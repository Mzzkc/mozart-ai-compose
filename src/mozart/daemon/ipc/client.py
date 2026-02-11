"""Async Unix domain socket client for Mozart daemon IPC.

Provides ``DaemonClient`` with two call patterns:

- ``call(method, params)``: send a JSON-RPC request, await a single response.
- ``stream(method, params)``: send a request, yield streaming notifications,
  and return the final result when the stream ends.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mozart.core.logging import get_logger
from mozart.daemon.exceptions import DaemonNotRunningError
from mozart.daemon.ipc.errors import rpc_error_to_exception
from mozart.daemon.ipc.protocol import JsonRpcRequest

_logger = get_logger("daemon.ipc.client")

# Counter for generating unique request IDs within a client session.
_request_id_counter = 0


def _next_request_id() -> int:
    """Return a monotonically increasing request ID."""
    global _request_id_counter  # noqa: PLW0603
    _request_id_counter += 1
    return _request_id_counter


class DaemonClient:
    """Async client for the Mozart daemon Unix socket IPC.

    Parameters
    ----------
    socket_path:
        Path to the Unix domain socket created by ``DaemonServer``.
    timeout:
        Seconds to wait for a response before raising ``TimeoutError``.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        timeout: float = 30.0,
    ) -> None:
        self._socket_path = socket_path
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def _connect(
        self,
    ) -> AsyncIterator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
        """Open a connection to the daemon socket.

        Raises ``DaemonNotRunningError`` if the socket doesn't exist or
        the connection is refused.
        """
        if not self._socket_path.exists():
            raise DaemonNotRunningError(
                f"Daemon socket not found: {self._socket_path}"
            )

        try:
            reader, writer = await asyncio.open_unix_connection(
                str(self._socket_path)
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError) as exc:
            raise DaemonNotRunningError(
                f"Cannot connect to daemon at {self._socket_path}: {exc}"
            ) from exc

        try:
            yield reader, writer
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass  # Best-effort cleanup

    # ------------------------------------------------------------------
    # RPC call (single request → single response)
    # ------------------------------------------------------------------

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send a JSON-RPC request and return the result.

        Raises:
            DaemonNotRunningError: socket unreachable
            DaemonError (subclass): server returned an error response
            TimeoutError: no response within ``self._timeout``
        """
        request = JsonRpcRequest(
            method=method,
            params=params,
            id=_next_request_id(),
        )

        async with self._connect() as (reader, writer):
            writer.write(request.model_dump_json().encode() + b"\n")
            await writer.drain()

            line = await asyncio.wait_for(
                reader.readline(), timeout=self._timeout
            )
            if not line:
                raise DaemonNotRunningError("Daemon closed connection")

            response = json.loads(line)
            if "error" in response:
                raise rpc_error_to_exception(response["error"])
            return response.get("result")

    # ------------------------------------------------------------------
    # Streaming RPC (request → notifications* → final response)
    # ------------------------------------------------------------------

    async def stream(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a JSON-RPC request and yield streaming notifications.

        Notifications (no ``id``) are yielded as dicts.  The final
        response (with ``id``) terminates the iteration.

        If the final response contains an error, raises the
        corresponding ``DaemonError``.
        """
        request = JsonRpcRequest(
            method=method,
            params=params,
            id=_next_request_id(),
        )

        async with self._connect() as (reader, writer):
            writer.write(request.model_dump_json().encode() + b"\n")
            await writer.drain()

            async for raw_line in reader:
                if not raw_line:
                    break

                msg = json.loads(raw_line)

                # Check if this is the final response (has our request id)
                msg_id = msg.get("id")
                if msg_id is not None:
                    # Final response — stream is done
                    if "error" in msg:
                        raise rpc_error_to_exception(msg["error"])
                    return

                # Notification — yield params to caller
                yield msg.get("params", {})


__all__ = ["DaemonClient"]
