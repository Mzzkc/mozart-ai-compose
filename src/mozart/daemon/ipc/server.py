"""Async Unix domain socket server for Mozart daemon IPC.

Binds a Unix socket, accepts concurrent client connections, reads
newline-delimited JSON-RPC 2.0 requests, dispatches them through
``RequestHandler``, and writes responses back.  Handles connection
lifecycle and socket cleanup on shutdown.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from mozart.core.logging import get_logger
from mozart.daemon.ipc.errors import invalid_request, parse_error
from mozart.daemon.ipc.handler import RequestHandler
from mozart.daemon.ipc.protocol import JsonRpcRequest

_logger = get_logger("daemon.ipc.server")

# Maximum size of a single JSON-RPC message (1 MB).
MAX_MESSAGE_BYTES = 1_048_576

# Default limit on concurrent client connections.
DEFAULT_MAX_CONNECTIONS = 20


class DaemonServer:
    """Async Unix domain socket server with JSON-RPC 2.0 routing.

    Parameters
    ----------
    socket_path:
        Filesystem path for the Unix domain socket.
    handler:
        ``RequestHandler`` that dispatches JSON-RPC methods.
    permissions:
        Octal file permissions applied to the socket after creation.
        Defaults to ``0o660`` (owner + group read/write).
    max_connections:
        Maximum concurrent client connections.
    """

    def __init__(
        self,
        socket_path: Path,
        handler: RequestHandler,
        *,
        permissions: int = 0o660,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ) -> None:
        self._socket_path = socket_path
        self._handler = handler
        self._permissions = permissions
        self._max_connections = max_connections
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._connection_semaphore = asyncio.Semaphore(max_connections)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind the Unix socket and start accepting connections."""
        # Ensure parent directory exists
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        # Reject symlinks to prevent redirection attacks
        if self._socket_path.is_symlink():
            raise OSError(
                f"Socket path is a symlink (possible attack): {self._socket_path}"
            )
        # Remove stale socket from a previous unclean shutdown
        self._socket_path.unlink(missing_ok=True)

        self._server = await asyncio.start_unix_server(
            self._accept_connection,
            path=str(self._socket_path),
            limit=MAX_MESSAGE_BYTES,
        )
        os.chmod(self._socket_path, self._permissions)

        _logger.info(
            "ipc_server_started",
            socket_path=str(self._socket_path),
            max_connections=self._max_connections,
        )

    async def stop(self) -> None:
        """Stop the server, cancel active connections, and remove the socket."""
        if self._server is None:
            return

        # Stop accepting new connections
        self._server.close()
        await self._server.wait_closed()

        # Cancel all active connection tasks
        for task in self._connections:
            task.cancel()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
        self._connections.clear()

        # Clean up socket file
        self._socket_path.unlink(missing_ok=True)
        self._server = None

        _logger.info("ipc_server_stopped")

    @property
    def is_running(self) -> bool:
        """Return whether the server is currently accepting connections."""
        return self._server is not None and self._server.is_serving()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    async def _accept_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Wrap each connection in a tracked task with a concurrency limit."""
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
            task.add_done_callback(self._on_connection_done)

        async with self._connection_semaphore:
            await self._handle_connection(reader, writer)

    def _on_connection_done(self, task: asyncio.Task[None]) -> None:
        """Log exceptions from connection tasks before discarding."""
        self._connections.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            _logger.warning(
                "connection_task_failed",
                error=str(exc),
                task_name=task.get_name(),
            )

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Process requests on a single client connection until it closes."""
        peer = writer.get_extra_info("peername") or "unknown"
        _logger.debug("client_connected", peer=str(peer))

        try:
            while True:
                try:
                    line = await reader.readline()
                except asyncio.LimitOverrunError:
                    # Message exceeded MAX_MESSAGE_BYTES; buffer is in an
                    # indeterminate state so we must close the connection.
                    _logger.warning("message_too_large", peer=str(peer))
                    error_resp = parse_error()
                    writer.write(
                        error_resp.model_dump_json().encode() + b"\n"
                    )
                    await writer.drain()
                    break

                if not line:
                    break  # Client disconnected

                # Enforce max message size (belt-and-suspenders)
                if len(line) > MAX_MESSAGE_BYTES:
                    error_resp = parse_error()
                    writer.write(
                        error_resp.model_dump_json().encode() + b"\n"
                    )
                    await writer.drain()
                    continue

                await self._process_message(line, writer)
        except ConnectionResetError:
            _logger.debug("client_disconnected", peer=str(peer), reason="reset")
        except asyncio.CancelledError:
            _logger.debug("client_disconnected", peer=str(peer), reason="cancelled")
        except Exception as exc:
            _logger.warning(
                "connection_error",
                peer=str(peer),
                error=str(exc),
            )
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass  # Best-effort cleanup
            _logger.debug("client_disconnected", peer=str(peer))

    async def _process_message(
        self,
        line: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Parse one NDJSON line and route through the handler."""
        # Parse JSON
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            error_resp = parse_error()
            writer.write(error_resp.model_dump_json().encode() + b"\n")
            await writer.drain()
            return

        # Validate JSON-RPC structure
        if not isinstance(raw, dict) or "method" not in raw:
            error_resp = invalid_request(
                raw.get("id") if isinstance(raw, dict) else None,
                "missing 'method' field",
            )
            writer.write(error_resp.model_dump_json().encode() + b"\n")
            await writer.drain()
            return

        # Build typed request
        try:
            request = JsonRpcRequest.model_validate(raw)
        except Exception as exc:
            error_resp = invalid_request(raw.get("id"), str(exc))
            writer.write(error_resp.model_dump_json().encode() + b"\n")
            await writer.drain()
            return

        # Dispatch to handler
        response = await self._handler.handle(request, writer)

        # Handler returns None for notifications or streaming methods
        if response is not None:
            writer.write(response.model_dump_json().encode() + b"\n")
            await writer.drain()


__all__ = ["DaemonServer"]
