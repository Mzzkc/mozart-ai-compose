"""Conductor detection and CLI routing — safe fallback to direct execution.

This module is used by CLI commands to auto-detect a running Mozart
conductor and route operations through it. When no conductor is detected,
the caller falls back to direct execution (existing behavior).

SAFETY: Every public function catches ALL exceptions and returns a
"not routed" result. This ensures that conductor bugs never break the CLI.
The CLI wiring wraps calls in try/except as a second layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mozart.core.logging import get_logger

_logger = get_logger("daemon.detect")


def _resolve_socket_path(socket_path: Path | None) -> Path:
    """Resolve socket path, falling back to SocketConfig default."""
    if socket_path is not None:
        return socket_path
    from mozart.daemon.config import SocketConfig

    return SocketConfig().path


async def is_daemon_available(socket_path: Path | None = None) -> bool:
    """Check if the Mozart conductor is running. Safe: returns False on any error."""
    try:
        from mozart.daemon.ipc.client import DaemonClient

        resolved = _resolve_socket_path(socket_path)
        client = DaemonClient(resolved)
        return await client.is_daemon_running()
    except (OSError, ConnectionError) as e:
        # Connection/socket errors — conductor not reachable.
        resolved = _resolve_socket_path(socket_path)
        level = "info" if resolved.exists() else "debug"
        getattr(_logger, level)("daemon_detection_failed", error=str(e))
        return False
    except ImportError:
        _logger.debug("daemon_detection_import_error")
        return False
    except Exception as e:
        # Check if this is a known DaemonError (guard against missing module)
        is_daemon_error = False
        try:
            from mozart.daemon.exceptions import DaemonError
            is_daemon_error = isinstance(e, DaemonError)
        except ImportError:
            pass

        if is_daemon_error:
            _logger.debug("daemon_detection_failed", error=str(e))
        else:
            _logger.warning("daemon_detection_unexpected", error=str(e), exc_info=True)
        return False


async def try_daemon_route(
    method: str,
    params: dict[str, Any],
    *,
    socket_path: Path | None = None,
) -> tuple[bool, Any]:
    """Try routing a CLI command through the conductor.

    Returns:
        (True, result) if conductor handled the request.
        (False, None) if conductor is not running or a connection error occurred.

    Raises:
        JobSubmissionError, ResourceExhaustedError: Business logic errors from
            a running conductor are re-raised so callers can handle them
            (e.g., "job not found" is different from "daemon not running").

    Connection-level errors never raise — they return (False, None).
    """
    try:
        from mozart.daemon.ipc.client import DaemonClient

        client = DaemonClient(_resolve_socket_path(socket_path))
        if not await client.is_daemon_running():
            return False, None
        result = await client.call(method, params)
        return True, result
    except (OSError, ConnectionError, TimeoutError, ValueError) as e:
        _logger.debug("daemon_route_failed", method=method, error=str(e))
        return False, None
    except Exception as e:
        from mozart.daemon.exceptions import JobSubmissionError, ResourceExhaustedError

        if isinstance(e, (JobSubmissionError, ResourceExhaustedError)):
            # Business logic errors from a *running* conductor — re-raise so
            # callers can distinguish "daemon unavailable" from "daemon rejected
            # the request" (e.g., "job not found" vs "conductor not running").
            raise

        from mozart.daemon.exceptions import DaemonError

        if isinstance(e, DaemonError):
            # All other daemon errors (not running, already running, unknown
            # method, protocol errors) — treat as "daemon not reachable".
            _logger.debug("daemon_route_failed", method=method, error=str(e))
            return False, None

        _logger.warning(
            "daemon_route_unexpected_error",
            method=method,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True,
        )
        return False, None
