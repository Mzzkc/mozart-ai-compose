"""Daemon detection and CLI routing — safe fallback to direct execution.

This module is used by CLI commands to auto-detect a running mozartd
daemon and route operations through it. When no daemon is detected,
the caller falls back to direct execution (existing behavior).

SAFETY: Every public function catches ALL exceptions and returns a
"not routed" result. This ensures that daemon bugs never break the CLI.
The CLI wiring (Phase 4) wraps calls in try/except as a second layer.
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
    """Check if mozartd is running. Safe: returns False on any error."""
    try:
        from mozart.daemon.ipc.client import DaemonClient

        resolved = _resolve_socket_path(socket_path)
        client = DaemonClient(resolved)
        return await client.is_daemon_running()
    except (OSError, ConnectionError) as e:
        # Connection/socket errors — daemon not reachable.
        resolved = _resolve_socket_path(socket_path)
        level = "info" if resolved.exists() else "debug"
        getattr(_logger, level)("daemon_detection_failed", error=str(e))
        return False
    except ImportError:
        _logger.debug("daemon_detection_import_error")
        return False
    except Exception as e:
        _logger.warning("daemon_detection_unexpected", error=str(e), exc_info=True)
        return False


async def try_daemon_route(
    method: str,
    params: dict[str, Any],
    *,
    socket_path: Path | None = None,
) -> tuple[bool, Any]:
    """Try routing a CLI command through the daemon.

    Returns:
        (True, result) if daemon handled the request.
        (False, None) if daemon is not running or any error occurred.

    This function NEVER raises — any exception returns (False, None).
    """
    try:
        from mozart.daemon.ipc.client import DaemonClient

        client = DaemonClient(_resolve_socket_path(socket_path))
        if not await client.is_daemon_running():
            return False, None
        result = await client.call(method, params)
        return True, result
    except Exception as e:
        _logger.debug("daemon_route_failed", method=method, error=str(e))
        return False, None
