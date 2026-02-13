"""Tests for mozart.daemon.detect — CLI daemon detection and routing.

The detect module provides a safe fallback layer: if the daemon is not
running, or any error occurs, the CLI falls back to direct execution.
These tests verify that contract.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mozart.daemon.detect import is_daemon_available, try_daemon_route
from mozart.daemon.ipc.handler import RequestHandler
from mozart.daemon.ipc.server import DaemonServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detect_handler() -> RequestHandler:
    """Build a handler with methods needed for detection tests."""
    handler = RequestHandler()

    async def _status(_params: dict[str, Any], _writer: asyncio.StreamWriter) -> Any:
        return {
            "pid": 9999,
            "uptime_seconds": 42.0,
            "running_jobs": 1,
            "total_jobs_active": 3,
            "memory_usage_mb": 128.0,
            "version": "0.1.0",
        }

    async def _echo(params: dict[str, Any], _writer: asyncio.StreamWriter) -> Any:
        return {"echo": params}

    handler.register("daemon.status", _status)
    handler.register("test.echo", _echo)
    return handler


# ---------------------------------------------------------------------------
# Tests: is_daemon_available
# ---------------------------------------------------------------------------


class TestIsDaemonAvailable:
    """Tests for is_daemon_available()."""

    @pytest.mark.asyncio
    async def test_returns_false_no_socket(self, tmp_path: Path):
        """Returns False when no socket file exists."""
        sock = tmp_path / "nonexistent.sock"
        result = await is_daemon_available(sock)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_stale_socket(self, tmp_path: Path):
        """Returns False when socket file exists but no server listens."""
        sock = tmp_path / "stale.sock"
        sock.touch()
        result = await is_daemon_available(sock)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_when_running(self, tmp_path: Path):
        """Returns True when a server is actually listening."""
        sock = tmp_path / "test.sock"
        server = DaemonServer(sock, _make_detect_handler())
        await server.start()
        try:
            result = await is_daemon_available(sock)
            assert result is True
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_never_raises(self, tmp_path: Path):
        """Function never raises, even with invalid input."""
        # Pass a directory instead of a file — should return False, not raise
        result = await is_daemon_available(tmp_path)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: try_daemon_route
# ---------------------------------------------------------------------------


class TestTryDaemonRoute:
    """Tests for try_daemon_route()."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_daemon(self, tmp_path: Path):
        """Returns (False, None) when daemon is not running."""
        sock = tmp_path / "nonexistent.sock"
        routed, result = await try_daemon_route(
            "daemon.status", {}, socket_path=sock
        )
        assert routed is False
        assert result is None

    @pytest.mark.asyncio
    async def test_routes_through_daemon(self, tmp_path: Path):
        """Returns (True, result) when daemon handles the request."""
        sock = tmp_path / "test.sock"
        server = DaemonServer(sock, _make_detect_handler())
        await server.start()
        try:
            routed, result = await try_daemon_route(
                "test.echo",
                {"message": "routed"},
                socket_path=sock,
            )
            assert routed is True
            assert result["echo"]["message"] == "routed"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_returns_false_on_unknown_method(self, tmp_path: Path):
        """Returns (False, None) when daemon returns an error for the method."""
        sock = tmp_path / "test.sock"
        server = DaemonServer(sock, _make_detect_handler())
        await server.start()
        try:
            # Unknown method causes DaemonError, which is caught
            routed, result = await try_daemon_route(
                "nonexistent.method",
                {},
                socket_path=sock,
            )
            assert routed is False
            assert result is None
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_never_raises(self, tmp_path: Path):
        """Function never raises, even on unexpected errors."""
        # Invalid path that will cause errors
        routed, result = await try_daemon_route(
            "daemon.status", {}, socket_path=tmp_path / "nope.sock"
        )
        assert routed is False
        assert result is None

    @pytest.mark.asyncio
    async def test_routes_status_request(self, tmp_path: Path):
        """Status request returns full daemon status through routing."""
        sock = tmp_path / "test.sock"
        server = DaemonServer(sock, _make_detect_handler())
        await server.start()
        try:
            routed, result = await try_daemon_route(
                "daemon.status", {}, socket_path=sock
            )
            assert routed is True
            assert result["pid"] == 9999
            assert result["version"] == "0.1.0"
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Tests: CLI non-daemon regression (subprocess-based)
# ---------------------------------------------------------------------------


class TestCliNonDaemonRegression:
    """Verify that Mozart CLI works correctly without a running daemon.

    Uses subprocess execution to test the real CLI entry point with
    no daemon socket present — the detect.py auto-routing must fall
    back to direct execution without error.
    """

    def test_dry_run_without_daemon(self):
        """mozart run --dry-run succeeds without a daemon socket.

        This is the critical regression test: if detect.py's fallback
        breaks, every CLI invocation fails.
        """
        result = subprocess.run(
            [
                sys.executable, "-m", "mozart", "run",
                "examples/simple-sheet.yaml", "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **__import__("os").environ,
                # Ensure no daemon socket is found
                "MOZART_SOCKET_PATH": "/tmp/nonexistent-mozart-test.sock",
            },
        )
        assert result.returncode == 0, (
            f"CLI dry-run failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
        # Verify expected output markers
        assert "simple-sheet" in result.stdout.lower() or "dry run" in result.stdout.lower(), (
            f"Expected output markers not found:\n{result.stdout[:500]}"
        )

    def test_validate_without_daemon(self):
        """mozart validate succeeds without a daemon socket."""
        result = subprocess.run(
            [
                sys.executable, "-m", "mozart", "validate",
                "examples/simple-sheet.yaml",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"CLI validate failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
