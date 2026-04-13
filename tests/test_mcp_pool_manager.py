"""Tests for the shared MCP pool manager — process lifecycle, socket bridging.

The MCP pool manages long-running MCP server processes for the conductor.
Each server is started as a subprocess and proxied behind a Unix socket.
Agents access the servers via bind-mounted sockets.

TDD: tests written before implementation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marianne.daemon.config import McpPoolConfig, McpServerEntry
from marianne.daemon.mcp_pool import McpPoolManager, McpServerState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def pool_config(tmp_path: Path) -> McpPoolConfig:
    """Create a McpPoolConfig with two test servers."""
    return McpPoolConfig(
        servers={
            "github": McpServerEntry(
                command="github-mcp-server",
                transport="stdio",
                socket=str(tmp_path / "github.sock"),
                restart_policy="on-failure",
            ),
            "filesystem": McpServerEntry(
                command="fs-mcp-server",
                transport="stdio",
                socket=str(tmp_path / "filesystem.sock"),
                restart_policy="never",
            ),
        },
    )


@pytest.fixture()
def empty_config() -> McpPoolConfig:
    """Empty pool with no servers."""
    return McpPoolConfig()


# =============================================================================
# Construction
# =============================================================================


class TestConstruction:
    """McpPoolManager can be created from config."""

    def test_creates_from_config(self, pool_config: McpPoolConfig) -> None:
        manager = McpPoolManager(pool_config)
        assert manager is not None

    def test_creates_with_empty_config(self, empty_config: McpPoolConfig) -> None:
        manager = McpPoolManager(empty_config)
        assert manager is not None

    def test_server_names(self, pool_config: McpPoolConfig) -> None:
        manager = McpPoolManager(pool_config)
        assert set(manager.server_names()) == {"github", "filesystem"}


# =============================================================================
# Start / Stop lifecycle
# =============================================================================


class TestLifecycle:
    """Server processes are started and stopped by the manager."""

    async def test_start_all_starts_servers(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await manager.start_all()

        # Both servers should be tracked
        assert manager.is_running("github")
        assert manager.is_running("filesystem")

    async def test_stop_all_terminates_servers(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await manager.start_all()

        await manager.stop_all()

        assert not manager.is_running("github")
        assert not manager.is_running("filesystem")

    async def test_start_empty_pool_is_noop(
        self, empty_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(empty_config)
        await manager.start_all()  # should not raise
        await manager.stop_all()

    async def test_stop_all_is_idempotent(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=0)

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await manager.start_all()

        await manager.stop_all()
        await manager.stop_all()  # should not raise


# =============================================================================
# Server state tracking
# =============================================================================


class TestServerState:
    """Server state is tracked correctly."""

    def test_initial_state_is_stopped(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)
        assert not manager.is_running("github")
        assert not manager.is_running("filesystem")

    def test_unknown_server_not_running(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)
        assert not manager.is_running("nonexistent")

    def test_server_state_enum(self) -> None:
        assert McpServerState.STOPPED.value == "stopped"
        assert McpServerState.RUNNING.value == "running"
        assert McpServerState.FAILED.value == "failed"

    async def test_get_status_returns_all_servers(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)
        status = manager.get_status()
        assert "github" in status
        assert "filesystem" in status
        assert status["github"] == McpServerState.STOPPED
        assert status["filesystem"] == McpServerState.STOPPED


# =============================================================================
# Process failure handling
# =============================================================================


class TestFailureHandling:
    """Process failures are detected and handled per restart policy."""

    async def test_start_failure_marks_server_failed(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("command not found"),
        ):
            await manager.start_all()

        # Servers should be marked as failed, not running
        status = manager.get_status()
        assert status["github"] == McpServerState.FAILED
        assert status["filesystem"] == McpServerState.FAILED

    async def test_stop_handles_already_dead_process(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 1  # Already exited
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = AsyncMock(return_value=1)

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await manager.start_all()

        # Should not raise even though process already exited
        await manager.stop_all()


# =============================================================================
# Socket path management
# =============================================================================


class TestSocketPaths:
    """Socket paths are managed correctly."""

    def test_get_socket_path(
        self, pool_config: McpPoolConfig, tmp_path: Path,
    ) -> None:
        manager = McpPoolManager(pool_config)
        assert manager.get_socket_path("github") == Path(
            str(tmp_path / "github.sock")
        )

    def test_get_socket_path_unknown_server(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)
        assert manager.get_socket_path("nonexistent") is None

    def test_get_all_socket_paths(
        self, pool_config: McpPoolConfig, tmp_path: Path,
    ) -> None:
        manager = McpPoolManager(pool_config)
        paths = manager.get_all_socket_paths()
        assert len(paths) == 2
        assert "github" in paths
        assert "filesystem" in paths


# =============================================================================
# Health check
# =============================================================================


class TestHealthCheck:
    """Health checks verify server processes are alive."""

    async def test_health_check_running_server(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None  # Still running
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await manager.start_all()

        assert await manager.health_check("github")

    async def test_health_check_dead_server(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)

        mock_proc = AsyncMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        with patch(
            "marianne.daemon.mcp_pool.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await manager.start_all()

        # Simulate process death
        mock_proc.returncode = 1

        assert not await manager.health_check("github")

    async def test_health_check_unknown_server(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)
        assert not await manager.health_check("nonexistent")

    async def test_health_check_not_started(
        self, pool_config: McpPoolConfig,
    ) -> None:
        manager = McpPoolManager(pool_config)
        assert not await manager.health_check("github")
