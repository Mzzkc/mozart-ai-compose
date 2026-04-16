"""Integration tests for the IPC ConnectionPool and pooled DaemonClient.

Tests use real Unix sockets with a minimal DaemonServer — no mocks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from marianne.daemon.ipc.client import ConnectionPool, DaemonClient
from marianne.daemon.ipc.handler import RequestHandler
from marianne.daemon.ipc.server import DaemonServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_echo_handler() -> RequestHandler:
    """Handler that echoes params back as result."""
    handler = RequestHandler()

    async def _echo(params: dict[str, Any], writer: asyncio.StreamWriter) -> dict[str, Any]:
        return params

    handler.register("test.echo", _echo)
    return handler


def _build_counter_handler() -> tuple[RequestHandler, dict[str, int]]:
    """Handler that counts total requests processed."""
    counter: dict[str, int] = {"total": 0}
    handler = RequestHandler()

    async def _count(params: dict[str, Any], writer: asyncio.StreamWriter) -> dict[str, Any]:
        counter["total"] += 1
        return {"count": counter["total"]}

    handler.register("test.count", _count)
    return handler, counter


@pytest.fixture()
def socket_path(tmp_path: Path) -> Path:
    return tmp_path / "pool-test.sock"


async def _start_server(
    socket_path: Path,
    handler: RequestHandler | None = None,
) -> DaemonServer:
    """Start and return a DaemonServer with the given handler."""
    if handler is None:
        handler = _build_echo_handler()
    server = DaemonServer(socket_path, handler)
    await server.start()
    return server


# ---------------------------------------------------------------------------
# ConnectionPool tests
# ---------------------------------------------------------------------------


async def test_pool_acquire_release_reuse(socket_path: Path) -> None:
    """Connections are reused after release (LIFO)."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=4)

        # Acquire a connection
        r1, w1 = await pool.acquire()
        w1.get_extra_info("sockname")

        # Release it
        pool.release(r1, w1)

        # Acquire again — should get the same connection back
        r2, w2 = await pool.acquire()
        assert w2 is w1  # Same object — reused from idle stack

        pool.release(r2, w2)
        await pool.close()
    finally:
        await server.stop()


async def test_pool_max_size_enforced(socket_path: Path) -> None:
    """Pool semaphore prevents acquiring more than max_size connections."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=2, acquire_timeout=0.5)

        c1 = await pool.acquire()
        c2 = await pool.acquire()

        # Third acquire should timeout
        with pytest.raises(TimeoutError, match="Pool exhausted"):
            await pool.acquire()

        pool.release(*c1)
        pool.release(*c2)
        await pool.close()
    finally:
        await server.stop()


async def test_pool_stale_connections_discarded(socket_path: Path) -> None:
    """Connections idle beyond max_idle_seconds are discarded on acquire."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(
            socket_path,
            max_size=4,
            max_idle_seconds=0.1,  # Very short for testing
        )

        r1, w1 = await pool.acquire()
        pool.release(r1, w1)

        # Wait for the connection to become stale
        await asyncio.sleep(0.2)

        # Next acquire should discard the stale connection and open a new one
        r2, w2 = await pool.acquire()
        assert w2 is not w1  # Different connection — stale one was discarded

        pool.release(r2, w2)
        await pool.close()
    finally:
        await server.stop()


async def test_pool_close_idempotent(socket_path: Path) -> None:
    """Closing a pool multiple times does not raise."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=2)
        r, w = await pool.acquire()
        pool.release(r, w)

        await pool.close()
        await pool.close()  # Should not raise
        assert pool.closed
    finally:
        await server.stop()


async def test_pool_acquire_after_close_raises(socket_path: Path) -> None:
    """Acquiring from a closed pool raises DaemonNotRunningError."""
    from marianne.daemon.exceptions import DaemonNotRunningError

    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=2)
        await pool.close()

        with pytest.raises(DaemonNotRunningError, match="pool is closed"):
            await pool.acquire()
    finally:
        await server.stop()


async def test_pool_discard_releases_semaphore(socket_path: Path) -> None:
    """Discarding a connection frees its semaphore slot."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=1, acquire_timeout=1.0)

        r, w = await pool.acquire()
        pool.discard(w)

        # Should be able to acquire again — slot was freed
        r2, w2 = await pool.acquire()
        pool.release(r2, w2)
        await pool.close()
    finally:
        await server.stop()


async def test_pool_broken_connection_skipped(socket_path: Path) -> None:
    """Broken idle connections (writer closing) are skipped on acquire."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=4)

        r1, w1 = await pool.acquire()
        # Artificially break the connection
        w1.close()
        pool.release(r1, w1)

        # Acquire should skip the broken one and open a new one
        r2, w2 = await pool.acquire()
        assert w2 is not w1

        pool.release(r2, w2)
        await pool.close()
    finally:
        await server.stop()


async def test_pool_constructor_validation() -> None:
    """Pool constructor validates parameters."""
    sock = Path("/tmp/pool-validate.sock")

    with pytest.raises(ValueError, match="max_size must be >= 1"):
        ConnectionPool(sock, max_size=0)

    with pytest.raises(ValueError, match="max_idle_seconds must be > 0"):
        ConnectionPool(sock, max_idle_seconds=0.0)

    with pytest.raises(ValueError, match="connect_timeout must be > 0"):
        ConnectionPool(sock, connect_timeout=0.0)

    with pytest.raises(ValueError, match="acquire_timeout must be > 0"):
        ConnectionPool(sock, acquire_timeout=0.0)


# ---------------------------------------------------------------------------
# DaemonClient pooled call() tests
# ---------------------------------------------------------------------------


async def test_client_call_uses_pool(socket_path: Path) -> None:
    """DaemonClient.call() reuses connections via the pool."""
    handler, counter = _build_counter_handler()
    server = await _start_server(socket_path, handler)
    try:
        async with DaemonClient(socket_path, pool_size=4) as client:
            # Multiple sequential calls should reuse connections
            for _ in range(5):
                result = await client.call("test.count")
                assert "count" in result

            assert counter["total"] == 5
    finally:
        await server.stop()


async def test_client_concurrent_calls(socket_path: Path) -> None:
    """Multiple concurrent calls work with the pool."""
    handler, counter = _build_counter_handler()
    server = await _start_server(socket_path, handler)
    try:
        async with DaemonClient(socket_path, pool_size=4) as client:
            results = await asyncio.gather(*[client.call("test.count") for _ in range(4)])
            assert len(results) == 4
            assert counter["total"] == 4
    finally:
        await server.stop()


async def test_client_survives_server_restart(socket_path: Path) -> None:
    """Client works across a server restart (pool recreated lazily)."""
    handler, counter = _build_counter_handler()
    server = await _start_server(socket_path, handler)
    try:
        client = DaemonClient(socket_path, pool_size=2)

        # Prime the pool
        await client.call("test.count")
        assert counter["total"] == 1

        # Close the pool so idle connections are dropped — this lets
        # the server's handler tasks finish so server.stop() completes.
        await client.close()

        # Restart server
        await server.stop()
        handler2, counter2 = _build_counter_handler()
        server = await _start_server(socket_path, handler2)

        # Pool is recreated lazily — new connections to new server
        result = await client.call("test.count")
        assert result["count"] == 1  # New counter starts at 1
        assert counter2["total"] == 1

        await client.close()
    finally:
        await server.stop()


async def test_client_handles_broken_pooled_connections(
    socket_path: Path,
) -> None:
    """Client recovers when idle pooled connections become broken.

    Simulates a broken connection by closing the writer on idle
    connections.  The pool detects them as broken during acquire and
    opens fresh connections transparently.
    """
    handler, counter = _build_counter_handler()
    server = await _start_server(socket_path, handler)
    try:
        async with DaemonClient(socket_path, pool_size=2) as client:
            # Prime the pool
            result1 = await client.call("test.count")
            assert result1["count"] == 1

            # Break all idle connections
            pool = client._get_pool()
            for _r, w, _ts in pool._idle:
                w.close()

            # Next call should work — pool detects broken connections
            # and opens a fresh one
            result2 = await client.call("test.count")
            assert result2["count"] == 2
    finally:
        await server.stop()


async def test_client_context_manager(socket_path: Path) -> None:
    """DaemonClient works as an async context manager."""
    server = await _start_server(socket_path)
    try:
        async with DaemonClient(socket_path) as client:
            result = await client.call("test.echo", {"hello": "world"})
            assert result == {"hello": "world"}

        # Pool should be closed after exiting context
        assert client._pool is None
    finally:
        await server.stop()


async def test_client_close_idempotent(socket_path: Path) -> None:
    """Closing a client multiple times does not raise."""
    server = await _start_server(socket_path)
    try:
        client = DaemonClient(socket_path)
        await client.call("test.echo", {"x": 1})
        await client.close()
        await client.close()  # Should not raise
    finally:
        await server.stop()


async def test_client_pool_independent_of_stream(socket_path: Path) -> None:
    """Pool-based call() works independently of _connect()-based methods."""
    handler = _build_echo_handler()
    server = await _start_server(socket_path, handler)
    try:
        async with DaemonClient(socket_path, pool_size=2) as client:
            # Pooled calls work
            result = await client.call("test.echo", {"a": 1})
            assert result == {"a": 1}

            # _connect() is still usable (used by stream/is_daemon_running)
            async with client._connect() as (reader, writer):
                assert not writer.is_closing()
                writer.close()

            # Pool still works after direct _connect() usage
            result2 = await client.call("test.echo", {"b": 2})
            assert result2 == {"b": 2}
    finally:
        await server.stop()


async def test_client_application_error_releases_connection(
    socket_path: Path,
) -> None:
    """Application-level errors (JSON-RPC error responses) release the
    connection back to the pool rather than discarding it."""
    from marianne.daemon.exceptions import DaemonError

    handler = RequestHandler()

    async def _fail(params: dict[str, Any], writer: asyncio.StreamWriter) -> dict[str, Any]:
        from marianne.daemon.exceptions import JobSubmissionError

        raise JobSubmissionError("test failure")

    handler.register("test.fail", _fail)

    # Also register a success method
    async def _ok(params: dict[str, Any], writer: asyncio.StreamWriter) -> dict[str, Any]:
        return {"ok": True}

    handler.register("test.ok", _ok)

    server = await _start_server(socket_path, handler)
    try:
        async with DaemonClient(socket_path, pool_size=1) as client:
            # Call the failing method — should raise but not break the pool
            with pytest.raises(DaemonError):
                await client.call("test.fail")

            # Pool should still work — connection was released, not discarded
            result = await client.call("test.ok")
            assert result == {"ok": True}
    finally:
        await server.stop()


async def test_pool_release_on_closed_pool(socket_path: Path) -> None:
    """Releasing a connection to a closed pool discards it."""
    server = await _start_server(socket_path)
    try:
        pool = ConnectionPool(socket_path, max_size=2)
        r, w = await pool.acquire()

        await pool.close()

        # Release after close should not raise — just discard
        pool.release(r, w)
    finally:
        await server.stop()
