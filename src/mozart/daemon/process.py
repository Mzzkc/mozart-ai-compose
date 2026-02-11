"""mozartd — Mozart daemon process.

Long-running orchestration service that manages job execution,
resources, and cross-job coordination.  Provides CLI commands
for starting, stopping, and checking daemon status.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import typer

from mozart.core.logging import get_logger
from mozart.daemon.config import DaemonConfig
from mozart.daemon.pgroup import ProcessGroupManager

_logger = get_logger("mozartd")

daemon_app = typer.Typer(name="mozartd", help="Mozart daemon service")


# ─── CLI Commands ─────────────────────────────────────────────────────


@daemon_app.command()
def start(
    config_file: Path | None = typer.Option(None, "--config", "-c"),
    foreground: bool = typer.Option(False, "--foreground", "-f"),
    log_level: str = typer.Option("info", "--log-level", "-l"),
) -> None:
    """Start the Mozart daemon."""
    config = _load_config(config_file)
    config.log_level = log_level

    # Check if already running (PID alive check + advisory lock probe)
    pid = _read_pid(config.pid_file)
    if pid is not None and _pid_alive(pid):
        typer.echo(f"mozartd is already running (PID {pid})")
        raise typer.Exit(1)

    # Detect concurrent start race via advisory lock
    if config.pid_file.exists() and not config.pid_file.is_symlink():
        try:
            probe_fd = os.open(str(config.pid_file), os.O_RDONLY)
            try:
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(probe_fd, fcntl.LOCK_UN)
            finally:
                os.close(probe_fd)
        except OSError:
            typer.echo("mozartd is starting up (PID file locked)")
            raise typer.Exit(1) from None

    # Configure logging
    from mozart.core.logging import configure_logging

    log_fmt = "console" if foreground else "json"
    configure_logging(
        level=log_level.upper(),  # type: ignore[arg-type]
        format=log_fmt,  # type: ignore[arg-type]
        file_path=config.log_file,
        include_timestamps=True,
    )

    if not foreground:
        _daemonize(config)
    else:
        _logger.info("daemon.starting", pid=os.getpid(), foreground=True)

    daemon = DaemonProcess(config)
    asyncio.run(daemon.run())


@daemon_app.command()
def stop(
    pid_file: Path = typer.Option(
        Path("/tmp/mozartd.pid"), "--pid-file",
    ),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Stop the running daemon."""
    pid = _read_pid(pid_file)
    if pid is None or not _pid_alive(pid):
        typer.echo("mozartd is not running")
        # Clean up stale PID file
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(1)

    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    typer.echo(f"Sent {'SIGKILL' if force else 'SIGTERM'} to mozartd (PID {pid})")


@daemon_app.command()
def status(
    pid_file: Path = typer.Option(
        Path("/tmp/mozartd.pid"), "--pid-file",
    ),
    socket_path: Path = typer.Option(
        Path("/tmp/mozartd.sock"), "--socket",
    ),
) -> None:
    """Check daemon status via health probes."""
    pid = _read_pid(pid_file)
    if pid is None or not _pid_alive(pid):
        typer.echo("mozartd is not running")
        pid_file.unlink(missing_ok=True)
        raise typer.Exit(1)

    typer.echo(f"mozartd is running (PID {pid})")

    # Query health probes via IPC
    from mozart.daemon.ipc.client import DaemonClient

    client = DaemonClient(socket_path)

    HealthResult = tuple[
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]

    async def _get_health() -> HealthResult:
        health = None
        ready = None
        daemon_info = None
        try:
            health = await client.call("daemon.health")
        except Exception:
            pass
        try:
            ready = await client.call("daemon.ready")
        except Exception:
            pass
        try:
            daemon_info = await client.call("daemon.status")
        except Exception:
            pass
        return health, ready, daemon_info

    try:
        health, ready, daemon_info = asyncio.run(_get_health())
    except Exception:
        typer.echo("  (Could not connect to daemon socket for details)")
        return

    if health:
        uptime = health.get("uptime_seconds", 0)
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        typer.echo(f"  Uptime: {hours}h {minutes}m {seconds}s")

    if ready:
        status_str = ready.get("status", "unknown")
        symbol = "+" if status_str == "ready" else "-"
        typer.echo(f"  [{symbol}] Readiness: {status_str}")
        typer.echo(f"  Running jobs: {ready.get('running_jobs', '?')}")
        typer.echo(f"  Memory: {ready.get('memory_mb', '?')} MB")
        typer.echo(f"  Child processes: {ready.get('child_processes', '?')}")
        typer.echo(f"  Accepting work: {ready.get('accepting_work', '?')}")

    if daemon_info:
        typer.echo(f"  Version: {daemon_info.get('version', '?')}")


# ─── DaemonProcess ────────────────────────────────────────────────────


class DaemonProcess:
    """Long-running Mozart daemon process.

    Composes DaemonServer (IPC), JobManager (job tracking),
    and ResourceMonitor (limits) into a single lifecycle.
    """

    def __init__(self, config: DaemonConfig) -> None:
        self._config = config
        self._shutdown_event = asyncio.Event()
        self._pgroup = ProcessGroupManager()
        self._start_time = time.monotonic()

    async def run(self) -> None:
        """Main daemon lifecycle: boot, serve, shutdown."""
        # 1. Write PID file
        _write_pid(self._config.pid_file)

        try:
            # 2. Set up process group (fixes issue #38 — orphan prevention)
            self._pgroup.setup()

            # 3. Create components
            from mozart.daemon.ipc.handler import RequestHandler
            from mozart.daemon.ipc.server import DaemonServer
            from mozart.daemon.manager import JobManager
            from mozart.daemon.monitor import ResourceMonitor

            manager = JobManager(self._config)
            monitor = ResourceMonitor(
                self._config.resource_limits, manager, pgroup=self._pgroup,
            )
            handler = RequestHandler()

            # 4. Create health checker
            from mozart.daemon.health import HealthChecker

            health = HealthChecker(manager, monitor, start_time=self._start_time)

            # 5. Register RPC methods (adapt JobManager to handler signature)
            self._register_methods(handler, manager, health)

            # 6. Start server
            server = DaemonServer(
                self._config.socket.path,
                handler,
                permissions=self._config.socket.permissions,
                max_connections=self._config.socket.backlog,
            )
            await server.start()

            # 7. Install signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(
                        self._handle_signal(s, manager, server),
                    ),
                )
            loop.add_signal_handler(
                signal.SIGHUP,
                lambda: asyncio.create_task(self._reload_config()),
            )

            # 8. Start resource monitor
            interval = getattr(
                self._config, "monitor_interval_seconds", 15.0,
            )
            await monitor.start(interval_seconds=interval)

            # 9. Run until shutdown
            _logger.info(
                "daemon.started",
                pid=os.getpid(),
                socket=str(self._config.socket.path),
            )
            await manager.wait_for_shutdown()

            # 10. Cleanup
            await monitor.stop()
            await server.stop()

            # 11. Kill remaining children in process group (issue #38)
            self._pgroup.kill_all_children()
            orphans = self._pgroup.cleanup_orphans()
            if orphans:
                _logger.info(
                    "daemon.shutdown_orphans_cleaned",
                    count=len(orphans),
                )

            _logger.info("daemon.stopped")
        finally:
            # Always remove PID file, even on crash
            self._config.pid_file.unlink(missing_ok=True)

    def _register_methods(
        self,
        handler: Any,
        manager: Any,
        health: Any = None,
    ) -> None:
        """Wire JSON-RPC methods to JobManager and HealthChecker."""
        from mozart.daemon.types import JobRequest

        async def handle_submit(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            request = JobRequest(**params)
            response = await manager.submit_job(request)
            return response.model_dump()

        async def handle_job_status(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            return await manager.get_job_status(
                params["job_id"], params.get("workspace"),
            )

        async def handle_pause(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            ok = await manager.pause_job(
                params["job_id"], params.get("workspace"),
            )
            return {"paused": ok}

        async def handle_resume(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            response = await manager.resume_job(
                params["job_id"], params.get("workspace"),
            )
            return response.model_dump()

        async def handle_cancel(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            ok = await manager.cancel_job(params["job_id"])
            return {"cancelled": ok}

        async def handle_list(_p: dict[str, Any], _w: Any) -> list[dict[str, Any]]:
            return await manager.list_jobs()

        async def handle_daemon_status(_p: dict[str, Any], _w: Any) -> dict[str, Any]:
            return await manager.get_daemon_status()

        async def handle_shutdown(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            graceful = params.get("graceful", True)
            asyncio.create_task(manager.shutdown(graceful=graceful))
            return {"shutting_down": True}

        handler.register("job.submit", handle_submit)
        handler.register("job.status", handle_job_status)
        handler.register("job.pause", handle_pause)
        handler.register("job.resume", handle_resume)
        handler.register("job.cancel", handle_cancel)
        handler.register("job.list", handle_list)
        handler.register("daemon.status", handle_daemon_status)
        handler.register("daemon.shutdown", handle_shutdown)

        # Health check probes
        if health is not None:
            async def handle_health(_p: dict[str, Any], _w: Any) -> dict[str, Any]:
                return await health.liveness()

            async def handle_ready(_p: dict[str, Any], _w: Any) -> dict[str, Any]:
                return await health.readiness()

            handler.register("daemon.health", handle_health)
            handler.register("daemon.ready", handle_ready)

    async def _handle_signal(
        self,
        sig: signal.Signals,
        manager: Any,
        server: Any,
    ) -> None:
        """Handle shutdown signals (SIGTERM, SIGINT)."""
        _logger.info("daemon.signal_received", signal=sig.name)
        await manager.shutdown(graceful=True)

    async def _reload_config(self) -> None:
        """Hot-reload daemon config on SIGHUP."""
        config_file = getattr(self._config, "config_file", None)
        if config_file and Path(config_file).exists():
            _logger.info("daemon.reloading_config", path=str(config_file))
            # Reload would update self._config in place
            # For now, just log the reload attempt
        else:
            _logger.debug("daemon.sighup_no_config_file")


# ─── Helpers ──────────────────────────────────────────────────────────


def _load_config(config_file: Path | None) -> DaemonConfig:
    """Load DaemonConfig from YAML file or return defaults."""
    if config_file and config_file.exists():
        import yaml

        with open(config_file) as f:
            data = yaml.safe_load(f) or {}
        return DaemonConfig.model_validate(data)
    return DaemonConfig()


def _write_pid(pid_file: Path) -> None:
    """Write current PID to file atomically with advisory lock.

    Uses fcntl.flock() to prevent TOCTOU races when two ``mozartd start``
    invocations run concurrently.  Also rejects symlinks to avoid a local
    attacker redirecting the write to an arbitrary file.
    """
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Reject symlinks (parity with socket symlink check in DaemonServer)
    if pid_file.is_symlink():
        raise OSError(f"PID file is a symlink (possible attack): {pid_file}")

    tmp = pid_file.with_suffix(".tmp")
    tmp.write_text(str(os.getpid()))
    tmp.rename(pid_file)

    # Advisory lock — held for daemon lifetime (released on fd close / exit)
    try:
        fd = os.open(str(pid_file), os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Store fd so the lock persists until the process exits
        _write_pid._lock_fd = fd  # type: ignore[attr-defined]
    except OSError:
        _logger.debug("pid_file.lock_failed", pid_file=str(pid_file))


def _read_pid(pid_file: Path) -> int | None:
    """Read PID from file, returning None if missing or invalid."""
    try:
        return int(pid_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we can't signal it


def _daemonize(config: DaemonConfig) -> None:
    """Double-fork to detach from terminal."""
    # First fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)  # Parent exits

    # New session
    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        sys.exit(0)  # First child exits

    # Redirect stdio to /dev/null
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)

    _logger.info(
        "daemon.daemonized",
        pid=os.getpid(),
        sid=os.getsid(0),
    )


__all__ = ["DaemonProcess", "daemon_app"]
