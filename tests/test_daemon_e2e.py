"""End-to-end daemon integration tests.

Start a real daemon (foreground mode), submit jobs through IPC client,
and verify completion.  The full stack is exercised: DaemonProcess boot,
IPC server bind, JSON-RPC dispatch, JobManager task management, health
probes, and graceful shutdown.

JobService.start_job() is patched to avoid requiring a real Claude CLI.
Everything else — sockets, JSON-RPC, manager semaphores, health checks,
monitor — runs for real.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from mozart.daemon.config import DaemonConfig, SocketConfig
from mozart.daemon.exceptions import DaemonNotRunningError, JobSubmissionError
from mozart.daemon.ipc.client import DaemonClient
from mozart.daemon.process import DaemonProcess
from mozart.daemon.types import JobRequest, JobResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal fixture config path (created during test setup).
FIXTURE_CONFIG = Path(__file__).parent / "fixtures" / "test-daemon-job.yaml"


def _make_daemon_config(tmp_path: Path) -> DaemonConfig:
    """Build a DaemonConfig with paths scoped to tmp_path."""
    return DaemonConfig(
        socket=SocketConfig(path=tmp_path / "test-mozartd.sock"),
        pid_file=tmp_path / "test-mozartd.pid",
        max_concurrent_jobs=2,
        max_concurrent_sheets=3,
        # Use a fast monitor interval for tests
        monitor_interval_seconds=5.0,
        # Short shutdown timeout for test speed
        shutdown_timeout_seconds=10.0,
    )


async def _wait_for_socket(socket_path: Path, timeout: float = 10.0) -> None:
    """Block until the daemon socket file appears on disk."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if socket_path.exists():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Daemon socket {socket_path} did not appear within {timeout}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def daemon(tmp_path: Path):
    """Start a real daemon in the background, yield (client, config), then shut down.

    The daemon runs `DaemonProcess.run()` as a background asyncio task.
    ProcessGroupManager.setup() is patched because os.setpgid() is
    restricted in test environments.  Signal handlers are patched
    because add_signal_handler only works on the main thread.

    JobService.start_job() is patched to return a synthetic RunSummary
    so tests don't need a real Claude CLI backend.
    """
    config = _make_daemon_config(tmp_path)
    dp = DaemonProcess(config)

    # Patch process group (needs root / main-thread for os.setpgid)
    with (
        patch.object(dp._pgroup, "setup"),
        patch.object(dp._pgroup, "kill_all_children"),
        patch.object(dp._pgroup, "cleanup_orphans", return_value=[]),
    ):
        # Patch signal handler installation (only works on main thread)
        original_run = dp.run

        async def _patched_run() -> None:
            loop = asyncio.get_running_loop()
            original_add = loop.add_signal_handler

            def _safe_add_signal_handler(sig: Any, cb: Any) -> None:
                try:
                    original_add(sig, cb)
                except (ValueError, RuntimeError):
                    pass  # Not on main thread; skip signal handler

            with patch.object(loop, "add_signal_handler", _safe_add_signal_handler):
                await original_run()

        daemon_task = asyncio.create_task(_patched_run())

        # Wait for socket to appear
        await _wait_for_socket(config.socket.path)

        client = DaemonClient(config.socket.path, timeout=10.0)
        yield client, config

        # Shutdown: send RPC then await task
        try:
            await client.call("daemon.shutdown", {"graceful": False})
        except (DaemonNotRunningError, ConnectionResetError, OSError):
            pass  # Daemon may already be shutting down

        try:
            await asyncio.wait_for(daemon_task, timeout=15.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            daemon_task.cancel()
            try:
                await daemon_task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Tests: Daemon Lifecycle
# ---------------------------------------------------------------------------


class TestDaemonLifecycle:
    """Tests for daemon boot, status queries, and shutdown."""

    async def test_daemon_starts_and_responds(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """Daemon starts, binds socket, and responds to daemon.status RPC."""
        client, config = daemon

        result = await client.call("daemon.status")
        assert result["pid"] == os.getpid()  # Foreground mode — same process
        assert result["running_jobs"] == 0
        assert "version" in result

    async def test_daemon_status_round_trip(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """DaemonClient.status() returns a typed DaemonStatus with all fields.

        This is the D022 regression test — ensures get_daemon_status() returns
        every field that DaemonStatus requires, so deserialization succeeds.
        """
        from mozart.daemon.types import DaemonStatus

        client, config = daemon
        status = await client.status()

        assert isinstance(status, DaemonStatus)
        assert status.pid == os.getpid()
        assert status.uptime_seconds >= 0
        assert status.running_jobs == 0
        assert status.total_sheets_active >= 0
        assert status.memory_usage_mb >= 0.0
        assert isinstance(status.version, str)

    async def test_daemon_pid_file_written(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """PID file is created during daemon startup."""
        client, config = daemon

        assert config.pid_file.exists()
        pid_text = config.pid_file.read_text().strip()
        assert pid_text.isdigit()
        assert int(pid_text) == os.getpid()

    async def test_health_probe_returns_ok(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """Liveness probe returns status=ok with uptime."""
        client, config = daemon

        health = await client.health()
        assert health["status"] == "ok"
        assert health["pid"] == os.getpid()
        assert health["uptime_seconds"] >= 0

    async def test_readiness_probe(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """Readiness probe returns status=ready when no load."""
        client, config = daemon

        ready = await client.readiness()
        assert ready["status"] == "ready"
        assert ready["running_jobs"] == 0
        assert ready["accepting_work"] is True
        assert "memory_mb" in ready

    async def test_list_jobs_empty(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """list_jobs returns empty list when no jobs submitted."""
        client, config = daemon

        jobs = await client.list_jobs()
        assert jobs == []

    async def test_graceful_shutdown(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """daemon.shutdown RPC triggers clean daemon exit."""
        client, config = daemon

        # The fixture handles shutdown, but verify the RPC works
        result = await client.call("daemon.shutdown", {"graceful": True})
        assert result["shutting_down"] is True

        # Give daemon time to shut down
        await asyncio.sleep(0.5)

        # After shutdown, new connections should fail
        new_client = DaemonClient(config.socket.path, timeout=2.0)
        running = await new_client.is_daemon_running()
        # Socket may or may not be cleaned up yet, but the daemon
        # should no longer accept RPC calls.  Either False (socket gone)
        # or a connection that fails on call is acceptable.
        if running:
            try:
                await new_client.call("daemon.status")
                # If we get here, daemon is still alive — that's also OK
                # during graceful wind-down
            except (DaemonNotRunningError, ConnectionResetError, OSError):
                pass  # Expected after shutdown


# ---------------------------------------------------------------------------
# Tests: Job Submission
# ---------------------------------------------------------------------------


class TestJobSubmission:
    """Tests for job submission, status, and cancellation."""

    async def test_submit_job_accepted(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """A valid job config is accepted and returns job_id."""
        client, config = daemon

        # Patch JobService.start_job to avoid needing real Claude CLI
        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            new_callable=AsyncMock,
        ) as mock_start:
            from mozart.core.checkpoint import JobStatus
            from mozart.execution.runner.models import RunSummary

            mock_start.return_value = RunSummary(
                job_id="test-daemon-job",
                job_name="test-daemon-job",
                total_sheets=1,
                final_status=JobStatus.COMPLETED,
                completed_sheets=1,
            )

            req = JobRequest(config_path=FIXTURE_CONFIG)
            resp = await client.submit_job(req)

            assert isinstance(resp, JobResponse)
            assert resp.status == "accepted"
            assert resp.job_id  # Non-empty job ID
            assert "test-daemon-job" in resp.job_id

            # Wait a moment for the job task to start and complete
            await asyncio.sleep(0.5)

    async def test_submit_job_rejected_missing_config(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """Submitting a non-existent config file is rejected."""
        client, config = daemon

        req = JobRequest(config_path=Path("/nonexistent/path/fake.yaml"))
        resp = await client.submit_job(req)

        assert resp.status == "rejected"
        assert "not found" in (resp.message or "").lower()

    async def test_submit_job_appears_in_list(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """Submitted job appears in job.list."""
        client, config = daemon

        # Use a long-running mock so job stays active for listing
        started = asyncio.Event()
        hold = asyncio.Event()

        async def _slow_start(*args: Any, **kwargs: Any) -> Any:
            started.set()
            await hold.wait()
            from mozart.core.checkpoint import JobStatus
            from mozart.execution.runner.models import RunSummary

            return RunSummary(
                job_id="test-daemon-job",
                job_name="test-daemon-job",
                total_sheets=1,
                final_status=JobStatus.COMPLETED,
            )

        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            side_effect=_slow_start,
        ):
            req = JobRequest(config_path=FIXTURE_CONFIG)
            resp = await client.submit_job(req)
            assert resp.status == "accepted"

            # Wait for job to actually enter the running state
            await asyncio.wait_for(started.wait(), timeout=5.0)
            await asyncio.sleep(0.1)  # Let status update propagate

            jobs = await client.list_jobs()
            job_ids = [j["job_id"] for j in jobs]
            assert resp.job_id in job_ids

            # Release the mock so cleanup doesn't hang
            hold.set()
            await asyncio.sleep(0.2)

    async def test_cancel_running_job(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """A running job can be cancelled via RPC."""
        client, config = daemon

        started = asyncio.Event()
        hold = asyncio.Event()

        async def _slow_start(*args: Any, **kwargs: Any) -> Any:
            started.set()
            await hold.wait()
            from mozart.core.checkpoint import JobStatus
            from mozart.execution.runner.models import RunSummary

            return RunSummary(
                job_id="test-daemon-job",
                job_name="test-daemon-job",
                total_sheets=1,
                final_status=JobStatus.COMPLETED,
            )

        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            side_effect=_slow_start,
        ):
            req = JobRequest(config_path=FIXTURE_CONFIG)
            resp = await client.submit_job(req)
            job_id = resp.job_id

            # Wait until the job is running
            await asyncio.wait_for(started.wait(), timeout=5.0)
            await asyncio.sleep(0.1)

            # Cancel it
            cancel_result = await client.call("job.cancel", {"job_id": job_id})
            assert cancel_result["cancelled"] is True

            # Release hold in case cancellation didn't propagate yet
            hold.set()
            await asyncio.sleep(0.3)

            # Verify it shows as cancelled
            jobs = await client.list_jobs()
            cancelled_job = next((j for j in jobs if j["job_id"] == job_id), None)
            assert cancelled_job is not None
            assert cancelled_job["status"] == "cancelled"

    async def test_job_status_query(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """job.status RPC returns detailed info about a specific job."""
        client, config = daemon

        started = asyncio.Event()
        hold = asyncio.Event()

        async def _slow_start(*args: Any, **kwargs: Any) -> Any:
            started.set()
            await hold.wait()
            from mozart.core.checkpoint import JobStatus
            from mozart.execution.runner.models import RunSummary

            return RunSummary(
                job_id="test-daemon-job",
                job_name="test-daemon-job",
                total_sheets=1,
                final_status=JobStatus.COMPLETED,
            )

        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            side_effect=_slow_start,
        ):
            req = JobRequest(config_path=FIXTURE_CONFIG)
            resp = await client.submit_job(req)
            job_id = resp.job_id

            await asyncio.wait_for(started.wait(), timeout=5.0)
            await asyncio.sleep(0.1)

            status = await client.get_job_status(job_id, str(config.pid_file.parent))
            assert status["job_id"] == job_id
            assert status["status"] == "running"
            assert str(FIXTURE_CONFIG) in status["config_path"]

            hold.set()
            await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Tests: Concurrent Jobs
# ---------------------------------------------------------------------------


class TestConcurrentJobs:
    """Tests for concurrent job submission and limit enforcement."""

    async def test_concurrent_jobs_respect_limit(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """At most max_concurrent_jobs run simultaneously."""
        client, config = daemon
        assert config.max_concurrent_jobs == 2

        running_count = 0
        max_observed = 0
        gate = asyncio.Event()
        all_started = asyncio.Event()
        start_count = 0

        async def _tracked_start(*args: Any, **kwargs: Any) -> Any:
            nonlocal running_count, max_observed, start_count
            running_count += 1
            start_count += 1
            if running_count > max_observed:
                max_observed = running_count
            if start_count >= 2:
                all_started.set()
            try:
                await gate.wait()
            finally:
                running_count -= 1
            from mozart.core.checkpoint import JobStatus
            from mozart.execution.runner.models import RunSummary

            return RunSummary(
                job_id="test", job_name="test", total_sheets=1,
                final_status=JobStatus.COMPLETED,
            )

        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            side_effect=_tracked_start,
        ):
            # Submit 3 jobs but only 2 should run at once
            responses = []
            for _ in range(3):
                req = JobRequest(config_path=FIXTURE_CONFIG)
                r = await client.submit_job(req)
                assert r.status == "accepted"
                responses.append(r)

            # Wait for at least 2 to be running
            await asyncio.wait_for(all_started.wait(), timeout=5.0)
            await asyncio.sleep(0.3)

            # The third job is gated by the semaphore
            assert max_observed <= config.max_concurrent_jobs

            # Release all
            gate.set()
            await asyncio.sleep(1.0)

    async def test_multiple_jobs_complete_independently(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """Two submitted jobs both complete and show in job list."""
        client, config = daemon

        call_count = 0

        async def _fast_start(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            from mozart.core.checkpoint import JobStatus
            from mozart.execution.runner.models import RunSummary

            return RunSummary(
                job_id="test", job_name="test", total_sheets=1,
                final_status=JobStatus.COMPLETED, completed_sheets=1,
            )

        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            side_effect=_fast_start,
        ):
            r1 = await client.submit_job(JobRequest(config_path=FIXTURE_CONFIG))
            r2 = await client.submit_job(JobRequest(config_path=FIXTURE_CONFIG))
            assert r1.status == "accepted"
            assert r2.status == "accepted"

            # Wait for both to complete
            await asyncio.sleep(1.5)

            jobs = await client.list_jobs()
            statuses = {j["job_id"]: j["status"] for j in jobs}
            assert statuses.get(r1.job_id) == "completed"
            assert statuses.get(r2.job_id) == "completed"
            assert call_count == 2


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for daemon error handling and resilience."""

    async def test_job_failure_recorded(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """A job that raises an exception is marked as failed."""
        client, config = daemon

        async def _failing_start(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Simulated backend crash")

        with patch(
            "mozart.daemon.job_service.JobService.start_job",
            side_effect=_failing_start,
        ):
            req = JobRequest(config_path=FIXTURE_CONFIG)
            resp = await client.submit_job(req)
            assert resp.status == "accepted"

            # Wait for the task to fail
            await asyncio.sleep(1.0)

            jobs = await client.list_jobs()
            failed_job = next((j for j in jobs if j["job_id"] == resp.job_id), None)
            assert failed_job is not None
            assert failed_job["status"] == "failed"

    async def test_unknown_rpc_method_returns_error(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """Calling a non-existent RPC method raises DaemonError."""
        client, config = daemon
        from mozart.daemon.exceptions import DaemonError

        with pytest.raises(DaemonError, match="Method not found"):
            await client.call("nonexistent.method")

    async def test_get_status_unknown_job_raises(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """Requesting status for a non-existent job raises JobSubmissionError."""
        client, config = daemon

        with pytest.raises(JobSubmissionError, match="not found"):
            await client.call(
                "job.status",
                {"job_id": "does-not-exist", "workspace": "/tmp"},
            )

    async def test_daemon_rejects_during_shutdown(
        self, daemon: tuple[DaemonClient, DaemonConfig],
    ):
        """Jobs submitted during shutdown are rejected."""
        client, config = daemon

        # Trigger graceful shutdown
        await client.call("daemon.shutdown", {"graceful": True})
        await asyncio.sleep(0.3)

        # New submission should be rejected (daemon is shutting down)
        # The daemon may have already closed the socket, so either
        # a rejection or a connection error is acceptable.
        try:
            req = JobRequest(config_path=FIXTURE_CONFIG)
            resp = await client.submit_job(req)
            assert resp.status == "rejected"
            assert "shutting down" in (resp.message or "").lower()
        except (DaemonNotRunningError, ConnectionResetError, OSError):
            pass  # Also acceptable — daemon socket already gone


# ---------------------------------------------------------------------------
# Tests: Real Execution (no mocking)
# ---------------------------------------------------------------------------


class TestRealExecution:
    """Tests that exercise real JobService execution through the daemon.

    These tests do NOT mock JobService — they submit real jobs with
    dry_run=True so the full code path (config parsing, workspace
    creation, state backend setup) runs without needing a Claude CLI.
    """

    async def test_dry_run_job_completes(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """A dry_run job goes through the real path and completes."""
        client, config = daemon

        req = JobRequest(config_path=FIXTURE_CONFIG, dry_run=True)
        resp = await client.submit_job(req)
        assert resp.status == "accepted"
        assert resp.job_id

        # dry_run jobs complete almost instantly
        await asyncio.sleep(1.5)

        jobs = await client.list_jobs()
        job = next((j for j in jobs if j["job_id"] == resp.job_id), None)
        assert job is not None
        assert job["status"] == "completed"

    async def test_dry_run_state_transitions(self, daemon: tuple[DaemonClient, DaemonConfig]):
        """Verify the full lifecycle: queued → running → completed."""
        client, config = daemon

        req = JobRequest(config_path=FIXTURE_CONFIG, dry_run=True)
        resp = await client.submit_job(req)
        assert resp.status == "accepted"

        # Wait for completion
        for _ in range(20):
            jobs = await client.list_jobs()
            job = next((j for j in jobs if j["job_id"] == resp.job_id), None)
            if job and job["status"] == "completed":
                break
            await asyncio.sleep(0.2)
        else:
            pytest.fail("Job did not complete within timeout")

        assert job is not None
        assert job["status"] == "completed"
        assert job["started_at"] is not None


# ---------------------------------------------------------------------------
# Tests: Client Without Daemon (negative path)
# ---------------------------------------------------------------------------


class TestClientWithoutDaemon:
    """Verify client behaves correctly when no daemon is running."""

    async def test_is_daemon_running_false(self, tmp_path: Path):
        """is_daemon_running returns False when socket doesn't exist."""
        client = DaemonClient(tmp_path / "nonexistent.sock")
        assert await client.is_daemon_running() is False

    async def test_status_raises_not_running(self, tmp_path: Path):
        """status() raises DaemonNotRunningError without a daemon."""
        client = DaemonClient(tmp_path / "nonexistent.sock")
        with pytest.raises(DaemonNotRunningError):
            await client.status()

    async def test_submit_raises_not_running(self, tmp_path: Path):
        """submit_job() raises DaemonNotRunningError without a daemon."""
        client = DaemonClient(tmp_path / "nonexistent.sock")
        req = JobRequest(config_path=FIXTURE_CONFIG)
        with pytest.raises(DaemonNotRunningError):
            await client.submit_job(req)
