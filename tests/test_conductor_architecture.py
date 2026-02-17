"""RED-phase reproducer tests for 6 conductor architecture bugs.

Each test asserts CORRECT behavior (post-fix). Against the current
buggy code, all 6 tests must FAIL — proving each bug exists.

Ref: .conductor-fix-workspace/01-audit.md
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mozart.core.checkpoint import CheckpointState
from mozart.daemon.config import DaemonConfig
from mozart.daemon.manager import DaemonJobStatus, JobManager, JobMeta


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def daemon_config(tmp_path: Path) -> DaemonConfig:
    """Create a DaemonConfig with low limits for testing."""
    return DaemonConfig(
        max_concurrent_jobs=2,
        pid_file=tmp_path / "test.pid",
        state_db_path=tmp_path / "test-registry.db",
    )


@pytest.fixture
async def manager(daemon_config: DaemonConfig) -> AsyncIterator[JobManager]:
    """Create a JobManager with mocked JobService and opened registry."""
    mgr = JobManager(daemon_config)
    await mgr._registry.open()
    mgr._service = MagicMock()
    # Prevent fire-and-forget save_checkpoint tasks from racing with
    # registry.close() in teardown.
    mgr._registry.save_checkpoint = AsyncMock()
    yield mgr
    await mgr._registry.close()


def _make_checkpoint(job_id: str, **kwargs) -> CheckpointState:
    """Create a minimal CheckpointState for testing."""
    defaults = {
        "job_id": job_id,
        "job_name": job_id,
        "status": "running",
        "last_completed_sheet": 0,
        "total_sheets": 5,
        "sheets": {},
        "start_time": 1000.0,
    }
    defaults.update(kwargs)
    return CheckpointState(**defaults)


# ─── BUG-1: State job_id must match conductor ID ──────────────────────


class TestBug1StateJobIdMatchesConductorId:
    """BUG-1: _resolve_conductor_key relies on a linear scan (first
    running job without live state) when state.job_id doesn't match
    any conductor key. With multiple running jobs, the scan picks
    the wrong job — insertion order, not the actual owner.

    The root cause: there is no explicit config.name → conductor_id
    mapping populated at job start time. The heuristic scan is fragile.

    FAILS because beta's state is misrouted to alpha (first in dict).
    """

    @pytest.mark.asyncio
    async def test_state_job_id_matches_conductor_id(
        self, manager: JobManager,
    ) -> None:
        # Two running jobs: "alpha" (from alpha.yaml) and "beta" (from beta.yaml)
        # Neither has live state yet.
        manager._job_meta["alpha"] = JobMeta(
            job_id="alpha",
            config_path=Path("/tmp/alpha.yaml"),
            workspace=Path("/tmp/ws1"),
            status=DaemonJobStatus.RUNNING,
        )
        manager._job_meta["beta"] = JobMeta(
            job_id="beta",
            config_path=Path("/tmp/beta.yaml"),
            workspace=Path("/tmp/ws2"),
            status=DaemonJobStatus.RUNNING,
        )

        # "beta"'s job publishes first — config.name = "code-reviewer"
        # differs from config_path.stem = "beta".
        # _resolve_conductor_key: "code-reviewer" not in _job_meta,
        # so it scans for the first running job without live state.
        # Dict insertion order: alpha, beta → picks "alpha" (WRONG).
        state = _make_checkpoint("code-reviewer", last_completed_sheet=2)
        manager._on_state_published(state)

        # BUG: beta's state was stored under "alpha" because the linear
        # scan has no way to know which job this state belongs to.
        # With an explicit config.name → conductor_id mapping, this
        # would route correctly to "beta".
        assert "beta" in manager._live_states, (
            "State from beta's job (config.name='code-reviewer') was "
            "misrouted to 'alpha' by the linear scan in "
            "_resolve_conductor_key. An explicit config.name → "
            "conductor_id mapping would prevent this."
        )


# ─── BUG-2: Concurrent same-config jobs must not clobber each other ───


class TestBug2ConcurrentSameConfigNoClobber:
    """BUG-2: Two concurrent jobs from the same config both publish
    states with job_id = config.name = "foo". After initial routing
    (which may work for the first publish), the second job's LATER
    publishes still go to "foo" because its job_id matches the
    _job_meta key directly and the sheet number is higher (no
    regression detected).

    The anti-clobber heuristic in _resolve_conductor_key only
    detects regression (lower sheet number). It cannot distinguish
    forward progress from different jobs that share the same
    config.name.

    FAILS because foo-2's sheet-4 publish overwrites foo's sheet-3 state.
    """

    @pytest.mark.asyncio
    async def test_concurrent_same_config_no_clobber(
        self, manager: JobManager,
    ) -> None:
        # Two concurrent jobs from the same config.
        # First job: conductor ID "foo", second: "foo-2" (deduped).
        manager._job_meta["foo"] = JobMeta(
            job_id="foo",
            config_path=Path("/tmp/foo.yaml"),
            workspace=Path("/tmp/ws1"),
            status=DaemonJobStatus.RUNNING,
        )
        manager._job_meta["foo-2"] = JobMeta(
            job_id="foo-2",
            config_path=Path("/tmp/foo.yaml"),
            workspace=Path("/tmp/ws2"),
            status=DaemonJobStatus.RUNNING,
        )

        # First job publishes sheet 3 (config.name = "foo")
        state_a = _make_checkpoint("foo", last_completed_sheet=3)
        manager._on_state_published(state_a)

        # Second job publishes sheet 1 (also job_id = "foo")
        # Anti-clobber routes this to "foo-2" (regression detected).
        state_b = _make_checkpoint("foo", last_completed_sheet=1)
        manager._on_state_published(state_b)

        # Now second job progresses to sheet 4, publishes again.
        # conductor_key = "foo" (direct match), 4 > 3 (no regression).
        # Stores under "foo" — CLOBBERS first job's state!
        state_c = _make_checkpoint("foo", last_completed_sheet=4)
        manager._on_state_published(state_c)

        # BUG: foo's state was overwritten by foo-2's publish.
        # First job's actual progress (sheet 3) is lost.
        assert manager._live_states["foo"].last_completed_sheet == 3, (
            f"Job 'foo' state was clobbered by the second job's "
            f"publish. Expected sheet 3, got "
            f"{manager._live_states['foo'].last_completed_sheet}. "
            "Without an explicit config.name → conductor_id mapping, "
            "_resolve_conductor_key cannot distinguish forward "
            "progress from different jobs with the same config.name."
        )


# ─── BUG-3: Completed job must retain live state ──────────────────────


class TestBug3CompletedJobRetainsLiveState:
    """BUG-3: _on_task_done retains live states for terminal jobs
    (completed, paused, failed) but never cleans them up.
    _prune_job_history only evicts from _job_meta, not _live_states.

    Over time, completed job states accumulate in _live_states with
    no eviction path, causing unbounded memory growth in long-running
    conductors.

    FAILS because pruning removes the job from _job_meta but leaves
    the orphaned live state in _live_states.
    """

    @pytest.mark.asyncio
    async def test_completed_job_retains_live_state(
        self, manager: JobManager,
    ) -> None:
        # Fill up more terminal jobs than max_job_history allows.
        # Default max_job_history is large, so set a lower value.
        manager._config = manager._config.model_copy(
            update={"max_job_history": 2},
        )

        # Create 4 completed jobs, each with live state.
        for i in range(4):
            jid = f"job-{i}"
            manager._live_states[jid] = _make_checkpoint(
                jid, last_completed_sheet=5, status="completed",
            )
            manager._job_meta[jid] = JobMeta(
                job_id=jid,
                config_path=Path(f"/tmp/test-{i}.yaml"),
                workspace=Path(f"/tmp/workspace-{i}"),
                status=DaemonJobStatus.COMPLETED,
                submitted_at=float(i),
            )

        # Simulate one more task completion to trigger pruning.
        async def _done():
            pass

        task = asyncio.create_task(_done())
        manager._jobs["job-3"] = task
        await task
        manager._on_task_done("job-3", task)

        # _prune_job_history evicts oldest from _job_meta (job-0, job-1)
        assert "job-0" not in manager._job_meta, "Sanity: job-0 should be pruned from meta"

        # BUG: _prune_job_history does NOT clean _live_states — orphaned
        # entries remain, consuming memory indefinitely.
        assert "job-0" not in manager._live_states, (
            "Pruned job 'job-0' was removed from _job_meta but its "
            "live state remains orphaned in _live_states. "
            "_prune_job_history must also evict from _live_states "
            "to prevent unbounded memory growth."
        )


# ─── BUG-4: submit_job must use _id_gen_lock ──────────────────────────


class TestBug4SubmitJobUsesLock:
    """BUG-4: _id_gen_lock exists (line 114) but is never acquired
    in submit_job(). This leaves a TOCTOU race window between the
    duplicate check (line 374) and the registry insert (line 447).

    FAILS because two concurrent submissions both pass the duplicate
    check before either reaches the insert.
    """

    @pytest.mark.asyncio
    async def test_submit_job_uses_id_gen_lock(self, tmp_path: Path) -> None:
        config = DaemonConfig(
            max_concurrent_jobs=4,
            state_db_path=tmp_path / "reg.db",
        )
        mgr = JobManager(config)
        await mgr._registry.open()
        mgr._service = MagicMock()

        try:
            # Create a valid config file
            config_file = tmp_path / "race-test.yaml"
            config_file.write_text(
                "name: race-test\n"
                "sheet:\n"
                "  size: 1\n"
                "  total_items: 1\n"
                "prompt:\n"
                "  template: test prompt\n"
            )

            # Make register_job slow to open the race window.
            # Between the duplicate check and the meta insert, there's an
            # `await` on register_job — if the lock isn't held, the second
            # coroutine passes the duplicate check during this await.
            original_register = mgr._registry.register_job

            async def slow_register(*args, **kwargs):
                await asyncio.sleep(0.05)
                return await original_register(*args, **kwargs)

            mgr._registry.register_job = slow_register

            from mozart.daemon.types import JobRequest

            r1 = JobRequest(config_path=config_file, workspace=tmp_path / "ws1")
            r2 = JobRequest(config_path=config_file, workspace=tmp_path / "ws2")

            # Submit both concurrently
            resp1, resp2 = await asyncio.gather(
                mgr.submit_job(r1),
                mgr.submit_job(r2),
            )

            # BUG: Without the lock, both pass the duplicate check
            statuses = sorted([resp1.status, resp2.status])
            assert statuses == ["accepted", "rejected"], (
                f"Expected one accepted + one rejected, got: "
                f"resp1={resp1.status}, resp2={resp2.status}. "
                "submit_job() doesn't acquire _id_gen_lock."
            )
        finally:
            await mgr._registry.close()


# ─── BUG-5: is_daemon_running must short-circuit on missing path ──────


class TestBug5IsDaemonRunningShortCircuit:
    """BUG-5: is_daemon_running() always calls open_unix_connection
    even when the socket file doesn't exist. Unlike _connect() which
    checks self._socket_path.exists() first, is_daemon_running() has
    its own inline connection logic without this guard.

    FAILS because open_unix_connection IS called for a non-existent
    path — the method relies on the exception handler instead of
    short-circuiting.
    """

    @pytest.mark.asyncio
    async def test_is_daemon_running_skips_connect_on_missing_path(
        self,
    ) -> None:
        from mozart.daemon.ipc.client import DaemonClient

        # Use a path that definitely doesn't exist
        client = DaemonClient(Path("/tmp/nonexistent-mozart-test-socket"))

        with patch("asyncio.open_unix_connection") as mock_connect:
            # The socket doesn't exist; should short-circuit and return
            # False WITHOUT calling open_unix_connection
            mock_connect.side_effect = FileNotFoundError("no such socket")
            result = await client.is_daemon_running()

        assert result is False

        # BUG: open_unix_connection WAS called even though the socket
        # path doesn't exist. is_daemon_running() should check
        # self._socket_path.exists() first, like _connect() does.
        mock_connect.assert_not_called(), (
            "is_daemon_running() should not call open_unix_connection "
            "when the socket path doesn't exist."
        )


# ─── BUG-6: ValueError must be logged at warning, not debug ───────────


class TestBug6TryDaemonRouteValueErrorHandling:
    """BUG-6: When a running daemon returns malformed JSON, the
    ValueError is caught and logged at DEBUG level as
    "daemon_route_protocol_error". Since the daemon IS running but
    misbehaving, this should be logged at WARNING level so operators
    notice the problem.

    FAILS because the current code logs at debug level.
    """

    @pytest.mark.asyncio
    async def test_try_daemon_route_valueerror_logged_as_warning(
        self,
    ) -> None:
        from mozart.daemon.detect import try_daemon_route

        with (
            patch(
                "mozart.daemon.ipc.client.DaemonClient",
            ) as MockClientClass,
            patch(
                "mozart.daemon.detect._logger",
            ) as mock_logger,
        ):
            mock_client = MagicMock()
            MockClientClass.return_value = mock_client
            mock_client.is_daemon_running = AsyncMock(return_value=True)
            mock_client.call = AsyncMock(
                side_effect=ValueError("bad json response"),
            )

            routed, result = await try_daemon_route("test.method", {})

        assert routed is False
        assert result is None

        # BUG: ValueError is logged via _logger.debug (line 97), not
        # _logger.warning. For a running daemon sending bad responses,
        # this should be warning-level.
        warning_msgs = [
            call.args[0] for call in mock_logger.warning.call_args_list
            if call.args
        ]
        assert "daemon_route_protocol_error" in warning_msgs, (
            "ValueError from a running daemon should be logged at WARNING "
            "level as 'daemon_route_protocol_error', but it was logged at "
            "DEBUG level instead. Operators won't notice the misbehaving daemon."
        )
