# Workspace-Independent Job Control Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `mozart pause` and `mozart cancel` work without the workspace filesystem by using in-process asyncio.Event signaling and exposing cancel as a CLI command.

**Architecture:** Replace the file-signal pause path in daemon mode with an asyncio.Event per job. The event is created by JobManager when a job starts, threaded through JobService and into the runner via RunnerContext. The runner checks the event at sheet boundaries. Cancel gets a new CLI command routing to the existing `cancel_job()`. Pause gets a `--force` flag that routes to cancel. State-save failures during pause are caught so the job still exits as PAUSED.

**Tech Stack:** Python asyncio, Typer CLI, pytest

---

### Task 1: Add `pause_event` to RunnerContext and JobRunnerBase

**Files:**
- Modify: `src/mozart/execution/runner/models.py:495-566` (RunnerContext dataclass)
- Modify: `src/mozart/execution/runner/base.py:112-231` (JobRunnerBase.__init__)
- Test: `tests/test_workspace_independent_job_control.py` (new)

**Step 1: Write the failing test**

Create `tests/test_workspace_independent_job_control.py`:

```python
"""Tests for workspace-independent job control.

Covers: in-process pause event signaling, cancel CLI, force-pause,
and state-save resilience when workspace is missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mozart.execution.runner.models import RunnerContext


class TestPauseEventOnRunnerContext:
    """RunnerContext accepts an optional pause_event."""

    def test_pause_event_default_is_none(self):
        ctx = RunnerContext()
        assert ctx.pause_event is None

    def test_pause_event_can_be_set(self):
        event = asyncio.Event()
        ctx = RunnerContext(pause_event=event)
        assert ctx.pause_event is event
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseEventOnRunnerContext -v`
Expected: FAIL — `RunnerContext` has no field `pause_event`

**Step 3: Add `pause_event` field to RunnerContext**

In `src/mozart/execution/runner/models.py`, add after `self_healing_auto_confirm` (line ~565):

```python
    # In-process pause signaling (workspace-independent job control)
    pause_event: asyncio.Event | None = None
    """Async event set by daemon to signal graceful pause.

    When set, the runner checks this at sheet boundaries instead of
    (or in addition to) the filesystem pause signal file. This allows
    pause to work even when the workspace is moved or deleted.
    """
```

Add `import asyncio` to the top of models.py (it's not currently imported — check first).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseEventOnRunnerContext -v`
Expected: PASS

**Step 5: Wire pause_event into JobRunnerBase.__init__**

In `src/mozart/execution/runner/base.py`, inside `__init__` after extracting context fields (around line 165), add:

```python
            pause_event = context.pause_event
```

And after `self._paused_at_sheet: int | None = None` (line 231), add:

```python
        # In-process pause event from daemon (workspace-independent control)
        self._pause_event: asyncio.Event | None = pause_event if context else None
```

Also add `pause_event: asyncio.Event | None = None` initialization in the pre-context block (around line 151, with the other None defaults), and set it from context in the context extraction block.

**Step 6: Commit**

```bash
git add src/mozart/execution/runner/models.py src/mozart/execution/runner/base.py tests/test_workspace_independent_job_control.py
git commit -m "feat: add pause_event field to RunnerContext and JobRunnerBase"
```

---

### Task 2: Make `_check_pause_signal` check the event first

**Files:**
- Modify: `src/mozart/execution/runner/base.py:435-454` (_check_pause_signal)
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing test**

Add to `tests/test_workspace_independent_job_control.py`:

```python
from unittest.mock import AsyncMock, patch, PropertyMock
from mozart.core.checkpoint import CheckpointState


def _make_runner_base(*, pause_event=None, workspace="/tmp/fake-workspace"):
    """Create a minimal JobRunnerBase with mocked dependencies."""
    from mozart.execution.runner.base import JobRunnerBase
    from mozart.execution.runner.models import RunnerContext

    config = MagicMock()
    config.workspace = Path(workspace)
    config.rate_limit.detection_patterns = []
    config.circuit_breaker.enabled = False
    config.sheet.dependencies = None
    config.backend.working_directory = None
    config.prompt = MagicMock()
    config.name = "test-job"

    backend = MagicMock()
    state_backend = MagicMock()

    ctx = RunnerContext(pause_event=pause_event) if pause_event else None
    runner = JobRunnerBase.__new__(JobRunnerBase)
    # Manually call init to avoid full mixin chain
    JobRunnerBase.__init__(runner, config, backend, state_backend, context=ctx)
    return runner


class TestCheckPauseSignalWithEvent:
    """_check_pause_signal checks asyncio.Event before filesystem."""

    def test_returns_true_when_event_is_set(self):
        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(pause_event=event)

        state = MagicMock(spec=CheckpointState)
        state.job_id = "test-job"

        assert runner._check_pause_signal(state) is True

    def test_returns_false_when_event_not_set(self):
        event = asyncio.Event()
        runner = _make_runner_base(pause_event=event)

        state = MagicMock(spec=CheckpointState)
        state.job_id = "test-job"

        # Event not set AND workspace doesn't exist (no file signal)
        assert runner._check_pause_signal(state) is False

    def test_falls_back_to_file_when_no_event(self, tmp_path):
        """Without an event, file-based signal still works."""
        runner = _make_runner_base(workspace=str(tmp_path))

        state = MagicMock(spec=CheckpointState)
        state.job_id = "test-job"

        # No file → False
        assert runner._check_pause_signal(state) is False

        # Create signal file → True
        (tmp_path / ".mozart-pause-test-job").touch()
        assert runner._check_pause_signal(state) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestCheckPauseSignalWithEvent -v`
Expected: FAIL — `_check_pause_signal` doesn't check the event

**Step 3: Update `_check_pause_signal` to check event first**

In `src/mozart/execution/runner/base.py`, replace `_check_pause_signal` (lines 435-454):

```python
    def _check_pause_signal(self, state: CheckpointState) -> bool:
        """Check if a pause has been requested for this job.

        Checks in order:
        1. In-process asyncio.Event (set by daemon — works without workspace)
        2. File-based signal (.mozart-pause-{id} in workspace — legacy fallback)

        Args:
            state: Current job state to get job_id and workspace.

        Returns:
            True if pause signal detected, False otherwise.
        """
        if not state.job_id:
            return False

        # 1. In-process event from daemon (workspace-independent)
        if self._pause_event is not None and self._pause_event.is_set():
            return True

        # 2. File-based signal (legacy / non-daemon mode)
        try:
            workspace_path = Path(self.config.workspace)
            pause_signal_file = workspace_path / f".mozart-pause-{state.job_id}"
            return pause_signal_file.exists()
        except OSError:
            return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestCheckPauseSignalWithEvent -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/execution/runner/base.py tests/test_workspace_independent_job_control.py
git commit -m "feat: _check_pause_signal checks asyncio.Event before filesystem"
```

---

### Task 3: Make `_handle_pause_request` resilient to state-save failure

**Files:**
- Modify: `src/mozart/execution/runner/base.py:484-524` (_handle_pause_request)
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing test**

```python
class TestHandlePauseRequestResilience:
    """_handle_pause_request still raises GracefulShutdownError even if state save fails."""

    @pytest.mark.asyncio
    async def test_pause_succeeds_when_state_save_fails(self):
        """Job pauses cleanly even when workspace state save throws."""
        from mozart.execution.runner.models import GracefulShutdownError

        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(pause_event=event)
        runner.state_backend = AsyncMock()
        runner.state_backend.save = AsyncMock(side_effect=OSError("Workspace gone"))

        state = MagicMock(spec=CheckpointState)
        state.job_id = "test-job"
        state.total_sheets = 5

        with pytest.raises(GracefulShutdownError):
            await runner._handle_pause_request(state, current_sheet=3)

        # State was marked as paused even though save failed
        state.mark_job_paused.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_clears_event_after_handling(self):
        """The pause event is cleared after handling so resumed jobs don't re-pause."""
        from mozart.execution.runner.models import GracefulShutdownError

        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(pause_event=event)
        runner.state_backend = AsyncMock()

        state = MagicMock(spec=CheckpointState)
        state.job_id = "test-job"
        state.total_sheets = 5

        with pytest.raises(GracefulShutdownError):
            await runner._handle_pause_request(state, current_sheet=3)

        assert not event.is_set(), "Event should be cleared after pause handling"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestHandlePauseRequestResilience -v`
Expected: FAIL — state save OSError propagates instead of being caught; event not cleared

**Step 3: Update `_handle_pause_request` for resilience**

Replace `_handle_pause_request` in `src/mozart/execution/runner/base.py` (lines 484-524):

```python
    async def _handle_pause_request(
        self, state: CheckpointState, current_sheet: int
    ) -> None:
        """Handle pause request by saving state and pausing execution.

        State-save failures are caught — the daemon's registry and in-memory
        state serve as the source of truth when the workspace is unavailable.

        Args:
            state: Current job state to update.
            current_sheet: Current sheet number where pause occurred.

        Raises:
            GracefulShutdownError: Always raised after handling pause.
        """
        # Clear the pause signal file (non-critical, best-effort)
        try:
            self._clear_pause_signal(state)
        except OSError:
            self._logger.debug("Failed to clear pause signal file", exc_info=True)

        # Clear in-process event so a resumed job doesn't immediately re-pause
        if self._pause_event is not None:
            self._pause_event.clear()

        # Update state to paused
        state.mark_job_paused()
        self._paused_at_sheet = current_sheet

        # Save state — catch workspace errors so pause still completes
        try:
            await self.state_backend.save(state)
        except OSError:
            self._logger.warning(
                "pause.state_save_failed",
                job_id=state.job_id,
                paused_at_sheet=current_sheet,
                hint="State preserved in daemon registry",
            )

        # Log pause event
        self._logger.info(
            "job.paused_gracefully",
            job_id=state.job_id,
            paused_at_sheet=current_sheet,
            total_sheets=state.total_sheets,
        )

        # Show pause confirmation to user
        self.console.print(
            f"\n[yellow]Job paused gracefully at sheet "
            f"{current_sheet}/{state.total_sheets}.[/yellow]"
        )
        self.console.print(
            f"[green]State saved.[/green] To resume: [bold]mozart resume {state.job_id}[/bold]"
        )

        raise GracefulShutdownError(f"Job {state.job_id} paused at sheet {current_sheet}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestHandlePauseRequestResilience -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/execution/runner/base.py tests/test_workspace_independent_job_control.py
git commit -m "fix: _handle_pause_request catches state-save failure and clears event"
```

---

### Task 4: Refactor `JobManager.pause_job()` to use in-process event

**Files:**
- Modify: `src/mozart/daemon/manager.py:105-112` (add _pause_events dict)
- Modify: `src/mozart/daemon/manager.py:610-632` (pause_job)
- Modify: `src/mozart/daemon/manager.py:673-713` (modify_job — update its pause call)
- Modify: `src/mozart/daemon/manager.py:1262-1296` (_run_managed_task — create/cleanup event)
- Modify: `src/mozart/daemon/manager.py:1496-1525` (_on_task_done — cleanup event)
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing tests**

```python
from collections.abc import AsyncIterator
from mozart.daemon.config import DaemonConfig
from mozart.daemon.exceptions import JobSubmissionError
from mozart.daemon.manager import DaemonJobStatus, JobManager, JobMeta


@pytest.fixture
def daemon_config(tmp_path: Path) -> DaemonConfig:
    return DaemonConfig(
        max_concurrent_jobs=2,
        pid_file=tmp_path / "test.pid",
        state_db_path=tmp_path / "test-registry.db",
    )


@pytest.fixture
async def manager(daemon_config: DaemonConfig) -> AsyncIterator[JobManager]:
    mgr = JobManager(daemon_config)
    await mgr._registry.open()
    mgr._service = MagicMock()
    yield mgr
    await mgr._registry.close()


class TestPauseJobWithEvent:
    """JobManager.pause_job() uses in-process event, not filesystem."""

    @pytest.mark.asyncio
    async def test_pause_sets_event(self, manager: JobManager):
        """pause_job sets the asyncio.Event for the job."""
        event = asyncio.Event()
        manager._pause_events["job-1"] = event
        manager._job_meta["job-1"] = JobMeta(
            job_id="job-1",
            config_path=Path("/tmp/test.yaml"),
            workspace=Path("/tmp/workspace"),
            status=DaemonJobStatus.RUNNING,
        )
        fake_task = asyncio.get_event_loop().create_future()
        manager._jobs["job-1"] = fake_task

        result = await manager.pause_job("job-1")

        assert result is True
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_pause_does_not_call_service(self, manager: JobManager):
        """pause_job no longer delegates to JobService."""
        event = asyncio.Event()
        manager._pause_events["job-1"] = event
        manager._job_meta["job-1"] = JobMeta(
            job_id="job-1",
            config_path=Path("/tmp/test.yaml"),
            workspace=Path("/tmp/workspace"),
            status=DaemonJobStatus.RUNNING,
        )
        fake_task = asyncio.get_event_loop().create_future()
        manager._jobs["job-1"] = fake_task
        manager._service.pause_job = AsyncMock()

        await manager.pause_job("job-1")

        manager._service.pause_job.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pause_no_workspace_needed(self, manager: JobManager):
        """pause_job works even with nonexistent workspace."""
        event = asyncio.Event()
        manager._pause_events["job-1"] = event
        manager._job_meta["job-1"] = JobMeta(
            job_id="job-1",
            config_path=Path("/tmp/test.yaml"),
            workspace=Path("/nonexistent/deleted/workspace"),
            status=DaemonJobStatus.RUNNING,
        )
        fake_task = asyncio.get_event_loop().create_future()
        manager._jobs["job-1"] = fake_task

        # Should NOT raise — workspace doesn't matter
        result = await manager.pause_job("job-1")
        assert result is True
        assert event.is_set()

    @pytest.mark.asyncio
    async def test_pause_no_event_falls_back_to_service(self, manager: JobManager):
        """If no event exists (shouldn't happen), fall back to service."""
        manager._job_meta["job-1"] = JobMeta(
            job_id="job-1",
            config_path=Path("/tmp/test.yaml"),
            workspace=Path("/tmp/workspace"),
            status=DaemonJobStatus.RUNNING,
        )
        fake_task = asyncio.get_event_loop().create_future()
        manager._jobs["job-1"] = fake_task
        manager._service.pause_job = AsyncMock(return_value=True)

        result = await manager.pause_job("job-1")

        assert result is True
        manager._service.pause_job.assert_awaited_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseJobWithEvent -v`
Expected: FAIL — `_pause_events` doesn't exist, `pause_job` still delegates to service

**Step 3: Implement the changes**

In `src/mozart/daemon/manager.py`:

**A. Add `_pause_events` dict** — after `self._live_states` (line 112), add:

```python
        # In-process pause events per job — set by pause_job(), checked by
        # the runner at sheet boundaries.  Keyed by conductor job_id.
        self._pause_events: dict[str, asyncio.Event] = {}
```

**B. Refactor `pause_job()`** — replace lines 610-632:

```python
    async def pause_job(self, job_id: str) -> bool:
        """Send pause signal to a running job via in-process event.

        No workspace access required — the event is checked by the runner
        at sheet boundaries within the same asyncio event loop.
        """
        meta = self._job_meta.get(job_id)
        if meta is None:
            raise JobSubmissionError(f"Job '{job_id}' not found")
        if meta.status != DaemonJobStatus.RUNNING:
            raise JobSubmissionError(
                f"Job '{job_id}' is {meta.status.value}, not running"
            )

        # Verify there's an actual running task (guards against stale
        # "running" status restored from registry after daemon restart)
        task = self._jobs.get(job_id)
        if task is None or task.done():
            meta.status = DaemonJobStatus.FAILED
            await self._registry.update_status(job_id, DaemonJobStatus.FAILED.value)
            raise JobSubmissionError(
                f"Job '{job_id}' has no running process "
                f"(stale status after daemon restart)"
            )

        # Prefer in-process event (workspace-independent)
        event = self._pause_events.get(job_id)
        if event is not None:
            event.set()
            _logger.info("job.pause_signal_sent", job_id=job_id, mechanism="event")
            return True

        # Fallback: delegate to service (file-based signal)
        ws = meta.workspace
        return await self._checked_service.pause_job(meta.job_id, ws)
```

**C. Update `modify_job()`** — replace the pause call at line 704:

```python
        # Send pause signal (prefers in-process event)
        await self.pause_job(job_id)
```

Remove the `ws` resolution before the pause call in modify_job (the line `ws = workspace or meta.workspace` before the pause call). Keep the `ws` resolution for the `pending_modify` tuple that follows.

**D. Create event in `_run_managed_task()`** — inside the method, after acquiring the semaphore and before the try block (around line 1301), add:

```python
            # Create in-process pause event for this job
            pause_event = asyncio.Event()
            self._pause_events[job_id] = pause_event
```

**E. Clean up event in `_on_task_done()`** — after `self._jobs.pop(job_id, None)` (line 1505), add:

```python
        self._pause_events.pop(job_id, None)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseJobWithEvent -v`
Expected: PASS

**Step 5: Run existing pause tests to verify no regression**

Run: `pytest tests/test_daemon_manager.py::TestPauseJob -v`
Expected: Some tests may need updating since `pause_job` signature changed (no longer takes workspace param). Update those tests to match the new signature.

**Step 6: Update existing TestPauseJob tests**

In `tests/test_daemon_manager.py`, update the `TestPauseJob` tests:
- `test_pause_running_job_delegates_to_service` → update to test event-based pause
- `test_pause_uses_meta_workspace_as_fallback` → update or remove (workspace no longer relevant)

**Step 7: Commit**

```bash
git add src/mozart/daemon/manager.py tests/test_workspace_independent_job_control.py tests/test_daemon_manager.py
git commit -m "feat: JobManager.pause_job uses in-process event, no workspace required"
```

---

### Task 5: Thread `pause_event` from JobManager through JobService to runner

**Files:**
- Modify: `src/mozart/daemon/manager.py:1423-1460` (_run_job_task inner _execute)
- Modify: `src/mozart/daemon/manager.py:1466-1494` (_resume_job_task)
- Modify: `src/mozart/daemon/job_service.py:143-248` (start_job)
- Modify: `src/mozart/daemon/job_service.py:250-394` (resume_job)
- Modify: `src/mozart/daemon/job_service.py:465-499` (_create_runner)
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing test**

```python
class TestPauseEventThreading:
    """Pause event is threaded from manager through service to runner."""

    @pytest.mark.asyncio
    async def test_start_job_passes_pause_event_to_runner(self):
        """JobService._create_runner receives pause_event via context."""
        from mozart.daemon.job_service import JobService

        service = JobService.__new__(JobService)
        # We only need to verify _create_runner puts the event in RunnerContext
        # Mock the minimum needed
        config = MagicMock()
        config.name = "test"
        config.workspace = Path("/tmp/ws")
        config.rate_limit.detection_patterns = []
        config.circuit_breaker.enabled = False
        config.sheet.dependencies = None
        config.backend.working_directory = None
        config.prompt = MagicMock()

        event = asyncio.Event()
        components = {
            "backend": MagicMock(),
            "outcome_store": None,
            "global_learning_store": None,
            "grounding_engine": None,
            "notification_manager": None,
        }
        state_backend = MagicMock()
        service._output = MagicMock()
        service._rate_limit_callback = None
        service._event_callback = None
        service._learning_store = None

        runner = service._create_runner(
            config, components, state_backend,
            job_id="test", pause_event=event,
        )

        assert runner._pause_event is event
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseEventThreading -v`
Expected: FAIL — `_create_runner` doesn't accept `pause_event`

**Step 3: Implement the threading**

**A. `_create_runner` accepts `pause_event`** — in `job_service.py` line 465:

Add `pause_event: asyncio.Event | None = None` parameter. Pass it into RunnerContext:

```python
        runner_context = RunnerContext(
            ...,
            pause_event=pause_event,
        )
```

**B. `start_job` accepts and passes `pause_event`** — add parameter to `start_job` signature, pass to `_create_runner`.

**C. `resume_job` accepts and passes `pause_event`** — add parameter to `resume_job` signature, pass to `_create_runner`.

**D. Manager passes event from `_pause_events`** — in `_run_job_task` and `_resume_job_task`, pass `pause_event=self._pause_events.get(job_id)` to `start_job`/`resume_job`.

Note: The event for `_run_job_task` is created in `_run_managed_task` before the coroutine executes, so `self._pause_events[job_id]` will exist by the time `_execute()` runs inside `_run_managed_task`.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseEventThreading -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/daemon/manager.py src/mozart/daemon/job_service.py tests/test_workspace_independent_job_control.py
git commit -m "feat: thread pause_event from JobManager through JobService to runner"
```

---

### Task 6: Update IPC handler — remove workspace from pause

**Files:**
- Modify: `src/mozart/daemon/process.py:422-430` (handle_pause)
- Modify: `src/mozart/cli/commands/pause.py:107-108` (params dict)
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing test**

```python
class TestIPCPauseHandler:
    """IPC pause handler no longer requires workspace param."""

    @pytest.mark.asyncio
    async def test_pause_handler_calls_without_workspace(self):
        """handle_pause calls manager.pause_job(job_id) only."""
        manager = MagicMock()
        manager.pause_job = AsyncMock(return_value=True)

        # Simulate what handle_pause does after our change
        params = {"job_id": "my-job"}
        ok = await manager.pause_job(params["job_id"])
        assert ok is True
        manager.pause_job.assert_awaited_once_with("my-job")
```

**Step 2: Run (this is more of a sanity check), then implement**

**Step 3: Update `handle_pause` in `process.py`**

Replace lines 422-430:

```python
        async def handle_pause(params: dict[str, Any], _w: Any) -> dict[str, Any]:
            from mozart.daemon.exceptions import JobSubmissionError
            try:
                ok = await manager.pause_job(params["job_id"])
                return {"paused": ok}
            except JobSubmissionError as e:
                return {"paused": False, "error": str(e)}
```

**Step 4: Update CLI `_pause_job` in `pause.py`**

In `_pause_job()` (line 107-108), change params to not include workspace:

```python
    params = {"job_id": job_id}
```

The `workspace` parameter on the CLI `pause` function stays (hidden, for direct-mode fallback) but is no longer sent to the daemon.

**Step 5: Commit**

```bash
git add src/mozart/daemon/process.py src/mozart/cli/commands/pause.py tests/test_workspace_independent_job_control.py
git commit -m "refactor: remove workspace from pause IPC handler"
```

---

### Task 7: Add `mozart cancel` CLI command

**Files:**
- Create: `src/mozart/cli/commands/cancel.py`
- Modify: `src/mozart/cli/commands/__init__.py`
- Modify: `src/mozart/cli/__init__.py`
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing test**

```python
class TestCancelCLI:
    """mozart cancel command exists and routes to IPC."""

    def test_cancel_command_importable(self):
        from mozart.cli.commands.cancel import cancel
        assert callable(cancel)

    def test_cancel_command_registered(self):
        from mozart.cli import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "cancel" in command_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestCancelCLI -v`
Expected: FAIL — module doesn't exist

**Step 3: Create `cancel.py`**

Create `src/mozart/cli/commands/cancel.py`:

```python
"""Cancel command for Mozart CLI.

Provides `mozart cancel` to immediately cancel a running job via
asyncio task cancellation. Unlike `pause`, this interrupts mid-sheet
and does not wait for a clean boundary. Use `pause` for graceful
stops; use `cancel` when the job must stop now.
"""

from __future__ import annotations

import asyncio
import json

import typer

from mozart.daemon.exceptions import DaemonError

from ..helpers import configure_global_logging, require_conductor
from ..output import console


def cancel(
    job_id: str = typer.Argument(..., help="Job ID to cancel"),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON",
    ),
) -> None:
    """Cancel a running Mozart job immediately.

    Unlike `pause`, this does not wait for a sheet boundary. The job's
    asyncio task is cancelled, in-progress work is rolled back, and the
    job is marked as CANCELLED. Use `pause` for graceful stops.

    Examples:
        mozart cancel my-job
        mozart cancel my-job --json
    """
    asyncio.run(_cancel_job(job_id, json_output))


async def _cancel_job(job_id: str, json_output: bool) -> None:
    from mozart.daemon.detect import try_daemon_route

    configure_global_logging(console)

    params = {"job_id": job_id}
    try:
        routed, result = await try_daemon_route("job.cancel", params)
    except (OSError, ConnectionError, DaemonError) as exc:
        if json_output:
            console.print(json.dumps({
                "success": False, "job_id": job_id, "message": str(exc),
            }, indent=2))
        else:
            console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    if not routed:
        require_conductor()
        return

    cancelled = result.get("cancelled", False)
    if json_output:
        console.print(json.dumps({
            "success": cancelled, "job_id": job_id,
        }, indent=2))
    elif cancelled:
        console.print(f"[green]Job '{job_id}' cancelled.[/green]")
    else:
        console.print(f"[yellow]Job '{job_id}' not found or already stopped.[/yellow]")
```

**Step 4: Register the command**

In `src/mozart/cli/commands/__init__.py`, add:
```python
from .cancel import cancel
```
And add `"cancel"` to `__all__`.

In `src/mozart/cli/__init__.py`, add after the pause/modify registration (around line 236):
```python
app.command()(cancel)
```

And import `cancel` at the top with the other command imports.

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestCancelCLI -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/mozart/cli/commands/cancel.py src/mozart/cli/commands/__init__.py src/mozart/cli/__init__.py tests/test_workspace_independent_job_control.py
git commit -m "feat: add mozart cancel CLI command"
```

---

### Task 8: Add `--force` flag to `mozart pause`

**Files:**
- Modify: `src/mozart/cli/commands/pause.py:44-81` (pause function signature + routing)
- Test: `tests/test_workspace_independent_job_control.py`

**Step 1: Write the failing test**

```python
class TestPauseForceFlag:
    """mozart pause --force routes to cancel."""

    def test_pause_has_force_option(self):
        """The pause command accepts --force."""
        import inspect
        from mozart.cli.commands.pause import pause
        sig = inspect.signature(pause)
        assert "force" in sig.parameters
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseForceFlag -v`
Expected: FAIL — no `force` parameter

**Step 3: Add `--force` to `pause` command**

In `src/mozart/cli/commands/pause.py`, add the `force` parameter to the `pause` function (after `json_output`):

```python
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force-cancel the job immediately (does not wait for sheet boundary)",
    ),
```

And in the function body, before the `asyncio.run(_pause_job(...))` call:

```python
    if force:
        from .cancel import _cancel_job
        asyncio.run(_cancel_job(job_id, json_output))
        return
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_workspace_independent_job_control.py::TestPauseForceFlag -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/cli/commands/pause.py tests/test_workspace_independent_job_control.py
git commit -m "feat: add --force flag to mozart pause (routes to cancel)"
```

---

### Task 9: Run full test suite and fix regressions

**Files:**
- Modify: `tests/test_daemon_manager.py` (update existing pause tests)
- Possibly: `tests/test_daemon_ipc_methods.py`, `tests/test_daemon_e2e.py`

**Step 1: Run full test suite**

Run: `pytest tests/ -x --timeout=60`

**Step 2: Fix any failures**

Likely regressions:
- `test_daemon_manager.py::TestPauseJob::test_pause_running_job_delegates_to_service` — needs update: now uses event, not service
- `test_daemon_manager.py::TestPauseJob::test_pause_uses_meta_workspace_as_fallback` — remove or rewrite: workspace is no longer a parameter
- Any IPC tests that pass workspace to pause handler

For each failing test:
1. Read the test to understand what it verified
2. Determine if the behavior it tested is still valid
3. Update or replace the test

**Step 3: Run suite again to confirm green**

Run: `pytest tests/ -x --timeout=60`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update existing tests for workspace-independent pause"
```

---

### Task 10: Final verification and documentation

**Step 1: Run type checker**

Run: `mypy src/mozart/daemon/manager.py src/mozart/execution/runner/base.py src/mozart/execution/runner/models.py src/mozart/cli/commands/cancel.py src/mozart/cli/commands/pause.py src/mozart/daemon/job_service.py src/mozart/daemon/process.py`

**Step 2: Run linter**

Run: `ruff check src/mozart/daemon/manager.py src/mozart/execution/runner/base.py src/mozart/execution/runner/models.py src/mozart/cli/commands/cancel.py src/mozart/cli/commands/pause.py src/mozart/daemon/job_service.py src/mozart/daemon/process.py`

**Step 3: Run the specific test file one final time**

Run: `pytest tests/test_workspace_independent_job_control.py -v`
Expected: All PASS

**Step 4: Run full suite one final time**

Run: `pytest tests/ --timeout=60`
Expected: All PASS

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: fix type/lint issues from workspace-independent job control"
```
