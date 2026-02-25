"""Tests for workspace-independent job control (pause_event on RunnerContext).

Tasks 1-3: Add asyncio.Event-based pause signaling to replace file-based
pause signals when running inside the daemon.

The design:
- RunnerContext gains a `pause_event` field (asyncio.Event | None)
- JobRunnerBase stores it as `self._pause_event`
- `_check_pause_signal` checks the event first, falls back to file-based
- `_handle_pause_request` clears the event and tolerates state-save failure
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mozart.core.checkpoint import CheckpointState
from mozart.core.config import JobConfig
from mozart.execution.runner.models import GracefulShutdownError, RunnerContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, overrides: dict[str, Any] | None = None) -> JobConfig:
    """Build a minimal JobConfig for testing."""
    base: dict[str, Any] = {
        "name": "test-job",
        "description": "Unit test job",
        "workspace": str(tmp_path / "workspace"),
        "backend": {"type": "claude_cli", "skip_permissions": True},
        "sheet": {"size": 5, "total_items": 10},
        "prompt": {"template": "Process sheet {{ sheet_num }}."},
        "retry": {
            "max_retries": 3,
            "max_completion_attempts": 2,
            "base_delay_seconds": 1.0,
            "exponential_base": 2.0,
            "max_delay_seconds": 60.0,
            "jitter": False,
            "completion_threshold_percent": 60,
        },
        "validations": [],
    }
    if overrides:
        for k, v in overrides.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k].update(v)
            else:
                base[k] = v
    (tmp_path / "workspace").mkdir(exist_ok=True)
    return JobConfig(**base)


def _make_state(job_id: str = "test-job-001", total_sheets: int = 5) -> CheckpointState:
    """Build a minimal CheckpointState for testing."""
    return CheckpointState(
        job_id=job_id,
        job_name="test-job",
        total_sheets=total_sheets,
    )


def _make_runner_base(
    tmp_path: Path,
    *,
    pause_event: asyncio.Event | None = None,
    workspace: str | None = None,
) -> Any:
    """Create a minimal JobRunnerBase with mocked dependencies.

    Uses JobRunnerBase directly (not the full JobRunner mixin chain) to
    keep tests focused on the base layer without pulling in all mixins.
    """
    from mozart.execution.runner.base import JobRunnerBase

    ws = workspace or str(tmp_path / "workspace")
    config = _make_config(tmp_path, {"workspace": ws})
    backend = MagicMock()
    state_backend = AsyncMock()

    ctx = RunnerContext(pause_event=pause_event)
    runner = JobRunnerBase(config, backend, state_backend, context=ctx)
    return runner


# ===========================================================================
# Task 1: pause_event on RunnerContext and JobRunnerBase
# ===========================================================================


class TestPauseEventOnRunnerContext:
    """Task 1: RunnerContext gains a pause_event field."""

    def test_pause_event_default_is_none(self) -> None:
        """RunnerContext() has pause_event == None by default."""
        ctx = RunnerContext()
        assert ctx.pause_event is None

    def test_pause_event_can_be_set(self) -> None:
        """RunnerContext(pause_event=event) stores the event."""
        event = asyncio.Event()
        ctx = RunnerContext(pause_event=event)
        assert ctx.pause_event is event


class TestPauseEventOnJobRunnerBase:
    """Task 1: JobRunnerBase.__init__ extracts pause_event from context."""

    def test_pause_event_stored_on_runner(self, tmp_path: Path) -> None:
        """pause_event is stored as runner._pause_event."""
        event = asyncio.Event()
        runner = _make_runner_base(tmp_path, pause_event=event)
        assert runner._pause_event is event

    def test_pause_event_none_when_no_context(self, tmp_path: Path) -> None:
        """_pause_event is None when no context is provided."""
        from mozart.execution.runner.base import JobRunnerBase

        config = _make_config(tmp_path)
        runner = JobRunnerBase(config, MagicMock(), AsyncMock())
        assert runner._pause_event is None

    def test_pause_event_none_when_context_omits_it(self, tmp_path: Path) -> None:
        """_pause_event is None when context has no pause_event."""
        runner = _make_runner_base(tmp_path, pause_event=None)
        assert runner._pause_event is None


# ===========================================================================
# Task 2: _check_pause_signal checks the event first
# ===========================================================================


class TestCheckPauseSignalWithEvent:
    """Task 2: _check_pause_signal prefers asyncio.Event over file-based signal."""

    def test_returns_true_when_event_is_set(self, tmp_path: Path) -> None:
        """When pause_event is set, _check_pause_signal returns True."""
        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(tmp_path, pause_event=event)
        state = _make_state()
        assert runner._check_pause_signal(state) is True

    def test_returns_false_when_event_not_set(self, tmp_path: Path) -> None:
        """When pause_event exists but is not set, returns False (no file either)."""
        event = asyncio.Event()
        runner = _make_runner_base(tmp_path, pause_event=event)
        state = _make_state()
        # Workspace exists but no pause file and event not set
        assert runner._check_pause_signal(state) is False

    def test_falls_back_to_file_when_no_event(self, tmp_path: Path) -> None:
        """Without pause_event, file-based signal detection still works."""
        runner = _make_runner_base(tmp_path)
        state = _make_state(job_id="file-test-job")

        # No file -> False
        assert runner._check_pause_signal(state) is False

        # Create pause signal file
        ws = Path(runner.config.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / ".mozart-pause-file-test-job").touch()
        assert runner._check_pause_signal(state) is True

    def test_file_check_tolerates_missing_workspace(self, tmp_path: Path) -> None:
        """File-based check doesn't crash if workspace directory doesn't exist."""
        runner = _make_runner_base(
            tmp_path, workspace=str(tmp_path / "nonexistent" / "workspace")
        )
        state = _make_state()
        # Should return False, not raise OSError
        assert runner._check_pause_signal(state) is False

    def test_event_takes_priority_over_file(self, tmp_path: Path) -> None:
        """When event is set, file is not checked (event takes priority)."""
        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(tmp_path)
        runner._pause_event = event

        state = _make_state()
        # Even without any file, event being set returns True
        assert runner._check_pause_signal(state) is True


# ===========================================================================
# Task 3: _handle_pause_request resilience
# ===========================================================================


class TestHandlePauseRequestResilience:
    """Task 3: _handle_pause_request tolerates state-save failure and clears event."""

    @pytest.mark.asyncio
    async def test_pause_succeeds_when_state_save_fails(self, tmp_path: Path) -> None:
        """GracefulShutdownError is raised even if state_backend.save raises OSError."""
        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(tmp_path, pause_event=event)

        # Make state_backend.save raise OSError
        runner.state_backend.save = AsyncMock(side_effect=OSError("disk full"))

        state = _make_state()
        with pytest.raises(GracefulShutdownError):
            await runner._handle_pause_request(state, current_sheet=2)

        # mark_job_paused was called before save attempt
        assert state.status.value == "paused"

    @pytest.mark.asyncio
    async def test_pause_clears_event_after_handling(self, tmp_path: Path) -> None:
        """After handling pause, the event is cleared (so it can be re-used)."""
        event = asyncio.Event()
        event.set()
        runner = _make_runner_base(tmp_path, pause_event=event)

        state = _make_state()
        with pytest.raises(GracefulShutdownError):
            await runner._handle_pause_request(state, current_sheet=3)

        assert not event.is_set(), "pause_event should be cleared after handling"

    @pytest.mark.asyncio
    async def test_pause_works_without_event(self, tmp_path: Path) -> None:
        """_handle_pause_request works normally when no pause_event is set."""
        runner = _make_runner_base(tmp_path)

        state = _make_state()
        with pytest.raises(GracefulShutdownError):
            await runner._handle_pause_request(state, current_sheet=1)

        assert state.status.value == "paused"

    @pytest.mark.asyncio
    async def test_pause_still_logs_on_save_failure(self, tmp_path: Path) -> None:
        """State-save failure is logged as warning, not swallowed silently."""
        runner = _make_runner_base(tmp_path)
        runner.state_backend.save = AsyncMock(side_effect=OSError("disk full"))

        state = _make_state()
        with pytest.raises(GracefulShutdownError):
            await runner._handle_pause_request(state, current_sheet=1)
