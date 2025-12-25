"""Tests for Mozart JobRunner with graceful shutdown and progress tracking."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mozart.backends.base import ExecutionResult
from mozart.core.checkpoint import BatchStatus, CheckpointState, JobStatus
from mozart.core.config import JobConfig
from mozart.execution.runner import (
    BatchExecutionMode,
    FatalError,
    GracefulShutdownError,
    JobRunner,
    RunSummary,
)
from mozart.execution.validation import BatchValidationResult, ValidationResult
from mozart.state.base import StateBackend


@pytest.fixture
def mock_backend() -> MagicMock:
    """Create a mock backend that returns successful execution results."""
    backend = AsyncMock()
    backend.execute = AsyncMock(
        return_value=ExecutionResult(
            success=True,
            stdout="Task completed",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
        )
    )
    backend.health_check = AsyncMock(return_value=True)
    return backend


@pytest.fixture
def mock_state_backend() -> MagicMock:
    """Create a mock state backend."""
    backend = AsyncMock()
    backend.load = AsyncMock(return_value=None)
    backend.save = AsyncMock()
    backend.list_jobs = AsyncMock(return_value=[])
    return backend


@pytest.fixture
def sample_config() -> JobConfig:
    """Create a sample job configuration for testing."""
    return JobConfig.model_validate({
        "name": "test-job",
        "description": "Test job for runner tests",
        "backend": {
            "type": "claude_cli",
            "skip_permissions": True,
        },
        "batch": {
            "size": 10,
            "total_items": 30,
        },
        "prompt": {
            "template": "Process batch {{ batch_num }} of {{ total_batches }}.",
        },
        "retry": {
            "max_retries": 2,
        },
        "validations": [],  # No validations for simpler tests
    })


class TestGracefulShutdownError:
    """Tests for graceful shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_flag_starts_false(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that shutdown flag is initially false."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )
        assert runner._shutdown_requested is False

    @pytest.mark.asyncio
    async def test_signal_handler_sets_shutdown_flag(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that signal handler sets the shutdown flag."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Call signal handler directly
        runner._signal_handler()

        assert runner._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_signal_handler_only_sets_once(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that signal handler only prints message on first call."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # First call should set flag
        runner._signal_handler()
        assert runner._shutdown_requested is True

        # Second call should not change anything (no exception)
        runner._signal_handler()
        assert runner._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_graceful_shutdown_saves_state_as_paused(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that graceful shutdown saves state with PAUSED status."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Create a state to save
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=3,
            last_completed_batch=1,
            status=JobStatus.RUNNING,
        )

        # Call the graceful shutdown handler
        with pytest.raises(GracefulShutdownError) as exc_info:
            await runner._handle_graceful_shutdown(state)

        # Verify state was saved as PAUSED
        assert state.status == JobStatus.PAUSED
        mock_state_backend.save.assert_called_once_with(state)
        assert "paused by user request" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interruptible_sleep_checks_shutdown_flag(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that interruptible sleep exits early when shutdown is requested."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Set shutdown flag
        runner._shutdown_requested = True

        # Sleep should return almost immediately
        import time
        start = time.monotonic()
        await runner._interruptible_sleep(10.0)  # Long sleep
        elapsed = time.monotonic() - start

        # Should have exited much earlier than 10 seconds
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_run_checks_shutdown_before_each_batch(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that run method checks shutdown flag before each batch."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Mock validation to always pass
        with patch.object(runner, "_execute_batch_with_recovery") as mock_exec:
            # Set shutdown after first batch completes
            async def set_shutdown_after_first(*args: Any) -> None:
                state = args[0]
                batch_num = args[1]
                # First mark the batch as started (creates the batch state)
                state.mark_batch_started(batch_num)
                state.mark_batch_completed(batch_num, validation_passed=True)
                runner._shutdown_requested = True

            mock_exec.side_effect = set_shutdown_after_first

            # Run should stop after first batch due to shutdown
            with pytest.raises(GracefulShutdownError):
                await runner.run()

            # Should have only executed one batch
            assert mock_exec.call_count == 1


class TestProgressTracking:
    """Tests for progress tracking functionality."""

    @pytest.mark.asyncio
    async def test_progress_callback_is_called(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that progress callback is called after each batch."""
        progress_updates: list[tuple[int, int, float | None]] = []

        def track_progress(completed: int, total: int, eta: float | None) -> None:
            progress_updates.append((completed, total, eta))

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            progress_callback=track_progress,
        )

        # Mock the batch execution to just mark batches complete
        with patch.object(runner, "_execute_batch_with_recovery") as mock_exec:
            async def mark_complete(state: CheckpointState, batch_num: int) -> None:
                # First mark the batch as started (creates the batch state)
                state.mark_batch_started(batch_num)
                state.mark_batch_completed(batch_num, validation_passed=True)

            mock_exec.side_effect = mark_complete

            # Run all 3 batches
            await runner.run()

        # Should have 3 progress updates
        assert len(progress_updates) == 3

        # Check progress values
        assert progress_updates[0][0] == 1  # 1 completed
        assert progress_updates[1][0] == 2  # 2 completed
        assert progress_updates[2][0] == 3  # 3 completed

        # All should have total=3
        assert all(total == 3 for _, total, _ in progress_updates)

    @pytest.mark.asyncio
    async def test_eta_is_calculated_from_batch_times(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that ETA is calculated from average batch time."""
        etas: list[float | None] = []

        def track_eta(completed: int, total: int, eta: float | None) -> None:
            etas.append(eta)

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            progress_callback=track_eta,
        )

        # Simulate batch times
        runner._batch_times = [1.0, 2.0, 3.0]  # Avg = 2.0

        # Create a state
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=5,
            last_completed_batch=3,
        )

        # Call progress update
        runner._update_progress(state)

        # ETA should be 2.0 * 2 remaining = 4.0
        assert len(etas) == 1
        assert etas[0] is not None
        assert abs(etas[0] - 4.0) < 0.01

    @pytest.mark.asyncio
    async def test_no_callback_no_error(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that no progress callback doesn't cause errors."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            progress_callback=None,
        )

        # Create a state
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=3,
            last_completed_batch=1,
        )

        # Should not raise
        runner._update_progress(state)


class TestGracefulShutdownErrorException:
    """Tests for the GracefulShutdownError exception."""

    def test_graceful_shutdown_message(self) -> None:
        """Test GracefulShutdownError exception message."""
        exc = GracefulShutdownError("Test message")
        assert str(exc) == "Test message"

    def test_graceful_shutdown_is_exception(self) -> None:
        """Test GracefulShutdownError inherits from Exception."""
        exc = GracefulShutdownError("Test")
        assert isinstance(exc, Exception)

    def test_graceful_shutdown_can_be_caught(self) -> None:
        """Test GracefulShutdownError can be caught separately from other exceptions."""
        caught = False

        try:
            raise GracefulShutdownError("User interrupt")
        except GracefulShutdownError:
            caught = True
        except Exception:
            pass

        assert caught


class TestBatchTiming:
    """Tests for batch timing and ETA calculation."""

    @pytest.mark.asyncio
    async def test_batch_times_are_tracked(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that batch execution times are recorded."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Initially empty
        assert len(runner._batch_times) == 0

        # Mock batch execution with a small delay
        with patch.object(runner, "_execute_batch_with_recovery") as mock_exec:
            async def slow_batch(state: CheckpointState, batch_num: int) -> None:
                await asyncio.sleep(0.1)  # 100ms
                # First mark the batch as started (creates the batch state)
                state.mark_batch_started(batch_num)
                state.mark_batch_completed(batch_num, validation_passed=True)

            mock_exec.side_effect = slow_batch

            # Run all batches
            await runner.run()

        # Should have 3 batch times recorded
        assert len(runner._batch_times) == 3

        # Each time should be at least 100ms
        assert all(t >= 0.1 for t in runner._batch_times)

    def test_eta_calculation_with_no_times(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test ETA returns None when no batch times recorded."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        etas: list[float | None] = []

        def track_eta(completed: int, total: int, eta: float | None) -> None:
            etas.append(eta)

        runner.progress_callback = track_eta

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_batches=3,
            last_completed_batch=1,
        )

        runner._update_progress(state)

        # No batch times = no ETA
        assert etas[0] is None


class TestRunSummary:
    """Tests for RunSummary dataclass and methods."""

    def test_run_summary_initialization(self) -> None:
        """Test RunSummary initializes with correct defaults."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
        )

        assert summary.job_id == "test-job"
        assert summary.job_name == "Test Job"
        assert summary.total_batches == 10
        assert summary.completed_batches == 0
        assert summary.failed_batches == 0
        assert summary.skipped_batches == 0
        assert summary.total_duration_seconds == 0.0
        assert summary.total_retries == 0
        assert summary.final_status == JobStatus.PENDING

    def test_success_rate_calculation(self) -> None:
        """Test success rate percentage calculation."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            completed_batches=8,
            failed_batches=2,
        )

        assert summary.success_rate == 80.0

    def test_success_rate_zero_batches(self) -> None:
        """Test success rate with zero batches returns 0."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=0,
        )

        assert summary.success_rate == 0.0

    def test_validation_pass_rate_calculation(self) -> None:
        """Test validation pass rate calculation."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            validation_pass_count=7,
            validation_fail_count=3,
        )

        assert summary.validation_pass_rate == 70.0

    def test_validation_pass_rate_no_validations(self) -> None:
        """Test validation pass rate with no validations returns 100%."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
        )

        # No validations = 100% pass
        assert summary.validation_pass_rate == 100.0

    def test_first_attempt_rate_calculation(self) -> None:
        """Test first attempt success rate calculation."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            completed_batches=8,
            first_attempt_successes=6,
        )

        assert summary.first_attempt_rate == 75.0

    def test_first_attempt_rate_no_completions(self) -> None:
        """Test first attempt rate with no completed batches returns 0."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            completed_batches=0,
        )

        assert summary.first_attempt_rate == 0.0

    def test_to_dict_structure(self) -> None:
        """Test to_dict returns correct structure."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_batches=10,
            completed_batches=8,
            failed_batches=1,
            skipped_batches=1,
            total_duration_seconds=120.5,
            total_retries=3,
            total_completion_attempts=2,
            rate_limit_waits=1,
            validation_pass_count=8,
            validation_fail_count=0,
            first_attempt_successes=6,
            final_status=JobStatus.COMPLETED,
        )

        result = summary.to_dict()

        assert result["job_id"] == "test-job"
        assert result["job_name"] == "Test Job"
        assert result["status"] == "completed"
        assert result["duration_seconds"] == 120.5
        assert result["duration_formatted"] == "2m 0s"

        assert result["batches"]["total"] == 10
        assert result["batches"]["completed"] == 8
        assert result["batches"]["failed"] == 1
        assert result["batches"]["skipped"] == 1
        assert result["batches"]["success_rate"] == 80.0

        assert result["validation"]["passed"] == 8
        assert result["validation"]["failed"] == 0
        assert result["validation"]["pass_rate"] == 100.0

        assert result["execution"]["total_retries"] == 3
        assert result["execution"]["completion_attempts"] == 2
        assert result["execution"]["rate_limit_waits"] == 1
        assert result["execution"]["first_attempt_successes"] == 6
        assert result["execution"]["first_attempt_rate"] == 75.0

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting for seconds."""
        assert RunSummary._format_duration(45.5) == "45.5s"

    def test_format_duration_minutes(self) -> None:
        """Test duration formatting for minutes."""
        assert RunSummary._format_duration(125) == "2m 5s"

    def test_format_duration_hours(self) -> None:
        """Test duration formatting for hours."""
        assert RunSummary._format_duration(3725) == "1h 2m"


class TestRunnerReturnsRunSummary:
    """Tests for JobRunner returning RunSummary."""

    @pytest.mark.asyncio
    async def test_run_returns_tuple(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that runner.run() returns tuple of (state, summary)."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Mock batch execution
        with patch.object(runner, "_execute_batch_with_recovery") as mock_exec:
            async def complete_batch(state: CheckpointState, batch_num: int) -> None:
                state.mark_batch_started(batch_num)
                state.mark_batch_completed(batch_num, validation_passed=True)

            mock_exec.side_effect = complete_batch

            result = await runner.run()

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        state, summary = result
        assert isinstance(state, CheckpointState)
        assert isinstance(summary, RunSummary)

    @pytest.mark.asyncio
    async def test_run_summary_populated(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that summary is correctly populated after run."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Mock batch execution to succeed with first_attempt_success
        with patch.object(runner, "_execute_batch_with_recovery") as mock_exec:
            async def complete_batch(state: CheckpointState, batch_num: int) -> None:
                state.mark_batch_started(batch_num)
                batch_state = state.batches[batch_num]
                batch_state.first_attempt_success = True
                state.mark_batch_completed(batch_num, validation_passed=True)

            mock_exec.side_effect = complete_batch

            state, summary = await runner.run()

        assert summary.job_id == "test-job"
        assert summary.job_name == "test-job"
        assert summary.total_batches == 3  # sample_config has 3 batches
        assert summary.completed_batches == 3
        assert summary.failed_batches == 0
        assert summary.final_status == JobStatus.COMPLETED
        assert summary.total_duration_seconds > 0

    @pytest.mark.asyncio
    async def test_get_summary_returns_current_summary(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test get_summary() returns the current summary after run."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Before run, summary is None
        assert runner.get_summary() is None

        # Mock batch execution
        with patch.object(runner, "_execute_batch_with_recovery") as mock_exec:
            async def complete_batch(state: CheckpointState, batch_num: int) -> None:
                state.mark_batch_started(batch_num)
                state.mark_batch_completed(batch_num, validation_passed=True)

            mock_exec.side_effect = complete_batch

            await runner.run()

        # After run, summary is populated
        summary = runner.get_summary()
        assert summary is not None
        assert summary.completed_batches == 3
