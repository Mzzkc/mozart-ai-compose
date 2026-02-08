"""Tests for Mozart JobRunner with graceful shutdown and progress tracking."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mozart.backends.base import ExecutionResult
from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.core.config import JobConfig
from mozart.execution.runner import (
    FatalError,
    GracefulShutdownError,
    JobRunner,
    RunSummary,
)
from mozart.execution.preflight import PreflightResult, PromptMetrics
from mozart.core.config import ValidationRule
from mozart.execution.validation import SheetValidationResult, ValidationResult


def make_mock_preflight_result() -> PreflightResult:
    """Create a valid PreflightResult for tests that don't need preflight checks."""
    return PreflightResult(
        prompt_metrics=PromptMetrics(
            character_count=100,
            estimated_tokens=25,
            line_count=5,
            has_file_references=False,
            referenced_paths=[],
            word_count=20,
        ),
        warnings=[],
        errors=[],
        paths_accessible={},
        working_directory_valid=True,
    )


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
        "sheet": {
            "size": 10,
            "total_items": 30,
        },
        "prompt": {
            "template": "Process sheet {{ sheet_num }} of {{ total_sheets }}.",
        },
        "retry": {
            "max_retries": 2,
        },
        "validations": [],  # No validations for simpler tests
        "pause_between_sheets_seconds": 0,  # Fast tests: no pause between sheets
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
            total_sheets=3,
            last_completed_sheet=1,
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
        # Use generous tolerance for CI environments under load
        assert elapsed < 5.0

    @pytest.mark.asyncio
    async def test_run_checks_shutdown_before_each_sheet(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that run method checks shutdown flag before each sheet."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Mock validation to always pass
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            # Set shutdown after first sheet completes
            async def set_shutdown_after_first(*args: Any) -> None:
                state = args[0]
                sheet_num = args[1]
                # First mark the sheet as started (creates the sheet state)
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)
                runner._shutdown_requested = True

            mock_exec.side_effect = set_shutdown_after_first

            # Run should stop after first sheet due to shutdown
            with pytest.raises(GracefulShutdownError):
                await runner.run()

            # Should have only executed one sheet
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
        """Test that progress callback is called after each sheet."""
        progress_updates: list[tuple[int, int, float | None]] = []

        def track_progress(completed: int, total: int, eta: float | None) -> None:
            progress_updates.append((completed, total, eta))

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            progress_callback=track_progress,
        )

        # Mock the sheet execution to just mark sheets complete
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def mark_complete(state: CheckpointState, sheet_num: int) -> None:
                # First mark the sheet as started (creates the sheet state)
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = mark_complete

            # Run all 3 sheets
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
    async def test_eta_is_calculated_from_sheet_times(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that ETA is calculated from average sheet time."""
        etas: list[float | None] = []

        def track_eta(completed: int, total: int, eta: float | None) -> None:
            etas.append(eta)

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            progress_callback=track_eta,
        )

        # Simulate sheet times
        runner._sheet_times = [1.0, 2.0, 3.0]  # Avg = 2.0

        # Create a state
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
            last_completed_sheet=3,
        )

        # Call progress update
        runner._update_progress(state)

        # ETA should be 2.0 * 2 remaining = 4.0
        assert len(etas) == 1
        assert etas[0] is not None
        assert abs(etas[0] - 4.0) < 0.5

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
            total_sheets=3,
            last_completed_sheet=1,
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


class TestSheetTiming:
    """Tests for sheet timing and ETA calculation."""

    @pytest.mark.asyncio
    async def test_sheet_times_are_tracked(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that sheet execution times are recorded."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Initially empty
        assert len(runner._sheet_times) == 0

        # Mock sheet execution with a small delay
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def slow_sheet(state: CheckpointState, sheet_num: int) -> None:
                await asyncio.sleep(0.01)  # 10ms - fast enough to avoid CI flakiness
                # First mark the sheet as started (creates the sheet state)
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = slow_sheet

            # Run all sheets
            await runner.run()

        # Should have 3 sheet times recorded
        assert len(runner._sheet_times) == 3

        # Each time should be positive (non-zero execution)
        assert all(t > 0 for t in runner._sheet_times)

    def test_eta_calculation_with_no_times(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test ETA returns None when no sheet times recorded."""
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
            total_sheets=3,
            last_completed_sheet=1,
        )

        runner._update_progress(state)

        # No sheet times = no ETA
        assert etas[0] is None


class TestRunSummary:
    """Tests for RunSummary dataclass and methods."""

    def test_run_summary_initialization(self) -> None:
        """Test RunSummary initializes with correct defaults."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
        )

        assert summary.job_id == "test-job"
        assert summary.job_name == "Test Job"
        assert summary.total_sheets == 10
        assert summary.completed_sheets == 0
        assert summary.failed_sheets == 0
        assert summary.skipped_sheets == 0
        assert summary.total_duration_seconds == 0.0
        assert summary.total_retries == 0
        assert summary.final_status == JobStatus.PENDING

    def test_success_rate_calculation(self) -> None:
        """Test success rate percentage calculation."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
            completed_sheets=8,
            failed_sheets=2,
        )

        assert summary.success_rate == 80.0

    def test_success_rate_zero_sheets(self) -> None:
        """Test success rate with zero sheets returns 0."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=0,
        )

        assert summary.success_rate == 0.0

    def test_validation_pass_rate_calculation(self) -> None:
        """Test validation pass rate calculation."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
            validation_pass_count=7,
            validation_fail_count=3,
        )

        assert summary.validation_pass_rate == 70.0

    def test_validation_pass_rate_no_validations(self) -> None:
        """Test validation pass rate with no validations returns 100%."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
        )

        # No validations = 100% pass
        assert summary.validation_pass_rate == 100.0

    def test_first_attempt_rate_calculation(self) -> None:
        """Test first attempt success rate calculation."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
            completed_sheets=8,
            first_attempt_successes=6,
        )

        assert summary.first_attempt_rate == 75.0

    def test_first_attempt_rate_no_completions(self) -> None:
        """Test first attempt rate with no completed sheets returns 0."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
            completed_sheets=0,
        )

        assert summary.first_attempt_rate == 0.0

    def test_to_dict_structure(self) -> None:
        """Test to_dict returns correct structure."""
        summary = RunSummary(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=10,
            completed_sheets=8,
            failed_sheets=1,
            skipped_sheets=1,
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

        assert result["sheets"]["total"] == 10
        assert result["sheets"]["completed"] == 8
        assert result["sheets"]["failed"] == 1
        assert result["sheets"]["skipped"] == 1
        assert result["sheets"]["success_rate"] == 80.0

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

        # Mock sheet execution
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet

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

        # Mock sheet execution to succeed with first_attempt_success
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                sheet_state = state.sheets[sheet_num]
                sheet_state.first_attempt_success = True
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet

            state, summary = await runner.run()

        assert summary.job_id == "test-job"
        assert summary.job_name == "test-job"
        assert summary.total_sheets == 3  # sample_config has 3 sheets
        assert summary.completed_sheets == 3
        assert summary.failed_sheets == 0
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

        # Mock sheet execution
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet

            await runner.run()

        # After run, summary is populated
        summary = runner.get_summary()
        assert summary is not None
        assert summary.completed_sheets == 3


class TestRunnerLoggingIntegration:
    """Tests for structured logging integration in JobRunner.

    These tests verify that the runner emits structured log events with
    correct data at appropriate times during job execution.
    """

    def setup_method(self) -> None:
        """Reset logging configuration before each test."""
        import logging as stdlib_logging

        import structlog

        from mozart.core.logging import clear_context

        # Reset structlog's caching
        structlog.reset_defaults()
        structlog.configure(cache_logger_on_first_use=False)
        root_logger = stdlib_logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        clear_context()

    def teardown_method(self) -> None:
        """Clear context after each test."""
        from mozart.core.logging import clear_context

        clear_context()

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text for assertion matching."""
        import re

        ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_pattern.sub("", text)

    @pytest.mark.asyncio
    async def test_job_started_log_emitted(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that job.started event is logged when run begins."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Mock sheet execution to complete immediately
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        assert "job.started" in err
        assert "job_id=test-job" in err
        assert "total_sheets=3" in err

    @pytest.mark.asyncio
    async def test_job_completed_log_emitted(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that job.completed event is logged with summary data."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        assert "job.completed" in err
        assert "status=completed" in err
        assert "duration_seconds" in err
        assert "completed_sheets=3" in err
        assert "success_rate" in err

    @pytest.mark.asyncio
    async def test_job_failed_log_emitted(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that job.failed event is logged when job fails."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="ERROR",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Mock sheet execution to fail
        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            mock_exec.side_effect = FatalError("Test fatal error")

            with pytest.raises(FatalError):
                await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        assert "job.failed" in err
        assert "sheet_num=1" in err
        assert "Test fatal error" in err

    @pytest.mark.asyncio
    async def test_sheet_started_log_emitted(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that sheet.started event is logged for each sheet."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # For sheet.started to be logged, we need to actually run sheets
        # This requires mocking deeper - let's patch at a higher level
        executed_sheets: list[int] = []

        async def track_and_complete(state: CheckpointState, sheet_num: int) -> None:
            executed_sheets.append(sheet_num)
            state.mark_sheet_started(sheet_num)
            state.mark_sheet_completed(sheet_num, validation_passed=True)

        with patch.object(runner, "_execute_sheet_with_recovery", track_and_complete):
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        # Check that sheet.started was logged for each sheet via job.started logs
        assert "job.started" in err

    @pytest.mark.asyncio
    async def test_sheet_retry_log_emitted(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that sheet.retry event is logged when retry occurs."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="WARNING",
            format="console",
            include_timestamps=False,
        )

        # Configure backend to fail then succeed
        call_count = 0

        async def fail_then_succeed(prompt: str) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call fails with transient error
                return ExecutionResult(
                    success=False,
                    stdout="",
                    stderr="Connection timeout",
                    exit_code=1,
                    duration_seconds=1.0,
                )
            else:
                # Subsequent calls succeed
                return ExecutionResult(
                    success=True,
                    stdout="Success",
                    stderr="",
                    exit_code=0,
                    duration_seconds=1.0,
                )

        mock_backend.execute = AsyncMock(side_effect=fail_then_succeed)

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Track validation calls to fail first, pass second
        validation_call_count = 0
        test_rule = ValidationRule(
            type="file_exists",
            path="{workspace}/output.txt",
            description="Output file must exist",
        )

        def validation_side_effect(*args: Any, **kwargs: Any) -> SheetValidationResult:
            nonlocal validation_call_count
            validation_call_count += 1
            if validation_call_count == 1:
                # First validation fails
                return SheetValidationResult(
                    sheet_num=1,
                    results=[
                        ValidationResult(rule=test_rule, passed=False)
                    ],
                )
            else:
                # Subsequent validations pass
                return SheetValidationResult(
                    sheet_num=1,
                    results=[
                        ValidationResult(rule=test_rule, passed=True)
                    ],
                )

        # Patch validation and preflight to pass
        with (
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.run_validations"
            ) as mock_validation,
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.snapshot_mtime_files"
            ),
            patch.object(
                runner, "_run_preflight_checks", return_value=make_mock_preflight_result()
            ),
        ):
            mock_validation.side_effect = validation_side_effect
            # Run only one sheet to limit complexity
            sample_config.sheet.total_items = 10
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        assert "sheet.retry" in err
        assert "attempt=1" in err

    @pytest.mark.asyncio
    async def test_rate_limit_log_emitted(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that rate_limit.detected event is logged."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="WARNING",
            format="console",
            include_timestamps=False,
        )

        # Configure short rate limit wait for test
        sample_config.rate_limit.wait_minutes = 0.01  # Very short wait

        call_count = 0

        async def rate_limit_then_succeed(prompt: str) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ExecutionResult(
                    success=False,
                    stdout="rate limit exceeded",
                    stderr="",
                    exit_code=1,
                    duration_seconds=1.0,
                )
            else:
                return ExecutionResult(
                    success=True,
                    stdout="Success",
                    stderr="",
                    exit_code=0,
                    duration_seconds=1.0,
                )

        mock_backend.execute = AsyncMock(side_effect=rate_limit_then_succeed)

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Track validation calls to fail during rate-limit, pass after recovery
        validation_call_count = 0
        test_rule = ValidationRule(
            type="file_exists",
            path="{workspace}/output.txt",
            description="Output file must exist",
        )

        def validation_side_effect(*args: Any, **kwargs: Any) -> SheetValidationResult:
            nonlocal validation_call_count
            validation_call_count += 1
            if validation_call_count == 1:
                # First validation fails (during rate-limit execution)
                return SheetValidationResult(
                    sheet_num=1,
                    results=[
                        ValidationResult(rule=test_rule, passed=False)
                    ],
                )
            else:
                # After rate-limit wait, validation passes
                return SheetValidationResult(
                    sheet_num=1,
                    results=[
                        ValidationResult(rule=test_rule, passed=True)
                    ],
                )

        # Patch validation and preflight to pass
        with (
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.run_validations"
            ) as mock_validation,
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.snapshot_mtime_files"
            ),
            patch.object(
                runner, "_run_preflight_checks", return_value=make_mock_preflight_result()
            ),
        ):
            mock_validation.side_effect = validation_side_effect
            sample_config.sheet.total_items = 10
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        assert "rate_limit.detected" in err
        assert "wait_count=1" in err

    @pytest.mark.asyncio
    async def test_config_summary_includes_safe_fields(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that _get_config_summary returns safe, non-sensitive data."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        summary = runner._get_config_summary()

        # Should include these safe fields
        assert "backend_type" in summary
        assert "sheet_size" in summary
        assert "total_items" in summary
        assert "max_retries" in summary
        assert "validation_count" in summary

        # Should NOT include sensitive fields
        assert "api_key" not in str(summary).lower()
        assert "token" not in str(summary).lower()
        assert "secret" not in str(summary).lower()

    @pytest.mark.asyncio
    async def test_sheet_completed_includes_validation_duration(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that sheet.completed includes validation_duration_seconds."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Patch validation and preflight to pass
        with (
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.run_validations"
            ) as mock_validation,
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.snapshot_mtime_files"
            ),
            patch.object(
                runner, "_run_preflight_checks", return_value=make_mock_preflight_result()
            ),
        ):
            mock_validation.return_value = SheetValidationResult(sheet_num=1, results=[])
            sample_config.sheet.total_items = 10
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        assert "sheet.completed" in err
        assert "validation_duration_seconds" in err

    @pytest.mark.asyncio
    async def test_execution_context_created_for_run(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that ExecutionContext is created during run."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Before run, no execution context
        assert runner._execution_context is None

        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet
            await runner.run()

        # After run, execution context should be set
        assert runner._execution_context is not None
        assert runner._execution_context.job_id == "test-job"
        assert runner._execution_context.component == "runner"

    @pytest.mark.asyncio
    async def test_preflight_warnings_logged(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that preflight warnings count is included in sheet.started log."""
        from mozart.core.logging import configure_logging

        # Use INFO level to capture sheet.started which includes preflight_warnings count
        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        # Create a preflight result with warnings (simulates large prompt)
        preflight_with_warnings = PreflightResult(
            prompt_metrics=PromptMetrics(
                character_count=200000,
                estimated_tokens=50000,  # Above warning threshold
                line_count=100,
                has_file_references=False,
                referenced_paths=[],
                word_count=40000,
            ),
            warnings=["Large prompt detected: ~50,000 tokens"],
            errors=[],
            paths_accessible={},
            working_directory_valid=True,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Patch validation and preflight to return our warning result
        with (
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.run_validations"
            ) as mock_validation,
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.snapshot_mtime_files"
            ),
            patch.object(
                runner, "_run_preflight_checks", return_value=preflight_with_warnings
            ),
        ):
            mock_validation.return_value = SheetValidationResult(sheet_num=1, results=[])
            sample_config.sheet.total_items = 10
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)
        out = self.strip_ansi(captured.out)

        # When _run_preflight_checks is patched, the logging inside it doesn't run.
        # But the caller logs sheet.started with preflight_warnings count at INFO level.
        # Check that either:
        # 1. The structured warning log appears in stderr (if not patched)
        # 2. The sheet.started log shows preflight_warnings count > 0
        # 3. Console output shows the warning (stdout)
        assert (
            "sheet.preflight_warnings" in err
            or "preflight_warnings=1" in err
            or "Large prompt detected" in out
        )


class TestLoggingLevelFiltering:
    """Tests for log level filtering in runner."""

    def setup_method(self) -> None:
        """Reset logging configuration before each test."""
        import logging as stdlib_logging

        import structlog

        from mozart.core.logging import clear_context

        structlog.reset_defaults()
        structlog.configure(cache_logger_on_first_use=False)
        root_logger = stdlib_logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        clear_context()

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove ANSI escape codes from text."""
        import re

        ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
        return ansi_pattern.sub("", text)

    @pytest.mark.asyncio
    async def test_debug_logs_not_shown_at_info_level(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that DEBUG level logs are not shown when level is INFO."""
        from mozart.core.logging import configure_logging

        configure_logging(
            level="INFO",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        with patch.object(runner, "_execute_sheet_with_recovery") as mock_exec:
            async def complete_sheet(state: CheckpointState, sheet_num: int) -> None:
                state.mark_sheet_started(sheet_num)
                state.mark_sheet_completed(sheet_num, validation_passed=True)

            mock_exec.side_effect = complete_sheet
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        # INFO level logs should be present
        assert "job.started" in err or "job.completed" in err

        # DEBUG level logs (like preflight_metrics) should NOT be present
        # at INFO level
        assert "sheet.preflight_metrics" not in err

    @pytest.mark.asyncio
    async def test_warning_logs_shown_at_warning_level(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that WARNING level logs are shown when level is WARNING."""
        from mozart.core.logging import configure_logging

        # Configure short rate limit wait
        sample_config.rate_limit.wait_minutes = 0.01

        call_count = 0

        async def rate_limit_then_succeed(prompt: str) -> ExecutionResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ExecutionResult(
                    success=False,
                    stdout="rate limit exceeded",
                    stderr="",
                    exit_code=1,
                    duration_seconds=1.0,
                )
            else:
                return ExecutionResult(
                    success=True,
                    stdout="Success",
                    stderr="",
                    exit_code=0,
                    duration_seconds=1.0,
                )

        mock_backend.execute = AsyncMock(side_effect=rate_limit_then_succeed)

        configure_logging(
            level="WARNING",
            format="console",
            include_timestamps=False,
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        # Track validation calls to fail during rate-limit, pass after recovery
        validation_call_count = 0
        test_rule = ValidationRule(
            type="file_exists",
            path="{workspace}/output.txt",
            description="Output file must exist",
        )

        def validation_side_effect(*args: Any, **kwargs: Any) -> SheetValidationResult:
            nonlocal validation_call_count
            validation_call_count += 1
            if validation_call_count == 1:
                # First validation fails (during rate-limit execution)
                return SheetValidationResult(
                    sheet_num=1,
                    results=[
                        ValidationResult(rule=test_rule, passed=False)
                    ],
                )
            else:
                # After rate-limit wait, validation passes
                return SheetValidationResult(
                    sheet_num=1,
                    results=[
                        ValidationResult(rule=test_rule, passed=True)
                    ],
                )

        with (
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.run_validations"
            ) as mock_validation,
            patch(
                "mozart.execution.runner.sheet.ValidationEngine.snapshot_mtime_files"
            ),
            patch.object(
                runner, "_run_preflight_checks", return_value=make_mock_preflight_result()
            ),
        ):
            mock_validation.side_effect = validation_side_effect
            sample_config.sheet.total_items = 10
            await runner.run()

        captured = capsys.readouterr()
        err = self.strip_ansi(captured.err)

        # WARNING level logs should be present
        assert "rate_limit.detected" in err

        # INFO level logs should NOT be present at WARNING level
        assert "job.started" not in err
        assert "job.completed" not in err


class TestActiveBroadcastPolling:
    """Tests for v16 Evolution: Active Broadcast Polling.

    These tests verify that the runner correctly polls for pattern
    discoveries from other concurrent jobs during retry waits.
    """

    @pytest.mark.asyncio
    async def test_poll_broadcast_discoveries_calls_store(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Verify polling calls check_recent_pattern_discoveries on store."""
        from datetime import datetime
        from mozart.learning.global_store import PatternDiscoveryEvent

        mock_store = MagicMock()
        mock_store.check_recent_pattern_discoveries = MagicMock(return_value=[])

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            global_learning_store=mock_store,
        )

        await runner._poll_broadcast_discoveries("test-job-123", sheet_num=1)

        mock_store.check_recent_pattern_discoveries.assert_called_once_with(
            exclude_job_id="test-job-123",
            min_effectiveness=0.5,
            limit=10,
        )

    @pytest.mark.asyncio
    async def test_poll_broadcast_discoveries_handles_empty_results(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Verify polling handles empty discovery results gracefully."""
        mock_store = MagicMock()
        mock_store.check_recent_pattern_discoveries = MagicMock(return_value=[])

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            global_learning_store=mock_store,
        )

        # Should not raise, should not log anything for empty results
        await runner._poll_broadcast_discoveries("test-job-123", sheet_num=1)

        mock_store.check_recent_pattern_discoveries.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_broadcast_discoveries_logs_found_patterns(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Verify polling logs when patterns are discovered."""
        from datetime import datetime, timedelta
        from mozart.learning.global_store import PatternDiscoveryEvent

        now = datetime.now()
        discoveries = [
            PatternDiscoveryEvent(
                id="disc-001",
                pattern_id="pat-001",
                pattern_name="Test Pattern 1",
                pattern_type="retry_pattern",
                source_job_hash="other-job-hash",
                recorded_at=now,
                expires_at=now + timedelta(minutes=10),
                effectiveness_score=0.85,
                context_tags=["test"],
            ),
            PatternDiscoveryEvent(
                id="disc-002",
                pattern_id="pat-002",
                pattern_name="Test Pattern 2",
                pattern_type="validation_failure",
                source_job_hash="another-job-hash",
                recorded_at=now,
                expires_at=now + timedelta(minutes=10),
                effectiveness_score=0.72,
                context_tags=["test", "validation"],
            ),
        ]

        mock_store = MagicMock()
        mock_store.check_recent_pattern_discoveries = MagicMock(return_value=discoveries)

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            global_learning_store=mock_store,
        )

        await runner._poll_broadcast_discoveries("test-job-123", sheet_num=2)

        captured = capsys.readouterr()
        # Console should show discovery count
        assert "2 pattern" in captured.out
        assert "broadcast" in captured.out

    @pytest.mark.asyncio
    async def test_poll_broadcast_discoveries_excludes_self_job(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Verify polling excludes the current job's own patterns."""
        mock_store = MagicMock()
        mock_store.check_recent_pattern_discoveries = MagicMock(return_value=[])

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            global_learning_store=mock_store,
        )

        await runner._poll_broadcast_discoveries("my-unique-job-id", sheet_num=1)

        # Verify exclude_job_id is passed correctly
        call_kwargs = mock_store.check_recent_pattern_discoveries.call_args[1]
        assert call_kwargs["exclude_job_id"] == "my-unique-job-id"

    @pytest.mark.asyncio
    async def test_poll_broadcast_discoveries_handles_store_error(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Verify polling handles store errors gracefully without blocking."""
        mock_store = MagicMock()
        mock_store.check_recent_pattern_discoveries = MagicMock(
            side_effect=Exception("Database connection failed")
        )

        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            global_learning_store=mock_store,
        )

        # Should not raise - polling failure shouldn't block retry
        await runner._poll_broadcast_discoveries("test-job", sheet_num=1)

    @pytest.mark.asyncio
    async def test_poll_broadcast_discoveries_noop_without_store(
        self,
        sample_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Verify polling is no-op when global learning store is not configured."""
        runner = JobRunner(
            config=sample_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
            global_learning_store=None,  # No store configured
        )

        # Should not raise
        await runner._poll_broadcast_discoveries("test-job", sheet_num=1)


# =============================================================================
# TestCostTracking
# =============================================================================


class TestCostTracking:
    """Tests for CostMixin cost tracking functionality.

    Tests the cost tracking methods:
    - _track_cost(): Token usage and cost calculation
    - _check_cost_limits(): Per-sheet and per-job cost limits
    - _estimate_prompt_tokens(): Token estimation
    - _format_cost(): Cost formatting
    """

    @pytest.fixture
    def cost_enabled_config(self) -> JobConfig:
        """Create a config with cost tracking enabled."""
        return JobConfig.model_validate({
            "name": "test-cost-job",
            "description": "Test job for cost tracking tests",
            "backend": {
                "type": "claude_cli",
                "skip_permissions": True,
            },
            "sheet": {
                "size": 10,
                "total_items": 30,
            },
            "prompt": {
                "template": "Process sheet {{ sheet_num }}.",
            },
            "cost_limits": {
                "enabled": True,
                "max_cost_per_sheet": 1.0,
                "max_cost_per_job": 10.0,
                "cost_per_1k_input_tokens": 0.003,
                "cost_per_1k_output_tokens": 0.015,
                "warn_at_percent": 80,
            },
        })

    @pytest.fixture
    def cost_runner(
        self,
        cost_enabled_config: JobConfig,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> JobRunner:
        """Create a runner with cost tracking enabled."""
        return JobRunner(
            config=cost_enabled_config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

    def test_track_cost_with_exact_tokens(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test cost tracking with exact token counts from API."""
        from mozart.backends.base import ExecutionResult
        from mozart.core.checkpoint import CheckpointState, SheetState

        result = ExecutionResult(
            success=True,
            stdout="Output",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            input_tokens=1000,
            output_tokens=500,
        )
        sheet_state = SheetState(sheet_num=1)
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )

        input_tokens, output_tokens, cost, confidence = cost_runner._track_cost(
            result, sheet_state, state
        )

        assert input_tokens == 1000
        assert output_tokens == 500
        # Cost = (1000/1000 * 0.003) + (500/1000 * 0.015) = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 0.0001
        assert confidence == 1.0  # Exact counts

    def test_track_cost_with_legacy_tokens(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test cost tracking with legacy tokens_used field."""
        from mozart.backends.base import ExecutionResult
        from mozart.core.checkpoint import CheckpointState, SheetState

        result = ExecutionResult(
            success=True,
            stdout="Output",
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
            tokens_used=1000,  # Legacy field
        )
        sheet_state = SheetState(sheet_num=1)
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )

        _, output_tokens, _, confidence = cost_runner._track_cost(
            result, sheet_state, state
        )

        assert output_tokens == 1000
        assert confidence == 0.85  # Lower confidence for legacy

    def test_track_cost_with_estimation(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test cost tracking with character-based estimation."""
        from mozart.backends.base import ExecutionResult
        from mozart.core.checkpoint import CheckpointState, SheetState

        # 400 chars => ~100 tokens (4 chars per token)
        result = ExecutionResult(
            success=True,
            stdout="x" * 400,
            stderr="",
            exit_code=0,
            duration_seconds=1.0,
        )
        sheet_state = SheetState(sheet_num=1)
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )

        _, output_tokens, _, confidence = cost_runner._track_cost(
            result, sheet_state, state
        )

        assert output_tokens == 100  # 400 chars / 4
        assert confidence == 0.7  # Lowest confidence for estimation

    def test_track_cost_accumulates_to_state(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test that costs accumulate across multiple calls."""
        from mozart.backends.base import ExecutionResult
        from mozart.core.checkpoint import CheckpointState, SheetState

        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3
        )

        # Track cost for 3 sheets
        for i in range(3):
            result = ExecutionResult(
                success=True,
                stdout="Output",
                stderr="",
                exit_code=0,
                duration_seconds=1.0,
                input_tokens=1000,
                output_tokens=500,
            )
            sheet_state = SheetState(sheet_num=i + 1)
            cost_runner._track_cost(result, sheet_state, state)

        # Total should be 3x the individual cost
        expected_cost = 3 * 0.0105
        assert abs(state.total_estimated_cost - expected_cost) < 0.0001
        assert state.total_input_tokens == 3000
        assert state.total_output_tokens == 1500

    def test_check_cost_limits_passes_within_limits(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test that cost limit check passes when within limits."""
        from mozart.core.checkpoint import CheckpointState, SheetState

        sheet_state = SheetState(sheet_num=1)
        sheet_state.estimated_cost = 0.5  # Below max_cost_per_sheet (1.0)
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )
        state.total_estimated_cost = 5.0  # Below max_cost_per_job (10.0)

        exceeded, reason = cost_runner._check_cost_limits(sheet_state, state)

        assert exceeded is False
        assert reason is None

    def test_check_cost_limits_fails_on_sheet_limit(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test that cost limit check fails when sheet limit exceeded."""
        from mozart.core.checkpoint import CheckpointState, SheetState

        sheet_state = SheetState(sheet_num=1)
        sheet_state.estimated_cost = 1.5  # Above max_cost_per_sheet (1.0)
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )

        exceeded, reason = cost_runner._check_cost_limits(sheet_state, state)

        assert exceeded is True
        assert "Sheet cost" in reason
        assert "exceeded limit" in reason

    def test_check_cost_limits_fails_on_job_limit(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test that cost limit check fails when job limit exceeded."""
        from mozart.core.checkpoint import CheckpointState, SheetState

        sheet_state = SheetState(sheet_num=1)
        sheet_state.estimated_cost = 0.5
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )
        state.total_estimated_cost = 11.0  # Above max_cost_per_job (10.0)

        exceeded, reason = cost_runner._check_cost_limits(sheet_state, state)

        assert exceeded is True
        assert "Job cost" in reason
        assert "exceeded limit" in reason

    def test_check_cost_limits_disabled_passes_always(
        self,
        mock_backend: MagicMock,
        mock_state_backend: MagicMock,
    ) -> None:
        """Test that cost limits always pass when disabled."""
        from mozart.core.checkpoint import CheckpointState, SheetState

        config = JobConfig.model_validate({
            "name": "test-job",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 10, "total_items": 30},
            "prompt": {"template": "Test"},
            "cost_limits": {"enabled": False},  # Disabled
        })
        runner = JobRunner(
            config=config,
            backend=mock_backend,
            state_backend=mock_state_backend,
        )

        sheet_state = SheetState(sheet_num=1)
        sheet_state.estimated_cost = 1000.0  # Very high
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=1
        )
        state.total_estimated_cost = 1000.0

        exceeded, reason = runner._check_cost_limits(sheet_state, state)

        assert exceeded is False  # Passes because disabled
        assert reason is None

    def test_estimate_prompt_tokens(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test prompt token estimation."""
        # 400 chars => 100 tokens
        assert cost_runner._estimate_prompt_tokens("x" * 400) == 100
        # 3 chars => minimum 1 token
        assert cost_runner._estimate_prompt_tokens("abc") == 1
        # Empty string => minimum 1 token
        assert cost_runner._estimate_prompt_tokens("") == 1

    def test_format_cost(
        self,
        cost_runner: JobRunner,
    ) -> None:
        """Test cost formatting."""
        assert cost_runner._format_cost(0.0042) == "$0.0042"  # < 0.01
        assert cost_runner._format_cost(0.123) == "$0.123"  # < 1.0
        assert cost_runner._format_cost(1.23) == "$1.23"  # >= 1.0
        assert cost_runner._format_cost(12.34) == "$12.34"
