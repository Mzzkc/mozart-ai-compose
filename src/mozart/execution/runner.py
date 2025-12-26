"""Job runner with partial completion recovery.

Orchestrates batch execution with validation, retry logic, and
automatic completion prompt generation for partial failures.
"""

import asyncio
import random
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from rich.console import Console

from mozart.backends.base import Backend, ExecutionResult
from mozart.core.checkpoint import BatchStatus, CheckpointState, JobStatus
from mozart.core.config import JobConfig
from mozart.core.errors import ClassifiedError, ErrorClassifier
from mozart.core.logging import ExecutionContext, MozartLogger, get_logger
from mozart.execution.circuit_breaker import CircuitBreaker
from mozart.execution.escalation import EscalationContext, EscalationHandler, EscalationResponse
from mozart.execution.preflight import PreflightChecker, PreflightResult
from mozart.execution.retry_strategy import (
    AdaptiveRetryStrategy,
    ErrorRecord,
    RetryStrategyConfig,
)
from mozart.execution.validation import BatchValidationResult, ValidationEngine
from mozart.learning.judgment import JudgmentClient, JudgmentQuery, JudgmentResponse
from mozart.learning.outcomes import BatchOutcome, OutcomeStore
from mozart.prompts.templating import BatchContext, CompletionContext, PromptBuilder
from mozart.state.base import StateBackend


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


@dataclass
class RunSummary:
    """Summary of a completed job run.

    Tracks key metrics for display at job completion:
    - Total execution time
    - Batch success/failure counts
    - Validation pass rate
    - Retry statistics
    """

    job_id: str
    job_name: str
    total_batches: int
    completed_batches: int = 0
    failed_batches: int = 0
    skipped_batches: int = 0
    total_duration_seconds: float = 0.0
    total_retries: int = 0
    total_completion_attempts: int = 0
    rate_limit_waits: int = 0
    validation_pass_count: int = 0
    validation_fail_count: int = 0
    first_attempt_successes: int = 0
    final_status: JobStatus = field(default=JobStatus.PENDING)

    @property
    def success_rate(self) -> float:
        """Calculate batch success rate as percentage."""
        if self.total_batches == 0:
            return 0.0
        return (self.completed_batches / self.total_batches) * 100

    @property
    def validation_pass_rate(self) -> float:
        """Calculate validation pass rate as percentage."""
        total = self.validation_pass_count + self.validation_fail_count
        if total == 0:
            return 100.0  # No validations = 100% pass
        return (self.validation_pass_count / total) * 100

    @property
    def first_attempt_rate(self) -> float:
        """Calculate first-attempt success rate as percentage."""
        if self.completed_batches == 0:
            return 0.0
        return (self.first_attempt_successes / self.completed_batches) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary for JSON output."""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.final_status.value,
            "duration_seconds": round(self.total_duration_seconds, 2),
            "duration_formatted": self._format_duration(self.total_duration_seconds),
            "batches": {
                "total": self.total_batches,
                "completed": self.completed_batches,
                "failed": self.failed_batches,
                "skipped": self.skipped_batches,
                "success_rate": round(self.success_rate, 1),
            },
            "validation": {
                "passed": self.validation_pass_count,
                "failed": self.validation_fail_count,
                "pass_rate": round(self.validation_pass_rate, 1),
            },
            "execution": {
                "total_retries": self.total_retries,
                "completion_attempts": self.total_completion_attempts,
                "rate_limit_waits": self.rate_limit_waits,
                "first_attempt_successes": self.first_attempt_successes,
                "first_attempt_rate": round(self.first_attempt_rate, 1),
            },
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class BatchExecutionMode(str, Enum):
    """Mode of batch execution."""

    NORMAL = "normal"
    """Standard first-time execution."""

    COMPLETION = "completion"
    """Completion mode after partial success - focused on missing items."""

    RETRY = "retry"
    """Full retry after completion mode exhausted or minority passed."""

    ESCALATE = "escalate"
    """Escalation mode - low confidence requires external decision."""


class FatalError(Exception):
    """Non-recoverable error that should stop the job."""

    pass


class GracefulShutdownError(Exception):
    """Raised when Ctrl+C is pressed to trigger graceful shutdown.

    This exception is caught by the runner to save state before exiting.
    """

    pass


class JobRunner:
    """Orchestrates batch execution with validation and partial recovery.

    The runner implements the following flow:
    1. Execute batch with standard prompt
    2. Run validations on expected outputs
    3. If all pass: mark complete, continue to next batch
    4. If majority pass: enter completion mode, generate focused prompt
    5. If minority pass: full retry with original prompt
    6. Track attempts separately for completion vs full retry
    """

    def __init__(
        self,
        config: JobConfig,
        backend: Backend,
        state_backend: StateBackend,
        console: Console | None = None,
        outcome_store: OutcomeStore | None = None,
        escalation_handler: EscalationHandler | None = None,
        judgment_client: JudgmentClient | None = None,
        progress_callback: Callable[[int, int, float | None], None] | None = None,
        execution_progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize job runner.

        Args:
            config: Job configuration.
            backend: Claude execution backend.
            state_backend: State persistence backend.
            console: Rich console for output (optional).
            outcome_store: Optional outcome store for learning (Phase 1).
            escalation_handler: Optional escalation handler for low-confidence
                decisions (Phase 2). If not provided, escalation is disabled.
            judgment_client: Optional judgment client for Recursive Light
                integration (Phase 4). Provides TDF-aligned execution decisions.
                If not provided, falls back to local _decide_next_action().
            progress_callback: Optional callback for progress updates. Called with
                (completed_batches, total_batches, eta_seconds). Used by CLI for
                progress bar updates.
            execution_progress_callback: Optional callback for real-time execution
                progress during long-running batch executions. Called with dict
                containing: batch_num, bytes_received, lines_received, elapsed_seconds,
                phase. Used by CLI to show "Still running... 5.2KB received".
        """
        self.config = config
        self.backend = backend
        self.state_backend = state_backend
        self.console = console or Console()
        self.outcome_store = outcome_store
        self.escalation_handler = escalation_handler
        self.judgment_client = judgment_client
        self.progress_callback = progress_callback
        self.execution_progress_callback = execution_progress_callback
        self.prompt_builder = PromptBuilder(config.prompt)
        self.error_classifier = ErrorClassifier.from_config(
            config.rate_limit.detection_patterns
        )
        self.preflight_checker = PreflightChecker(
            workspace=config.workspace,
            working_directory=config.backend.working_directory or config.workspace,
        )

        # Graceful shutdown state
        self._shutdown_requested = False
        self._current_state: CheckpointState | None = None
        self._batch_times: list[float] = []  # Track batch durations for ETA

        # Summary tracking for run statistics
        self._summary: RunSummary | None = None
        self._run_start_time: float = 0.0

        # Execution progress tracking (Task 4)
        self._current_batch_num: int | None = None
        self._execution_progress_snapshots: list[dict[str, Any]] = []

        # Structured logging (Task 8: Logging Integration)
        self._logger: MozartLogger = get_logger("runner")
        self._execution_context: ExecutionContext | None = None

        # Circuit breaker for resilient execution (Task 12)
        self._circuit_breaker: CircuitBreaker | None = None
        if config.circuit_breaker.enabled:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker.failure_threshold,
                recovery_timeout=config.circuit_breaker.recovery_timeout_seconds,
                name=config.name,
            )

        # Adaptive retry strategy (Task 13)
        self._retry_strategy = AdaptiveRetryStrategy(
            config=RetryStrategyConfig(
                base_delay=config.retry.base_delay_seconds,
                max_delay=config.retry.max_delay_seconds,
                exponential_base=config.retry.exponential_base,
                jitter_factor=0.25,  # 25% jitter
            )
        )

    async def run(
        self,
        start_batch: int | None = None,
        config_path: str | None = None,
    ) -> tuple[CheckpointState, RunSummary]:
        """Run the job from start or resume point.

        Args:
            start_batch: Optional batch number to start from (overrides state).
            config_path: Optional path to the original config file (for resume).

        Returns:
            Tuple of (CheckpointState, RunSummary) with final job state and
            execution statistics.

        Raises:
            FatalError: If an unrecoverable error occurs.
            GracefulShutdownError: If Ctrl+C was pressed and job was paused.
        """
        # Initialize timing and summary tracking
        self._run_start_time = time.monotonic()
        state = await self._initialize_state(start_batch, config_path)
        self._current_state = state

        # Initialize run summary
        self._summary = RunSummary(
            job_id=state.job_id,
            job_name=state.job_name,
            total_batches=state.total_batches,
        )

        # Create execution context for correlation (Task 8: Logging Integration)
        self._execution_context = ExecutionContext(
            job_id=state.job_id,
            component="runner",
        )

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

        # Log job start with config summary
        config_summary = self._get_config_summary()
        self._logger.info(
            "job.started",
            job_id=state.job_id,
            total_batches=state.total_batches,
            resume_from=start_batch,
            config=config_summary,
        )

        try:
            next_batch = state.get_next_batch()
            while next_batch is not None and next_batch <= state.total_batches:
                # Check for shutdown request before starting batch
                if self._shutdown_requested:
                    await self._handle_graceful_shutdown(state)

                batch_start = time.monotonic()

                try:
                    await self._execute_batch_with_recovery(state, next_batch)
                except FatalError as e:
                    state.mark_job_failed(str(e))
                    await self.state_backend.save(state)
                    # Update summary with failure before raising
                    self._finalize_summary(state)
                    # Log job failure
                    self._logger.error(
                        "job.failed",
                        job_id=state.job_id,
                        batch_num=next_batch,
                        error=str(e),
                        duration_seconds=round(time.monotonic() - self._run_start_time, 2),
                        completed_batches=self._summary.completed_batches if self._summary else 0,
                    )
                    raise

                # Track batch timing for ETA calculation
                batch_duration = time.monotonic() - batch_start
                self._batch_times.append(batch_duration)

                # Update progress callback
                self._update_progress(state)

                # Pause between batches
                if next_batch < state.total_batches:
                    await self._interruptible_sleep(
                        self.config.pause_between_batches_seconds
                    )

                next_batch = state.get_next_batch()

            # Mark job complete if we processed all batches
            if state.status == JobStatus.RUNNING:
                state.status = JobStatus.COMPLETED
                await self.state_backend.save(state)

            # Finalize summary
            self._finalize_summary(state)

            # Log job completion with summary
            self._logger.info(
                "job.completed",
                job_id=state.job_id,
                status=state.status.value,
                duration_seconds=round(self._summary.total_duration_seconds, 2),
                completed_batches=self._summary.completed_batches,
                failed_batches=self._summary.failed_batches,
                success_rate=round(self._summary.success_rate, 1),
                validation_pass_rate=round(self._summary.validation_pass_rate, 1),
                total_retries=self._summary.total_retries,
            )

            return state, self._summary
        finally:
            # Remove signal handlers
            self._remove_signal_handlers()

    def _finalize_summary(self, state: CheckpointState) -> None:
        """Finalize run summary with statistics from completed job.

        Args:
            state: Final job state to extract statistics from.
        """
        if self._summary is None:
            return

        # Calculate total duration
        self._summary.total_duration_seconds = time.monotonic() - self._run_start_time
        self._summary.final_status = state.status

        # Aggregate batch statistics
        for batch_state in state.batches.values():
            if batch_state.status == BatchStatus.COMPLETED:
                self._summary.completed_batches += 1
                if batch_state.first_attempt_success:
                    self._summary.first_attempt_successes += 1
                # Track completion attempts
                if batch_state.completion_attempts:
                    self._summary.total_completion_attempts += batch_state.completion_attempts
            elif batch_state.status == BatchStatus.FAILED:
                self._summary.failed_batches += 1
            elif batch_state.status == BatchStatus.SKIPPED:
                self._summary.skipped_batches += 1

            # Track retries (attempts - 1)
            if batch_state.attempt_count > 1:
                self._summary.total_retries += batch_state.attempt_count - 1

            # Track validation results
            if batch_state.validation_passed is True:
                self._summary.validation_pass_count += 1
            elif batch_state.validation_passed is False:
                self._summary.validation_fail_count += 1

        # Copy rate limit waits from state
        self._summary.rate_limit_waits = state.rate_limit_waits

    def get_summary(self) -> RunSummary | None:
        """Get the current run summary.

        Returns:
            RunSummary if a run has been executed, None otherwise.
        """
        return self._summary

    def _get_config_summary(self) -> dict[str, Any]:
        """Build a safe config summary for logging.

        Returns a dictionary with key configuration values that are safe
        to log (no sensitive data like API keys or tokens).

        Returns:
            Dictionary with non-sensitive configuration summary.
        """
        return {
            "backend_type": self.config.backend.type,
            "batch_size": self.config.batch.size,
            "total_items": self.config.batch.total_items,
            "max_retries": self.config.retry.max_retries,
            "max_completion_attempts": self.config.retry.max_completion_attempts,
            "workspace": str(self.config.workspace),
            "validation_count": len(self.config.validations),
            "rate_limit_wait_minutes": self.config.rate_limit.wait_minutes,
            "learning_enabled": (
                self.config.learning.enabled if hasattr(self.config, "learning") else False
            ),
            "circuit_breaker_enabled": self.config.circuit_breaker.enabled,
            "circuit_breaker_threshold": self.config.circuit_breaker.failure_threshold,
        }

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown.

        On Unix, uses SIGINT handler. On Windows, we rely on KeyboardInterrupt.
        """
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(
                    signal.SIGINT,
                    self._signal_handler,
                )
            except (RuntimeError, NotImplementedError):
                # Running in a thread or platform doesn't support signals
                pass

    def _remove_signal_handlers(self) -> None:
        """Remove signal handlers."""
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                loop.remove_signal_handler(signal.SIGINT)
            except (RuntimeError, NotImplementedError, ValueError):
                # Not running or no handler installed
                pass

    def _signal_handler(self) -> None:
        """Handle SIGINT by setting shutdown flag."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self.console.print(
                "\n[yellow]Ctrl+C received. Finishing current batch and saving state...[/yellow]"
            )

    async def _handle_graceful_shutdown(self, state: CheckpointState) -> None:
        """Handle graceful shutdown by saving state and raising GracefulShutdown.

        Args:
            state: Current job state to save.

        Raises:
            GracefulShutdownError: Always raised after saving state.
        """
        state.status = JobStatus.PAUSED
        await self.state_backend.save(state)

        # Show resume hint
        self.console.print(
            f"\n[green]State saved.[/green] Job paused at batch "
            f"{state.last_completed_batch + 1}/{state.total_batches}."
        )
        self.console.print(
            f"\n[bold]To resume:[/bold] mozart resume {state.job_id}"
        )

        raise GracefulShutdownError(f"Job {state.job_id} paused by user request")

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that can be interrupted by shutdown request.

        Args:
            seconds: Time to sleep in seconds.
        """
        # Sleep in small increments to check for shutdown
        increment = min(0.5, seconds)
        elapsed = 0.0

        while elapsed < seconds and not self._shutdown_requested:
            await asyncio.sleep(increment)
            elapsed += increment

    def _update_progress(self, state: CheckpointState) -> None:
        """Update progress callback with current state.

        Args:
            state: Current job state.
        """
        if self.progress_callback is None:
            return

        # Calculate ETA based on average batch time
        eta: float | None = None
        if self._batch_times:
            avg_time = sum(self._batch_times) / len(self._batch_times)
            remaining_batches = state.total_batches - state.last_completed_batch
            eta = avg_time * remaining_batches

        self.progress_callback(
            state.last_completed_batch,
            state.total_batches,
            eta,
        )

    def _handle_execution_progress(self, progress: dict[str, Any]) -> None:
        """Handle execution progress update from backend.

        Called by backend during long-running executions to report
        bytes/lines received and elapsed time.

        Args:
            progress: Dict with bytes_received, lines_received, elapsed_seconds, phase.
        """
        # Add batch_num to progress info
        progress_with_batch = {
            "batch_num": self._current_batch_num,
            **progress,
        }

        # Record snapshot for persistence (every 30 seconds of significant change)
        should_snapshot = (
            not self._execution_progress_snapshots
            or progress.get("bytes_received", 0)
            - self._execution_progress_snapshots[-1].get("bytes_received", 0)
            > 1024  # At least 1KB of new data
        )
        if should_snapshot:
            snapshot = {
                **progress_with_batch,
                "snapshot_at": _utc_now().isoformat(),
            }
            self._execution_progress_snapshots.append(snapshot)

        # Forward to CLI callback if set
        if self.execution_progress_callback is not None:
            self.execution_progress_callback(progress_with_batch)

    async def _initialize_state(
        self, start_batch: int | None,
        config_path: str | None = None,
    ) -> CheckpointState:
        """Initialize or load job state.

        Args:
            start_batch: Optional override for starting batch.
            config_path: Optional path to the original config file.

        Returns:
            CheckpointState ready for execution.
        """
        job_id = self.config.name
        state = await self.state_backend.load(job_id)

        if state is None:
            # Serialize JobConfig for resume capability (Task 3: Config Storage)
            config_snapshot = self.config.model_dump(mode="json")

            state = CheckpointState(
                job_id=job_id,
                job_name=self.config.name,
                total_batches=self.config.batch.total_batches,
                config_snapshot=config_snapshot,
                config_path=config_path,
            )
            await self.state_backend.save(state)
            self.console.print("[green]Created new job state[/green]")
        else:
            self.console.print(
                f"[yellow]Resuming from batch {state.last_completed_batch + 1}[/yellow]"
            )

        # Override starting batch if specified
        if start_batch is not None:
            state.last_completed_batch = start_batch - 1

        return state

    async def _execute_batch_with_recovery(
        self,
        state: CheckpointState,
        batch_num: int,
    ) -> None:
        """Execute a single batch with full retry/completion logic.

        Flow:
        1. Execute batch normally
        2. Run validations
        3. If all pass -> complete
        4. If majority pass -> enter completion mode
        5. If minority pass -> full retry

        Args:
            state: Current job state.
            batch_num: Batch number to execute.

        Raises:
            FatalError: If all retries exhausted or fatal error encountered.
        """
        # Track execution timing for learning
        execution_start_time = time.monotonic()

        # Build batch context
        batch_context = self._build_batch_context(batch_num)
        validation_engine = ValidationEngine(
            self.config.workspace,
            batch_context.to_dict(),
        )

        # Track attempts
        normal_attempts = 0
        completion_attempts = 0
        max_retries = self.config.retry.max_retries
        max_completion = self.config.retry.max_completion_attempts

        # Track execution history for judgment (Phase 4)
        execution_history: list[ExecutionResult] = []

        # Track error history for adaptive retry (Task 13)
        error_history: list[ErrorRecord] = []

        # Build original prompt
        original_prompt = self.prompt_builder.build_batch_prompt(batch_context)
        current_prompt = original_prompt
        current_mode = BatchExecutionMode.NORMAL

        # Run preflight checks before first execution (Task 2: Prompt Metrics)
        preflight_result = self._run_preflight_checks(
            prompt=original_prompt,
            batch_context=batch_context.to_dict(),
            batch_num=batch_num,
            state=state,
        )

        # Check for fatal preflight errors
        if preflight_result.has_errors:
            error_summary = "; ".join(preflight_result.errors)
            state.mark_batch_failed(
                batch_num,
                f"Preflight check failed: {error_summary}",
                "preflight",
            )
            await self.state_backend.save(state)
            # Log preflight failure
            self._logger.error(
                "batch.preflight_failed",
                batch_num=batch_num,
                errors=preflight_result.errors,
            )
            raise FatalError(f"Batch {batch_num}: Preflight check failed - {error_summary}")

        # Log batch start (Task 8: Logging Integration)
        self._logger.info(
            "batch.started",
            batch_num=batch_num,
            execution_mode=current_mode.value,
            prompt_tokens=preflight_result.prompt_metrics.estimated_tokens,
            prompt_lines=preflight_result.prompt_metrics.line_count,
            preflight_warnings=len(preflight_result.warnings),
        )

        while True:
            # Mark batch started
            state.mark_batch_started(batch_num)
            batch_state = state.batches[batch_num]
            batch_state.execution_mode = current_mode.value
            await self.state_backend.save(state)

            # Initialize execution progress tracking (Task 4)
            self._current_batch_num = batch_num
            self._execution_progress_snapshots.clear()

            # Snapshot mtimes before execution (for file_modified checks)
            validation_engine.snapshot_mtime_files(self.config.validations)

            # Check circuit breaker before execution (Task 12)
            if (
                self._circuit_breaker is not None
                and not self._circuit_breaker.can_execute()
            ):
                wait_time = self._circuit_breaker.time_until_retry()
                cb_state = self._circuit_breaker.get_state()
                self._logger.warning(
                    "circuit_breaker.blocked",
                    batch_num=batch_num,
                    state=cb_state.value,
                    wait_seconds=wait_time,
                )
                self.console.print(
                    f"[yellow]Batch {batch_num}: Circuit breaker OPEN - "
                    f"waiting {wait_time:.0f}s for recovery[/yellow]"
                )
                if wait_time and wait_time > 0:
                    await self._interruptible_sleep(wait_time)
                continue  # Re-check circuit breaker after wait

            # Execute
            self.console.print(
                f"[blue]Batch {batch_num}: {current_mode.value} execution[/blue]"
            )
            result = await self.backend.execute(current_prompt)

            # Store execution progress snapshots in batch state (Task 4)
            if self._execution_progress_snapshots:
                batch_state.progress_snapshots = self._execution_progress_snapshots.copy()
                batch_state.last_activity_at = _utc_now()

            # Capture raw output for debugging (Task 1: Raw Output Capture)
            batch_state.capture_output(result.stdout, result.stderr)

            # Track execution result for judgment (Phase 4)
            execution_history.append(result)

            if not result.success:
                # Execution failed (not validation failure)
                error = self._classify_error(result)

                # Track error for adaptive retry (Task 13)
                error_record = ErrorRecord.from_classified_error(
                    error=error,
                    batch_num=batch_num,
                    attempt_num=normal_attempts + 1,
                )
                error_history.append(error_record)

                # Record failure in circuit breaker (Task 12)
                # Don't record rate limits as they're handled specially
                if self._circuit_breaker is not None and not error.is_rate_limit:
                    self._circuit_breaker.record_failure()

                if error.is_rate_limit:
                    # Rate limit handling uses its own strategy
                    error_history.clear()  # Reset on rate limit
                    await self._handle_rate_limit(state)
                    continue  # Retry same execution

                if not error.should_retry:
                    state.mark_batch_failed(
                        batch_num,
                        error.message,
                        error.category.value,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        execution_duration_seconds=result.duration_seconds,
                    )
                    await self.state_backend.save(state)
                    # Log fatal batch failure
                    self._logger.error(
                        "batch.failed",
                        batch_num=batch_num,
                        error_category=error.category.value,
                        error_message=error.message,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        duration_seconds=round(result.duration_seconds or 0, 2),
                        stdout_tail=result.stdout[-500:] if result.stdout else None,
                    )
                    raise FatalError(f"Batch {batch_num}: {error.message}")

                # Transient error - get adaptive retry recommendation (Task 13)
                retry_recommendation = self._retry_strategy.analyze(
                    error_history=error_history,
                    max_retries=max_retries,
                )

                normal_attempts += 1

                # Check both max retries and adaptive strategy recommendation
                if normal_attempts >= max_retries:
                    state.mark_batch_failed(
                        batch_num,
                        f"Failed after {max_retries} retries: {error.message}",
                        error.category.value,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        execution_duration_seconds=result.duration_seconds,
                    )
                    await self.state_backend.save(state)
                    # Log retry exhaustion failure
                    self._logger.error(
                        "batch.failed",
                        batch_num=batch_num,
                        error_category=error.category.value,
                        error_message=f"Retries exhausted: {error.message}",
                        attempt=normal_attempts,
                        max_retries=max_retries,
                        exit_code=result.exit_code,
                        duration_seconds=round(result.duration_seconds or 0, 2),
                    )
                    raise FatalError(
                        f"Batch {batch_num} failed after {max_retries} retries"
                    )

                # Check if adaptive strategy recommends stopping early
                if not retry_recommendation.should_retry:
                    state.mark_batch_failed(
                        batch_num,
                        f"Adaptive retry aborted: {retry_recommendation.reason}",
                        error.category.value,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        execution_duration_seconds=result.duration_seconds,
                    )
                    await self.state_backend.save(state)
                    # Log adaptive strategy abort
                    self._logger.warning(
                        "batch.adaptive_retry_aborted",
                        batch_num=batch_num,
                        error_category=error.category.value,
                        pattern=retry_recommendation.detected_pattern.value,
                        reason=retry_recommendation.reason,
                        confidence=round(retry_recommendation.confidence, 3),
                        strategy=retry_recommendation.strategy_used,
                        attempt=normal_attempts,
                    )
                    raise FatalError(
                        f"Batch {batch_num} aborted: {retry_recommendation.reason}"
                    )

                # Log retry attempt with adaptive strategy info
                self._logger.warning(
                    "batch.retry",
                    batch_num=batch_num,
                    attempt=normal_attempts,
                    max_retries=max_retries,
                    error_category=error.category.value,
                    reason=error.message,
                    # Adaptive retry strategy fields (Task 13)
                    retry_delay_seconds=round(retry_recommendation.delay_seconds, 2),
                    retry_confidence=round(retry_recommendation.confidence, 3),
                    retry_pattern=retry_recommendation.detected_pattern.value,
                    retry_strategy=retry_recommendation.strategy_used,
                )
                self.console.print(
                    f"[yellow]Batch {batch_num}: {retry_recommendation.detected_pattern.value} - "
                    f"retry {normal_attempts}/{max_retries} "
                    f"(delay: {retry_recommendation.delay_seconds:.1f}s, "
                    f"confidence: {retry_recommendation.confidence:.0%})[/yellow]"
                )
                await asyncio.sleep(retry_recommendation.delay_seconds)
                continue

            # Execution succeeded - record success in circuit breaker (Task 12)
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_success()

            # Execution succeeded - run validations
            validation_start = time.monotonic()
            validation_result = validation_engine.run_validations(
                self.config.validations
            )
            validation_duration = time.monotonic() - validation_start

            # Update state with validation details
            self._update_batch_validation_state(state, batch_num, validation_result)
            await self.state_backend.save(state)

            if validation_result.all_passed:
                # SUCCESS - all validations passed
                execution_duration = time.monotonic() - execution_start_time

                # Determine outcome category based on execution path
                first_attempt_success = (normal_attempts == 1 and completion_attempts == 0)
                if first_attempt_success:
                    outcome_category = "success_first_try"
                elif completion_attempts > 0:
                    outcome_category = "success_completion"
                else:
                    outcome_category = "success_retry"

                # Populate BatchState learning fields
                batch_state = state.batches[batch_num]
                batch_state.first_attempt_success = first_attempt_success
                batch_state.outcome_category = outcome_category
                batch_state.confidence_score = validation_result.pass_percentage / 100.0

                state.mark_batch_completed(
                    batch_num,
                    validation_passed=True,
                    validation_details=validation_result.to_dict_list(),
                )

                # Record outcome for learning if store is available
                await self._record_batch_outcome(
                    batch_num=batch_num,
                    job_id=state.job_id,
                    validation_result=validation_result,
                    execution_duration=execution_duration,
                    normal_attempts=normal_attempts,
                    completion_attempts=completion_attempts,
                    first_attempt_success=first_attempt_success,
                    final_status=BatchStatus.COMPLETED,
                )

                await self.state_backend.save(state)
                # Log successful batch completion
                self._logger.info(
                    "batch.completed",
                    batch_num=batch_num,
                    duration_seconds=round(execution_duration, 2),
                    validation_duration_seconds=round(validation_duration, 3),
                    validation_count=len(validation_result.results),
                    validation_pass_rate=100.0,
                    outcome_category=outcome_category,
                    retry_count=normal_attempts - 1 if normal_attempts > 0 else 0,
                    completion_attempts=completion_attempts,
                    first_attempt_success=first_attempt_success,
                )
                self.console.print(
                    f"[green]Batch {batch_num}: All {len(validation_result.results)} "
                    f"validations passed[/green]"
                )
                return

            # Some validations failed - use judgment-based decision logic (Phase 4)
            next_mode, decision_reason, prompt_modifications = await self._decide_with_judgment(
                batch_num=batch_num,
                validation_result=validation_result,
                execution_history=execution_history,
                normal_attempts=normal_attempts,
                completion_attempts=completion_attempts,
            )
            pass_pct = validation_result.pass_percentage

            self.console.print(
                f"[dim]Batch {batch_num}: Decision: {next_mode.value} - {decision_reason}[/dim]"
            )

            # Apply prompt modifications from judgment if provided
            if prompt_modifications and next_mode == BatchExecutionMode.RETRY:
                # Join modifications into additional instructions
                modification_text = "\n".join(prompt_modifications)
                current_prompt = (
                    original_prompt + "\n\n---\nJudgment modifications:\n" + modification_text
                )
                self.console.print(
                    f"[blue]Batch {batch_num}: Applying {len(prompt_modifications)} "
                    f"prompt modifications from judgment[/blue]"
                )

            if next_mode == BatchExecutionMode.COMPLETION:
                # COMPLETION MODE: High/medium confidence with majority passed
                completion_attempts += 1
                batch_state.completion_attempts = completion_attempts
                current_mode = BatchExecutionMode.COMPLETION

                # Pass ValidationResult objects (not just rules) to get expanded paths
                completion_ctx = CompletionContext(
                    batch_num=batch_num,
                    total_batches=state.total_batches,
                    passed_validations=validation_result.get_passed_results(),
                    failed_validations=validation_result.get_failed_results(),
                    completion_attempt=completion_attempts,
                    max_completion_attempts=max_completion,
                    original_prompt=original_prompt,
                    workspace=self.config.workspace,
                )
                current_prompt = self.prompt_builder.build_completion_prompt(
                    completion_ctx
                )

                self.console.print(
                    f"[yellow]Batch {batch_num}: Entering completion mode "
                    f"({validation_result.passed_count}/{len(validation_result.results)} passed, "
                    f"{pass_pct:.0f}%). Attempt {completion_attempts}/{max_completion}[/yellow]"
                )

                await asyncio.sleep(self.config.retry.completion_delay_seconds)
                continue

            elif next_mode == BatchExecutionMode.ESCALATE:
                # ESCALATE MODE: Low confidence requires external decision
                # Track error messages for escalation context
                escalation_error_history: list[str] = []
                if batch_state.error_message:
                    escalation_error_history.append(batch_state.error_message)

                response = await self._handle_escalation(
                    state=state,
                    batch_num=batch_num,
                    validation_result=validation_result,
                    current_prompt=current_prompt,
                    error_history=escalation_error_history,
                    normal_attempts=normal_attempts,
                )

                # Apply escalation response
                if response.action == "retry":
                    normal_attempts += 1
                    if normal_attempts >= max_retries:
                        state.mark_batch_failed(
                            batch_num,
                            f"Escalation retry exhausted after {max_retries} attempts",
                            "escalation",
                        )
                        await self.state_backend.save(state)
                        raise FatalError(
                            f"Batch {batch_num} exhausted retries after escalation"
                        )
                    current_mode = BatchExecutionMode.RETRY
                    current_prompt = original_prompt
                    await asyncio.sleep(self._get_retry_delay(normal_attempts))
                    continue

                elif response.action == "skip":
                    # Skip this batch and move to next
                    state.mark_batch_completed(
                        batch_num,
                        validation_passed=False,
                        validation_details=validation_result.to_dict_list(),
                    )
                    batch_state = state.batches[batch_num]
                    batch_state.outcome_category = "skipped_by_escalation"
                    await self.state_backend.save(state)
                    self.console.print(
                        f"[yellow]Batch {batch_num}: Skipped via escalation[/yellow]"
                    )
                    return

                elif response.action == "abort":
                    state.mark_batch_failed(
                        batch_num,
                        "Aborted via escalation",
                        "escalation",
                    )
                    await self.state_backend.save(state)
                    raise FatalError(
                        f"Batch {batch_num}: Job aborted via escalation"
                    )

                elif response.action == "modify_prompt":
                    if response.modified_prompt is None:
                        # Fall back to retry if no modified prompt provided
                        self.console.print(
                            f"[yellow]Batch {batch_num}: No modified prompt provided, "
                            f"falling back to retry[/yellow]"
                        )
                        normal_attempts += 1
                        current_mode = BatchExecutionMode.RETRY
                        current_prompt = original_prompt
                    else:
                        current_mode = BatchExecutionMode.RETRY
                        current_prompt = response.modified_prompt
                        self.console.print(
                            f"[blue]Batch {batch_num}: Retrying with modified prompt[/blue]"
                        )
                    await asyncio.sleep(self._get_retry_delay(normal_attempts))
                    continue

            else:
                # RETRY MODE: Fall through to full retry
                normal_attempts += 1
                if normal_attempts >= max_retries:
                    state.mark_batch_failed(
                        batch_num,
                        f"Validation failed after {max_retries} retries and "
                        f"{completion_attempts} completion attempts "
                        f"({validation_result.failed_count} validations still failing)",
                        "validation",
                    )
                    await self.state_backend.save(state)
                    raise FatalError(
                        f"Batch {batch_num} exhausted all retry options "
                        f"({validation_result.failed_count} validations failing)"
                    )

                self.console.print(
                    f"[red]Batch {batch_num}: {validation_result.failed_count} validations failed "
                    f"({pass_pct:.0f}% passed). Full retry {normal_attempts}/{max_retries}[/red]"
                )

                current_mode = BatchExecutionMode.RETRY
                current_prompt = original_prompt
                await asyncio.sleep(self._get_retry_delay(normal_attempts))

    def _build_batch_context(self, batch_num: int) -> BatchContext:
        """Build batch context for template expansion.

        Args:
            batch_num: Current batch number.

        Returns:
            BatchContext with item range and workspace.
        """
        return self.prompt_builder.build_batch_context(
            batch_num=batch_num,
            total_batches=self.config.batch.total_batches,
            batch_size=self.config.batch.size,
            total_items=self.config.batch.total_items,
            start_item=self.config.batch.start_item,
            workspace=self.config.workspace,
        )

    def _run_preflight_checks(
        self,
        prompt: str,
        batch_context: dict[str, Any],
        batch_num: int,
        state: CheckpointState,
    ) -> PreflightResult:
        """Run preflight checks on a prompt before execution.

        Analyzes the prompt for potential issues like excessive size,
        missing file references, and invalid working directories.

        Updates BatchState with metrics and any warnings.

        Args:
            prompt: The prompt text to analyze.
            batch_context: Context dictionary for template expansion.
            batch_num: Current batch number.
            state: Current job state to update.

        Returns:
            PreflightResult with metrics and any warnings/errors.
        """
        # Ensure batch state exists
        if batch_num not in state.batches:
            from mozart.core.checkpoint import BatchState
            state.batches[batch_num] = BatchState(batch_num=batch_num)

        batch_state = state.batches[batch_num]

        # Run preflight checks
        result = self.preflight_checker.check(prompt, batch_context)

        # Store metrics in batch state
        batch_state.prompt_metrics = result.prompt_metrics.to_dict()
        batch_state.preflight_warnings = result.warnings.copy()

        metrics = result.prompt_metrics

        # Structured logging for preflight results (Task 8)
        if result.has_warnings:
            self._logger.warning(
                "batch.preflight_warnings",
                batch_num=batch_num,
                warnings=result.warnings,
                estimated_tokens=metrics.estimated_tokens,
            )

        # Debug-level log for metrics
        self._logger.debug(
            "batch.preflight_metrics",
            batch_num=batch_num,
            character_count=metrics.character_count,
            estimated_tokens=metrics.estimated_tokens,
            line_count=metrics.line_count,
            word_count=metrics.word_count,
            has_file_references=metrics.has_file_references,
            referenced_path_count=len(metrics.referenced_paths),
        )

        # Console output (preserved for CLI feedback)
        for warning in result.warnings:
            self.console.print(f"[yellow]Batch {batch_num} preflight: {warning}[/yellow]")

        for error in result.errors:
            self.console.print(f"[red]Batch {batch_num} preflight ERROR: {error}[/red]")

        self.console.print(
            f"[dim]Batch {batch_num}: ~{metrics.estimated_tokens:,} tokens, "
            f"{metrics.line_count:,} lines, "
            f"{len(metrics.referenced_paths)} file refs[/dim]"
        )

        return result

    def _update_batch_validation_state(
        self,
        state: CheckpointState,
        batch_num: int,
        result: BatchValidationResult,
    ) -> None:
        """Update batch state with validation tracking.

        Args:
            state: Current job state.
            batch_num: Batch number.
            result: Validation results.
        """
        batch = state.batches.get(batch_num)
        if batch:
            batch.validation_passed = result.all_passed
            batch.validation_details = result.to_dict_list()
            batch.last_pass_percentage = result.pass_percentage

            # Store validation descriptions for display
            batch.passed_validations = [
                r.rule.description or r.rule.path or "validation"
                for r in result.get_passed_results()
            ]
            batch.failed_validations = [
                r.rule.description or r.rule.path or "validation"
                for r in result.get_failed_results()
            ]

    def _classify_error(self, result: ExecutionResult) -> ClassifiedError:
        """Classify execution error.

        Args:
            result: Execution result with error details.

        Returns:
            ClassifiedError with category and retry info.
        """
        return self.error_classifier.classify(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            exit_signal=result.exit_signal,
            exit_reason=result.exit_reason,
        )

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in seconds.
        """
        base = self.config.retry.base_delay_seconds
        exp = self.config.retry.exponential_base
        max_delay = self.config.retry.max_delay_seconds

        delay = min(base * (exp ** (attempt - 1)), max_delay)

        if self.config.retry.jitter:
            # Add 50-100% jitter
            delay *= 0.5 + random.random()

        return delay

    async def _handle_rate_limit(self, state: CheckpointState) -> None:
        """Handle rate limit by waiting and health checking.

        Args:
            state: Current job state.

        Raises:
            FatalError: If max waits exceeded or health check fails.
        """
        wait_minutes = self.config.rate_limit.wait_minutes
        state.rate_limit_waits += 1
        await self.state_backend.save(state)

        # Log rate limit detection
        self._logger.warning(
            "rate_limit.detected",
            job_id=state.job_id,
            wait_count=state.rate_limit_waits,
            max_waits=self.config.rate_limit.max_waits,
            wait_minutes=wait_minutes,
        )

        if state.rate_limit_waits >= self.config.rate_limit.max_waits:
            self._logger.error(
                "rate_limit.exhausted",
                job_id=state.job_id,
                wait_count=state.rate_limit_waits,
                max_waits=self.config.rate_limit.max_waits,
            )
            raise FatalError(
                f"Exceeded maximum rate limit waits ({self.config.rate_limit.max_waits})"
            )

        self.console.print(
            f"[yellow]Rate limited. Waiting {wait_minutes} minutes... "
            f"(wait {state.rate_limit_waits}/{self.config.rate_limit.max_waits})[/yellow]"
        )
        await asyncio.sleep(wait_minutes * 60)

        # Health check before resuming
        self.console.print("[blue]Running health check...[/blue]")
        if not await self.backend.health_check():
            self._logger.error(
                "rate_limit.health_check_failed",
                job_id=state.job_id,
            )
            raise FatalError("Backend health check failed after rate limit wait")

        self._logger.info(
            "rate_limit.resumed",
            job_id=state.job_id,
            wait_count=state.rate_limit_waits,
        )
        self.console.print("[green]Health check passed, resuming...[/green]")

    async def _record_batch_outcome(
        self,
        batch_num: int,
        job_id: str,
        validation_result: BatchValidationResult,
        execution_duration: float,
        normal_attempts: int,
        completion_attempts: int,
        first_attempt_success: bool,
        final_status: BatchStatus,
    ) -> None:
        """Record batch outcome for learning if outcome store is available.

        Args:
            batch_num: Batch number that completed.
            job_id: Job identifier.
            validation_result: Validation results from the batch.
            execution_duration: Total execution time in seconds.
            normal_attempts: Number of normal/retry attempts.
            completion_attempts: Number of completion mode attempts.
            first_attempt_success: Whether batch succeeded on first attempt.
            final_status: Final batch status.
        """
        if self.outcome_store is None:
            return

        outcome = BatchOutcome(
            batch_id=f"{job_id}_batch_{batch_num}",
            job_id=job_id,
            validation_results=validation_result.to_dict_list(),
            execution_duration=execution_duration,
            retry_count=normal_attempts - 1 if normal_attempts > 0 else 0,
            completion_mode_used=completion_attempts > 0,
            final_status=final_status,
            validation_pass_rate=validation_result.pass_percentage,
            first_attempt_success=first_attempt_success,
            patterns_detected=[],  # Future: pattern detection logic
            timestamp=_utc_now(),
        )

        await self.outcome_store.record(outcome)

    def _decide_next_action(
        self,
        validation_result: BatchValidationResult,
        normal_attempts: int,
        completion_attempts: int,
    ) -> tuple[BatchExecutionMode, str]:
        """Decide the next action based on confidence and pass percentage.

        This method implements adaptive retry strategy using validation confidence
        and pass percentage to decide between COMPLETION, RETRY, or ESCALATE modes.

        Decision logic:
        - If confidence > high_threshold AND pass_pct > threshold: COMPLETION mode
        - If confidence < min_threshold: ESCALATE if handler exists, else RETRY
        - Otherwise: Use existing logic (RETRY or COMPLETION based on pass_pct)

        Args:
            validation_result: Results from validation engine with confidence scores.
            normal_attempts: Number of retry attempts already made.
            completion_attempts: Number of completion mode attempts already made.

        Returns:
            Tuple of (BatchExecutionMode, reason string explaining the decision).
        """
        confidence = validation_result.aggregate_confidence
        pass_pct = validation_result.pass_percentage
        completion_threshold = self.config.retry.completion_threshold_percent
        max_completion = self.config.retry.max_completion_attempts

        # Get thresholds from learning config
        high_threshold = self.config.learning.high_confidence_threshold
        min_threshold = self.config.learning.min_confidence_threshold

        # High confidence + majority passed -> completion mode (focused approach)
        if confidence > high_threshold and pass_pct > completion_threshold:
            if completion_attempts < max_completion:
                return (
                    BatchExecutionMode.COMPLETION,
                    f"high confidence ({confidence:.2f}) with {pass_pct:.0f}% passed, "
                    f"attempting focused completion",
                )
            else:
                # Completion attempts exhausted, fall back to retry
                return (
                    BatchExecutionMode.RETRY,
                    f"completion attempts exhausted ({completion_attempts}/{max_completion}), "
                    f"falling back to full retry",
                )

        # Low confidence -> escalate if enabled and handler available, else retry
        if confidence < min_threshold:
            escalation_available = (
                self.config.learning.escalation_enabled
                and self.escalation_handler is not None
            )
            if escalation_available:
                return (
                    BatchExecutionMode.ESCALATE,
                    f"low confidence ({confidence:.2f}) requires escalation",
                )
            else:
                return (
                    BatchExecutionMode.RETRY,
                    f"low confidence ({confidence:.2f}) but escalation not available, "
                    f"attempting retry",
                )

        # Medium confidence zone: use pass percentage to decide
        if pass_pct > completion_threshold and completion_attempts < max_completion:
            return (
                BatchExecutionMode.COMPLETION,
                f"medium confidence ({confidence:.2f}) with {pass_pct:.0f}% passed, "
                f"attempting completion mode",
            )
        else:
            return (
                BatchExecutionMode.RETRY,
                f"medium confidence ({confidence:.2f}) with {pass_pct:.0f}% passed, "
                f"full retry needed",
            )

    async def _decide_with_judgment(
        self,
        batch_num: int,
        validation_result: BatchValidationResult,
        execution_history: list[ExecutionResult],
        normal_attempts: int,
        completion_attempts: int,
    ) -> tuple[BatchExecutionMode, str, list[str] | None]:
        """Decide next action using Recursive Light judgment if available.

        Consults the judgment_client (Recursive Light) for TDF-aligned decisions.
        Falls back to local _decide_next_action() if no judgment client is configured
        or if the judgment client encounters errors.

        Args:
            batch_num: Current batch number.
            validation_result: Results from validation engine with confidence scores.
            execution_history: List of ExecutionResults from previous attempts.
            normal_attempts: Number of retry attempts already made.
            completion_attempts: Number of completion mode attempts already made.

        Returns:
            Tuple of (BatchExecutionMode, reason string, optional prompt_modifications).
            The prompt_modifications list is provided if judgment recommends it.
        """
        # Fall back to local decision if no judgment client
        if self.judgment_client is None:
            mode, reason = self._decide_next_action(
                validation_result, normal_attempts, completion_attempts
            )
            return mode, reason, None

        # Build JudgmentQuery from current state
        query = self._build_judgment_query(
            batch_num=batch_num,
            validation_result=validation_result,
            execution_history=execution_history,
            normal_attempts=normal_attempts,
        )

        try:
            # Get judgment from Recursive Light
            response = await self.judgment_client.get_judgment(query)

            # Log any patterns learned
            if response.patterns_learned:
                for pattern in response.patterns_learned:
                    self.console.print(
                        f"[dim]Batch {batch_num}: Pattern learned: {pattern}[/dim]"
                    )

            # Map JudgmentResponse.recommended_action to BatchExecutionMode
            mode = self._map_judgment_to_mode(response, completion_attempts)
            reason = f"RL judgment ({response.confidence:.2f}): {response.reasoning}"

            return mode, reason, response.prompt_modifications

        except Exception as e:
            # On any error, fall back to local decision
            self.console.print(
                f"[yellow]Batch {batch_num}: Judgment client error: {e}, "
                f"falling back to local decision[/yellow]"
            )
            mode, reason = self._decide_next_action(
                validation_result, normal_attempts, completion_attempts
            )
            return mode, reason + " (judgment fallback)", None

    def _build_judgment_query(
        self,
        batch_num: int,
        validation_result: BatchValidationResult,
        execution_history: list[ExecutionResult],
        normal_attempts: int,
    ) -> JudgmentQuery:
        """Build a JudgmentQuery from current execution state.

        Args:
            batch_num: Current batch number.
            validation_result: Validation results with confidence.
            execution_history: Previous execution results for this batch.
            normal_attempts: Number of retry attempts made.

        Returns:
            JudgmentQuery populated with current state.
        """
        # Extract error patterns from execution history
        error_patterns: list[str] = []
        for result in execution_history:
            if not result.success and result.stderr:
                # Extract first line as pattern identifier
                first_line = result.stderr.split("\n")[0][:100]
                if first_line and first_line not in error_patterns:
                    error_patterns.append(first_line)

        # Serialize execution history for query
        history_dicts: list[dict[str, Any]] = []
        for result in execution_history:
            history_dicts.append({
                "success": result.success,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "has_stdout": bool(result.stdout),
                "has_stderr": bool(result.stderr),
            })

        # Get similar outcomes from outcome store if available
        similar_outcomes: list[dict[str, Any]] = []
        # Note: Async call to outcome_store.query_similar() could be added here
        # For now, leave empty - future enhancement

        return JudgmentQuery(
            job_id=self.config.name,
            batch_num=batch_num,
            validation_results=validation_result.to_dict_list(),
            execution_history=history_dicts,
            error_patterns=error_patterns,
            retry_count=normal_attempts,
            confidence=validation_result.aggregate_confidence,
            similar_outcomes=similar_outcomes,
        )

    def _map_judgment_to_mode(
        self,
        response: JudgmentResponse,
        completion_attempts: int,
    ) -> BatchExecutionMode:
        """Map JudgmentResponse.recommended_action to BatchExecutionMode.

        Args:
            response: JudgmentResponse from Recursive Light.
            completion_attempts: Number of completion attempts already made.

        Returns:
            BatchExecutionMode corresponding to the recommended action.

        Raises:
            FatalError: If action is 'abort'.
        """
        action = response.recommended_action
        max_completion = self.config.retry.max_completion_attempts

        if action == "proceed":
            # Proceed means validation passed - this shouldn't happen in decision flow
            # but handle gracefully by treating as completion success
            return BatchExecutionMode.NORMAL

        elif action == "retry":
            return BatchExecutionMode.RETRY

        elif action == "completion":
            # Check if we have completion attempts left
            if completion_attempts < max_completion:
                return BatchExecutionMode.COMPLETION
            else:
                # Exhausted completion attempts, fall back to retry
                return BatchExecutionMode.RETRY

        elif action == "escalate":
            return BatchExecutionMode.ESCALATE

        elif action == "abort":
            # Abort raises FatalError to stop the job
            raise FatalError(
                f"RL judgment recommends abort: {response.reasoning}"
            )

        else:
            # Unknown action - fall back to retry
            return BatchExecutionMode.RETRY

    async def _handle_escalation(
        self,
        state: CheckpointState,
        batch_num: int,
        validation_result: BatchValidationResult,
        current_prompt: str,
        error_history: list[str],
        normal_attempts: int,
    ) -> EscalationResponse:
        """Handle escalation for low-confidence batch decisions.

        Builds escalation context and invokes the escalation handler
        to get a decision on how to proceed.

        Args:
            state: Current job state.
            batch_num: Batch number being processed.
            validation_result: Validation results from the batch.
            current_prompt: The prompt that was used for execution.
            error_history: List of previous error messages.
            normal_attempts: Number of retry attempts already made.

        Returns:
            EscalationResponse with the action to take.

        Raises:
            FatalError: If no escalation handler is configured.
        """
        if self.escalation_handler is None:
            raise FatalError(
                f"Batch {batch_num}: Escalation requested but no handler configured"
            )

        batch_state = state.batches.get(batch_num)
        if batch_state is None:
            raise FatalError(f"Batch {batch_num}: No batch state found for escalation")

        # Build escalation context
        context = EscalationContext(
            job_id=state.job_id,
            batch_num=batch_num,
            validation_results=validation_result.to_dict_list(),
            confidence=validation_result.aggregate_confidence,
            retry_count=normal_attempts,
            error_history=error_history,
            prompt_used=current_prompt,
            output_summary=batch_state.error_message or "",
        )

        self.console.print(
            f"[yellow]Batch {batch_num}: Escalating due to low confidence "
            f"({context.confidence:.1%})[/yellow]"
        )

        # Get response from escalation handler
        response = await self.escalation_handler.escalate(context)

        self.console.print(
            f"[blue]Batch {batch_num}: Escalation response: {response.action}[/blue]"
        )
        if response.guidance:
            self.console.print(f"[dim]Guidance: {response.guidance}[/dim]")

        return response
