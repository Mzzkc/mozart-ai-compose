"""Checkpoint and state management models.

Defines the state that gets persisted between runs for resumable orchestration.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from mozart.core.logging import get_logger

# Module-level logger for checkpoint operations
_logger = get_logger("checkpoint")

# Constants for output capture
MAX_OUTPUT_CAPTURE_BYTES: int = 10240  # 10KB - last N bytes of stdout/stderr to capture

# Constants for error history (Task 10: Error History Model)
MAX_ERROR_HISTORY: int = 10  # Maximum number of error records to keep per sheet

# Type alias for error types
ErrorType = Literal["transient", "rate_limit", "permanent"]


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Replacement for deprecated datetime.utcnow().
    """
    return datetime.now(UTC)


class SheetStatus(str, Enum):
    """Status of a single sheet."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStatus(str, Enum):
    """Status of an entire job run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ErrorRecord(BaseModel):
    """Record of a single error occurrence during sheet execution.

    Stores structured error information for debugging and pattern analysis.
    Error history is trimmed to MAX_ERROR_HISTORY records per sheet to
    prevent unbounded state growth.
    """

    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="When the error occurred (UTC)",
    )
    error_type: ErrorType = Field(
        description="Error classification: transient, rate_limit, or permanent",
    )
    error_code: str = Field(
        description="Error code for categorization (e.g., E001, E002)",
    )
    error_message: str = Field(
        description="Human-readable error description",
    )
    attempt_number: int = Field(
        ge=1,
        description="Which attempt this error occurred on (1-based)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (exit_code, signal, category, etc.)",
    )
    stdout_tail: str | None = Field(
        default=None,
        description="Last portion of stdout when error occurred",
    )
    stderr_tail: str | None = Field(
        default=None,
        description="Last portion of stderr when error occurred",
    )
    stack_trace: str | None = Field(
        default=None,
        description="Stack trace if exception was caught",
    )


class SheetState(BaseModel):
    """State for a single sheet."""

    sheet_num: int
    status: SheetStatus = SheetStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    attempt_count: int = 0
    exit_code: int | None = None
    error_message: str | None = None
    error_category: str | None = None
    validation_passed: bool | None = None
    validation_details: list[dict[str, Any]] | None = None

    # Exit signal differentiation (Task 3: Exit Signal Differentiation)
    exit_signal: int | None = Field(
        default=None,
        description="Signal number if process was killed (e.g., 9=SIGKILL, 15=SIGTERM)",
    )
    exit_reason: str | None = Field(
        default=None,
        description="Why execution ended: completed, timeout, killed, or error",
    )
    execution_duration_seconds: float | None = Field(
        default=None,
        description="How long the sheet execution took in seconds",
    )

    # Partial completion tracking
    completion_attempts: int = Field(
        default=0,
        description="Number of completion prompt attempts for partial recovery",
    )
    passed_validations: list[str] = Field(
        default_factory=list,
        description="Descriptions of validations that passed",
    )
    failed_validations: list[str] = Field(
        default_factory=list,
        description="Descriptions of validations that failed",
    )
    last_pass_percentage: float | None = Field(
        default=None,
        description="Last validation pass percentage",
    )
    execution_mode: str | None = Field(
        default=None,
        description="Last execution mode: normal, completion, or retry",
    )

    # Learning metadata (Phase 1: Learning Foundation)
    outcome_data: dict[str, Any] | None = Field(
        default=None,
        description="Structured outcome data for learning and pattern recognition",
    )
    confidence_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate confidence in outcome quality (0.0-1.0)",
    )
    learned_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns recognized from this sheet execution",
    )
    similar_outcomes_count: int = Field(
        default=0,
        description="Number of similar historical outcomes found",
    )
    first_attempt_success: bool = Field(
        default=False,
        description="Whether sheet succeeded on first attempt (no retries/completion)",
    )
    outcome_category: str | None = Field(
        default=None,
        description=(
            "Outcome classification: success_first_try, success_completion, "
            "success_retry, failed_exhausted, failed_fatal"
        ),
    )

    # Raw output capture for debugging (last N bytes to avoid memory issues)
    stdout_tail: str | None = Field(
        default=None,
        description="Last 10KB of stdout for debugging failed executions",
    )
    stderr_tail: str | None = Field(
        default=None,
        description="Last 10KB of stderr for debugging failed executions",
    )
    output_truncated: bool = Field(
        default=False,
        description="True if output was larger than capture limit and was truncated",
    )

    # Preflight metrics (Task 2: Prompt Metrics and Pre-flight Checks)
    prompt_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Prompt analysis metrics (character_count, estimated_tokens, etc.)",
    )
    preflight_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from preflight checks (large prompts, missing files, etc.)",
    )

    # Execution progress tracking (Task 4: Execution Progress Tracking)
    progress_snapshots: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Periodic progress records during execution (bytes, lines, phase)",
    )
    last_activity_at: datetime | None = Field(
        default=None,
        description="Last time activity was observed during execution",
    )

    # Error history tracking (Task 10: Error History Model)
    error_history: list[ErrorRecord] = Field(
        default_factory=list,
        description="History of errors encountered during sheet execution (max 10)",
    )

    def capture_output(
        self,
        stdout: str,
        stderr: str,
        max_bytes: int = MAX_OUTPUT_CAPTURE_BYTES,
    ) -> None:
        """Capture tail of stdout/stderr for debugging.

        Stores the last `max_bytes` of each output stream. Sets output_truncated
        to True if either stream was larger than the limit.

        Args:
            stdout: Full stdout string from execution.
            stderr: Full stderr string from execution.
            max_bytes: Maximum bytes to capture per stream (default 10KB).
        """
        # Convert to bytes to measure actual size, then slice
        stdout_bytes = stdout.encode("utf-8", errors="replace")
        stderr_bytes = stderr.encode("utf-8", errors="replace")

        stdout_truncated = len(stdout_bytes) > max_bytes
        stderr_truncated = len(stderr_bytes) > max_bytes

        # Capture tail (last N bytes) and decode back to string
        if stdout_truncated:
            self.stdout_tail = stdout_bytes[-max_bytes:].decode(
                "utf-8", errors="replace"
            )
        else:
            self.stdout_tail = stdout if stdout else None

        if stderr_truncated:
            self.stderr_tail = stderr_bytes[-max_bytes:].decode(
                "utf-8", errors="replace"
            )
        else:
            self.stderr_tail = stderr if stderr else None

        self.output_truncated = stdout_truncated or stderr_truncated

    def record_error(
        self,
        error_type: ErrorType,
        error_code: str,
        error_message: str,
        attempt: int,
        *,
        stdout_tail: str | None = None,
        stderr_tail: str | None = None,
        stack_trace: str | None = None,
        **context: Any,
    ) -> None:
        """Record an error with context, trimming to max history.

        Creates an ErrorRecord and appends it to error_history. If the history
        exceeds MAX_ERROR_HISTORY, the oldest records are removed.

        Logs the error at WARNING level for observability.

        Args:
            error_type: Classification of the error (transient, rate_limit, permanent).
            error_code: Error code for categorization (e.g., E001, E002).
            error_message: Human-readable error description.
            attempt: Which attempt this error occurred on (1-based).
            stdout_tail: Optional tail of stdout when error occurred.
            stderr_tail: Optional tail of stderr when error occurred.
            stack_trace: Optional stack trace if exception was caught.
            **context: Additional context to store (exit_code, signal, etc.).
        """
        record = ErrorRecord(
            error_type=error_type,
            error_code=error_code,
            error_message=error_message,
            attempt_number=attempt,
            context=context,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            stack_trace=stack_trace,
        )

        self.error_history.append(record)

        # Trim to max history (keep most recent)
        if len(self.error_history) > MAX_ERROR_HISTORY:
            self.error_history = self.error_history[-MAX_ERROR_HISTORY:]

        # Log at WARNING level for observability
        _logger.warning(
            "error_recorded",
            sheet_num=self.sheet_num,
            error_type=error_type,
            error_code=error_code,
            attempt=attempt,
            history_size=len(self.error_history),
            error_message=error_message[:100] if error_message else None,
            **{k: v for k, v in context.items() if k not in ("stdout_tail", "stderr_tail")},
        )


class CheckpointState(BaseModel):
    """Complete checkpoint state for a job run.

    This is the primary state object that gets persisted and restored
    for resumable job execution.

    Zombie Detection:
        A job is considered a "zombie" when the state shows RUNNING status
        but the associated process (tracked by `pid`) is no longer alive.
        This can happen when:
        - External timeout wrapper sends SIGKILL
        - System crash or forced termination
        - WSL shutdown while job running

        Use `is_zombie()` to detect this state, and `mark_zombie_detected()`
        to recover from it.
    """

    # Job identification
    job_id: str = Field(description="Unique ID for this job run")
    job_name: str = Field(description="Name from job config")
    config_hash: str | None = Field(default=None, description="Hash of config for change detection")

    # Config storage for resume (Task 3: Config Storage)
    config_snapshot: dict[str, Any] | None = Field(
        default=None,
        description="Serialized JobConfig for resume without config file",
    )
    config_path: str | None = Field(
        default=None,
        description="Original config file path for fallback and debugging",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Progress tracking
    total_sheets: int
    last_completed_sheet: int = Field(
        default=0, description="Last successfully completed sheet number"
    )
    current_sheet: int | None = Field(default=None, description="Currently processing sheet")
    status: JobStatus = JobStatus.PENDING

    # Sheet-level state
    sheets: dict[int, SheetState] = Field(default_factory=dict)

    # Execution metadata
    pid: int | None = Field(default=None, description="Process ID of running orchestrator")
    error_message: str | None = None
    total_retry_count: int = Field(default=0, description="Total retries across all sheets")
    rate_limit_waits: int = Field(default=0, description="Number of rate limit waits")

    def get_next_sheet(self) -> int | None:
        """Determine the next sheet to process.

        Returns None if all sheets are complete.
        """
        if self.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
            return None

        # Check for in-progress sheet (resume from crash)
        if self.current_sheet is not None:
            sheet_state = self.sheets.get(self.current_sheet)
            if sheet_state and sheet_state.status == SheetStatus.IN_PROGRESS:
                return self.current_sheet

        # Find next pending sheet after last completed
        for sheet_num in range(self.last_completed_sheet + 1, self.total_sheets + 1):
            sheet_state = self.sheets.get(sheet_num)
            if sheet_state is None:
                return sheet_num
            if sheet_state.status in (SheetStatus.PENDING, SheetStatus.FAILED):
                return sheet_num

        return None

    def mark_sheet_started(self, sheet_num: int) -> None:
        """Mark a sheet as started."""
        previous_status = self.status
        self.current_sheet = sheet_num
        self.status = JobStatus.RUNNING
        self.updated_at = _utc_now()

        if sheet_num not in self.sheets:
            self.sheets[sheet_num] = SheetState(sheet_num=sheet_num)

        sheet = self.sheets[sheet_num]
        sheet.status = SheetStatus.IN_PROGRESS
        sheet.started_at = _utc_now()
        sheet.attempt_count += 1

        _logger.debug(
            "sheet_started",
            job_id=self.job_id,
            sheet_num=sheet_num,
            attempt_count=sheet.attempt_count,
            previous_status=previous_status.value,
            total_sheets=self.total_sheets,
        )

    def mark_sheet_completed(
        self,
        sheet_num: int,
        validation_passed: bool = True,
        validation_details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Mark a sheet as completed."""
        self.updated_at = _utc_now()

        sheet = self.sheets[sheet_num]
        sheet.status = SheetStatus.COMPLETED
        sheet.completed_at = _utc_now()
        sheet.exit_code = 0
        sheet.validation_passed = validation_passed
        sheet.validation_details = validation_details

        self.last_completed_sheet = sheet_num
        self.current_sheet = None

        # Check if job is complete
        job_completed = sheet_num >= self.total_sheets
        if job_completed:
            self.status = JobStatus.COMPLETED
            self.completed_at = _utc_now()

        _logger.debug(
            "sheet_completed",
            job_id=self.job_id,
            sheet_num=sheet_num,
            validation_passed=validation_passed,
            attempt_count=sheet.attempt_count,
            job_completed=job_completed,
            progress=f"{self.last_completed_sheet}/{self.total_sheets}",
        )

    def mark_sheet_failed(
        self,
        sheet_num: int,
        error_message: str,
        error_category: str | None = None,
        exit_code: int | None = None,
        exit_signal: int | None = None,
        exit_reason: str | None = None,
        execution_duration_seconds: float | None = None,
    ) -> None:
        """Mark a sheet as failed.

        Args:
            sheet_num: Sheet number that failed.
            error_message: Human-readable error description.
            error_category: Error category from ErrorClassifier (e.g., "signal", "timeout").
            exit_code: Process exit code (None if killed by signal).
            exit_signal: Signal number if killed by signal (e.g., 9=SIGKILL, 15=SIGTERM).
            exit_reason: Why execution ended ("completed", "timeout", "killed", "error").
            execution_duration_seconds: How long the sheet execution took.
        """
        self.updated_at = _utc_now()

        sheet = self.sheets[sheet_num]
        sheet.status = SheetStatus.FAILED
        sheet.completed_at = _utc_now()
        sheet.error_message = error_message
        sheet.error_category = error_category
        sheet.exit_code = exit_code
        sheet.exit_signal = exit_signal
        sheet.exit_reason = exit_reason
        if execution_duration_seconds is not None:
            sheet.execution_duration_seconds = execution_duration_seconds

        self.current_sheet = None
        self.total_retry_count += 1

        _logger.debug(
            "sheet_failed",
            job_id=self.job_id,
            sheet_num=sheet_num,
            error_category=error_category,
            exit_code=exit_code,
            exit_signal=exit_signal,
            exit_reason=exit_reason,
            attempt_count=sheet.attempt_count,
            total_retry_count=self.total_retry_count,
            error_message=error_message[:100] if error_message else None,
        )

    def mark_job_failed(self, error_message: str) -> None:
        """Mark the entire job as failed."""
        previous_status = self.status
        self.status = JobStatus.FAILED
        self.error_message = error_message
        self.completed_at = _utc_now()
        self.updated_at = _utc_now()

        _logger.error(
            "job_failed",
            job_id=self.job_id,
            previous_status=previous_status.value,
            last_completed_sheet=self.last_completed_sheet,
            total_sheets=self.total_sheets,
            total_retry_count=self.total_retry_count,
            error_message=error_message[:200] if error_message else None,
        )

    def mark_job_paused(self) -> None:
        """Mark the job as paused."""
        previous_status = self.status
        self.status = JobStatus.PAUSED
        self.updated_at = _utc_now()

        _logger.info(
            "job_paused",
            job_id=self.job_id,
            previous_status=previous_status.value,
            last_completed_sheet=self.last_completed_sheet,
            total_sheets=self.total_sheets,
            current_sheet=self.current_sheet,
        )

    def get_progress(self) -> tuple[int, int]:
        """Get progress as (completed, total)."""
        completed = sum(
            1 for b in self.sheets.values()
            if b.status == SheetStatus.COMPLETED
        )
        return completed, self.total_sheets

    def get_progress_percent(self) -> float:
        """Get progress as percentage."""
        completed, total = self.get_progress()
        return (completed / total * 100) if total > 0 else 0.0

    def is_zombie(self, stale_threshold_seconds: float = 300.0) -> bool:
        """Check if this job is a zombie (RUNNING but process dead).

        A zombie state occurs when:
        1. Status is RUNNING
        2. PID is set
        3. Process with that PID is no longer alive

        Additionally, if the PID is alive but belongs to a different process
        (PID recycling), we check if updated_at is stale (default 5 minutes).

        Args:
            stale_threshold_seconds: Time after which a running job with no
                updates is considered potentially stale (default 300s = 5min).

        Returns:
            True if job appears to be a zombie, False otherwise.
        """
        import os

        # Only RUNNING jobs can be zombies
        if self.status != JobStatus.RUNNING:
            return False

        # If no PID recorded, can't determine - not a zombie by this check
        if self.pid is None:
            return False

        # Check if process is alive
        try:
            # os.kill with signal 0 checks if process exists without killing it
            os.kill(self.pid, 0)
            # Process exists - check for stale updates (PID recycling protection)
            if self.updated_at:
                now = _utc_now()
                elapsed = (now - self.updated_at).total_seconds()
                # If process is alive but updates are stale, might be recycled PID
                # This is a heuristic - real process would update more frequently
                if elapsed > stale_threshold_seconds:
                    _logger.warning(
                        "zombie_stale_pid_detected",
                        job_id=self.job_id,
                        pid=self.pid,
                        elapsed_seconds=round(elapsed, 1),
                        threshold_seconds=stale_threshold_seconds,
                    )
                    return True
            return False  # Process alive and recent updates
        except ProcessLookupError:
            # Process doesn't exist - definite zombie
            _logger.warning(
                "zombie_dead_pid_detected",
                job_id=self.job_id,
                pid=self.pid,
            )
            return True
        except PermissionError:
            # Can't check (different user) - assume not zombie
            return False
        except OSError:
            # Other OS error - assume not zombie to be safe
            return False

    def mark_zombie_detected(self, reason: str | None = None) -> None:
        """Mark this job as recovered from zombie state.

        Changes status from RUNNING to PAUSED, clears PID, and records
        the zombie recovery in the error message.

        Args:
            reason: Optional additional context about why zombie was detected.
        """
        previous_status = self.status
        previous_pid = self.pid

        self.status = JobStatus.PAUSED
        self.pid = None
        self.updated_at = _utc_now()

        # Build zombie recovery message
        zombie_msg = f"Zombie recovery: job was RUNNING (PID {previous_pid}) but process dead"
        if reason:
            zombie_msg += f". {reason}"

        # Preserve any existing error message
        if self.error_message:
            self.error_message = f"{zombie_msg}. Previous error: {self.error_message}"
        else:
            self.error_message = zombie_msg

        _logger.warning(
            "zombie_recovered",
            job_id=self.job_id,
            previous_status=previous_status.value,
            previous_pid=previous_pid,
            reason=reason,
        )

    def set_running_pid(self, pid: int | None = None) -> None:
        """Set the PID of the running orchestrator process.

        Call this when starting job execution to enable zombie detection.
        If pid is None, uses the current process PID.

        Args:
            pid: Process ID to record. Defaults to current process.
        """
        import os

        self.pid = pid if pid is not None else os.getpid()
        self.updated_at = _utc_now()

        _logger.debug(
            "running_pid_set",
            job_id=self.job_id,
            pid=self.pid,
        )
