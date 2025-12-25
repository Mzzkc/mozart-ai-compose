"""Checkpoint and state management models.

Defines the state that gets persisted between runs for resumable orchestration.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from mozart.core.logging import get_logger

# Module-level logger for checkpoint operations
_logger = get_logger("checkpoint")

# Constants for output capture
MAX_OUTPUT_CAPTURE_BYTES: int = 10240  # 10KB - last N bytes of stdout/stderr to capture


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Replacement for deprecated datetime.utcnow().
    """
    return datetime.now(UTC)


class BatchStatus(str, Enum):
    """Status of a single batch."""

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


class BatchState(BaseModel):
    """State for a single batch."""

    batch_num: int
    status: BatchStatus = BatchStatus.PENDING
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
        description="How long the batch execution took in seconds",
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
        description="Patterns recognized from this batch execution",
    )
    similar_outcomes_count: int = Field(
        default=0,
        description="Number of similar historical outcomes found",
    )
    first_attempt_success: bool = Field(
        default=False,
        description="Whether batch succeeded on first attempt (no retries/completion)",
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


class CheckpointState(BaseModel):
    """Complete checkpoint state for a job run.

    This is the primary state object that gets persisted and restored
    for resumable job execution.
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
    total_batches: int
    last_completed_batch: int = Field(
        default=0, description="Last successfully completed batch number"
    )
    current_batch: int | None = Field(default=None, description="Currently processing batch")
    status: JobStatus = JobStatus.PENDING

    # Batch-level state
    batches: dict[int, BatchState] = Field(default_factory=dict)

    # Execution metadata
    pid: int | None = Field(default=None, description="Process ID of running orchestrator")
    error_message: str | None = None
    total_retry_count: int = Field(default=0, description="Total retries across all batches")
    rate_limit_waits: int = Field(default=0, description="Number of rate limit waits")

    def get_next_batch(self) -> int | None:
        """Determine the next batch to process.

        Returns None if all batches are complete.
        """
        if self.status in (JobStatus.COMPLETED, JobStatus.CANCELLED):
            return None

        # Check for in-progress batch (resume from crash)
        if self.current_batch is not None:
            batch_state = self.batches.get(self.current_batch)
            if batch_state and batch_state.status == BatchStatus.IN_PROGRESS:
                return self.current_batch

        # Find next pending batch after last completed
        for batch_num in range(self.last_completed_batch + 1, self.total_batches + 1):
            batch_state = self.batches.get(batch_num)
            if batch_state is None:
                return batch_num
            if batch_state.status in (BatchStatus.PENDING, BatchStatus.FAILED):
                return batch_num

        return None

    def mark_batch_started(self, batch_num: int) -> None:
        """Mark a batch as started."""
        previous_status = self.status
        self.current_batch = batch_num
        self.status = JobStatus.RUNNING
        self.updated_at = _utc_now()

        if batch_num not in self.batches:
            self.batches[batch_num] = BatchState(batch_num=batch_num)

        batch = self.batches[batch_num]
        batch.status = BatchStatus.IN_PROGRESS
        batch.started_at = _utc_now()
        batch.attempt_count += 1

        _logger.debug(
            "batch_started",
            job_id=self.job_id,
            batch_num=batch_num,
            attempt_count=batch.attempt_count,
            previous_status=previous_status.value,
            total_batches=self.total_batches,
        )

    def mark_batch_completed(
        self,
        batch_num: int,
        validation_passed: bool = True,
        validation_details: list[dict[str, Any]] | None = None,
    ) -> None:
        """Mark a batch as completed."""
        self.updated_at = _utc_now()

        batch = self.batches[batch_num]
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = _utc_now()
        batch.exit_code = 0
        batch.validation_passed = validation_passed
        batch.validation_details = validation_details

        self.last_completed_batch = batch_num
        self.current_batch = None

        # Check if job is complete
        job_completed = batch_num >= self.total_batches
        if job_completed:
            self.status = JobStatus.COMPLETED
            self.completed_at = _utc_now()

        _logger.debug(
            "batch_completed",
            job_id=self.job_id,
            batch_num=batch_num,
            validation_passed=validation_passed,
            attempt_count=batch.attempt_count,
            job_completed=job_completed,
            progress=f"{self.last_completed_batch}/{self.total_batches}",
        )

    def mark_batch_failed(
        self,
        batch_num: int,
        error_message: str,
        error_category: str | None = None,
        exit_code: int | None = None,
        exit_signal: int | None = None,
        exit_reason: str | None = None,
        execution_duration_seconds: float | None = None,
    ) -> None:
        """Mark a batch as failed.

        Args:
            batch_num: Batch number that failed.
            error_message: Human-readable error description.
            error_category: Error category from ErrorClassifier (e.g., "signal", "timeout").
            exit_code: Process exit code (None if killed by signal).
            exit_signal: Signal number if killed by signal (e.g., 9=SIGKILL, 15=SIGTERM).
            exit_reason: Why execution ended ("completed", "timeout", "killed", "error").
            execution_duration_seconds: How long the batch execution took.
        """
        self.updated_at = _utc_now()

        batch = self.batches[batch_num]
        batch.status = BatchStatus.FAILED
        batch.completed_at = _utc_now()
        batch.error_message = error_message
        batch.error_category = error_category
        batch.exit_code = exit_code
        batch.exit_signal = exit_signal
        batch.exit_reason = exit_reason
        if execution_duration_seconds is not None:
            batch.execution_duration_seconds = execution_duration_seconds

        self.current_batch = None
        self.total_retry_count += 1

        _logger.debug(
            "batch_failed",
            job_id=self.job_id,
            batch_num=batch_num,
            error_category=error_category,
            exit_code=exit_code,
            exit_signal=exit_signal,
            exit_reason=exit_reason,
            attempt_count=batch.attempt_count,
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
            last_completed_batch=self.last_completed_batch,
            total_batches=self.total_batches,
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
            last_completed_batch=self.last_completed_batch,
            total_batches=self.total_batches,
            current_batch=self.current_batch,
        )

    def get_progress(self) -> tuple[int, int]:
        """Get progress as (completed, total)."""
        completed = sum(
            1 for b in self.batches.values()
            if b.status == BatchStatus.COMPLETED
        )
        return completed, self.total_batches

    def get_progress_percent(self) -> float:
        """Get progress as percentage."""
        completed, total = self.get_progress()
        return (completed / total * 100) if total > 0 else 0.0
