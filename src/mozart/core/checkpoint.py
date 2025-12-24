"""Checkpoint and state management models.

Defines the state that gets persisted between runs for resumable orchestration.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    validation_passed: Optional[bool] = None
    validation_details: Optional[list[dict]] = None

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
    last_pass_percentage: Optional[float] = Field(
        default=None,
        description="Last validation pass percentage",
    )
    execution_mode: Optional[str] = Field(
        default=None,
        description="Last execution mode: normal, completion, or retry",
    )

    # Learning metadata (Phase 1: Learning Foundation)
    outcome_data: Optional[dict] = Field(
        default=None,
        description="Structured outcome data for learning and pattern recognition",
    )
    confidence_score: Optional[float] = Field(
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
    outcome_category: Optional[str] = Field(
        default=None,
        description="Outcome classification: success_first_try, success_completion, success_retry, failed_exhausted, failed_fatal",
    )


class CheckpointState(BaseModel):
    """Complete checkpoint state for a job run.

    This is the primary state object that gets persisted and restored
    for resumable job execution.
    """

    # Job identification
    job_id: str = Field(description="Unique ID for this job run")
    job_name: str = Field(description="Name from job config")
    config_hash: Optional[str] = Field(default=None, description="Hash of config for change detection")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Progress tracking
    total_batches: int
    last_completed_batch: int = Field(default=0, description="Last successfully completed batch number")
    current_batch: Optional[int] = Field(default=None, description="Currently processing batch")
    status: JobStatus = JobStatus.PENDING

    # Batch-level state
    batches: dict[int, BatchState] = Field(default_factory=dict)

    # Execution metadata
    pid: Optional[int] = Field(default=None, description="Process ID of running orchestrator")
    error_message: Optional[str] = None
    total_retry_count: int = Field(default=0, description="Total retries across all batches")
    rate_limit_waits: int = Field(default=0, description="Number of rate limit waits")

    def get_next_batch(self) -> Optional[int]:
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
            if batch_state is None or batch_state.status in (BatchStatus.PENDING, BatchStatus.FAILED):
                return batch_num

        return None

    def mark_batch_started(self, batch_num: int) -> None:
        """Mark a batch as started."""
        self.current_batch = batch_num
        self.status = JobStatus.RUNNING
        self.updated_at = datetime.utcnow()

        if batch_num not in self.batches:
            self.batches[batch_num] = BatchState(batch_num=batch_num)

        batch = self.batches[batch_num]
        batch.status = BatchStatus.IN_PROGRESS
        batch.started_at = datetime.utcnow()
        batch.attempt_count += 1

    def mark_batch_completed(
        self,
        batch_num: int,
        validation_passed: bool = True,
        validation_details: Optional[list[dict]] = None,
    ) -> None:
        """Mark a batch as completed."""
        self.updated_at = datetime.utcnow()

        batch = self.batches[batch_num]
        batch.status = BatchStatus.COMPLETED
        batch.completed_at = datetime.utcnow()
        batch.exit_code = 0
        batch.validation_passed = validation_passed
        batch.validation_details = validation_details

        self.last_completed_batch = batch_num
        self.current_batch = None

        # Check if job is complete
        if batch_num >= self.total_batches:
            self.status = JobStatus.COMPLETED
            self.completed_at = datetime.utcnow()

    def mark_batch_failed(
        self,
        batch_num: int,
        error_message: str,
        error_category: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> None:
        """Mark a batch as failed."""
        self.updated_at = datetime.utcnow()

        batch = self.batches[batch_num]
        batch.status = BatchStatus.FAILED
        batch.completed_at = datetime.utcnow()
        batch.error_message = error_message
        batch.error_category = error_category
        batch.exit_code = exit_code

        self.current_batch = None
        self.total_retry_count += 1

    def mark_job_failed(self, error_message: str) -> None:
        """Mark the entire job as failed."""
        self.status = JobStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_job_paused(self) -> None:
        """Mark the job as paused."""
        self.status = JobStatus.PAUSED
        self.updated_at = datetime.utcnow()

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
