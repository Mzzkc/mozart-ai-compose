"""Dashboard API routes.

All routes are prefixed with /api for clear API namespace separation.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mozart.core.checkpoint import BatchStatus, CheckpointState, JobStatus
from mozart.dashboard.app import get_state_backend
from mozart.state.base import StateBackend

router = APIRouter(prefix="/api", tags=["Jobs"])


# ============================================================================
# Response Models (Pydantic schemas for API responses)
# ============================================================================


class BatchSummary(BaseModel):
    """Summarized batch information for list views."""

    batch_num: int
    status: BatchStatus
    attempt_count: int = 0
    validation_passed: bool | None = None


class JobSummary(BaseModel):
    """Summarized job information for list views."""

    job_id: str
    job_name: str
    status: JobStatus
    total_batches: int
    completed_batches: int
    progress_percent: float
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_checkpoint(cls, state: CheckpointState) -> "JobSummary":
        """Create from CheckpointState."""
        completed, total = state.get_progress()
        return cls(
            job_id=state.job_id,
            job_name=state.job_name,
            status=state.status,
            total_batches=total,
            completed_batches=completed,
            progress_percent=state.get_progress_percent(),
            created_at=state.created_at,
            updated_at=state.updated_at,
        )


class JobDetail(BaseModel):
    """Full job details including batch information."""

    job_id: str
    job_name: str
    status: JobStatus
    total_batches: int
    last_completed_batch: int
    current_batch: int | None
    progress_percent: float
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    total_retry_count: int
    rate_limit_waits: int
    batches: list[BatchSummary]

    @classmethod
    def from_checkpoint(cls, state: CheckpointState) -> "JobDetail":
        """Create from CheckpointState."""
        batches = [
            BatchSummary(
                batch_num=b.batch_num,
                status=b.status,
                attempt_count=b.attempt_count,
                validation_passed=b.validation_passed,
            )
            for b in sorted(state.batches.values(), key=lambda x: x.batch_num)
        ]
        return cls(
            job_id=state.job_id,
            job_name=state.job_name,
            status=state.status,
            total_batches=state.total_batches,
            last_completed_batch=state.last_completed_batch,
            current_batch=state.current_batch,
            progress_percent=state.get_progress_percent(),
            created_at=state.created_at,
            updated_at=state.updated_at,
            started_at=state.started_at,
            completed_at=state.completed_at,
            error_message=state.error_message,
            total_retry_count=state.total_retry_count,
            rate_limit_waits=state.rate_limit_waits,
            batches=batches,
        )


class JobStatusResponse(BaseModel):
    """Focused status information for job monitoring."""

    job_id: str
    status: JobStatus
    progress_percent: float
    completed_batches: int
    total_batches: int
    current_batch: int | None
    error_message: str | None
    updated_at: datetime

    @classmethod
    def from_checkpoint(cls, state: CheckpointState) -> "JobStatusResponse":
        """Create from CheckpointState."""
        completed, total = state.get_progress()
        return cls(
            job_id=state.job_id,
            status=state.status,
            progress_percent=state.get_progress_percent(),
            completed_batches=completed,
            total_batches=total,
            current_batch=state.current_batch,
            error_message=state.error_message,
            updated_at=state.updated_at,
        )


class JobListResponse(BaseModel):
    """Response for job list endpoint."""

    jobs: list[JobSummary]
    total: int


# ============================================================================
# API Endpoints
# ============================================================================


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: JobStatus | None = None,
    limit: int = 50,
    backend: StateBackend = Depends(get_state_backend),
) -> JobListResponse:
    """List all jobs with optional status filter.

    Args:
        status: Filter by job status (optional)
        limit: Maximum number of jobs to return
        backend: State backend (injected)

    Returns:
        List of job summaries
    """
    all_jobs = await backend.list_jobs()

    # Apply status filter
    if status is not None:
        all_jobs = [j for j in all_jobs if j.status == status]

    # Apply limit
    limited_jobs = all_jobs[:limit]

    return JobListResponse(
        jobs=[JobSummary.from_checkpoint(j) for j in limited_jobs],
        total=len(all_jobs),
    )


@router.get("/jobs/{job_id}", response_model=JobDetail)
async def get_job(
    job_id: str,
    backend: StateBackend = Depends(get_state_backend),
) -> JobDetail:
    """Get detailed information about a specific job.

    Args:
        job_id: Unique job identifier
        backend: State backend (injected)

    Returns:
        Full job details

    Raises:
        HTTPException: 404 if job not found
    """
    state = await backend.load(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobDetail.from_checkpoint(state)


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    backend: StateBackend = Depends(get_state_backend),
) -> JobStatusResponse:
    """Get focused status information for a job.

    Lightweight endpoint for polling job progress.

    Args:
        job_id: Unique job identifier
        backend: State backend (injected)

    Returns:
        Job status summary

    Raises:
        HTTPException: 404 if job not found
    """
    state = await backend.load(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse.from_checkpoint(state)
