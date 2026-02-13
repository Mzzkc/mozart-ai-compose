"""Shared data types for the Mozart daemon.

Defines request/response models and status types used across daemon components
(server, service, CLI bridge). All models are Pydantic v2 BaseModel for
serialization over IPC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class JobRequest(BaseModel):
    """Request to submit a job to the daemon.

    Sent by clients (CLI, dashboard) to the daemon over IPC.
    The daemon validates the config and either accepts or rejects.
    """

    config_path: Path = Field(
        description="Path to the job configuration YAML file",
    )
    workspace: Path | None = Field(
        default=None,
        description="Override workspace directory. "
        "If None, uses the workspace specified in the job config.",
    )
    fresh: bool = Field(
        default=False,
        description="Start with clean state, ignoring any existing checkpoint",
    )
    self_healing: bool = Field(
        default=False,
        description="Enable self-healing mode for automatic error recovery",
    )
    self_healing_auto_confirm: bool = Field(
        default=False,
        description="Auto-confirm suggested fixes in self-healing mode",
    )
    dry_run: bool = Field(
        default=False,
        description="Validate config and return without executing sheets",
    )


class JobResponse(BaseModel):
    """Response from the daemon after a job submission.

    Returned immediately — does not wait for job completion.
    Clients poll status separately via DaemonStatus or job-specific queries.
    """

    job_id: str = Field(
        description="Unique identifier for the submitted job",
    )
    status: Literal["accepted", "rejected", "error"] = Field(
        description="Submission result: accepted (queued), "
        "rejected (validation failed), or error (daemon fault)",
    )
    message: str | None = Field(
        default=None,
        description="Human-readable detail about the submission result",
    )


class DaemonStatus(BaseModel):
    """Current status snapshot of the running daemon.

    Returned by health check / status queries. Provides a lightweight
    overview without per-job detail.
    """

    pid: int = Field(
        description="Process ID of the daemon",
    )
    uptime_seconds: float = Field(
        description="Seconds since daemon started",
    )
    running_jobs: int = Field(
        description="Number of currently executing jobs",
    )
    total_jobs_active: int = Field(
        description="Total active jobs (proxy for sheet count until Phase 3 scheduler is wired)",
    )
    memory_usage_mb: float = Field(
        description="Current RSS memory usage in MB",
    )
    version: str = Field(
        description="Mozart version string",
    )
