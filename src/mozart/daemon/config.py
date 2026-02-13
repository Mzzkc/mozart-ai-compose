"""Configuration models for the Mozart daemon (mozartd).

Defines Pydantic v2 models for daemon-specific settings: socket configuration,
resource limits, concurrency controls, and state backend selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ResourceLimitConfig(BaseModel):
    """Resource constraints for the daemon process.

    Prevents runaway resource consumption by capping memory, processes,
    and API call rates at the daemon level.
    """

    max_memory_mb: int = Field(
        default=8192,
        ge=512,
        description="Maximum RSS memory in MB before daemon triggers backpressure",
    )
    max_processes: int = Field(
        default=50,
        ge=5,
        description="Maximum child processes (backends + validation commands)",
    )
    max_api_calls_per_minute: int = Field(
        default=60,
        ge=1,
        description="Global rate limit for API calls across all jobs. "
        "NOT YET ENFORCED — rate limiting currently works through "
        "externally-reported events via RateLimitCoordinator. "
        "Setting a non-default value will log a warning at startup.",
    )


class SocketConfig(BaseModel):
    """Unix domain socket configuration for daemon IPC.

    Uses Unix sockets (not TCP) for security — no network exposure.
    The socket file is created on daemon start and removed on clean shutdown.
    """

    path: Path = Field(
        default=Path("/tmp/mozartd.sock"),
        description="Unix domain socket path for client-daemon communication",
    )
    permissions: int = Field(
        default=0o660,
        description="File permissions for the socket (octal). "
        "0o660 = owner+group read/write, no world access.",
    )
    backlog: int = Field(
        default=5,
        ge=1,
        description="Maximum pending connections in the socket listen queue",
    )


class DaemonConfig(BaseModel):
    """Top-level configuration for the Mozart daemon (mozartd).

    Controls socket binding, PID file location, concurrency limits,
    resource constraints, and state backend selection. Follows the
    same Field() conventions as mozart.core.config.
    """

    socket: SocketConfig = Field(
        default_factory=SocketConfig,
        description="Unix domain socket configuration for IPC",
    )
    pid_file: Path = Field(
        default=Path("/tmp/mozartd.pid"),
        description="PID file for daemon process management. "
        "Used to detect already-running daemons and for signal delivery.",
    )
    max_concurrent_jobs: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum jobs executing simultaneously. "
        "Each job runs as an asyncio.Task in the daemon event loop.",
    )
    max_concurrent_sheets: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Reserved for Phase 3 scheduler — NOT YET ENFORCED. "
        "Will set the global limit on parallel sheet executions across all jobs. "
        "Setting a non-default value will log a warning at startup.",
    )
    resource_limits: ResourceLimitConfig = Field(
        default_factory=ResourceLimitConfig,
        description="Resource constraints for the daemon process",
    )
    state_backend_type: Literal["sqlite"] = Field(
        default="sqlite",
        description="Reserved — will enable persistent daemon state in a future "
        "release. Currently frozen to 'sqlite'; changing has no effect. "
        "Setting a non-default value will log a warning at startup.",
    )
    state_db_path: Path = Field(
        default=Path("~/.mozart/daemon-state.db"),
        description="Reserved for future use; not yet implemented. "
        "Will be the path for daemon state database. "
        "Tilde is expanded at runtime. Stores job registry and metrics.",
    )
    log_level: str = Field(
        default="info",
        description="Minimum log level for daemon structlog output. "
        "One of: debug, info, warning, error.",
    )
    log_file: Path | None = Field(
        default=None,
        description="Log file path. None means log to stderr only.",
    )
    job_timeout_seconds: float = Field(
        default=21600.0,
        ge=60.0,
        description="Maximum wall-clock time for a single job task. "
        "Jobs exceeding this limit are cancelled with FAILED status. "
        "Default is 6 hours (21600s). Set higher for known long-running jobs.",
    )
    shutdown_timeout_seconds: float = Field(
        default=300.0,
        ge=10.0,
        description="Max seconds to wait for running jobs during graceful shutdown",
    )
    monitor_interval_seconds: float = Field(
        default=15.0,
        ge=5.0,
        description="Interval between resource monitor checks",
    )
    max_job_history: int = Field(
        default=1000,
        ge=10,
        description="Maximum completed/failed/cancelled jobs to keep in memory. "
        "Oldest terminal jobs are evicted when the limit is reached.",
    )
    config_file: Path | None = Field(
        default=None,
        description="Reserved for future config reload support. "
        "Not yet functional — SIGHUP reload is not implemented.",
    )
