"""Shared utilities for Mozart CLI commands.

This module contains helpers used across multiple CLI command modules:
- Logger setup and configuration
- Backend creation helpers
- Config loading utilities
- Console/Rich setup
- State backend discovery
- Common utility functions

★ Insight ─────────────────────────────────────
1. **Module-level state pattern**: Global state (_output_level, _log_level, etc.)
   is used to share CLI configuration across commands without passing it through
   every function. This trades explicit dependencies for convenience.

2. **Factory pattern for backends**: `create_state_backend_from_config` and
   `create_notifiers_from_config` encapsulate the creation logic, making it
   easy to add new backend types without modifying command code.

3. **Error message constants**: Using a class (`ErrorMessages`) for error strings
   enables consistent user-facing messages and makes localization easier.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import typer
from rich.console import Console

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.core.config import JobConfig
from mozart.core.logging import configure_logging, get_logger
from mozart.state import JsonStateBackend, SQLiteStateBackend, StateBackend

if TYPE_CHECKING:
    from datetime import datetime


# =============================================================================
# Module-level logger
# =============================================================================

_logger = get_logger("cli")


# =============================================================================
# Error message constants
# =============================================================================


class ErrorMessages:
    """Constants for CLI error messages.

    Centralizing error messages ensures consistent user-facing output
    and simplifies potential localization.
    """

    JOB_NOT_FOUND = "Job not found"
    CONFIG_LOAD_ERROR = "Error loading config"
    WORKSPACE_NOT_FOUND = "Workspace not found"
    STATE_FILE_NOT_FOUND = "State file not found"


# =============================================================================
# Output level management
# =============================================================================


class OutputLevel(str, Enum):
    """Output verbosity level."""

    QUIET = "quiet"  # Minimal output (errors only)
    NORMAL = "normal"  # Default output
    VERBOSE = "verbose"  # Detailed output


# Global output level state
_output_level: OutputLevel = OutputLevel.NORMAL


def get_output_level() -> OutputLevel:
    """Get current output level."""
    return _output_level


def set_output_level(level: OutputLevel) -> None:
    """Set the output level.

    Args:
        level: The new output level.
    """
    global _output_level
    _output_level = level


def is_verbose() -> bool:
    """Check if verbose output is enabled."""
    return _output_level == OutputLevel.VERBOSE


def is_quiet() -> bool:
    """Check if quiet output is enabled."""
    return _output_level == OutputLevel.QUIET


# =============================================================================
# Logging configuration
# =============================================================================


@dataclass
class CliLoggingConfig:
    """Centralized CLI logging configuration state.

    Replaces 4 module-level variables with a single typed dataclass.
    All getter/setter functions below delegate to this instance.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
    file: Path | None = None
    format: Literal["json", "console", "both"] = "console"
    configured: bool = False


# Single global config instance
_log_config = CliLoggingConfig()


def get_log_level() -> str:
    """Get current log level."""
    return _log_config.level


def set_log_level(level: str) -> None:
    """Set the log level.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    _log_config.level = level  # type: ignore[assignment]


def get_log_file() -> Path | None:
    """Get current log file path."""
    return _log_config.file


def set_log_file(path: Path | None) -> None:
    """Set the log file path.

    When a log file is specified, logs are written to the file in
    console format. Rich CLI output (progress bars, status tables)
    is separate from structured logging and still displays on console.

    Args:
        path: Path for log file output, or None to disable file logging.
    """
    _log_config.file = path
    if path:
        _log_config.format = "console"


def get_log_format() -> str:
    """Get current log format."""
    return _log_config.format


def set_log_format(fmt: str) -> None:
    """Set the log format.

    Args:
        fmt: Log format string (json, console, both).
    """
    _log_config.format = fmt  # type: ignore[assignment]


def configure_global_logging(console: Console) -> None:
    """Configure logging based on global CLI options.

    This is called after all callbacks have processed their options.
    Only configures once per session.

    Args:
        console: Rich console for error output.

    Raises:
        typer.Exit: If logging configuration fails.
    """
    if _log_config.configured:
        return

    try:
        configure_logging(
            level=_log_config.level,
            format=_log_config.format,
            file_path=_log_config.file,
        )
        _log_config.configured = True
        # Note: Intentionally not logging here to avoid polluting --json output
        # If debugging is needed, use --log-file to redirect logs
    except ValueError as e:
        # Handle configuration errors (e.g., format="both" without file_path)
        console.print(f"[red]Logging configuration error:[/red] {e}")
        raise typer.Exit(1) from None


def reset_logging_state() -> None:
    """Reset logging state (primarily for testing).

    This resets the configured flag so logging can be
    reconfigured in tests.
    """
    _log_config.configured = False


# =============================================================================
# Default paths
# =============================================================================

# Default state directory when no config is available
DEFAULT_STATE_DIR = Path.home() / ".mozart" / "state"


# =============================================================================
# Backend creation helpers
# =============================================================================


def create_state_backend_from_config(config: JobConfig) -> StateBackend:
    """Create appropriate state backend based on configuration.

    Args:
        config: Job configuration specifying backend type and path.

    Returns:
        StateBackend instance (SQLite or JSON based on config).
    """
    state_path = config.get_state_path()

    if config.state_backend == "sqlite":
        return SQLiteStateBackend(state_path)
    else:
        # JSON backend uses directory, not file path
        return JsonStateBackend(state_path.parent if state_path.suffix else state_path)


def get_default_state_backend() -> StateBackend:
    """Get the default state backend for listing jobs without a config.

    Returns:
        SQLiteStateBackend pointing to the global Mozart state directory.
    """
    DEFAULT_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return SQLiteStateBackend(DEFAULT_STATE_DIR / "mozart.db")


# Re-exported from notifications module for backwards compatibility.
# The canonical implementation lives in mozart.notifications.factory.
from mozart.notifications.factory import (
    create_notifiers_from_config as create_notifiers_from_config,  # noqa: F401, E501
)

# =============================================================================
# State discovery helpers
# =============================================================================


def find_job_workspace(job_id: str, hint: Path | None = None) -> Path | None:
    """Find workspace containing a job's state file.

    Search order:
    1. Provided hint path
    2. Current directory
    3. Common workspace patterns (./workspace, ./{job_id}-workspace)

    Args:
        job_id: Job identifier to search for.
        hint: Optional hint path to check first.

    Returns:
        Path to workspace containing job state, or None if not found.
    """
    search_paths: list[Path] = []

    # Priority 1: Provided hint
    if hint:
        search_paths.append(hint)

    # Priority 2: Current directory
    cwd = Path.cwd()
    search_paths.append(cwd)

    # Priority 3: Common workspace patterns
    search_paths.append(cwd / "workspace")
    search_paths.append(cwd / f"{job_id}-workspace")

    for path in search_paths:
        if not path.exists():
            continue

        # Check for JSON state file
        json_state = path / f"{job_id}.json"
        if json_state.exists():
            return path

        # Check for SQLite backend
        sqlite_state = path / ".mozart-state.db"
        if sqlite_state.exists():
            # SQLite backend exists, check if it contains this job
            # We defer actual job lookup to the caller
            return path

    return None


async def find_job_state(
    job_id: str,
    workspace: Path | None,
) -> tuple[CheckpointState | None, StateBackend | None]:
    """Find job state in available backends.

    Args:
        job_id: Job ID to find.
        workspace: Optional workspace directory to search.

    Returns:
        Tuple of (CheckpointState, StateBackend) or (None, None) if not found.
    """
    backends: list[StateBackend] = []

    if workspace:
        if not workspace.exists():
            return None, None
        sqlite_path = workspace / ".mozart-state.db"
        if sqlite_path.exists():
            backends.append(SQLiteStateBackend(sqlite_path))
        backends.append(JsonStateBackend(workspace))
    else:
        cwd = Path.cwd()
        backends.append(JsonStateBackend(cwd))
        sqlite_cwd = cwd / ".mozart-state.db"
        if sqlite_cwd.exists():
            backends.append(SQLiteStateBackend(sqlite_cwd))

    for backend in backends:
        try:
            job = await backend.load(job_id)
            if job:
                return job, backend
        except Exception as e:
            _logger.warning(
                "error_querying_backend",
                job_id=job_id,
                backend=type(backend).__name__,
                error=str(e),
                exc_info=True,
            )
            continue

    return None, None


async def require_job_state(
    job_id: str,
    workspace: Path | None,
    *,
    json_output: bool = False,
) -> tuple[CheckpointState, StateBackend]:
    """Find job state or exit with a formatted error.

    Combines the repeated workspace-check → find-state → error-on-missing
    pattern used across CLI commands (status, resume, pause, diagnose, errors).

    Args:
        job_id: Job ID to find.
        workspace: Optional workspace directory to search.
        json_output: If True, format error as JSON; otherwise use Rich markup.

    Returns:
        Tuple of (CheckpointState, StateBackend).

    Raises:
        typer.Exit(1): If workspace doesn't exist or job not found.
    """
    from .output import output_error

    if workspace and not workspace.exists():
        output_error(
            f"{ErrorMessages.WORKSPACE_NOT_FOUND}: {workspace}",
            error_code="E501",
            hints=["Check the workspace path exists"],
            json_output=json_output,
        )
        raise typer.Exit(1)

    found_state, found_backend = await find_job_state(job_id, workspace)

    if found_state is None or found_backend is None:
        output_error(
            f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}",
            error_code="E501",
            hints=[
                "Use --workspace to specify the directory containing the job state",
                "Run 'mozart list' to see available jobs",
            ],
            json_output=json_output,
            job_id=job_id,
        )
        raise typer.Exit(1)

    return found_state, found_backend


def get_state_backends(workspace: Path | None) -> list[StateBackend]:
    """Get list of state backends to search.

    Args:
        workspace: Optional workspace directory to search.

    Returns:
        List of StateBackend instances to query.
    """
    backends: list[StateBackend] = []

    if workspace:
        if workspace.exists():
            sqlite_path = workspace / ".mozart-state.db"
            if sqlite_path.exists():
                backends.append(SQLiteStateBackend(sqlite_path))
            backends.append(JsonStateBackend(workspace))
    else:
        cwd = Path.cwd()
        backends.append(JsonStateBackend(cwd))
        sqlite_cwd = cwd / ".mozart-state.db"
        if sqlite_cwd.exists():
            backends.append(SQLiteStateBackend(sqlite_cwd))

    return backends


# =============================================================================
# Pause signal helpers
# =============================================================================


def create_pause_signal(workspace: Path, job_id: str) -> Path:
    """Create pause signal file for a job.

    Args:
        workspace: Workspace directory.
        job_id: Job identifier.

    Returns:
        Path to created signal file.

    Raises:
        PermissionError: If workspace is not writable.
        OSError: If signal file cannot be created.
    """
    signal_file = workspace / f".mozart-pause-{job_id}"
    signal_file.touch()
    return signal_file


async def wait_for_pause_ack(
    state_backend: StateBackend,
    job_id: str,
    timeout: int,
) -> bool:
    """Wait for job to acknowledge pause signal.

    Polls the state backend until the job status changes to PAUSED
    or the timeout is reached.

    Args:
        state_backend: State backend to poll.
        job_id: Job identifier.
        timeout: Maximum seconds to wait.

    Returns:
        True if job paused successfully, False if timeout reached.
    """
    start_time = asyncio.get_event_loop().time()
    poll_interval = 1.0  # Poll every second

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed >= timeout:
            return False

        try:
            state = await state_backend.load(job_id)
            if state and state.status == JobStatus.PAUSED:
                return True
            if state and state.status not in {JobStatus.RUNNING, JobStatus.PAUSED}:
                # Job changed to a non-running/paused state (completed, failed, etc.)
                # Consider this as "acknowledged" since it's no longer running
                return True
        except Exception:
            _logger.warning("Error polling pause state", exc_info=True)

        await asyncio.sleep(poll_interval)


# =============================================================================
# Last activity time
# =============================================================================


def get_last_activity_time(job: CheckpointState) -> datetime | None:
    """Get the most recent activity timestamp from the job.

    Checks sheet last_activity_at fields and updated_at.

    Args:
        job: CheckpointState to check.

    Returns:
        datetime of last activity, or None if not available.
    """
    from datetime import datetime as dt

    candidates: list[dt] = []

    # Check updated_at
    if job.updated_at:
        candidates.append(job.updated_at)

    # Check sheet-level last_activity_at
    for sheet in job.sheets.values():
        if sheet.last_activity_at:
            candidates.append(sheet.last_activity_at)

    if candidates:
        return max(candidates)
    return None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Error messages
    "ErrorMessages",
    # Output level
    "OutputLevel",
    "get_output_level",
    "set_output_level",
    "is_verbose",
    "is_quiet",
    # Logging
    "get_log_level",
    "set_log_level",
    "get_log_file",
    "set_log_file",
    "get_log_format",
    "set_log_format",
    "configure_global_logging",
    "reset_logging_state",
    # Paths
    "DEFAULT_STATE_DIR",
    # Backend creation
    "create_state_backend_from_config",
    "get_default_state_backend",
    "create_notifiers_from_config",
    # State discovery
    "find_job_workspace",
    "find_job_state",
    "require_job_state",
    "get_state_backends",
    # Pause signals
    "create_pause_signal",
    "wait_for_pause_ack",
    # Utilities
    "get_last_activity_time",
    # Logger
    "_logger",
]
