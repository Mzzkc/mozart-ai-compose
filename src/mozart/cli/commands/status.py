"""Status commands for Mozart CLI.

This module implements job status display commands:
- `mozart status <job-id>` - Show detailed status for a specific job
- `mozart list` (list_jobs) - List all jobs in the workspace

★ Insight ─────────────────────────────────────
1. **Dual output modes**: Both JSON and Rich output are supported for all
   status commands. JSON mode enables scripting and CI/CD integration while
   Rich mode provides a human-friendly interactive experience.

2. **Watch mode implementation**: The status --watch flag uses a polling loop
   with screen clearing. This is simpler than event-driven updates but works
   universally across state backends (JSON files or SQLite).

3. **Circuit breaker inference**: The actual CircuitBreaker is a runtime object
   not persisted to state. We infer its likely state from failure patterns in
   the persisted sheet states to give operators visibility into retry behavior.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
)

from mozart.core.checkpoint import ErrorRecord, JobStatus, SheetStatus
from mozart.state import JsonStateBackend, SQLiteStateBackend, StateBackend

from ..helpers import ErrorMessages
from ..output import (
    StatusColors,
    console,
    format_duration,
    format_timestamp,
    format_validation_status,
    create_jobs_table,
    create_sheet_details_table,
    create_synthesis_table,
)

if TYPE_CHECKING:
    from mozart.core.checkpoint import CheckpointState

_logger = logging.getLogger(__name__)

# =============================================================================
# CLI Commands
# =============================================================================


def status(
    job_id: str = typer.Argument(
        ...,
        help="Job ID to check status for",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output status as JSON for machine parsing",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        "-W",
        help="Continuously monitor status with live updates",
    ),
    watch_interval: int = typer.Option(
        5,
        "--interval",
        "-i",
        help="Refresh interval in seconds for --watch mode (default: 5)",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to search for job state",
    ),
) -> None:
    """Show detailed status of a specific job.

    Displays job progress, sheet states, timing information, and any errors.
    Use --json for machine-readable output in scripts.
    Use --watch for continuous monitoring (updates every 5 seconds by default).

    Examples:
        mozart status my-job
        mozart status my-job --json
        mozart status my-job --watch
        mozart status my-job --watch --interval 10
    """
    if watch:
        asyncio.run(_status_job_watch(job_id, json_output, watch_interval, workspace))
    else:
        asyncio.run(_status_job(job_id, json_output, workspace))


def list_jobs(
    status_filter: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by job status (pending, running, completed, failed, paused)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of jobs to display",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to scan for jobs (scans JSON state files)",
    ),
) -> None:
    """List all jobs.

    By default, searches for job state files in the workspace directories.
    Use --workspace to specify a particular directory to scan.
    """
    asyncio.run(_list_jobs(status_filter, limit, workspace))


# =============================================================================
# Async Implementation Functions
# =============================================================================


async def _status_job(
    job_id: str,
    json_output: bool,
    workspace: Path | None,
) -> None:
    """Asynchronously get and display status for a specific job."""
    # Determine which backends to query
    backends: list[StateBackend] = []

    if workspace:
        if not workspace.exists():
            console.print(f"[red]Workspace not found:[/red] {workspace}")
            raise typer.Exit(1)

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

    # Search for the job in all backends
    found_job: CheckpointState | None = None

    for backend in backends:
        try:
            job = await backend.load(job_id)
            if job:
                found_job = job
                break
        except Exception as e:
            _logger.debug("Error querying backend for %s: %s", job_id, e)
            continue

    if not found_job:
        if json_output:
            err_msg = f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}"
            console.print(json.dumps({"error": err_msg}, indent=2))
        else:
            console.print(f"[red]{ErrorMessages.JOB_NOT_FOUND}:[/red] {job_id}")
            console.print(
                "\n[dim]Hint: Use --workspace to specify the directory "
                "containing the job state.[/dim]"
            )
        raise typer.Exit(1)

    # Output as JSON if requested
    if json_output:
        _output_status_json(found_job)
        return

    # Display rich status output
    _output_status_rich(found_job)


async def _status_job_watch(
    job_id: str,
    json_output: bool,
    interval: int,
    workspace: Path | None,
) -> None:
    """Continuously monitor job status with live updates.

    Args:
        job_id: Job ID to monitor.
        json_output: Output as JSON instead of rich formatting.
        interval: Refresh interval in seconds.
        workspace: Optional workspace directory to search.
    """
    console.print(f"[dim]Watching job [bold]{job_id}[/bold] (Ctrl+C to stop)[/dim]\n")

    try:
        while True:
            # Find and load job state
            backends: list[StateBackend] = []

            if workspace:
                if not workspace.exists():
                    console.print(f"[red]Workspace not found:[/red] {workspace}")
                    raise typer.Exit(1)

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

            found_job: CheckpointState | None = None
            for backend in backends:
                try:
                    job = await backend.load(job_id)
                    if job:
                        found_job = job
                        break
                except Exception:
                    continue

            # Clear screen and show status
            console.clear()

            if not found_job:
                if json_output:
                    err_msg = f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}"
                    console.print(json.dumps({"error": err_msg}, indent=2))
                else:
                    console.print(f"[red]{ErrorMessages.JOB_NOT_FOUND}:[/red] {job_id}")
                    console.print(
                        "\n[dim]Hint: Use --workspace to specify the directory "
                        "containing the job state.[/dim]"
                    )
            else:
                if json_output:
                    _output_status_json(found_job)
                else:
                    _output_status_rich(found_job)

                # Show watch mode indicator
                now = datetime.now(UTC)
                console.print(
                    f"\n[dim]Last updated: {now.strftime('%H:%M:%S')} "
                    f"| Refreshing every {interval}s | Press Ctrl+C to stop[/dim]"
                )

                # Exit watch mode if job is completed or failed
                if found_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    console.print(
                        f"\n[yellow]Job {found_job.status.value} - exiting watch mode[/yellow]"
                    )
                    break

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[dim]Watch mode stopped[/dim]")


async def _list_jobs(
    status_filter: str | None,
    limit: int,
    workspace: Path | None,
) -> None:
    """Asynchronously list jobs from state backends."""
    # Determine which backends to query
    backends: list[tuple[str, StateBackend]] = []

    if workspace:
        if not workspace.exists():
            console.print(f"[red]Workspace not found:[/red] {workspace}")
            raise typer.Exit(1)

        sqlite_path = workspace / ".mozart-state.db"
        if sqlite_path.exists():
            backends.append((str(workspace), SQLiteStateBackend(sqlite_path)))

        json_backend = JsonStateBackend(workspace)
        backends.append((str(workspace), json_backend))
    else:
        cwd = Path.cwd()
        json_backend = JsonStateBackend(cwd)
        backends.append((".", json_backend))

        sqlite_cwd = cwd / ".mozart-state.db"
        if sqlite_cwd.exists():
            backends.append((".", SQLiteStateBackend(sqlite_cwd)))


    # Collect all jobs
    all_jobs: list[tuple[str, CheckpointState]] = []

    for source, backend in backends:
        try:
            jobs = await backend.list_jobs()
            for job in jobs:
                all_jobs.append((source, job))
        except Exception:
            continue

    # Remove duplicates (same job_id from different backends)
    seen_ids: set[str] = set()
    unique_jobs: list[tuple[str, CheckpointState]] = []
    for source, job in all_jobs:
        if job.job_id not in seen_ids:
            seen_ids.add(job.job_id)
            unique_jobs.append((source, job))

    # Filter by status if specified
    if status_filter:
        try:
            target_status = JobStatus(status_filter.lower())
            unique_jobs = [
                (s, j) for s, j in unique_jobs if j.status == target_status
            ]
        except ValueError:
            console.print(
                f"[red]Invalid status:[/red] {status_filter}\n"
                f"Valid values: pending, running, completed, failed, paused"
            )
            raise typer.Exit(1) from None

    # Sort by updated_at descending and limit
    unique_jobs.sort(key=lambda x: x[1].updated_at or datetime.min.replace(tzinfo=UTC), reverse=True)
    unique_jobs = unique_jobs[:limit]

    if not unique_jobs:
        console.print("[dim]No jobs found.[/dim]")
        if status_filter:
            console.print("[dim]Try without --status filter or check a different workspace.[/dim]")
        return

    # Build table
    table = create_jobs_table()

    for _source, job in unique_jobs:
        # Format status with color
        status_color = StatusColors.get_job_color(job.status)
        status_str = f"[{status_color}]{job.status.value}[/{status_color}]"

        # Format progress
        progress = f"{job.last_completed_sheet}/{job.total_sheets}"

        # Format updated time
        updated = job.updated_at.strftime("%Y-%m-%d %H:%M") if job.updated_at else "-"

        table.add_row(job.job_id, status_str, progress, updated)

    console.print(table)
    console.print(f"\n[dim]Showing {len(unique_jobs)} job(s)[/dim]")


# =============================================================================
# Output Formatters
# =============================================================================


def _output_status_json(job: CheckpointState) -> None:
    """Output job status as JSON."""
    # Build a clean JSON representation
    # Use last_completed_sheet for progress since it's more reliable than counting sheets dict
    completed = job.last_completed_sheet
    total = job.total_sheets
    percent = (completed / total * 100) if total > 0 else 0.0

    # Collect recent errors for JSON output
    recent_errors_data: list[dict[str, Any]] = []
    for sheet_num, error in _collect_recent_errors(job, limit=5):
        recent_errors_data.append({
            "sheet_num": sheet_num,
            "timestamp": error.timestamp.isoformat() if error.timestamp else None,
            "error_type": error.error_type,
            "error_code": error.error_code,
            "error_message": error.error_message,
        })

    # Get last activity time
    last_activity = _get_last_activity_time(job)

    # Infer circuit breaker state
    cb_state = _infer_circuit_breaker_state(job)

    output = {
        "job_id": job.job_id,
        "job_name": job.job_name,
        "status": job.status.value,
        "progress": {
            "completed": completed,
            "total": total,
            "percent": round(percent, 1),
        },
        "timing": {
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "updated_at": job.updated_at.isoformat() if job.updated_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "last_activity": last_activity.isoformat() if last_activity else None,
        },
        "execution": {
            "current_sheet": job.current_sheet,
            "total_retry_count": job.total_retry_count,
            "rate_limit_waits": job.rate_limit_waits,
        },
        "circuit_breaker": cb_state,
        "recent_errors": recent_errors_data,
        "error": job.error_message,
        "sheets": {
            str(num): {
                "status": sheet.status.value,
                "attempt_count": sheet.attempt_count,
                "validation_passed": sheet.validation_passed,
                "error_message": sheet.error_message,
                "error_category": sheet.error_category,
            }
            for num, sheet in job.sheets.items()
        },
    }
    console.print(json.dumps(output, indent=2))


def _output_status_rich(job: CheckpointState) -> None:
    """Output job status with rich formatting."""
    # Status color mapping
    status_color = StatusColors.get_job_color(job.status)

    # Build header content
    header_lines = [
        f"[bold]{job.job_name}[/bold]",
        f"ID: [cyan]{job.job_id}[/cyan]",
        f"Status: [{status_color}]{job.status.value.upper()}[/{status_color}]",
    ]

    # Calculate duration
    if job.started_at:
        if job.completed_at:
            duration = job.completed_at - job.started_at
            duration_str = format_duration(duration.total_seconds())
            header_lines.append(f"Duration: {duration_str}")
        elif job.status == JobStatus.RUNNING and job.updated_at:
            elapsed = datetime.now(UTC) - job.started_at
            elapsed_str = format_duration(elapsed.total_seconds())
            header_lines.append(f"Running for: {elapsed_str}")

    console.print(Panel("\n".join(header_lines), title="Job Status"))

    # Progress bar - use last_completed_sheet for consistency with JSON output
    completed = job.last_completed_sheet
    total = job.total_sheets

    console.print("\n[bold]Progress[/bold]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        transient=False,
    ) as progress:
        progress.add_task("Sheets", total=total, completed=completed)
        progress.refresh()

    # Parallel execution info (v17 evolution)
    if job.parallel_enabled:
        console.print("\n[bold]Parallel Execution[/bold]")
        console.print("  Mode: [cyan]Enabled[/cyan]")
        console.print(f"  Max concurrent: {job.parallel_max_concurrent}")
        console.print(f"  Batches executed: {job.parallel_batches_executed}")
        if job.sheets_in_progress:
            in_progress_str = ", ".join(str(s) for s in job.sheets_in_progress)
            console.print(f"  Currently running: [blue]{in_progress_str}[/blue]")

    # Synthesis results (v18 evolution: Result Synthesizer Pattern)
    if job.synthesis_results:
        console.print("\n[bold]Synthesis Results[/bold]")
        synth_table = create_synthesis_table()

        for batch_id, result in job.synthesis_results.items():
            sheets = result.get("sheets", [])
            sheets_str = ", ".join(str(s) for s in sheets[:4])
            if len(sheets) > 4:
                sheets_str += f" (+{len(sheets) - 4})"

            strategy = result.get("strategy", "merge")
            synth_status = result.get("status", "pending")
            synth_status_color = StatusColors.SYNTHESIS_STATUS.get(synth_status, "white")

            synth_table.add_row(
                batch_id[:12],
                sheets_str,
                strategy,
                f"[{synth_status_color}]{synth_status}[/{synth_status_color}]",
            )

        console.print(synth_table)

    # Sheet details table
    if job.sheets:
        console.print("\n[bold]Sheet Details[/bold]")
        sheet_table = create_sheet_details_table()

        for sheet_num in sorted(job.sheets.keys()):
            sheet = job.sheets[sheet_num]
            sheet_color = StatusColors.get_sheet_color(sheet.status)

            # Format validation status
            val_str = format_validation_status(sheet.validation_passed)

            # Truncate error message for table
            error_str = ""
            if sheet.error_message:
                error_str = sheet.error_message[:50]
                if len(sheet.error_message) > 50:
                    error_str += "..."

            sheet_table.add_row(
                str(sheet_num),
                f"[{sheet_color}]{sheet.status.value}[/{sheet_color}]",
                str(sheet.attempt_count),
                val_str,
                error_str,
            )

        console.print(sheet_table)

    # Show error/info message
    if job.error_message:
        # Distinguish between actual errors and informational recovery messages
        if "Recovered from stale" in job.error_message or "ready to resume" in job.error_message:
            console.print(f"\n[bold cyan]Note:[/bold cyan] {job.error_message}")
        else:
            console.print(f"\n[bold red]Error:[/bold red] {job.error_message}")

    # Timing info
    console.print("\n[bold]Timing[/bold]")
    if job.created_at:
        console.print(f"  Created:  {format_timestamp(job.created_at)}")
    if job.started_at:
        console.print(f"  Started:  {format_timestamp(job.started_at)}")
    if job.updated_at:
        console.print(f"  Updated:  {format_timestamp(job.updated_at)}")
    if job.completed_at:
        console.print(f"  Completed: {format_timestamp(job.completed_at)}")

    # Execution stats
    quota_waits = getattr(job, "quota_waits", 0)
    if job.total_retry_count > 0 or job.rate_limit_waits > 0 or quota_waits > 0:
        console.print("\n[bold]Execution Stats[/bold]")
        console.print(f"  Total retries: {job.total_retry_count}")
        console.print(f"  Rate limit waits: {job.rate_limit_waits}")
        if quota_waits > 0:
            console.print(f"  Quota exhaustion waits: {quota_waits}")

    # Recent errors section - show last 3 errors from any sheet
    recent_errors = _collect_recent_errors(job, limit=3)
    if recent_errors:
        console.print("\n[bold red]Recent Errors[/bold red]")
        for sheet_num, error in recent_errors:
            # Format with color based on error type
            type_style = StatusColors.get_error_color(error.error_type)

            # Truncate message
            message = error.error_message or ""
            if len(message) > 60:
                message = message[:57] + "..."

            console.print(
                f"  [{type_style}]\u2022[/{type_style}] Sheet {sheet_num}: "
                f"[{type_style}]{error.error_code}[/{type_style}] - {message}"
            )

        # Hint for more details
        console.print(
            f"\n[dim]  Use 'mozart errors {job.job_id}' for complete error history[/dim]"
        )

    # Last activity timestamp
    last_activity = _get_last_activity_time(job)
    if last_activity:
        console.print("\n[bold]Last Activity[/bold]")
        console.print(f"  {format_timestamp(last_activity)}")

    # Circuit breaker state indicator
    cb_state = _infer_circuit_breaker_state(job)
    if cb_state:
        cb_color = {"open": "red", "half_open": "yellow", "closed": "green"}.get(
            cb_state["state"], "white"
        )
        console.print("\n[bold]Circuit Breaker (inferred)[/bold]")
        console.print(f"  State: [{cb_color}]{cb_state['state'].upper()}[/{cb_color}]")
        console.print(f"  Consecutive failures: {cb_state['consecutive_failures']}")
        if cb_state.get("reason"):
            console.print(f"  [dim]{cb_state['reason']}[/dim]")


# =============================================================================
# Helper Functions
# =============================================================================


def _collect_recent_errors(
    job: CheckpointState,
    limit: int = 3,
) -> list[tuple[int, ErrorRecord]]:
    """Collect the most recent errors from sheet states.

    Args:
        job: CheckpointState to collect errors from.
        limit: Maximum number of errors to return.

    Returns:
        List of (sheet_num, ErrorRecord) tuples, sorted by timestamp descending.
    """
    all_errors: list[tuple[int, ErrorRecord]] = []

    for sheet_num, sheet in job.sheets.items():
        # Collect from error_history
        for error in sheet.error_history:
            all_errors.append((sheet_num, error))

        # If no history but has error_message, create synthetic record
        if not sheet.error_history and sheet.error_message:
            synthetic = ErrorRecord(
                error_type=_infer_error_type(sheet.error_category),
                error_code=sheet.error_category or "E999",
                error_message=sheet.error_message,
                attempt_number=sheet.attempt_count,
                context={
                    "exit_code": sheet.exit_code,
                    "exit_signal": sheet.exit_signal,
                },
            )
            if sheet.completed_at:
                synthetic.timestamp = sheet.completed_at
            all_errors.append((sheet_num, synthetic))

    # Sort by timestamp (most recent first) and take limit
    all_errors.sort(key=lambda x: x[1].timestamp, reverse=True)
    return all_errors[:limit]


def _get_last_activity_time(job: CheckpointState) -> datetime | None:
    """Get the most recent activity timestamp from the job.

    Checks sheet last_activity_at fields and updated_at.

    Args:
        job: CheckpointState to check.

    Returns:
        datetime of last activity, or None if not available.
    """
    candidates: list[datetime] = []

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


def _infer_error_type(
    error_category: str | None,
) -> Literal["transient", "rate_limit", "permanent"]:
    """Infer error type from error category string.

    Args:
        error_category: Error category from sheet state.

    Returns:
        Error type literal: transient, rate_limit, or permanent.
    """
    if error_category is None:
        return "permanent"

    category_lower = error_category.lower()
    if "rate" in category_lower or "limit" in category_lower:
        return "rate_limit"
    if category_lower in ("transient", "timeout", "network", "signal"):
        return "transient"
    return "permanent"


def _infer_circuit_breaker_state(job: CheckpointState) -> dict[str, Any] | None:
    """Infer likely circuit breaker state from job state.

    The actual CircuitBreaker is a runtime object and not persisted.
    We can infer the likely state based on failure patterns:
    - If last N sheets all failed -> likely OPEN
    - If mix of success/failure -> likely CLOSED
    - If recovering from failures -> likely HALF_OPEN

    Args:
        job: CheckpointState to analyze.

    Returns:
        Dict with inferred state info, or None if no relevant data.
    """
    if not job.sheets:
        return None

    # Count consecutive failures from the end
    sorted_sheets = sorted(job.sheets.items(), key=lambda x: x[0], reverse=True)
    consecutive_failures = 0

    for _sheet_num, sheet in sorted_sheets:
        if sheet.status == SheetStatus.FAILED:
            consecutive_failures += 1
        elif sheet.status == SheetStatus.COMPLETED:
            break
        # PENDING/IN_PROGRESS don't count

    if consecutive_failures == 0:
        return None  # No failures, circuit likely closed, nothing special to show

    # Default threshold is 5 (from CircuitBreaker)
    threshold = 5

    if consecutive_failures >= threshold:
        return {
            "state": "open",
            "consecutive_failures": consecutive_failures,
            "reason": f"\u2265{threshold} consecutive failures detected",
        }
    elif consecutive_failures > 0:
        return {
            "state": "closed",
            "consecutive_failures": consecutive_failures,
            "reason": f"Under threshold ({consecutive_failures}/{threshold})",
        }

    return None


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "status",
    "list_jobs",
]
