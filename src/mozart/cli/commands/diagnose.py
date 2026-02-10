"""Diagnostic commands for Mozart CLI.

This module implements commands for inspecting job state and debugging issues:
- `logs`: View and follow log files
- `errors`: List errors for a job with filtering
- `diagnose`: Generate comprehensive diagnostic reports

★ Insight ─────────────────────────────────────
1. **Layered debugging approach**: The three commands form a debugging hierarchy:
   `logs` for real-time streaming, `errors` for filtered error lists, and
   `diagnose` for comprehensive reports. Users typically progress through
   these as they narrow down issues.

2. **Synthetic error records**: When older state files lack error_history,
   the commands synthesize ErrorRecord objects from sheet-level error_message
   fields. This maintains backward compatibility with pre-history state files.

3. **Error type inference**: The `infer_error_type` function categorizes errors
   into permanent/transient/rate_limit based on error_category strings. This
   enables appropriate color-coding and helps users understand retry behavior.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import gzip
import json as json_module
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.table import Table

from mozart.core.checkpoint import (
    CheckpointState,
    ErrorRecord,
    SheetStatus,
)
from mozart.core.logging import find_log_files, get_default_log_path

from ..helpers import (
    configure_global_logging,
    is_quiet,
    require_job_state,
)
from ..output import (
    StatusColors,
    console,
    create_diagnostic_panel,
    create_errors_table,
    create_timeline_table,
    format_duration,
    format_error_details,
    infer_error_type,
)

# =============================================================================
# logs command
# =============================================================================


def logs(
    job_id: str | None = typer.Argument(
        None,
        help="Job ID to filter logs for (optional, shows all if not specified)",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to find logs (defaults to current directory)",
    ),
    log_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Specific log file path (overrides workspace default)",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-F",
        help="Follow the log file for new entries (like tail -f)",
    ),
    lines: int = typer.Option(
        50,
        "--lines",
        "-n",
        help="Number of lines to show (0 for all)",
    ),
    level: str | None = typer.Option(
        None,
        "--level",
        "-l",
        help="Filter by minimum log level (DEBUG, INFO, WARNING, ERROR)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output raw JSON log entries",
    ),
) -> None:
    """Show or tail log files for a job.

    Displays log entries from Mozart log files. Supports both current log files
    and compressed rotated logs (.gz).

    Examples:
        mozart logs                         # Show recent logs
        mozart logs my-job                  # Filter by job ID
        mozart logs --follow                # Follow log file (like tail -f)
        mozart logs --lines 100             # Show last 100 lines
        mozart logs --level ERROR           # Show only ERROR and above
        mozart logs --json                  # Output raw JSON entries
        mozart logs --workspace ./workspace # Use specific workspace

    Note:
        Log files are stored at {workspace}/logs/mozart.log by default.
        Use --file to specify a different log file path.
    """
    configure_global_logging(console)

    # Determine log file path
    ws = workspace or Path.cwd()
    target_log = log_file or get_default_log_path(ws)

    # Check if log file exists
    if not target_log.exists():
        # Try to find any log files in the workspace
        available_logs = find_log_files(ws, target_log)
        if not available_logs:
            console.print(f"[yellow]No log files found at:[/yellow] {target_log}")
            console.print(
                "\n[dim]Hint: Logs are created when running jobs with file logging enabled.\n"
                "Use --log-file or --log-format=both with mozart run to enable file logging.[/dim]"
            )
            raise typer.Exit(1)
        # Use the first available log
        target_log = available_logs[0]

    # Parse log level filter
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    min_level = 0
    if level:
        level_upper = level.upper()
        if level_upper not in level_order:
            console.print(
                f"[red]Invalid log level:[/red] {level}\n"
                "Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
            raise typer.Exit(1)
        min_level = level_order[level_upper]

    def parse_log_line(line: str) -> dict[str, Any] | None:
        """Parse a JSON log line, returning None if invalid."""
        line = line.strip()
        if not line:
            return None
        try:
            result: dict[str, Any] = json_module.loads(line)
            return result
        except json_module.JSONDecodeError:
            # Not a JSON line, return as plain text entry
            return {"event": line, "_raw": True}

    def should_include(entry: dict[str, Any]) -> bool:
        """Check if a log entry passes the filters."""
        # Filter by job_id if specified
        if job_id:
            entry_job_id = entry.get("job_id", "")
            if entry_job_id != job_id:
                return False

        # Filter by log level
        entry_level = entry.get("level", "INFO").upper()
        entry_level_num = level_order.get(entry_level, 1)
        return entry_level_num >= min_level

    def format_entry(entry: dict[str, Any]) -> str:
        """Format a log entry for display."""
        if json_output:
            return json_module.dumps(entry)

        # Raw/non-JSON line
        if entry.get("_raw"):
            return str(entry.get("event", ""))

        # Format structured log entry
        timestamp = entry.get("timestamp", "")
        level_str = entry.get("level", "INFO").upper()
        event = entry.get("event", "")
        component = entry.get("component", "")
        entry_job_id = entry.get("job_id", "")
        sheet_num = entry.get("sheet_num")

        # Color for level
        level_colors = {
            "DEBUG": "dim",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red bold",
        }
        level_color = level_colors.get(level_str, "white")

        # Build formatted line
        parts: list[str] = []
        if timestamp:
            # Shorten timestamp for display
            if "T" in timestamp:
                ts_short = timestamp.split("T")[1].split("+")[0].split(".")[0]
                parts.append(f"[dim]{ts_short}[/dim]")
            else:
                parts.append(f"[dim]{timestamp[:19]}[/dim]")

        parts.append(f"[{level_color}]{level_str:7}[/{level_color}]")

        if component:
            parts.append(f"[cyan]{component}[/cyan]")

        if entry_job_id:
            parts.append(f"[magenta]{entry_job_id}[/magenta]")

        if sheet_num is not None:
            parts.append(f"[green]sheet:{sheet_num}[/green]")

        parts.append(event)

        # Add extra context fields
        exclude_keys = {
            "timestamp", "level", "event", "component",
            "job_id", "sheet_num", "run_id", "parent_run_id", "_raw",
        }
        extras = {k: v for k, v in entry.items() if k not in exclude_keys}
        if extras:
            extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
            parts.append(f"[dim]{extras_str}[/dim]")

        return " ".join(parts)

    def read_log_lines(path: Path, num_lines: int | None = None) -> list[str]:
        """Read lines from a log file (handles .gz compression)."""
        is_gzip_file = path.suffix == ".gz"
        all_lines: list[str] = []

        try:
            if is_gzip_file:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    all_lines = f.readlines()
            else:
                with open(path, encoding="utf-8") as f:
                    all_lines = f.readlines()
        except OSError as e:
            console.print(f"[red]Error reading log file:[/red] {e}")
            return []

        if num_lines and num_lines > 0:
            return all_lines[-num_lines:]
        return all_lines

    def display_logs() -> None:
        """Display filtered log entries."""
        raw_lines = read_log_lines(target_log, lines if lines > 0 else None)

        if not raw_lines:
            console.print("[dim]No log entries found.[/dim]")
            return

        displayed = 0
        for line in raw_lines:
            entry = parse_log_line(line)
            if entry and should_include(entry):
                console.print(format_entry(entry))
                displayed += 1

        if displayed == 0:
            console.print("[dim]No log entries match the specified filters.[/dim]")
            if job_id:
                console.print(f"[dim]Job ID filter: {job_id}[/dim]")
            if level:
                console.print(f"[dim]Level filter: {level.upper()}+[/dim]")

    def follow_logs() -> None:
        """Follow log file for new entries (like tail -f)."""
        console.print(f"[dim]Following log file: {target_log}[/dim]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        file_handle = None
        try:
            # Open file and go to end
            file_handle = open(target_log, encoding="utf-8")  # noqa: SIM115
            file_handle.seek(0, 2)

            while True:
                line = file_handle.readline()
                if line:
                    entry = parse_log_line(line)
                    if entry and should_include(entry):
                        console.print(format_entry(entry))
                else:
                    # No new data, wait a bit
                    time.sleep(0.5)

                    # Check if file was rotated (inode changed or file deleted)
                    if not target_log.exists():
                        console.print(
                            "[yellow]Log file rotated. Waiting for new file...[/yellow]"
                        )
                        file_handle.close()

                        # Wait for new file to appear
                        for _ in range(10):
                            time.sleep(1)
                            if target_log.exists():
                                file_handle = open(target_log, encoding="utf-8")  # noqa: SIM115
                                break
                        else:
                            console.print(
                                "[yellow]Log file not recreated. Stopping.[/yellow]"
                            )
                            return
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped following logs.[/dim]")
        except OSError as e:
            console.print(f"[red]Error following log file:[/red] {e}")
            raise typer.Exit(1) from None
        finally:
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass

    # Show log file info
    if not is_quiet() and not json_output:
        console.print(f"[dim]Log file: {target_log}[/dim]")

    # Either follow or display
    if follow:
        follow_logs()
    else:
        display_logs()


# =============================================================================
# errors command
# =============================================================================


def errors(
    job_id: str = typer.Argument(..., help="Job ID to show errors for"),
    sheet: int | None = typer.Option(
        None,
        "--sheet",
        "-b",
        help="Filter errors by specific sheet number",
    ),
    error_type: str | None = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by error type: transient, rate_limit, or permanent",
    ),
    error_code: str | None = typer.Option(
        None,
        "--code",
        "-c",
        help="Filter by error code (e.g., E001, E101)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show full stdout/stderr tails for each error",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to search for job state",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output errors as JSON",
    ),
) -> None:
    """List all errors for a job with detailed information.

    Displays errors grouped by sheet, with color-coding by error type:
    - Red: Permanent errors (non-retriable, fatal)
    - Yellow: Transient errors (retriable with backoff)
    - Blue: Rate limit errors (retriable after wait)

    Examples:
        mozart errors my-job                   # Show all errors
        mozart errors my-job --sheet 3         # Errors for sheet 3 only
        mozart errors my-job --type transient  # Only transient errors
        mozart errors my-job --code E001       # Only timeout errors
        mozart errors my-job --verbose         # Show stdout/stderr details
    """
    asyncio.run(_errors_job(job_id, sheet, error_type, error_code, verbose, workspace, json_output))


async def _errors_job(
    job_id: str,
    sheet_filter: int | None,
    error_type_filter: str | None,
    error_code_filter: str | None,
    verbose: bool,
    workspace: Path | None,
    json_output: bool,
) -> None:
    """Asynchronously display errors for a job."""
    configure_global_logging(console)

    # Find job state
    found_job, _ = await require_job_state(job_id, workspace, json_output=json_output)

    # Collect all errors from sheet states
    all_errors: list[tuple[int, ErrorRecord]] = []

    for sheet_num, sheet_state in found_job.sheets.items():
        # Apply sheet filter if specified
        if sheet_filter is not None and sheet_num != sheet_filter:
            continue

        # Collect from error_history field
        for error in sheet_state.error_history:
            # Apply type filter
            if error_type_filter is not None and error.error_type != error_type_filter:
                continue
            # Apply code filter
            if error_code_filter is not None and error.error_code != error_code_filter:
                continue
            all_errors.append((sheet_num, error))

    # If no errors in history, check for error_message on failed sheets
    if not all_errors:
        for sheet_num, sheet_state in found_job.sheets.items():
            if sheet_filter is not None and sheet_num != sheet_filter:
                continue

            if sheet_state.error_message:
                # Create a synthetic ErrorRecord from sheet error_message
                # This handles older state files that don't have error_history populated
                synthetic_error = ErrorRecord(
                    error_type=infer_error_type(sheet_state.error_category),
                    error_code=sheet_state.error_category or "E999",
                    error_message=sheet_state.error_message,
                    attempt_number=sheet_state.attempt_count,
                    stdout_tail=sheet_state.stdout_tail,
                    stderr_tail=sheet_state.stderr_tail,
                    context={
                        "exit_code": sheet_state.exit_code,
                        "exit_signal": sheet_state.exit_signal,
                        "exit_reason": sheet_state.exit_reason,
                    },
                )
                # Apply filters
                type_mismatch = (
                    error_type_filter is not None
                    and synthetic_error.error_type != error_type_filter
                )
                code_mismatch = (
                    error_code_filter is not None
                    and synthetic_error.error_code != error_code_filter
                )
                if type_mismatch or code_mismatch:
                    continue
                all_errors.append((sheet_num, synthetic_error))

    # Sort by sheet number, then timestamp
    all_errors.sort(key=lambda x: (x[0], x[1].timestamp))

    # Output as JSON if requested
    if json_output:
        output: dict[str, Any] = {
            "job_id": job_id,
            "total_errors": len(all_errors),
            "errors": [
                {
                    "sheet_num": sheet_num,
                    "timestamp": error.timestamp.isoformat() if error.timestamp else None,
                    "error_type": error.error_type,
                    "error_code": error.error_code,
                    "error_message": error.error_message,
                    "attempt_number": error.attempt_number,
                    "context": error.context,
                    "stdout_tail": error.stdout_tail if verbose else None,
                    "stderr_tail": error.stderr_tail if verbose else None,
                }
                for sheet_num, error in all_errors
            ],
        }
        console.print(json_module.dumps(output, indent=2, default=str))
        return

    # Display with Rich table
    if not all_errors:
        console.print(f"[green]No errors found for job:[/green] {job_id}")
        if sheet_filter is not None:
            console.print(f"[dim]Sheet filter: {sheet_filter}[/dim]")
        if error_type_filter is not None:
            console.print(f"[dim]Type filter: {error_type_filter}[/dim]")
        if error_code_filter is not None:
            console.print(f"[dim]Code filter: {error_code_filter}[/dim]")
        return

    # Build errors table
    table = Table(title=f"Errors for Job: {job_id}")
    table.add_column("Sheet", justify="right", style="cyan", width=6)
    table.add_column("Time", style="dim", width=8)
    table.add_column("Type", width=10)
    table.add_column("Code", width=6)
    table.add_column("Attempt", justify="right", width=7)
    table.add_column("Message", style="white", no_wrap=False)

    for sheet_num, error in all_errors:
        # Format timestamp (just time, not date)
        time_str = ""
        if error.timestamp:
            time_str = error.timestamp.strftime("%H:%M:%S")

        # Format error type with color
        type_style = StatusColors.get_error_color(error.error_type)
        type_str = f"[{type_style}]{error.error_type}[/{type_style}]"

        # Truncate message for table
        message = error.error_message or ""
        if len(message) > 60 and not verbose:
            message = message[:57] + "..."

        table.add_row(
            str(sheet_num),
            time_str,
            type_str,
            error.error_code,
            str(error.attempt_number),
            message,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(all_errors)} error(s)[/dim]")

    # Show verbose details if requested
    if verbose:
        console.print("\n[bold]Error Details[/bold]")
        for sheet_num, error in all_errors:
            border_style = StatusColors.get_error_color(error.error_type)
            console.print(
                Panel(
                    format_error_details(error),
                    title=f"Sheet {sheet_num} - {error.error_code}",
                    border_style=border_style,
                )
            )


# =============================================================================
# diagnose command
# =============================================================================


def diagnose(
    job_id: str = typer.Argument(..., help="Job ID to diagnose"),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to search for job state",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output diagnostic report as JSON",
    ),
    include_logs: bool = typer.Option(
        False,
        "--include-logs",
        help="Inline the last 50 lines from each sheet/hook log file in the output",
    ),
) -> None:
    """Generate a comprehensive diagnostic report for a job.

    The diagnostic report includes:
    - Job overview and current status
    - Preflight warnings from all sheets
    - Prompt metrics (token counts, line counts)
    - Execution timeline with timing information
    - All errors with full context and output tails
    - Log file locations, sizes, and modification times
    - (with --include-logs) Inline log content from each log file

    This command is particularly useful for debugging failed jobs
    or understanding why a job is running slowly.

    Examples:
        mozart diagnose my-job                 # Full diagnostic report
        mozart diagnose my-job --json          # Machine-readable output
        mozart diagnose my-job --workspace .   # Specify workspace
        mozart diagnose my-job --include-logs  # Include inline log content
    """
    asyncio.run(_diagnose_job(job_id, workspace, json_output, include_logs=include_logs))


async def _diagnose_job(
    job_id: str,
    workspace: Path | None,
    json_output: bool,
    include_logs: bool = False,
) -> None:
    """Asynchronously generate diagnostic report for a job."""
    configure_global_logging(console)

    # Find job state
    found_job, found_backend = await require_job_state(job_id, workspace, json_output=json_output)

    # Build diagnostic report
    report: dict[str, Any] = _build_diagnostic_report(found_job, workspace=workspace)

    # Inline log content if requested
    if include_logs:
        _attach_log_contents(report)

    # Add execution history count if backend supports it
    if hasattr(found_backend, 'get_execution_history_count'):
        try:
            history_count = await found_backend.get_execution_history_count(  # type: ignore[attr-defined]
                job_id
            )
            report["execution_history_count"] = history_count
        except Exception:
            report["execution_history_count"] = None
    else:
        report["execution_history_count"] = None

    if json_output:
        # soft_wrap, highlight=False, markup=False prevent Rich from inserting
        # newlines or interpreting brackets in JSON output (e.g., log contents).
        console.print(
            json_module.dumps(report, indent=2, default=str),
            soft_wrap=True,
            highlight=False,
            markup=False,
        )
        return

    # Display formatted report
    _display_diagnostic_report(found_job, report)


def _discover_log_files(workspace: Path | None) -> list[dict[str, Any]]:
    """Scan workspace for log files and report metadata.

    Discovers sheet log files in {workspace}/logs/ and hook log files
    in {workspace}/hooks/, returning path, size, and modification time
    for each.

    Args:
        workspace: Workspace directory to scan. Returns empty list if None.

    Returns:
        List of dicts with path, size_bytes, modified_at, and category keys.
    """
    if workspace is None or not workspace.exists():
        return []

    discovered: list[dict[str, Any]] = []

    # Scan {workspace}/logs/ for sheet log files
    logs_dir = workspace / "logs"
    if logs_dir.is_dir():
        for log_file in sorted(logs_dir.iterdir()):
            if log_file.is_file() and log_file.suffix == ".log":
                stat = log_file.stat()
                discovered.append({
                    "path": str(log_file),
                    "name": log_file.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    "category": "sheet_log",
                })

    # Scan {workspace}/hooks/ for hook log files
    hooks_dir = workspace / "hooks"
    if hooks_dir.is_dir():
        for hook_file in sorted(hooks_dir.iterdir()):
            if hook_file.is_file() and hook_file.suffix == ".log":
                stat = hook_file.stat()
                discovered.append({
                    "path": str(hook_file),
                    "name": hook_file.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                    "category": "hook_log",
                })

    return discovered


# Default number of tail lines to inline from each log file
_INCLUDE_LOGS_TAIL_LINES = 50


def _attach_log_contents(report: dict[str, Any]) -> None:
    """Inline the last N lines of each discovered log file into the report.

    Mutates *report* in-place, adding a ``log_contents`` key that maps each
    log file path to its trailing lines.

    Args:
        report: Diagnostic report dict (must already contain ``log_files``).
    """
    log_contents: dict[str, str] = {}

    for log_info in report.get("log_files", []):
        log_path = Path(log_info["path"])
        if not log_path.exists():
            continue
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = lines[-_INCLUDE_LOGS_TAIL_LINES:]
            log_contents[str(log_path)] = "\n".join(tail)
        except OSError:
            log_contents[str(log_path)] = "<read error>"

    report["log_contents"] = log_contents


def _build_diagnostic_report(
    job: CheckpointState,
    *,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Build comprehensive diagnostic report from job state.

    Args:
        job: CheckpointState to analyze.
        workspace: Optional workspace directory for log file discovery.

    Returns:
        Dictionary with diagnostic information.
    """
    report: dict[str, Any] = {
        "job_id": job.job_id,
        "job_name": job.job_name,
        "status": job.status.value,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    # Progress summary
    completed_count = sum(
        1 for sheet in job.sheets.values() if sheet.status == SheetStatus.COMPLETED
    )
    failed_count = sum(
        1 for sheet in job.sheets.values() if sheet.status == SheetStatus.FAILED
    )
    report["progress"] = {
        "total_sheets": job.total_sheets,
        "completed": completed_count,
        "failed": failed_count,
        "last_completed": job.last_completed_sheet,
        "percent": (
            round(job.last_completed_sheet / job.total_sheets * 100, 1)
            if job.total_sheets > 0 else 0
        ),
    }

    # Timing - typed as Any to allow mixed types (str, None, float)
    timing_data: dict[str, Any] = {
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    # Calculate duration
    if job.started_at:
        end_time = job.completed_at or datetime.now(UTC)
        duration = (end_time - job.started_at).total_seconds()
        timing_data["duration_seconds"] = round(duration, 2)

    report["timing"] = timing_data

    # Collect preflight warnings across all sheets
    all_warnings: list[dict[str, Any]] = []
    for sheet_num, sheet in job.sheets.items():
        for warning in sheet.preflight_warnings:
            all_warnings.append({
                "sheet_num": sheet_num,
                "warning": warning,
            })
    report["preflight_warnings"] = all_warnings

    # Collect prompt metrics from all sheets
    prompt_metrics: list[dict[str, Any]] = []
    for sheet_num, sheet in job.sheets.items():
        if sheet.prompt_metrics:
            prompt_metrics.append({
                "sheet_num": sheet_num,
                **sheet.prompt_metrics,
            })
    report["prompt_metrics"] = prompt_metrics

    # Token statistics
    if prompt_metrics:
        tokens = [m.get("estimated_tokens", 0) for m in prompt_metrics]
        report["token_statistics"] = {
            "min": min(tokens),
            "max": max(tokens),
            "avg": round(sum(tokens) / len(tokens), 0),
            "total": sum(tokens),
        }

    # Execution timeline
    timeline: list[dict[str, Any]] = []
    for sheet_num in sorted(job.sheets.keys()):
        sheet = job.sheets[sheet_num]
        entry = {
            "sheet_num": sheet_num,
            "status": sheet.status.value,
            "started_at": sheet.started_at.isoformat() if sheet.started_at else None,
            "completed_at": sheet.completed_at.isoformat() if sheet.completed_at else None,
            "duration_seconds": sheet.execution_duration_seconds,
            "attempt_count": sheet.attempt_count,
            "completion_attempts": sheet.completion_attempts,
            "execution_mode": sheet.execution_mode,
            "outcome_category": sheet.outcome_category,
        }
        timeline.append(entry)
    report["execution_timeline"] = timeline

    # Execution statistics
    report["execution_stats"] = {
        "total_retry_count": job.total_retry_count,
        "rate_limit_waits": job.rate_limit_waits,
    }

    # All errors with full context
    all_errors: list[dict[str, Any]] = []
    for sheet_num, sheet in job.sheets.items():
        for error in sheet.error_history:
            all_errors.append({
                "sheet_num": sheet_num,
                "timestamp": error.timestamp.isoformat() if error.timestamp else None,
                "error_type": error.error_type,
                "error_code": error.error_code,
                "error_message": error.error_message,
                "attempt_number": error.attempt_number,
                "context": error.context,
                "stdout_tail": error.stdout_tail,
                "stderr_tail": error.stderr_tail,
                "stack_trace": error.stack_trace,
            })

        # Add sheet-level error if no history exists
        if not sheet.error_history and sheet.error_message:
            all_errors.append({
                "sheet_num": sheet_num,
                "timestamp": sheet.completed_at.isoformat() if sheet.completed_at else None,
                "error_type": infer_error_type(sheet.error_category),
                "error_code": sheet.error_category or "E999",
                "error_message": sheet.error_message,
                "attempt_number": sheet.attempt_count,
                "context": {
                    "exit_code": sheet.exit_code,
                    "exit_signal": sheet.exit_signal,
                    "exit_reason": sheet.exit_reason,
                },
                "stdout_tail": sheet.stdout_tail,
                "stderr_tail": sheet.stderr_tail,
            })

    report["errors"] = all_errors
    report["error_count"] = len(all_errors)

    # Job error message
    if job.error_message:
        report["job_error"] = job.error_message

    # Log file discovery
    log_files = _discover_log_files(workspace)
    report["log_files"] = log_files

    # Log rotation hint metadata
    hook_log_count = sum(1 for lf in log_files if lf.get("category") == "hook_log")
    if hook_log_count > 20:
        report["log_rotation_hint"] = (
            f"{hook_log_count} hook log files found — consider cleanup"
        )

    return report


def _display_diagnostic_report(job: CheckpointState, report: dict[str, Any]) -> None:
    """Display formatted diagnostic report.

    Args:
        job: Job state for additional context.
        report: Diagnostic report dictionary.
    """
    # Header panel
    console.print(create_diagnostic_panel(job.job_name, job.job_id, job.status))

    # Progress section
    progress = report.get("progress", {})
    console.print("\n[bold cyan]Progress[/bold cyan]")
    console.print(
        f"  Sheets: {progress.get('completed', 0)}/{progress.get('total_sheets', 0)} "
        f"completed ({progress.get('percent', 0):.1f}%)"
    )
    if progress.get("failed", 0) > 0:
        console.print(f"  Failed: [red]{progress.get('failed', 0)}[/red]")

    # Timing section
    timing = report.get("timing", {})
    if timing.get("duration_seconds"):
        console.print("\n[bold cyan]Timing[/bold cyan]")
        console.print(f"  Duration: {format_duration(timing['duration_seconds'])}")
        if timing.get("started_at"):
            console.print(f"  Started: {timing['started_at'][:19]}")
        if timing.get("completed_at"):
            console.print(f"  Completed: {timing['completed_at'][:19]}")

    # Preflight warnings
    warnings = report.get("preflight_warnings", [])
    if warnings:
        console.print(f"\n[bold yellow]Preflight Warnings ({len(warnings)})[/bold yellow]")
        for w in warnings[:20]:  # Limit display
            console.print(f"  [yellow]•[/yellow] Sheet {w['sheet_num']}: {w['warning']}")
        if len(warnings) > 20:
            console.print(f"  [dim]... and {len(warnings) - 20} more warnings[/dim]")

    # Token statistics
    token_stats = report.get("token_statistics")
    if token_stats:
        console.print("\n[bold cyan]Prompt Metrics[/bold cyan]")
        console.print(
            f"  Tokens: min={token_stats['min']:,}, max={token_stats['max']:,}, "
            f"avg={token_stats['avg']:,.0f}"
        )
        console.print(f"  Total tokens processed: {token_stats['total']:,}")

    # Execution timeline summary
    timeline = report.get("execution_timeline", [])
    if timeline:
        console.print("\n[bold cyan]Execution Timeline[/bold cyan]")
        timeline_table = create_timeline_table()

        for entry in timeline[:20]:  # Limit display
            status = entry.get("status", "unknown")
            status_style = StatusColors.TIMELINE_STATUS.get(status, "white")

            duration = entry.get("duration_seconds")
            duration_str = f"{duration:.1f}s" if duration else "-"

            attempts = entry.get("attempt_count", 0)
            comp_attempts = entry.get("completion_attempts", 0)
            attempts_str = str(attempts)
            if comp_attempts > 0:
                attempts_str += f"+{comp_attempts}c"

            timeline_table.add_row(
                str(entry.get("sheet_num", "")),
                f"[{status_style}]{status}[/{status_style}]",
                duration_str,
                attempts_str,
                entry.get("execution_mode") or "-",
                entry.get("outcome_category") or "-",
            )

        console.print(timeline_table)
        if len(timeline) > 20:
            console.print(f"[dim]... and {len(timeline) - 20} more sheets[/dim]")

    # Execution stats
    stats = report.get("execution_stats", {})
    history_count = report.get("execution_history_count")
    has_stats = (
        stats.get("total_retry_count", 0) > 0
        or stats.get("rate_limit_waits", 0) > 0
        or history_count is not None
    )
    if has_stats:
        console.print("\n[bold cyan]Execution Statistics[/bold cyan]")
        if history_count is not None:
            console.print(f"  Execution History: {history_count} recorded attempt(s)")
        if stats.get("total_retry_count", 0) > 0:
            console.print(f"  Total Retries: {stats['total_retry_count']}")
        if stats.get("rate_limit_waits", 0) > 0:
            console.print(f"  Rate Limit Waits: [yellow]{stats['rate_limit_waits']}[/yellow]")

    # Errors section
    errors_list = report.get("errors", [])
    if errors_list:
        console.print(f"\n[bold red]Errors ({len(errors_list)})[/bold red]")

        error_table = create_errors_table(title="")
        # Remove duplicate "Sheet" column header (already in table factory)
        # Just add rows directly
        for err in errors_list[:15]:  # Limit display
            err_type = err.get("error_type", "unknown")
            type_style = StatusColors.get_error_color(err_type)

            message = err.get("error_message", "")[:80]
            if len(err.get("error_message", "")) > 80:
                message += "..."

            error_table.add_row(
                str(err.get("sheet_num", "")),
                f"[{type_style}]{err_type}[/{type_style}]",
                err.get("error_code", ""),
                str(err.get("attempt_number", "")),
                message,
            )

        console.print(error_table)

        if len(errors_list) > 15:
            console.print(f"[dim]... and {len(errors_list) - 15} more errors[/dim]")
        console.print(
            f"\n[dim]Use 'mozart errors {job.job_id} --verbose' for full error details[/dim]"
        )

    # Job-level error
    if report.get("job_error"):
        console.print(f"\n[bold red]Job Error:[/bold red] {report['job_error']}")

    # Log files section
    log_files = report.get("log_files", [])
    if log_files:
        console.print(f"\n[bold cyan]Log Files ({len(log_files)})[/bold cyan]")
        log_table = Table()
        log_table.add_column("File", style="cyan", no_wrap=True)
        log_table.add_column("Size", justify="right", width=10)
        log_table.add_column("Modified", style="dim", width=19)
        log_table.add_column("Category", width=12)

        for lf in log_files:
            size = lf.get("size_bytes", 0)
            if size >= 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"

            modified = lf.get("modified_at", "")[:19] if lf.get("modified_at") else "-"

            log_table.add_row(
                lf.get("name", ""),
                size_str,
                modified,
                lf.get("category", ""),
            )

        console.print(log_table)

        # Log rotation hint when hook logs accumulate
        hook_log_count = sum(1 for lf in log_files if lf.get("category") == "hook_log")
        if hook_log_count > 20:
            console.print(
                f"\n[yellow]Hint:[/yellow] {hook_log_count} hook log files found. "
                "Consider cleaning up old logs to save disk space."
            )

        console.print(
            f"\n[dim]Use 'mozart diagnose {job.job_id} --include-logs' "
            "to inline log content[/dim]"
        )

    # Inline log contents (when --include-logs is used)
    log_contents = report.get("log_contents", {})
    if log_contents:
        tail = _INCLUDE_LOGS_TAIL_LINES
        console.print(f"\n[bold cyan]Log Contents (last {tail} lines each)[/bold cyan]")
        for log_path, content in log_contents.items():
            log_name = Path(log_path).name
            console.print(
                Panel(
                    content or "[dim]<empty>[/dim]",
                    title=log_name,
                    border_style="dim",
                )
            )


# =============================================================================
# history command
# =============================================================================


def history(
    job_id: str = typer.Argument(..., help="Job ID to show execution history for"),
    sheet: int | None = typer.Option(
        None,
        "--sheet",
        "-b",
        help="Filter by specific sheet number",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum number of records to show",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to search for job state",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output history as JSON",
    ),
) -> None:
    """Show execution history for a job.

    Displays a table of past execution attempts from the SQLite state backend,
    including sheet number, attempt number, exit code, duration, and timestamp.

    Requires the SQLite state backend (execution history is not available with
    the JSON backend).

    Examples:
        mozart history my-job                  # Show all history
        mozart history my-job --sheet 3        # History for sheet 3 only
        mozart history my-job --limit 100      # Show more records
        mozart history my-job --json           # Machine-readable output
    """
    asyncio.run(_history_job(job_id, sheet, limit, workspace, json_output))


async def _history_job(
    job_id: str,
    sheet_filter: int | None,
    limit: int,
    workspace: Path | None,
    json_output: bool,
) -> None:
    """Asynchronously display execution history for a job."""
    configure_global_logging(console)

    # Find job state and backend
    _, found_backend = await require_job_state(job_id, workspace, json_output=json_output)

    # Check if backend supports execution history
    if not hasattr(found_backend, 'get_execution_history'):
        if json_output:
            console.print(json_module.dumps({
                "error": "Execution history requires SQLite state backend",
                "hint": "Re-run the job with SQLite state to enable history tracking",
            }))
        else:
            console.print(
                "[yellow]Execution history requires the SQLite state backend.[/yellow]\n"
                "[dim]History is only recorded when using SQLite state storage.[/dim]"
            )
        raise typer.Exit(1)

    # Query execution history
    records = await found_backend.get_execution_history(  # type: ignore[attr-defined]
        job_id=job_id,
        sheet_num=sheet_filter,
        limit=limit,
    )

    # Output as JSON if requested
    if json_output:
        output: dict[str, Any] = {
            "job_id": job_id,
            "total_records": len(records),
            "records": records,
        }
        if sheet_filter is not None:
            output["sheet_filter"] = sheet_filter
        console.print(json_module.dumps(output, indent=2, default=str))
        return

    # Display with Rich table
    if not records:
        console.print(f"[dim]No execution history found for job:[/dim] {job_id}")
        if sheet_filter is not None:
            console.print(f"[dim]Sheet filter: {sheet_filter}[/dim]")
        console.print(
            "\n[dim]Hint: Execution history is recorded when the job runs "
            "with the SQLite state backend.[/dim]"
        )
        return

    table = Table(title=f"Execution History: {job_id}")
    table.add_column("Sheet", justify="right", style="cyan", width=6)
    table.add_column("Attempt", justify="right", width=7)
    table.add_column("Exit Code", justify="right", width=9)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Timestamp", style="dim", width=19)

    for record in records:
        exit_code = record.get("exit_code")
        exit_str = str(exit_code) if exit_code is not None else "-"
        exit_style = "green" if exit_code == 0 else "red" if exit_code is not None else "dim"

        duration = record.get("duration_seconds")
        duration_str = f"{duration:.1f}s" if duration is not None else "-"

        timestamp = record.get("executed_at", "")
        if timestamp and len(timestamp) > 19:
            timestamp = timestamp[:19]

        table.add_row(
            str(record.get("sheet_num", "")),
            str(record.get("attempt_num", "")),
            f"[{exit_style}]{exit_str}[/{exit_style}]",
            duration_str,
            timestamp,
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(records)} record(s)[/dim]")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "logs",
    "errors",
    "diagnose",
    "history",
]
