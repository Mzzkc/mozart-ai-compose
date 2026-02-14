"""Rich output formatting for Mozart CLI.

This module centralizes all Rich-based formatting utilities for the CLI:
- Color schemes for status values
- Table builders with consistent styling
- Progress bar configurations
- Panel formatting helpers
- Status and duration formatters

★ Insight ─────────────────────────────────────
1. **Color scheme consistency**: Centralizing color mappings in one module
   ensures visual consistency across all CLI commands. Users associate
   colors with meanings (green=success, red=error), so consistency builds
   intuitive understanding.

2. **Table factory pattern**: Instead of duplicating Table() configurations
   across commands, factory functions encapsulate styling decisions. This
   makes it easy to adjust all tables' appearance in one place.

3. **Progress bar customization**: Rich's Progress supports extensive
   customization via column composition. We define reusable configurations
   for different progress display needs (simple vs detailed with ETA).
─────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from mozart.core.checkpoint import JobStatus, SheetStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mozart.core.checkpoint import CheckpointErrorRecord, CheckpointState
    from mozart.execution.runner.models import RunSummary

# =============================================================================
# Shared console instance
# =============================================================================

# NOTE: Command modules should use this console or accept it as a parameter.
# Quiet/JSON modes are handled by the is_quiet()/is_json() guards in each
# command, NOT by this Console instance itself.
console = Console()


# =============================================================================
# Color schemes for status values
# =============================================================================


class StatusColors:
    """Color mappings for various status types.

    Using a class with attributes makes it easy to see all available
    color mappings and maintain consistency across the codebase.
    """

    # Job-level status colors
    JOB_STATUS: dict[JobStatus, str] = {
        JobStatus.PENDING: "yellow",
        JobStatus.RUNNING: "blue",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.PAUSED: "magenta",
        JobStatus.CANCELLED: "dim",
    }

    # Sheet-level status colors
    SHEET_STATUS: dict[SheetStatus, str] = {
        SheetStatus.PENDING: "yellow",
        SheetStatus.IN_PROGRESS: "blue",
        SheetStatus.COMPLETED: "green",
        SheetStatus.FAILED: "red",
        SheetStatus.SKIPPED: "dim",
    }

    # Synthesis result status colors (v18 evolution)
    SYNTHESIS_STATUS: dict[str, str] = {
        "pending": "yellow",
        "ready": "blue",
        "done": "green",
        "failed": "red",
    }

    # Error type colors
    ERROR_TYPE: dict[str, str] = {
        "permanent": "red",
        "transient": "yellow",
        "rate_limit": "blue",
    }

    # Timeline status colors (string-based, for diagnostic reports)
    TIMELINE_STATUS: dict[str, str] = {
        "pending": "yellow",
        "in_progress": "blue",
        "completed": "green",
        "failed": "red",
        "skipped": "dim",
    }

    @classmethod
    def get_job_color(cls, status: JobStatus) -> str:
        """Get color for a job status value."""
        return cls.JOB_STATUS.get(status, "white")

    @classmethod
    def get_sheet_color(cls, status: SheetStatus) -> str:
        """Get color for a sheet status value."""
        return cls.SHEET_STATUS.get(status, "white")

    @classmethod
    def get_error_color(cls, error_type: str) -> str:
        """Get color for an error type."""
        return cls.ERROR_TYPE.get(error_type, "white")


# =============================================================================
# Duration formatting
# =============================================================================


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds, or None.

    Returns:
        Human-readable duration string (e.g., "5.2s", "3m 12s", "1h 30m").
    """
    if seconds is None:
        return "N/A"

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


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Human-readable size string (e.g., "128B", "1.5KB", "2.3MB").
    """
    if num_bytes < 1024:
        return f"{num_bytes}B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f}KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f}MB"


def format_timestamp(dt: datetime | None, include_tz: bool = True) -> str:
    """Format a datetime for display.

    Args:
        dt: datetime to format, or None.
        include_tz: Whether to include timezone suffix.

    Returns:
        Formatted timestamp string, or "-" if None.
    """
    if dt is None:
        return "-"

    fmt = "%Y-%m-%d %H:%M:%S"
    if include_tz:
        fmt += " UTC"
    return dt.strftime(fmt)


# =============================================================================
# Validation status formatting
# =============================================================================


def format_validation_status(passed: bool | None) -> str:
    """Format validation status with appropriate styling.

    Args:
        passed: True if passed, False if failed, None if not run.

    Returns:
        Rich-formatted validation status string.
    """
    if passed is None:
        return "-"
    elif passed:
        return "[green]\u2713 Pass[/green]"
    else:
        return "[red]\u2717 Fail[/red]"


# =============================================================================
# Error type inference
# =============================================================================


def infer_error_type(
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


# =============================================================================
# Table builders
# =============================================================================


def create_jobs_table() -> Table:
    """Create a styled table for job listings.

    Returns:
        Rich Table configured for job list display.
    """
    table = Table(title="Mozart Jobs")
    table.add_column("Job ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Workspace", style="dim", no_wrap=True)
    table.add_column("Submitted", style="dim")
    return table


def create_sheet_plan_table() -> Table:
    """Create a styled table for dry-run sheet plan display.

    Returns:
        Rich Table configured for sheet plan display.
    """
    table = Table(title="Sheet Plan")
    table.add_column("Sheet", style="cyan")
    table.add_column("Items", style="green")
    table.add_column("Validations", style="yellow")
    return table


def create_sheet_details_table() -> Table:
    """Create a styled table for detailed sheet status.

    Returns:
        Rich Table configured for sheet details display.
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", style="cyan", width=4)
    table.add_column("Status", width=12)
    table.add_column("Attempts", justify="right", width=8)
    table.add_column("Validation", width=10)
    table.add_column("Error", style="dim", no_wrap=False)
    return table


def create_synthesis_table() -> Table:
    """Create a styled table for synthesis results (v18 evolution).

    Returns:
        Rich Table configured for synthesis result display.
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Batch ID", style="cyan", width=12)
    table.add_column("Sheets", width=15)
    table.add_column("Strategy", width=12)
    table.add_column("Status", width=10)
    return table


def create_errors_table(title: str = "Errors") -> Table:
    """Create a styled table for error display.

    Args:
        title: Optional title for the table.

    Returns:
        Rich Table configured for error display.
    """
    table = Table(title=title if title else None, show_header=True, header_style="bold")
    table.add_column("Sheet", justify="right", style="cyan", width=5)
    table.add_column("Type", width=10)
    table.add_column("Code", width=6)
    table.add_column("Attempt", justify="right", width=7)
    table.add_column("Message", no_wrap=False)
    return table


def create_timeline_table() -> Table:
    """Create a styled table for execution timeline.

    Returns:
        Rich Table configured for timeline display.
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=4)
    table.add_column("Status", width=12)
    table.add_column("Duration", justify="right", width=10)
    table.add_column("Attempts", justify="right", width=8)
    table.add_column("Mode", width=12)
    table.add_column("Outcome", width=18)
    return table


def create_patterns_table(title: str = "Learned Patterns") -> Table:
    """Create a styled table for learning pattern display.

    Args:
        title: Title for the patterns table.

    Returns:
        Rich Table configured for pattern display.
    """
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("Pattern", style="cyan", no_wrap=False)
    table.add_column("Confidence", justify="right", width=12)
    table.add_column("Applied", justify="right", width=8)
    table.add_column("Success Rate", justify="right", width=12)
    return table


def create_simple_table(show_header: bool = False) -> Table:
    """Create a simple table without box styling.

    Useful for key-value displays or compact listings.

    Args:
        show_header: Whether to show column headers.

    Returns:
        Rich Table with minimal styling.
    """
    return Table(show_header=show_header, box=None)


# =============================================================================
# Progress bar configurations
# =============================================================================


def create_execution_progress(console_instance: Console | None = None) -> Progress:
    """Create a detailed progress bar for job execution.

    Includes spinner, percentage, sheet count, elapsed time, ETA,
    and execution status display.

    Args:
        console_instance: Optional console to use. Defaults to module console.

    Returns:
        Configured Progress instance (not yet started).
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("\u2022"),  # bullet
        TextColumn("{task.completed}/{task.total} sheets"),
        TextColumn("\u2022"),
        TimeElapsedColumn(),
        TextColumn("\u2022"),
        TextColumn("ETA: {task.fields[eta]}"),
        TextColumn("\u2022"),
        TextColumn("[dim]{task.fields[exec_status]}[/dim]"),
        console=console_instance or console,
        transient=False,
    )


def create_status_progress(console_instance: Console | None = None) -> Progress:
    """Create a simple progress bar for status display.

    Shows description, bar, percentage, and count.

    Args:
        console_instance: Optional console to use. Defaults to module console.

    Returns:
        Configured Progress instance (not yet started).
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console_instance or console,
        transient=False,
    )


# =============================================================================
# Panel builders
# =============================================================================


def create_header_panel(
    lines: Sequence[str],
    title: str,
    border_style: str = "default",
) -> Panel:
    """Create a header panel with consistent styling.

    Args:
        lines: Lines of text to display in the panel.
        title: Panel title.
        border_style: Border color/style (e.g., "cyan", "green", "yellow").

    Returns:
        Configured Panel instance.
    """
    return Panel("\n".join(lines), title=title, border_style=border_style)


def create_run_summary_panel(
    summary: RunSummary,
    job_status: JobStatus,
) -> Panel:
    """Create a run summary panel.

    Args:
        summary: RunSummary instance with job completion data.
        job_status: Final job status for border color.

    Returns:
        Panel with styled run summary.
    """
    status_color = StatusColors.get_job_color(job_status)
    status_text = f"[{status_color}]{job_status.value.upper()}[/{status_color}]"

    lines = [
        f"[bold]{summary.job_name}[/bold]",
        f"Status: {status_text}",
        "",
        "[bold]Sheets[/bold]",
        f"  Completed: {summary.completed_sheets}/{summary.total_sheets}",
        f"  Failed: {summary.failed_sheets}",
        f"  Remaining: {summary.total_sheets - summary.completed_sheets
                        - summary.failed_sheets - summary.skipped_sheets}",
    ]

    # Add validation info if available
    if hasattr(summary, "validation_passed") and summary.validation_passed is not None:
        val_status = (
            "[green]All Passed[/green]"
            if summary.validation_passed
            else "[red]Some Failed[/red]"
        )
        lines.extend(["", "[bold]Validation[/bold]", f"  Status: {val_status}"])

    # Add execution time if available
    if hasattr(summary, "duration_seconds") and summary.duration_seconds:
        duration_str = format_duration(summary.duration_seconds)
        lines.extend(["", "[bold]Execution[/bold]", f"  Duration: {duration_str}"])

    border = "green" if job_status == JobStatus.COMPLETED else "yellow"
    return Panel("\n".join(lines), title="Run Summary", border_style=border)


def create_diagnostic_panel(
    job_name: str,
    job_id: str,
    status: JobStatus,
) -> Panel:
    """Create a diagnostic report header panel.

    Args:
        job_name: Name of the job.
        job_id: Job identifier.
        status: Current job status.

    Returns:
        Panel with diagnostic header info.
    """
    status_color = StatusColors.get_job_color(status)
    lines = [
        f"[bold]{job_name}[/bold]",
        f"ID: {job_id}",
        f"Status: [{status_color}]{status.value.upper()}[/{status_color}]",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    return Panel("\n".join(lines), title="Diagnostic Report", border_style="cyan")


def create_server_panel(
    title: str,
    server_name: str,
    info_lines: Sequence[str],
) -> Panel:
    """Create a server startup info panel.

    Args:
        title: Panel title.
        server_name: Name of the server being started.
        info_lines: Additional info lines (URLs, settings, etc.).

    Returns:
        Panel with server info.
    """
    lines = [f"[bold]{server_name}[/bold]", ""]
    lines.extend(info_lines)
    lines.extend(["", "[dim]Press Ctrl+C to stop[/dim]"])
    return Panel("\n".join(lines), title=title)


# =============================================================================
# Error formatting
# =============================================================================


def output_error(
    message: str,
    *,
    error_code: str | None = None,
    hints: list[str] | None = None,
    severity: Literal["error", "warning"] = "error",
    json_output: bool = False,
    console_instance: Console | None = None,
    **json_extras: str | int | float | bool | None,
) -> None:
    """Output a formatted error/warning with optional hints and JSON alternative.

    Consolidates the common CLI error output pattern:
    - Rich mode: colored error prefix, blank line, dim hints
    - JSON mode: structured dict with error_code, message, hints

    Args:
        message: The error message to display.
        error_code: Optional error code (e.g., "E501").
        hints: Optional list of hint strings for the user.
        severity: "error" (red) or "warning" (yellow).
        json_output: If True, output as JSON instead of Rich markup.
        console_instance: Console to print to. Defaults to module console.
        **json_extras: Extra key-value pairs included in JSON output only.
    """
    out = console_instance or console
    color = "red" if severity == "error" else "yellow"
    label = "Error" if severity == "error" else "Warning"

    if json_output:
        result: dict[str, str | int | float | bool | list[str] | None] = {
            "success": False,
            "message": message,
        }
        if error_code:
            result["error_code"] = error_code
        if hints:
            result["hints"] = hints
        for k, v in json_extras.items():
            result[k] = v
        import json as _json

        out.print(_json.dumps(result, indent=2))
        return

    if error_code:
        prefix = f"[{color}]{label} [{error_code}]:[/{color}] "
    else:
        prefix = f"[{color}]{label}:[/{color}] "
    out.print(f"{prefix}{message}")

    if hints:
        out.print()
        out.print("[dim]Hints:[/dim]")
        for hint in hints:
            out.print(f"  - {hint}")


def format_error_details(error: CheckpointErrorRecord) -> str:
    """Format detailed error information for display.

    Args:
        error: ErrorRecord object.

    Returns:
        Formatted string with error details.
    """
    lines = [
        f"[bold]Message:[/bold] {error.error_message or 'N/A'}",
        f"[bold]Type:[/bold] {error.error_type}",
        f"[bold]Code:[/bold] {error.error_code}",
        f"[bold]Attempt:[/bold] {error.attempt_number}",
    ]

    if error.timestamp:
        lines.append(
            f"[bold]Time:[/bold] {error.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    if error.context:
        context_str = ", ".join(
            f"{k}={v}" for k, v in error.context.items() if v is not None
        )
        if context_str:
            lines.append(f"[bold]Context:[/bold] {context_str}")

    if error.stdout_tail:
        from mozart.core.constants import TRUNCATE_STDOUT_TAIL_CHARS
        lines.append(
            f"\n[bold]Stdout (tail):[/bold]\n"
            f"[dim]{error.stdout_tail[:TRUNCATE_STDOUT_TAIL_CHARS]}[/dim]"
        )

    if error.stderr_tail:
        from mozart.core.constants import TRUNCATE_STDOUT_TAIL_CHARS
        lines.append(
            f"\n[bold]Stderr (tail):[/bold]\n"
            f"[red dim]{error.stderr_tail[:TRUNCATE_STDOUT_TAIL_CHARS]}[/red dim]"
        )

    if error.stack_trace:
        lines.append(f"\n[bold]Stack Trace:[/bold]\n[dim]{error.stack_trace[:800]}[/dim]")

    return "\n".join(lines)


# =============================================================================
# Status output helpers
# =============================================================================


def format_job_status_line(job: CheckpointState) -> str:
    """Format a single job's status as a styled line.

    Args:
        job: CheckpointState to format.

    Returns:
        Rich-formatted status line.
    """
    status_color = StatusColors.get_job_color(job.status)
    return (
        f"[{status_color}]{job.status.value}[/{status_color}] - "
        f"[cyan]{job.job_id}[/cyan] ({job.last_completed_sheet}/{job.total_sheets})"
    )


def print_job_status_header(
    job: CheckpointState,
    console_instance: Console | None = None,
) -> None:
    """Print job status header panel.

    Args:
        job: CheckpointState to display.
        console_instance: Console to print to. Defaults to module console.
    """
    out = console_instance or console
    status_color = StatusColors.get_job_color(job.status)

    lines = [
        f"[bold]{job.job_name}[/bold]",
        f"ID: [cyan]{job.job_id}[/cyan]",
        f"Status: [{status_color}]{job.status.value.upper()}[/{status_color}]",
    ]

    # Add duration if available
    if job.started_at:
        if job.completed_at:
            duration = job.completed_at - job.started_at
            duration_str = format_duration(duration.total_seconds())
            lines.append(f"Duration: {duration_str}")
        elif job.status == JobStatus.RUNNING and job.updated_at:
            elapsed = datetime.now(UTC) - job.started_at
            elapsed_str = format_duration(elapsed.total_seconds())
            lines.append(f"Running for: {elapsed_str}")

    out.print(Panel("\n".join(lines), title="Job Status"))


def print_timing_section(
    job: CheckpointState,
    console_instance: Console | None = None,
) -> None:
    """Print job timing information.

    Args:
        job: CheckpointState with timing data.
        console_instance: Console to print to. Defaults to module console.
    """
    out = console_instance or console
    out.print("\n[bold]Timing[/bold]")

    if job.created_at:
        out.print(f"  Created:  {format_timestamp(job.created_at)}")
    if job.started_at:
        out.print(f"  Started:  {format_timestamp(job.started_at)}")
    if job.updated_at:
        out.print(f"  Updated:  {format_timestamp(job.updated_at)}")
    if job.completed_at:
        out.print(f"  Completed: {format_timestamp(job.completed_at)}")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Console
    "console",
    # Color schemes
    "StatusColors",
    # Formatters
    "format_duration",
    "format_bytes",
    "format_timestamp",
    "format_validation_status",
    "output_error",
    "format_error_details",
    "format_job_status_line",
    "infer_error_type",
    # Table builders
    "create_jobs_table",
    "create_sheet_plan_table",
    "create_sheet_details_table",
    "create_synthesis_table",
    "create_errors_table",
    "create_timeline_table",
    "create_patterns_table",
    "create_simple_table",
    # Progress builders
    "create_execution_progress",
    "create_status_progress",
    # Panel builders
    "create_header_panel",
    "create_run_summary_panel",
    "create_diagnostic_panel",
    "create_server_panel",
    # Output helpers
    "print_job_status_header",
    "print_timing_section",
    # Rich re-exports for convenience
    "Console",
    "Panel",
    "Progress",
    "Table",
    "BarColumn",
    "SpinnerColumn",
    "TextColumn",
    "TimeElapsedColumn",
]
