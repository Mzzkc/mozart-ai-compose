"""Mozart CLI - Orchestration tool for Claude AI sessions.

Commands:
    run       Run a job from a YAML configuration file
    status    Show status of running or completed jobs
    resume    Resume a paused or failed job
    list      List all jobs
    validate  Validate a job configuration file
    dashboard Start the web dashboard
"""

import asyncio
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from mozart import __version__
from mozart.core.checkpoint import CheckpointState, JobStatus, SheetStatus
from mozart.core.config import JobConfig, NotificationConfig
from mozart.core.logging import configure_logging, get_logger
from mozart.execution.runner import RunSummary
from mozart.notifications import (
    DesktopNotifier,
    NotificationManager,
    Notifier,
    SlackNotifier,
    WebhookNotifier,
)
from mozart.state import JsonStateBackend, SQLiteStateBackend, StateBackend

# Module-level logger for CLI
_logger = get_logger("cli")


# Error message constants for consistent user-facing output
class ErrorMessages:
    """Constants for CLI error messages."""

    JOB_NOT_FOUND = "Job not found"
    CONFIG_LOAD_ERROR = "Error loading config"
    WORKSPACE_NOT_FOUND = "Workspace not found"
    STATE_FILE_NOT_FOUND = "State file not found"


class OutputLevel(str, Enum):
    """Output verbosity level."""

    QUIET = "quiet"  # Minimal output (errors only)
    NORMAL = "normal"  # Default output
    VERBOSE = "verbose"  # Detailed output


# Global output level state
_output_level: OutputLevel = OutputLevel.NORMAL

# Global logging configuration state
_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
_log_file: Path | None = None
_log_format: Literal["json", "console", "both"] = "console"
_logging_configured: bool = False

# Default state directory when no config is available
DEFAULT_STATE_DIR = Path.home() / ".mozart" / "state"


def _configure_global_logging() -> None:
    """Configure logging based on global CLI options.

    This is called after all callbacks have processed their options.
    Only configures once per session.
    """
    global _logging_configured
    if _logging_configured:
        return

    try:
        configure_logging(
            level=_log_level,
            format=_log_format,
            file_path=_log_file,
        )
        _logging_configured = True
        # Note: Intentionally not logging here to avoid polluting --json output
        # If debugging is needed, use --log-file to redirect logs
    except ValueError as e:
        # Handle configuration errors (e.g., format="both" without file_path)
        console.print(f"[red]Logging configuration error:[/red] {e}")
        raise typer.Exit(1) from None


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


def create_notifiers_from_config(
    notification_configs: list[NotificationConfig],
) -> list[Notifier]:
    """Create Notifier instances from notification configuration.

    Args:
        notification_configs: List of NotificationConfig from job config.

    Returns:
        List of configured Notifier instances.
    """
    notifiers: list[Notifier] = []

    for config in notification_configs:
        notifier: Notifier | None = None
        # Cast Literal list to str list for from_config methods
        events: list[str] = list(config.on_events)

        if config.type == "desktop":
            notifier = DesktopNotifier.from_config(
                on_events=events,
                config=config.config,
            )
        elif config.type == "slack":
            notifier = SlackNotifier.from_config(
                on_events=events,
                config=config.config,
            )
        elif config.type == "webhook":
            notifier = WebhookNotifier.from_config(
                on_events=events,
                config=config.config,
            )
        else:
            _logger.warning(f"Unknown notification type: {config.type}")
            continue

        if notifier:
            notifiers.append(notifier)

    return notifiers


app = typer.Typer(
    name="mozart",
    help="Orchestration tool for Claude AI sessions",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Mozart AI Compose v{__version__}")
        raise typer.Exit()


def verbose_callback(value: bool) -> None:
    """Set verbose output mode."""
    global _output_level
    if value:
        _output_level = OutputLevel.VERBOSE


def quiet_callback(value: bool) -> None:
    """Set quiet output mode."""
    global _output_level
    if value:
        _output_level = OutputLevel.QUIET


def log_level_callback(value: str | None) -> str | None:
    """Set log level from CLI option."""
    global _log_level
    if value:
        _log_level = value  # type: ignore[assignment]
    return value


def log_file_callback(value: Path | None) -> Path | None:
    """Set log file path from CLI option.

    When a log file is specified, logs are written to the file in
    human-readable format. Rich CLI output (progress bars, status tables)
    is separate from structured logging and still displays on console.
    """
    global _log_file, _log_format
    if value:
        _log_file = value
        # Keep console format for readable logs
        # File handler is now created automatically when file_path is set
        _log_format = "console"
    return value


def log_format_callback(value: str | None) -> str | None:
    """Set log format from CLI option."""
    global _log_format
    if value:
        _log_format = value  # type: ignore[assignment]
    return value


def get_output_level() -> OutputLevel:
    """Get current output level."""
    return _output_level


def is_verbose() -> bool:
    """Check if verbose output is enabled."""
    return _output_level == OutputLevel.VERBOSE


def is_quiet() -> bool:
    """Check if quiet output is enabled."""
    return _output_level == OutputLevel.QUIET


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        callback=verbose_callback,
        is_eager=True,
        help="Show detailed output with additional information",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        callback=quiet_callback,
        is_eager=True,
        help="Show minimal output (errors only)",
    ),
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            "-L",
            callback=log_level_callback,
            help="Logging level (DEBUG, INFO, WARNING, ERROR)",
            envvar="MOZART_LOG_LEVEL",
        ),
    ] = None,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            callback=log_file_callback,
            help="Path for log file output",
            envvar="MOZART_LOG_FILE",
        ),
    ] = None,
    log_format: Annotated[
        str | None,
        typer.Option(
            "--log-format",
            callback=log_format_callback,
            help="Log format: json, console, or both",
            envvar="MOZART_LOG_FORMAT",
        ),
    ] = None,
) -> None:
    """Mozart AI Compose - Orchestration tool for Claude AI sessions."""
    # Configure logging based on CLI options (called once)
    _configure_global_logging()


@app.command()
def run(
    config_file: Path = typer.Argument(
        ...,
        help="Path to YAML job configuration file",
        exists=True,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Show what would be executed without running",
    ),
    start_sheet: int | None = typer.Option(
        None,
        "--start-sheet",
        "-s",
        help="Override starting sheet number",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON for machine parsing",
    ),
    escalation: bool = typer.Option(
        False,
        "--escalation",
        "-e",
        help="Enable human-in-the-loop escalation for low-confidence sheets",
    ),
) -> None:
    """Run a job from a YAML configuration file."""
    from mozart.core.config import JobConfig

    try:
        config = JobConfig.from_yaml(config_file)
    except Exception as e:
        if json_output:
            console.print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from None

    # In quiet mode, skip the config panel
    if not is_quiet() and not json_output:
        console.print(Panel(
            f"[bold]{config.name}[/bold]\n"
            f"{config.description or 'No description'}\n\n"
            f"Backend: {config.backend.type}\n"
            f"Sheets: {config.sheet.total_sheets} "
            f"({config.sheet.size} items each)\n"
            f"Workspace: {config.workspace}",
            title="Job Configuration",
        ))

    # Validate flag compatibility
    if json_output and escalation:
        console.print(json.dumps({
            "error": "--escalation is incompatible with --json output mode. "
            "Escalation requires interactive console prompts."
        }, indent=2))
        raise typer.Exit(1)

    if dry_run:
        if not json_output:
            console.print("\n[yellow]Dry run - not executing[/yellow]")
            _show_dry_run(config)
        else:
            console.print(json.dumps({
                "dry_run": True,
                "job_name": config.name,
                "total_sheets": config.sheet.total_sheets,
            }, indent=2))
        return

    # Actually run the job
    if not is_quiet() and not json_output:
        console.print("\n[green]Starting job...[/green]")
    asyncio.run(_run_job(config, start_sheet, json_output, escalation))


async def _run_job(
    config: JobConfig,
    start_sheet: int | None,
    json_output: bool = False,
    escalation: bool = False,
) -> None:
    """Run the job asynchronously using the JobRunner with progress display."""
    from mozart.backends.anthropic_api import AnthropicApiBackend
    from mozart.backends.base import Backend
    from mozart.backends.claude_cli import ClaudeCliBackend
    from mozart.backends.recursive_light import RecursiveLightBackend
    from mozart.execution.runner import FatalError, GracefulShutdownError, JobRunner
    from mozart.learning.outcomes import JsonOutcomeStore

    # Ensure workspace exists
    config.workspace.mkdir(parents=True, exist_ok=True)

    # Setup backends based on config
    state_backend = create_state_backend_from_config(config)

    # Create appropriate backend based on type
    backend: Backend
    if config.backend.type == "recursive_light":
        rl_config = config.backend.recursive_light
        backend = RecursiveLightBackend(
            rl_endpoint=rl_config.endpoint,
            user_id=rl_config.user_id,
            timeout=rl_config.timeout,
        )
        if is_verbose() and not json_output:
            console.print(
                f"[dim]Using Recursive Light backend at {rl_config.endpoint}[/dim]"
            )
    elif config.backend.type == "anthropic_api":
        backend = AnthropicApiBackend.from_config(config.backend)
        if is_verbose() and not json_output:
            console.print(
                f"[dim]Using Anthropic API backend with model {config.backend.model}[/dim]"
            )
    else:
        # Default to ClaudeCliBackend (claude_cli)
        backend = ClaudeCliBackend.from_config(config.backend)

    # Execution progress state for CLI display (Task 4)
    execution_status: dict[str, Any] = {
        "sheet_num": None,
        "bytes_received": 0,
        "lines_received": 0,
        "elapsed_seconds": 0.0,
        "phase": "idle",
    }

    # Setup outcome store for learning if enabled
    outcome_store = None
    if config.learning.enabled:
        outcome_store_path = config.get_outcome_store_path()
        if config.learning.outcome_store_type == "json":
            outcome_store = JsonOutcomeStore(outcome_store_path)
        # Future: add SqliteOutcomeStore when implemented
        if is_verbose() and not json_output:
            console.print(
                f"[dim]Learning enabled: outcomes will be stored at {outcome_store_path}[/dim]"
            )

    # Setup global learning store for cross-workspace learning
    global_learning_store = None
    if config.learning.enabled:
        from mozart.learning.global_store import get_global_store
        global_learning_store = get_global_store()
        if is_verbose() and not json_output:
            console.print(
                "[dim]Global learning enabled: cross-workspace patterns active[/dim]"
            )

    # Setup notification manager from config
    notification_manager: NotificationManager | None = None
    if config.notifications:
        notifiers = create_notifiers_from_config(config.notifications)
        if notifiers:
            notification_manager = NotificationManager(notifiers)
            if is_verbose() and not json_output:
                console.print(
                    f"[dim]Notifications enabled: {len(notifiers)} channel(s) configured[/dim]"
                )

    # Create progress bar for sheet tracking (skip in quiet/json mode)
    progress: Progress | None = None
    progress_task_id: TaskID | None = None

    if not is_quiet() and not json_output:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} sheets"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("ETA: {task.fields[eta]}"),
            TextColumn("•"),
            TextColumn("[dim]{task.fields[exec_status]}[/dim]"),
            console=console,
            transient=False,
        )

    def _format_exec_status() -> str:
        """Format execution status for progress display."""
        if execution_status["phase"] == "idle":
            return ""
        if execution_status["phase"] == "starting":
            return "starting..."
        if execution_status["phase"] == "completed":
            return ""

        # Format bytes received
        bytes_recv = execution_status.get("bytes_received", 0)
        if bytes_recv < 1024:
            bytes_str = f"{bytes_recv}B"
        elif bytes_recv < 1024 * 1024:
            bytes_str = f"{bytes_recv / 1024:.1f}KB"
        else:
            bytes_str = f"{bytes_recv / (1024 * 1024):.1f}MB"

        return f"{bytes_str} received"

    def update_progress(completed: int, total: int, eta_seconds: float | None) -> None:
        """Update progress bar with current sheet progress."""
        nonlocal progress_task_id
        if progress is not None and progress_task_id is not None:
            eta_str = _format_duration(eta_seconds) if eta_seconds else "calculating..."
            exec_status = _format_exec_status()
            progress.update(
                progress_task_id,
                completed=completed,
                total=total,
                eta=eta_str,
                exec_status=exec_status,
            )

    def update_execution_display(progress_info: dict[str, Any]) -> None:
        """Update progress bar with execution status (called by backend)."""
        execution_status.update(progress_info)
        # Refresh the progress bar with new execution status
        if progress is not None and progress_task_id is not None:
            exec_status = _format_exec_status()
            progress.update(progress_task_id, exec_status=exec_status)

    # Override the execution progress callback to update display
    if isinstance(backend, ClaudeCliBackend) and not is_quiet() and not json_output:
        backend.progress_callback = update_execution_display

    # Setup escalation handler if enabled
    escalation_handler = None
    if escalation:
        from mozart.execution.escalation import ConsoleEscalationHandler

        # Enable escalation in config - required for runner to use the handler
        config.learning.escalation_enabled = True

        escalation_handler = ConsoleEscalationHandler(
            confidence_threshold=config.learning.min_confidence_threshold,
            auto_retry_on_first_failure=True,
        )
        if is_verbose() and not json_output:
            console.print(
                "[dim]Escalation enabled: low-confidence sheets will prompt for decisions[/dim]"
            )

    # Setup grounding engine if enabled (v8 Evolution: External Grounding Hooks)
    # v9 Evolution: Wire up hook registration from config
    grounding_engine = None
    if config.grounding.enabled:
        from mozart.execution.grounding import GroundingEngine, create_hook_from_config

        grounding_engine = GroundingEngine(hooks=[], config=config.grounding)

        # Register hooks from configuration (v9: Integration-only completion)
        for hook_config in config.grounding.hooks:
            try:
                hook = create_hook_from_config(hook_config)
                grounding_engine.add_hook(hook)
                if is_verbose() and not json_output:
                    console.print(f"[dim]  Registered hook: {hook.name}[/dim]")
            except ValueError as e:
                console.print(f"[yellow]Warning: Failed to create hook: {e}[/yellow]")

        if is_verbose() and not json_output:
            hook_count = grounding_engine.get_hook_count()
            console.print(
                f"[dim]Grounding enabled: {hook_count} hook(s) registered[/dim]"
            )

    # Create runner with progress callback
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=state_backend,
        console=console if not json_output else Console(quiet=True),
        outcome_store=outcome_store,
        escalation_handler=escalation_handler,
        progress_callback=update_progress if progress else None,
        global_learning_store=global_learning_store,
        grounding_engine=grounding_engine,
    )

    job_id = config.name  # Use job name as ID for now
    summary: RunSummary | None = None

    try:
        # Send job start notification
        if notification_manager:
            await notification_manager.notify_job_start(
                job_id=job_id,
                job_name=config.name,
                total_sheets=config.sheet.total_sheets,
            )

        # Start progress display
        if progress:
            progress.start()
            starting_sheet = start_sheet or 1
            initial_completed = starting_sheet - 1
            progress_task_id = progress.add_task(
                f"[cyan]{config.name}[/cyan]",
                total=config.sheet.total_sheets,
                completed=initial_completed,
                eta="calculating...",
                exec_status="",  # Initial empty execution status
            )

        # Run job with validation and completion recovery
        state, summary = await runner.run(start_sheet=start_sheet)

        # Stop progress and show final state
        if progress:
            progress.stop()

        if json_output:
            # Output summary as JSON
            console.print(json.dumps(summary.to_dict(), indent=2))
        elif state.status == JobStatus.COMPLETED:
            _display_run_summary(summary)

            # Send job complete notification
            if notification_manager:
                await notification_manager.notify_job_complete(
                    job_id=job_id,
                    job_name=config.name,
                    success_count=summary.completed_sheets,
                    failure_count=summary.failed_sheets,
                    duration_seconds=summary.total_duration_seconds,
                )
        else:
            if not is_quiet():
                console.print(
                    f"[yellow]Job ended with status: {state.status.value}[/yellow]"
                )
                _display_run_summary(summary)

            # Send job failed notification if not completed
            if notification_manager and state.status == JobStatus.FAILED:
                await notification_manager.notify_job_failed(
                    job_id=job_id,
                    job_name=config.name,
                    error_message=f"Job failed with status: {state.status.value}",
                    sheet_num=state.current_sheet,
                )

    except GracefulShutdownError:
        # Graceful shutdown already saved state and printed resume hint
        if progress:
            progress.stop()
        if not json_output:
            console.print("[yellow]Job paused. Exiting gracefully.[/yellow]")
        else:
            # Get summary from runner even on shutdown
            summary = runner.get_summary()
            if summary:
                summary.final_status = JobStatus.PAUSED
                console.print(json.dumps(summary.to_dict(), indent=2))
        raise typer.Exit(0) from None

    except FatalError as e:
        if progress:
            progress.stop()

        if json_output:
            summary = runner.get_summary()
            if summary:
                summary.final_status = JobStatus.FAILED
                output = summary.to_dict()
                output["error"] = str(e)
                console.print(json.dumps(output, indent=2))
            else:
                console.print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(f"[red]Fatal error: {e}[/red]")

        # Send failure notification
        if notification_manager:
            await notification_manager.notify_job_failed(
                job_id=job_id,
                job_name=config.name,
                error_message=str(e),
            )

        raise typer.Exit(1) from None

    finally:
        # Ensure progress is stopped
        if progress and progress.live.is_started:
            progress.stop()

        # Clean up notification resources
        if notification_manager:
            await notification_manager.close()


def _display_run_summary(summary: RunSummary) -> None:
    """Display run summary as a rich panel.

    Args:
        summary: Run summary with execution statistics.
    """
    if is_quiet():
        return

    # Build status indicator
    status_color = {
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.PAUSED: "yellow",
    }.get(summary.final_status, "white")

    status_text = f"[{status_color}]{summary.final_status.value.upper()}[/{status_color}]"

    # Build summary content
    lines = [
        f"[bold]{summary.job_name}[/bold]",
        f"Status: {status_text}",
        f"Duration: {summary._format_duration(summary.total_duration_seconds)}",
        "",
        "[bold]Sheets[/bold]",
        f"  Completed: [green]{summary.completed_sheets}[/green]/{summary.total_sheets}",
    ]

    if summary.failed_sheets > 0:
        lines.append(f"  Failed: [red]{summary.failed_sheets}[/red]")
    if summary.skipped_sheets > 0:
        lines.append(f"  Skipped: [yellow]{summary.skipped_sheets}[/yellow]")

    lines.append(f"  Success Rate: {summary.success_rate:.1f}%")

    # Validation stats
    if summary.validation_pass_count + summary.validation_fail_count > 0:
        lines.extend([
            "",
            "[bold]Validation[/bold]",
            f"  Pass Rate: {summary.validation_pass_rate:.1f}%",
        ])

    # Execution stats (show in verbose mode or if notable)
    if is_verbose() or summary.total_retries > 0 or summary.rate_limit_waits > 0:
        lines.extend([
            "",
            "[bold]Execution[/bold]",
        ])
        if summary.first_attempt_successes > 0:
            lines.append(
                f"  First Attempt Success: {summary.first_attempt_rate:.0f}% "
                f"({summary.first_attempt_successes}/{summary.completed_sheets})"
            )
        if summary.total_retries > 0:
            lines.append(f"  Retries Used: {summary.total_retries}")
        if summary.total_completion_attempts > 0:
            lines.append(f"  Completion Attempts: {summary.total_completion_attempts}")
        if summary.rate_limit_waits > 0:
            lines.append(f"  Rate Limit Waits: [yellow]{summary.rate_limit_waits}[/yellow]")

    console.print(Panel(
        "\n".join(lines),
        title="Run Summary",
        border_style="green" if summary.final_status == JobStatus.COMPLETED else "yellow",
    ))


def _show_dry_run(config: JobConfig) -> None:
    """Show what would be executed in dry run mode."""
    table = Table(title="Sheet Plan")
    table.add_column("Sheet", style="cyan")
    table.add_column("Items", style="green")
    table.add_column("Validations", style="yellow")

    for sheet_num in range(1, config.sheet.total_sheets + 1):
        start = (sheet_num - 1) * config.sheet.size + config.sheet.start_item
        end = min(start + config.sheet.size - 1, config.sheet.total_items)
        table.add_row(
            str(sheet_num),
            f"{start}-{end}",
            str(len(config.validations)),
        )

    console.print(table)


@app.command()
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


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (optional if config_snapshot exists in state)",
        exists=True,
        readable=True,
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to search for job state",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force resume even if job appears completed",
    ),
    escalation: bool = typer.Option(
        False,
        "--escalation",
        "-e",
        help="Enable human-in-the-loop escalation for low-confidence sheets",
    ),
    reload_config: bool = typer.Option(
        False,
        "--reload-config",
        "-r",
        help="Reload config from yaml file instead of using cached snapshot. "
        "Use with --config to specify a new file, or it will reload from the original path.",
    ),
) -> None:
    """Resume a paused or failed job.

    Loads the job state from the state backend and continues execution
    from where it left off. The job configuration is reconstructed from
    the stored config_snapshot, or you can provide a config file with --config.

    Examples:
        mozart resume my-job
        mozart resume my-job --config job.yaml
        mozart resume my-job --workspace ./workspace
        mozart resume my-job --escalation
        mozart resume my-job --reload-config  # Reload from original yaml
        mozart resume my-job -r --config updated.yaml  # Use updated config
    """
    asyncio.run(_resume_job(job_id, config_file, workspace, force, escalation, reload_config))


async def _resume_job(
    job_id: str,
    config_file: Path | None,
    workspace: Path | None,
    force: bool,
    escalation: bool = False,
    reload_config: bool = False,
) -> None:
    """Resume a paused or failed job.

    Args:
        job_id: Job ID to resume.
        config_file: Optional path to config file.
        workspace: Optional workspace directory to search.
        force: Force resume even if job appears completed.
        escalation: Enable human-in-the-loop escalation for low-confidence sheets.
        reload_config: If True, reload config from yaml file instead of cached snapshot.
    """
    from mozart.backends.anthropic_api import AnthropicApiBackend
    from mozart.backends.base import Backend
    from mozart.backends.claude_cli import ClaudeCliBackend
    from mozart.backends.recursive_light import RecursiveLightBackend
    from mozart.execution.runner import FatalError, GracefulShutdownError, JobRunner
    from mozart.learning.outcomes import JsonOutcomeStore

    # Find job state in backends
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

    # Find job in backends
    found_state: CheckpointState | None = None
    found_backend: StateBackend | None = None

    for state_bknd in backends:
        try:
            state = await state_bknd.load(job_id)
            if state:
                found_state = state
                found_backend = state_bknd
                break
        except Exception as e:
            _logger.debug(f"Error querying backend for {job_id}: {e}")
            continue

    if found_state is None or found_backend is None:
        console.print(f"[red]{ErrorMessages.JOB_NOT_FOUND}:[/red] {job_id}")
        console.print(
            "\n[dim]Hint: Use --workspace to specify the directory "
            "containing the job state.[/dim]"
        )
        raise typer.Exit(1)

    # Check if job is in a resumable state
    resumable_statuses = {JobStatus.PAUSED, JobStatus.FAILED, JobStatus.RUNNING}
    if found_state.status not in resumable_statuses:
        if found_state.status == JobStatus.COMPLETED and not force:
            console.print(
                f"[yellow]Job '{job_id}' is already completed.[/yellow]"
            )
            console.print(
                "[dim]Use --force to resume anyway (will restart from last sheet).[/dim]"
            )
            raise typer.Exit(1)
        elif found_state.status == JobStatus.PENDING:
            console.print(
                f"[yellow]Job '{job_id}' has not been started yet.[/yellow]"
            )
            console.print("[dim]Use 'mozart run' to start the job.[/dim]")
            raise typer.Exit(1)

    # Reconstruct JobConfig
    config: JobConfig | None = None
    config_was_reloaded = False

    # Priority 1: Use provided config file (always takes precedence)
    if config_file:
        try:
            config = JobConfig.from_yaml(config_file)
            console.print(f"[dim]Using config from: {config_file}[/dim]")
            config_was_reloaded = True
        except Exception as e:
            console.print(f"[red]Error loading config file:[/red] {e}")
            raise typer.Exit(1) from None

    # Priority 2: If reload_config, force reload from config_path
    elif reload_config:
        if found_state.config_path:
            config_path = Path(found_state.config_path)
            if config_path.exists():
                try:
                    config = JobConfig.from_yaml(config_path)
                    console.print(
                        f"[cyan]Reloaded config from:[/cyan] {config_path}"
                    )
                    config_was_reloaded = True
                except Exception as e:
                    console.print(f"[red]Error reloading config:[/red] {e}")
                    raise typer.Exit(1) from None
            else:
                console.print(
                    f"[red]Cannot reload: config file not found:[/red] {config_path}\n"
                    "[dim]Hint: Use --config to specify a new config file.[/dim]"
                )
                raise typer.Exit(1)
        else:
            console.print(
                "[red]Cannot reload: no config_path stored in state.[/red]\n"
                "[dim]Hint: Use --config to specify a config file.[/dim]"
            )
            raise typer.Exit(1)

    # Priority 3: Reconstruct from config_snapshot (default)
    elif found_state.config_snapshot:
        try:
            config = JobConfig.model_validate(found_state.config_snapshot)
            console.print("[dim]Reconstructed config from saved state[/dim]")
        except Exception as e:
            console.print(f"[red]Error reconstructing config from snapshot:[/red] {e}")
            console.print(
                "[dim]Hint: Provide a config file with --config flag.[/dim]"
            )
            raise typer.Exit(1) from None

    # Priority 4: Try to load from stored config_path as last resort
    elif found_state.config_path:
        config_path = Path(found_state.config_path)
        if config_path.exists():
            try:
                config = JobConfig.from_yaml(config_path)
                console.print(f"[dim]Loaded config from stored path: {config_path}[/dim]")
            except Exception as e:
                console.print(f"[red]Error loading stored config:[/red] {e}")
                raise typer.Exit(1) from None
        else:
            console.print(
                f"[yellow]Stored config file not found:[/yellow] {config_path}"
            )
            console.print("[dim]Hint: Provide a config file with --config flag.[/dim]")
            raise typer.Exit(1)
    else:
        console.print(
            "[red]Cannot resume: No config available.[/red]\n"
            "The job state doesn't contain a config snapshot.\n"
            "Please provide a config file with --config flag."
        )
        raise typer.Exit(1)

    # Update config_snapshot in state if config was reloaded
    if config_was_reloaded and config:
        found_state.config_snapshot = config.model_dump(mode="json")
        console.print("[dim]Updated cached config snapshot[/dim]")

    # Calculate resume point
    resume_sheet = found_state.last_completed_sheet + 1
    if resume_sheet > found_state.total_sheets:
        if force:
            # For force resume, restart from last sheet
            resume_sheet = found_state.total_sheets
            console.print(
                f"[yellow]Job was completed. Force restarting sheet {resume_sheet}.[/yellow]"
            )
        else:
            console.print("[green]Job is already fully completed.[/green]")
            return

    # Display resume info
    console.print(Panel(
        f"[bold]{config.name}[/bold]\n"
        f"Status: {found_state.status.value}\n"
        f"Progress: {found_state.last_completed_sheet}/{found_state.total_sheets} sheets\n"
        f"Resuming from sheet: {resume_sheet}",
        title="Resume Job",
    ))

    # Reset job status to RUNNING for resume
    found_state.status = JobStatus.RUNNING
    found_state.error_message = None  # Clear previous error
    await found_backend.save(found_state)

    # Setup backends for execution
    backend: Backend
    if config.backend.type == "recursive_light":
        rl_config = config.backend.recursive_light
        backend = RecursiveLightBackend(
            rl_endpoint=rl_config.endpoint,
            user_id=rl_config.user_id,
            timeout=rl_config.timeout,
        )
        console.print(
            f"[dim]Using Recursive Light backend at {rl_config.endpoint}[/dim]"
        )
    elif config.backend.type == "anthropic_api":
        backend = AnthropicApiBackend.from_config(config.backend)
        console.print(
            f"[dim]Using Anthropic API backend with model {config.backend.model}[/dim]"
        )
    else:
        backend = ClaudeCliBackend.from_config(config.backend)

    # Setup outcome store for learning if enabled
    outcome_store = None
    if config.learning.enabled:
        outcome_store_path = config.get_outcome_store_path()
        if config.learning.outcome_store_type == "json":
            outcome_store = JsonOutcomeStore(outcome_store_path)
        console.print(
            f"[dim]Learning enabled: outcomes will be stored at {outcome_store_path}[/dim]"
        )

    # Setup global learning store for cross-workspace learning
    global_learning_store = None
    if config.learning.enabled:
        from mozart.learning.global_store import get_global_store
        global_learning_store = get_global_store()
        console.print(
            "[dim]Global learning enabled: cross-workspace patterns active[/dim]"
        )

    # Setup notification manager from config
    notification_manager: NotificationManager | None = None
    if config.notifications:
        notifiers = create_notifiers_from_config(config.notifications)
        if notifiers:
            notification_manager = NotificationManager(notifiers)
            console.print(
                f"[dim]Notifications enabled: {len(notifiers)} channel(s) configured[/dim]"
            )

    # Create progress bar for sheet tracking
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} sheets"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("ETA: {task.fields[eta]}"),
        console=console,
        transient=False,
    )

    # Progress callback to update the progress bar
    progress_task_id: TaskID | None = None

    def update_progress(completed: int, total: int, eta_seconds: float | None) -> None:
        """Update progress bar with current sheet progress."""
        nonlocal progress_task_id
        if progress_task_id is not None:
            eta_str = _format_duration(eta_seconds) if eta_seconds else "calculating..."
            progress.update(
                progress_task_id,
                completed=completed,
                total=total,
                eta=eta_str,
            )

    # Setup escalation handler if enabled
    escalation_handler = None
    if escalation:
        from mozart.execution.escalation import ConsoleEscalationHandler

        # Enable escalation in config - required for runner to use the handler
        config.learning.escalation_enabled = True

        escalation_handler = ConsoleEscalationHandler(
            confidence_threshold=config.learning.min_confidence_threshold,
            auto_retry_on_first_failure=True,
        )
        if is_verbose():
            console.print(
                "[dim]Escalation enabled: low-confidence sheets will prompt for decisions[/dim]"
            )

    # Setup grounding engine if enabled (v8 Evolution: External Grounding Hooks)
    # v9 Evolution: Wire up hook registration from config
    grounding_engine = None
    if config.grounding.enabled:
        from mozart.execution.grounding import GroundingEngine, create_hook_from_config

        grounding_engine = GroundingEngine(hooks=[], config=config.grounding)

        # Register hooks from configuration (v9: Integration-only completion)
        for hook_config in config.grounding.hooks:
            try:
                hook = create_hook_from_config(hook_config)
                grounding_engine.add_hook(hook)
                if is_verbose():
                    console.print(f"[dim]  Registered hook: {hook.name}[/dim]")
            except ValueError as e:
                console.print(f"[yellow]Warning: Failed to create hook: {e}[/yellow]")

        if is_verbose():
            hook_count = grounding_engine.get_hook_count()
            console.print(
                f"[dim]Grounding enabled: {hook_count} hook(s) registered[/dim]"
            )

    # Create runner with progress callback
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=found_backend,
        console=console,
        outcome_store=outcome_store,
        escalation_handler=escalation_handler,
        progress_callback=update_progress,
        global_learning_store=global_learning_store,
        grounding_engine=grounding_engine,
    )

    try:
        # Send job resume notification (use job_start event)
        if notification_manager:
            remaining_sheets = found_state.total_sheets - found_state.last_completed_sheet
            await notification_manager.notify_job_start(
                job_id=job_id,
                job_name=config.name,
                total_sheets=remaining_sheets,
            )

        # Start progress display
        progress.start()
        progress_task_id = progress.add_task(
            f"[cyan]{config.name}[/cyan] (resuming)",
            total=found_state.total_sheets,
            completed=found_state.last_completed_sheet,
            eta="calculating...",
        )

        # Resume from the next sheet
        if not is_quiet():
            console.print(f"\n[green]Resuming from sheet {resume_sheet}...[/green]")
        state, summary = await runner.run(
            start_sheet=resume_sheet,
            config_path=str(config_file) if config_file else found_state.config_path,
        )

        # Stop progress and show final state
        progress.stop()

        if state.status == JobStatus.COMPLETED:
            _display_run_summary(summary)

            # Send job complete notification
            if notification_manager:
                await notification_manager.notify_job_complete(
                    job_id=job_id,
                    job_name=config.name,
                    success_count=summary.completed_sheets,
                    failure_count=summary.failed_sheets,
                    duration_seconds=summary.total_duration_seconds,
                )
        else:
            if not is_quiet():
                console.print(
                    f"[yellow]Job ended with status: {state.status.value}[/yellow]"
                )
                _display_run_summary(summary)

            # Send job failed notification if not completed
            if notification_manager and state.status == JobStatus.FAILED:
                await notification_manager.notify_job_failed(
                    job_id=job_id,
                    job_name=config.name,
                    error_message=f"Job failed with status: {state.status.value}",
                    sheet_num=state.current_sheet,
                )

    except GracefulShutdownError:
        # Graceful shutdown already saved state and printed resume hint
        progress.stop()
        console.print("[yellow]Job paused. Exiting gracefully.[/yellow]")
        raise typer.Exit(0) from None

    except FatalError as e:
        progress.stop()
        console.print(f"[red]Fatal error: {e}[/red]")

        # Send failure notification
        if notification_manager:
            await notification_manager.notify_job_failed(
                job_id=job_id,
                job_name=config.name,
                error_message=str(e),
            )

        raise typer.Exit(1) from None

    finally:
        # Ensure progress is stopped
        if progress.live.is_started:
            progress.stop()

        if notification_manager:
            await notification_manager.close()


@app.command(hidden=True)
def recover(
    job_id: str = typer.Argument(..., help="Job ID to recover"),
    sheet: int | None = typer.Option(
        None,
        "--sheet",
        "-s",
        help="Specific sheet number to recover (default: all failed sheets)",
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing job state",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-n",
        help="Check validations without modifying state",
    ),
) -> None:
    """Recover sheets that completed work but were incorrectly marked as failed.

    This command runs validations for failed sheets without re-executing them.
    If validations pass, the sheet is marked as complete.

    This is useful when:
    - Claude CLI returned a non-zero exit code but the work was done
    - A transient error caused failure after files were created
    - You want to check if a failed sheet actually succeeded

    Examples:
        mozart recover my-job                    # Recover all failed sheets
        mozart recover my-job --sheet 6         # Recover specific sheet
        mozart recover my-job --dry-run         # Check without modifying
    """
    asyncio.run(_recover_job(job_id, sheet, workspace, dry_run))


async def _recover_job(
    job_id: str,
    sheet_num: int | None,
    workspace: Path | None,
    dry_run: bool,
) -> None:
    """Recover sheets by running validations without re-executing.

    Args:
        job_id: Job ID to recover.
        sheet_num: Specific sheet to recover, or None for all failed sheets.
        workspace: Optional workspace directory.
        dry_run: If True, only check validations without modifying state.
    """
    from mozart.core.config import JobConfig
    from mozart.execution.validation import ValidationEngine

    _configure_global_logging()

    # Find job state
    state_file = None
    search_paths = []

    if workspace:
        search_paths.append(workspace)
    else:
        search_paths.extend([
            Path.cwd(),
            Path.cwd() / job_id,
            Path.home() / ".mozart" / "state",
        ])

    for search_path in search_paths:
        candidate = search_path / f"{job_id}.json"
        if candidate.exists():
            state_file = candidate
            break

    if not state_file:
        console.print(f"[red]Job state not found: {job_id}[/red]")
        console.print(f"[dim]Searched: {', '.join(str(p) for p in search_paths)}[/dim]")
        raise typer.Exit(1)

    # Load state
    state_backend = JsonStateBackend(state_file.parent)
    state = await state_backend.load(job_id)

    if not state:
        console.print(f"[red]Could not load state for job: {job_id}[/red]")
        raise typer.Exit(1)

    # Reconstruct config from snapshot
    if not state.config_snapshot:
        console.print("[red]No config snapshot in state - cannot run validations[/red]")
        raise typer.Exit(1)

    config = JobConfig.model_validate(state.config_snapshot)

    # Determine which sheets to check
    sheets_to_check: list[int] = []
    if sheet_num is not None:
        sheets_to_check = [sheet_num]
    else:
        # Find all failed sheets
        for snum, sheet_state in state.sheets.items():
            if sheet_state.status == SheetStatus.FAILED:
                sheets_to_check.append(int(snum))

    if not sheets_to_check:
        console.print("[green]No failed sheets to recover[/green]")
        raise typer.Exit(0)

    console.print(Panel(
        f"[bold]Recover Job: {job_id}[/bold]\n"
        f"Sheets to check: {sheets_to_check}\n"
        f"Dry run: {dry_run}",
        title="Recovery",
    ))

    # Create validation engine
    validation_engine = ValidationEngine(
        context_factory=lambda snum: ValidationEngine.create_context(
            sheet_num=snum,
            workspace=config.workspace,
            project_root=config.backend.working_directory or Path.cwd(),
        )
    )

    recovered_count = 0
    for snum in sorted(sheets_to_check):
        console.print(f"\n[bold]Sheet {snum}:[/bold]")

        # Run validations for this sheet
        validation_engine.context = ValidationEngine.create_context(
            sheet_num=snum,
            workspace=config.workspace,
            project_root=config.backend.working_directory or Path.cwd(),
        )
        result = validation_engine.run_validations(config.validations)

        # Show results
        for vr in result.results:
            status = "[green]✓[/green]" if vr.passed else "[red]✗[/red]"
            console.print(f"  {status} {vr.rule.description}")

        if result.all_passed:
            console.print(f"  [green]All {len(result.results)} validations passed![/green]")

            if not dry_run:
                # Update state to mark sheet as completed
                state.sheets[snum].status = SheetStatus.COMPLETED
                state.sheets[snum].validation_passed = True
                state.sheets[snum].validation_details = result.to_dict_list()
                state.sheets[snum].error_message = None
                state.sheets[snum].error_category = None

                # Update last_completed_sheet if this extends it
                if snum > state.last_completed_sheet:
                    state.last_completed_sheet = snum

                recovered_count += 1
                console.print(f"  [blue]→ Marked as completed[/blue]")
            else:
                console.print(f"  [yellow]→ Would mark as completed (dry-run)[/yellow]")
        else:
            failed_count = len([r for r in result.results if not r.passed])
            console.print(
                f"  [red]{failed_count} validation(s) failed - cannot recover[/red]"
            )

    # Save state if not dry run
    if not dry_run and recovered_count > 0:
        # Update job status if all sheets now complete
        all_complete = all(
            s.status == SheetStatus.COMPLETED
            for s in state.sheets.values()
        )
        if all_complete:
            state.status = JobStatus.COMPLETED
        elif state.status == JobStatus.FAILED:
            state.status = JobStatus.PAUSED  # Allow resume

        await state_backend.save(state)
        console.print(f"\n[green]Recovered {recovered_count} sheet(s)[/green]")
    elif dry_run:
        console.print(f"\n[yellow]Dry run complete - no changes made[/yellow]")
    else:
        console.print(f"\n[yellow]No sheets could be recovered[/yellow]")


@app.command(name="list")
def list_jobs(
    status: str | None = typer.Option(
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
    asyncio.run(_list_jobs(status, limit, workspace))


async def _list_jobs(
    status_filter: str | None,
    limit: int,
    workspace: Path | None,
) -> None:
    """Asynchronously list jobs from state backends."""
    # Determine which backends to query
    backends: list[tuple[str, StateBackend]] = []

    if workspace:
        # Use specified workspace
        if not workspace.exists():
            console.print(f"[red]Workspace not found:[/red] {workspace}")
            raise typer.Exit(1)

        # Check for SQLite state file
        sqlite_path = workspace / ".mozart-state.db"
        if sqlite_path.exists():
            backends.append((str(workspace), SQLiteStateBackend(sqlite_path)))

        # Also check for JSON state files
        json_backend = JsonStateBackend(workspace)
        backends.append((str(workspace), json_backend))
    else:
        # Try default locations
        # 1. Current directory
        cwd = Path.cwd()
        json_backend = JsonStateBackend(cwd)
        backends.append((".", json_backend))

        # 2. SQLite in current directory
        sqlite_cwd = cwd / ".mozart-state.db"
        if sqlite_cwd.exists():
            backends.append((".", SQLiteStateBackend(sqlite_cwd)))

    from mozart.core.checkpoint import CheckpointState

    # Collect all jobs
    all_jobs: list[tuple[str, CheckpointState]] = []

    for source, backend in backends:
        try:
            jobs = await backend.list_jobs()
            for job in jobs:
                all_jobs.append((source, job))
        except Exception as e:
            _logger.debug(f"Error querying backend {source}: {e}")
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
    unique_jobs.sort(key=lambda x: x[1].updated_at, reverse=True)
    unique_jobs = unique_jobs[:limit]

    if not unique_jobs:
        console.print("[dim]No jobs found.[/dim]")
        if status_filter:
            console.print("[dim]Try without --status filter or check a different workspace.[/dim]")
        return

    # Build table
    table = Table(title="Mozart Jobs")
    table.add_column("Job ID", style="cyan", no_wrap=True)
    table.add_column("Status", style="bold")
    table.add_column("Progress", justify="right")
    table.add_column("Updated", style="dim")

    for _source, job in unique_jobs:
        # Format status with color
        status_style = {
            JobStatus.PENDING: "yellow",
            JobStatus.RUNNING: "blue",
            JobStatus.COMPLETED: "green",
            JobStatus.FAILED: "red",
            JobStatus.PAUSED: "magenta",
        }.get(job.status, "white")
        status_str = f"[{status_style}]{job.status.value}[/{status_style}]"

        # Format progress
        progress = f"{job.last_completed_sheet}/{job.total_sheets}"

        # Format updated time
        updated = job.updated_at.strftime("%Y-%m-%d %H:%M") if job.updated_at else "-"

        table.add_row(job.job_id, status_str, progress, updated)

    console.print(table)
    console.print(f"\n[dim]Showing {len(unique_jobs)} job(s)[/dim]")


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
                    console.print(json.dumps({"error": f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}"}, indent=2))
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


async def _status_job(
    job_id: str,
    json_output: bool,
    workspace: Path | None,
) -> None:
    """Asynchronously get and display status for a specific job."""
    # Determine which backends to query
    backends: list[StateBackend] = []

    if workspace:
        # Use specified workspace
        if not workspace.exists():
            console.print(f"[red]Workspace not found:[/red] {workspace}")
            raise typer.Exit(1)

        # Check for SQLite state file
        sqlite_path = workspace / ".mozart-state.db"
        if sqlite_path.exists():
            backends.append(SQLiteStateBackend(sqlite_path))

        # Also check for JSON state files
        backends.append(JsonStateBackend(workspace))
    else:
        # Try default locations
        cwd = Path.cwd()
        backends.append(JsonStateBackend(cwd))

        # SQLite in current directory
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
            _logger.debug(f"Error querying backend for {job_id}: {e}")
            continue

    if not found_job:
        if json_output:
            console.print(json.dumps({"error": f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}"}, indent=2))
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


def _output_status_json(job: CheckpointState) -> None:
    """Output job status as JSON."""

    # Build a clean JSON representation
    # Use last_completed_sheet for progress since it's more reliable than counting sheets dict
    completed = job.last_completed_sheet
    total = job.total_sheets
    percent = (completed / total * 100) if total > 0 else 0.0

    # Collect recent errors for JSON output
    recent_errors_data: list[dict[str, Any]] = []
    for sheet_num, error in _collect_recent_errors_for_json(job, limit=5):
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


def _collect_recent_errors_for_json(
    job: CheckpointState,
    limit: int = 5,
) -> list[tuple[int, Any]]:
    """Collect recent errors for JSON output (no rich formatting).

    This is a variant of _collect_recent_errors that works without
    importing the full ErrorRecord type at module level.

    Args:
        job: CheckpointState to collect errors from.
        limit: Maximum number of errors to return.

    Returns:
        List of (sheet_num, error) tuples, sorted by timestamp descending.
    """
    from mozart.core.checkpoint import ErrorRecord

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


def _output_status_rich(job: CheckpointState) -> None:
    """Output job status with rich formatting."""

    # Status color mapping
    status_colors = {
        JobStatus.PENDING: "yellow",
        JobStatus.RUNNING: "blue",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.PAUSED: "magenta",
        JobStatus.CANCELLED: "dim",
    }
    status_color = status_colors.get(job.status, "white")

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
            duration_str = _format_duration(duration.total_seconds())
            header_lines.append(f"Duration: {duration_str}")
        elif job.status == JobStatus.RUNNING and job.updated_at:
            from datetime import UTC, datetime
            elapsed = datetime.now(UTC) - job.started_at
            elapsed_str = _format_duration(elapsed.total_seconds())
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
        # Force refresh to display
        progress.refresh()

    # Sheet details table
    if job.sheets:
        console.print("\n[bold]Sheet Details[/bold]")
        sheet_table = Table(show_header=True, header_style="bold")
        sheet_table.add_column("#", justify="right", style="cyan", width=4)
        sheet_table.add_column("Status", width=12)
        sheet_table.add_column("Attempts", justify="right", width=8)
        sheet_table.add_column("Validation", width=10)
        sheet_table.add_column("Error", style="dim", no_wrap=False)

        sheet_status_colors = {
            SheetStatus.PENDING: "yellow",
            SheetStatus.IN_PROGRESS: "blue",
            SheetStatus.COMPLETED: "green",
            SheetStatus.FAILED: "red",
            SheetStatus.SKIPPED: "dim",
        }

        for sheet_num in sorted(job.sheets.keys()):
            sheet = job.sheets[sheet_num]
            sheet_color = sheet_status_colors.get(sheet.status, "white")

            # Format validation status
            if sheet.validation_passed is None:
                val_str = "-"
            elif sheet.validation_passed:
                val_str = "[green]✓ Pass[/green]"
            else:
                val_str = "[red]✗ Fail[/red]"

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

    # Show error message if job failed
    if job.error_message:
        console.print(f"\n[bold red]Error:[/bold red] {job.error_message}")

    # Timing info
    console.print("\n[bold]Timing[/bold]")
    if job.created_at:
        console.print(f"  Created:  {job.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if job.started_at:
        console.print(f"  Started:  {job.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if job.updated_at:
        console.print(f"  Updated:  {job.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if job.completed_at:
        console.print(f"  Completed: {job.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

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
            type_styles = {
                "permanent": "red",
                "transient": "yellow",
                "rate_limit": "blue",
            }
            type_style = type_styles.get(error.error_type, "white")

            # Truncate message
            message = error.error_message or ""
            if len(message) > 60:
                message = message[:57] + "..."

            console.print(
                f"  [{type_style}]•[/{type_style}] Sheet {sheet_num}: "
                f"[{type_style}]{error.error_code}[/{type_style}] - {message}"
            )

        # Hint for more details
        console.print(
            f"\n[dim]  Use 'mozart errors {job.job_id}' for complete error history[/dim]"
        )

    # Last activity timestamp (from sheet progress_snapshots or last_activity_at)
    last_activity = _get_last_activity_time(job)
    if last_activity:
        console.print("\n[bold]Last Activity[/bold]")
        console.print(f"  {last_activity.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Circuit breaker state indicator (based on failure patterns)
    # Note: The actual CircuitBreaker object is runtime-only and not persisted.
    # We can infer the likely state from consecutive failures in the state.
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


def _collect_recent_errors(
    job: CheckpointState,
    limit: int = 3,
) -> list[tuple[int, Any]]:
    """Collect the most recent errors from sheet states.

    Args:
        job: CheckpointState to collect errors from.
        limit: Maximum number of errors to return.

    Returns:
        List of (sheet_num, ErrorRecord) tuples, sorted by timestamp descending.
    """
    from mozart.core.checkpoint import ErrorRecord

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


def _get_last_activity_time(job: CheckpointState) -> Any | None:
    """Get the most recent activity timestamp from the job.

    Checks sheet last_activity_at fields and updated_at.

    Args:
        job: CheckpointState to check.

    Returns:
        datetime of last activity, or None if not available.
    """
    from datetime import datetime

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
            "reason": f"≥{threshold} consecutive failures detected",
        }
    elif consecutive_failures > 0:
        return {
            "state": "closed",
            "consecutive_failures": consecutive_failures,
            "reason": f"Under threshold ({consecutive_failures}/{threshold})",
        }

    return None


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string."""
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


@app.command()
def validate(
    config_file: Path = typer.Argument(
        ...,
        help="Path to YAML job configuration file",
        exists=True,
        readable=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output validation results as JSON",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed validation output",
    ),
) -> None:
    """Validate a job configuration file.

    Performs comprehensive validation including:
    - YAML syntax and Pydantic schema validation
    - Jinja template syntax checking
    - Path existence verification
    - Regex pattern compilation
    - Configuration completeness checks

    Exit codes:
      0: Valid (warnings/info OK)
      1: Invalid (one or more errors)
      2: Cannot validate (file not found, YAML unparseable)
    """
    import yaml

    from mozart.core.config import JobConfig
    from mozart.validation import (
        ValidationReporter,
        ValidationRunner,
        ValidationSeverity,
        create_default_checks,
    )

    # First try to read and parse YAML
    try:
        raw_yaml = config_file.read_text()
    except Exception as e:
        if json_output:
            console.print('{"valid": false, "error": "Cannot read file: ' + str(e) + '"}')
        else:
            console.print(f"[red]Cannot read config file:[/red] {e}")
        raise typer.Exit(2) from None

    # Try to parse YAML
    try:
        yaml.safe_load(raw_yaml)
    except yaml.YAMLError as e:
        if json_output:
            console.print('{"valid": false, "error": "YAML syntax error: ' + str(e) + '"}')
        else:
            console.print(f"[red]YAML syntax error:[/red] {e}")
        raise typer.Exit(2) from None

    # Try Pydantic validation
    try:
        config = JobConfig.from_yaml(config_file)
    except Exception as e:
        if json_output:
            console.print('{"valid": false, "error": "Schema validation failed: ' + str(e) + '"}')
        else:
            console.print(f"[red]Schema validation failed:[/red] {e}")
        raise typer.Exit(2) from None

    # Show basic info first
    if not json_output:
        console.print(f"\nValidating [cyan]{config.name}[/cyan]...")
        console.print()
        console.print("[green]✓[/green] YAML syntax valid")
        console.print("[green]✓[/green] Schema validation passed (Pydantic)")
        console.print()
        console.print("Running extended validation checks...")

    # Run extended validation checks
    runner = ValidationRunner(create_default_checks())
    issues = runner.validate(config, config_file, raw_yaml)

    # Output results
    reporter = ValidationReporter(console)

    if json_output:
        console.print(reporter.report_json(issues))
    else:
        reporter.report_terminal(issues, config.name)

        # Show config summary if no errors
        if not runner.has_errors(issues):
            console.print()
            console.print("[dim]Configuration summary:[/dim]")
            console.print(f"  Sheets: {config.sheet.total_sheets}")
            console.print(f"  Backend: {config.backend.type}")
            console.print(f"  Validations: {len(config.validations)}")
            console.print(f"  Notifications: {len(config.notifications)}")

    # Exit with appropriate code
    exit_code = runner.get_exit_code(issues)
    if exit_code != 0:
        raise typer.Exit(exit_code) from None


@app.command()
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
    import gzip
    import json as json_module
    import time

    from mozart.core.logging import find_log_files, get_default_log_path

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
        parts = []
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
        is_gzip = path.suffix == ".gz"
        all_lines: list[str] = []

        try:
            if is_gzip:
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


@app.command()
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
    import json as json_module

    from mozart.core.checkpoint import ErrorRecord

    # Find job state
    found_job, _backend = await _find_job_state(job_id, workspace)
    if found_job is None:
        if json_output:
            console.print(json_module.dumps({"error": f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}"}, indent=2))
        else:
            console.print(f"[red]{ErrorMessages.JOB_NOT_FOUND}:[/red] {job_id}")
            console.print(
                "\n[dim]Hint: Use --workspace to specify the directory "
                "containing the job state.[/dim]"
            )
        raise typer.Exit(1)

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
                from mozart.core.checkpoint import ErrorRecord as ErrRec

                synthetic_error = ErrRec(
                    error_type=_infer_error_type(sheet_state.error_category),
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

    # Color mapping for error types
    type_styles = {
        "permanent": "red bold",
        "transient": "yellow",
        "rate_limit": "blue",
    }

    for sheet_num, error in all_errors:
        # Format timestamp (just time, not date)
        time_str = ""
        if error.timestamp:
            time_str = error.timestamp.strftime("%H:%M:%S")

        # Format error type with color
        type_style = type_styles.get(error.error_type, "white")
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
            console.print(
                Panel(
                    _format_error_details(error),
                    title=f"Sheet {sheet_num} - {error.error_code}",
                    border_style=type_styles.get(error.error_type, "white").split()[0],
                )
            )


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


def _format_error_details(error: Any) -> str:
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
        lines.append(f"[bold]Time:[/bold] {error.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    if error.context:
        context_str = ", ".join(f"{k}={v}" for k, v in error.context.items() if v is not None)
        if context_str:
            lines.append(f"[bold]Context:[/bold] {context_str}")

    if error.stdout_tail:
        lines.append(f"\n[bold]Stdout (tail):[/bold]\n[dim]{error.stdout_tail[:500]}[/dim]")

    if error.stderr_tail:
        lines.append(f"\n[bold]Stderr (tail):[/bold]\n[red dim]{error.stderr_tail[:500]}[/red dim]")

    if error.stack_trace:
        lines.append(f"\n[bold]Stack Trace:[/bold]\n[dim]{error.stack_trace[:800]}[/dim]")

    return "\n".join(lines)


async def _find_job_state(
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
        except Exception:
            continue

    return None, None


@app.command()
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
) -> None:
    """Generate a comprehensive diagnostic report for a job.

    The diagnostic report includes:
    - Job overview and current status
    - Preflight warnings from all sheets
    - Prompt metrics (token counts, line counts)
    - Execution timeline with timing information
    - All errors with full context and output tails

    This command is particularly useful for debugging failed jobs
    or understanding why a job is running slowly.

    Examples:
        mozart diagnose my-job                 # Full diagnostic report
        mozart diagnose my-job --json          # Machine-readable output
        mozart diagnose my-job --workspace .   # Specify workspace
    """
    asyncio.run(_diagnose_job(job_id, workspace, json_output))


async def _diagnose_job(
    job_id: str,
    workspace: Path | None,
    json_output: bool,
) -> None:
    """Asynchronously generate diagnostic report for a job."""
    import json as json_module

    # Find job state
    found_job, _backend = await _find_job_state(job_id, workspace)
    if found_job is None:
        if json_output:
            console.print(json_module.dumps({"error": f"{ErrorMessages.JOB_NOT_FOUND}: {job_id}"}, indent=2))
        else:
            console.print(f"[red]{ErrorMessages.JOB_NOT_FOUND}:[/red] {job_id}")
        raise typer.Exit(1)

    # Build diagnostic report
    report: dict[str, Any] = _build_diagnostic_report(found_job)

    if json_output:
        console.print(json_module.dumps(report, indent=2, default=str))
        return

    # Display formatted report
    _display_diagnostic_report(found_job, report)


def _build_diagnostic_report(job: CheckpointState) -> dict[str, Any]:
    """Build comprehensive diagnostic report from job state.

    Args:
        job: CheckpointState to analyze.

    Returns:
        Dictionary with diagnostic information.
    """
    from datetime import UTC, datetime

    report: dict[str, Any] = {
        "job_id": job.job_id,
        "job_name": job.job_name,
        "status": job.status.value,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    # Progress summary
    completed_count = sum(
        1 for b in job.sheets.values() if b.status == SheetStatus.COMPLETED
    )
    failed_count = sum(
        1 for b in job.sheets.values() if b.status == SheetStatus.FAILED
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

    # Timing
    report["timing"] = {
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    # Calculate duration
    if job.started_at:
        end_time = job.completed_at or datetime.now(UTC)
        duration = (end_time - job.started_at).total_seconds()
        report["timing"]["duration_seconds"] = round(duration, 2)

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
                "error_type": _infer_error_type(sheet.error_category),
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

    return report


def _display_diagnostic_report(job: CheckpointState, report: dict[str, Any]) -> None:
    """Display formatted diagnostic report.

    Args:
        job: Job state for additional context.
        report: Diagnostic report dictionary.
    """
    from datetime import UTC, datetime

    # Status colors
    status_colors = {
        JobStatus.PENDING: "yellow",
        JobStatus.RUNNING: "blue",
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.PAUSED: "magenta",
        JobStatus.CANCELLED: "dim",
    }
    status_color = status_colors.get(job.status, "white")

    # Header panel
    header_lines = [
        f"[bold]{job.job_name}[/bold]",
        f"ID: {job.job_id}",
        f"Status: [{status_color}]{job.status.value.upper()}[/{status_color}]",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    console.print(Panel("\n".join(header_lines), title="Diagnostic Report", border_style="cyan"))

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
        console.print(f"  Duration: {_format_duration(timing['duration_seconds'])}")
        if timing.get("started_at"):
            console.print(f"  Started: {timing['started_at'][:19]}")
        if timing.get("completed_at"):
            console.print(f"  Completed: {timing['completed_at'][:19]}")

    # Preflight warnings
    warnings = report.get("preflight_warnings", [])
    if warnings:
        console.print(f"\n[bold yellow]Preflight Warnings ({len(warnings)})[/bold yellow]")
        for w in warnings[:10]:  # Limit display
            console.print(f"  [yellow]•[/yellow] Sheet {w['sheet_num']}: {w['warning']}")
        if len(warnings) > 10:
            console.print(f"  [dim]... and {len(warnings) - 10} more[/dim]")

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
        timeline_table = Table(show_header=True, header_style="bold")
        timeline_table.add_column("#", justify="right", width=4)
        timeline_table.add_column("Status", width=12)
        timeline_table.add_column("Duration", justify="right", width=10)
        timeline_table.add_column("Attempts", justify="right", width=8)
        timeline_table.add_column("Mode", width=12)
        timeline_table.add_column("Outcome", width=18)

        sheet_status_colors = {
            "pending": "yellow",
            "in_progress": "blue",
            "completed": "green",
            "failed": "red",
            "skipped": "dim",
        }

        for entry in timeline[:20]:  # Limit display
            status = entry.get("status", "unknown")
            status_style = sheet_status_colors.get(status, "white")

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
    if stats.get("total_retry_count", 0) > 0 or stats.get("rate_limit_waits", 0) > 0:
        console.print("\n[bold cyan]Execution Statistics[/bold cyan]")
        if stats.get("total_retry_count", 0) > 0:
            console.print(f"  Total Retries: {stats['total_retry_count']}")
        if stats.get("rate_limit_waits", 0) > 0:
            console.print(f"  Rate Limit Waits: [yellow]{stats['rate_limit_waits']}[/yellow]")

    # Errors section
    errors = report.get("errors", [])
    if errors:
        console.print(f"\n[bold red]Errors ({len(errors)})[/bold red]")

        error_table = Table(show_header=True, header_style="bold")
        error_table.add_column("Sheet", justify="right", width=5)
        error_table.add_column("Type", width=10)
        error_table.add_column("Code", width=6)
        error_table.add_column("Message", no_wrap=False)

        type_styles = {
            "permanent": "red",
            "transient": "yellow",
            "rate_limit": "blue",
        }

        for err in errors[:15]:  # Limit display
            err_type = err.get("error_type", "unknown")
            type_style = type_styles.get(err_type, "white")

            message = err.get("error_message", "")[:80]
            if len(err.get("error_message", "")) > 80:
                message += "..."

            error_table.add_row(
                str(err.get("sheet_num", "")),
                f"[{type_style}]{err_type}[/{type_style}]",
                err.get("error_code", ""),
                message,
            )

        console.print(error_table)

        if len(errors) > 15:
            console.print(f"[dim]... and {len(errors) - 15} more errors[/dim]")
        console.print(
            "\n[dim]Use 'mozart errors " + job.job_id + " --verbose' for full error details[/dim]"
        )

    # Job-level error
    if report.get("job_error"):
        console.print(f"\n[bold red]Job Error:[/bold red] {report['job_error']}")


@app.command()
def dashboard(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run dashboard on"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to"),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory for job state (defaults to current directory)",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development",
    ),
) -> None:
    """Start the web dashboard.

    Launches the Mozart dashboard API server for job monitoring and control.
    The API provides endpoints for listing, viewing, and managing jobs.

    Examples:
        mozart dashboard                    # Start on localhost:8000
        mozart dashboard --port 3000        # Custom port
        mozart dashboard --host 0.0.0.0     # Allow external connections
        mozart dashboard --workspace ./jobs # Use specific state directory
    """
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Error:[/red] uvicorn is required for the dashboard.\n"
            "Install it with: pip install uvicorn"
        )
        raise typer.Exit(1) from None

    from mozart.dashboard import create_app
    from mozart.state import JsonStateBackend, SQLiteStateBackend
    from mozart.state.base import StateBackend

    # Determine state directory
    state_dir = workspace or Path.cwd()

    # Create state backend (prefer SQLite if exists, otherwise JSON)
    state_backend: StateBackend
    sqlite_path = state_dir / ".mozart-state.db"
    if sqlite_path.exists():
        state_backend = SQLiteStateBackend(sqlite_path)
        console.print(f"[dim]Using SQLite state backend: {sqlite_path}[/dim]")
    else:
        state_backend = JsonStateBackend(state_dir)
        console.print(f"[dim]Using JSON state backend: {state_dir}[/dim]")

    # Create the FastAPI app
    fastapi_app = create_app(
        state_backend=state_backend,
        title="Mozart Dashboard",
        version=__version__,
    )

    # Display startup info
    console.print(
        Panel(
            f"[bold]Mozart Dashboard[/bold]\n\n"
            f"API: http://{host}:{port}\n"
            f"Docs: http://{host}:{port}/docs\n"
            f"OpenAPI: http://{host}:{port}/openapi.json\n\n"
            f"[dim]Press Ctrl+C to stop[/dim]",
            title="Starting Server",
        )
    )

    # Run the server
    try:
        uvicorn.run(
            fastapi_app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")


# =============================================================================
# Global Learning Commands (Movement IV-B)
# =============================================================================


@app.command()
def patterns(
    global_patterns: bool = typer.Option(
        True,
        "--global/--local",
        "-g/-l",
        help="Show global patterns (default) or local workspace patterns",
    ),
    min_priority: float = typer.Option(
        0.0,
        "--min-priority",
        "-p",
        help="Minimum priority score to display (0.0-1.0)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of patterns to display",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """View global learning patterns.

    Displays patterns learned from job executions across all workspaces.
    These patterns inform retry strategies, wait times, and validation.

    Examples:
        mozart patterns                  # Show global patterns
        mozart patterns --min-priority 0.5  # Only high-priority patterns
        mozart patterns --json           # JSON output for scripting
        mozart patterns --local          # Local workspace patterns only
    """
    from mozart.learning.global_store import get_global_store

    store = get_global_store()

    if json_output:
        patterns_list = store.get_patterns(min_priority=min_priority)[:limit]
        import json
        output = []
        for p in patterns_list:
            output.append({
                "id": p.id,
                "pattern_type": p.pattern_type,
                "pattern_name": p.pattern_name,
                "description": p.description,
                "occurrence_count": p.occurrence_count,
                "effectiveness_score": round(p.effectiveness_score, 3),
                "priority_score": round(p.priority_score, 3),
                "context_tags": list(p.context_tags) if p.context_tags else [],
            })
        console.print(json.dumps(output, indent=2))
        return

    # Get patterns from global store
    patterns_list = store.get_patterns(min_priority=min_priority)[:limit]

    if not patterns_list:
        console.print("[dim]No patterns found in global learning store.[/dim]")
        console.print(
            "\n[dim]Hint: Patterns are learned from job executions. "
            "Run jobs with learning enabled to build patterns.[/dim]"
        )
        return

    # Display patterns table
    table = Table(title="Global Learning Patterns")
    table.add_column("ID", style="cyan", no_wrap=True, width=10)
    table.add_column("Type", style="yellow", width=15)
    table.add_column("Name", style="bold", width=25)
    table.add_column("Count", justify="right", width=6)
    table.add_column("Effect", justify="right", width=8)
    table.add_column("Priority", justify="right", width=8)
    table.add_column("Tags", style="dim", width=20)

    for p in patterns_list:
        # Format effectiveness with color
        eff = p.effectiveness_score
        eff_color = "green" if eff > 0.7 else "yellow" if eff > 0.4 else "red"
        eff_str = f"[{eff_color}]{eff:.2f}[/{eff_color}]"

        # Format priority with color
        pri = p.priority_score
        pri_color = "green" if pri > 0.7 else "yellow" if pri > 0.4 else "dim"
        pri_str = f"[{pri_color}]{pri:.2f}[/{pri_color}]"

        # Truncate tags
        tags = ", ".join(p.context_tags[:3]) if p.context_tags else ""
        if len(tags) > 20:
            tags = tags[:17] + "..."

        table.add_row(
            p.id[:10],
            p.pattern_type,
            p.pattern_name[:25],
            str(p.occurrence_count),
            eff_str,
            pri_str,
            tags,
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(patterns_list)} pattern(s)[/dim]")


@app.command("aggregate-patterns")
def aggregate_patterns(
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Specific workspace to aggregate from (defaults to all discovered)",
    ),
    additional_paths: list[Path] = typer.Option(
        [],
        "--path",
        "-p",
        help="Additional paths to scan for outcomes",
    ),
) -> None:
    """Aggregate patterns from workspace-local outcomes to global store.

    Scans workspaces for .mozart-outcomes.json files and imports them
    into the global learning store, then runs pattern detection.

    Examples:
        mozart aggregate-patterns                  # Scan all common locations
        mozart aggregate-patterns -w ./workspace  # Specific workspace
        mozart aggregate-patterns -p /path/to/outcomes.json  # Additional path
    """
    from mozart.learning.global_store import get_global_store
    from mozart.learning.migration import OutcomeMigrator
    from mozart.learning.aggregator import PatternAggregator

    store = get_global_store()
    # Wire up pattern aggregator for pattern detection after migration
    aggregator = PatternAggregator(store)
    migrator = OutcomeMigrator(store, aggregator=aggregator)

    console.print("[bold]Aggregating patterns from workspaces...[/bold]\n")

    if workspace:
        # Migrate specific workspace
        result = migrator.migrate_workspace(workspace)
    else:
        # Migrate all discovered workspaces
        result = migrator.migrate_all(additional_paths=list(additional_paths) if additional_paths else None)

    # Display results
    console.print(Panel(
        f"Workspaces found: [cyan]{result.workspaces_found}[/cyan]\n"
        f"Outcomes imported: [green]{result.outcomes_imported}[/green]\n"
        f"Patterns detected: [yellow]{result.patterns_detected}[/yellow]\n"
        f"Workspaces imported: [green]{len(result.imported_workspaces)}[/green]\n"
        f"Workspaces skipped: [dim]{len(result.skipped_workspaces)}[/dim]",
        title="Aggregation Complete",
        border_style="green" if result.outcomes_imported > 0 else "yellow",
    ))

    if result.errors:
        console.print("\n[yellow]Warnings:[/yellow]")
        for error in result.errors[:5]:
            console.print(f"  [dim]• {error}[/dim]")

    if result.imported_workspaces:
        console.print("\n[dim]Imported from:[/dim]")
        for ws in result.imported_workspaces[:5]:
            console.print(f"  [cyan]• {ws}[/cyan]")


@app.command("learning-stats")
def learning_stats(
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """View global learning statistics.

    Shows summary statistics about the global learning store including
    execution counts, pattern counts, and effectiveness metrics.

    Examples:
        mozart learning-stats         # Human-readable summary
        mozart learning-stats --json  # JSON output for scripting
    """
    from mozart.learning.global_store import get_global_store
    from mozart.learning.migration import check_migration_status

    store = get_global_store()
    stats = store.get_execution_stats()
    migration = check_migration_status(store)

    if json_output:
        import json
        output = {
            "executions": {
                "total": stats.get("total_executions", 0),
                "first_attempt_success_rate": round(stats.get("first_attempt_success_rate", 0) * 100, 1),
            },
            "patterns": {
                "total": stats.get("total_patterns", 0),
                "avg_effectiveness": round(stats.get("avg_pattern_effectiveness", 0), 3),
            },
            "workspaces": {
                "unique": stats.get("unique_workspaces", 0),
            },
            "error_recoveries": {
                "total": stats.get("total_error_recoveries", 0),
                "success_rate": round(stats.get("error_recovery_success_rate", 0) * 100, 1),
            },
            "migration_needed": migration.get("needs_migration", False),
        }
        console.print(json.dumps(output, indent=2))
        return

    # Human-readable output
    console.print("[bold]Global Learning Statistics[/bold]\n")

    # Execution stats
    console.print("[bold cyan]Executions[/bold cyan]")
    console.print(f"  Total recorded: [green]{stats.get('total_executions', 0)}[/green]")
    success_rate = stats.get("first_attempt_success_rate", 0) * 100
    console.print(f"  First-attempt success: [{'green' if success_rate > 70 else 'yellow'}]{success_rate:.1f}%[/]")

    # Pattern stats with data source breakdown
    console.print("\n[bold cyan]Patterns[/bold cyan]")
    console.print(f"  Total learned: [yellow]{stats.get('total_patterns', 0)}[/yellow]")
    avg_eff = stats.get("avg_pattern_effectiveness", 0)
    console.print(f"  Avg effectiveness: [{'green' if avg_eff > 0.6 else 'yellow'}]{avg_eff:.2f}[/]")

    # Count patterns by type for data source visibility
    all_patterns = store.get_patterns(limit=1000)
    output_patterns = sum(1 for p in all_patterns if p.pattern_type == "output_pattern")
    error_code_patterns = sum(1 for p in all_patterns if "error_code" in (p.pattern_name or ""))
    semantic_patterns = sum(1 for p in all_patterns if p.pattern_type == "semantic_failure")

    console.print("\n[bold cyan]Data Sources[/bold cyan]")
    console.print(f"  Output patterns extracted: [cyan]{output_patterns}[/cyan]")
    console.print(f"  Error code patterns: [cyan]{error_code_patterns}[/cyan]")
    console.print(f"  Semantic failure patterns: [cyan]{semantic_patterns}[/cyan]")

    # Workspace coverage
    console.print("\n[bold cyan]Workspaces[/bold cyan]")
    console.print(f"  Unique workspaces: [cyan]{stats.get('unique_workspaces', 0)}[/cyan]")

    # Error recovery stats
    console.print("\n[bold cyan]Error Recovery Learning[/bold cyan]")
    console.print(f"  Recoveries recorded: {stats.get('total_error_recoveries', 0)}")
    recovery_rate = stats.get("error_recovery_success_rate", 0) * 100
    console.print(f"  Recovery success rate: [{'green' if recovery_rate > 70 else 'yellow'}]{recovery_rate:.1f}%[/]")

    # Migration status
    if migration.get("needs_migration"):
        console.print(
            "\n[yellow]⚠ Migration needed:[/yellow] Run 'mozart aggregate-patterns' "
            "to import workspace-local outcomes"
        )


@app.command("learning-insights")
def learning_insights(
    limit: Annotated[int, typer.Option(help="Max patterns to show")] = 10,
    pattern_type: Annotated[str | None, typer.Option(help="Filter by type")] = None,
) -> None:
    """Show actionable insights from learning data.

    Displays patterns extracted from execution history including:
    - Output patterns (from stdout/stderr analysis)
    - Error code patterns (aggregated error statistics)
    - Success predictors (factors that correlate with success)

    Examples:
        mozart learning-insights
        mozart learning-insights --pattern-type output_pattern
        mozart learning-insights --limit 20
    """
    from mozart.learning.global_store import GlobalLearningStore

    console.print("[bold]Learning Insights[/bold]")
    console.print()

    store = GlobalLearningStore()
    patterns = store.get_patterns(
        pattern_type=pattern_type,
        limit=limit,
    )

    if not patterns:
        console.print("[dim]No patterns learned yet. Run some jobs![/dim]")
        return

    # Display table of patterns
    table = Table(title="Learned Patterns")
    table.add_column("Type", style="cyan")
    table.add_column("Description")
    table.add_column("Freq", justify="right")
    table.add_column("Effectiveness", justify="right")

    for p in patterns:
        desc_str = p.description or ""
        desc = desc_str[:45] + "..." if len(desc_str) > 45 else desc_str
        table.add_row(
            p.pattern_type,
            desc or "[no description]",
            str(p.occurrence_count),
            f"{p.effectiveness_score:.0%}" if p.effectiveness_score else "-",
        )

    console.print(table)


@app.command("learning-activity")
def learning_activity(
    hours: int = typer.Option(
        24,
        "--hours",
        "-h",
        help="Show activity from the last N hours",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON for machine parsing",
    ),
) -> None:
    """View recent learning activity and pattern applications.

    Learning Activation: Shows what patterns have been applied, their
    effectiveness, and provides insight into how the learning system
    is improving execution outcomes.

    Examples:
        mozart learning-activity           # Last 24 hours of activity
        mozart learning-activity -h 48     # Last 48 hours
        mozart learning-activity --json    # JSON output
    """
    from datetime import datetime, timedelta

    from mozart.learning.global_store import get_global_store

    store = get_global_store()
    stats = store.get_execution_stats()

    # Get optimal execution window analysis
    window = store.get_optimal_execution_window()

    # Get recent similar executions for activity display
    cutoff = datetime.now() - timedelta(hours=hours)
    recent_executions = store.get_similar_executions(limit=20)
    recent_count = sum(
        1 for e in recent_executions
        if e.completed_at and e.completed_at > cutoff
    )

    if json_output:
        import json
        output = {
            "period_hours": hours,
            "recent_executions": recent_count,
            "first_attempt_success_rate": round(
                stats.get("first_attempt_success_rate", 0) * 100, 1
            ),
            "patterns_active": stats.get("total_patterns", 0),
            "optimal_hours": window.get("optimal_hours", []),
            "avoid_hours": window.get("avoid_hours", []),
            "scheduling_confidence": round(window.get("confidence", 0), 2),
        }
        console.print(json.dumps(output, indent=2))
        return

    # Human-readable output
    console.print(f"[bold]Learning Activity (last {hours} hours)[/bold]\n")

    # Recent activity
    console.print("[bold cyan]Recent Executions[/bold cyan]")
    console.print(f"  Executions in period: [green]{recent_count}[/green]")
    success_rate = stats.get("first_attempt_success_rate", 0) * 100
    console.print(
        f"  First-attempt success: "
        f"[{'green' if success_rate > 70 else 'yellow'}]{success_rate:.1f}%[/]"
    )

    # Pattern application info
    console.print("\n[bold cyan]Pattern Application[/bold cyan]")
    pattern_count = stats.get("total_patterns", 0)
    if pattern_count > 0:
        console.print(f"  Active patterns: [yellow]{pattern_count}[/yellow]")
        avg_eff = stats.get("avg_pattern_effectiveness", 0)
        console.print(
            f"  Avg effectiveness: "
            f"[{'green' if avg_eff > 0.6 else 'yellow'}]{avg_eff:.2f}[/]"
        )
    else:
        console.print("  [dim]No patterns learned yet[/dim]")

    # Time-aware scheduling insights
    console.print("\n[bold cyan]Optimal Execution Windows[/bold cyan]")
    if window.get("confidence", 0) > 0.3:
        optimal = window.get("optimal_hours", [])
        avoid = window.get("avoid_hours", [])

        if optimal:
            optimal_str = ", ".join(f"{h:02d}:00" for h in optimal)
            console.print(f"  [green]✓ Best hours:[/green] {optimal_str}")
        if avoid:
            avoid_str = ", ".join(f"{h:02d}:00" for h in avoid)
            console.print(f"  [red]✗ Avoid hours:[/red] {avoid_str}")

        console.print(
            f"  Confidence: [cyan]{window.get('confidence', 0):.0%}[/cyan] "
            f"(based on {window.get('sample_count', 0)} samples)"
        )
    else:
        console.print("  [dim]Insufficient data for scheduling recommendations[/dim]")

    # Learning status summary
    console.print("\n[bold cyan]Learning Status[/bold cyan]")
    total_executions = stats.get("total_executions", 0)
    if total_executions >= 50:
        console.print("  [green]✓ Learning system is well-trained[/green]")
    elif total_executions >= 10:
        console.print("  [yellow]○ Learning system is gathering data[/yellow]")
    else:
        console.print("  [dim]○ Learning system is in early training[/dim]")


if __name__ == "__main__":
    app()
