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
from mozart.core.checkpoint import BatchStatus, CheckpointState, JobStatus
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

# Get logger for CLI module - will be configured in main()
logger = get_logger("cli")


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
            logger.warning(f"Unknown notification type: {config.type}")
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
    """Set log file path from CLI option."""
    global _log_file
    if value:
        _log_file = value
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
    start_batch: int | None = typer.Option(
        None,
        "--start-batch",
        "-s",
        help="Override starting batch number",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON for machine parsing",
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
            f"Batches: {config.batch.total_batches} "
            f"({config.batch.size} items each)\n"
            f"Workspace: {config.workspace}",
            title="Job Configuration",
        ))

    if dry_run:
        if not json_output:
            console.print("\n[yellow]Dry run - not executing[/yellow]")
            _show_dry_run(config)
        else:
            console.print(json.dumps({
                "dry_run": True,
                "job_name": config.name,
                "total_batches": config.batch.total_batches,
            }, indent=2))
        return

    # Actually run the job
    if not is_quiet() and not json_output:
        console.print("\n[green]Starting job...[/green]")
    asyncio.run(_run_job(config, start_batch, json_output))


async def _run_job(
    config: JobConfig,
    start_batch: int | None,
    json_output: bool = False,
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
        "batch_num": None,
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

    # Create progress bar for batch tracking (skip in quiet/json mode)
    progress: Progress | None = None
    progress_task_id: TaskID | None = None

    if not is_quiet() and not json_output:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed}/{task.total} batches"),
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
        """Update progress bar with current batch progress."""
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

    # Create runner with progress callback
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=state_backend,
        console=console if not json_output else Console(quiet=True),
        outcome_store=outcome_store,
        progress_callback=update_progress if progress else None,
    )

    job_id = config.name  # Use job name as ID for now
    summary: RunSummary | None = None

    try:
        # Send job start notification
        if notification_manager:
            await notification_manager.notify_job_start(
                job_id=job_id,
                job_name=config.name,
                total_batches=config.batch.total_batches,
            )

        # Start progress display
        if progress:
            progress.start()
            starting_batch = start_batch or 1
            initial_completed = starting_batch - 1
            progress_task_id = progress.add_task(
                f"[cyan]{config.name}[/cyan]",
                total=config.batch.total_batches,
                completed=initial_completed,
                eta="calculating...",
                exec_status="",  # Initial empty execution status
            )

        # Run job with validation and completion recovery
        state, summary = await runner.run(start_batch=start_batch)

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
                    success_count=summary.completed_batches,
                    failure_count=summary.failed_batches,
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
                    batch_num=state.current_batch,
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
        "[bold]Batches[/bold]",
        f"  Completed: [green]{summary.completed_batches}[/green]/{summary.total_batches}",
    ]

    if summary.failed_batches > 0:
        lines.append(f"  Failed: [red]{summary.failed_batches}[/red]")
    if summary.skipped_batches > 0:
        lines.append(f"  Skipped: [yellow]{summary.skipped_batches}[/yellow]")

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
                f"({summary.first_attempt_successes}/{summary.completed_batches})"
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
    table = Table(title="Batch Plan")
    table.add_column("Batch", style="cyan")
    table.add_column("Items", style="green")
    table.add_column("Validations", style="yellow")

    for batch_num in range(1, config.batch.total_batches + 1):
        start = (batch_num - 1) * config.batch.size + config.batch.start_item
        end = min(start + config.batch.size - 1, config.batch.total_items)
        table.add_row(
            str(batch_num),
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
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to search for job state",
    ),
) -> None:
    """Show detailed status of a specific job.

    Displays job progress, batch states, timing information, and any errors.
    Use --json for machine-readable output in scripts.
    """
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
) -> None:
    """Resume a paused or failed job.

    Loads the job state from the state backend and continues execution
    from where it left off. The job configuration is reconstructed from
    the stored config_snapshot, or you can provide a config file with --config.

    Examples:
        mozart resume my-job
        mozart resume my-job --config job.yaml
        mozart resume my-job --workspace ./workspace
    """
    asyncio.run(_resume_job(job_id, config_file, workspace, force))


async def _resume_job(
    job_id: str,
    config_file: Path | None,
    workspace: Path | None,
    force: bool,
) -> None:
    """Resume a paused or failed job.

    Args:
        job_id: Job ID to resume.
        config_file: Optional path to config file.
        workspace: Optional workspace directory to search.
        force: Force resume even if job appears completed.
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
            logger.debug(f"Error querying backend for {job_id}: {e}")
            continue

    if found_state is None or found_backend is None:
        console.print(f"[red]Job not found:[/red] {job_id}")
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
                "[dim]Use --force to resume anyway (will restart from last batch).[/dim]"
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

    # Priority 1: Use provided config file
    if config_file:
        try:
            config = JobConfig.from_yaml(config_file)
            console.print(f"[dim]Using config from: {config_file}[/dim]")
        except Exception as e:
            console.print(f"[red]Error loading config file:[/red] {e}")
            raise typer.Exit(1) from None

    # Priority 2: Reconstruct from config_snapshot
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

    # Priority 3: Try to load from stored config_path
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

    # Calculate resume point
    resume_batch = found_state.last_completed_batch + 1
    if resume_batch > found_state.total_batches:
        if force:
            # For force resume, restart from last batch
            resume_batch = found_state.total_batches
            console.print(
                f"[yellow]Job was completed. Force restarting batch {resume_batch}.[/yellow]"
            )
        else:
            console.print("[green]Job is already fully completed.[/green]")
            return

    # Display resume info
    console.print(Panel(
        f"[bold]{config.name}[/bold]\n"
        f"Status: {found_state.status.value}\n"
        f"Progress: {found_state.last_completed_batch}/{found_state.total_batches} batches\n"
        f"Resuming from batch: {resume_batch}",
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

    # Setup notification manager from config
    notification_manager: NotificationManager | None = None
    if config.notifications:
        notifiers = create_notifiers_from_config(config.notifications)
        if notifiers:
            notification_manager = NotificationManager(notifiers)
            console.print(
                f"[dim]Notifications enabled: {len(notifiers)} channel(s) configured[/dim]"
            )

    # Create progress bar for batch tracking
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} batches"),
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
        """Update progress bar with current batch progress."""
        nonlocal progress_task_id
        if progress_task_id is not None:
            eta_str = _format_duration(eta_seconds) if eta_seconds else "calculating..."
            progress.update(
                progress_task_id,
                completed=completed,
                total=total,
                eta=eta_str,
            )

    # Create runner with progress callback
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=found_backend,
        console=console,
        outcome_store=outcome_store,
        progress_callback=update_progress,
    )

    try:
        # Send job resume notification (use job_start event)
        if notification_manager:
            remaining_batches = found_state.total_batches - found_state.last_completed_batch
            await notification_manager.notify_job_start(
                job_id=job_id,
                job_name=config.name,
                total_batches=remaining_batches,
            )

        # Start progress display
        progress.start()
        progress_task_id = progress.add_task(
            f"[cyan]{config.name}[/cyan] (resuming)",
            total=found_state.total_batches,
            completed=found_state.last_completed_batch,
            eta="calculating...",
        )

        # Resume from the next batch
        if not is_quiet():
            console.print(f"\n[green]Resuming from batch {resume_batch}...[/green]")
        state, summary = await runner.run(
            start_batch=resume_batch,
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
                    success_count=summary.completed_batches,
                    failure_count=summary.failed_batches,
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
                    batch_num=state.current_batch,
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
            logger.debug(f"Error querying backend {source}: {e}")
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
        progress = f"{job.last_completed_batch}/{job.total_batches}"

        # Format updated time
        updated = job.updated_at.strftime("%Y-%m-%d %H:%M") if job.updated_at else "-"

        table.add_row(job.job_id, status_str, progress, updated)

    console.print(table)
    console.print(f"\n[dim]Showing {len(unique_jobs)} job(s)[/dim]")


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
            logger.debug(f"Error querying backend for {job_id}: {e}")
            continue

    if not found_job:
        if json_output:
            console.print(json.dumps({"error": f"Job not found: {job_id}"}, indent=2))
        else:
            console.print(f"[red]Job not found:[/red] {job_id}")
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
    # Use last_completed_batch for progress since it's more reliable than counting batches dict
    completed = job.last_completed_batch
    total = job.total_batches
    percent = (completed / total * 100) if total > 0 else 0.0

    # Collect recent errors for JSON output
    recent_errors_data: list[dict[str, Any]] = []
    for batch_num, error in _collect_recent_errors_for_json(job, limit=5):
        recent_errors_data.append({
            "batch_num": batch_num,
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
            "current_batch": job.current_batch,
            "total_retry_count": job.total_retry_count,
            "rate_limit_waits": job.rate_limit_waits,
        },
        "circuit_breaker": cb_state,
        "recent_errors": recent_errors_data,
        "error": job.error_message,
        "batches": {
            str(num): {
                "status": batch.status.value,
                "attempt_count": batch.attempt_count,
                "validation_passed": batch.validation_passed,
                "error_message": batch.error_message,
                "error_category": batch.error_category,
            }
            for num, batch in job.batches.items()
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
        List of (batch_num, error) tuples, sorted by timestamp descending.
    """
    from mozart.core.checkpoint import ErrorRecord

    all_errors: list[tuple[int, ErrorRecord]] = []

    for batch_num, batch in job.batches.items():
        # Collect from error_history
        for error in batch.error_history:
            all_errors.append((batch_num, error))

        # If no history but has error_message, create synthetic record
        if not batch.error_history and batch.error_message:
            synthetic = ErrorRecord(
                error_type=_infer_error_type(batch.error_category),
                error_code=batch.error_category or "E999",
                error_message=batch.error_message,
                attempt_number=batch.attempt_count,
                context={
                    "exit_code": batch.exit_code,
                    "exit_signal": batch.exit_signal,
                },
            )
            if batch.completed_at:
                synthetic.timestamp = batch.completed_at
            all_errors.append((batch_num, synthetic))

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

    # Progress bar - use last_completed_batch for consistency with JSON output
    completed = job.last_completed_batch
    total = job.total_batches

    console.print("\n[bold]Progress[/bold]")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
        transient=False,
    ) as progress:
        progress.add_task("Batches", total=total, completed=completed)
        # Force refresh to display
        progress.refresh()

    # Batch details table
    if job.batches:
        console.print("\n[bold]Batch Details[/bold]")
        batch_table = Table(show_header=True, header_style="bold")
        batch_table.add_column("#", justify="right", style="cyan", width=4)
        batch_table.add_column("Status", width=12)
        batch_table.add_column("Attempts", justify="right", width=8)
        batch_table.add_column("Validation", width=10)
        batch_table.add_column("Error", style="dim", no_wrap=False)

        batch_status_colors = {
            BatchStatus.PENDING: "yellow",
            BatchStatus.IN_PROGRESS: "blue",
            BatchStatus.COMPLETED: "green",
            BatchStatus.FAILED: "red",
            BatchStatus.SKIPPED: "dim",
        }

        for batch_num in sorted(job.batches.keys()):
            batch = job.batches[batch_num]
            batch_color = batch_status_colors.get(batch.status, "white")

            # Format validation status
            if batch.validation_passed is None:
                val_str = "-"
            elif batch.validation_passed:
                val_str = "[green]✓ Pass[/green]"
            else:
                val_str = "[red]✗ Fail[/red]"

            # Truncate error message for table
            error_str = ""
            if batch.error_message:
                error_str = batch.error_message[:50]
                if len(batch.error_message) > 50:
                    error_str += "..."

            batch_table.add_row(
                str(batch_num),
                f"[{batch_color}]{batch.status.value}[/{batch_color}]",
                str(batch.attempt_count),
                val_str,
                error_str,
            )

        console.print(batch_table)

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
    if job.total_retry_count > 0 or job.rate_limit_waits > 0:
        console.print("\n[bold]Execution Stats[/bold]")
        console.print(f"  Total retries: {job.total_retry_count}")
        console.print(f"  Rate limit waits: {job.rate_limit_waits}")

    # Recent errors section - show last 3 errors from any batch
    recent_errors = _collect_recent_errors(job, limit=3)
    if recent_errors:
        console.print("\n[bold red]Recent Errors[/bold red]")
        for batch_num, error in recent_errors:
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
                f"  [{type_style}]•[/{type_style}] Batch {batch_num}: "
                f"[{type_style}]{error.error_code}[/{type_style}] - {message}"
            )

        # Hint for more details
        console.print(
            f"\n[dim]  Use 'mozart errors {job.job_id}' for complete error history[/dim]"
        )

    # Last activity timestamp (from batch progress_snapshots or last_activity_at)
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
    """Collect the most recent errors from batch states.

    Args:
        job: CheckpointState to collect errors from.
        limit: Maximum number of errors to return.

    Returns:
        List of (batch_num, ErrorRecord) tuples, sorted by timestamp descending.
    """
    from mozart.core.checkpoint import ErrorRecord

    all_errors: list[tuple[int, ErrorRecord]] = []

    for batch_num, batch in job.batches.items():
        # Collect from error_history
        for error in batch.error_history:
            all_errors.append((batch_num, error))

        # If no history but has error_message, create synthetic record
        if not batch.error_history and batch.error_message:
            synthetic = ErrorRecord(
                error_type=_infer_error_type(batch.error_category),
                error_code=batch.error_category or "E999",
                error_message=batch.error_message,
                attempt_number=batch.attempt_count,
                context={
                    "exit_code": batch.exit_code,
                    "exit_signal": batch.exit_signal,
                },
            )
            if batch.completed_at:
                synthetic.timestamp = batch.completed_at
            all_errors.append((batch_num, synthetic))

    # Sort by timestamp (most recent first) and take limit
    all_errors.sort(key=lambda x: x[1].timestamp, reverse=True)
    return all_errors[:limit]


def _get_last_activity_time(job: CheckpointState) -> Any | None:
    """Get the most recent activity timestamp from the job.

    Checks batch last_activity_at fields and updated_at.

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

    # Check batch-level last_activity_at
    for batch in job.batches.values():
        if batch.last_activity_at:
            candidates.append(batch.last_activity_at)

    if candidates:
        return max(candidates)
    return None


def _infer_circuit_breaker_state(job: CheckpointState) -> dict[str, Any] | None:
    """Infer likely circuit breaker state from job state.

    The actual CircuitBreaker is a runtime object and not persisted.
    We can infer the likely state based on failure patterns:
    - If last N batches all failed -> likely OPEN
    - If mix of success/failure -> likely CLOSED
    - If recovering from failures -> likely HALF_OPEN

    Args:
        job: CheckpointState to analyze.

    Returns:
        Dict with inferred state info, or None if no relevant data.
    """
    if not job.batches:
        return None

    # Count consecutive failures from the end
    sorted_batches = sorted(job.batches.items(), key=lambda x: x[0], reverse=True)
    consecutive_failures = 0

    for _batch_num, batch in sorted_batches:
        if batch.status == BatchStatus.FAILED:
            consecutive_failures += 1
        elif batch.status == BatchStatus.COMPLETED:
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
) -> None:
    """Validate a job configuration file."""
    from mozart.core.config import JobConfig

    try:
        config = JobConfig.from_yaml(config_file)
        console.print(f"[green]Valid configuration:[/green] {config.name}")
        console.print(f"  Batches: {config.batch.total_batches}")
        console.print(f"  Backend: {config.backend.type}")
        console.print(f"  Validations: {len(config.validations)}")
        console.print(f"  Notifications: {len(config.notifications)}")
    except Exception as e:
        console.print(f"[red]Invalid configuration:[/red] {e}")
        raise typer.Exit(1) from None


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
        batch_num = entry.get("batch_num")

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

        if batch_num is not None:
            parts.append(f"[green]batch:{batch_num}[/green]")

        parts.append(event)

        # Add extra context fields
        exclude_keys = {
            "timestamp", "level", "event", "component",
            "job_id", "batch_num", "run_id", "parent_run_id", "_raw",
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
    batch: int | None = typer.Option(
        None,
        "--batch",
        "-b",
        help="Filter errors by specific batch number",
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

    Displays errors grouped by batch, with color-coding by error type:
    - Red: Permanent errors (non-retriable, fatal)
    - Yellow: Transient errors (retriable with backoff)
    - Blue: Rate limit errors (retriable after wait)

    Examples:
        mozart errors my-job                   # Show all errors
        mozart errors my-job --batch 3         # Errors for batch 3 only
        mozart errors my-job --type transient  # Only transient errors
        mozart errors my-job --code E001       # Only timeout errors
        mozart errors my-job --verbose         # Show stdout/stderr details
    """
    asyncio.run(_errors_job(job_id, batch, error_type, error_code, verbose, workspace, json_output))


async def _errors_job(
    job_id: str,
    batch_filter: int | None,
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
            console.print(json_module.dumps({"error": f"Job not found: {job_id}"}, indent=2))
        else:
            console.print(f"[red]Job not found:[/red] {job_id}")
            console.print(
                "\n[dim]Hint: Use --workspace to specify the directory "
                "containing the job state.[/dim]"
            )
        raise typer.Exit(1)

    # Collect all errors from batch states
    all_errors: list[tuple[int, ErrorRecord]] = []

    for batch_num, batch_state in found_job.batches.items():
        # Apply batch filter if specified
        if batch_filter is not None and batch_num != batch_filter:
            continue

        # Collect from error_history field
        for error in batch_state.error_history:
            # Apply type filter
            if error_type_filter is not None and error.error_type != error_type_filter:
                continue
            # Apply code filter
            if error_code_filter is not None and error.error_code != error_code_filter:
                continue
            all_errors.append((batch_num, error))

    # If no errors in history, check for error_message on failed batches
    if not all_errors:
        for batch_num, batch_state in found_job.batches.items():
            if batch_filter is not None and batch_num != batch_filter:
                continue

            if batch_state.error_message:
                # Create a synthetic ErrorRecord from batch error_message
                # This handles older state files that don't have error_history populated
                from mozart.core.checkpoint import ErrorRecord as ErrRec

                synthetic_error = ErrRec(
                    error_type=_infer_error_type(batch_state.error_category),
                    error_code=batch_state.error_category or "E999",
                    error_message=batch_state.error_message,
                    attempt_number=batch_state.attempt_count,
                    stdout_tail=batch_state.stdout_tail,
                    stderr_tail=batch_state.stderr_tail,
                    context={
                        "exit_code": batch_state.exit_code,
                        "exit_signal": batch_state.exit_signal,
                        "exit_reason": batch_state.exit_reason,
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
                all_errors.append((batch_num, synthetic_error))

    # Sort by batch number, then timestamp
    all_errors.sort(key=lambda x: (x[0], x[1].timestamp))

    # Output as JSON if requested
    if json_output:
        output: dict[str, Any] = {
            "job_id": job_id,
            "total_errors": len(all_errors),
            "errors": [
                {
                    "batch_num": batch_num,
                    "timestamp": error.timestamp.isoformat() if error.timestamp else None,
                    "error_type": error.error_type,
                    "error_code": error.error_code,
                    "error_message": error.error_message,
                    "attempt_number": error.attempt_number,
                    "context": error.context,
                    "stdout_tail": error.stdout_tail if verbose else None,
                    "stderr_tail": error.stderr_tail if verbose else None,
                }
                for batch_num, error in all_errors
            ],
        }
        console.print(json_module.dumps(output, indent=2, default=str))
        return

    # Display with Rich table
    if not all_errors:
        console.print(f"[green]No errors found for job:[/green] {job_id}")
        if batch_filter is not None:
            console.print(f"[dim]Batch filter: {batch_filter}[/dim]")
        if error_type_filter is not None:
            console.print(f"[dim]Type filter: {error_type_filter}[/dim]")
        if error_code_filter is not None:
            console.print(f"[dim]Code filter: {error_code_filter}[/dim]")
        return

    # Build errors table
    table = Table(title=f"Errors for Job: {job_id}")
    table.add_column("Batch", justify="right", style="cyan", width=6)
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

    for batch_num, error in all_errors:
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
            str(batch_num),
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
        for batch_num, error in all_errors:
            console.print(
                Panel(
                    _format_error_details(error),
                    title=f"Batch {batch_num} - {error.error_code}",
                    border_style=type_styles.get(error.error_type, "white").split()[0],
                )
            )


def _infer_error_type(
    error_category: str | None,
) -> Literal["transient", "rate_limit", "permanent"]:
    """Infer error type from error category string.

    Args:
        error_category: Error category from batch state.

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
    - Preflight warnings from all batches
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
            console.print(json_module.dumps({"error": f"Job not found: {job_id}"}, indent=2))
        else:
            console.print(f"[red]Job not found:[/red] {job_id}")
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
        1 for b in job.batches.values() if b.status == BatchStatus.COMPLETED
    )
    failed_count = sum(
        1 for b in job.batches.values() if b.status == BatchStatus.FAILED
    )
    report["progress"] = {
        "total_batches": job.total_batches,
        "completed": completed_count,
        "failed": failed_count,
        "last_completed": job.last_completed_batch,
        "percent": (
            round(job.last_completed_batch / job.total_batches * 100, 1)
            if job.total_batches > 0 else 0
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

    # Collect preflight warnings across all batches
    all_warnings: list[dict[str, Any]] = []
    for batch_num, batch in job.batches.items():
        for warning in batch.preflight_warnings:
            all_warnings.append({
                "batch_num": batch_num,
                "warning": warning,
            })
    report["preflight_warnings"] = all_warnings

    # Collect prompt metrics from all batches
    prompt_metrics: list[dict[str, Any]] = []
    for batch_num, batch in job.batches.items():
        if batch.prompt_metrics:
            prompt_metrics.append({
                "batch_num": batch_num,
                **batch.prompt_metrics,
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
    for batch_num in sorted(job.batches.keys()):
        batch = job.batches[batch_num]
        entry = {
            "batch_num": batch_num,
            "status": batch.status.value,
            "started_at": batch.started_at.isoformat() if batch.started_at else None,
            "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
            "duration_seconds": batch.execution_duration_seconds,
            "attempt_count": batch.attempt_count,
            "completion_attempts": batch.completion_attempts,
            "execution_mode": batch.execution_mode,
            "outcome_category": batch.outcome_category,
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
    for batch_num, batch in job.batches.items():
        for error in batch.error_history:
            all_errors.append({
                "batch_num": batch_num,
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

        # Add batch-level error if no history exists
        if not batch.error_history and batch.error_message:
            all_errors.append({
                "batch_num": batch_num,
                "timestamp": batch.completed_at.isoformat() if batch.completed_at else None,
                "error_type": _infer_error_type(batch.error_category),
                "error_code": batch.error_category or "E999",
                "error_message": batch.error_message,
                "attempt_number": batch.attempt_count,
                "context": {
                    "exit_code": batch.exit_code,
                    "exit_signal": batch.exit_signal,
                    "exit_reason": batch.exit_reason,
                },
                "stdout_tail": batch.stdout_tail,
                "stderr_tail": batch.stderr_tail,
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
        f"  Batches: {progress.get('completed', 0)}/{progress.get('total_batches', 0)} "
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
            console.print(f"  [yellow]•[/yellow] Batch {w['batch_num']}: {w['warning']}")
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

        batch_status_colors = {
            "pending": "yellow",
            "in_progress": "blue",
            "completed": "green",
            "failed": "red",
            "skipped": "dim",
        }

        for entry in timeline[:20]:  # Limit display
            status = entry.get("status", "unknown")
            status_style = batch_status_colors.get(status, "white")

            duration = entry.get("duration_seconds")
            duration_str = f"{duration:.1f}s" if duration else "-"

            attempts = entry.get("attempt_count", 0)
            comp_attempts = entry.get("completion_attempts", 0)
            attempts_str = str(attempts)
            if comp_attempts > 0:
                attempts_str += f"+{comp_attempts}c"

            timeline_table.add_row(
                str(entry.get("batch_num", "")),
                f"[{status_style}]{status}[/{status_style}]",
                duration_str,
                attempts_str,
                entry.get("execution_mode") or "-",
                entry.get("outcome_category") or "-",
            )

        console.print(timeline_table)
        if len(timeline) > 20:
            console.print(f"[dim]... and {len(timeline) - 20} more batches[/dim]")

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
        error_table.add_column("Batch", justify="right", width=5)
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
                str(err.get("batch_num", "")),
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


if __name__ == "__main__":
    app()
