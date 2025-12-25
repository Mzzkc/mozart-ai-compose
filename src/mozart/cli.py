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
import logging
from enum import Enum
from pathlib import Path
from typing import Any

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
from mozart.execution.runner import RunSummary
from mozart.notifications import (
    DesktopNotifier,
    NotificationManager,
    Notifier,
    SlackNotifier,
    WebhookNotifier,
)
from mozart.state import JsonStateBackend, SQLiteStateBackend, StateBackend

logger = logging.getLogger(__name__)


class OutputLevel(str, Enum):
    """Output verbosity level."""

    QUIET = "quiet"  # Minimal output (errors only)
    NORMAL = "normal"  # Default output
    VERBOSE = "verbose"  # Detailed output


# Global output level state
_output_level: OutputLevel = OutputLevel.NORMAL

# Default state directory when no config is available
DEFAULT_STATE_DIR = Path.home() / ".mozart" / "state"


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
) -> None:
    """Mozart AI Compose - Orchestration tool for Claude AI sessions."""
    pass


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
        },
        "execution": {
            "current_batch": job.current_batch,
            "total_retry_count": job.total_retry_count,
            "rate_limit_waits": job.rate_limit_waits,
        },
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
