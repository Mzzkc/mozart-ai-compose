"""Run command for Mozart CLI.

This module implements the `mozart run` command which executes jobs
from YAML configuration files.

★ Insight ─────────────────────────────────────
1. **Callback injection pattern**: Progress callbacks are injected into
   backends at runtime to decouple display logic from execution logic.
   This allows the same backend to work with or without progress display.

2. **Context object pattern**: RunnerContext bundles all optional
   components (outcome store, escalation handler, grounding engine)
   to avoid parameter explosion in JobRunner.__init__.

3. **Graceful shutdown handling**: GracefulShutdownError is caught
   separately from FatalError to allow clean exit with state preserved.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    TaskID,
)
from rich.table import Table

from mozart.core.checkpoint import JobStatus

from ..helpers import (
    create_state_backend_from_config,
    is_quiet,
    is_verbose,
)
from ..output import console, format_duration
from ._shared import (
    create_progress_bar,
    handle_job_completion,
    setup_all,
)

if TYPE_CHECKING:
    from mozart.core.config import JobConfig
    from mozart.execution.runner import RunSummary

_logger = logging.getLogger(__name__)


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
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Override workspace directory. Creates the directory if it doesn't exist. "
        "Takes precedence over the workspace defined in the YAML config.",
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
    self_healing: bool = typer.Option(
        False,
        "--self-healing",
        "-H",
        help="Enable automatic diagnosis and remediation when retries are exhausted",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-confirm suggested fixes when using --self-healing",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Delete existing state before running, ensuring a fresh start. "
        "Use this for self-chaining jobs or when you want to re-run a completed job "
        "from scratch without resuming from previous state.",
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

    # Override workspace from CLI if provided
    if workspace is not None:
        config.workspace = Path(workspace).resolve()

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
                "workspace": str(config.workspace),
            }, indent=2))
        return

    # Actually run the job
    if not is_quiet() and not json_output:
        console.print("\n[green]Starting job...[/green]")
    asyncio.run(_run_job(config, start_sheet, json_output, escalation, self_healing, yes, fresh))


async def _run_job(
    config: JobConfig,
    start_sheet: int | None,
    json_output: bool = False,
    escalation: bool = False,
    self_healing: bool = False,
    auto_confirm: bool = False,
    fresh: bool = False,
) -> None:
    """Run the job asynchronously using the JobRunner with progress display."""
    from mozart.backends.claude_cli import ClaudeCliBackend
    from mozart.execution.runner import FatalError, GracefulShutdownError, JobRunner

    # Ensure workspace exists
    config.workspace.mkdir(parents=True, exist_ok=True)

    # Setup backends based on config
    state_backend = create_state_backend_from_config(config)

    # Delete existing state if --fresh flag is set (clean start)
    if fresh:
        deleted = await state_backend.delete(config.name)
        if deleted and not is_quiet() and not json_output:
            console.print(
                f"[yellow]--fresh: Deleted existing state for '{config.name}'[/yellow]"
            )
        elif not deleted and is_verbose() and not json_output:
            console.print(
                f"[dim]--fresh: No existing state found for '{config.name}'[/dim]"
            )

        # Archive workspace files if configured
        if config.workspace_lifecycle.archive_on_fresh:
            from mozart.workspace.lifecycle import WorkspaceArchiver

            archiver = WorkspaceArchiver(config.workspace, config.workspace_lifecycle)
            archive_path = archiver.archive()
            if archive_path and not is_quiet() and not json_output:
                console.print(
                    f"[yellow]--fresh: Archived workspace to {archive_path.name}[/yellow]"
                )

    quiet = json_output
    components = setup_all(
        config, escalation=escalation, quiet=quiet, console=console,
    )
    backend = components.backend
    outcome_store = components.outcome_store
    global_learning_store = components.global_learning_store
    notification_manager = components.notification_manager
    escalation_handler = components.escalation_handler
    grounding_engine = components.grounding_engine

    # Execution progress state for CLI display (Task 4)
    execution_status: dict[str, Any] = {
        "sheet_num": None,
        "bytes_received": 0,
        "lines_received": 0,
        "elapsed_seconds": 0.0,
        "phase": "idle",
    }

    # Create progress bar for sheet tracking (skip in quiet/json mode)
    progress: Progress | None = None
    progress_task_id: TaskID | None = None

    if not is_quiet() and not json_output:
        progress = create_progress_bar(
            console=console,
            include_exec_status=True,
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
            eta_str = format_duration(eta_seconds) if eta_seconds else "calculating..."
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

    # Create runner context with all optional components
    from mozart.execution.runner import RunnerContext
    runner_context = RunnerContext(
        console=console if not json_output else Console(quiet=True),
        outcome_store=outcome_store,
        escalation_handler=escalation_handler,
        progress_callback=update_progress if progress else None,
        global_learning_store=global_learning_store,
        grounding_engine=grounding_engine,
        self_healing_enabled=self_healing,
        self_healing_auto_confirm=auto_confirm,
    )

    # Create runner with progress callback
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=state_backend,
        context=runner_context,
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
        else:
            await handle_job_completion(
                state=state,
                summary=summary,
                notification_manager=notification_manager,
                job_id=job_id,
                job_name=config.name,
                console=console,
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

        # Send failure notification (must not mask the original error)
        if notification_manager:
            try:
                await notification_manager.notify_job_failed(
                    job_id=job_id,
                    job_name=config.name,
                    error_message=str(e),
                )
            except Exception as notify_err:
                _logger.error(
                    "notification_failed_during_error_handling",
                    job_id=job_id,
                    notification_error=str(notify_err),
                    original_error=str(e),
                )

        raise typer.Exit(1) from None

    finally:
        # Ensure progress is stopped
        if progress and progress.live.is_started:
            progress.stop()

        # Clean up notification resources
        if notification_manager:
            try:
                await notification_manager.close()
            except Exception:
                pass  # Don't mask errors during final cleanup


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


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "run",
]
