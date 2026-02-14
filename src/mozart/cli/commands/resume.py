"""Resume command for Mozart CLI.

This module implements the `mozart resume` command which continues
execution of paused or failed jobs.

★ Insight ─────────────────────────────────────
1. **Config reconstruction hierarchy**: The resume command has a clear priority
   for config sources: (1) provided --config file, (2) --reload-config from
   original path, (3) cached config_snapshot, (4) stored config_path. This
   fallback chain ensures maximum flexibility while maintaining state consistency.

2. **Resumable state validation**: Only PAUSED, FAILED, and RUNNING jobs can be
   resumed. COMPLETED jobs require --force flag to override. This prevents
   accidental re-execution while still allowing intentional retries.

3. **Progress callback injection**: The same progress_callback pattern from run.py
   is used here, enabling seamless visual continuity between fresh runs and resumes.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.panel import Panel
from rich.progress import TaskID

from mozart.core.checkpoint import CheckpointState, JobStatus
from mozart.core.config import JobConfig
from mozart.state import StateBackend

from ..helpers import (
    configure_global_logging,
    is_quiet,
    require_job_state,
)
from ..output import console, format_duration
from ._shared import (
    create_progress_bar,
    handle_job_completion,
    setup_all,
)

_logger = logging.getLogger(__name__)


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
    asyncio.run(
        _resume_job(
            job_id, config_file, workspace, force, escalation,
            reload_config, self_healing, yes
        )
    )


async def _find_job_state(
    job_id: str,
    workspace: Path | None,
    force: bool,
) -> tuple[CheckpointState, StateBackend]:
    """Find and validate job state from available backends.

    Uses require_job_state for the find+error pattern, then validates
    that the job is in a resumable status.

    Args:
        job_id: Job ID to find.
        workspace: Optional workspace directory to search.
        force: Allow resuming completed jobs.

    Returns:
        Tuple of (found_state, found_backend).

    Raises:
        typer.Exit: If job not found or not in resumable state.
    """
    found_state, found_backend = await require_job_state(job_id, workspace)

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

    return found_state, found_backend


def _reconstruct_config(
    found_state: CheckpointState,
    config_file: Path | None,
    reload_config: bool,
) -> tuple[JobConfig, bool]:
    """Reconstruct JobConfig using 4-tier priority fallback.

    Priority order:
    1. Provided --config file
    2. --reload-config from original path
    3. Cached config_snapshot in state
    4. Stored config_path (last resort)

    Args:
        found_state: Job checkpoint state with config_snapshot/config_path.
        config_file: Optional explicit config file path.
        reload_config: Whether to reload from original yaml path.

    Returns:
        Tuple of (config, was_reloaded).

    Raises:
        typer.Exit: If no config source available or loading fails.
    """
    # Priority 1: Use provided config file (always takes precedence)
    if config_file:
        try:
            config = JobConfig.from_yaml(config_file)
            console.print(f"[dim]Using config from: {config_file}[/dim]")
            return config, True
        except Exception as e:
            console.print(f"[red]Error loading config file:[/red] {e}")
            raise typer.Exit(1) from None

    # Priority 2: If reload_config, force reload from config_path
    if reload_config:
        if found_state.config_path:
            config_path = Path(found_state.config_path)
            if config_path.exists():
                try:
                    config = JobConfig.from_yaml(config_path)
                    console.print(
                        f"[cyan]Reloaded config from:[/cyan] {config_path}"
                    )
                    return config, True
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
    if found_state.config_snapshot:
        try:
            config = JobConfig.model_validate(found_state.config_snapshot)
            console.print("[dim]Reconstructed config from saved state[/dim]")
            return config, False
        except Exception as e:
            console.print(f"[red]Error reconstructing config from snapshot:[/red] {e}")
            console.print(
                "[dim]Hint: Provide a config file with --config flag.[/dim]"
            )
            raise typer.Exit(1) from None

    # Priority 4: Try to load from stored config_path as last resort
    if found_state.config_path:
        config_path = Path(found_state.config_path)
        if config_path.exists():
            try:
                config = JobConfig.from_yaml(config_path)
                console.print(f"[dim]Loaded config from stored path: {config_path}[/dim]")
                return config, False
            except Exception as e:
                console.print(f"[red]Error loading stored config:[/red] {e}")
                raise typer.Exit(1) from None
        else:
            console.print(
                f"[yellow]Stored config file not found:[/yellow] {config_path}"
            )
            console.print("[dim]Hint: Provide a config file with --config flag.[/dim]")
            raise typer.Exit(1)

    console.print(
        "[red]Cannot resume: No config available.[/red]\n"
        "The job state doesn't contain a config snapshot.\n"
        "Please provide a config file with --config flag."
    )
    raise typer.Exit(1)


async def _resume_job(
    job_id: str,
    config_file: Path | None,
    workspace: Path | None,
    force: bool,
    escalation: bool = False,
    reload_config: bool = False,
    self_healing: bool = False,
    auto_confirm: bool = False,
) -> None:
    """Resume a paused or failed job.

    Args:
        job_id: Job ID to resume.
        config_file: Optional path to config file.
        workspace: Optional workspace directory to search.
        force: Force resume even if job appears completed.
        escalation: Enable human-in-the-loop escalation for low-confidence sheets.
        reload_config: If True, reload config from yaml file instead of cached snapshot.
        self_healing: Enable automatic diagnosis and remediation.
        auto_confirm: Auto-confirm suggested fixes.
    """
    from mozart.execution.runner import FatalError, GracefulShutdownError, JobRunner

    configure_global_logging(console)

    # Phase 1: Find and validate job state
    found_state, found_backend = await _find_job_state(job_id, workspace, force)

    # Phase 2: Reconstruct config
    config, config_was_reloaded = _reconstruct_config(
        found_state, config_file, reload_config
    )

    # Update config_snapshot in state if config was reloaded
    if config_was_reloaded:
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

    # Phase 3: Setup backends and features for execution (shared with run.py)
    components = setup_all(
        config, escalation=escalation, console=console,
    )
    backend = components.backend
    outcome_store = components.outcome_store
    global_learning_store = components.global_learning_store
    notification_manager = components.notification_manager
    escalation_handler = components.escalation_handler
    grounding_engine = components.grounding_engine

    # Create progress bar for sheet tracking
    progress = create_progress_bar(console=console)

    # Progress callback to update the progress bar
    progress_task_id: TaskID | None = None

    def update_progress(completed: int, total: int, eta_seconds: float | None) -> None:
        """Update progress bar with current sheet progress."""
        nonlocal progress_task_id
        if progress_task_id is not None:
            eta_str = format_duration(eta_seconds) if eta_seconds else "calculating..."
            progress.update(
                progress_task_id,
                completed=completed,
                total=total,
                eta=eta_str,
            )

    # Create runner context with all optional components
    from mozart.execution.runner import RunnerContext
    runner_context = RunnerContext(
        console=console,
        outcome_store=outcome_store,
        escalation_handler=escalation_handler,
        progress_callback=update_progress,
        global_learning_store=global_learning_store,
        grounding_engine=grounding_engine,
        self_healing_enabled=self_healing,
        self_healing_auto_confirm=auto_confirm,
    )

    # Create runner with progress callback
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=found_backend,
        context=runner_context,
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
        progress.stop()
        console.print("[yellow]Job paused. Exiting gracefully.[/yellow]")
        raise typer.Exit(0) from None

    except FatalError as e:
        progress.stop()
        console.print(f"[red]Fatal error: {e}[/red]")

        # Send failure notification (must not mask the original error)
        if notification_manager:
            try:
                await notification_manager.notify_job_failed(
                    job_id=job_id,
                    job_name=config.name,
                    error_message=str(e),
                )
            except Exception:
                _logger.debug("Notification failed during error handling", exc_info=True)

        raise typer.Exit(1) from None

    finally:
        # Ensure progress is stopped
        if progress.live.is_started:
            progress.stop()

        if notification_manager:
            try:
                await notification_manager.close()
            except OSError:
                pass  # Don't mask errors during final cleanup


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "resume",
]
