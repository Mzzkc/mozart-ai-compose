"""Pause and modify commands for Mozart CLI.

This module implements the `mozart pause` and `mozart modify` commands
for gracefully pausing running jobs and updating their configuration.

★ Insight ─────────────────────────────────────
1. **Signal-based pause mechanism**: Rather than interrupting execution directly,
   Mozart uses a file-based signal (.mozart-pause-{job_id}). The runner polls for
   this file at sheet boundaries, enabling clean checkpoints without data loss.

2. **Atomic state transitions**: The pause command only works on RUNNING jobs,
   and modify can handle RUNNING, PAUSED, or FAILED states. This state machine
   prevents race conditions and invalid state transitions.

3. **Modify as composition**: The modify command is essentially pause + resume
   composed together with config validation in between. This pattern keeps the
   individual commands focused while providing a convenient compound operation.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from mozart.core.checkpoint import JobStatus
from mozart.core.config import JobConfig

from ..helpers import (
    configure_global_logging,
    create_pause_signal,
    find_job_workspace,
    require_job_state,
    wait_for_pause_ack,
)
from ..output import console, output_error


def pause(
    job_id: str = typer.Argument(..., help="Job ID to pause"),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing job state",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Wait for job to acknowledge pause signal",
    ),
    timeout: int = typer.Option(
        60,
        "--timeout",
        "-t",
        help="Timeout in seconds when using --wait",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON",
    ),
) -> None:
    """Pause a running Mozart job gracefully.

    Creates a pause signal that the job will detect at the next sheet boundary.
    The job saves its state and can be resumed with `mozart resume`.

    Examples:
        mozart pause my-job
        mozart pause my-job --workspace ./workspace
        mozart pause my-job --wait --timeout 30
        mozart pause my-job --json
    """
    asyncio.run(_pause_job(job_id, workspace, wait, timeout, json_output))


async def _pause_job(
    job_id: str,
    workspace: Path | None,
    wait: bool,
    timeout: int,
    json_output: bool,
) -> None:
    """Pause a running job.

    Args:
        job_id: Job ID to pause.
        workspace: Optional workspace directory.
        wait: Whether to wait for pause acknowledgment.
        timeout: Timeout in seconds for wait.
        json_output: Output in JSON format.
    """
    configure_global_logging(console)

    # Find job state using shared discovery helper
    found_state, found_backend = await require_job_state(
        job_id, workspace, json_output=json_output,
    )

    # Resolve workspace directory for pause signal file
    found_workspace = find_job_workspace(job_id, workspace)
    if not found_workspace:
        # Shouldn't happen since require_job_state succeeded, but guard anyway
        raise typer.Exit(1)

    # Check if job is in a pausable state
    if found_state.status != JobStatus.RUNNING:
        status_str = found_state.status.value
        if json_output:
            hints: list[str] = []
            if found_state.status == JobStatus.PAUSED:
                hints.append("Job is already paused")
                hints.append(f"Use 'mozart resume {job_id}' to resume")
            elif found_state.status == JobStatus.PENDING:
                hints.append("Use 'mozart run' to start the job")
            elif found_state.status == JobStatus.COMPLETED:
                hints.append("Job has already completed")
            result = {
                "success": False,
                "error_code": "E502",
                "job_id": job_id,
                "status": status_str,
                "message": f"Job '{job_id}' is {status_str}, not running",
                "hints": hints,
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(
                f"[red]Error [E502]:[/red] Job '{job_id}' is {status_str}, not running"
            )
            console.print()
            if found_state.status == JobStatus.PAUSED:
                console.print("[dim]Hint: Job is already paused.[/dim]")
                console.print(f"[dim]Use 'mozart resume {job_id}' to resume.[/dim]")
            elif found_state.status == JobStatus.PENDING:
                console.print("[dim]Hint: Use 'mozart run' to start the job.[/dim]")
            elif found_state.status == JobStatus.COMPLETED:
                console.print("[dim]Hint: Job has already completed.[/dim]")
        raise typer.Exit(1)

    # Create pause signal file
    try:
        signal_file = create_pause_signal(found_workspace, job_id)
    except (PermissionError, OSError) as e:
        output_error(
            f"Cannot create pause signal: {e}",
            error_code="E503",
            hints=["Check workspace write permissions"],
            json_output=json_output,
            job_id=job_id,
        )
        raise typer.Exit(1) from None

    # Optionally wait for pause acknowledgment
    was_acknowledged = False
    if wait and found_backend:
        if not json_output:
            console.print(
                f"[dim]Waiting for job to pause (timeout: {timeout}s)...[/dim]"
            )
        was_acknowledged = await wait_for_pause_ack(found_backend, job_id, timeout)
        if not was_acknowledged:
            if json_output:
                result = {
                    "success": False,
                    "error_code": "E504",
                    "job_id": job_id,
                    "message": f"Pause not acknowledged within {timeout}s",
                    "signal_file": str(signal_file),
                    "hints": [
                        "Job may have completed before processing signal",
                        "Check 'mozart status' for current job state",
                    ],
                }
                console.print(json.dumps(result, indent=2))
            else:
                console.print(
                    f"[yellow]Warning [E504]:[/yellow] "
                    f"Pause not acknowledged within {timeout}s"
                )
                console.print()
                console.print("[dim]Hints:[/dim]")
                console.print(
                    "  - Job may have completed before processing signal"
                )
                console.print("  - Check 'mozart status' for current job state")
            raise typer.Exit(2)

    # Output success
    if json_output:
        result = {
            "success": True,
            "job_id": job_id,
            "status": "running" if not was_acknowledged else "paused",
            "message": "Pause signal sent. Job will pause at next sheet boundary.",
            "signal_file": str(signal_file),
            "acknowledged": was_acknowledged,
        }
        console.print(json.dumps(result, indent=2))
    else:
        if was_acknowledged:
            console.print(f"[green]Job '{job_id}' paused successfully.[/green]")
        else:
            console.print(f"Pause signal sent to job '[cyan]{job_id}[/cyan]'.")
            console.print("Job will pause at next sheet boundary.")
        console.print()
        console.print(f"To resume: [bold]mozart resume {job_id}[/bold]")
        console.print(
            f"To resume with new config: "
            f"[bold]mozart resume {job_id} -r --config new.yaml[/bold]"
        )


def modify(
    job_id: str = typer.Argument(..., help="Job ID to modify"),
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="New configuration file",
        exists=True,
        readable=True,
    ),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing job state",
    ),
    resume_flag: bool = typer.Option(
        False,
        "--resume",
        "-r",
        help="Immediately resume with new config after pausing",
    ),
    wait: bool = typer.Option(
        False,
        "--wait",
        help="Wait for job to pause before resuming (when --resume)",
    ),
    timeout: int = typer.Option(
        60,
        "--timeout",
        "-t",
        help="Timeout in seconds for pause acknowledgment",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON",
    ),
) -> None:
    """Modify a job's configuration and optionally resume execution.

    This is a convenience command that combines pause + config validation.
    If the job is running, it will be paused first.
    Use --resume to immediately resume with the new configuration.

    Examples:
        mozart modify my-job --config updated.yaml
        mozart modify my-job -c new-config.yaml --resume
        mozart modify my-job -c updated.yaml -r --workspace ./workspace
        mozart modify my-job -c updated.yaml -r --wait
    """
    asyncio.run(
        _modify_job(job_id, config, workspace, resume_flag, wait, timeout, json_output)
    )


async def _modify_job(
    job_id: str,
    config_file: Path,
    workspace: Path | None,
    resume_flag: bool,
    wait: bool,
    timeout: int,
    json_output: bool,
) -> None:
    """Modify a job's configuration.

    Args:
        job_id: Job ID to modify.
        config_file: New config file path.
        workspace: Optional workspace directory.
        resume_flag: Whether to resume after pausing.
        wait: Whether to wait for pause acknowledgment before resuming.
        timeout: Timeout in seconds for pause wait.
        json_output: Output in JSON format.
    """
    from .resume import _resume_job

    configure_global_logging(console)

    # Validate the new config file first
    try:
        new_config = JobConfig.from_yaml(config_file)
    except Exception as e:
        if json_output:
            result = {
                "success": False,
                "error_code": "E505",
                "job_id": job_id,
                "message": f"Invalid config file: {e}",
                "config_file": str(config_file),
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(f"[red]Error [E505]:[/red] Invalid config file: {e}")
        raise typer.Exit(1) from None

    # Find job state using shared discovery helper
    found_state, found_backend = await require_job_state(
        job_id, workspace, json_output=json_output,
    )

    # Resolve workspace directory for pause signal file
    found_workspace = find_job_workspace(job_id, workspace)
    if not found_workspace:
        # Shouldn't happen since require_job_state succeeded, but guard anyway
        raise typer.Exit(1)

    # Handle job based on its current state
    job_was_running = found_state.status == JobStatus.RUNNING

    if job_was_running:
        # Pause the running job first
        try:
            create_pause_signal(found_workspace, job_id)
            if not json_output:
                console.print(
                    f"Pause signal sent to job '[cyan]{job_id}[/cyan]'."
                )
        except (PermissionError, OSError) as e:
            if json_output:
                result = {
                    "success": False,
                    "error_code": "E503",
                    "job_id": job_id,
                    "message": f"Cannot create pause signal: {e}",
                }
                console.print(json.dumps(result, indent=2))
            else:
                console.print(f"[red]Error [E503]:[/red] Cannot create pause signal: {e}")
            raise typer.Exit(1) from None

        # Wait for pause if requested and resuming
        if wait and resume_flag and found_backend:
            if not json_output:
                console.print(
                    f"[dim]Waiting for job to pause (timeout: {timeout}s)...[/dim]"
                )
            was_acknowledged = await wait_for_pause_ack(found_backend, job_id, timeout)
            if not was_acknowledged:
                if json_output:
                    result = {
                        "success": False,
                        "error_code": "E504",
                        "job_id": job_id,
                        "message": f"Pause not acknowledged within {timeout}s",
                    }
                    console.print(json.dumps(result, indent=2))
                else:
                    console.print(
                        f"[yellow]Warning:[/yellow] Pause not acknowledged within {timeout}s"
                    )
                raise typer.Exit(2)

    elif found_state.status not in {JobStatus.PAUSED, JobStatus.FAILED}:
        # Job is in a state that can't be modified (completed, pending)
        status_str = found_state.status.value
        if json_output:
            result = {
                "success": False,
                "error_code": "E502",
                "job_id": job_id,
                "status": status_str,
                "message": f"Job '{job_id}' is {status_str}, cannot modify",
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print(
                f"[red]Error [E502]:[/red] Job '{job_id}' is {status_str}"
            )
            if found_state.status == JobStatus.COMPLETED:
                console.print("[dim]Hint: Job has already completed.[/dim]")
            elif found_state.status == JobStatus.PENDING:
                console.print("[dim]Hint: Use 'mozart run' to start the job.[/dim]")
        raise typer.Exit(1)

    # Output success message
    if not json_output:
        console.print(f"[green]Config validated:[/green] {config_file}")
        console.print(f"[dim]Job name: {new_config.name}[/dim]")
        console.print(f"[dim]Sheets: {new_config.sheet.total_sheets}[/dim]")

    # If resume flag is set, call resume logic
    if resume_flag:
        if not json_output:
            console.print()
            console.print("[cyan]Resuming with new config...[/cyan]")

        # Call resume with reload_config
        await _resume_job(
            job_id=job_id,
            config_file=config_file,
            workspace=found_workspace,
            force=False,
            escalation=False,
            reload_config=True,
            self_healing=False,
            auto_confirm=False,
        )
    else:
        # Just show instructions
        if json_output:
            result = {
                "success": True,
                "job_id": job_id,
                "status": found_state.status.value,
                "config_validated": True,
                "config_file": str(config_file),
                "message": "Config validated. Job paused and ready to resume.",
            }
            console.print(json.dumps(result, indent=2))
        else:
            console.print()
            console.print("When ready to resume with new config:")
            console.print(
                f"  [bold]mozart resume {job_id} -r --config {config_file}[/bold]"
            )
            console.print()
            console.print("Or to resume with original config:")
            console.print(f"  [bold]mozart resume {job_id}[/bold]")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "pause",
    "modify",
]
