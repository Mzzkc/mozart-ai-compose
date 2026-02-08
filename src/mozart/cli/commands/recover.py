"""Recover command for Mozart CLI.

This module implements the hidden `mozart recover` command for recovering
sheets that completed work but were incorrectly marked as failed.

★ Insight ─────────────────────────────────────
1. **Non-destructive recovery**: The recover command re-runs validations without
   re-executing the backend. This is useful when work was completed but the
   process failed afterwards (e.g., transient network error after writing files).

2. **State machine transitions**: The command can transition sheets from FAILED
   to COMPLETED, and the job from FAILED to PAUSED. This allows the job to be
   resumed normally after recovery.

3. **Dry-run safety**: The --dry-run flag runs validations without modifying
   state. This lets users preview what would be recovered before committing.
─────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel

from mozart.core.checkpoint import JobStatus, SheetStatus
from mozart.core.config import JobConfig
from mozart.execution.validation import ValidationEngine
from mozart.state import JsonStateBackend

from ..helpers import configure_global_logging
from ..output import console


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
    configure_global_logging(console)

    # Find job state
    state_file = None
    search_paths: list[Path] = []

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

    recovered_count = 0
    for snum in sorted(sheets_to_check):
        console.print(f"\n[bold]Sheet {snum}:[/bold]")

        # Create validation engine for this sheet
        sheet_context: dict[str, Any] = {
            "sheet_num": snum,
            "start_item": None,
            "end_item": None,
        }
        validation_engine = ValidationEngine(
            workspace=config.workspace,
            sheet_context=sheet_context,
        )
        result = await validation_engine.run_validations(config.validations)

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
                console.print("  [blue]→ Marked as completed[/blue]")
            else:
                console.print("  [yellow]→ Would mark as completed (dry-run)[/yellow]")
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
        console.print("\n[yellow]Dry run complete - no changes made[/yellow]")
    else:
        console.print("\n[yellow]No sheets could be recovered[/yellow]")


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "recover",
]
