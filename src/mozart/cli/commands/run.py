"""Run command for Mozart CLI.

This module implements the ``mozart run`` command which routes jobs
through a running mozartd daemon.  Direct execution is not supported —
a running daemon is required (like docker requires dockerd).

The only exception is ``--dry-run``, which validates and displays the
job plan without executing anything (no daemon needed).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.panel import Panel
from rich.table import Table

from ..helpers import is_quiet
from ..output import console

if TYPE_CHECKING:
    from mozart.core.config import JobConfig


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
    if escalation:
        _msg = (
            "--escalation requires interactive console prompts which are "
            "not available in daemon mode. Escalation is not currently "
            "supported."
        )
        if json_output:
            console.print(json.dumps({"error": _msg}, indent=2))
        else:
            console.print(f"[red]Error:[/red] {_msg}")
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

    # Route through daemon (required)
    routed = asyncio.run(
        _try_daemon_submit(
            config_file, workspace, fresh, self_healing, yes, json_output,
            start_sheet=start_sheet,
        ),
    )
    if routed:
        return

    # Daemon not available or submission failed
    if json_output:
        console.print(json.dumps({
            "error": "Mozart daemon is not running. Start with: mozartd start",
        }))
    else:
        console.print("[red]Error:[/red] Mozart daemon is not running.")
        console.print("Start it with: [bold]mozartd start[/bold]")
    raise typer.Exit(1)


async def _try_daemon_submit(
    config_file: Path,
    workspace: Path | None,
    fresh: bool,
    self_healing: bool,
    auto_confirm: bool,
    json_output: bool,
    *,
    start_sheet: int | None = None,
) -> bool:
    """Submit a job to the running mozartd daemon.

    Returns True if the daemon accepted the submission, False if the
    daemon is not reachable or rejected the job.  Never raises — all
    errors return False so the caller can emit an appropriate error.
    """
    try:
        from mozart.daemon.detect import is_daemon_available, try_daemon_route

        if not await is_daemon_available():
            return False

        params: dict[str, Any] = {
            "config_path": str(config_file.resolve()),
            "fresh": fresh,
            "self_healing": self_healing,
            "self_healing_auto_confirm": auto_confirm,
        }
        if workspace is not None:
            params["workspace"] = str(Path(workspace).resolve())
        if start_sheet is not None:
            params["start_sheet"] = start_sheet

        routed, result = await try_daemon_route("job.submit", params)
        if not routed:
            return False

        if json_output:
            console.print(json.dumps(result, indent=2))
        else:
            status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
            job_id = result.get("job_id", "?") if isinstance(result, dict) else "?"
            msg = result.get("message", "") if isinstance(result, dict) else ""
            if status == "accepted":
                console.print(f"[green]Job submitted to daemon:[/green] {job_id}")
                if msg:
                    console.print(f"  {msg}")
                # Monitoring guidance
                ws = workspace or config_file.parent / "workspace"
                console.print(
                    f"\n[dim]Monitor with:[/dim] mozart status {job_id} -w {ws} --watch"
                )
            else:
                console.print(f"[yellow]Daemon rejected job:[/yellow] {msg}")
                return False

        return True
    except Exception:
        return False


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
