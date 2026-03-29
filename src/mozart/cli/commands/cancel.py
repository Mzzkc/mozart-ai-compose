"""Cancel command for Mozart CLI.

Provides `mozart cancel` to immediately cancel a running job via
asyncio task cancellation. Unlike `pause`, this interrupts mid-sheet
and does not wait for a clean boundary. Use `pause` for graceful
stops; use `cancel` when the job must stop now.
"""

from __future__ import annotations

import asyncio

import typer

from mozart.daemon.exceptions import DaemonError

from ..helpers import configure_global_logging, require_conductor
from ..output import console, output_error, output_json


def cancel(
    job_id: str = typer.Argument(..., help="Score ID to cancel"),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output result as JSON",
    ),
) -> None:
    """Cancel a running Mozart score immediately.

    Unlike `pause`, this does not wait for a sheet boundary. The score's
    asyncio task is cancelled, in-progress work is rolled back, and the
    score is marked as CANCELLED. Use `pause` for graceful stops.

    Examples:
        mozart cancel my-job
        mozart cancel my-job --json
    """
    from ._shared import validate_job_id

    job_id = validate_job_id(job_id)
    asyncio.run(_cancel_job(job_id, json_output))


async def _cancel_job(job_id: str, json_output: bool) -> None:
    from mozart.daemon.detect import try_daemon_route

    configure_global_logging(console)

    params = {"job_id": job_id}
    try:
        routed, result = await try_daemon_route("job.cancel", params)
    except (OSError, ConnectionError, DaemonError) as exc:
        output_error(
            str(exc),
            hints=["Check conductor status: mozart conductor-status"],
            json_output=json_output,
        )
        raise typer.Exit(1) from None

    if not routed:
        require_conductor(routed, json_output=json_output)
        return

    cancelled = result.get("cancelled", False)
    if json_output:
        output_json({"success": cancelled, "job_id": job_id})
    elif cancelled:
        console.print(f"[green]Score '{job_id}' cancelled.[/green]")
    else:
        console.print(f"[yellow]Score '{job_id}' not found or already stopped.[/yellow]")
