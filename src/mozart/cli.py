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
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from mozart import __version__

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


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
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
    start_batch: Optional[int] = typer.Option(
        None,
        "--start-batch",
        "-s",
        help="Override starting batch number",
    ),
) -> None:
    """Run a job from a YAML configuration file."""
    from mozart.core.config import JobConfig

    try:
        config = JobConfig.from_yaml(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)

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
        console.print("\n[yellow]Dry run - not executing[/yellow]")
        _show_dry_run(config)
        return

    # Actually run the job
    console.print("\n[green]Starting job...[/green]")
    asyncio.run(_run_job(config, start_batch))


async def _run_job(config, start_batch: Optional[int]) -> None:
    """Run the job asynchronously using the JobRunner."""
    from mozart.backends.claude_cli import ClaudeCliBackend
    from mozart.execution.runner import FatalError, JobRunner
    from mozart.learning.outcomes import JsonOutcomeStore
    from mozart.state.json_backend import JsonStateBackend

    # Ensure workspace exists
    config.workspace.mkdir(parents=True, exist_ok=True)

    # Setup backends
    state_backend = JsonStateBackend(config.workspace)
    backend = ClaudeCliBackend.from_config(config.backend)

    # Setup outcome store for learning if enabled
    outcome_store = None
    if config.learning.enabled:
        outcome_store_path = config.get_outcome_store_path()
        if config.learning.outcome_store_type == "json":
            outcome_store = JsonOutcomeStore(outcome_store_path)
        # Future: add SqliteOutcomeStore when implemented
        console.print(
            f"[dim]Learning enabled: outcomes will be stored at {outcome_store_path}[/dim]"
        )

    # Create runner with partial completion support and learning
    runner = JobRunner(
        config=config,
        backend=backend,
        state_backend=state_backend,
        console=console,
        outcome_store=outcome_store,
    )

    try:
        # Run job with validation and completion recovery
        state = await runner.run(start_batch=start_batch)

        if state.status.value == "completed":
            console.print(Panel("[bold green]Job completed successfully![/bold green]"))
        else:
            console.print(
                f"[yellow]Job ended with status: {state.status.value}[/yellow]"
            )

    except FatalError as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(1)


def _show_dry_run(config) -> None:
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
    job_id: Optional[str] = typer.Argument(
        None,
        help="Job ID to check (shows all if not specified)",
    ),
) -> None:
    """Show status of running or completed jobs."""
    # TODO: Implement status command
    console.print("[yellow]Status command not yet implemented[/yellow]")


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Job ID to resume"),
) -> None:
    """Resume a paused or failed job."""
    # TODO: Implement resume command
    console.print("[yellow]Resume command not yet implemented[/yellow]")


@app.command(name="list")
def list_jobs() -> None:
    """List all jobs."""
    # TODO: Implement list command
    console.print("[yellow]List command not yet implemented[/yellow]")


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
        raise typer.Exit(1)


@app.command()
def dashboard(
    port: int = typer.Option(8000, "--port", "-p", help="Port to run dashboard on"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
) -> None:
    """Start the web dashboard."""
    # TODO: Implement dashboard
    console.print(f"[yellow]Dashboard not yet implemented. Would run on {host}:{port}[/yellow]")


if __name__ == "__main__":
    app()
