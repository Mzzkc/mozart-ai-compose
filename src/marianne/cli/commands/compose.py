"""Compose command — ``mzt compose``.

Compiles semantic agent definitions into Mozart scores. Takes a YAML config
describing agents as people (voice, focus, meditation, techniques) and produces
complete self-chaining scores with identity seeding, technique wiring,
instrument resolution, and validation generation.

Usage::

    mzt compose config.yaml --output scores/my-project/
    mzt compose config.yaml --output scores/ --fleet
    mzt compose config.yaml --dry-run
    mzt compose config.yaml --seed-only
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.table import Table

from marianne.core.logging import get_logger

from ..output import console, output_error

if TYPE_CHECKING:
    from marianne.compose.pipeline import CompilationPipeline

_logger = get_logger("cli.compose")


def compose(
    config: Path = typer.Argument(
        ...,
        help="Path to the semantic agent config YAML",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output directory for generated scores",
    ),
    fleet: bool = typer.Option(
        False,
        "--fleet",
        help="Force generation of a fleet config (auto-generated for multi-agent configs)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be generated without writing files",
    ),
    seed_only: bool = typer.Option(
        False,
        "--seed-only",
        help="Only seed agent identities, don't generate scores",
    ),
    agents_dir: Path | None = typer.Option(
        None,
        "--agents-dir",
        help="Override agents identity directory (default: ~/.mzt/agents)",
    ),
    techniques_dir: Path | None = typer.Option(
        None,
        "--techniques-dir",
        help="Path to technique module documents",
    ),
) -> None:
    """Compile semantic agent definitions into Mozart scores.

    Takes a YAML config describing agents and produces complete
    self-chaining scores with identity, techniques, instruments,
    and validations.
    """
    import yaml

    from marianne.compose.pipeline import CompilationPipeline

    # Load config
    try:
        with open(config) as f:
            config_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        output_error(
            f"YAML syntax error: {e}",
            hints=[f"Check indentation and syntax in {config}"],
        )
        raise typer.Exit(1) from None
    except FileNotFoundError:
        output_error(
            f"Config file not found: {config}",
            hints=["Check the file path and try again"],
        )
        raise typer.Exit(1) from None

    if not isinstance(config_data, dict):
        output_error(
            "Config file must contain a YAML mapping",
            hints=["Ensure the file starts with key-value pairs, not a list"],
        )
        raise typer.Exit(1) from None

    agents = config_data.get("agents", [])
    if not agents:
        output_error(
            "Config must contain at least one agent",
            hints=["Add an 'agents' list with at least one agent definition"],
        )
        raise typer.Exit(1)

    project = config_data.get("project", {})
    project_name = project.get("name", config.stem) if isinstance(project, dict) else config.stem

    if dry_run:
        _show_dry_run(config_data, project_name)
        return

    pipeline = CompilationPipeline(
        agents_dir=agents_dir,
        techniques_dir=techniques_dir,
    )

    if seed_only:
        _seed_identities(pipeline, config_data)
        return

    # Compile all agent scores
    output_dir = output or (config.parent / "scores")
    score_paths = pipeline.compile_config(config_data, output_dir)

    # Force fleet generation when --fleet is passed and pipeline didn't already
    # generate one (pipeline auto-generates fleet for multi-agent configs)
    if fleet and not any(p.name == "fleet.yaml" for p in score_paths):
        from marianne.compose.fleet import FleetGenerator

        fleet_path = output_dir / "fleet.yaml"
        FleetGenerator().write(config_data, output_dir, fleet_path)
        score_paths.append(fleet_path)
        _logger.info("fleet_config_forced", path=str(fleet_path))

    console.print(f"\n[green]Compiled {len(agents)} agent scores[/green]")
    for path in score_paths:
        console.print(f"  {path}")
    console.print(f"\nScores written to: {output_dir}")


def _show_dry_run(config_data: dict[str, Any], project_name: str) -> None:
    """Display dry-run summary without generating files."""
    agents = config_data.get("agents", [])
    defaults = config_data.get("defaults", {})

    console.print("\n[bold]Composition Compiler — Dry Run[/bold]")
    console.print(f"{'=' * 50}")
    console.print(f"Project:  {project_name}")
    console.print(f"Agents:   {len(agents)}")

    table = Table(title="Agent Roster")
    table.add_column("Name", style="cyan")
    table.add_column("Focus")
    table.add_column("Voice", max_width=40)
    table.add_column("Overrides")

    for agent in agents:
        if not isinstance(agent, dict):
            continue
        name = agent.get("name", "?")
        focus = agent.get("focus", "")
        voice = agent.get("voice", "")
        overrides: list[str] = []
        if agent.get("instruments"):
            overrides.append("instruments")
        if agent.get("techniques"):
            overrides.append("techniques")
        if agent.get("patterns"):
            overrides.append("patterns")
        table.add_row(
            str(name),
            str(focus),
            str(voice)[:40],
            ", ".join(overrides) if overrides else "-",
        )

    console.print(table)

    # Show defaults summary
    instruments = defaults.get("instruments", {}) if isinstance(defaults, dict) else {}
    techniques = defaults.get("techniques", {}) if isinstance(defaults, dict) else {}
    console.print(f"\nDefault instruments: {len(instruments)} tiers")
    console.print(f"Default techniques:  {len(techniques)} declared")
    console.print("Sheets per cycle:    12")


def _seed_identities(
    pipeline: CompilationPipeline,
    config_data: dict[str, Any],
) -> None:
    """Seed identities only without generating scores."""
    agents = config_data.get("agents", [])
    if not isinstance(agents, list):
        return

    for agent_def in agents:
        if not isinstance(agent_def, dict):
            continue
        path = pipeline.seed_identity(agent_def)
        console.print(f"  Seeded: {path}")

    console.print(f"\n[green]Seeded {len(agents)} agent identities[/green]")


__all__ = ["compose"]
