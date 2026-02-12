"""Daemon configuration management commands for Mozart CLI.

This module implements the `mozart config` command group for viewing
and managing the mozartd daemon configuration file.

Subcommands:
- `mozart config show`  — Display current config as a Rich table
- `mozart config set`   — Update a config value
- `mozart config path`  — Show config file location
- `mozart config init`  — Create a default config file
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from rich.table import Table

from ..output import console

config_app = typer.Typer(
    name="config",
    help="Manage daemon (mozartd) configuration.",
    invoke_without_command=True,
)

DEFAULT_CONFIG_DIR = Path("~/.mozart")
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "daemon.yaml"


def _resolve_config_path(config_file: Path | None) -> Path:
    """Resolve the config file path, expanding ~."""
    path = config_file or DEFAULT_CONFIG_FILE
    return path.expanduser()


def _load_config_data(path: Path) -> dict[str, Any]:
    """Load config YAML from disk, returning empty dict if missing."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_config_data(path: Path, data: dict[str, Any]) -> None:
    """Write config data to YAML file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    tmp.rename(path)


def _get_nested(data: dict[str, Any], dotted_key: str) -> Any:
    """Get a value from a nested dict using dot notation."""
    keys = dotted_key.split(".")
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dot notation."""
    keys = dotted_key.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _coerce_value(raw: str) -> Any:
    """Coerce a string value to the appropriate Python type."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    """Manage daemon (mozartd) configuration."""
    if ctx.invoked_subcommand is None:
        # No subcommand provided — show help
        console.print(ctx.get_help())
        raise typer.Exit(0)


@config_app.command()
def show(
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to daemon config file (default: ~/.mozart/daemon.yaml)",
    ),
) -> None:
    """Display current daemon configuration as a table.

    Shows all configuration values with their current and default settings.
    Values loaded from the config file are highlighted; defaults shown in dim.

    Examples:
        mozart config show
        mozart config show --config /etc/mozart/daemon.yaml
    """
    from mozart.daemon.config import DaemonConfig

    path = _resolve_config_path(config_file)
    file_data = _load_config_data(path)

    # Build the effective config (file values + defaults)
    try:
        effective = DaemonConfig.model_validate(file_data)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from None

    source_label = f"[dim]{path}[/dim]" if path.exists() else "[dim](defaults)[/dim]"
    console.print(f"\nDaemon configuration — {source_label}\n")

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("Key", style="white", min_width=30)
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Flatten the config for display
    flat = _flatten_model(effective.model_dump(), prefix="")

    for key, value in flat.items():
        file_value = _get_nested(file_data, key)
        source = "file" if file_value is not None else "default"
        source_display = f"[dim]{source}[/dim]" if source == "default" else source
        table.add_row(key, str(value), source_display)

    console.print(table)


def _flatten_model(
    data: dict[str, Any], prefix: str,
) -> dict[str, Any]:
    """Flatten a nested dict into dot-notation keys."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            result.update(_flatten_model(value, full_key))
        else:
            result[full_key] = value
    return result


@config_app.command("set")
def set_value(
    key: str = typer.Argument(
        ...,
        help="Config key in dot notation (e.g., socket.path, max_concurrent_jobs)",
    ),
    value: str = typer.Argument(
        ...,
        help="New value to set",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to daemon config file (default: ~/.mozart/daemon.yaml)",
    ),
) -> None:
    """Update a daemon configuration value.

    Values are validated against the DaemonConfig schema before saving.
    Use dot notation for nested keys (e.g., socket.path, resource_limits.max_memory_mb).

    Examples:
        mozart config set max_concurrent_jobs 10
        mozart config set socket.path /tmp/custom.sock
        mozart config set resource_limits.max_memory_mb 4096
        mozart config set log_level debug
    """
    from mozart.daemon.config import DaemonConfig

    path = _resolve_config_path(config_file)
    data = _load_config_data(path)

    coerced = _coerce_value(value)
    _set_nested(data, key, coerced)

    # Validate the full config before saving
    try:
        DaemonConfig.model_validate(data)
    except Exception as e:
        console.print(f"[red]Invalid value:[/red] {e}")
        raise typer.Exit(1) from None

    _save_config_data(path, data)
    console.print(f"[green]Set[/green] {key} = {coerced!r} in {path}")


@config_app.command()
def path(
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to daemon config file (default: ~/.mozart/daemon.yaml)",
    ),
) -> None:
    """Show the daemon config file location.

    Displays the resolved path and whether the file exists.

    Examples:
        mozart config path
        mozart config path --config /etc/mozart/daemon.yaml
    """
    resolved = _resolve_config_path(config_file)
    exists = resolved.exists()
    status = "[green]exists[/green]" if exists else "[yellow]not created[/yellow]"
    console.print(f"{resolved}  ({status})")


@config_app.command()
def init(
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to create config file (default: ~/.mozart/daemon.yaml)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config file",
    ),
) -> None:
    """Create a default daemon config file.

    Generates a YAML config file with all default values and
    descriptive comments. Refuses to overwrite unless --force is given.

    Examples:
        mozart config init
        mozart config init --config /etc/mozart/daemon.yaml
        mozart config init --force
    """
    from mozart.daemon.config import DaemonConfig

    resolved = _resolve_config_path(config_file)

    if resolved.exists() and not force:
        console.print(
            f"[yellow]Config file already exists:[/yellow] {resolved}\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(1)

    defaults = DaemonConfig()
    data = defaults.model_dump()

    # Convert Path objects to strings for YAML serialization
    _stringify_paths(data)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    _save_config_data(resolved, data)
    console.print(f"[green]Created default config:[/green] {resolved}")


def _stringify_paths(data: dict[str, Any]) -> None:
    """Recursively convert Path values to strings for YAML output."""
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
        elif isinstance(value, dict):
            _stringify_paths(value)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "config_app",
]
