"""Mozart CLI - modular command structure.

This package provides a modular CLI implementation for Mozart AI Compose.
The CLI is built using Typer and organized into command modules for maintainability.

★ Insight ─────────────────────────────────────
1. **Typer callback for global options**: The @app.callback() decorator defines
   options that apply to ALL commands (--verbose, --quiet, --version). These
   callbacks run BEFORE any command, setting up shared state like output level
   and logging configuration.

2. **Command registration via app.command()**: Each command function is registered
   with the Typer app using app.command(). The function name becomes the CLI
   command name (with underscores converted to hyphens). Custom names are set
   via the `name` parameter.

3. **Modular command imports**: Commands are organized in separate modules
   (run.py, status.py, etc.) and imported here for registration. This keeps
   the main __init__.py focused on assembly while command logic lives in
   dedicated files.
─────────────────────────────────────────────────

Package structure:
    cli/
    ├── __init__.py           # This file - app assembly (~150 LOC)
    ├── helpers.py            # Shared utilities (~600 LOC)
    ├── output.py             # Rich formatting (~400 LOC)
    └── commands/
        ├── __init__.py       # Command exports
        ├── run.py            # run command
        ├── status.py         # status, list_jobs commands
        ├── resume.py         # resume command
        ├── pause.py          # pause, modify commands
        ├── validate.py       # validate command
        ├── recover.py        # recover command (hidden)
        ├── diagnose.py       # logs, errors, diagnose commands
        ├── dashboard.py      # dashboard, mcp commands
        └── learning.py       # patterns-*, learning-* commands
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from mozart import __version__

# Re-export helpers module for direct access to internal state (conftest.py needs this)
from . import helpers as helpers
from .commands import (
    # diagnose.py
    diagnose,
    errors,
    history,
    # status.py
    list_jobs,
    logs,
    # pause.py
    modify,
    pause,
    # recover.py
    recover,
    # resume.py
    resume,
    # run.py
    run,
    status,
    # validate.py
    validate,
)
from .commands.conductor import (
    conductor_status,
    restart,
    start,
    stop,
)
from .commands.config_cmd import config_app
from .commands.dashboard import dashboard, mcp
from .commands.learning import (
    entropy_status,
    learning_activity,
    learning_drift,
    learning_epistemic_drift,
    learning_insights,
    learning_stats,
    patterns_budget,
    patterns_entropy,
    patterns_list,
    patterns_why,
)
from .commands.status import _output_status_rich

# Import helper functions for re-export
from .helpers import (
    OutputLevel,
    configure_global_logging,
    create_notifiers_from_config,
    set_log_file,
    set_log_format,
    set_log_level,
    set_output_level,
)
from .output import console

# =============================================================================
# Typer app definition
# =============================================================================

app = typer.Typer(
    name="mozart",
    help="Orchestration tool for Claude AI sessions",
    add_completion=False,
)


# =============================================================================
# Global option callbacks
# =============================================================================


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Mozart AI Compose v{__version__}")
        raise typer.Exit()


def verbose_callback(value: bool) -> None:
    """Set verbose output mode."""
    if value:
        set_output_level(OutputLevel.VERBOSE)


def quiet_callback(value: bool) -> None:
    """Set quiet output mode."""
    if value:
        set_output_level(OutputLevel.QUIET)


def log_level_callback(value: str | None) -> str | None:
    """Set log level from CLI option."""
    if value:
        set_log_level(value)
    return value


def log_file_callback(value: Path | None) -> Path | None:
    """Set log file path from CLI option."""
    if value:
        set_log_file(value)
    return value


def log_format_callback(value: str | None) -> str | None:
    """Set log format from CLI option."""
    if value:
        set_log_format(value)
    return value


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
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            "-L",
            callback=log_level_callback,
            help="Logging level (DEBUG, INFO, WARNING, ERROR)",
            envvar="MOZART_LOG_LEVEL",
        ),
    ] = None,
    log_file: Annotated[
        Path | None,
        typer.Option(
            "--log-file",
            callback=log_file_callback,
            help="Path for log file output",
            envvar="MOZART_LOG_FILE",
        ),
    ] = None,
    log_format: Annotated[
        str | None,
        typer.Option(
            "--log-format",
            callback=log_format_callback,
            help="Log format: json, console, or both",
            envvar="MOZART_LOG_FORMAT",
        ),
    ] = None,
) -> None:
    """Mozart AI Compose - Orchestration tool for Claude AI sessions."""
    # Configure logging based on CLI options (called once)
    configure_global_logging(console)


# =============================================================================
# Core command registration
# =============================================================================

# Job execution commands
app.command()(run)
app.command()(resume)
app.command()(pause)
app.command()(modify)

# Job status commands
app.command()(status)
app.command(name="list")(list_jobs)

# Validation and recovery
app.command()(validate)
app.command(hidden=True)(recover)  # Hidden - recovery is advanced operation

# Diagnostic commands
app.command()(logs)
app.command()(errors)
app.command()(diagnose)
app.command()(history)

# Server commands
app.command()(dashboard)
app.command()(mcp)

# Conductor lifecycle commands
app.command()(start)
app.command()(stop)
app.command()(restart)
app.command(name="conductor-status")(conductor_status)

# Daemon configuration
app.add_typer(config_app)

# =============================================================================
# Learning system commands
# =============================================================================

# Pattern analysis commands
app.command(name="patterns-list")(patterns_list)
app.command(name="patterns-why")(patterns_why)
app.command(name="patterns-entropy")(patterns_entropy)
app.command(name="patterns-budget")(patterns_budget)

# Learning statistics and insights
app.command(name="learning-stats")(learning_stats)
app.command(name="learning-insights")(learning_insights)
app.command(name="learning-drift")(learning_drift)
app.command(name="learning-epistemic-drift")(learning_epistemic_drift)
app.command(name="learning-activity")(learning_activity)

# System health monitoring
app.command(name="entropy-status")(entropy_status)


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "app",
    "main",
    "console",
    # Re-export helpers for convenience
    "OutputLevel",
    # Helper functions (used by tests)
    "create_notifiers_from_config",
    "_output_status_rich",
]
