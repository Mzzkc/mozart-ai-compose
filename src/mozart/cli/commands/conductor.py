"""Conductor commands — ``mozart start/stop/restart/conductor-status``.

These commands consolidate
all daemon lifecycle management into the main ``mozart`` CLI.
The core logic lives in ``mozart.daemon.process`` (shared functions);
this module provides thin Typer command wrappers.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..output import output_error


def start(
    config_file: Path | None = typer.Option(None, "--config", "-c", help="YAML config file"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
    profile: str | None = typer.Option(
        None, "--profile", "-p",
        help="Daemon operational profile (dev, intensive, minimal). "
        "Overrides config file defaults.",
    ),
) -> None:
    """Start the Mozart conductor."""
    from mozart.daemon.process import start_conductor

    start_conductor(
        config_file=config_file,
        foreground=foreground,
        log_level=log_level,
        profile=profile,
    )


def stop(
    pid_file: Path | None = typer.Option(None, "--pid-file", help="PID file path"),
    force: bool = typer.Option(False, "--force", help="Send SIGKILL instead of SIGTERM"),
) -> None:
    """Stop the Mozart conductor."""
    from mozart.daemon.process import stop_conductor

    stop_conductor(pid_file=pid_file, force=force)


def restart(
    config_file: Path | None = typer.Option(None, "--config", "-c", help="YAML config file"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
    pid_file: Path | None = typer.Option(None, "--pid-file", help="PID file path"),
    profile: str | None = typer.Option(
        None, "--profile", "-p",
        help="Daemon operational profile (dev, intensive, minimal). "
        "Overrides config file defaults.",
    ),
) -> None:
    """Restart the Mozart conductor (stop + start)."""
    from mozart.daemon.process import (
        start_conductor,
        stop_conductor,
        wait_for_conductor_exit,
    )

    # Stop (ignore exit if not running)
    try:
        stop_conductor(pid_file=pid_file)
    except SystemExit:
        pass

    # Wait for old process to fully exit before starting the new one.
    # Without this, start_conductor sees the dying process and says
    # "already running" (race condition).
    if not wait_for_conductor_exit(pid_file, timeout=30.0):
        output_error(
            "Old conductor did not exit within 30 seconds.",
            hints=["Try 'mozart stop --force' to send SIGKILL."],
        )
        raise typer.Exit(1)

    start_conductor(
        config_file=config_file,
        foreground=foreground,
        log_level=log_level,
        profile=profile,
    )


def conductor_status(
    pid_file: Path | None = typer.Option(None, "--pid-file", help="PID file path"),
    socket_path: Path | None = typer.Option(None, "--socket", help="Unix socket path"),
) -> None:
    """Check Mozart conductor status."""
    from mozart.daemon.process import get_conductor_status

    get_conductor_status(pid_file=pid_file, socket_path=socket_path)
