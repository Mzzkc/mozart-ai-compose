"""Shared execution setup logic for run and resume commands.

Extracts the common infrastructure setup that both _run_job() and _resume_job()
need: backend creation, learning stores, notification manager, escalation handler,
grounding engine, and summary display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from mozart.core.checkpoint import JobStatus
from mozart.notifications import NotificationManager

from ..helpers import (
    create_notifiers_from_config,
    is_quiet,
    is_verbose,
)
from ..output import console as default_console

if TYPE_CHECKING:
    from mozart.backends.base import Backend
    from mozart.core.config import JobConfig
    from mozart.execution.escalation import ConsoleEscalationHandler
    from mozart.execution.grounding import GroundingEngine
    from mozart.execution.runner import RunSummary
    from mozart.learning.global_store import GlobalLearningStore
    from mozart.learning.outcomes import OutcomeStore


def create_backend(
    config: JobConfig,
    *,
    quiet: bool = False,
    console: Console | None = None,
) -> Backend:
    """Create the appropriate execution backend from job config.

    Args:
        config: Job configuration with backend settings.
        quiet: If True, suppress verbose logging.
        console: Console for verbose output.

    Returns:
        Configured Backend instance.
    """
    from mozart.backends.anthropic_api import AnthropicApiBackend
    from mozart.backends.claude_cli import ClaudeCliBackend
    from mozart.backends.recursive_light import RecursiveLightBackend

    if config.backend.type == "recursive_light":
        rl_config = config.backend.recursive_light
        backend: Backend = RecursiveLightBackend.from_config(config.backend)
        if not quiet and is_verbose() and console:
            console.print(
                f"[dim]Using Recursive Light backend at {rl_config.endpoint}[/dim]"
            )
    elif config.backend.type == "anthropic_api":
        backend = AnthropicApiBackend.from_config(config.backend)
        if not quiet and is_verbose() and console:
            console.print(
                f"[dim]Using Anthropic API backend with model {config.backend.model}[/dim]"
            )
    else:
        backend = ClaudeCliBackend.from_config(config.backend)

    return backend


def setup_learning(
    config: JobConfig,
    *,
    quiet: bool = False,
    console: Console | None = None,
) -> tuple[OutcomeStore | None, GlobalLearningStore | None]:
    """Setup outcome store and global learning store if learning is enabled.

    Args:
        config: Job configuration with learning settings.
        quiet: If True, suppress verbose logging.
        console: Console for verbose output.

    Returns:
        Tuple of (outcome_store, global_learning_store), either may be None.
    """
    outcome_store: OutcomeStore | None = None
    global_learning_store: GlobalLearningStore | None = None

    if config.learning.enabled:
        from mozart.learning.global_store import get_global_store
        from mozart.learning.outcomes import JsonOutcomeStore

        outcome_store_path = config.get_outcome_store_path()
        if config.learning.outcome_store_type == "json":
            outcome_store = JsonOutcomeStore(outcome_store_path)

        global_learning_store = get_global_store()

        if not quiet and is_verbose() and console:
            console.print(
                f"[dim]Learning enabled: outcomes at {outcome_store_path}[/dim]"
            )
            console.print(
                "[dim]Global learning enabled: cross-workspace patterns active[/dim]"
            )

    return outcome_store, global_learning_store


def setup_notifications(
    config: JobConfig,
    *,
    quiet: bool = False,
    console: Console | None = None,
) -> NotificationManager | None:
    """Setup notification manager from config.

    Args:
        config: Job configuration with notification settings.
        quiet: If True, suppress verbose logging.
        console: Console for verbose output.

    Returns:
        NotificationManager if notifications configured, else None.
    """
    if not config.notifications:
        return None

    notifiers = create_notifiers_from_config(config.notifications)
    if not notifiers:
        return None

    notification_manager = NotificationManager(notifiers)
    if not quiet and is_verbose() and console:
        console.print(
            f"[dim]Notifications enabled: {len(notifiers)} channel(s) configured[/dim]"
        )
    return notification_manager


def setup_escalation(
    config: JobConfig,
    *,
    enabled: bool = False,
    quiet: bool = False,
    console: Console | None = None,
) -> ConsoleEscalationHandler | None:
    """Setup escalation handler if enabled.

    Args:
        config: Job configuration with learning settings.
        enabled: Whether escalation is explicitly enabled via CLI flag.
        quiet: If True, suppress verbose logging.
        console: Console for verbose output.

    Returns:
        ConsoleEscalationHandler if enabled, else None.
    """
    if not enabled:
        return None

    from mozart.execution.escalation import ConsoleEscalationHandler

    config.learning.escalation_enabled = True
    handler = ConsoleEscalationHandler(
        confidence_threshold=config.learning.min_confidence_threshold,
        auto_retry_on_first_failure=True,
    )
    if not quiet and is_verbose() and console:
        console.print(
            "[dim]Escalation enabled: low-confidence sheets will prompt for decisions[/dim]"
        )
    return handler


def setup_grounding(
    config: JobConfig,
    *,
    quiet: bool = False,
    console: Console | None = None,
) -> GroundingEngine | None:
    """Setup grounding engine with hooks from config.

    Args:
        config: Job configuration with grounding settings.
        quiet: If True, suppress verbose logging.
        console: Console for verbose output.

    Returns:
        GroundingEngine if grounding enabled, else None.
    """
    if not config.grounding.enabled:
        return None

    from mozart.execution.grounding import GroundingEngine, create_hook_from_config

    engine = GroundingEngine(hooks=[], config=config.grounding)

    for hook_config in config.grounding.hooks:
        try:
            hook = create_hook_from_config(hook_config)
            engine.add_hook(hook)
            if not quiet and is_verbose() and console:
                console.print(f"[dim]  Registered hook: {hook.name}[/dim]")
        except ValueError as e:
            if console:
                console.print(f"[yellow]Warning: Failed to create hook: {e}[/yellow]")

    if not quiet and is_verbose() and console:
        hook_count = engine.get_hook_count()
        console.print(
            f"[dim]Grounding enabled: {hook_count} hook(s) registered[/dim]"
        )

    return engine


def display_run_summary(summary: RunSummary) -> None:
    """Display run summary as a rich panel.

    Shared by both `run` and `resume` commands to avoid duplication.

    Args:
        summary: Run summary with execution statistics.
    """
    if is_quiet():
        return

    status_color = {
        JobStatus.COMPLETED: "green",
        JobStatus.FAILED: "red",
        JobStatus.PAUSED: "yellow",
    }.get(summary.final_status, "white")

    status_text = f"[{status_color}]{summary.final_status.value.upper()}[/{status_color}]"

    lines = [
        f"[bold]{summary.job_name}[/bold]",
        f"Status: {status_text}",
        f"Duration: {summary._format_duration(summary.total_duration_seconds)}",
        "",
        "[bold]Sheets[/bold]",
        f"  Completed: [green]{summary.completed_sheets}[/green]/{summary.total_sheets}",
    ]

    if summary.failed_sheets > 0:
        lines.append(f"  Failed: [red]{summary.failed_sheets}[/red]")
    if summary.skipped_sheets > 0:
        lines.append(f"  Skipped: [yellow]{summary.skipped_sheets}[/yellow]")

    lines.append(f"  Success Rate: {summary.success_rate:.1f}%")

    if summary.validation_pass_count + summary.validation_fail_count > 0:
        lines.extend([
            "",
            "[bold]Validation[/bold]",
            f"  Pass Rate: {summary.validation_pass_rate:.1f}%",
        ])

    if is_verbose() or summary.total_retries > 0 or summary.rate_limit_waits > 0:
        lines.extend([
            "",
            "[bold]Execution[/bold]",
        ])
        if summary.first_attempt_successes > 0:
            lines.append(
                f"  First Attempt Success: {summary.first_attempt_rate:.0f}% "
                f"({summary.first_attempt_successes}/{summary.completed_sheets})"
            )
        if summary.total_retries > 0:
            lines.append(f"  Retries Used: {summary.total_retries}")
        if summary.total_completion_attempts > 0:
            lines.append(f"  Completion Attempts: {summary.total_completion_attempts}")
        if summary.rate_limit_waits > 0:
            lines.append(f"  Rate Limit Waits: [yellow]{summary.rate_limit_waits}[/yellow]")

    default_console.print(Panel(
        "\n".join(lines),
        title="Run Summary",
        border_style="green" if summary.final_status == JobStatus.COMPLETED else "yellow",
    ))
