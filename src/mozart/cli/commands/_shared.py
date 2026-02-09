"""Shared execution setup logic for run and resume commands.

Extracts the common infrastructure setup that both _run_job() and _resume_job()
need: backend creation, learning stores, notification manager, escalation handler,
and grounding engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from mozart.notifications import NotificationManager

from ..helpers import (
    create_notifiers_from_config,
    is_verbose,
)

if TYPE_CHECKING:
    from mozart.backends.base import Backend
    from mozart.core.config import JobConfig
    from mozart.execution.escalation import ConsoleEscalationHandler
    from mozart.execution.grounding import GroundingEngine
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
        backend: Backend = RecursiveLightBackend(
            rl_endpoint=rl_config.endpoint,
            user_id=rl_config.user_id,
            timeout=rl_config.timeout,
        )
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
