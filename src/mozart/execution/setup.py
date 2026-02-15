"""Shared execution setup — pure functions with no UI dependencies.

Consolidates the component creation logic that was duplicated between
``cli/commands/_shared.py`` and ``daemon/job_service.py``.  Both paths
now call these functions:

- **CLI** wraps them with Rich console output for verbosity.
- **Daemon** calls them directly (no console).

This eliminates the "mirrors _shared.py" comments in job_service.py and
ensures that adding a new backend type only requires updating one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mozart.core.logging import get_logger

if TYPE_CHECKING:
    from mozart.backends.base import Backend
    from mozart.core.config import JobConfig
    from mozart.execution.grounding import GroundingEngine
    from mozart.learning.global_store import GlobalLearningStore
    from mozart.learning.outcomes import OutcomeStore
    from mozart.notifications.base import NotificationManager
    from mozart.state.base import StateBackend

_logger = get_logger("execution.setup")


def create_backend(config: JobConfig) -> Backend:
    """Create the appropriate execution backend from job config.

    Supports: claude_cli, anthropic_api, recursive_light.

    Args:
        config: Job configuration with backend settings.

    Returns:
        Configured Backend instance.
    """
    from mozart.backends.anthropic_api import AnthropicApiBackend
    from mozart.backends.claude_cli import ClaudeCliBackend
    from mozart.backends.recursive_light import RecursiveLightBackend

    if config.backend.type == "recursive_light":
        return RecursiveLightBackend.from_config(config.backend)
    elif config.backend.type == "anthropic_api":
        return AnthropicApiBackend.from_config(config.backend)
    else:
        return ClaudeCliBackend.from_config(config.backend)


def setup_learning(
    config: JobConfig,
    *,
    global_learning_store_override: GlobalLearningStore | None = None,
) -> tuple[OutcomeStore | None, GlobalLearningStore | None]:
    """Setup outcome store and global learning store if learning is enabled.

    Args:
        config: Job configuration with learning settings.
        global_learning_store_override: If provided, use this store instead
            of the module-level singleton.  The daemon injects its shared
            LearningHub store here to avoid opening a second SQLite connection.

    Returns:
        Tuple of (outcome_store, global_learning_store), either may be None.
    """
    if not config.learning.enabled:
        return None, None

    from mozart.learning.outcomes import JsonOutcomeStore

    outcome_store: OutcomeStore | None = None
    outcome_store_path = config.get_outcome_store_path()
    if config.learning.outcome_store_type == "json":
        outcome_store = JsonOutcomeStore(outcome_store_path)

    # Prefer injected store (from daemon LearningHub) over the
    # module-level singleton.  This avoids opening a second
    # SQLite connection when the daemon already owns one.
    if global_learning_store_override is not None:
        global_learning_store = global_learning_store_override
    else:
        from mozart.learning.global_store import get_global_store

        global_learning_store = get_global_store()

    return outcome_store, global_learning_store


def setup_notifications(config: JobConfig) -> NotificationManager | None:
    """Setup notification manager from config.

    Args:
        config: Job configuration with notification settings.

    Returns:
        NotificationManager if notifications configured, else None.
    """
    if not config.notifications:
        return None

    from mozart.notifications import NotificationManager
    from mozart.notifications.factory import create_notifiers_from_config

    notifiers = create_notifiers_from_config(config.notifications)
    if not notifiers:
        return None

    return NotificationManager(notifiers)


def setup_grounding(config: JobConfig) -> GroundingEngine | None:
    """Setup grounding engine with hooks from config.

    Args:
        config: Job configuration with grounding settings.

    Returns:
        GroundingEngine if grounding enabled, else None.
    """
    if not config.grounding.enabled:
        return None

    from mozart.execution.grounding import GroundingEngine, create_hook_from_config

    engine = GroundingEngine(hooks=[], config=config.grounding)
    failed_count = 0

    for hook_config in config.grounding.hooks:
        try:
            hook = create_hook_from_config(hook_config)
            engine.add_hook(hook)
        except ValueError as e:
            failed_count += 1
            _logger.warning(
                "failed_to_create_hook",
                hook_type=getattr(hook_config, "type", "unknown"),
                error=str(e),
                exc_info=True,
            )

    if failed_count:
        _logger.error(
            "grounding_hooks_partial_failure",
            failed=failed_count,
            total=len(config.grounding.hooks),
            loaded=len(config.grounding.hooks) - failed_count,
        )

    return engine


def create_state_backend(
    workspace: Path,
    backend_type: str = "json",
) -> StateBackend:
    """Create state persistence backend.

    Args:
        workspace: Workspace directory for state files.
        backend_type: "json" or "sqlite".

    Returns:
        Configured StateBackend instance.
    """
    from mozart.state import JsonStateBackend, SQLiteStateBackend

    if backend_type == "sqlite":
        return SQLiteStateBackend(workspace / ".mozart-state.db")
    else:
        return JsonStateBackend(workspace)


__all__ = [
    "create_backend",
    "create_state_backend",
    "setup_grounding",
    "setup_learning",
    "setup_notifications",
]
