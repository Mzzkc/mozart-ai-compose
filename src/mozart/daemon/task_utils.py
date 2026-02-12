"""Shared utilities for asyncio.Task lifecycle in the daemon.

Provides ``log_task_exception`` to extract and log exceptions from
completed tasks — used by done-callbacks across the daemon to avoid
silently losing errors from background tasks.
"""

from __future__ import annotations

import asyncio
from typing import Any


def log_task_exception(
    task: asyncio.Task[Any],
    logger: Any,
    event: str,
    *,
    level: str = "error",
) -> BaseException | None:
    """Extract and log an exception from a completed task.

    Call this at the top of any ``add_done_callback`` handler to
    consistently surface exceptions from background tasks.

    Args:
        task: The completed task to inspect.
        logger: A structlog or stdlib logger with ``.error()``/``.warning()`` methods.
        event: Structlog-style event name (e.g. ``"monitor.loop_died"``).
        level: Log method name — ``"error"`` (default) or ``"warning"``.

    Returns:
        The exception if one was found, ``None`` if the task completed
        normally or was cancelled.
    """
    if task.cancelled():
        return None
    exc = task.exception()
    if exc is not None:
        log_fn = getattr(logger, level, logger.error)
        log_fn(event, error=str(exc), task_name=task.get_name())
    return exc
