"""Observer event recorder — persists per-job observer events to JSONL.

Subscribes to ``observer.*`` events on the EventBus, applies path
exclusion and coalescing, then writes to per-job JSONL files inside
each job's workspace. Also maintains an in-memory ring buffer for
real-time TUI consumption via IPC.
"""

from __future__ import annotations

import fnmatch
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mozart.core.logging import get_logger
from mozart.daemon.config import ObserverConfig

if TYPE_CHECKING:
    from mozart.daemon.event_bus import EventBus
    from mozart.daemon.types import ObserverEvent

_logger = get_logger("daemon.observer_recorder")

_FILE_EVENTS = frozenset({
    "observer.file_created",
    "observer.file_modified",
    "observer.file_deleted",
})

_PROCESS_EVENTS = frozenset({
    "observer.process_spawned",
    "observer.process_exited",
})

_ALL_OBSERVER_EVENTS = _FILE_EVENTS | _PROCESS_EVENTS


class ObserverRecorder:
    """EventBus subscriber that persists observer events to per-job JSONL files."""

    def __init__(self, config: ObserverConfig, event_bus: EventBus) -> None:
        self._config = config
        self._event_bus = event_bus
        self._sub_id: str | None = None
        self._jobs: dict[str, _JobRecorderState] = {}

    def _should_exclude(self, rel_path: str) -> bool:
        """Check if a relative path matches any exclusion pattern."""
        for pattern in self._config.exclude_patterns:
            # Match against full path
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            # Match with wildcard prefix (e.g. */pattern)
            if fnmatch.fnmatch(rel_path, f"*/{pattern}"):
                return True
            # Check each path component (for directory patterns like ".git/")
            parts = rel_path.replace("\\", "/").split("/")
            for part in parts:
                if fnmatch.fnmatch(part + "/", pattern):
                    return True
                if fnmatch.fnmatch(part, pattern):
                    return True
        return False


class _JobRecorderState:
    """Per-job state for the observer recorder."""

    __slots__ = ("job_id", "workspace", "file_handle", "recent_events", "coalesce_buffer")

    def __init__(self, job_id: str, workspace: Path) -> None:
        self.job_id = job_id
        self.workspace = workspace
        self.file_handle: Any = None
        self.recent_events: deque[ObserverEvent] = deque(maxlen=200)
        self.coalesce_buffer: dict[str, tuple[float, ObserverEvent]] = {}


__all__ = ["ObserverRecorder"]
