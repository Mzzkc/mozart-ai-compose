"""Observer event recorder — persists per-job observer events to JSONL.

Subscribes to ``observer.*`` events on the EventBus, applies path
exclusion and coalescing, then writes to per-job JSONL files inside
each job's workspace. Also maintains an in-memory ring buffer for
real-time TUI consumption via IPC.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
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

    def register_job(self, job_id: str, workspace: Path) -> None:
        """Start recording events for a job."""
        if job_id in self._jobs:
            return
        state = _JobRecorderState(job_id, workspace)
        if self._config.persist_events:
            jsonl_path = workspace / ".mozart-observer.jsonl"
            try:
                state.file_handle = open(jsonl_path, "a+", encoding="utf-8")  # noqa: SIM115
            except OSError:
                _logger.warning(
                    "observer_recorder.open_failed",
                    job_id=job_id,
                    path=str(jsonl_path),
                    exc_info=True,
                )
        self._jobs[job_id] = state
        _logger.info("observer_recorder.registered", job_id=job_id)

    def unregister_job(self, job_id: str) -> None:
        """Stop recording events for a job, flush and close file."""
        state = self._jobs.pop(job_id, None)
        if state is None:
            return
        self._flush_state(state)
        self._close_state(state)
        _logger.info("observer_recorder.unregistered", job_id=job_id)

    def flush(self, job_id: str) -> None:
        """Flush coalesce buffer and file handle for a job."""
        state = self._jobs.get(job_id)
        if state is None:
            return
        self._flush_state(state)

    def _handle_event(self, event: ObserverEvent) -> None:
        """EventBus callback — route events through coalescing or direct write."""
        job_id = event.get("job_id", "")
        state = self._jobs.get(job_id)
        if state is None:
            _logger.debug(
                "observer_recorder.event_after_unregister",
                job_id=job_id,
                event=event.get("event"),
            )
            return

        # Check path exclusion for file events
        event_type = event.get("event", "")
        if event_type in _FILE_EVENTS:
            data = event.get("data") or {}
            rel_path = data.get("path", "")
            if rel_path and self._should_exclude(rel_path):
                return

        # REVIEW FIX 3: Always add to ring buffer immediately, even during
        # coalescing. Without this, the TUI hides active files.
        state.recent_events.append(event)

        # Only file_modified events are coalesced
        if (
            event_type == "observer.file_modified"
            and self._config.coalesce_window_seconds > 0
        ):
            self._coalesce_or_buffer(state, event)
        else:
            self._write_event_to_file(state, event)

        # REVIEW FIX 2: Check size cap after writes; defer truncation to avoid
        # blocking the EventBus drain loop.
        if (
            state.file_handle is not None
            and not state.truncating
            and state.file_handle.tell() > self._config.max_timeline_bytes
        ):
            state.truncating = True
            try:
                asyncio.create_task(self._enforce_size_cap_async(job_id))
            except RuntimeError:
                # No running event loop (e.g. sync test context) — run inline
                self._enforce_size_cap_sync(job_id)
                state.truncating = False

    def _coalesce_or_buffer(
        self, state: _JobRecorderState, event: ObserverEvent,
    ) -> None:
        """Buffer a file_modified event, coalescing with prior same-path event."""
        data = event.get("data") or {}
        path = data.get("path", "")
        # REVIEW FIX 4: Compare event['timestamp'], not wall clock
        event_ts = event["timestamp"]

        if path in state.coalesce_buffer:
            prev_ts, _prev_event = state.coalesce_buffer[path]
            if (event_ts - prev_ts) <= self._config.coalesce_window_seconds:
                # Within window — replace buffered event (coalesce)
                state.coalesce_buffer[path] = (event_ts, event)
                return
            # Window expired — flush previous event to file, then buffer new
            self._write_event_to_file(state, _prev_event)

        state.coalesce_buffer[path] = (event_ts, event)

    def _write_event(self, job_id: str, event: ObserverEvent) -> None:
        """Write a single event to JSONL and ring buffer (if not excluded)."""
        state = self._jobs.get(job_id)
        if state is None:
            _logger.debug(
                "observer_recorder.event_for_unregistered_job",
                job_id=job_id,
                event_type=event.get("event", ""),
            )
            return

        # Check path exclusion for file events
        event_type = event.get("event", "")
        if event_type in _FILE_EVENTS:
            data = event.get("data") or {}
            rel_path = data.get("path", "")
            if rel_path and self._should_exclude(rel_path):
                return

        # Add to ring buffer BEFORE file write attempt.
        state.recent_events.append(event)

        # Write to JSONL
        self._write_event_to_file(state, event)

    def _write_event_to_file(
        self, state: _JobRecorderState, event: ObserverEvent,
    ) -> None:
        """Write a single event to the JSONL file handle."""
        if state.file_handle is not None:
            try:
                line = json.dumps(event, separators=(",", ":")) + "\n"
                state.file_handle.write(line)
            except OSError:
                _logger.warning(
                    "observer_recorder.write_failed",
                    job_id=state.job_id,
                    exc_info=True,
                )

    def _enforce_size_cap_sync(self, job_id: str) -> None:
        """Truncate JSONL preserving line boundaries (synchronous).

        REVIEW FIX 1: After seeking to midpoint, readline() consumes the
        partial line so every surviving line is a complete JSON object.

        Because the file is opened in append mode (writes always go to end),
        we close the handle, rewrite via a temp file, then reopen in a+ mode.
        """
        state = self._jobs.get(job_id)
        if state is None or state.file_handle is None:
            return
        try:
            fh = state.file_handle
            cap = self._config.max_timeline_bytes
            fh.flush()

            jsonl_path = state.workspace / ".mozart-observer.jsonl"
            size = jsonl_path.stat().st_size
            if size <= cap:
                return

            # Close the append-mode handle so we can rewrite
            fh.close()
            state.file_handle = None

            # Read the file content and find the surviving tail
            content = jsonl_path.read_text(encoding="utf-8")
            # Repeatedly halve until under cap
            while len(content.encode("utf-8")) > cap:
                mid = len(content) // 2
                nl = content.find("\n", mid)
                if nl == -1:
                    content = ""
                    break
                content = content[nl + 1:]

            # Rewrite the file with surviving content
            jsonl_path.write_text(content, encoding="utf-8")

            # Reopen in a+ mode for continued appending
            state.file_handle = open(jsonl_path, "a+", encoding="utf-8")  # noqa: SIM115
        except OSError:
            _logger.warning(
                "observer_recorder.truncate_failed",
                job_id=job_id,
                exc_info=True,
            )

    async def _enforce_size_cap_async(self, job_id: str) -> None:
        """REVIEW FIX 2: Async wrapper for size cap enforcement.

        Defers truncation to a background task so it doesn't block the
        EventBus drain loop.
        """
        try:
            self._enforce_size_cap_sync(job_id)
        finally:
            state = self._jobs.get(job_id)
            if state is not None:
                state.truncating = False

    def _flush_state(self, state: _JobRecorderState) -> None:
        """Flush coalesce buffer and file handle."""
        # Drain coalesce buffer
        for _path, (_ts, event) in state.coalesce_buffer.items():
            if state.file_handle is not None:
                try:
                    line = json.dumps(event, separators=(",", ":")) + "\n"
                    state.file_handle.write(line)
                except OSError:
                    pass
        state.coalesce_buffer.clear()

        # Flush file to OS page cache
        if state.file_handle is not None:
            try:
                state.file_handle.flush()
            except OSError:
                pass

    def _close_state(self, state: _JobRecorderState) -> None:
        """Close the file handle."""
        if state.file_handle is not None:
            try:
                state.file_handle.close()
            except OSError:
                pass
            state.file_handle = None

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

    __slots__ = (
        "job_id", "workspace", "file_handle", "recent_events",
        "coalesce_buffer", "truncating",
    )

    def __init__(self, job_id: str, workspace: Path) -> None:
        self.job_id = job_id
        self.workspace = workspace
        self.file_handle: Any = None
        self.recent_events: deque[ObserverEvent] = deque(maxlen=200)
        self.coalesce_buffer: dict[str, tuple[float, ObserverEvent]] = {}
        self.truncating: bool = False


__all__ = ["ObserverRecorder"]
