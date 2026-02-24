# Observer Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire JobObserver events into persistent per-job JSONL storage, `mozart top` TUI display, and completion snapshots — closing all three gaps from [#102](https://github.com/Mzzkc/mozart-ai-compose/issues/102).

**Architecture:** New `ObserverRecorder` EventBus subscriber handles persistence (per-job JSONL) and TUI data access (in-memory ring buffer). Snapshot enrichment adds JSONL capture + git context. TUI integration routes process events to timeline, file events to detail panel.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, Textual (TUI), watchfiles/psutil (optional deps)

**Design Doc:** `docs/plans/2026-02-24-observer-integration-design.md`

---

### Task 1: ObserverConfig — Add Persistence Fields

**Files:**
- Modify: `src/mozart/daemon/config.py:84-105` (ObserverConfig class)
- Test: `tests/test_observer_recorder.py` (new file, first tests)

**Step 1: Write the failing test**

Create `tests/test_observer_recorder.py`:

```python
"""Tests for ObserverRecorder and ObserverConfig persistence fields."""

from __future__ import annotations

import pytest

from mozart.daemon.config import ObserverConfig


class TestObserverConfigPersistence:
    """Verify new persistence fields on ObserverConfig."""

    def test_defaults(self) -> None:
        config = ObserverConfig()
        assert config.persist_events is True
        assert ".git/" in config.exclude_patterns
        assert "__pycache__/" in config.exclude_patterns
        assert "node_modules/" in config.exclude_patterns
        assert ".venv/" in config.exclude_patterns
        assert "*.pyc" in config.exclude_patterns
        assert config.coalesce_window_seconds == 2.0
        assert config.max_timeline_bytes == 10_485_760

    def test_disable_persistence(self) -> None:
        config = ObserverConfig(persist_events=False)
        assert config.persist_events is False

    def test_custom_exclude_patterns(self) -> None:
        config = ObserverConfig(exclude_patterns=[".git/", "build/"])
        assert config.exclude_patterns == [".git/", "build/"]

    def test_coalesce_window_minimum(self) -> None:
        config = ObserverConfig(coalesce_window_seconds=0.0)
        assert config.coalesce_window_seconds == 0.0

    def test_max_timeline_bytes_minimum(self) -> None:
        with pytest.raises(Exception):  # Pydantic validation error
            ObserverConfig(max_timeline_bytes=100)  # Below 1MB minimum
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_observer_recorder.py::TestObserverConfigPersistence -v`
Expected: FAIL — `persist_events` attribute not found on ObserverConfig

**Step 3: Write minimal implementation**

In `src/mozart/daemon/config.py`, add fields to `ObserverConfig` (after line 105):

```python
class ObserverConfig(BaseModel):
    """Configuration for the job observer and event bus."""

    enabled: bool = Field(
        default=True,
        description="Enable event bus and observer infrastructure.",
    )
    watch_interval_seconds: float = Field(
        default=2.0,
        ge=0.5,
        description="Interval between observer filesystem checks.",
    )
    snapshot_ttl_hours: int = Field(
        default=168,
        ge=1,
        description="Hours to keep completion snapshots (default 1 week).",
    )
    max_queue_size: int = Field(
        default=1000,
        ge=100,
        description="Maximum events per subscriber before drop-oldest.",
    )
    persist_events: bool = Field(
        default=True,
        description="Persist observer events to per-job JSONL timeline files.",
    )
    exclude_patterns: list[str] = Field(
        default=[".git/", "__pycache__/", "node_modules/", ".venv/", "*.pyc"],
        description="Glob patterns for workspace paths to exclude from event persistence. "
        "Matched against relative paths using fnmatch.",
    )
    coalesce_window_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Window for coalescing rapid same-file modification events. "
        "Multiple edits to the same file within this window produce one event.",
    )
    max_timeline_bytes: int = Field(
        default=10_485_760,
        ge=1_048_576,
        description="Maximum JSONL timeline file size per job (default 10MB). "
        "Oldest events are truncated when exceeded.",
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_observer_recorder.py::TestObserverConfigPersistence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/daemon/config.py tests/test_observer_recorder.py
git commit -m "feat(observer): add persistence config fields to ObserverConfig (#102)"
```

---

### Task 2: ObserverRecorder — Path Exclusion + Core Structure

**Files:**
- Create: `src/mozart/daemon/observer_recorder.py`
- Test: `tests/test_observer_recorder.py` (add tests)

**Step 1: Write the failing tests**

Add to `tests/test_observer_recorder.py`:

```python
import fnmatch
from pathlib import Path
from unittest.mock import AsyncMock

from mozart.daemon.config import ObserverConfig
from mozart.daemon.observer_recorder import ObserverRecorder


class TestPathExclusion:
    """Verify path exclusion filtering."""

    def _make_recorder(self, **config_kwargs: object) -> ObserverRecorder:
        config = ObserverConfig(**config_kwargs)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_default_excludes_git(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude(".git/objects/abc123")

    def test_default_excludes_pycache(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude("src/__pycache__/foo.cpython-312.pyc")

    def test_default_excludes_node_modules(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude("node_modules/lodash/index.js")

    def test_default_excludes_venv(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude(".venv/lib/python3.12/site.py")

    def test_default_excludes_pyc(self) -> None:
        recorder = self._make_recorder()
        assert recorder._should_exclude("src/foo.pyc")

    def test_allows_normal_paths(self) -> None:
        recorder = self._make_recorder()
        assert not recorder._should_exclude("src/main.py")
        assert not recorder._should_exclude("output-3.md")
        assert not recorder._should_exclude("tests/test_foo.py")

    def test_custom_patterns(self) -> None:
        recorder = self._make_recorder(exclude_patterns=["build/", "*.tmp"])
        assert recorder._should_exclude("build/output.js")
        assert recorder._should_exclude("data.tmp")
        assert not recorder._should_exclude(".git/HEAD")  # Not in custom list

    def test_empty_patterns_excludes_nothing(self) -> None:
        recorder = self._make_recorder(exclude_patterns=[])
        assert not recorder._should_exclude(".git/HEAD")
        assert not recorder._should_exclude("src/__pycache__/foo.pyc")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_observer_recorder.py::TestPathExclusion -v`
Expected: FAIL — `observer_recorder` module not found

**Step 3: Write minimal implementation**

Create `src/mozart/daemon/observer_recorder.py`:

```python
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
            # Match against full path and each path component
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(rel_path, f"*/{pattern}"):
                return True
            # Check if any path component matches (for directory patterns like ".git/")
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_observer_recorder.py::TestPathExclusion -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/daemon/observer_recorder.py tests/test_observer_recorder.py
git commit -m "feat(observer): add ObserverRecorder with path exclusion (#102)"
```

---

### Task 3: ObserverRecorder — JSONL Persistence + Register/Unregister

**Files:**
- Modify: `src/mozart/daemon/observer_recorder.py`
- Test: `tests/test_observer_recorder.py` (add tests)

**Step 1: Write the failing tests**

Add to `tests/test_observer_recorder.py`:

```python
import json
import time

class TestJSONLPersistence:
    """Verify JSONL write, register/unregister lifecycle."""

    def _make_recorder(self, **config_kwargs: object) -> ObserverRecorder:
        config = ObserverConfig(**config_kwargs)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_register_creates_state(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        assert "job-1" in recorder._jobs

    def test_register_opens_jsonl(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        assert jsonl_path.exists()

    def test_unregister_closes_file(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        recorder.unregister_job("job-1")
        assert "job-1" not in recorder._jobs

    def test_write_event_produces_jsonl_line(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_created",
            "data": {"path": "output.md"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event"] == "observer.file_created"

    def test_excluded_path_not_written(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1",
            "sheet_num": 0,
            "event": "observer.file_modified",
            "data": {"path": ".git/objects/abc123"},
            "timestamp": time.time(),
        }
        recorder._write_event("job-1", event)
        recorder.flush("job-1")

        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        content = jsonl_path.read_text().strip()
        assert content == ""

    def test_unregister_unknown_job_is_noop(self) -> None:
        recorder = self._make_recorder()
        recorder.unregister_job("nonexistent")  # Should not raise

    def test_disabled_persistence_skips_file(self, tmp_path: Path) -> None:
        recorder = self._make_recorder(persist_events=False)
        recorder.register_job("job-1", tmp_path)
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        assert not jsonl_path.exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_observer_recorder.py::TestJSONLPersistence -v`
Expected: FAIL — `register_job` not implemented

**Step 3: Write implementation**

Add to `ObserverRecorder` in `observer_recorder.py`:

```python
import json
import time

class ObserverRecorder:
    # ... existing __init__ and _should_exclude ...

    def register_job(self, job_id: str, workspace: Path) -> None:
        """Start recording events for a job."""
        if job_id in self._jobs:
            return
        state = _JobRecorderState(job_id, workspace)
        if self._config.persist_events:
            jsonl_path = workspace / ".mozart-observer.jsonl"
            try:
                state.file_handle = open(jsonl_path, "a", encoding="utf-8")
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
        """Flush coalesce buffer and fsync for a job."""
        state = self._jobs.get(job_id)
        if state is None:
            return
        self._flush_state(state)

    def _write_event(self, job_id: str, event: ObserverEvent) -> None:
        """Write a single event to JSONL and ring buffer (if not excluded)."""
        state = self._jobs.get(job_id)
        if state is None:
            return

        # Check path exclusion for file events
        event_type = event.get("event", "")
        if event_type in _FILE_EVENTS:
            data = event.get("data") or {}
            rel_path = data.get("path", "")
            if rel_path and self._should_exclude(rel_path):
                return

        # Add to ring buffer
        state.recent_events.append(event)

        # Write to JSONL
        if state.file_handle is not None:
            try:
                line = json.dumps(event, separators=(",", ":")) + "\n"
                state.file_handle.write(line)
            except OSError:
                _logger.warning(
                    "observer_recorder.write_failed",
                    job_id=job_id,
                    exc_info=True,
                )

    def _flush_state(self, state: _JobRecorderState) -> None:
        """Flush coalesce buffer and file handle."""
        # Flush coalesce buffer
        for _path, (_ts, event) in state.coalesce_buffer.items():
            if state.file_handle is not None:
                try:
                    line = json.dumps(event, separators=(",", ":")) + "\n"
                    state.file_handle.write(line)
                except OSError:
                    pass
        state.coalesce_buffer.clear()

        # Fsync file
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_observer_recorder.py::TestJSONLPersistence -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/daemon/observer_recorder.py tests/test_observer_recorder.py
git commit -m "feat(observer): JSONL persistence with register/unregister lifecycle (#102)"
```

---

### Task 4: ObserverRecorder — Coalescing + Size Cap

**Files:**
- Modify: `src/mozart/daemon/observer_recorder.py`
- Test: `tests/test_observer_recorder.py` (add tests)

**Step 1: Write the failing tests**

```python
class TestCoalescing:
    """Verify same-file modification coalescing."""

    def _make_recorder(self, **kw: object) -> ObserverRecorder:
        config = ObserverConfig(**kw)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_rapid_same_file_mods_coalesce(self, tmp_path: Path) -> None:
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(5):
            event: ObserverEvent = {
                "job_id": "job-1", "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": "same-file.md"},
                "timestamp": now + i * 0.1,  # 100ms apart
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [l for l in jsonl_path.read_text().strip().split("\n") if l]
        # Should produce 1 coalesced event, not 5
        assert len(lines) == 1

    def test_different_files_not_coalesced(self, tmp_path: Path) -> None:
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            event: ObserverEvent = {
                "job_id": "job-1", "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": f"file-{i}.md"},
                "timestamp": now + i * 0.1,
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [l for l in jsonl_path.read_text().strip().split("\n") if l]
        assert len(lines) == 3

    def test_process_events_not_coalesced(self, tmp_path: Path) -> None:
        recorder = self._make_recorder(coalesce_window_seconds=2.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            event: ObserverEvent = {
                "job_id": "job-1", "sheet_num": 0,
                "event": "observer.process_spawned",
                "data": {"pid": 1000 + i, "name": "pytest"},
                "timestamp": now + i * 0.1,
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [l for l in jsonl_path.read_text().strip().split("\n") if l]
        assert len(lines) == 3  # Process events always write immediately

    def test_zero_coalesce_window_disables(self, tmp_path: Path) -> None:
        recorder = self._make_recorder(coalesce_window_seconds=0.0)
        recorder.register_job("job-1", tmp_path)
        now = time.time()
        for i in range(3):
            event: ObserverEvent = {
                "job_id": "job-1", "sheet_num": 0,
                "event": "observer.file_modified",
                "data": {"path": "same-file.md"},
                "timestamp": now + i * 0.1,
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        lines = [l for l in jsonl_path.read_text().strip().split("\n") if l]
        assert len(lines) == 3  # No coalescing


class TestSizeCap:
    """Verify JSONL size cap and truncation."""

    def _make_recorder(self, **kw: object) -> ObserverRecorder:
        config = ObserverConfig(**kw)
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_truncates_when_over_cap(self, tmp_path: Path) -> None:
        # 2KB cap — will trigger truncation quickly
        recorder = self._make_recorder(
            max_timeline_bytes=2048, coalesce_window_seconds=0.0,
        )
        recorder.register_job("job-1", tmp_path)

        # Write enough events to exceed 2KB
        for i in range(100):
            event: ObserverEvent = {
                "job_id": "job-1", "sheet_num": 0,
                "event": "observer.file_created",
                "data": {"path": f"file-{i:04d}.md"},
                "timestamp": time.time(),
            }
            recorder._handle_event(event)

        recorder.flush("job-1")
        jsonl_path = tmp_path / ".mozart-observer.jsonl"
        size = jsonl_path.stat().st_size
        assert size <= 2048
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_observer_recorder.py::TestCoalescing tests/test_observer_recorder.py::TestSizeCap -v`
Expected: FAIL — `_handle_event` not defined

**Step 3: Implement coalescing and size cap**

Add `_handle_event()` method to `ObserverRecorder` (the EventBus callback) and the truncation logic. This is the method that the EventBus calls — it routes to coalesce buffer for file_modified, direct write for everything else, and checks size cap after writes.

The key logic:
- `_handle_event(event)` — EventBus callback. Extracts job_id, dispatches to `_write_event` or coalesce buffer.
- For `observer.file_modified` with coalesce window > 0: buffer the event keyed by path. If a previous event for the same path exists within the window, replace it.
- For all other events: call `_write_event` directly.
- After writing, check file size against cap. If exceeded, truncate oldest half.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_observer_recorder.py::TestCoalescing tests/test_observer_recorder.py::TestSizeCap -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/mozart/daemon/observer_recorder.py tests/test_observer_recorder.py
git commit -m "feat(observer): coalescing + JSONL size cap in ObserverRecorder (#102)"
```

---

### Task 5: ObserverRecorder — EventBus Start/Stop + get_recent_events

**Files:**
- Modify: `src/mozart/daemon/observer_recorder.py`
- Test: `tests/test_observer_recorder.py`

**Step 1: Write the failing tests**

```python
import asyncio

class TestLifecycle:
    """Verify start/stop subscribes/unsubscribes from EventBus."""

    @pytest.mark.asyncio
    async def test_start_subscribes(self) -> None:
        config = ObserverConfig()
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-123"
        recorder = ObserverRecorder(config=config, event_bus=bus)
        await recorder.start()
        assert recorder._sub_id == "sub-123"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        config = ObserverConfig()
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-123"
        bus.unsubscribe = lambda sub_id: True
        recorder = ObserverRecorder(config=config, event_bus=bus)
        await recorder.start()
        await recorder.stop()
        assert recorder._sub_id is None

    @pytest.mark.asyncio
    async def test_disabled_does_not_subscribe(self) -> None:
        config = ObserverConfig(enabled=False)
        bus = AsyncMock()
        recorder = ObserverRecorder(config=config, event_bus=bus)
        await recorder.start()
        assert recorder._sub_id is None


class TestGetRecentEvents:
    """Verify in-memory ring buffer retrieval."""

    def _make_recorder(self) -> ObserverRecorder:
        config = ObserverConfig()
        bus = AsyncMock()
        bus.subscribe = lambda callback, event_filter=None: "sub-id"
        return ObserverRecorder(config=config, event_bus=bus)

    def test_returns_events_for_job(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        event: ObserverEvent = {
            "job_id": "job-1", "sheet_num": 0,
            "event": "observer.process_spawned",
            "data": {"pid": 1234, "name": "pytest"},
            "timestamp": time.time(),
        }
        recorder._handle_event(event)
        recent = recorder.get_recent_events("job-1", limit=10)
        assert len(recent) == 1
        assert recent[0]["event"] == "observer.process_spawned"

    def test_respects_limit(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        recorder.register_job("job-1", tmp_path)
        for i in range(20):
            event: ObserverEvent = {
                "job_id": "job-1", "sheet_num": 0,
                "event": "observer.file_created",
                "data": {"path": f"file-{i}.md"},
                "timestamp": time.time(),
            }
            recorder._handle_event(event)
        recent = recorder.get_recent_events("job-1", limit=5)
        assert len(recent) == 5

    def test_none_job_id_returns_all(self, tmp_path: Path) -> None:
        recorder = self._make_recorder()
        ws1, ws2 = tmp_path / "ws1", tmp_path / "ws2"
        ws1.mkdir()
        ws2.mkdir()
        recorder.register_job("job-1", ws1)
        recorder.register_job("job-2", ws2)
        for jid in ("job-1", "job-2"):
            event: ObserverEvent = {
                "job_id": jid, "sheet_num": 0,
                "event": "observer.process_spawned",
                "data": {"pid": 1234, "name": "test"},
                "timestamp": time.time(),
            }
            recorder._handle_event(event)
        recent = recorder.get_recent_events(None, limit=50)
        assert len(recent) == 2

    def test_unknown_job_returns_empty(self) -> None:
        recorder = self._make_recorder()
        assert recorder.get_recent_events("nonexistent", limit=10) == []
```

**Step 2: Run tests, verify fail, implement, verify pass, commit**

Run: `pytest tests/test_observer_recorder.py::TestLifecycle tests/test_observer_recorder.py::TestGetRecentEvents -v`

Implement `start()`, `stop()`, `get_recent_events()` in `ObserverRecorder`.

```bash
git commit -m "feat(observer): EventBus lifecycle + get_recent_events (#102)"
```

---

### Task 6: Wire ObserverRecorder into JobManager

**Files:**
- Modify: `src/mozart/daemon/manager.py:130-260` (init + start), `manager.py:929-955` (shutdown), `manager.py:1196-1220` (snapshot capture)
- Test: existing daemon tests should still pass

**Step 1: Add recorder to JobManager.__init__**

After `self._snapshot_manager = SnapshotManager()` (line 163):
```python
# Observer event recorder — persists per-job observer events to JSONL.
# Initialized eagerly, started in start() after event bus.
self._observer_recorder: ObserverRecorder | None = None
```

**Step 2: Start recorder in JobManager.start()**

After semantic analyzer start (after line 253):
```python
# Start observer recorder after event bus.
if self._config.observer.persist_events:
    from mozart.daemon.observer_recorder import ObserverRecorder
    self._observer_recorder = ObserverRecorder(
        config=self._config.observer,
        event_bus=self._event_bus,
    )
    await self._observer_recorder.start()
```

**Step 3: Register job when observer starts**

In `_start_observer()` (around line 1042), after `await observer.start()`:
```python
if self._observer_recorder is not None:
    self._observer_recorder.register_job(job_id, meta.workspace)
```

**Step 4: Flush recorder before snapshot, unregister after**

In `_run_managed_task()`, before `self._snapshot_manager.capture()` (line 1216):
```python
# Flush observer recorder to ensure JSONL is complete before snapshot
if self._observer_recorder is not None:
    try:
        self._observer_recorder.flush(job_id)
    except Exception:
        _logger.warning("observer_recorder.flush_failed", job_id=job_id, exc_info=True)
```

In the `finally` block (line 1284), after `await self._stop_observer(job_id)`:
```python
if self._observer_recorder is not None:
    self._observer_recorder.unregister_job(job_id)
```

**Step 5: Stop recorder in shutdown()**

After stopping observers (line 932), before stopping semantic analyzer:
```python
if self._observer_recorder is not None:
    try:
        await self._observer_recorder.stop()
    except (OSError, RuntimeError):
        _logger.warning("manager.observer_recorder_stop_failed", exc_info=True)
```

**Step 6: Expose recorder for IPC**

Add property to JobManager:
```python
@property
def observer_recorder(self) -> ObserverRecorder | None:
    return self._observer_recorder
```

**Step 7: Run full test suite**

Run: `pytest tests/ -x --timeout=60`
Expected: All existing tests pass. No regressions.

**Step 8: Commit**

```bash
git add src/mozart/daemon/manager.py
git commit -m "feat(observer): wire ObserverRecorder into JobManager lifecycle (#102)"
```

---

### Task 7: Snapshot Enrichment — JSONL Capture + Git Context

**Files:**
- Modify: `src/mozart/daemon/snapshot.py`
- Test: `tests/test_daemon_snapshot.py`

**Step 1: Write the failing tests**

Add to `tests/test_daemon_snapshot.py`:

```python
class TestObserverJSONLCapture:
    """Verify observer JSONL is included in snapshots."""

    def test_captures_observer_jsonl(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        # Create a mock observer JSONL
        jsonl = workspace / ".mozart-observer.jsonl"
        jsonl.write_text('{"event":"observer.file_created"}\n')
        # Also create the required state file
        (workspace / "test-job.json").write_text("{}")

        manager = SnapshotManager(base_dir=tmp_path / "snapshots")
        snapshot_path = manager.capture("test-job", workspace)
        assert snapshot_path is not None
        captured = Path(snapshot_path) / ".mozart-observer.jsonl"
        assert captured.exists()


class TestGitContextCapture:
    """Verify git context capture in snapshots."""

    def test_captures_git_context_in_repo(self, tmp_path: Path) -> None:
        import subprocess
        workspace = tmp_path / "repo"
        workspace.mkdir()
        # Initialize a real git repo
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=workspace, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, capture_output=True)
        (workspace / "file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=workspace, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=workspace, capture_output=True)

        (workspace / "test-job.json").write_text("{}")

        manager = SnapshotManager(base_dir=tmp_path / "snapshots")
        snapshot_path = manager.capture("test-job", workspace)
        assert snapshot_path is not None
        git_ctx = Path(snapshot_path) / "git-context.json"
        assert git_ctx.exists()
        data = json.loads(git_ctx.read_text())
        assert "head_sha" in data
        assert "branch" in data

    def test_no_git_context_outside_repo(self, tmp_path: Path) -> None:
        workspace = tmp_path / "not-a-repo"
        workspace.mkdir()
        (workspace / "test-job.json").write_text("{}")

        manager = SnapshotManager(base_dir=tmp_path / "snapshots")
        snapshot_path = manager.capture("test-job", workspace)
        assert snapshot_path is not None
        git_ctx = Path(snapshot_path) / "git-context.json"
        assert not git_ctx.exists()  # Gracefully skipped
```

**Step 2: Run tests, verify fail**

**Step 3: Implement**

In `snapshot.py`:
- Add `.mozart-observer.jsonl` to `_CAPTURE_PATTERNS`
- Add `_capture_git_context(workspace, snapshot_dir)` method
- Call it from `capture()` after copying files

**Step 4: Run tests, verify pass, commit**

```bash
git commit -m "feat(observer): snapshot enrichment with JSONL + git context (#102)"
```

---

### Task 8: IPC Method — daemon.observer_events

**Files:**
- Modify: `src/mozart/daemon/process.py:507-528` (add handler)
- Test: integration test with mock

**Step 1: Add IPC handler**

In `_register_methods()` in `process.py`, after the `handler.register("daemon.events", handle_events)` block (line 528):

```python
# Observer event recorder IPC — per-job behavioral events
async def handle_observer_events(
    params: dict[str, Any], _w: Any,
) -> dict[str, Any]:
    if manager.observer_recorder is None:
        return {"events": []}
    job_id = params.get("job_id")
    limit = params.get("limit", 50)
    events = manager.observer_recorder.get_recent_events(
        job_id, limit=limit,
    )
    return {"events": events}

handler.register("daemon.observer_events", handle_observer_events)
```

**Step 2: Test + commit**

```bash
git commit -m "feat(observer): add daemon.observer_events IPC handler (#102)"
```

---

### Task 9: TUI Integration — Timeline + Detail Panel

**Files:**
- Modify: `src/mozart/tui/reader.py` (add `get_observer_events`)
- Modify: `src/mozart/tui/panels/timeline.py` (add PROC entry type)
- Modify: `src/mozart/tui/panels/detail.py` (add file activity section)
- Modify: `src/mozart/tui/app.py` (poll observer events)

**Step 1: Add get_observer_events to MonitorReader**

In `reader.py`, add method:
```python
async def get_observer_events(
    self, job_id: str | None = None, limit: int = 50,
) -> list[dict[str, Any]]:
    """Get recent observer events via IPC."""
    await self._ensure_source()
    if self._source == "ipc" and self._ipc_client is not None:
        try:
            result = await self._ipc_client.call(
                "daemon.observer_events",
                {"job_id": job_id, "limit": limit},
            )
            if result and isinstance(result, dict):
                return result.get("events", [])
        except Exception:
            _logger.debug("reader.observer_events_failed", exc_info=True)
    return []
```

**Step 2: Add PROC entries to TimelinePanel**

In `timeline.py`, add color mapping:
```python
_EVENT_COLORS: dict[str, str] = {
    # ... existing entries ...
    "observer_process": "magenta",
}
```

In `update_data()`, add `observer_events` parameter. For each event where `event` starts with `observer.process_`, render a PROC entry in the timeline.

**Step 3: Add file activity to DetailPanel**

In `detail.py`, add a section that shows recent file events when a job is selected. Render as a list with timestamps, event type, and path.

**Step 4: Poll observer events in MonitorApp refresh**

In `app.py`, add `reader.get_observer_events()` call in the refresh cycle. Route process events to TimelinePanel, file events to DetailPanel.

**Step 5: Test + commit**

```bash
git commit -m "feat(observer): wire observer events into mozart top TUI (#102)"
```

---

### Task 10: Full Integration Test + Cleanup

**Step 1: Run the full test suite**

```bash
pytest tests/ -x --timeout=60
```

**Step 2: Run mypy**

```bash
mypy src/mozart/daemon/observer_recorder.py src/mozart/daemon/snapshot.py
```

**Step 3: Run ruff**

```bash
ruff check src/mozart/daemon/observer_recorder.py src/mozart/daemon/snapshot.py src/mozart/daemon/manager.py src/mozart/daemon/process.py
```

**Step 4: Final commit if any fixes needed**

```bash
git commit -m "fix(observer): address lint/type issues from integration (#102)"
```

**Step 5: Update issue #102 with implementation summary**

---

## Dependency Order

```
Task 1 (config) → Task 2 (core structure) → Task 3 (JSONL) → Task 4 (coalesce + cap)
    → Task 5 (lifecycle + get_recent) → Task 6 (manager wiring) → Task 7 (snapshot)
    → Task 8 (IPC) → Task 9 (TUI) → Task 10 (integration test)
```

All tasks are sequential — each builds on the previous.
