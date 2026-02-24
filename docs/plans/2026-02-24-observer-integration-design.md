# Observer Integration Design

**Date:** 2026-02-24
**Issue:** [#102](https://github.com/Mzzkc/mozart-ai-compose/issues/102)
**Status:** Approved

---

## Problem

The JobObserver produces per-job behavioral events (filesystem changes, child process spawns/exits) and publishes them to the EventBus. Three gaps exist:

1. **Events are ephemeral** — no subscriber persists them to disk
2. **`mozart top` doesn't show them** — the TUI timeline only shows profiler events
3. **Completion snapshots don't include them** — post-mortem analysis loses behavioral data

## Approach: New ObserverRecorder Subscriber

A new `ObserverRecorder` class subscribes to `observer.*` events on the EventBus and bridges them to per-job JSONL persistence and TUI access. This follows the existing architecture pattern where each concern is a separate EventBus subscriber (like `SemanticAnalyzer` and `ProfilerCollector`).

**Why not extend ProfilerCollector?** The profiler answers "how much resources are being used?" (system-wide). The observer answers "what behavioral actions did the agent take?" (per-job). Different questions, different components. The profiler is already 756 lines. Coupling per-job behavioral data into a system-wide collector crosses an intentional boundary.

**Why not make JobObserver write directly?** The observer is a pure event producer by design. Adding I/O responsibility couples the sensor to a storage strategy. The EventBus exists precisely to decouple producers from consumers.

---

## Layer 1: ObserverRecorder (Persistence)

**New file:** `src/mozart/daemon/observer_recorder.py`

### Class Design

```python
class ObserverRecorder:
    """EventBus subscriber that persists observer events to per-job JSONL files."""

    def __init__(self, config: ObserverConfig, event_bus: EventBus) -> None: ...
    async def start(self) -> None: ...       # Subscribe to observer.* events
    async def stop(self) -> None: ...        # Unsubscribe, flush all, close files
    def register_job(self, job_id: str, workspace: Path) -> None: ...  # Open JSONL
    def unregister_job(self, job_id: str) -> None: ...                 # Flush + close
    def flush(self, job_id: str) -> None: ...                          # Flush coalesce buffer + fsync
    def get_recent_events(self, job_id: str | None, limit: int) -> list[ObserverEvent]: ...
```

### Per-Job State

Each registered job gets:
- Open file handle to `{workspace}/.mozart-observer.jsonl`
- Coalesce buffer: `dict[str, (timestamp, event)]` keyed by path, flushed when window expires
- Ring buffer: `deque(maxlen=200)` of recent events for TUI consumption

### Filtering

**Path exclusion** (applied before write):
- Default patterns: `.git/`, `__pycache__/`, `node_modules/`, `.venv/`, `*.pyc`
- Uses `fnmatch` against the relative path from the observer event

**Coalescing** (same-file modifications):
- When `observer.file_modified` arrives for a path already in the coalesce buffer within `coalesce_window_seconds` (default: 2.0s), replace the buffered event instead of writing both
- Flush on: window expiry, job unregister, explicit flush call

**Size cap:**
- Per-job JSONL capped at `max_timeline_bytes` (default: 10MB)
- On overflow: truncate oldest half of the file (seek, read tail, rewrite)

### Config Additions

```python
class ObserverConfig(BaseModel):
    # ... existing fields ...
    persist_events: bool = True
    exclude_patterns: list[str] = [".git/", "__pycache__/", "node_modules/", ".venv/", "*.pyc"]
    coalesce_window_seconds: float = 2.0
    max_timeline_bytes: int = 10_485_760  # 10MB
```

---

## Layer 2: `mozart top` Integration

### Data Flow

```
ObserverRecorder.get_recent_events(job_id, limit)
    ↓ (via IPC)
daemon.observer_events → {events: [...]}
    ↓
MonitorReader.get_observer_events() → list[ObserverEvent]
    ↓
TimelinePanel: process events (PROC)
DetailPanel: file events (FILE) on job drill-down
```

### TUI Display Split (TDF-informed)

The timeline panel is for "at a glance" monitoring. File events are high-frequency noise at that level. Process events are high-signal, low-frequency.

| Event Type | Display Location | Icon/Color |
|---|---|---|
| `observer.process_spawned` | Timeline panel | `⚙ PROC` purple |
| `observer.process_exited` | Timeline panel | `⚙ PROC` purple |
| `observer.file_created` | Detail panel (job drill-down) | `📄 FILE` cyan |
| `observer.file_modified` | Detail panel (job drill-down) | `📄 FILE` cyan |
| `observer.file_deleted` | Detail panel (job drill-down) | `📄 FILE` dim cyan |

### IPC Method

**`daemon.observer_events`** — new handler in `process.py`:
- Params: `{"job_id": str | null, "limit": int}` (null = all jobs)
- Returns: `{"events": [ObserverEvent, ...]}`
- Source: `ObserverRecorder.get_recent_events()`

### JSON Mode

`mozart top --json` includes all observer events in the NDJSON stream (no display split — machine consumers get everything).

---

## Layer 3: Snapshot Enrichment

### Observer JSONL Capture

Add `.mozart-observer.jsonl` to `SnapshotManager._CAPTURE_PATTERNS`:

```python
_CAPTURE_PATTERNS = [
    "*.json",
    "mozart.log",
    "*.log",
    ".mozart-observer.jsonl",  # Observer timeline
]
```

The recorder writes the JSONL inside the workspace, so the existing glob-based capture picks it up automatically.

### Git Context Capture

New method `SnapshotManager._capture_git_context(workspace, snapshot_dir)`:
- Runs: `git rev-parse --abbrev-ref HEAD`, `git rev-parse HEAD`, `git status --porcelain`
- Writes `git-context.json` to snapshot directory
- Fails silently if workspace is not a git repo

Contents:
```json
{
  "branch": "main",
  "head_sha": "abc123...",
  "uncommitted_files": ["src/foo.py", "tests/test_foo.py"],
  "captured_at": 1740000000.0
}
```

### Process Tree Snapshot

**Not included.** The profiler already captures system-wide process metrics every 5 seconds (including at job completion). Duplicating per-job is redundant. Use `mozart top --history` to query profiler data for the job's time window.

### Completion Barrier (Fail-Open)

In `manager.py:_run_managed_task()`, after job finishes:

```python
# 1. Flush observer recorder (ensures JSONL is complete)
self._observer_recorder.flush(job_id)

# 2. Capture snapshot (includes flushed JSONL + git context)
snapshot_path = self._snapshot_manager.capture(job_id, workspace)

# 3. Proceed with on_success hooks regardless of snapshot success
await self._run_on_success_hooks(...)
```

If flush or capture fails, log warning and continue. Observability never blocks execution.

---

## Error Handling

All fail-open:
- JSONL write failure → log warning, skip event
- Coalesce buffer flush failure → log warning, clear buffer
- Snapshot capture failure → log warning, proceed with hooks
- IPC method failure → TUI shows stale/empty data
- Missing watchfiles/psutil → observer produces no events, recorder has nothing to record

---

## Files

### Create
| File | Purpose |
|---|---|
| `src/mozart/daemon/observer_recorder.py` | ObserverRecorder class |
| `tests/test_observer_recorder.py` | Unit + integration tests |

### Modify
| File | Change |
|---|---|
| `src/mozart/daemon/config.py` | Add persistence fields to ObserverConfig |
| `src/mozart/daemon/manager.py` | Start/stop recorder, register/unregister per job, flush before snapshot |
| `src/mozart/daemon/snapshot.py` | Add JSONL to capture patterns, add git context capture |
| `src/mozart/daemon/process.py` | Add `daemon.observer_events` IPC handler |
| `src/mozart/daemon/ipc/handler.py` | Route new IPC method |
| `src/mozart/tui/reader.py` | Add `get_observer_events()` |
| `src/mozart/tui/panels/timeline.py` | Add PROC entry type |
| `src/mozart/tui/panels/detail.py` | Add file activity section |
| `src/mozart/tui/app.py` | Poll observer events in refresh cycle |
| `tests/test_daemon_snapshot.py` | Git context + JSONL capture tests |

---

## Testing

| Area | Tests |
|---|---|
| Path exclusion | Verify `.git/`, `__pycache__/` events filtered, normal paths pass |
| Coalesce window | Rapid same-file mods produce one event, different files produce separate events |
| JSONL write + cap | Verify write, verify truncation at 10MB, verify rotation preserves recent events |
| Register/unregister | File opened on register, flushed and closed on unregister |
| Ring buffer | get_recent_events returns correct subset, respects limit |
| Snapshot capture | JSONL included in snapshot, git-context.json generated |
| Fail-open barrier | Flush failure doesn't prevent snapshot, snapshot failure doesn't prevent hooks |
| IPC round-trip | daemon.observer_events returns events from recorder |
| TUI rendering | TimelinePanel renders PROC entries, DetailPanel renders FILE entries |
