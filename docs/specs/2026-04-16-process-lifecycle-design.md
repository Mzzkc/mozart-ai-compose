# Process Lifecycle Architecture — Design Spec

**Status:** DRAFT
**Date:** 2026-04-16
**Author:** Legion
**Triggered by:** GH TBD — process leakage during score2-completion concert

## Problem Statement

The baton treats OS process lifecycle as invisible infrastructure. No layer of the
system owns "kill the process when the sheet is done." This produces exponential
process accumulation under pause/resume/fallback cycles, eventually triggering
resource monitor pauses that themselves never auto-resume.

### Observed Failure Mode

1. Sheets dispatched — musician tasks spawn CLI backend processes (claude-code, gemini-cli, opencode)
2. Validation commands (`command_succeeds`) spawn bash — pytest — N xdist workers
3. Resource monitor detects process count > 95% of limit — calls `manager.pause_job()`
4. `pause_job()` — baton `PauseJob` event — sets `job.paused = True` AND `job.user_paused = True`
5. Running processes **continue** — only new dispatch is suppressed
6. `resume_job()` — cancels old asyncio task — `CancelledError` propagates
7. Old subprocess **not killed** — `CancelledError` skips the happy-path cleanup
8. `recover_job()` resets DISPATCHED sheets to PENDING — dispatches NEW processes
9. Old + new processes coexist — process count higher — monitor pauses again — cycle repeats

### Root Causes

**RC-1: Process identity not tracked in sheet state.**
`SheetState` has no `pid`, `pgid`, or process reference. The only link between a
sheet and its process is an asyncio task in `_active_tasks[(job_id, sheet_num)]`.
This link breaks on task cancellation, conductor restart, and stale recovery.

**RC-2: No subprocess kill on task cancellation.**
`cli_backend.py:709-725` and `engine.py:544-566` both use `try/except TimeoutError`
that properly kills the process on timeout, but have no `CancelledError` handler
and no `finally` block. When `task.cancel()` fires, the subprocess survives.

**RC-3: Validation commands have no process group.**
`engine.py:536` spawns `bash -c <command>` without `start_new_session=True`.
Even killing bash does not kill its children (pytest xdist workers). No tree-kill
mechanism exists.

**RC-4: Monitor pauses are indistinguishable from user pauses.**
`monitor.py:376` calls `manager.pause_job()` which sets `user_paused = True`.
The baton auto-resume paths (rate limit recovery, cost recovery) check
`if not job.user_paused` — so monitor-paused jobs never auto-resume.

**RC-5: Orphan reaper disabled.**
`pgroup.py:243-281` — `reap_orphaned_backends()` is a no-op. The F-481 rewrite
disabled it because ancestry-based detection killed legitimate processes on WSL2.
The planned replacement (per-job PID tracking) was never implemented.

## Design

### Principle: Process Identity is Part of Sheet Lifecycle

A sheet's process (PID, process group) is as fundamental as its status or attempt
count. When the baton transitions a sheet's status, it must also transition the
sheet's process. "Sheet completed" means "process exited cleanly." "Sheet cancelled"
means "process killed." No status transition leaves a process in an unknown state.

### Change 1: PID Tracking in SheetState

Add process identity fields to `SheetState` (Pydantic model in `core/checkpoint.py`):

```python
# Process lifecycle tracking
process_pid: int | None = Field(
    default=None,
    description="PID of the running backend process. Set on dispatch, "
    "cleared on completion/kill. Persisted for conductor restart recovery.",
)
process_pgid: int | None = Field(
    default=None,
    description="Process group ID for tree-kill. When start_new_session=True, "
    "this equals process_pid. Used by killpg() to terminate the entire "
    "process tree including validation command children.",
)
```

**Lifecycle:**
- Set in `_dispatch_callback()` via a callback from `backend.execute()` — `on_process_spawned(pid)`
- Cleared in `_on_musician_done()` or on explicit kill
- Persisted to checkpoint (survives conductor restart)

**Checkpoint schema impact:** Additive — new optional fields with defaults. Existing
checkpoints load fine (M-005 backward compatibility). This is NOT E-007 (non-additive
checkpoint change).

### Change 2: Process Group per Subprocess

All subprocesses spawned by the baton path use `start_new_session=True`:

**cli_backend.py:** Force `start_new_session=True` for baton-path execution.
The existing per-instrument `start_new_session` config remains for non-baton
usage, but the baton adapter overrides it. This ensures every musician process
is a session leader with a killable process group.

**engine.py:** Add `start_new_session=True` to validation `command_succeeds`
spawns. This makes bash — pytest — xdist workers a single killable group.

**Rationale:** `proc.kill()` only kills the direct child. `os.killpg(pgid, SIGTERM)`
kills the entire tree. Without process groups, killing bash leaves 24 xdist workers
orphaned. WSL2 concerns from pgroup.py (line 252) were about killing the daemon's
own group — killing a child's group is safe.

### Change 3: Kill-on-Exit in Backend and Validation

Add `finally` blocks that kill the process group on any exit:

```python
# cli_backend.py execute()
proc = await asyncio.create_subprocess_exec(*cmd, start_new_session=True, ...)
try:
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=...)
except TimeoutError:
    ...  # existing handler
finally:
    if proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
```

Same pattern in `engine.py _check_command_succeeds()`.

**Why SIGTERM before SIGKILL:** SIGTERM gives the process group a chance to clean
up (pytest workers, MCP servers). `proc.kill()` (SIGKILL) is the fallback for the
direct child if SIGTERM did not reach it.

### Change 4: Kill-on-Pause and Kill-on-Deregister

**Pause flow** (`_handle_pause_job` in `core.py`):
After setting `job.paused = True`, emit a new event `KillJobProcesses(job_id)`.
The adapter handles this by iterating all DISPATCHED sheets for the job, killing
their process groups via `os.killpg(sheet.process_pgid, SIGTERM)`, and cancelling
their asyncio tasks.

**Deregister flow** (`deregister_job` in `adapter.py`):
Before cancelling asyncio tasks (line 567), kill process groups for all active sheets.
This ensures the subprocess is dead before the task's `CancelledError` propagates.

**Cancel flow** (`_handle_cancel_job` in `core.py`):
Same as deregister — kill process groups before status transitions.

### Change 5: Monitor Pause Distinction

Add a `monitor_paused` flag to `BatonJobState`:

```python
@dataclass
class BatonJobState:
    paused: bool = False
    user_paused: bool = False
    monitor_paused: bool = False  # NEW: set by resource monitor
```

**New event:** `MonitorPauseJob(job_id, reason)` — distinct from `PauseJob`.
Handler sets `monitor_paused = True` and `paused = True` but NOT `user_paused`.

**Monitor resume:** In the resource monitor's `_check_limits()` loop, after
detecting pressure has dropped below the warn threshold, resume any jobs that
have `monitor_paused = True`:

```python
# In monitor._check_limits():
if current_processes < warn_threshold:
    for job_id in self._monitor_paused_jobs:
        await self._manager.resume_job(job_id)
    self._monitor_paused_jobs.clear()
```

**Interaction with user pause:** If a user pauses a monitor-paused job, both
`user_paused` and `monitor_paused` are True. Auto-resume checks `user_paused`
first — if True, does not auto-resume. User must explicitly resume.

### Change 6: Recovery on Conductor Restart

In `recover_job()` (adapter.py:601), after loading checkpoint state:

1. For each sheet with a non-None `process_pid`:
   a. Check if the PID is still alive: `os.kill(pid, 0)`
   b. If alive: kill the process group `os.killpg(pgid, SIGTERM)`, wait briefly
   c. Clear `process_pid` and `process_pgid` from the sheet state
2. Then proceed with normal recovery (reset DISPATCHED to PENDING)

This prevents the "old process still running when new one is dispatched" scenario
after conductor restart.

**PID reuse concern:** PIDs can be reused by the OS. Between conductor restart and
recovery, the old PID might now be a different process. Mitigation: check the process
command line (`/proc/{pid}/cmdline`) to verify it is a Marianne-spawned process before
killing. If uncertain, log a warning and clear the PID without killing.

### Change 7: Stale Detection Enhancement

In `_handle_stale_check()` (adapter.py:1684), in addition to checking if the asyncio
task is alive, also check if the OS process is alive:

```python
# If task is dead but PID is alive: zombie process
if (task is None or task.done()) and state.process_pid is not None:
    try:
        os.kill(state.process_pid, 0)  # Alive check
        os.killpg(state.process_pgid, SIGTERM)  # Kill zombie
    except (ProcessLookupError, PermissionError):
        pass
    state.process_pid = None
    state.process_pgid = None
```

## File Change Map

| File | Change | Scope |
|------|--------|-------|
| `core/checkpoint.py` | Add `process_pid`, `process_pgid` to SheetState | Small |
| `daemon/baton/state.py` | Add `monitor_paused` to BatonJobState | Small |
| `daemon/baton/events.py` | Add `MonitorPauseJob`, `KillJobProcesses` events | Small |
| `daemon/baton/core.py` | Handle new events, kill processes on pause/cancel | Medium |
| `daemon/baton/adapter.py` | Kill processes in deregister, recovery PID check, stale enhancement | Medium |
| `daemon/baton/dispatch.py` | No change (dispatch is correct — it only prevents new dispatch) | None |
| `daemon/baton/musician.py` | Pass PID callback, store PID in sheet state after spawn | Small |
| `daemon/monitor.py` | Use `MonitorPauseJob`, track monitor-paused jobs, auto-resume | Medium |
| `daemon/manager.py` | Route monitor pause through new event type | Small |
| `execution/instruments/cli_backend.py` | Force `start_new_session`, `finally` with `killpg` | Medium |
| `execution/validation/engine.py` | `start_new_session=True`, `finally` with `killpg` | Small |
| `daemon/pgroup.py` | Can eventually remove disabled reaper (replaced by this system) | Cleanup |

## Phasing

**Phase 1 (defense-in-depth, no schema change):**
1. `start_new_session=True` + `finally`/`killpg` in cli_backend.py and engine.py
2. Kill processes in `deregister_job()` before cancelling asyncio tasks

**Phase 2 (PID in sheet state):**
3. Add `process_pid`/`process_pgid` to SheetState
4. Wire PID storage from `on_process_spawned` through adapter to sheet state
5. Kill-on-pause via `KillJobProcesses` event

**Phase 3 (monitor autonomy):**
6. `MonitorPauseJob` event + `monitor_paused` flag
7. Auto-resume in resource monitor when pressure clears

**Phase 4 (restart recovery):**
8. PID-based cleanup in `recover_job()`
9. Stale detection enhancement

## Escalation Notes

- **E-007 concern:** `process_pid`/`process_pgid` are additive optional fields
  with None defaults. Existing checkpoints load without modification. This is
  NOT a non-additive checkpoint change.
- **E-002 concern:** No CLI command changes.
- **E-001 concern:** No IPC protocol changes.
- **E-003 concern:** No daemon lifecycle changes (the conductor starts/stops
  the same way; process cleanup is internal).

## Risks

| Risk | Mitigation |
|------|-----------|
| WSL2 process group sensitivity | Only killpg on child session groups, never the daemon's own group |
| PID reuse after conductor restart | Verify cmdline before killing; log warning if uncertain |
| Race between kill and natural exit | `ProcessLookupError` caught everywhere; idempotent |
| `start_new_session` breaking instruments | Instrument profiles can opt out via existing config flag |
| Validation command children surviving SIGTERM | Follow up with SIGKILL after brief delay if needed |

## Test Strategy

- Unit: SheetState with PID fields serializes/deserializes correctly
- Unit: killpg mock in cli_backend.py finally block
- Unit: MonitorPauseJob vs PauseJob event handling
- Integration: spawn subprocess with start_new_session, cancel task, verify no orphans
- Integration: monitor pauses job, pressure clears, job auto-resumes
- Adversarial: rapid pause/resume cycles do not leak processes
- Adversarial: conductor restart with live processes, all cleaned up
