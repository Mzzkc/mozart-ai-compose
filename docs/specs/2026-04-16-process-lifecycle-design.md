# Process Lifecycle Architecture — Design Spec

**Status:** DRAFT v3
**Date:** 2026-04-16
**Author:** Legion
**Reviewer:** Claude Opus 4.6 (1M context) — v2 review applied in v3
**Triggered by:** GH TBD — process leakage during runner-removal concert
**Related:** [Rate Limit Primary](2026-04-16-rate-limit-primary-design.md) — shares edit territory in `daemon/baton/` (see Coordination section)

## Problem Statement

The baton treats OS process lifecycle as invisible infrastructure. No layer of the
system owns "kill the process when the sheet is done." This produces process
accumulation under pause/resume/fallback cycles, eventually triggering resource
monitor pauses that themselves never auto-resume.

**Unblock target:** runner-removal is PAUSED at 73% because of this failure mode.
Phase 1 alone is expected to unblock it. Phases 2-4 harden the system further.

**Scope:** This spec governs Marianne-owned process lifecycle — subprocesses the
daemon spawns directly (CLI musicians, `command_succeeds` validation). It does
NOT govern process behavior *inside* a musician (e.g., a musician's own `subprocess`
calls). Kill-on-pause is deliberately disabled to protect in-flight musician work;
if a musician itself is the bloat source, monitor pause is a no-op by design and
the pressure persists. That is a separate problem (see Future Work).

### Observed Failure Mode

1. Sheets dispatched — musician tasks spawn CLI backend processes (claude-code, gemini-cli, opencode)
2. Validation commands (`command_succeeds`) spawn bash to pytest to N xdist workers
3. Resource monitor detects process count > 95% of limit — calls `manager.pause_job()`
4. `pause_job()` then baton `PauseJob` event — sets `job.paused = True` AND `job.user_paused = True`
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

### Guiding Principles

**1. Process cleanup is the subprocess owner's responsibility.**
The code that spawns a process must ensure it dies on every exit path — clean
completion, timeout, cancellation, error. This is achieved via `finally` blocks,
not external reapers or monitors.

**2. Pause means "stop dispatching," not "kill running work."**
A musician that is 45 minutes into a type bridge migration should not be killed
because the process monitor spiked during pytest. Pause suppresses new dispatch.
Running sheets complete and clean up after themselves. Only explicit cancel
(`mzt cancel`) or job deregistration kills in-flight work.

**3. Monitor pauses are automatic and reversible.**
When the resource monitor pauses a job due to pressure, it must auto-resume when
pressure clears. This happens without tearing down baton state — the manager task
stays alive, the baton event loop keeps running, and sheet state is preserved.

**4. Capture, never compute, process identifiers.**
Process group IDs are captured at spawn time (`os.getpgid(proc.pid)` immediately
after `create_subprocess_exec`). They are stored in memory for the process's
lifetime and never re-derived. Passing `proc.pid` to `killpg` is banned — that
assumes `pgid == pid`, which is only true by convention and is unsafe if any
layer between us and the kernel changes it.

### The Happy Path (for reader orientation)

Before the failure-mode details, here is a clean-run sheet lifecycle:

1. Baton dispatch fires, which calls `musician.run_sheet()`, which calls `backend.execute(...)`
2. `backend.execute` spawns subprocess with `start_new_session=True`, captures
   `pgid = os.getpgid(proc.pid)` (Change 1), registers `(job_id, sheet_num)` mapped to `(pid, pgid)`
   in `_active_pids` (Change 3's in-memory table) and in `SheetState.process_pid/pgid` (Phase 3)
3. `await proc.communicate()` returns normally with stdout/stderr/exit_code
4. `finally` block (Change 2) runs. `proc.returncode is not None`, so the kill path is skipped
5. `finally` clears the `_active_pids` entry and `SheetState.process_pid/pgid`
6. Musician emits `SheetAttemptResult`, baton runs `_handle_attempt_result`, emits `SheetCompleted`
7. `_on_musician_done` callback performs task cleanup, no process work (already cleaned)

The `finally` block is the one gate that cleanup always flows through. Changes 3-7
exist only for cases where the `finally` block could not run (external kill,
conductor restart, task cancellation that races completion).

### Change 1: Process Group per Subprocess

All subprocesses spawned by the baton path use `start_new_session=True`:

**cli_backend.py:** Force `start_new_session=True` for baton-path execution.
The existing per-instrument `start_new_session` config remains for non-baton
usage, but the baton adapter overrides it. This ensures every musician process
is a session leader with a killable process group.

**engine.py:** Add `start_new_session=True` to validation `command_succeeds`
spawns. This makes bash then pytest then xdist workers a single killable group.

Validation commands are the primary source of process bloat: each `pytest -n auto`
spawns 15-24 workers. Without process groups, killing bash on timeout leaves all
workers orphaned. With process groups, `os.killpg(pgid, SIGTERM)` kills the
entire tree in one call.

**Rationale:** `proc.kill()` only kills the direct child. `os.killpg(pgid, SIGTERM)`
kills the entire tree. Without process groups, killing bash leaves 24 xdist workers
orphaned. WSL2 concerns from pgroup.py (line 252) were about killing the daemon's
own group — killing a child's group is safe (see safety check below).

### Change 2: Kill-on-Exit in Backend and Validation

Add `finally` blocks that kill the process group on any exit path. The sequence
is **SIGTERM then grace then SIGKILL**, not SIGTERM immediately chased by SIGKILL.
The v2 spec had a bug here: sending SIGKILL on the next line defeated the purpose
of SIGTERM.

```python
# cli_backend.py execute()
proc = await asyncio.create_subprocess_exec(
    *cmd, start_new_session=True, ...
)

# Capture pgid at spawn — never derive from proc.pid later.
try:
    pgid = os.getpgid(proc.pid)
except ProcessLookupError:
    pgid = None  # Process already died; nothing to kill later.

# Safety: never kill the daemon's own group.
daemon_pgid = os.getpgid(0)
if pgid == daemon_pgid:
    # Something is wrong — start_new_session failed or the kernel ignored it.
    # Abort the spawn rather than risk killing ourselves later.
    proc.kill()
    await proc.wait()
    raise RuntimeError(
        f"Spawned process shares daemon pgid ({pgid}); refusing to continue"
    )

try:
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(), timeout=...
    )
except TimeoutError:
    ...  # existing handler
finally:
    if proc.returncode is None and pgid is not None:
        # SIGTERM the group, give it a chance to clean up.
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            # Did not exit in time — SIGKILL the group.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                await proc.wait()
            except ProcessLookupError:
                pass
```

Same pattern in `engine.py _check_command_succeeds()`. Validation commands deserve
equal attention — each `pytest -n auto` spawns a full worker pool that must
be killed as a tree.

**Why SIGTERM before SIGKILL:** SIGTERM gives pytest workers and MCP servers a
chance to shut down cleanly (flush logs, release locks, close file descriptors).
SIGKILL is the hammer for processes that ignore SIGTERM.

**Why 2 seconds grace:** Empirically long enough for pytest-xdist workers and
claude-code/opencode to exit cleanly. Short enough not to stall cancel paths.
If longer grace proves necessary, it becomes a tunable.

**Why `finally` solves the core problem:** The `finally` block runs on every exit
path — clean completion, `TimeoutError`, `CancelledError`, arbitrary exceptions.
This single change closes RC-2 (no kill on cancellation) because `CancelledError`
propagates through `await proc.communicate()`, the `finally` fires, and the process
group is killed. No separate `CancelledError` handler needed.

### Change 3: Kill Processes on Deregister and Cancel

**Deregister flow** (`deregister_job` in `adapter.py`):
Before cancelling asyncio tasks, kill process groups for all active sheets of
that job. This ensures the subprocess is dead before the task's `CancelledError`
propagates.

**Cancel flow** (`_handle_cancel_job` in `core.py`):
Same as deregister — kill process groups before status transitions.

**Why preempt when `finally` already cleans up?** `finally` fires only when the
await wakes up to run it. Task cancellation delivers `CancelledError`, but under
some conditions the `await proc.communicate()` may not yield promptly (blocked on
a slow syscall, uninterruptible I/O). The preemptive `killpg` forces the process
to exit, which unblocks `communicate`, which wakes the await, which runs `finally`,
which performs the already-done-is-idempotent cleanup. Belt and suspenders:
preempt for responsiveness, `finally` for correctness.

**NOT on pause.** Pause is a soft signal — "stop dispatching new sheets." Running
musicians continue to completion and clean up via the `finally` block (Change 2)
when they exit normally. Killing on pause would destroy in-progress work that
the user likely wants to preserve:

- A musician 40 minutes into a refactoring stage loses all uncommitted edits
- A musician about to commit after passing tests has its commit killed
- Re-execution on resume repeats all that work from scratch
- Partial edits left in the working tree corrupt the next musician's view

The `finally` block in Change 2 already ensures cleanup happens when the process
exits — whether from natural completion, timeout, or task cancellation. Kill-on-pause
adds a premature kill that fires even when the user just wants to temporarily stop
dispatch.

**In-memory PID table (Phase 1 scope):**
Deregister/cancel need to know *which* processes to kill. Since `SheetState`
does not gain PID fields until Phase 3, Phase 1 introduces a minimal in-memory
structure owned by the adapter:

```python
# adapter.py — Phase 1, in-memory only, not persisted
_active_pids: dict[tuple[str, int], tuple[int, int]] = {}
# (job_id, sheet_num) -> (pid, pgid)
```

- Populated by `on_process_spawned(job_id, sheet_num, pid, pgid)` callback
  passed to `backend.execute()`.
- Cleared by the spawn site's `finally` block and by `_on_musician_done`.
- Read by `deregister_job` and `_handle_cancel_job` to drive `killpg`.
- Lost on conductor restart — that is what Phase 4 recovery is for.

Phase 3 replaces this with persisted `SheetState.process_pid/pgid`. The in-memory
table is then removed (or kept as a fast-path cache — decided at Phase 3 implementation).

### Change 4: Monitor Pause Distinction (Lightweight Auto-Resume)

The resource monitor needs its own pause/resume path that does not set `user_paused`
and does not tear down baton state.

**New events:** `MonitorPauseJob(job_id, reason)` and `MonitorResumeJob(job_id)` —
distinct from `PauseJob` / `ResumeJob`.

**New field on `_JobRecord`:**
```python
@dataclass
class _JobRecord:
    paused: bool = False
    user_paused: bool = False
    monitor_paused: bool = False  # NEW: set by resource monitor
```

**Pause path:**
1. Monitor detects process count > 95% of limit
2. Monitor calls `manager.monitor_pause_job(job_id)` (new method)
3. Manager sends `MonitorPauseJob(job_id, reason)` to baton inbox
4. Manager sets status to PAUSED via `_set_job_status`
5. Baton handler sets `monitor_paused = True` and `paused = True` but NOT `user_paused`
6. Running sheets continue to completion — only new dispatch is suppressed
7. The manager task stays alive on `wait_for_completion` — no cancellation

**Resume path:**
1. Monitor's next check: process count below warn threshold (80%)
2. Monitor calls `manager.monitor_resume_job(job_id)` (new method)
3. Manager sets status to RUNNING via `_set_job_status`
4. Manager sends `MonitorResumeJob(job_id)` to baton inbox
5. Baton handler clears `monitor_paused`, clears `paused` (unless also `user_paused`)
6. Injects `DispatchRetry` to wake the dispatch cycle
7. Pending sheets get dispatched on the next cycle
8. No task cancellation. No checkpoint reload. No state rebuild. All in-flight
   sheet results, completion tracking, and musician tasks are preserved

**Why this is better than `manager.resume_job()`:**
`resume_job()` cancels the old manager task, which causes the old task's CancelledError handler
to call `deregister_job()`, which destroys baton state. The new task calls `recover_job()`,
rebuilds from checkpoint, resets DISPATCHED sheets to PENDING, and re-dispatches.
This loses in-flight results and re-executes sheets that may have already finished.

The lightweight path preserves everything. The baton just unpauses and continues
where it left off. The process spike that triggered the pause has already subsided
(the running sheets' processes exited cleanly via Change 2's `finally` blocks).

**Interaction with user pause:** If a user pauses a monitor-paused job, both
`user_paused` and `monitor_paused` become True. Auto-resume checks `user_paused`
first — if True, does not auto-resume. User must explicitly resume.

**Interaction with rate limit and cost auto-resume:** Existing auto-resume paths
(rate-limit timer, cost-recovery timer) check `if not job.user_paused`. They must
also check `if not job.monitor_paused` — otherwise the rate-limit timer could
race the monitor and resume a job the monitor still wants paused. Applies to:
- `daemon/baton/core.py` rate limit recovery handler
- `daemon/baton/core.py` cost recovery handler
- Any other auto-resume path gated on `user_paused`

**Oscillation guard (anti-chatter):** If the monitor pauses the same job more
than N times within M minutes (proposed defaults: N=3, M=5), the monitor switches
to exponential backoff before auto-resuming and emits a diagnostic event. After
a configurable ceiling (proposed: 10 pauses in 10 minutes), the monitor stops
auto-resuming and marks the job as requiring manual composer intervention. This
prevents spike-pause-resume-spike loops from burning CPU indefinitely, and makes
self-driving scores (e.g., evolution) safe under sustained pressure.

**Composer-facing display:**
- `mzt status` and dashboard distinguish `PAUSED (user)`, `PAUSED (monitor)`,
  `PAUSED (user+monitor)`. Exact label text is an implementation choice — the
  requirement is that a composer returning to a paused score can tell whether
  they need to do anything or the system will resume on its own.
- When monitor oscillation backoff kicks in, show `PAUSED (monitor, backoff Ns)`.
- When the manual-intervention ceiling is hit, show `PAUSED (monitor, needs attention)`.

**Monitor tracking:** The monitor maintains a `_monitor_paused_jobs: dict[str, _MonitorPauseState]`
to track which jobs it paused and their oscillation state (pause count, first
pause time, backoff seconds). On resume, it removes the job from this dict. On
conductor restart, the dict is lost (in-memory only) — but orphan recovery handles
paused jobs normally, and the user can `mzt resume` manually.

### Change 5: PID Tracking in SheetState

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
    description="Process group ID for tree-kill. Captured via os.getpgid(pid) "
    "immediately after spawn (never derived from pid later). Used by killpg() "
    "to terminate the entire process tree including validation command children.",
)
process_start_time: float | None = Field(
    default=None,
    description="Unix timestamp when the process was observed alive (derived "
    "from /proc/{pid}/stat field 22 on Linux). Used together with sheet.started_at "
    "to detect PID reuse during restart recovery.",
)
```

**Lifecycle:**
- Set in `_dispatch_callback()` via a callback from `backend.execute()` — `on_process_spawned(pid, pgid, start_time)`
- Cleared in `_on_musician_done()` or on explicit kill
- Persisted to checkpoint (survives conductor restart)

**Checkpoint schema impact:** Additive — new optional fields with defaults. Existing
checkpoints load fine (M-005 backward compatibility). This is NOT E-007 (non-additive
checkpoint change).

### Change 6: Recovery on Conductor Restart

In `recover_job()` (adapter.py:601), after loading checkpoint state:

1. For each sheet with a non-None `process_pid`:
   a. Check if the PID is still alive: `os.kill(pid, 0)`
   b. If alive, verify it is a Marianne-spawned process before killing:
      - Read `/proc/{pid}/stat` field 22 (starttime in clock ticks since boot)
      - Convert to wall-clock time: `current_process_start`
      - Compute reuse decision:
        - If `current_process_start > sheet.started_at + PID_REUSE_TOLERANCE_SECONDS`
          (default: 2.0s, accounts for clock skew and stat-read jitter), PID was
          reused — do not kill
        - If `abs(current_process_start - checkpoint.process_start_time) < 1.0s`
          (process matches the one we recorded at spawn), same process, safe to kill
        - Otherwise, uncertain — log warning, clear PID without killing
      - Read `/proc/{pid}/cmdline` as secondary check — verify it contains a known
        backend name (claude, opencode, gemini, goose, etc.). On mismatch, uncertain.
      - If confident (start-time match AND cmdline plausible): `os.killpg(pgid, SIGTERM)`,
        wait briefly, then SIGKILL if needed (same grace sequence as Change 2)
      - Always verify `pgid != os.getpgid(0)` before `killpg` (daemon-own-group guard)
   c. Clear `process_pid`, `process_pgid`, `process_start_time` from sheet state
2. Then proceed with normal recovery (reset DISPATCHED to PENDING)

This prevents the "old process still running when new one is dispatched" scenario
after conductor restart.

**Why dual check (start time + cmdline):** PID reuse is rare but real. Checking
only cmdline could match a user's unrelated opencode session. Checking only start
time has edge cases around clock skew. The explicit numeric tolerance
(`PID_REUSE_TOLERANCE_SECONDS = 2.0`) makes the decision deterministic and testable.
On WSL2 where `/proc` behavior can be quirky, failing either check safely falls
through to "log and clear without killing."

**Narrow-window acknowledgement:** If the conductor is SIGKILLed between
subprocess spawn and the line that persists `process_pid/pgid/start_time` to
`SheetState`, the orphan exists but has no persistent identifier. Recovery
cannot find it. This window is measured in microseconds and we accept it as a
known limitation. Future work (cgroups / systemd scopes) would close it.

### Change 7: Stale Detection Enhancement

In `_handle_stale_check()` (adapter.py:1684), in addition to checking if the asyncio
task is alive, also check if the OS process is alive:

```python
# If task is dead but PID is alive: zombie process
if (task is None or task.done()) and state.process_pid is not None:
    # Race-safe: verify start_time matches before killing, else PID may be reused.
    try:
        current_start = _read_process_start_time(state.process_pid)
    except ProcessLookupError:
        current_start = None

    if current_start is None:
        pass  # Process gone; just clear state.
    elif (
        state.process_start_time is not None
        and abs(current_start - state.process_start_time) < PID_REUSE_TOLERANCE_SECONDS
        and state.process_pgid != os.getpgid(0)  # never kill daemon's own group
    ):
        # Same process, safe to kill.
        try:
            os.killpg(state.process_pgid, signal.SIGTERM)
            # Follow with SIGKILL after brief grace if needed (same pattern as Change 2).
        except (ProcessLookupError, PermissionError):
            pass

    state.process_pid = None
    state.process_pgid = None
    state.process_start_time = None
```

Liveness-then-kill without the start-time check is racy: PID reuse can occur
between `os.kill(pid, 0)` and the `killpg`. Always validate identity before
killing.

## File Change Map

| File | Change | Scope | Phase |
|------|--------|-------|-------|
| `execution/instruments/cli_backend.py` | Force `start_new_session`, `finally` with SIGTERM grace SIGKILL, capture pgid at spawn, daemon-own-group safety check, `on_process_spawned` callback | Medium | 1, 3 |
| `execution/validation/engine.py` | `start_new_session=True`, `finally` with SIGTERM grace SIGKILL, pgid capture | Medium | 1 |
| `daemon/baton/adapter.py` | `_active_pids` dict (Phase 1), kill in `deregister_job` before task cancel, recovery PID check (Phase 4), stale enhancement with start-time dual check (Phase 3) | Medium | 1, 3, 4 |
| `daemon/monitor.py` | Use `MonitorPauseJob` / `MonitorResumeJob`, track `_monitor_paused_jobs` dict with oscillation state, exponential backoff, manual-intervention ceiling | Medium | 2 |
| `daemon/manager.py` | Add `monitor_pause_job` / `monitor_resume_job`, route through new events | Small | 2 |
| `daemon/baton/events.py` | Add `MonitorPauseJob`, `MonitorResumeJob` events | Small | 2 |
| `daemon/baton/core.py` | Handle new events, kill processes on cancel/deregister (NOT pause). **Rate-limit and cost auto-resume paths must also check `monitor_paused`.** | Medium | 1, 2 |
| `core/checkpoint.py` | Add `process_pid`, `process_pgid`, `process_start_time` to SheetState | Small | 3 |
| `daemon/baton/musician.py` | Pass PID callback, store PID, pgid, and start_time in sheet state after spawn | Small | 1, 3 |
| `cli/commands/status.py` + dashboard | `PAUSED (user)` vs `PAUSED (monitor)` vs `PAUSED (user+monitor)` rendering; backoff/ceiling states | Small | 2 |
| `daemon/baton/dispatch.py` | No change (dispatch is correct — it only prevents new dispatch) | None | — |
| `daemon/pgroup.py` | Can eventually remove disabled reaper (replaced by this system) | Cleanup | 4 |

## Phasing

### Phase 1: Process Cleanup (no schema change, immediate relief, **unblocks runner-removal**)

Fixes RC-2 and RC-3. Prevents process leaks at the source.

1. `start_new_session=True` in cli_backend.py (baton path) and engine.py (validation commands)
2. pgid capture at spawn; daemon-own-group safety check
3. `finally` blocks with SIGTERM then 2s grace then SIGKILL in both files
4. In-memory `_active_pids` dict in adapter; `on_process_spawned` callback
5. Kill process groups in `deregister_job()` before cancelling asyncio tasks

**Expected outcome:** Each sheet's processes are cleaned up promptly on completion,
timeout, or cancellation. Process count stays bounded. The monitor pause trigger
becomes rare because the bloat that caused it no longer accumulates.

**runner-removal linkage:** runner-removal is PAUSED at 73% because of leaked
validation-command processes triggering monitor pauses that never auto-resume.
Phase 1 removes the leak. The auto-resume issue is Phase 2, but Phase 2 is not
required to unblock runner-removal — once the leak is fixed, monitor pauses
become rare enough that manual resume is acceptable. Phase 1 alone is the
unblocker.

**Falsifier / phase-out predicate for Phase 2:**
After Phase 1 ships, run the runner-removal concert plus 2 additional full-fleet
concerts (minimum 6 sheets, fan-out pattern, `pytest -n auto` validation each).
Record: number of monitor pauses, steady-state process count, peak process count.
- If **0 monitor pauses** across all 3 concerts, Phase 2 can be deferred.
- If **1 or more monitor pause** but pressure clears within 30s, Phase 2 is
  desirable but not urgent.
- If **monitor pauses persist longer than 30s** (i.e., system gets stuck),
  Phase 2 is required.

### Phase 2: Monitor Autonomy (no schema change)

Fixes RC-4. Makes monitor pauses self-healing.

6. `MonitorPauseJob` / `MonitorResumeJob` events + `monitor_paused` flag on `_JobRecord`
7. `monitor_pause_job()` / `monitor_resume_job()` methods on manager
8. Monitor tracks `_monitor_paused_jobs` dict, auto-resumes when pressure clears
9. Lightweight resume path: status update + baton event, no task teardown
10. Oscillation guard: exponential backoff after N=3 pauses in M=5 min, manual-intervention ceiling at 10 pauses in 10 min
11. Rate-limit and cost auto-resume paths gated on `not monitor_paused`
12. CLI/dashboard distinguishes pause origins

**Expected outcome:** Even if process spikes still trigger monitor pauses (e.g.,
during brief `pytest -n auto` peaks), the job auto-resumes within one monitor cycle
(default: every 5 seconds) after the spike subsides. No user intervention needed
unless the chatter threshold is crossed.

### Phase 3: PID Tracking (schema change, defense-in-depth)

Fixes RC-1 and RC-5. Adds explicit process identity to sheet lifecycle.

13. Add `process_pid` / `process_pgid` / `process_start_time` to SheetState
14. Wire PID + pgid + start_time storage from `on_process_spawned` through adapter
15. Migrate `_active_pids` callers to prefer `SheetState.process_pid/pgid` (keep in-memory table as fast-path cache or remove)
16. Stale detection: check OS process liveness + start-time match when asyncio task is dead

**Expected outcome:** Every sheet knows its process. Stale detection can kill zombie
processes that escaped the `finally` block (e.g., conductor SIGKILL). PID tracking
enables observability — `mzt status` could show which process is running each sheet.

### Phase 4: Restart Recovery

17. PID-based cleanup in `recover_job()` — kill orphans from previous conductor
18. Dual verification (start time + cmdline) with explicit `PID_REUSE_TOLERANCE_SECONDS` to prevent PID reuse kills on WSL2
19. Retire `pgroup.py` reaper (replaced by this system)

**Expected outcome:** Conductor restart leaves no orphan processes. Combined with
Phase 1's `finally` blocks (which handle normal exits) and Phase 3's PID tracking
(which enables targeted cleanup), this closes the last gap: processes that survived
a conductor crash without running their `finally` blocks.

## Escalation Notes

- **E-007 concern:** `process_pid` / `process_pgid` / `process_start_time` (Phase 3)
  are additive optional fields with None defaults. Existing checkpoints load
  without modification. This is NOT a non-additive checkpoint change.
- **E-002 concern:** No CLI command changes (status text is a display-only tweak).
- **E-001 concern:** No IPC protocol changes.
- **E-003 concern:** No daemon lifecycle changes (the conductor starts/stops
  the same way; process cleanup is internal).

## Risks

| Risk | Mitigation |
|------|-----------|
| WSL2 process group sensitivity | Only killpg on child session groups; daemon-own-group guard refuses to kill `os.getpgid(0)` |
| PID reuse after conductor restart | Dual verification: process start time (with `PID_REUSE_TOLERANCE_SECONDS` = 2s) + cmdline check |
| Race between kill and natural exit | `ProcessLookupError` caught everywhere; idempotent |
| `start_new_session` breaking instruments | Instrument profiles can opt out via existing config flag; baton-path override is explicit |
| SIGKILL race before pgid captured (daemon crashed mid-spawn) | Acknowledged narrow window; future work (cgroups) would close it |
| Validation command children surviving SIGTERM | 2-second grace then SIGKILL the group |
| Monitor auto-resume set lost on restart | Acceptable — orphan recovery handles paused jobs; user can `mzt resume` manually |
| Lightweight resume races with user pause | Check `user_paused` before auto-resuming; user pause always wins |
| Rate-limit / cost auto-resume races monitor pause | Both paths gated on `not monitor_paused` (Phase 2) |
| Monitor pause chatter on sustained pressure | Exponential backoff + manual-intervention ceiling |
| Musician-internal process bloat (out of scope) | Spec scoped to Marianne-owned subprocesses; see Future Work |

## Test Strategy

- **Unit:** `finally` block fires on CancelledError — mock subprocess, verify `killpg(pgid, SIGTERM)` then 2s wait then `killpg(pgid, SIGKILL)` sequence
- **Unit:** pgid captured at spawn, daemon-own-group safety check raises if pgid matches
- **Unit:** `start_new_session=True` propagated to subprocess_exec and subprocess_shell
- **Unit:** engine.py validation spawn uses process groups and kills tree on timeout
- **Unit:** `MonitorPauseJob` vs `PauseJob` — verify `monitor_paused` vs `user_paused` flags
- **Unit:** `MonitorResumeJob` clears pause without tearing down baton state
- **Unit:** rate-limit auto-resume skips when `monitor_paused=True`; cost auto-resume same
- **Unit:** oscillation guard — 4th pause within 5 minutes triggers exponential backoff; 11th within 10 minutes halts auto-resume
- **Unit:** SheetState with PID + pgid + start_time fields serializes/deserializes correctly (Phase 3)
- **Unit:** stale detection with start-time mismatch does NOT kill (PID reuse simulation)
- **Integration:** spawn subprocess with start_new_session, cancel task, verify no orphans
- **Integration:** monitor pauses job on pressure, pressure clears, job auto-resumes within one cycle
- **Integration:** user pauses a monitor-paused job, auto-resume skips it
- **Integration:** CLI shows correct pause-origin label in each combination
- **Adversarial — leak:** rapid pause/resume cycles do not leak processes (assert process count for known backends returns to baseline within 5s of cycle end)
- **Adversarial — bounds:** 3-sheet fan-out all running `pytest -n auto`. Peak process count no greater than `baseline + (sheets * xdist_workers_per_sheet) + 8`; returns to `baseline + 4` within 15s of concert completion. Exact numbers finalized at implementation; the point is bounded peak and bounded recovery time, not "bounded" as a qualitative claim.
- **Adversarial — chatter:** simulate sustained pressure; verify exponential backoff triggers and ceiling halts auto-resume cleanly
- **Adversarial — restart:** conductor SIGKILL with live processes, restart cleans up orphans with PID reuse guard (Phase 4)
- **Accepted false negative:** Phase 3 narrow-window orphan (SIGKILL between spawn and PID persist) — documented, not tested (cannot be deterministically reproduced)
- **Compound (with Rate Limit Primary spec):** rate limit mid-sheet, `rate_limit_primary=False`, monitor pressure spikes from retry, monitor pause, auto-resume, sheet completes cleanly without leaked processes

## Coordination with Rate Limit Primary Spec

Both specs modify `daemon/baton/events.py`, `daemon/baton/musician.py`, `daemon/baton/core.py`.
Landing order:

1. **Process Lifecycle Phase 1** (this spec) — unblocks runner-removal; touches
   `core.py` (deregister path). No new events in Phase 1.
2. **Rate Limit Primary B6** — adds `rate_limit_primary` to `SheetAttemptResult`
   (in `events.py`) and baton attempt handler (in `core.py`)
3. **Process Lifecycle Phase 2** — adds `MonitorPauseJob` / `MonitorResumeJob`
   events to `events.py` and handlers to `core.py`
4. **Process Lifecycle Phases 3-4** — schema + restart recovery

This ordering avoids conflicting edits on the same file regions. If timing forces
a different order, a rebase-and-reconcile step must explicitly re-verify the
rate-limit/cost auto-resume paths still gate on `monitor_paused`.

## Future Work

- **OS-level isolation (cgroups v2 / systemd scopes):** Both this spec and the
  rate-limit spec are engineering workarounds for the absence of OS-level process
  isolation. On systemd-managed Linux hosts, each job could become a scope unit;
  scope death would reap all member processes unconditionally. WSL2 is the target
  that makes this non-trivial today. Worth reconsidering when WSL2 either gains
  systemd parity or becomes a minority host.
- **Validation command ceiling:** Score YAML currently lets `command_succeeds:
  pytest -n auto` implicitly claim 1/N of the machine. A composer-configurable
  per-score `validation_max_parallelism` would let Marianne cap worker counts
  without requiring every score author to think about it.
- **Musician-internal process behavior:** If a musician's own child spawns
  (e.g., a musician running `make test` that forks heavily) becomes the bloat
  source, monitor pause is a no-op (the musician's children are not in the
  baton's `_active_pids`). Options include: extending the pgid grouping one
  level up (the musician process itself is the group), sandboxing musicians in
  bwrap or firejail, or resource limits via `setrlimit`.
- **Streaming / long-lived backend connections:** No current backend is streaming,
  but if one lands, the `finally`-block model needs rethinking — a stream may
  outlive a single `communicate()` call.
- **Narrow-window orphan reaper as last resort:** The SIGKILL-between-spawn-and-persist
  window cannot be closed by this design alone. A coarse periodic reaper
  (scanning `/proc` for known-backend cmdlines that are not in `_active_pids`
  with start-time-based false-positive filtering) could close it. Out of scope
  for Phase 1-4; revisit if the narrow window causes observable pain.
