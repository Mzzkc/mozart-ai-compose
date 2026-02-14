# Known Limitations

An honest accounting of what Mozart doesn't do, what's incomplete, and where sharp edges exist. Each limitation includes the technical reason it exists and any available workaround.

---

## Execution Model

### Daemon Required for Execution

`mozart run` routes all jobs through the `mozartd` daemon. There is no standalone execution mode.

**What this means:** You must start the daemon (`mozartd start`) before running any job. Only `mozart validate` and `mozart run --dry-run` work without a running daemon.

**Why:** Centralized resource management, rate-limit coordination, and backpressure control require a single process to track all active jobs. The daemon also enables the persistent job registry and crash recovery.

**Workaround:** None. Start the daemon first:

```bash
mozartd start           # background
mozart run my-job.yaml  # now works
```

**Status:** Permanent design choice.

---

### Escalation Incompatible with Daemon

The `--escalation` flag (human-in-the-loop prompts for low-confidence sheets) is explicitly blocked when the daemon routes jobs.

**What this means:** You cannot use `mozart run my-job.yaml --escalation`. The CLI exits with an error:

> `--escalation requires interactive console prompts which are not available in daemon mode. Escalation is not currently supported.`

**Why:** The daemon runs jobs as background tasks without an attached terminal. Interactive console prompts have no UI surface to display through.

**Workaround:** None currently. The escalation feature exists in the codebase (`src/mozart/learning/store/escalation.py`) but has no supported execution path.

**Status:** Blocked until the dashboard or IPC layer supports interactive prompts.

---

### No Output Streaming

Execution is batch-oriented. stdout and stderr are captured after each sheet completes, not streamed in real time.

**What this means:**

- You cannot watch AI output as it generates
- `stdout_tail` in the state file is truncated to the last 500 characters (for log display) or ~10 KB (for self-healing diagnostic context)
- Long-running sheets provide no progress indication until completion

**Why:** The `asyncio.create_subprocess_exec` backend collects output via `communicate()`, which buffers until process exit. Streaming would require line-by-line reads with significant complexity for timeout handling.

**Relevant constants** (from `src/mozart/core/constants.py`):

| Constant | Value | Purpose |
|----------|-------|---------|
| `TRUNCATE_STDOUT_TAIL_CHARS` | 500 | Log display truncation |
| `HEALING_CONTEXT_TAIL_CHARS` | 10,000 | Self-healing diagnostic context |

**Workaround:** For long jobs, use `mozart status <job> --watch` to poll completion state, or `tail -f workspace/mozart.log` for log-level updates.

**Status:** Permanent for the current architecture.

---

## Daemon Internals

### Phase 3 Components Not Wired

Two major daemon subsystems are fully built, tested, and present in the codebase but not connected to the execution path:

| Component | File | What It Does |
|-----------|------|-------------|
| `GlobalSheetScheduler` | `src/mozart/daemon/scheduler.py` | Cross-job sheet scheduling with priority, fair-share, and DAG awareness |
| `RateLimitCoordinator` | `src/mozart/daemon/rate_coordinator.py` | Shares rate-limit state across concurrent jobs |

**What this means:** Jobs currently run monolithically via `JobService.start_job()`. The scheduler is instantiated lazily but doesn't drive execution. The rate coordinator's write path IS active (rate limit events flow from runners through `RunnerContext.rate_limit_callback` to the coordinator via `JobManager._on_rate_limit`), but its read path is not wired — the scheduler doesn't consume the collected data yet.

**Why:** These were built as Phase 3 infrastructure. Integration requires replacing the monolithic job execution path with per-sheet dispatch through the scheduler — a significant change that hasn't been prioritized.

**Impact:** Without the scheduler, concurrent daemon jobs don't share rate-limit intelligence or do cross-job fair-share scheduling. Each job manages its own retries independently.

**Status:** Planned for future integration. Infrastructure is ready.

---

### Single-Machine Only

Mozart runs on a single machine. There is no distributed execution, remote workers, or cluster mode.

**What this means:**

- The daemon binds to a local Unix socket for IPC
- Worktree isolation creates local git worktrees
- All concurrent jobs share the same machine's CPU, memory, and network

**Why:** The Unix socket IPC layer (`src/mozart/daemon/ipc/`) is inherently local. Distributed coordination would require a fundamentally different architecture (message queues, consensus protocols, distributed state).

**Workaround:** Run separate Mozart instances on different machines with different workspaces. They won't coordinate rate limits, but they'll operate independently.

**Status:** Permanent design choice. Distributed execution is out of scope.

---

## Backend Support

### Claude-Centric Design

Mozart was designed around the Claude CLI and Anthropic API. While the backend is pluggable (abstract `Backend` class), the error classification and rate-limit detection are tuned for Claude's output patterns.

**Available backends:**

| Backend | File | Maturity |
|---------|------|----------|
| `ClaudeCliBackend` | `backends/claude_cli.py` | Primary, battle-tested |
| `AnthropicApiBackend` | `backends/anthropic_api.py` | Functional, less tested |
| `OllamaBackend` | `backends/ollama.py` | Community-contributed |
| `RecursiveLightBackend` | `backends/recursive_light.py` | Experimental |

**What this means:**

- Default rate-limit patterns (e.g., `hit.*limit`, `limit.*resets?`, `daily.*limit`) were derived from Claude CLI output
- The error classifier's exit-code handling has Claude-specific logic (exit code 1 = "task complete but with issues" in Claude CLI text mode)
- Non-Claude backends work but may not get optimal error recovery

**Why:** Mozart originated as an orchestrator for Claude CLI jobs. The patterns are generic enough to catch common HTTP 429/rate-limit messages from any backend, but edge cases in non-Claude backends may be missed.

**Workaround:** You can supply custom `rate_limit_patterns` via the job config's `error_handling` section. The error classifier accepts user-defined regex patterns that override or extend the defaults.

**Status:** Permanent primary focus. Other backends are supported but secondary.

---

## Architecture Complexity

### Runner Mixin Architecture

The `JobRunner` class is composed of 6 mixins plus a base class (7 classes total) via multiple inheritance:

```
JobRunner(
    SheetExecutionMixin,  # Sheet execution (~3,000 lines)
    LifecycleMixin,       # run() orchestration
    RecoveryMixin,        # Error classification and retry
    PatternsMixin,        # Pattern management
    CostMixin,            # Cost tracking
    IsolationMixin,       # Worktree isolation
    JobRunnerBase,        # Core initialization
)
```

**What this means:**

- Debugging requires tracing method calls across multiple files in `src/mozart/execution/runner/`
- MRO (Method Resolution Order) determines which mixin's method wins
- State is shared across mixins via `self`, with no encapsulation between them
- `SheetExecutionMixin` alone (`sheet.py`) is ~3,000 lines

**Why:** The runner grew organically. Each concern (cost, isolation, recovery) was added as a mixin to avoid a single 5,000+ line file.

**Workaround:** When debugging, start from `base.py` (initialization) and `lifecycle.py` (`run()` entry point), then trace into specific mixins.

**Status:** Working but complex. No refactor planned.

---

### Learning System Complexity

The learning store has 14 modules in `src/mozart/learning/store/`:

```
base.py                   budget.py
drift.py                  escalation.py
executions.py             models.py
patterns.py               patterns_broadcast.py
patterns_crud.py           patterns_quarantine.py
patterns_query.py          patterns_success_factors.py
patterns_trust.py          rate_limits.py
```

**What this means:** The learning system tracks pattern drift, entropy, trust scores, quarantine state, success factors, and budget constraints. For simple jobs (run 5 sheets sequentially), most of this infrastructure is unused overhead.

**Why:** Designed for long-running, repeated jobs where learning from past failures materially improves success rates. The complexity pays for itself in those scenarios.

**Workaround:** Learning is opt-in via the `learning:` config section. Omit it for simple jobs and the store still initializes but doesn't collect meaningful data.

**Status:** Permanent. The learning system is a core differentiator.

---

## Dashboard

### Dashboard Coverage

The dashboard UI is functional but has limited coverage:

**What works:**

- Job listing and detail views
- Sheet status display with stdout/stderr tails
- Score editor with real-time validation
- SSE streaming infrastructure (`SSEManager` with heartbeats, broadcasts, per-job client tracking)
- SSE integration in job detail and dashboard index pages via HTMX

**What's limited:**

- Real-time updates depend on the SSE connection; no polling fallback
- No historical trend visualization
- No cost tracking display (CostMixin tracks costs, but the dashboard doesn't surface them)
- No learning insights visualization

**Why:** The dashboard was built as a monitoring tool, not a full management UI. The SSE infrastructure is solid but the UI only consumes a subset of available events.

**Status:** Functional for monitoring. Not a priority for expansion.

---

## Validation

### Validation Condition Expressions

Validation `condition` fields support comparison expressions with any context variable and boolean AND:

```yaml
validations:
  - type: file_exists
    path: "output.md"
    condition: "sheet_num >= 3"                        # Simple comparison
    condition: "stage == 2 and instance == 1"          # Boolean AND with fan-out variables
    condition: "sheet_num >= 3 and sheet_num <= 5"     # Range check
```

**Supported operators:** `>=`, `<=`, `==`, `!=`, `>`, `<`

**Supported variables:** Any variable from the sheet context — `sheet_num`, `stage`, `instance`, `fan_count`, `total_stages`, and any user-defined `prompt.variables`.

**What's NOT supported:**

- Complex expressions: `sheet_num in [1, 3, 5]`
- Boolean OR: `sheet_num == 3 or sheet_num == 5`
- Nested expressions: `(sheet_num > 2) and (stage < 3)`

**Why:** The condition evaluator in `engine.py` splits on `" and "` and evaluates each clause as a `variable operator value` triple. Unrecognized conditions silently fall back to "always apply" behavior.

**Workaround:** For OR logic, use multiple validation entries, each with a simple condition.

**Status:** Sufficient for current use cases.

---

## Configuration

### Validation Stage Limits

Validation stages are capped at 1-10. You cannot define more than 10 sequential validation stages per sheet.

**Why:** A practical limit to prevent unbounded validation chains. The `stage` field is validated with `ge=1, le=10`.

**Status:** Permanent. 10 stages covers all practical use cases.

### Process Timeout Default

The default backend timeout is **1800 seconds** (30 minutes), set via `BackendConfig.timeout_seconds`. This is the timeout users should be aware of.

There is also an internal constant `PROCESS_DEFAULT_TIMEOUT_SECONDS = 300` used as a fallback when no config is loaded, but this is never reached in normal operation — the Pydantic model always provides the 1800s default.

**Workaround:** Override the timeout in your job config:

```yaml
backend:
  timeout_seconds: 3600  # 60 minutes for long-running sheets
```

**Status:** Permanent. Explicit timeouts are safer than high defaults.
