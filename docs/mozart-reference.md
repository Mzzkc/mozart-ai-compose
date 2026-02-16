# Mozart AI Compose: The Definitive Reference

> **Purpose:** Comprehensive reference for AI assistants and developers working on Mozart.
> This document should be kept current by the `docs-generator` score.
>
> **Last updated:** 2026-02-14

---

## What Mozart IS

Mozart is a declarative orchestration framework that turns YAML job definitions into resilient, resumable, self-improving AI execution pipelines. It wraps Claude (CLI or API) in a sophisticated state machine that handles the messy reality of running AI at scale: rate limits, crashes, partial failures, validation, retry, and cross-job learning.

The musical metaphor runs deep. Jobs are "scores" (sheet music), units of work are "sheets" (pages of music), multiple jobs chain into "concerts," and the vision describes AI persons as "conductors." This isn't just naming — it shapes the architecture. A concert is an improvisational composition where jobs dynamically generate what comes next.

---

## The Complete Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer (Typer + Rich)                 │
│  26 commands: run, status, resume, pause, modify, validate,    │
│  diagnose, errors, recover, list, logs, history, dashboard,    │
│  mcp, config + 10 learning commands                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  Conductor (mozart start)                        │
│  Long-running process managing concurrent jobs via Unix socket  │
│  ┌──────────┬───────────┬──────────┬──────────┬──────────┐     │
│  │ Manager  │ Registry  │Scheduler │  Rate    │Backpress.│     │
│  │ job CRUD │ SQLite    │ DAG-aware│ Coordin. │ load mgmt│     │
│  │ lifecycle│ persist   │ priority │ cross-job│ memory   │     │
│  └──────────┴───────────┴──────────┴──────────┴──────────┘     │
│  ┌──────────┬──────────────────────────────────────────────┐   │
│  │IPC Server│ Supporting: Health, Monitor, PGroup,         │   │
│  │Unix sock │ LearningHub, Detection, Output Protocol      │   │
│  └──────────┴──────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────────────────┐
│                    Execution Engine (Runner)                     │
│  JobRunner: 6 mixins + base class via multiple inheritance                   │
│  ┌──────────┬───────────┬──────────┬──────────┬──────────┐     │
│  │Lifecycle │  Sheet    │ Recovery │ Patterns │   Cost   │     │
│  │ run()    │ execute() │ healing  │ query()  │ track()  │     │
│  │ modes    │ validate()│ retry    │ feedback │ limits   │     │
│  └──────────┴───────────┴──────────┴──────────┴──────────┘     │
│  ┌──────────┬──────────────────────────────────────────────┐   │
│  │Isolation │ Supporting: DAG, Parallel, CircuitBreaker,   │   │
│  │worktree  │ Escalation, Grounding, Synthesizer, Hooks    │   │
│  └──────────┴──────────────────────────────────────────────┘   │
└────────┬──────────┬─────────────┬──────────────┬───────────────┘
         │          │             │              │
┌────────▼──┐ ┌────▼─────┐ ┌────▼──────┐ ┌────▼─────────────┐
│ Backends  │ │  State   │ │ Learning  │ │  Notifications   │
│ Claude CLI│ │ JSON     │ │ Global    │ │  Desktop         │
│ Anthro API│ │ SQLite   │ │ Store     │ │  Slack           │
│ Ollama    │ │ Memory   │ │ (SQLite)  │ │  Webhook         │
│ Rec.Light │ │          │ │ Patterns  │ │                  │
└───────────┘ └──────────┘ └───────────┘ └──────────────────┘
```

---

## The Execution Flow (What Actually Happens)

When you run `mozart run job.yaml`:

1. **Parse & Validate** — YAML -> Pydantic JobConfig with 40+ config models. Fan-out expansion happens at parse time (e.g., stage 2 with fan_count=3 becomes sheets 2, 3, 4). Dependencies validated for cycles.

2. **Daemon Detection** — CLI checks for a running daemon via Unix socket. If found, job is submitted to daemon for execution. If not, the command exits with an error (daemon is required). Only `--dry-run` works without a daemon.

3. **Initialize State** — Creates CheckpointState or loads existing. Zombie detection: checks if PID in state file is actually alive via `os.kill(pid, 0)`. Dead PID -> recover to PAUSED.

4. **Setup Isolation** — If worktree mode enabled, creates git worktree (~24ms), overrides backend working directory. Locked to prevent accidental removal.

5. **Execute Sheets** — Two modes:
   - **Sequential:** DAG-aware ordering, one at a time
   - **Parallel:** Batches of independent sheets (up to max_concurrent), state protected by asyncio.Lock

   Per-sheet state machine:
   ```
   Setup -> Preflight -> Pattern Injection -> Execute -> Validate -> Decide
     |                                                            |
     |    +------------- COMPLETION (partial pass) <--------------+
     |    |                                                        |
     |    +------------- RETRY (exponential backoff) <------------+
     |    |                                                        |
     |    +------------- SELF-HEALING (retries exhausted) <-------+
     |    |                                                        |
     |    +------------- ESCALATE (low confidence) <--------------+
     |
     +-- Proactive Checkpoint (before dangerous ops)
   ```

6. **Error Classification** — Multi-phase: structured JSON errors -> exit code/signal -> regex fallback -> root cause selection across 50 error codes with priority-based tiebreaking. Rate limits get parsed reset times.

7. **Learning Aggregation** — On completion (or failure — survivorship bias fix), outcomes flow to GlobalLearningStore (~/.mozart/global-learning.db). Patterns detected, merged, trust-scored.

8. **Post-Success Hooks** — Concert chaining: completed jobs can spawn new jobs. Detached `run_job` hooks route through the daemon IPC when available, so chained jobs are tracked, rate-coordinated, and visible in `mozart list`. Falls back to subprocess when no daemon is running. `fresh: true` prevents infinite loops. Zero-work guard as defense-in-depth.

> **Defense-in-depth example:** When a job chains to itself, the child loads the parent's COMPLETED state. Without `--fresh`, it executes zero sheets and triggers on_success again — infinite loop. Mozart fixes this at two independent layers: `--fresh` deletes state (root cause), AND a zero-work guard skips hooks when nothing was done (symptom prevention). Either fix alone is sufficient.

---

## The Conductor (mozart start)

Mozart runs all jobs through a long-running conductor process. The conductor manages concurrent jobs, coordinates rate limits across jobs, and provides a Unix socket IPC interface.

### Quick Start

```bash
# Start the conductor (foreground, for development)
mozart start --foreground

# Start the conductor (background, production)
mozart start

# Submit a job (auto-routes through conductor)
mozart run my-job.yaml

# Check conductor status
mozart conductor-status

# List jobs
mozart list          # Active jobs only
mozart list --all    # Include completed/failed

# Stop the conductor
mozart stop
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Manager | `daemon/manager.py` | Job lifecycle, concurrency limits |
| Registry | `daemon/registry.py` | SQLite-backed persistent job tracking |
| Job Service | `daemon/job_service.py` | Core execution, decoupled from CLI |
| IPC Server | `daemon/ipc/server.py` | Unix socket + JSON-RPC 2.0 |
| Detection | `daemon/detect.py` | Auto-detect running daemon |
| Scheduler | `daemon/scheduler.py` | Cross-job sheet scheduling (built, not yet wired) |
| Rate Coordinator | `daemon/rate_coordinator.py` | Shared rate limit state (built, not yet wired) |
| Backpressure | `daemon/backpressure.py` | Memory-based load management |
| Monitor | `daemon/monitor.py` | CPU/memory/process tracking |
| Process Groups | `daemon/pgroup.py` | Prevent orphan processes |
| Learning Hub | `daemon/learning_hub.py` | Cross-job pattern sharing |
| Health | `daemon/health.py` | Health checks |

### Job IDs

Job IDs are human-friendly: the config file stem (e.g., `quality-continuous`). Duplicate names get `-2`, `-3` suffixes.

---

## The Learning System (Mozart's Brain)

Mozart doesn't just retry — it learns.

### Pattern Detection

8 pattern types extracted from outcomes: validation failures, retry successes, completion mode effectiveness, first-attempt successes, confidence patterns, semantic failures, output patterns (regex against stdout/stderr), and error code patterns.

### Pattern Lifecycle

```
Detected -> PENDING -> QUARANTINED --> VALIDATED --> RETIRED
                           |                |
                           +-- (cleared) ---+
```

### Trust Scoring

```
trust = 0.5 (base prior)
      + success_rate x 0.3
      - failure_rate x 0.4
      + age_factor x 0.2
      +/- quarantine adjustment
```

### Pattern Application

Epsilon-greedy: with probability epsilon (default 15%), lower-priority patterns are included to collect effectiveness data. This prevents cold-start death where new patterns never get tested.

### Cross-Job Coordination

- **Rate limit broadcasting:** Job A hits rate limit -> records to SQLite -> Job B checks before retrying
- **Pattern discovery broadcasting:** TTL-based (5 min) real-time pattern sharing between concurrent jobs
- **Learned wait times:** Average successful recovery waits, bounded and requiring minimum samples

### Entropy Monitoring

Shannon entropy over pattern application distribution. Low entropy triggers automatic response: boost exploration budget, revisit quarantined patterns. This prevents the learning system from collapsing to a single dominant pattern.

> **Laplace smoothing is critical.** New patterns get `effectiveness = (successes + 0.5) / (total + 1)`. Without the +0.5 prior, a pattern that succeeds once has 100% effectiveness and dominates. The prior makes new patterns start neutral (0.5) and converge to their true rate as evidence accumulates. This is Bayesian reasoning applied to AI orchestration.

---

## Score Anatomy

Every score needs exactly 3 required top-level fields plus optional configuration:

```yaml
name: "job-name"              # REQUIRED: unique identifier
sheet:                        # REQUIRED: how work is divided
  size: 1                     # items per sheet
  total_items: 9              # total items = 9 sheets when size=1
prompt:                       # REQUIRED: what the AI should do
  template: |                 # Jinja2 template with {{sheet_num}}, {{total_sheets}}, etc.
    ...
```

Everything else has sensible defaults.

### The 6 Score Archetypes

1. **Simple Task** (1-3 sheets, linear) — Quick one-off tasks
2. **Multi-Phase Pipeline** (5-10 sheets, strict dependencies) — Refactoring, building features
3. **Expert Review** (parallel fan-out + synthesis) — Code review, quality improvement
4. **Self-Improving Opus** (9 sheets, 6 movements, recursive) — Recursive self-improvement
5. **Concert Chain** (jobs spawning jobs) — Infinite improvement loops
6. **Issue-Driven Fixer** (dynamic scope) — Bug fixing, addressing deferred issues

### Key Template Variables

```
{{ sheet_num }}         - Current sheet number (1-indexed)
{{ total_sheets }}      - Total number of sheets
{{ start_item }}        - First item number for this sheet
{{ end_item }}          - Last item number for this sheet
{{ workspace }}         - Workspace directory path
{{ stage }}             - Original stage number (fan-out aware)
{{ instance }}          - Fan-out instance (1-indexed)
{{ fan_count }}         - Total instances for this stage
{{ previous_outputs }}  - Dict of previous sheet stdout (cross_sheet)
{{ previous_files }}    - Dict of captured file contents (cross_sheet)
+ any custom variables from prompt.variables
```

### Validation System — 5 Types

```yaml
validations:
  - type: file_exists       # Does the file exist?
    path: "{workspace}/result.md"

  - type: file_modified     # Was the file recently changed?
    path: "{workspace}/TRACKING.md"

  - type: content_contains  # Does file contain literal text?
    path: "{workspace}/result.md"
    pattern: "IMPLEMENTATION_COMPLETE: yes"

  - type: content_regex     # Does file match regex?
    path: "{workspace}/result.md"
    pattern: "FIXES_APPLIED: [0-9]+"

  - type: command_succeeds  # Does shell command return 0?
    command: "pytest -x -q --tb=no"
```

Key validation features:
- **Conditional:** `condition: "sheet_num >= 6"` applies only to later sheets
- **Retry with delay:** `retry_count: 3, retry_delay_ms: 200` for filesystem race conditions

### Advanced Features

| Feature | Config Key | Example |
|---------|-----------|---------|
| Fan-out parallelism | `sheet.fan_out: {2: 5}` | 5 parallel expert reviews |
| Sheet dependencies (DAG) | `sheet.dependencies: {3: [1, 2]}` | Fan-in after parallel |
| Cross-sheet context | `cross_sheet.auto_capture_stdout: true` | Pass outputs forward |
| Worktree isolation | `isolation.enabled: true` | Parallel-safe git ops |
| Concert chaining | `on_success: [{type: run_job}]` | Self-chaining loops |
| Workspace lifecycle | `workspace_lifecycle.archive_on_fresh: true` | Clean restarts |
| Cost limits | `cost_limits.max_cost_per_job: 100` | Budget enforcement |
| Backend selection | `backend.type: claude_cli` | LLM provider choice |
| Timeout overrides | `backend.timeout_overrides: {5: 7200}` | Per-sheet timeouts |
| Allowed tools | `backend.allowed_tools: [Read, Grep]` | Restrict agent tools |

### Production Best Practices

1. Always use `setsid` for detached execution
2. `backend.timeout_seconds: 3000` for complex sheets (50 minutes)
3. `isolation.enabled: true` for anything running in parallel
4. `retry.max_completion_attempts: 3` enables partial completion recovery
5. `cross_sheet.lookback_sheets: 3` when fan-out stages need all parallel outputs
6. Use `preamble` variable for shared context across all sheets
7. `workspace_lifecycle.archive_on_fresh: true` for self-chaining jobs
8. Validation markers (structured key=value files) for reliable completion checking

---

## Strengths

### Resilience Engineering
- Checkpoint after every sheet — crash anywhere, resume exactly
- Zombie detection via PID checking (not time-based — jobs can run for days)
- Atomic state saves (temp file + rename)
- Circuit breaker prevents cascading failures
- Graceful shutdown on SIGINT/SIGTERM; live config reload on SIGHUP
- Self-healing: auto-diagnosis + remediation when retries exhausted

### Observability
- Structured logging via structlog with context propagation
- Runner callback events at every lifecycle point (`sheet.started/completed/failed/retrying`, `sheet.validation_passed/failed`, `job.cost_update`, `job.iteration`) published via async event bus
- 50 error codes across 9 categories
- Execution history in SQLite for post-mortem
- Cost tracking with token-level granularity
- Web dashboard with SSE for real-time updates

### Composability
- Backend protocol: plug in any LLM (Claude CLI, API, Ollama, Recursive Light)
- State protocol: JSON for simplicity, SQLite for queries, Memory for tests
- Validation: file exists, file modified, content regex, command execution
- Notification: desktop, Slack, webhook
- Everything is YAML-configurable with sensible defaults

### Learning System
- Cross-workspace pattern sharing
- Trust scoring with quarantine lifecycle
- Epsilon-greedy exploration prevents local optima
- Entropy monitoring prevents convergence collapse

### Parallel Safety
- Git worktree isolation (~24ms overhead, shared objects)
- Locking during execution with stale lock recovery
- State mutex for concurrent sheet writes
- DAG-aware batch scheduling

---

## Known Weaknesses

1. **Resource Consumption** — Each Claude CLI process loads all plugins. Multiple concurrent processes can saturate memory.
2. **Claude-Focused** — Despite multiple backends, prompt templating and error classification are tuned for Claude.
3. **Learning Complexity** — 15-table SQLite database with 8 pattern types, hard to debug when patterns misbehave.
4. **Single-Machine** — No distributed execution. Learning store is local SQLite.
5. **No Streaming** — Batch-oriented. Long sheets provide minimal feedback beyond byte counters.
6. **Mixin Complexity** — JobRunner is 8 mixins via multiple inheritance. Understanding full flow requires reading 8 files sharing implicit state through `self`.

---

## By The Numbers

| Metric | Value |
|--------|-------|
| Source files | 182 Python files |
| Test files | 115 |
| CLI commands | 29 mozart commands |
| Packages | 20 under src/mozart/ |
| Config models | 40+ Pydantic models |
| Error codes | 50 across 9 categories |
| Learning tables | 15 SQLite tables |
| Pattern types | 8 |
| Backend types | 4 (Claude CLI, API, Ollama, Recursive Light) |
| State backends | 3 (JSON, SQLite, Memory) |
| Notification channels | 3 implemented (Desktop, Slack, Webhook). Email type exists in config schema but has no backend implementation. |

---

## The Vision: Federated AGI Infrastructure

VISION.md reveals Mozart's ultimate trajectory: infrastructure for collaborative intelligence. The Recursive Light Framework (RLF) creates LLMPerson entities with persistent identity, developmental stages, and autonomous judgment. Mozart becomes the substrate where these AI persons collaborate:

- Multiple conductors per concert (AI + human)
- Sheet-level conductor assignment
- Consensus mode: multiple perspectives required
- Person-aware learning: pattern effectiveness tracked per conductor
- Self-orchestration: AI persons initiate their own concerts
