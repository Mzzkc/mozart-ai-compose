# Mozart Usage Skill

> **Purpose**: Run, monitor, debug, and recover Mozart jobs. Covers the conductor, job lifecycle, diagnostics, config caching, recovery, and the anti-patterns that lose work.

---

## Triggers

| Use This Skill | Skip This Skill |
|---|---|
| Running/monitoring Mozart jobs | Writing new score configs (use mozart-score-authoring.md) |
| Debugging failed or stuck jobs | |
| Resuming interrupted jobs | |
| Understanding validation failures | |
| Conductor operations | |

---

## Conductor: The Required Foundation

**The conductor (daemon) is required for `mozart run`.** Without a running conductor, only `--dry-run` and `mozart validate` work.

### Starting the Conductor

```bash
# Foreground (development, see logs directly)
mozart start --foreground

# Background (production)
mozart start

# Detached from scripts (survives session end)
setsid mozart start &
```

### Conductor Commands

| Command | Purpose |
|---|---|
| `mozart start` | Start the conductor daemon |
| `mozart start --foreground` | Start in foreground (development) |
| `mozart start --profile dev` | Start with dev profile (debug logging, strace on) |
| `mozart start --profile intensive` | Start with intensive profile (48h timeout, high limits) |
| `mozart start --profile minimal` | Start with minimal profile (profiler + learning off) |
| `mozart stop` | Stop the conductor (**only when no jobs are running**) |
| `mozart stop --force` | Force-kill the conductor — **NEVER with active jobs** |
| `mozart restart` | Stop and restart |
| `mozart restart --profile dev` | Restart with a profile |
| `mozart conductor-status` | Check if conductor is running |

### How Jobs Route Through the Conductor

When you run `mozart run config.yaml`, the CLI checks for a running conductor via Unix socket. If found, the job is submitted through IPC and the CLI returns. The conductor manages job lifecycle, rate limit coordination across concurrent jobs, and event routing. If no conductor is found, the command exits with an error.

**The conductor runs your jobs.** `mozart run` is a client that submits work and returns. The job continues in the daemon regardless of whether your terminal stays open.

**NEVER stop the conductor while jobs are actively running.** Killing the daemon orphans all in-flight Claude agent processes and corrupts job state — sheets get stuck as `in_progress` with no validation or cleanup. To reload config on a running job, use `mozart modify -c new.yaml --resume --wait`. To safely stop: pause all jobs first, wait for pauses to take effect, then `mozart stop`.

---

## Job Lifecycle

### Submitting a Job

```bash
# Submit (routes through conductor)
mozart run my-score.yaml

# Dry run (no conductor needed, shows rendered prompts)
mozart run my-score.yaml --dry-run

# With self-healing enabled
mozart run my-score.yaml --self-healing

# Auto-confirm self-healing fixes
mozart run my-score.yaml --self-healing --yes

# Fresh start (clears previous state)
mozart run my-score.yaml --fresh

# Override workspace directory
mozart run my-score.yaml -w /absolute/path/to/workspace

# Start from specific sheet
mozart run my-score.yaml --start-sheet 3
```

`mozart run` is the only command that accepts `-w`/`--workspace`. All other commands resolve job context from the conductor's registry using the job ID.

### Monitoring

```bash
# Check status
mozart status my-job

# Live watch (refreshes every 5 seconds)
mozart status my-job --watch

# Custom refresh interval
mozart status my-job --watch --interval 10

# Machine-readable output
mozart status my-job --json

# List all active jobs
mozart list

# List all jobs including completed/failed
mozart list --all

# Filter by status
mozart list --status running
mozart list --status paused

# View job logs
mozart logs my-job

# Tail logs (follow mode)
mozart logs my-job --follow

# Show more lines
mozart logs my-job --lines 200

# Filter by level
mozart logs my-job --level ERROR

# View execution history
mozart history my-job
```

### Pausing

```bash
# Graceful pause at next sheet boundary
mozart pause my-job

# Wait for acknowledgment
mozart pause my-job --wait

# With timeout
mozart pause my-job --wait -t 60
```

### Resuming

```bash
# Resume from checkpoint
mozart resume my-job

# Resume (auto-reloads config from original YAML if it exists)
mozart resume my-job

# Resume with a different config file
mozart resume my-job -c fixed.yaml

# Resume using cached snapshot (skip auto-reload)
mozart resume my-job --no-reload

# Force resume a completed job
mozart resume my-job --force

# Resume with self-healing
mozart resume my-job --self-healing
```

### Modifying Running Jobs

`mozart modify` requires a new config file (`-c` is mandatory).

```bash
# Pause job and apply new config
mozart modify my-job -c updated.yaml

# Pause, apply new config, and immediately resume
mozart modify my-job -c updated.yaml --resume

# With wait for pause acknowledgment
mozart modify my-job -c updated.yaml --resume --wait
```

### `--fresh` vs `resume`

| Situation | Use |
|---|---|
| Job interrupted mid-progress | `mozart resume my-job` |
| Job failed, config needs fixing | `mozart resume my-job -c fixed.yaml` (auto-reloads) |
| Self-chaining: completed an iteration | `--fresh` (via hook config) |
| User explicitly wants to start over | `mozart run my-score.yaml --fresh` |
| Job was cancelled or partially failed | `mozart resume my-job` (try first) |

**`--fresh` deletes checkpoint state and archives workspace artifacts.** It wipes hours of work if used on an interrupted job. When in doubt, try `resume` first.

---

## Debugging Protocol (Mandatory Order)

**ALWAYS follow this sequence. Do NOT skip to manual investigation.**

```bash
# 1. ALWAYS start here
mozart status my-job

# 2. If failed --- get diagnostics
mozart diagnose my-job

# 3. Error details
mozart errors my-job --verbose

# 4. Filter errors by sheet, type, or code
mozart errors my-job --sheet 3
mozart errors my-job --type rate_limit
mozart errors my-job --code E201

# 5. Include log snippets in diagnostic
mozart diagnose my-job --include-logs

# 6. THEN manual investigation if needed
```

### Understanding `mozart status` Output

Key fields to check:
- **Status**: RUNNING, COMPLETED, FAILED, PAUSED, CANCELLED
- **Validation**: Pass/Fail per sheet (a sheet can execute successfully but fail validation)
- **Sheets**: N/M completed, which failed, which skipped
- **Rate limits**: Current wait count

**Critical insight**: `exit_code=0` does NOT mean success. Only `validation_passed=true` means success. A sheet can run to completion (exit 0) but fail its validations.

### Getting Validation Details

```bash
# Primary: use the errors command with verbose flag
mozart errors my-job --verbose

# Filter to just validation failures
mozart errors my-job --type permanent --verbose
```

### Common Failure Patterns

| Symptom | Likely Cause | Fix |
|---|---|---|
| File exists but "missing" | Wrong path syntax (`{{ }}` vs `{}`) | Check validation path syntax |
| Validation always passes | No validations configured, or too broad | Add meaningful validations |
| Pattern doesn't match | Regex anchors, escaping, or content changed | Test regex with `python3 -c "import re; ..."` |
| Command fails | Wrong `working_directory` or shell assumptions | Check CWD, test command manually |
| Sheet "passes" but work is bad | Validations too weak (file_exists only) | Add content checks or command validations |
| Config changes ignored | `--no-reload` used or file deleted | Config auto-reloads by default; check YAML file exists |
| Job hangs forever | Missing `skip_permissions: true` | Set in backend config |
| Chained job does nothing | Missing `fresh: true` in hook | Add `fresh: true` to self-chain hooks |

---

## Config Auto-Reload on Resume

**Mozart auto-reloads config from the original YAML file on resume.** The cached `config_snapshot` is a fallback when the file no longer exists on disk.

### Priority Order

1. Explicit `--config file.yaml` (always wins)
2. Auto-reload from stored `config_path` (default, if file exists)
3. Cached `config_snapshot` (fallback when file is gone or `--no-reload`)
4. Error (nothing available)

### When to Use `--no-reload`

Use `--no-reload` for deterministic replay from the cached snapshot:

```bash
# Use cached config (skip auto-reload even if YAML exists)
mozart resume my-job --no-reload

# Provide a different config file (overrides auto-reload)
mozart resume my-job -c fixed.yaml

# Nuclear --- start fresh (loses progress)
mozart run job.yaml --fresh
```

---

## Recovery Procedures

### Rate Limit Recovery

Mozart auto-waits when rate limited (default: 60 minutes, up to 24 cycles).

```bash
# Check if rate limited
mozart status my-job    # Shows PAUSED (rate_limited)

# If max_waits exhausted, just resume
mozart resume my-job
```

### Validation Failure Recovery

```bash
# 1. Check WHICH validation failed
mozart errors my-job --verbose

# 2a. Work is complete but validation config is wrong
mozart resume my-job -c fixed.yaml

# 2b. Work is incomplete --- Mozart retries automatically
mozart resume my-job
```

### Interrupted Job Recovery

```bash
# First: always try resume
mozart resume my-job

# If resume fails with stale PID --- auto-clears since fix b474d45
mozart resume my-job    # Retrying usually works

# If job is truly stuck, force resume
mozart resume my-job --force
```

### Registry Cleanup

```bash
# Clear completed/failed/cancelled jobs from the conductor registry
mozart clear

# Clear specific job(s)
mozart clear --job my-job

# Clear only failed jobs
mozart clear --status failed

# Clear jobs older than 1 hour (3600 seconds)
mozart clear --older-than 3600

# Skip confirmation
mozart clear --yes
```

---

## Self-Healing

```bash
# Enable automatic diagnosis and remediation
mozart run job.yaml --self-healing

# Auto-confirm suggested fixes
mozart run job.yaml --self-healing --yes

# Works with resume too
mozart resume my-job --self-healing
```

**How it works**: After all retries exhausted, diagnostic context is collected, applicable remedies identified and ranked, automatic fixes applied without prompting, suggested fixes prompt unless `--yes`.

**Built-in remedies**: create missing workspace, create parent directories, fix path separators (backslashes on Unix), suggest Jinja fixes, diagnose auth/CLI errors.

---

## Detached Execution

The conductor should be detached for long-running or unattended operation:

```bash
# CORRECT: Fully detached conductor
setsid mozart start &

# Development: foreground (stays attached to terminal)
mozart start --foreground
```

**Why setsid for the conductor?** Creates an independent session group. The conductor survives terminal close, context compaction, and session end.

**Jobs don't need setsid.** `mozart run` submits work to the conductor and returns. The job runs in the daemon regardless of your terminal session. Only the conductor itself needs to be detached.

```bash
# WRONG: External timeout corrupts state
timeout 600 mozart run my-score.yaml
```

Always let Mozart handle timeouts internally via `backend.timeout_seconds`.

---

## Error Codes

```
E0xx Execution    E4xx State
E1xx Rate Limit   E5xx Backend
E2xx Validation   E6xx Preflight
E3xx Config       E9xx Network
```

| Code | Retry? | Meaning |
|---|---|---|
| E001 | Yes | Timeout |
| E101 | Yes | API rate limit (~1hr wait) |
| E102 | Yes | CLI rate limit (~15min) |
| E201 | Yes | Expected file missing |
| E202 | Yes | Content doesn't match pattern |
| E301 | No | Invalid configuration |
| E401 | No | Checkpoint corruption |
| E601 | No | Required path missing |

---

## Concert/Chaining Operations

### Understanding Self-Chaining

Self-chaining scores chain into themselves via `on_success` hooks for iterative improvement:

```yaml
on_success:
  - type: run_job
    job_path: "examples/quality-continuous.yaml"
    detached: true
    fresh: true              # REQUIRED --- clears state
concert:
  enabled: true
  max_chain_depth: 10        # Safety limit
```

**`fresh: true` is mandatory.** Without it, the chained job loads COMPLETED state and does zero work.

### Monitoring Chains

```bash
# See all jobs (current and chained)
mozart list --all

# Check specific job in chain
mozart status quality-continuous-3

# View history of a chained job
mozart history quality-continuous-3
```

---

## Quick Reference

### Commands

| Command | Purpose |
|---|---|
| `mozart start` | Start conductor |
| `mozart stop` | Stop conductor |
| `mozart conductor-status` | Check conductor |
| `mozart run config.yaml` | Submit job |
| `mozart run config.yaml --dry-run` | Simulate (no conductor needed) |
| `mozart run config.yaml --fresh` | Submit with clean state |
| `mozart validate config.yaml` | Pre-flight check (no conductor needed) |
| `mozart status job` | Check progress |
| `mozart status job --watch` | Live monitoring |
| `mozart pause job` | Graceful pause |
| `mozart resume job` | Continue from checkpoint |
| `mozart resume job` | Resume (auto-reloads config from YAML) |
| `mozart resume job -c new.yaml` | Resume with different config file |
| `mozart resume job --no-reload` | Resume using cached snapshot |
| `mozart modify job -c new.yaml --resume` | Pause + resume with new config |
| `mozart diagnose job` | Full diagnostic report |
| `mozart errors job --verbose` | Error details with stdout/stderr |
| `mozart list` | List active jobs |
| `mozart list --all` | List all jobs (including finished) |
| `mozart clear --job job` | Remove job from registry |
| `mozart logs job` | View log entries |
| `mozart logs job --follow` | Tail logs |
| `mozart history job` | Execution history |
| `mozart dashboard` | Start web dashboard |

### Critical Rules

| Never | Always |
|---|---|
| `timeout 600 mozart run ...` | Let Mozart handle timeouts internally |
| Assume exit_code=0 is success | Check `validation_details` |
| Debug manually first | Use `status` -> `diagnose` -> `errors` |
| Edit YAML then plain `resume` | Config auto-reloads; just `mozart resume` |
| Kill running job (SIGKILL) | Use `mozart pause` for graceful stop |
| Edit config during run | Pause first, then `mozart modify` |
| Use `--fresh` on interrupted jobs | Use `resume` (try first, fresh loses progress) |

---

## Reference

- Score authoring: `~/.claude/skills/mozart-score-authoring.md`
- Full CLI reference: `docs/cli-reference.md`
- Daemon guide: `docs/daemon-guide.md`
- Config reference: `docs/configuration-reference.md`
- Limitations: `docs/limitations.md`

---

*Mozart Usage Skill --- operational guide for running, monitoring, and debugging Mozart AI Compose jobs.*
