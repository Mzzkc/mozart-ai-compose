# Mozart AI Compose - Usage Skill

> **TL;DR**: Mozart is a **general-purpose cognitive orchestration system** that orchestrates Claude prompts across sheets with validation, retry, and state management. It works for coding, research, writing, data curation, strategic planning, and any task with multi-phase workflows and clear validation criteria. Use `mozart status` → `diagnose` → `errors` BEFORE manual investigation.

---

## Triggers

| Use This Skill | Skip This Skill |
|----------------|-----------------|
| Running/debugging Mozart jobs | Single Claude CLI calls |
| Understanding validation failures | Non-orchestrated AI work |
| Resuming interrupted jobs | Unrelated batch tools |
| Writing job configurations | |
| **Multi-phase research projects** | |
| **Long-form writing orchestration** | |
| **Data curation pipelines** | |
| **Strategic planning workflows** | |

---

## Quick Reference

### Commands

| Command | Purpose |
|---------|---------|
| `mozart run config.yaml` | Execute job |
| `mozart status job-id -w ./ws` | Check progress |
| `mozart resume job-id -w ./ws` | Continue from checkpoint |
| `mozart pause job-id -w ./ws` | Pause running job |
| `mozart modify job-id -c new.yaml` | Modify config and resume |
| `mozart diagnose job-id -w ./ws` | Full diagnostic report |
| `mozart errors job-id -V` | List errors with details |
| `mozart validate config.yaml` | Validate config syntax |
| `mozart list -w ./ws` | List all jobs |
| `mozart logs job-id -w ./ws` | View/tail log files |
| `mozart dashboard -w ./ws` | Start web dashboard |
| `mozart patterns` | View global learning patterns |
| `mozart pattern-show <id>` | Show pattern details with provenance |
| `mozart pattern-quarantine <id>` | Quarantine a suspicious pattern |
| `mozart pattern-validate <id>` | Validate a quarantined pattern |
| `mozart recalculate-trust` | Recalculate all pattern trust scores |

### Global Options

All commands support these global options:

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version |
| `--verbose` | `-v` | Detailed output |
| `--quiet` | `-q` | Errors only |
| `--log-level` | `-L` | DEBUG, INFO, WARNING, ERROR |
| `--log-file` | | Path for log output |
| `--log-format` | | json, console, or both |

### Debug Order (MANDATORY)

```bash
mozart status job-id -w ./ws      # 1. Always start here
mozart diagnose job-id -w ./ws    # 2. If failed
mozart errors job-id -V           # 3. Error details
# 4. THEN manual investigation
```

### Critical Rules

| Never | Always |
|-------|--------|
| `timeout 600 mozart run ...` | `mozart run ...` (internal timeout) |
| `mozart run job.yaml &` | `setsid mozart run ... &` (fully detached) |
| Assume exit_code=0 → success | Check `validation_details` |
| Debug manually first | Use Mozart tools first |

---

## Command Reference

### `mozart run`

```bash
mozart run config.yaml              # Execute job
mozart run config.yaml --dry-run    # Preview without running
mozart run config.yaml -s 5         # Start from sheet 5
mozart run config.yaml -j           # JSON output
```

| Option | Short | Description |
|--------|-------|-------------|
| `--dry-run` | `-n` | Preview execution |
| `--start-sheet` | `-s` | Override starting sheet |
| `--json` | `-j` | Machine-readable output |

### `mozart status`

```bash
mozart status job-id -w ./ws         # Check status
mozart status job-id --watch         # Live monitoring
mozart status job-id -W -i 10        # Watch with 10s interval
mozart status job-id -j              # JSON output
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--json` | `-j` | JSON output |
| `--watch` | `-W` | Continuous monitoring |
| `--interval` | `-i` | Watch refresh seconds (default: 5) |

### `mozart pause`

Pause a running job gracefully. Creates a pause signal that the job detects at the next sheet boundary.

```bash
mozart pause job-id -w ./ws         # Pause running job
mozart pause job-id --wait          # Wait for acknowledgment
mozart pause job-id --wait -t 30    # Wait with 30s timeout
mozart pause job-id -j              # JSON output
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--wait` | | Wait for pause acknowledgment |
| `--timeout` | `-t` | Wait timeout seconds (default: 60) |
| `--json` | `-j` | JSON output |

**How it works:**
1. Creates a `.mozart-pause-{job_id}` signal file in the workspace
2. Running job checks for this file between sheets
3. Job saves state and transitions to PAUSED status
4. Signal file is removed after acknowledgment

**Use cases:**
- Gracefully stop a job to inspect intermediate results
- Prepare for config modifications
- Free up resources temporarily

### `mozart modify`

Modify a job's configuration and optionally resume execution. Combines pause + config validation in one command.

```bash
mozart modify job-id -c updated.yaml              # Modify config only
mozart modify job-id -c new.yaml -r               # Modify and resume
mozart modify job-id -c updated.yaml -r -w ./ws   # With workspace
mozart modify job-id -c updated.yaml -r --wait    # Wait for pause before resume
```

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | New configuration file (required) |
| `--workspace` | `-w` | Workspace directory |
| `--resume` | `-r` | Immediately resume with new config after pausing |
| `--wait` | | Wait for job to pause before resuming (when --resume) |
| `--timeout` | `-t` | Timeout in seconds for pause acknowledgment (default: 60) |
| `--json` | `-j` | JSON output |

**How it works:**
1. Validates the new config file
2. If job is running, sends pause signal
3. Optionally waits for pause acknowledgment
4. If `--resume`, starts resume with new config

**Use cases:**
- Change prompts mid-job based on early results
- Adjust timeouts or retry settings
- Fix configuration errors without losing progress
- Swap models or backend settings

### `mozart resume`

```bash
mozart resume job-id -w ./ws         # Resume from checkpoint
mozart resume job-id -c new.yaml     # Resume with new config
mozart resume job-id --force         # Force resume completed job
mozart resume job-id -r              # Reload config from original YAML
mozart resume job-id -r -c fix.yaml  # Reload from different YAML file
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--config` | `-c` | Override config file |
| `--force` | `-f` | Resume even if completed |
| `--reload-config` | `-r` | Reload config from YAML instead of using cached snapshot |

### `mozart errors`

```bash
mozart errors job-id -V              # Verbose with stdout/stderr
mozart errors job-id -b 3            # Sheet 3 only
mozart errors job-id -t transient    # Only transient errors
mozart errors job-id -c E001         # Only timeout errors
mozart errors job-id -j              # JSON output
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--verbose` | `-V` | Show stdout/stderr tails |
| `--sheet` | `-b` | Filter by sheet number |
| `--type` | `-t` | Filter: transient, rate_limit, permanent |
| `--code` | `-c` | Filter by error code (E001, E101, etc.) |
| `--json` | `-j` | JSON output |

### `mozart list`

```bash
mozart list -w ./ws                  # List all jobs
mozart list -s running               # Only running jobs
mozart list -l 50                    # Show 50 jobs max
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--status` | `-s` | Filter: pending, running, completed, failed, paused |
| `--limit` | `-l` | Max jobs to display (default: 20) |

### `mozart logs`

```bash
mozart logs -w ./ws                  # Recent logs (all jobs)
mozart logs job-id -w ./ws           # Filter by job
mozart logs -F                       # Follow (like tail -f)
mozart logs -n 100                   # Last 100 lines
mozart logs -l ERROR                 # ERROR and above
mozart logs -j                       # Raw JSON entries
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--file` | `-f` | Specific log file path |
| `--follow` | `-F` | Tail the log file |
| `--lines` | `-n` | Lines to show (default: 50, 0=all) |
| `--level` | `-l` | Minimum level: DEBUG, INFO, WARNING, ERROR |
| `--json` | `-j` | Raw JSON entries |

### `mozart dashboard`

```bash
mozart dashboard                     # Start on localhost:8000
mozart dashboard -p 3000             # Custom port
mozart dashboard --host 0.0.0.0      # External connections
mozart dashboard -r                  # Auto-reload (dev mode)
```

| Option | Short | Description |
|--------|-------|-------------|
| `--port` | `-p` | Port (default: 8000) |
| `--host` | | Bind address (default: 127.0.0.1) |
| `--workspace` | `-w` | Workspace directory |
| `--reload` | `-r` | Auto-reload for development |

### `mozart diagnose`

```bash
mozart diagnose job-id -w ./ws       # Full diagnostic report
mozart diagnose job-id -j            # JSON output
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--json` | `-j` | JSON output |

---

## Pause/Modify Workflow

### Common Patterns

**Pause and inspect:**
```bash
mozart pause my-job -w ./workspace
# Inspect intermediate results...
cat workspace/sheet-3-output.md
# Resume when ready
mozart resume my-job -w ./workspace
```

**Modify config on the fly:**
```bash
# Running job needs different timeout
mozart modify my-job -c updated.yaml -r -w ./workspace
```

**Two-step modify for inspection:**
```bash
mozart pause my-job -w ./workspace
# Review state, make config changes...
mozart resume my-job --reload-config -c fixed.yaml -w ./workspace
```

**Graceful overnight pause:**
```bash
# End of day - pause all jobs
for job in $(mozart list -s running -w ./workspace --json | jq -r '.[].job_id'); do
  mozart pause "$job" -w ./workspace
done

# Morning - resume
for job in $(mozart list -s paused -w ./workspace --json | jq -r '.[].job_id'); do
  mozart resume "$job" -w ./workspace
done
```

### When to Use Pause vs Modify

| Scenario | Use |
|----------|-----|
| Just want to stop temporarily | `pause` |
| Need to change config and continue | `modify -r` |
| Want to inspect before resuming | `pause`, then `resume --reload-config` |
| Config error causing failures | `modify -c fixed.yaml -r` |

---

## Error Codes

Error codes are organized by category (first digit after E):

### E0xx: Execution Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E001 | Yes | Timeout (command exceeded time limit) |
| E002 | Yes | Killed by signal (external termination) |
| E003 | No | Crash (segfault, bus error, abort) |
| E004 | No | Interrupted by user (SIGINT/Ctrl+C) |
| E005 | No | Out of memory (OOM killer) |
| E009 | Yes | Unknown execution error |

### E1xx: Rate Limit / Capacity

| Code | Retry? | Meaning |
|------|--------|---------|
| E101 | Yes | API rate limit (wait ~1hr) |
| E102 | Yes | CLI rate limit (wait ~15min) |
| E103 | Yes | Capacity exceeded (wait ~5min) |

### E2xx: Validation Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E201 | Yes | Expected file missing |
| E202 | Yes | Content doesn't match pattern |
| E203 | Yes | Validation command failed |
| E204 | Yes | Validation timed out |
| E209 | Yes | Generic validation needed |

### E3xx: Configuration Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E301 | No | Invalid configuration |
| E302 | No | Missing required field |
| E303 | No | Config file not found |
| E304 | No | YAML/JSON parse error |
| E305 | No | MCP/plugin error |
| E306 | No | CLI mode mismatch |

### E4xx: State Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E401 | No | Checkpoint corruption |
| E402 | Yes | State load failed |
| E403 | Yes | State save failed |
| E404 | No | State version mismatch |

### E5xx: Backend Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E501 | Yes | Connection failed / Job not found |
| E502 | No | Auth/authorization failed / Job not in valid state |
| E503 | Yes | Invalid response / Cannot create signal |
| E504 | Yes | Backend timeout / Pause not acknowledged |
| E505 | No | Backend not found (ENOENT) / Invalid config |

### E6xx: Preflight Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E601 | No | Required path missing |
| E602 | No | Prompt too large |
| E603 | No | Working directory invalid |
| E604 | No | Validation setup invalid |

### E9xx: Network Errors

| Code | Retry? | Meaning |
|------|--------|---------|
| E901 | Yes | Connection failed/refused |
| E902 | Yes | DNS resolution failed |
| E903 | Yes | SSL/TLS error |
| E904 | Yes | Network timeout |
| E999 | Yes | Unknown error |

---

## Configuration

### Minimal Config

```yaml
name: "my-job"
workspace: "./workspace"
sheet:
  size: 10
  total_items: 100
prompt:
  template: |
    Process items {{ start_item }} to {{ end_item }}.
```

### Fan-Out: Parameterized Stage Instantiation

Fan-out lets one stage definition instantiate N parallel copies, each receiving unique `{{ stage }}`, `{{ instance }}`, and `{{ fan_count }}` variables. This avoids duplicating nearly-identical sheets for parallel work.

```yaml
sheet:
  size: 1                    # Required: must be 1 for fan-out
  total_items: 7             # 7 logical stages
  fan_out:
    2: 3                     # Stage 2 → 3 parallel instances
    4: 3                     # Stage 4 → 3 parallel instances
    5: 3                     # Stage 5 → 3 parallel instances
  dependencies:
    2: [1]                   # Investigate depends on survey
    3: [2]                   # Adversarial waits for ALL investigations (fan-in)
    4: [3]                   # Execute depends on adversarial
    5: [4]                   # Review.i depends on execute.i (instance-matched)
    6: [5]                   # Finalize waits for ALL reviews (fan-in)
    7: [6]                   # Commit depends on finalize

parallel:
  enabled: true
  max_concurrent: 3

prompt:
  template: |
    {% if stage == 1 %}
    Pick 3 issues to fix...
    {% elif stage == 2 %}
    You are investigator {{ instance }} of {{ fan_count }}.
    Investigate issue {{ instance }}...
    {% elif stage == 3 %}
    Review all {{ fan_count }} investigations...
    {% endif %}
```

**Expansion:** 7 stages → 13 concrete sheets (1 + 3 + 1 + 3 + 3 + 1 + 1).

**Dependency patterns:**

| Pattern | When | Rule |
|---------|------|------|
| Fan-out (1→N) | Single stage → fanned stage | Each instance depends on the single source |
| Fan-in (N→1) | Fanned stage → single stage | Target depends on ALL instances |
| Instance-matched (N→N) | Same fan count | Instance i depends on instance i |
| Cross-fan (N→M) | Different fan counts, both >1 | All-to-all (conservative) |

**Constraints:** `fan_out` requires `size: 1` and `start_item: 1`.

**Resume safety:** After expansion, `fan_out` is cleared and `fan_out_stage_map` stores per-sheet metadata. Resuming a paused job re-parses the expanded config without re-expanding.

### Full Config Reference

```yaml
# === REQUIRED ===
name: "job-name"
description: "Human-readable job description"
workspace: "./workspace"
sheet:
  size: 10
  total_items: 100
  start_item: 1
  # Fan-out (optional): stage → instance count
  # fan_out: { 2: 3, 4: 3 }
  # dependencies: { 2: [1], 3: [2] }

prompt:
  template: |                    # OR template_file: ./prompt.j2
    Process {{ start_item }}-{{ end_item }}.
  variables: { key: value }
  stakes: "STAKES: ..."
  thinking_method: "..."

# === BACKEND ===
backend:
  type: claude_cli              # claude_cli | anthropic_api | recursive_light

  # CLI options (type: claude_cli)
  skip_permissions: true        # --dangerously-skip-permissions
  disable_mcp: true             # Disable MCP servers (~2x faster)
  output_format: json           # json | text | stream-json
  cli_model: claude-sonnet-4-20250514  # Override model
  allowed_tools:                # Restrict to specific tools
    - Read
    - Grep
    - Glob
  system_prompt_file: ./system.md  # Custom system prompt
  working_directory: ./code
  timeout_seconds: 1800
  cli_extra_args: ["--flag"]    # Escape hatch for new flags

# === TIMING ===
pause_between_sheets_seconds: 10

# === RETRY ===
retry:
  max_retries: 3
  base_delay_seconds: 10
  max_delay_seconds: 3600
  exponential_base: 2.0
  jitter: true

# === RATE LIMIT ===
rate_limit:
  wait_minutes: 60
  max_waits: 24
  detection_patterns: ["rate.?limit", "429"]

# === VALIDATIONS ===
validations:
  - type: file_exists
    path: "{workspace}/output-{sheet_num}.md"
    description: "Output created"
    stage: 1

  - type: content_contains
    path: "{workspace}/output.md"
    pattern: "COMPLETE"
    stage: 2

  - type: command_succeeds
    command: "cargo test"
    working_directory: "./code"
    stage: 3

# === NOTIFICATIONS ===
notifications:
  - type: desktop
    on_events:
      - job_complete
      - job_failed
      - job_paused
    config: {}
```

### Built-in Template Variables

| Variable | Example | Description |
|----------|---------|-------------|
| `{{ sheet_num }}` | `5` | Current sheet number (concrete, after expansion) |
| `{{ total_sheets }}` | `13` | Total sheets in job (after fan-out expansion) |
| `{{ start_item }}` | `41` | First item in this sheet |
| `{{ end_item }}` | `50` | Last item in this sheet |
| `{{ workspace }}` | `./workspace` | Workspace directory path |
| `{{ job_name }}` | `my-job` | Job identifier |
| `{{ stage }}` | `2` | Logical stage number (equals `sheet_num` without fan-out) |
| `{{ instance }}` | `1` | Instance within fan-out group (1-indexed, default 1) |
| `{{ fan_count }}` | `3` | Total instances in this stage's fan-out (default 1) |
| `{{ total_stages }}` | `7` | Original stage count before expansion |

---

## Recovery Procedures

### Rate Limit Recovery

```bash
# Mozart auto-waits. If max_waits exhausted:
mozart status job-id -w ./ws     # Shows: PAUSED (rate_limited)
# Wait manually, then:
mozart resume job-id -w ./ws
```

### Validation Failure Recovery

```bash
# 1. Check which validation failed
mozart errors job-id -V

# 2a. If work complete but validation wrong → fix config
mozart modify job-id -c fixed-config.yaml -r -w ./ws

# 2b. If work incomplete → just resume (Mozart retries)
mozart resume job-id -w ./ws
```

### State Corruption Recovery

```bash
# Start fresh from specific sheet
rm workspace/.mozart-state.db    # SQLite backend
# OR: rm workspace/.mozart-state.json  # JSON backend
mozart run job.yaml --start-sheet N
```

---

## Anti-Patterns

| Don't | Why | Do Instead |
|-------|-----|------------|
| `timeout 600 mozart run ...` | SIGKILL corrupts state | Mozart handles timeout internally |
| `mozart run job.yaml &` | Dies on session end | `setsid mozart --log-file ws/mozart.log run job.yaml &` |
| Check `exit_code` for success | Validations may have failed | Check `validation_details` |
| Debug manually first | Mozart tools provide context | `status` → `diagnose` → `errors` |
| Edit YAML then `resume` | Config is **cached** in state file | Use `--reload-config` or `modify -r` |

---

## Config Caching (Major Footgun!)

**Mozart caches the config snapshot in the state file on first run.** Subsequent `resume` commands use this cached snapshot, NOT the current YAML file.

### When This Bites You

1. You run `mozart run job.yaml` - config is snapshotted to state
2. Sheet fails validation due to a bad command
3. You fix the YAML file
4. You run `mozart resume job-id` - **uses the OLD cached config**
5. Same validation keeps failing despite your fix

### How to Fix

**Option 1: Use `mozart modify -r` (Recommended for running jobs)**
```bash
# Pause, update config, and resume in one command
mozart modify job-id -c fixed.yaml -r -w ./ws
```

**Option 2: Use `--reload-config` (For already paused jobs)**
```bash
# Reload from the original YAML file
mozart resume job-id -w ./ws --reload-config

# Or reload from a different/fixed YAML file
mozart resume job-id -w ./ws --reload-config --config fixed.yaml
```

**Option 3: Delete state and restart**
```bash
rm workspace/job.json
mozart run job.yaml
```

---

## Common Patterns

### Long-Running Jobs

```bash
setsid mozart --log-file workspace/mozart.log run job.yaml &
mozart status job-id --watch -w ./workspace
```

### Disable MCP for Speed

```yaml
backend:
  disable_mcp: true   # ~2x faster execution
```

### Read-Only Execution

```yaml
backend:
  allowed_tools:
    - Read
    - Grep
    - Glob
    - LS
```

---

## Quick Reference Card

```
COMMANDS                          DEBUGGING ORDER
--------                          ---------------
run <config>     Execute          1. mozart status ...
status <job>     Check status     2. mozart diagnose ...
resume <job>     Continue         3. mozart errors -V
pause <job>      Stop gracefully  4. Manual investigation
modify <job>     Change config
diagnose <job>   Full report      ERROR CATEGORIES
errors <job>     List errors      ----------------
validate <cfg>   Check config     E0xx Execution
list             List all jobs    E1xx Rate limit
logs [job]       View logs        E2xx Validation
dashboard        Web UI           E3xx Config
                                  E4xx State
GLOBAL OPTIONS                    E5xx Backend
--------------                    E6xx Preflight
-v, --verbose    Detailed         E9xx Network
-q, --quiet      Errors only
-L, --log-level  Level filter
--log-file       Log path
--log-format     json/console

COMMON OPTIONS                    NOTIFICATION EVENTS
--------------                    -------------------
-w, --workspace  Directory        job_start, job_complete
-j, --json       JSON output      job_failed, job_paused
-V, --verbose    Details          sheet_start, sheet_complete
                                  sheet_failed
```

---

*Mozart AI Compose - Updated 2026-02-09 with fan-out parameterized stage instantiation*
