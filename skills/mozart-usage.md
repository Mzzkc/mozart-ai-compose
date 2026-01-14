# Mozart AI Compose - Usage Skill

> **TL;DR**: Mozart orchestrates Claude prompts across sheets with validation, retry, and state management. Use `mozart status` → `diagnose` → `errors` BEFORE manual investigation.

---

## Triggers

| Use This Skill | Skip This Skill |
|----------------|-----------------|
| Running/debugging Mozart jobs | Single Claude CLI calls |
| Understanding validation failures | Non-orchestrated AI work |
| Resuming interrupted jobs | Unrelated batch tools |
| Writing job configurations | |

---

## Quick Reference

### Commands

| Command | Purpose |
|---------|---------|
| `mozart run config.yaml` | Execute job |
| `mozart status job-id -w ./ws` | Check progress |
| `mozart resume job-id -w ./ws` | Continue from checkpoint |
| `mozart diagnose job-id -w ./ws` | Full diagnostic report |
| `mozart errors job-id -V` | List errors with details |
| `mozart validate config.yaml` | Validate config syntax |
| `mozart list -w ./ws` | List all jobs |
| `mozart logs job-id -w ./ws` | View/tail log files |
| `mozart dashboard -w ./ws` | Start web dashboard |

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
| `mozart run job.yaml &` | `nohup mozart run ... &` (detached) |
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

### `mozart resume`

```bash
mozart resume job-id -w ./ws         # Resume from checkpoint
mozart resume job-id -c new.yaml     # Resume with new config
mozart resume job-id --force         # Force resume completed job
```

| Option | Short | Description |
|--------|-------|-------------|
| `--workspace` | `-w` | Workspace directory |
| `--config` | `-c` | Override config file |
| `--force` | `-f` | Resume even if completed |

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

## Mental Models

### 1. Sheet Model

```
Job: 100 items, sheet_size=10 → 10 sheets
├── Sheet 1: items 1-10    → own prompt, validations, retry budget
├── Sheet 2: items 11-20
└── ...
```

### 2. Validation-First Model

**Exit code ≠ Success**. Only validation pass = success.

```
Execution → Run validations → All pass? → SUCCESS
                           → Any fail? → Retry or FAIL
```

### 3. Retry Hierarchy

```
1. Completion Mode (>50% validations pass)
   └── Send completion prompt, preserve partial work

2. Full Retry (<50% pass or completion exhausted)
   └── Re-execute entire sheet with backoff

3. Rate Limit Wait (separate from retries)
   └── PAUSED state, auto-resume after wait
```

### 4. State Machine

```
Sheet: PENDING → RUNNING → COMPLETED | FAILED | RATE_LIMITED
Job:   PENDING → RUNNING → COMPLETED | FAILED | PAUSED
```

---

## Error Codes

| Code | Category | Retry? | Meaning |
|------|----------|--------|---------|
| E001 | Execution | Yes | Timeout |
| E002 | Execution | Yes | Killed by signal |
| E003 | Execution | No | Crash (segfault) |
| E101 | Rate Limit | Yes | API quota (wait 1hr) |
| E102 | Rate Limit | Yes | CLI throttle (wait 15min) |
| E201 | Validation | Yes | File missing |
| E202 | Validation | Yes | Content mismatch |
| E203 | Validation | Yes | Command failed |
| E301 | Config | No | Invalid config |
| E305 | Config | No | MCP/plugin error |
| E401 | State | No | Checkpoint corruption |
| E502 | Backend | No | Auth failed |
| E505 | Backend | No | Claude CLI not found |
| E9xx | Network | Yes | Connection/DNS/SSL |

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

  # API options (type: anthropic_api)
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY
  max_tokens: 8192
  temperature: 0.7

  # Recursive Light options (type: recursive_light)
  recursive_light:
    endpoint: "http://localhost:8080"
    user_id: "mozart-instance-1"
    timeout: 30.0

# === STATE ===
state_backend: sqlite           # json | sqlite
state_path: ./workspace/.mozart-state.db

# === TIMING ===
pause_between_sheets_seconds: 10

# === RETRY ===
retry:
  max_retries: 3
  base_delay_seconds: 10
  max_delay_seconds: 3600
  exponential_base: 2.0
  jitter: true
  max_completion_attempts: 3
  completion_delay_seconds: 5.0
  completion_threshold_percent: 50

# === RATE LIMIT ===
rate_limit:
  wait_minutes: 60
  max_waits: 24
  detection_patterns: ["rate.?limit", "429"]

# === CIRCUIT BREAKER ===
circuit_breaker:
  enabled: true
  failure_threshold: 5
  recovery_timeout_seconds: 300

# === COST LIMITS ===
cost_limits:
  enabled: false
  max_cost_per_sheet: 5.00
  max_cost_per_job: 100.00
  cost_per_1k_input_tokens: 0.003   # Sonnet default
  cost_per_1k_output_tokens: 0.015  # Sonnet default
  warn_at_percent: 80.0
  # Note: For Opus, use 0.015 input, 0.075 output

# === LOGGING ===
logging:
  level: INFO                   # DEBUG | INFO | WARNING | ERROR
  format: console               # json | console | both
  file_path: ./workspace/logs/mozart.log
  max_file_size_mb: 50
  backup_count: 5
  include_timestamps: true
  include_context: true

# === AI REVIEW ===
ai_review:
  enabled: false
  min_score: 60                 # Below this triggers action
  target_score: 80              # Above this is high quality
  on_low_score: warn            # retry | warn | fail
  max_retry_for_review: 2
  review_prompt_template: null  # Custom review prompt

# === LEARNING ===
learning:
  enabled: true
  outcome_store_type: json      # json | sqlite
  outcome_store_path: null      # Default: workspace/.mozart-outcomes.json
  min_confidence_threshold: 0.3
  high_confidence_threshold: 0.7
  escalation_enabled: false

# === VALIDATIONS ===
validations:
  - type: file_exists
    path: "{workspace}/output-{sheet_num}.md"
    description: "Output created"
    stage: 1

  - type: file_modified
    path: "{workspace}/output.md"
    description: "File was updated"
    stage: 1

  - type: content_contains
    path: "{workspace}/output.md"
    pattern: "COMPLETE"
    stage: 2

  - type: content_regex
    path: "{workspace}/output.md"
    pattern: "Score:\\s*\\d+"
    stage: 2

  - type: command_succeeds
    command: "cargo test"
    working_directory: "./code"
    stage: 3
    condition: "sheet_num >= 2"

# === NOTIFICATIONS ===
notifications:
  - type: desktop               # desktop | slack | webhook | email
    on_events:
      - job_complete
      - job_failed
    config: {}

  - type: slack
    on_events:
      - job_start
      - job_complete
      - job_failed
      - job_paused
    config:
      webhook_url: "https://hooks.slack.com/..."
      channel: "#mozart-jobs"

  - type: webhook
    on_events:
      - sheet_complete
      - sheet_failed
    config:
      url: "https://api.example.com/webhook"
      headers:
        Authorization: "Bearer ${WEBHOOK_TOKEN}"

# === POST-SUCCESS HOOKS ===
on_success:
  - type: run_job
    job_path: "{workspace}/next.yaml"
    job_workspace: "{workspace}/next"
    inherit_learning: true
    description: "Chain to next phase"
    on_failure: continue        # continue | abort
    timeout_seconds: 300

  - type: run_command
    command: "curl -X POST https://api/notify"
    working_directory: "./workspace"
    description: "Notify external system"
    on_failure: continue
    timeout_seconds: 60

  - type: run_script
    command: "./deploy.sh {job_id} {sheet_count}"
    working_directory: "./scripts"
    description: "Run deployment script"
    on_failure: abort
    timeout_seconds: 600

# === CONCERT (job chaining) ===
concert:
  enabled: false
  max_chain_depth: 5
  cooldown_between_jobs_seconds: 30
  inherit_workspace: true
  concert_log_path: ./workspace/concert.log
  abort_concert_on_hook_failure: false
```

### Template Variables

| Variable | Example |
|----------|---------|
| `{{ sheet_num }}` | `5` |
| `{{ total_sheets }}` | `10` |
| `{{ start_item }}` | `41` |
| `{{ end_item }}` | `50` |
| `{{ workspace }}` | `./workspace` |
| `{{ job_name }}` | `my-job` |

---

## Validation Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `file_exists` | File was created | `path` |
| `file_modified` | File mtime changed | `path` |
| `content_contains` | Has literal string | `path`, `pattern` |
| `content_regex` | Matches regex | `path`, `pattern` |
| `command_succeeds` | Exit code 0 | `command`, `working_directory` |

### Staged Validation

```yaml
# Stage 1 runs first; if any fail, higher stages skip
- type: file_exists
  stage: 1              # Basic existence
- type: content_contains
  stage: 2              # Content checks
- type: command_succeeds
  stage: 3              # Build/test
```

### Conditional Validation

```yaml
- type: file_exists
  condition: "sheet_num == 1"    # Only sheet 1
- type: command_succeeds
  condition: "sheet_num >= 2"    # Sheets 2+
```

---

## Notification Events

| Event | When |
|-------|------|
| `job_start` | Job begins execution |
| `sheet_start` | Sheet begins execution |
| `sheet_complete` | Sheet passes all validations |
| `sheet_failed` | Sheet exhausts retries |
| `job_complete` | All sheets complete |
| `job_failed` | Job cannot continue |
| `job_paused` | Job paused (rate limit, etc.) |

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
mozart resume job-id -c fixed-config.yaml

# 2b. If work incomplete → just resume (Mozart retries)
mozart resume job-id
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
| `mozart run job.yaml &` | Dies on session end | `nohup mozart --log-file ws/mozart.log run job.yaml &` |
| Check `exit_code` for success | Validations may have failed | Check `validation_details` |
| Debug manually first | Mozart tools provide context | `status` → `diagnose` → `errors` |
| `pattern: ".*"` | Matches everything | Specific regex patterns |
| Omit `description` | Hard to debug | Always describe validations |
| `source .venv/bin/activate && ...` in validations | `source` is bash-only, fails on `/bin/sh` | Use `python` directly (venv is on PATH) |

---

## Common Patterns

### Long-Running Jobs

```bash
nohup mozart --log-file workspace/mozart.log run job.yaml &
mozart status job-id --watch -w ./workspace
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

### Disable MCP for Speed

```yaml
backend:
  disable_mcp: true   # ~2x faster execution
```

### Multi-Agent per Sheet

```yaml
prompt:
  template: |
    LAUNCH 3 AGENTS IN PARALLEL:
    1. Security → {workspace}/sheet{sheet_num}-security.md
    2. Architecture → {workspace}/sheet{sheet_num}-arch.md
    3. Quality → {workspace}/sheet{sheet_num}-quality.md

validations:
  - type: file_exists
    path: "{workspace}/sheet{sheet_num}-security.md"
  - type: file_exists
    path: "{workspace}/sheet{sheet_num}-arch.md"
  - type: file_exists
    path: "{workspace}/sheet{sheet_num}-quality.md"
```

### Progressive Build Pipeline

```yaml
validations:
  - { type: command_succeeds, command: "cargo fmt --check", stage: 1 }
  - { type: command_succeeds, command: "cargo build", stage: 2 }
  - { type: command_succeeds, command: "cargo test", stage: 3 }
  - { type: command_succeeds, command: "cargo clippy -- -D warnings", stage: 4 }
```

### Job Chaining (Concert)

```yaml
on_success:
  - type: run_job
    job_path: "{workspace}/next-phase.yaml"

concert:
  enabled: true
  max_chain_depth: 10
```

### AI Code Review

```yaml
ai_review:
  enabled: true
  min_score: 60
  target_score: 80
  on_low_score: retry
  max_retry_for_review: 2
```

### Cost Tracking

```yaml
cost_limits:
  enabled: true
  max_cost_per_job: 50.00
  warn_at_percent: 80.0
```

---

## Architecture

```
CLI (cli.py)
    │
    ▼
Runner (runner.py)
├── Sheet iteration, retry logic, validation orchestration
    │
    ├── Backend (claude_cli.py, anthropic_api.py)
    ├── Validation (validation.py)
    └── State (json_backend.py, sqlite_backend.py)
```

### Key Files

| File | Purpose |
|------|---------|
| `cli.py` | CLI commands |
| `core/config.py` | Pydantic models |
| `core/errors.py` | Error classification |
| `core/checkpoint.py` | State models |
| `execution/runner.py` | Orchestration |
| `execution/validation.py` | Validation engine |
| `backends/claude_cli.py` | Claude CLI backend |
| `backends/anthropic_api.py` | API backend |

### State File Structure

```json
{
  "job_id": "my-job",
  "status": "running",
  "current_sheet": 3,
  "sheets": {
    "1": {
      "status": "completed",
      "attempts": 1,
      "validation_details": [...],
      "stdout_tail": "...",
      "stderr_tail": "..."
    }
  },
  "errors": [...],
  "total_cost_usd": 0.45
}
```

---

## Quick Reference Card

```
COMMANDS                          DEBUGGING ORDER
────────                          ───────────────
run <config>     Execute          1. mozart status ...
status <job>     Check status     2. mozart diagnose ...
resume <job>     Continue         3. mozart errors -V
diagnose <job>   Full report      4. Manual investigation
errors <job>     List errors
validate <cfg>   Check config     ERROR CATEGORIES
list             List all jobs    ────────────────
logs [job]       View logs        E0xx Execution
dashboard        Web UI           E1xx Rate limit
                                  E2xx Validation
GLOBAL OPTIONS                    E3xx Config
──────────────                    E4xx State
-v, --verbose    Detailed         E5xx Backend
-q, --quiet      Errors only      E9xx Network
-L, --log-level  Level filter
--log-file       Log path         NOTIFICATION EVENTS
--log-format     json/console     ───────────────────
                                  job_start, job_complete
COMMON OPTIONS                    job_failed, job_paused
──────────────                    sheet_start, sheet_complete
-w, --workspace  Directory        sheet_failed
-j, --json       JSON output
-V, --verbose    Details
```

---

*Mozart AI Compose v0.x - Generated from source*
