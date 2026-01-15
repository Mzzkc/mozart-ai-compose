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
| E501 | Yes | Connection failed |
| E502 | No | Auth/authorization failed |
| E503 | Yes | Invalid response |
| E504 | Yes | Backend timeout |
| E505 | No | Backend not found (ENOENT) |

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

### Quick Category Reference

```
E0xx Execution    E4xx State
E1xx Rate Limit   E5xx Backend
E2xx Validation   E6xx Preflight
E3xx Config       E9xx Network
```

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
| `mozart run job.yaml &` | Dies on session end | `setsid mozart --log-file ws/mozart.log run job.yaml &` |
| Check `exit_code` for success | Validations may have failed | Check `validation_details` |
| Debug manually first | Mozart tools provide context | `status` → `diagnose` → `errors` |
| `pattern: ".*"` | Matches everything | Specific regex patterns |
| Omit `description` | Hard to debug | Always describe validations |
| `source .venv/bin/activate && ...` in validations | `source` is bash-only, fails on `/bin/sh` | Use `python` directly (venv is on PATH) |
| Edit YAML then `resume` | Config is **cached** in state file | Use `--reload-config` flag |

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

**Option 1: Use `--reload-config` (Recommended)**
```bash
# Reload from the original YAML file
mozart resume job-id -w ./ws --reload-config

# Or reload from a different/fixed YAML file
mozart resume job-id -w ./ws --reload-config --config fixed.yaml
```

This updates the cached `config_snapshot` in the state file with the current YAML contents, then continues execution.

**Option 2: Update config_snapshot manually**
```python
import json, yaml
with open('job.yaml') as f:
    yaml_config = yaml.safe_load(f)
with open('workspace/job.json') as f:
    state = json.load(f)
state['config_snapshot']['validations'] = yaml_config['validations']
with open('workspace/job.json', 'w') as f:
    json.dump(state, f, indent=2)
```

**Option 3: Delete state and restart**
```bash
rm workspace/job.json
mozart run job.yaml
```

**Option 4: Start fresh on specific sheet**
```bash
mozart run job.yaml --start-sheet 5
```

---

## Jinja Template Pitfalls

Mozart uses Jinja2 for template rendering. These are common mistakes that cause template parsing failures:

### Problem: Literal `{{` in Template Content

If your template contains literal `{{` (e.g., in code examples or documentation), Jinja tries to parse it as a variable.

**Error:**
```
TemplateSyntaxError: unexpected char '`' at position X
```

**Wrong:**
```yaml
prompt:
  template: |
    | Example | Description |
    | `{{ foo }}` | Shows a variable |  # FAILS - Jinja tries to parse this
```

**Fix - Use Jinja escaping:**
```yaml
prompt:
  template: |
    | Example | Description |
    | `{{ '{{' }} foo {{ '}}' }}` | Shows a variable |  # Outputs: {{ foo }}
```

**Alternative - Use raw blocks for large sections:**
```yaml
prompt:
  template: |
    {% raw %}
    Here's a Jinja example: {{ variable }}
    And another: {% for item in items %}
    {% endraw %}
```

### Problem: Unclosed Tags

**Wrong:**
```yaml
prompt:
  template: |
    {% if sheet_num == 1 %}
    Do something
    # Missing {% endif %}
```

**Fix:** Always close your control structures:
- `{% if %}` needs `{% endif %}`
- `{% for %}` needs `{% endfor %}`
- `{% block %}` needs `{% endblock %}`

### Problem: Undefined Variables

**Error:**
```
UndefinedError: 'unknown_var' is undefined
```

**Available Variables:**
| Variable | Example Value |
|----------|---------------|
| `sheet_num` | `5` |
| `total_sheets` | `10` |
| `start_item` | `41` |
| `end_item` | `50` |
| `workspace` | `./workspace` |
| `job_name` | `my-job` |

Custom variables must be defined in `prompt.variables`.

### Problem: Whitespace Control

Jinja adds whitespace from control structures. Use `-` to strip whitespace:

**Messy output:**
```yaml
template: |
  {% for i in range(3) %}
  Item {{ i }}
  {% endfor %}
```

**Clean output:**
```yaml
template: |
  {%- for i in range(3) %}
  Item {{ i }}
  {%- endfor %}
```

### Problem: YAML Multiline + Jinja

When using `|` for multiline YAML, watch indentation:

**Wrong:**
```yaml
prompt:
  template: |
{% if condition %}  # No indentation - breaks YAML
content
{% endif %}
```

**Correct:**
```yaml
prompt:
  template: |
    {% if condition %}
    content
    {% endif %}
```

### Quick Reference: Escaping Jinja Syntax

| To Output | Write |
|-----------|-------|
| `{{` | `{{ '{{' }}` |
| `}}` | `{{ '}}' }}` |
| `{%` | `{{ '{%' }}` |
| `%}` | `{{ '%}' }}` |
| `{#` | `{{ '{#' }}` |
| `#}` | `{{ '#}' }}` |

---

## Common Patterns

### Long-Running Jobs

```bash
setsid mozart --log-file workspace/mozart.log run job.yaml &
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

## Self-Improving Jobs: Use What You Build

When running self-improving or evolution jobs (like `mozart-opus-evolution-vN.yaml`), ensure the jobs actually USE the features they implement. This is critical for closing the feedback loop.

### The Problem

Evolution jobs often build infrastructure without enabling it:

| Cycle | Feature Implemented | Enabled in Config? |
|-------|--------------------|--------------------|
| v9 | Grounding hooks | No `grounding:` section |
| v9 | Pattern feedback loop | Learning enabled, but no patterns applied |
| v11 | Escalation learning | Being implemented (can't use yet) |

This creates "infrastructure without usage" - features that could work but aren't tested in production.

### The Fix

When generating the next evolution config (Sheet 8), enable features from N-1 or earlier:

```yaml
# v12 should enable v9's grounding hooks:
grounding:
  enabled: true
  hooks:
    - type: file_checksum
      paths:
        - "{workspace}/*.md"

# v12 should enable v10's pattern learning:
learning:
  enabled: true
  pattern_injection: true  # Actually inject patterns into prompts
```

### Checklist for Evolution Jobs

Before running vN, verify:

1. **Features from v(N-2) are enabled** - Give features one cycle to stabilize
2. **Config sections exist** - Not just prompt instructions, actual runtime config
3. **Validations test the features** - Add validation rules that exercise new features

### Anti-Pattern: Building Without Using

```yaml
# BAD: v12 instructions mention grounding but don't enable it
prompt:
  template: |
    Use the grounding engine to validate...  # Just words

# GOOD: v12 actually configures grounding
grounding:
  enabled: true
  hooks:
    - type: file_checksum
```

---

## Self-Healing & Enhanced Validation

Mozart includes two complementary error recovery systems:

### Enhanced Validation (`mozart validate`)

Pre-execution checks that catch configuration issues before jobs run:

```bash
# Basic validation
mozart validate job.yaml

# JSON output (for CI/CD)
mozart validate job.yaml --json
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | Valid (warnings/info OK) |
| 1 | Invalid (has ERROR-severity issues) |
| 2 | Cannot validate (file not found, YAML broken) |

**Validation Checks:**

| Check ID | Description | Severity | Auto-fixable |
|----------|-------------|----------|--------------|
| V001 | Jinja syntax errors (unclosed blocks, expressions) | ERROR | Manual |
| V002 | Workspace parent directory missing | ERROR | Yes |
| V003 | Template file missing | ERROR | Manual |
| V004 | System prompt file missing | ERROR | Manual |
| V005 | Working directory invalid | ERROR | Yes |
| V007 | Invalid regex patterns | ERROR | Suggested |
| V008 | Missing required validation fields | ERROR | Manual |
| V101 | Undefined Jinja variable (with typo suggestions) | WARNING | Suggested |
| V103 | Very short timeout (<60s) | WARNING | Suggested |
| V104 | Very long timeout (>2h) | INFO | N/A |
| V106 | Empty pattern in validation | WARNING | Manual |
| V107 | Referenced files missing | WARNING | Manual |

**Example Output:**
```
$ mozart validate my-job.yaml

Validating my-job...

✓ YAML syntax valid
✓ Schema validation passed (Pydantic)

Running extended validation checks...

ERRORS (must fix before running):
  ✗ [V001] Line 15: Jinja syntax error - unexpected end of template
         {{ sheet_num of {{ total_sheets }}
         Suggestion: Add closing '}}' to the expression

WARNINGS (may cause issues):
  ! [V101] Undefined variable 'shee_num' in prompt.template
         Suggestion: Did you mean 'sheet_num'?

Summary: 1 error (must fix), 1 warning (should fix)

Validation: FAILED
```

### Self-Healing (`--self-healing`)

Automatic diagnosis and remediation when retries are exhausted:

```bash
# Enable self-healing
mozart run job.yaml --self-healing

# Auto-confirm suggested fixes (non-interactive)
mozart run job.yaml --self-healing --yes

# Works with resume too
mozart resume job-id --self-healing
```

**How It Works:**
1. Normal retry flow happens first (retries must be exhausted)
2. Error context collected (error code, stdout/stderr, config)
3. Diagnosis engine finds applicable remedies (sorted by confidence)
4. **Automatic remedies**: Applied without prompting (low-risk, reversible)
5. **Suggested remedies**: Prompt user unless `--yes` flag
6. **Diagnostic remedies**: Show guidance only (cannot auto-fix)
7. If any remedy succeeds, retry counter resets and sheet re-executes

**Built-in Remedies:**

| Remedy | Category | Triggers On | Action |
|--------|----------|-------------|--------|
| Create workspace | Automatic | E601 - workspace missing | `mkdir` the workspace |
| Create parent dirs | Automatic | E601/E201 - path parent missing | `mkdir -p` parents |
| Fix path separators | Automatic | Backslash paths on Unix | Suggest correction |
| Suggest Jinja fix | Suggested | E304/E305 - template errors | Typo suggestions |
| Diagnose auth errors | Diagnostic | E101/E102/E401 | Troubleshooting guidance |
| Diagnose missing CLI | Diagnostic | CLI not found | Installation guidance |

**Healing Report:**
```
═══════════════════════════════════════════════════════════════════════════
SELF-HEALING REPORT: Sheet 5
═══════════════════════════════════════════════════════════════════════════

Error Diagnosed:
  Code: E601 (PREFLIGHT_PATH_MISSING)
  Message: Workspace directory does not exist: ./my-workspace

Remedies Applied:
  ✓ [AUTO] mkdir ./my-workspace: Created workspace directory

Result: HEALED - Retrying sheet
═══════════════════════════════════════════════════════════════════════════
```

**Configuration (Optional):**
```yaml
# In job config YAML
self_healing:
  enabled: true                    # Enable without CLI flag
  auto_confirm: false              # Equivalent to --yes
  disabled_remedies:               # Skip specific remedies
    - suggest_jinja_fix            # Don't auto-correct typos
  max_healing_attempts: 2          # Limit healing cycles
```

**Key Files:**
| File | Purpose |
|------|---------|
| `src/mozart/validation/` | Enhanced validation module |
| `src/mozart/healing/` | Self-healing module |
| `src/mozart/healing/remedies/` | Individual remedy implementations |
| `tests/test_validation_checks.py` | Validation tests (29 tests) |
| `tests/test_healing.py` | Healing tests (32 tests) |

### When to Use Which

| Scenario | Use |
|----------|-----|
| Before running a new config | `mozart validate job.yaml` |
| CI/CD pipeline check | `mozart validate job.yaml --json` |
| Unattended long-running jobs | `mozart run job.yaml --self-healing --yes` |
| Interactive debugging | `mozart run job.yaml --self-healing` (prompted) |
| Known-good configs | No flags needed |

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
-q, --quiet      Errors only      E6xx Preflight
-L, --log-level  Level filter     E9xx Network
--log-file       Log path
--log-format     json/console

COMMON OPTIONS                    NOTIFICATION EVENTS
──────────────                    ───────────────────
-w, --workspace  Directory        job_start, job_complete
-j, --json       JSON output      job_failed, job_paused
-V, --verbose    Details          sheet_start, sheet_complete
                                  sheet_failed
```

---

*Mozart AI Compose v0.x - Generated from source*
