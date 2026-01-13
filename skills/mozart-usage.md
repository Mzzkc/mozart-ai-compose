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

### Debug Order (MANDATORY)

```bash
mozart status job-id -w ./ws      # 1. Always start here
mozart diagnose job-id -w ./ws    # 2. If failed
mozart errors job-id -V           # 3. Error details
# 4. THEN manual investigation
```

### Critical Rules

| ✗ Never | ✓ Always |
|---------|----------|
| `timeout 600 mozart run ...` | `mozart run ...` (internal timeout) |
| `mozart run job.yaml &` | `nohup mozart run ... &` (detached) |
| Assume exit_code=0 → success | Check `validation_details` |
| Debug manually first | Use Mozart tools first |

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
  type: claude_cli              # claude_cli | anthropic_api
  skip_permissions: true
  timeout_seconds: 1800
  output_format: json
  working_directory: ./code
  cli_extra_args: ["--flag"]
  # API only:
  model: claude-sonnet-4-20250514
  max_tokens: 8192

# === RETRY ===
retry:
  max_retries: 3
  base_delay_seconds: 10
  max_delay_seconds: 3600
  exponential_base: 2.0
  jitter: true
  max_completion_attempts: 3
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
  max_cost_per_job: 100.00

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
    condition: "sheet_num >= 2"

# === POST-SUCCESS HOOKS ===
on_success:
  - type: run_job
    job_path: "{workspace}/next.yaml"

  - type: run_command
    command: "curl -X POST https://api/notify"

# === CONCERT (job chaining) ===
concert:
  enabled: false
  max_chain_depth: 5
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
rm workspace/job-id.json         # JSON backend
# OR: rm workspace/.mozart-state.db  # SQLite backend
mozart run job.yaml --start-sheet N
```

---

## Anti-Patterns

| Don't | Why | Do Instead |
|-------|-----|------------|
| `timeout 600 mozart run ...` | SIGKILL corrupts state | Mozart handles timeout internally |
| `mozart run job.yaml &` | Dies on session end | `nohup mozart run ... > log 2>&1 &` |
| Check `exit_code` for success | Validations may have failed | Check `validation_details` |
| Debug manually first | Mozart tools provide context | `status` → `diagnose` → `errors` |
| `pattern: ".*"` | Matches everything | Specific regex patterns |
| Omit `description` | Hard to debug | Always describe validations |

---

## Common Patterns

### Long-Running Jobs

```bash
nohup mozart run job.yaml > workspace/mozart.log 2>&1 &
mozart status job-id --watch -w ./workspace
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
                                  E0xx Execution
OPTIONS                           E1xx Rate limit
───────                           E2xx Validation
-w, --workspace  Directory        E3xx Config
-j, --json       JSON output      E4xx State
-V, --verbose    Details          E5xx Backend
                                  E9xx Network
```

---

*Mozart AI Compose v0.x*
