# Mozart AI Compose - Implementation Plan

**Created:** 2025-12-18 (Planning session)
**Status:** Phase 1 Complete, Phases 2-4 Pending

---

## Overview

Python orchestration tool for running multiple Claude sessions with configurable prompts. Generalizes patterns from `run-batch-review.sh` into a reusable framework.

## User Requirements

- **Config**: YAML files for job definitions
- **Backends**: Claude CLI + Anthropic API support
- **Observability**: Full web dashboard

---

## Project Structure

```
~/Projects/mozart-ai-compose/
├── pyproject.toml
├── README.md
├── src/mozart/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                    # Typer CLI
│   ├── core/
│   │   ├── config.py             # Pydantic models + YAML loading
│   │   ├── job.py                # Job domain model
│   │   ├── checkpoint.py         # State management
│   │   └── errors.py             # Error classification
│   ├── backends/
│   │   ├── base.py               # Protocol
│   │   ├── claude_cli.py         # subprocess wrapper
│   │   └── anthropic_api.py      # async httpx
│   ├── execution/
│   │   ├── runner.py             # Main orchestration loop
│   │   ├── validation.py         # Output validation
│   │   └── retry.py              # Retry strategies
│   ├── prompts/
│   │   └── templating.py         # Jinja2 engine
│   ├── notifications/
│   │   ├── desktop.py
│   │   ├── webhook.py
│   │   └── slack.py
│   ├── state/
│   │   ├── json_backend.py
│   │   └── sqlite_backend.py
│   └── dashboard/
│       ├── app.py                # FastAPI
│       ├── routes/
│       ├── models.py
│       └── database.py
├── tests/
└── examples/
    ├── batch-review.yaml
    └── templates/
```

---

## Key Abstractions

### 1. JobConfig (Pydantic)
```python
class JobConfig(BaseModel):
    name: str
    workspace: Path
    backend: BackendConfig       # cli or api
    batch: BatchConfig           # size, total_items
    prompt: PromptConfig         # template, variables, stakes
    retry: RetryConfig           # max_retries, backoff
    rate_limit: RateLimitConfig  # patterns, wait_minutes
    validations: list[ValidationRule]
    notifications: list[NotificationConfig]
```

### 2. Backend Protocol
```python
class Backend(Protocol):
    async def execute(prompt: str) -> ExecutionResult
    async def health_check() -> bool
```

### 3. State Backend Protocol
```python
class StateBackend(Protocol):
    async def load(job_id) -> CheckpointState
    async def save(state) -> None
    async def mark_batch_status(job_id, batch_num, status) -> None
```

### 4. Error Classification
- RATE_LIMIT: wait + retry (no count decrement)
- TRANSIENT: retry with backoff
- VALIDATION: retry (Claude ran but output invalid)
- FATAL: stop job

---

## YAML Job Schema (Example)

```yaml
name: commit-batch-review
backend:
  type: claude_cli
  skip_permissions: true
batch:
  size: 10
  total_items: 552
prompt:
  template_file: templates/review.j2
  stakes: "1T$ tip for success, wolves for failure"
validations:
  - type: file_exists
    path: "{workspace}/batch{batch_num}-security-report.md"
  - type: file_modified
    path: "{workspace}/LONGFORM-REVIEW.md"
notifications:
  - type: desktop
    on_events: [job_complete]
```

---

## Database Schema (SQLite)

- `jobs` - Job definitions from YAML
- `job_runs` - Each invocation with status, progress
- `batches` - Per-batch execution history (stdout, stderr, validations)
- `notifications` - Notification log

---

## Dashboard Pages

1. **Home** - Jobs list with status badges
2. **Job Detail** - Config + run history
3. **Run Detail** - Progress bar, batch timeline, live logs
4. **Batch Detail** - Full output, validation results
5. **History** - Filterable run history

---

## Build Sequence

### Phase 1: Foundation ✅ COMPLETE
1. ✅ Project scaffolding (pyproject.toml, structure)
2. ✅ Config loading (Pydantic + YAML)
3. ✅ JSON state backend
4. ✅ Claude CLI backend

### Phase 2: Execution Engine ⏳ PENDING
5. Validation framework
6. Error classification (done in Phase 1)
7. Retry logic with rate limit handling
8. Main runner loop
9. CLI commands (run, status, resume)

### Phase 3: Extensions ⏳ PENDING
10. Anthropic API backend
11. Notifications (desktop, webhook)
12. SQLite state backend

### Phase 4: Dashboard ⏳ PENDING
13. FastAPI scaffold
14. SQLAlchemy models
15. API endpoints
16. Frontend (Jinja2 templates)

---

## Dependencies

```toml
dependencies = [
    "pydantic>=2.5.0",
    "pyyaml>=6.0",
    "jinja2>=3.1.0",
    "typer[all]>=0.9.0",
    "rich>=13.0.0",
    "httpx>=0.27.0",
    "anthropic>=0.18.0",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "sqlalchemy>=2.0.0",
    "aiosqlite>=0.19.0",
    "plyer>=2.1.0",
]
```

---

## CLI Usage

```bash
mozart run examples/batch-review.yaml
mozart status
mozart resume job-123
mozart dashboard --port 8000
```

---

## Reference Files

- `/home/emzi/Projects/Naurva/run-batch-review.sh` - State, retry, validation patterns
- `/home/emzi/.claude/skills/multi-agent-coordination.md` - Coordination patterns

---

*Original plan created during planning session 2025-12-18*
*Source: /home/emzi/.claude/plans/logical-percolating-bunny.md*
