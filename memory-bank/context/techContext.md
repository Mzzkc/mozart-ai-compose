# Mozart AI Compose - Technical Context

Architecture and implementation details for developers.

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   CLI Layer (Typer)                     │
│   mozart run|validate|status|dashboard  │
├─────────────────────────────────────────┤
│   Orchestration Layer                   │
│   Runner → Validation → Retry           │
├─────────────────────────────────────────┤
│   Backend Layer (Protocol-based)        │
│   ClaudeCliBackend | AnthropicApiBackend│
├─────────────────────────────────────────┤
│   State Layer (Protocol-based)          │
│   JsonStateBackend | SqliteStateBackend │
└─────────────────────────────────────────┘
```

---

## Core Abstractions

### JobConfig (src/mozart/core/config.py)

Root configuration loaded from YAML. Contains:
- `name`, `description`, `workspace`
- `backend: BackendConfig` - CLI or API settings
- `batch: BatchConfig` - size, total_items, start_item
- `prompt: PromptConfig` - template, variables, stakes
- `retry: RetryConfig` - max_retries, backoff settings
- `rate_limit: RateLimitConfig` - patterns, wait times
- `validations: list[ValidationRule]`
- `notifications: list[NotificationConfig]`

### CheckpointState (src/mozart/core/checkpoint.py)

Persisted state for resumable execution:
- `job_id`, `job_name`
- `total_batches`, `last_completed_batch`, `current_batch`
- `status: JobStatus` (pending|running|completed|failed|paused)
- `batches: dict[int, BatchState]` - Per-batch status
- Methods: `get_next_batch()`, `mark_batch_*()`, `get_progress()`

### Backend Protocol (src/mozart/backends/base.py)

```python
class Backend(Protocol):
    async def execute(prompt: str) -> ExecutionResult
    async def health_check() -> bool
```

ExecutionResult contains: success, exit_code, stdout, stderr, duration_seconds, rate_limited

### StateBackend Protocol (src/mozart/state/base.py)

```python
class StateBackend(Protocol):
    async def load(job_id) -> Optional[CheckpointState]
    async def save(state) -> None
    async def list_jobs() -> list[CheckpointState]
    async def mark_batch_status(job_id, batch_num, status)
```

---

## Error Classification (src/mozart/core/errors.py)

Categories determine retry behavior:

| Category | Retriable | Behavior |
|----------|-----------|----------|
| RATE_LIMIT | Yes | Long wait (1hr), health check before retry |
| TRANSIENT | Yes | Short backoff, normal retry |
| VALIDATION | Yes | Claude ran but output invalid, retry |
| NETWORK | Yes | Connection issues, backoff retry |
| TIMEOUT | Yes | Execution timed out, retry |
| AUTH | No | API key/permission issue, fatal |
| FATAL | No | Unknown error, stop job |

Pattern matching on stdout/stderr for classification.

---

## Prompt Templating

Jinja2 templates with these variables:
- `batch_num` - Current batch (1-indexed)
- `total_batches` - Total batch count
- `start_item`, `end_item` - Item range for batch
- `workspace` - Workspace directory path
- `stakes` - Stakes text from config
- `thinking_method` - Thinking method from config
- Any key in `prompt.variables`

---

## Validation Types

| Type | Check |
|------|-------|
| file_exists | Path exists after batch |
| file_modified | mtime changed (before/after comparison) |
| content_contains | File contains pattern |
| content_regex | File matches regex |

Path supports templating: `{workspace}/batch{batch_num}-report.md`

---

## State File Format (JSON)

```json
{
  "job_id": "commit-batch-review",
  "job_name": "commit-batch-review",
  "created_at": "2025-12-18T12:00:00",
  "updated_at": "2025-12-18T12:30:00",
  "total_batches": 56,
  "last_completed_batch": 5,
  "current_batch": null,
  "status": "running",
  "batches": {
    "1": {"batch_num": 1, "status": "completed", ...},
    "2": {"batch_num": 2, "status": "completed", ...}
  }
}
```

---

## Dependencies

**Core:**
- pydantic>=2.5.0 - Validation, serialization
- pyyaml>=6.0 - Config loading
- jinja2>=3.1.0 - Prompt templating

**CLI:**
- typer[all]>=0.9.0 - CLI framework
- rich>=13.0.0 - Terminal output

**HTTP:**
- httpx>=0.27.0 - Async HTTP client
- anthropic>=0.18.0 - Anthropic SDK

**Dashboard:**
- fastapi>=0.109.0 - Web framework
- uvicorn>=0.27.0 - ASGI server
- sqlalchemy>=2.0.0 - ORM
- aiosqlite>=0.19.0 - Async SQLite

**Notifications:**
- plyer>=2.1.0 - Desktop notifications

---

## Security Notes

**Subprocess Safety:** Claude CLI backend uses `asyncio.create_subprocess_exec` (not shell=True). Arguments passed as list, no shell injection risk.

**Secrets:** API keys via environment variables (api_key_env config). Never stored in config files.

**Rate Limit Patterns:** Configurable patterns for detection. Defaults cover common cases.

---

*Last Updated: 2025-12-18*
