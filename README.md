# Mozart AI Compose

Orchestration tool for running multiple Claude AI sessions with configurable prompts.

## Features

- **YAML Configuration**: Define jobs declaratively with Jinja2 templating
- **Multiple Backends**: Claude CLI or direct Anthropic API
- **Resumable Execution**: Checkpoint-based state management with resume command
- **Smart Retry**: Error classification with rate limit detection
- **Output Validation**: Verify expected files with multiple validation types
- **Progress Tracking**: Real-time progress bar with ETA
- **Graceful Shutdown**: Ctrl+C saves state for later resume
- **Notifications**: Desktop, Slack, webhook support
- **Web Dashboard**: REST API for monitoring and job management

## Installation

```bash
# From source
git clone https://github.com/yourusername/mozart-ai-compose.git
cd mozart-ai-compose
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Quick Start

1. Create a job configuration file:

```yaml
# my-job.yaml
name: "my-batch-job"
description: "Process items in batches"

backend:
  type: claude_cli
  skip_permissions: true

batch:
  size: 10
  total_items: 100

prompt:
  template: |
    Process batch {{ batch_num }} of {{ total_batches }}.
    Items {{ start_item }} to {{ end_item }}.

    {{ stakes }}

  stakes: "Complete all items correctly for a reward!"

retry:
  max_retries: 2

validations:
  - type: file_exists
    path: "output/batch{{ batch_num }}-result.md"
```

2. Validate and run:

```bash
# Validate configuration
mozart validate my-job.yaml

# Preview with dry run
mozart run my-job.yaml --dry-run

# Run the job
mozart run my-job.yaml
```

3. Monitor and manage:

```bash
# Check job status
mozart status my-batch-job

# List all jobs
mozart list

# Resume if interrupted
mozart resume my-batch-job

# Start web dashboard
mozart dashboard
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `mozart run <config>` | Execute a batch job |
| `mozart run <config> -n` | Dry run (show plan) |
| `mozart status <job-id>` | Show job status and progress |
| `mozart resume <job-id>` | Resume paused/failed job |
| `mozart list` | List all jobs |
| `mozart validate <config>` | Validate configuration |
| `mozart logs [job-id]` | View/tail log files |
| `mozart errors [job-id]` | View error history |
| `mozart diagnose <job-id>` | Comprehensive diagnostics |
| `mozart dashboard` | Start web dashboard |

### Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable detailed output |
| `-q, --quiet` | Suppress non-essential output |
| `-V, --version` | Show version |

## Configuration Reference

### Backend Options

```yaml
backend:
  type: claude_cli  # or anthropic_api

  # CLI options
  skip_permissions: true
  output_format: json  # optional
  working_directory: ./workspace

  # API options (when type: anthropic_api)
  model: claude-sonnet-4-20250514
  api_key_env: ANTHROPIC_API_KEY
  max_tokens: 8192
  temperature: 0.7
```

### Retry Configuration

```yaml
retry:
  max_retries: 3
  base_delay_seconds: 10.0
  max_delay_seconds: 3600.0
  exponential_base: 2.0
  jitter: true
```

### Rate Limit Handling

```yaml
rate_limit:
  wait_minutes: 60
  max_waits: 24  # 24 hours max
  detection_patterns:
    - "rate.?limit"
    - "429"
    - "quota"
```

### Validations

```yaml
validations:
  # Check file exists
  - type: file_exists
    path: "{workspace}/batch{batch_num}-output.md"

  # Check file was modified (mtime changed)
  - type: file_modified
    path: "{workspace}/tracking.md"

  # Check file contains pattern
  - type: content_contains
    path: "{workspace}/report.md"
    pattern: "## Summary"

  # Run command and check exit code
  - type: command_succeeds
    command: "python validate.py {batch_num}"
```

### Notifications

```yaml
notifications:
  - type: desktop
    on_events: [job_complete, job_failed]

  - type: slack
    on_events: [batch_failed]
    config:
      webhook_url_env: SLACK_WEBHOOK_URL
      channel: "#alerts"

  - type: webhook
    on_events: [job_complete]
    config:
      url: https://example.com/webhook
```

## Template Variables

Available in prompt templates:

| Variable | Description |
|----------|-------------|
| `batch_num` | Current batch number (1-indexed) |
| `total_batches` | Total number of batches |
| `start_item` | First item number in batch |
| `end_item` | Last item number in batch |
| `workspace` | Workspace directory path |
| `stakes` | Stakes text from config |
| `thinking_method` | Thinking method text from config |
| Any key in `variables` | Custom variables from config |

## Architecture

```
mozart/
├── core/           # Domain models, config, errors
├── backends/       # Claude CLI and API backends
├── execution/      # Runner, validation, retry logic
├── state/          # JSON and SQLite state backends
├── prompts/        # Jinja2 templating
├── notifications/  # Desktop, Slack, webhook
├── dashboard/      # FastAPI web interface
└── learning/       # Outcome tracking (experimental)
```

## Observability

Mozart includes comprehensive observability features for debugging and monitoring batch jobs.

### Structured Logging

Configure structured JSON logging with automatic rotation:

```yaml
logging:
  level: DEBUG        # DEBUG, INFO, WARNING, ERROR
  format: json        # json, console, or both
  file_path: logs/mozart.log
  max_file_size_mb: 50
  backup_count: 5
  include_timestamps: true
  include_context: true  # Adds job_id, run_id, batch_num to logs
```

View logs with built-in CLI:

```bash
# Show recent logs
mozart logs

# Follow logs in real-time
mozart logs -f

# Filter by level or job
mozart logs --level ERROR
mozart logs my-job-id

# Output as JSON for processing
mozart logs --json | jq '.event'
```

### Error Codes Reference

Mozart uses structured error codes for classification:

| Code | Category | Description | Retriable |
|------|----------|-------------|-----------|
| E001 | TRANSIENT | Generic transient error | Yes |
| E002 | TRANSIENT | Connection/network error | Yes |
| E003 | TRANSIENT | Temporary unavailable | Yes |
| E101 | RATE_LIMIT | API rate limit hit | Yes |
| E102 | RATE_LIMIT | Quota exceeded | Yes |
| E201 | VALIDATION | Output validation failed | No |
| E301 | TIMEOUT | Execution timed out | Yes |
| E302 | TIMEOUT | Process killed (timeout) | Yes |
| E401 | SIGNAL | Process killed by signal | No |
| E402 | SIGNAL | Segmentation fault | No |
| E501 | FATAL | Authentication failed | No |
| E502 | FATAL | Configuration error | No |
| E999 | FATAL | Unknown/unclassified error | No |

### Diagnostics Command

Get comprehensive diagnostic information for troubleshooting:

```bash
# Run diagnostics on a job
mozart diagnose my-job-id

# Include raw output capture
mozart diagnose my-job-id --include-output

# Export as JSON for analysis
mozart diagnose my-job-id --json > diagnostics.json
```

The diagnose command shows:
- Job status and progress
- Error history with codes and messages
- Batch execution times and retry counts
- Raw stdout/stderr capture (last 10KB per batch)
- Prompt metrics (token estimates, warnings)
- Circuit breaker status

### Error History

View error history for debugging patterns:

```bash
# Show recent errors across all jobs
mozart errors

# Filter by job
mozart errors my-job-id

# Filter by error code
mozart errors --code E101

# Include raw output
mozart errors --verbose
```

### Circuit Breaker

Mozart includes a circuit breaker to prevent cascading failures:

- Opens after consecutive failures (default: 5)
- Prevents new batch execution while open
- Automatically recovers after timeout (default: 5 minutes)
- Stats available in `mozart diagnose` output

### Preflight Checks

Before each batch, Mozart runs preflight checks:

- Validates working directory exists
- Estimates prompt token count
- Warns if prompt exceeds thresholds:
  - Warning: >50K tokens
  - Error: >150K tokens
- Checks referenced file paths exist
- Results stored in batch state for diagnostics

### Output Capture

Raw stdout/stderr is captured for debugging:

- Last 10KB of each stream preserved
- Available in checkpoint state
- Shown in `mozart diagnose --include-output`
- Useful for debugging failed batches

## Dashboard API

Start the dashboard:

```bash
mozart dashboard --port 8000
```

Available endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/jobs` | List jobs |
| `GET /api/jobs/{id}` | Job details |
| `GET /api/jobs/{id}/status` | Job status (lightweight) |
| `GET /docs` | Swagger UI |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [CLI Reference](docs/cli-reference.md)
- [Changelog](CHANGELOG.md)

## License

MIT
