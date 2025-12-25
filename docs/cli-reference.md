# Mozart CLI Reference

Complete reference for all Mozart CLI commands and options.

## Global Options

These options work with all commands:

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | `-V` | Show version and exit |
| `--verbose` | `-v` | Enable verbose output |
| `--quiet` | `-q` | Suppress non-essential output |
| `--help` | | Show help message |

## Commands

### `mozart run`

Execute a batch job from a configuration file.

```bash
mozart run <config> [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `config` | Yes | Path to YAML configuration file |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--dry-run` | `-n` | false | Show plan without executing |
| `--workspace` | `-w` | `.` | Workspace directory for state |
| `--json` | `-j` | false | Output results as JSON |
| `--start-batch` | | 1 | Start from specific batch number |

#### Examples

```bash
# Basic run
mozart run job.yaml

# Dry run to preview
mozart run job.yaml --dry-run

# Custom workspace
mozart run job.yaml --workspace ./output

# Start from batch 3
mozart run job.yaml --start-batch 3

# JSON output for scripting
mozart run job.yaml --json
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success or graceful shutdown |
| 1 | Configuration error or job failure |

---

### `mozart status`

Show status of a specific job.

```bash
mozart status <job-id> [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `job-id` | Yes | Job identifier |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--workspace` | `-w` | `.` | Workspace directory |
| `--json` | `-j` | false | Output as JSON |

#### Examples

```bash
# Show job status
mozart status my-batch-job

# JSON output
mozart status my-batch-job --json

# Custom workspace
mozart status my-batch-job --workspace ./jobs
```

#### Output

Standard output includes:
- Job name and ID
- Status (running, completed, failed, paused, pending)
- Progress bar with batch counts
- Timing information
- Error messages (if any)
- Batch details table

JSON output structure:
```json
{
  "job_id": "my-batch-job",
  "job_name": "My Batch Job",
  "status": "running",
  "progress": {
    "completed": 5,
    "total": 10,
    "percent": 50.0
  },
  "batches": {
    "1": {"status": "completed", "attempt_count": 1},
    "2": {"status": "in_progress", "attempt_count": 1}
  }
}
```

---

### `mozart resume`

Resume a paused or failed job.

```bash
mozart resume <job-id> [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `job-id` | Yes | Job identifier to resume |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--workspace` | `-w` | `.` | Workspace directory |
| `--config` | `-c` | | Override config file path |
| `--force` | `-f` | false | Force resume completed jobs |

#### Examples

```bash
# Resume paused job
mozart resume my-batch-job

# Resume with explicit config
mozart resume my-batch-job --config job.yaml

# Force restart completed job
mozart resume my-batch-job --force
```

#### Resumable States

| Status | Resumable | Notes |
|--------|-----------|-------|
| `paused` | Yes | Continues from last batch |
| `failed` | Yes | Retries from failed batch |
| `running` | Yes | Continues from last batch |
| `completed` | With `--force` | Restarts entire job |
| `pending` | No | Use `run` instead |

---

### `mozart list`

List all jobs in workspace.

```bash
mozart list [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--workspace` | `-w` | `.` | Workspace directory |
| `--status` | `-s` | | Filter by status |
| `--limit` | `-l` | 50 | Maximum jobs to show |
| `--json` | `-j` | false | Output as JSON |

#### Examples

```bash
# List all jobs
mozart list

# Filter by status
mozart list --status completed
mozart list --status failed
mozart list --status paused

# Limit results
mozart list --limit 10

# JSON output
mozart list --json
```

#### Status Filters

Valid status values:
- `pending` - Not yet started
- `running` - Currently executing
- `paused` - Gracefully interrupted
- `completed` - Successfully finished
- `failed` - Ended with error

---

### `mozart validate`

Validate a job configuration file.

```bash
mozart validate <config> [OPTIONS]
```

#### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `config` | Yes | Path to YAML configuration file |

#### Examples

```bash
# Validate config
mozart validate job.yaml
```

#### Output

On success:
```
Valid configuration: my-batch-job
  Batches: 10 (5 items each)
  Validations: 2
```

On error:
```
Invalid configuration: job.yaml
  Error: batch.size must be positive
```

---

### `mozart dashboard`

Start the web dashboard server.

```bash
mozart dashboard [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--workspace` | `-w` | `.` | Workspace directory |
| `--host` | `-h` | `127.0.0.1` | Host to bind to |
| `--port` | `-p` | `8000` | Port to listen on |

#### Examples

```bash
# Start with defaults
mozart dashboard

# Custom port
mozart dashboard --port 3000

# Bind to all interfaces
mozart dashboard --host 0.0.0.0

# Custom workspace
mozart dashboard --workspace ./jobs
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{id}` | Get job details |
| GET | `/api/jobs/{id}/status` | Get job status (lightweight) |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | OpenAPI schema |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MOZART_WORKSPACE` | Default workspace directory |
| `ANTHROPIC_API_KEY` | API key for Anthropic backend |

## Configuration File Reference

See [Configuration Guide](configuration.md) for full YAML options.

### Minimal Example

```yaml
name: "my-job"
batch:
  size: 10
  total_items: 100
prompt:
  template: "Process batch {{ batch_num }}"
```

### Full Example

```yaml
name: "full-example"
description: "Comprehensive job example"

backend:
  type: claude_cli
  skip_permissions: true
  working_directory: ./workspace

batch:
  size: 10
  total_items: 100

prompt:
  template: |
    Process batch {{ batch_num }} of {{ total_batches }}.
    Items: {{ start_item }} to {{ end_item }}.
    {{ stakes }}
  stakes: "Be thorough!"

retry:
  max_retries: 3
  base_delay_seconds: 30
  exponential_base: 2.0

rate_limit:
  wait_minutes: 60
  max_waits: 24

validations:
  - type: file_exists
    path: "output/batch{{ batch_num }}.md"

notifications:
  - type: desktop
    on_events: [job_complete, job_failed]
```

## Keyboard Shortcuts

During execution:

| Key | Action |
|-----|--------|
| `Ctrl+C` | Graceful shutdown (saves state) |
| `Ctrl+C` (x2) | Force quit |

## Tips

### Verbose Mode

Use `-v` for detailed output:
```bash
mozart -v run job.yaml
```

Shows:
- Backend configuration
- Batch execution details
- Validation results
- Timing information

### Quiet Mode

Use `-q` for minimal output:
```bash
mozart -q run job.yaml
```

Shows only:
- Errors
- Final status

### JSON Output for Scripts

Combine `--json` with `jq` for scripting:
```bash
# Get job status
mozart status my-job --json | jq '.status'

# List failed jobs
mozart list --status failed --json | jq '.jobs[].job_id'
```
