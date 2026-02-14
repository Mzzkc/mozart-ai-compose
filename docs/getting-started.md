# Getting Started with Mozart AI Compose

Mozart AI Compose is an orchestration tool for running multiple AI sessions with configurable prompts. It handles sheet-based execution, retries, state management, and more.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/Mzzkc/mozart-ai-compose.git
cd mozart-ai-compose

# Install with daemon support (required for job execution)
./setup.sh --daemon

# Or manually:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[daemon]"
```

### With Development Dependencies

```bash
./setup.sh --dev --daemon
# Or: pip install -e ".[dev,daemon]"
```

### Verify Installation

```bash
mozart --version
```

## Prerequisites

Mozart uses Claude CLI as its default backend. Ensure you have:

1. **Claude CLI installed**: Follow [Claude CLI installation guide](https://docs.anthropic.com/claude-code)
2. **API access configured**: Claude CLI should be authenticated
3. **Daemon support installed**: `pip install -e ".[daemon]"` or `./setup.sh --daemon`

## Your First Job

### Step 1: Create a Configuration File

Create a file called `my-first-job.yaml`:

```yaml
# my-first-job.yaml
name: "my-first-job"
description: "Process files in sheets of 10"

backend:
  type: claude_cli
  skip_permissions: true

sheet:
  size: 10
  total_items: 30

prompt:
  template: |
    Process sheet {{ sheet_num }} of {{ total_sheets }}.

    Items in this sheet: {{ start_item }} to {{ end_item }}.

    Please:
    1. Review the items
    2. Generate a summary
    3. Save to output/sheet{{ sheet_num }}.md

    {{ stakes }}

  stakes: "Take your time and be thorough!"

retry:
  max_retries: 2

validations:
  - type: file_exists
    path: "output/sheet{{ sheet_num }}.md"
    description: "Output file created"
```

### Step 2: Validate Your Configuration

Before running, validate the configuration:

```bash
mozart validate my-first-job.yaml
```

You should see:
```
Valid configuration: my-first-job
  Sheets: 3 (10 items each)
  Validations: 1
```

### Step 3: Dry Run

Preview what Mozart will do without executing:

```bash
mozart run my-first-job.yaml --dry-run
```

This shows:
- Job configuration summary
- Sheet plan with item ranges
- Prompt that will be sent to the backend

### Step 4: Start the Daemon

The Mozart daemon is required for job execution:

```bash
mozartd start
mozartd status   # Verify it's running
```

### Step 5: Run the Job

Execute the job:

```bash
mozart run my-first-job.yaml
```

You'll see:
- Progress bar with ETA
- Current sheet status
- Validation results

### Step 6: Check Status

While running (or after), check job status:

```bash
# Show specific job details
mozart status my-first-job

# List all active daemon jobs
mozart list
```

## Handling Interruptions

### Graceful Shutdown

Press `Ctrl+C` during execution for a graceful shutdown:

```
Ctrl+C received. Finishing current sheet and saving state...

State saved. Job paused at sheet 2/3.

To resume: mozart resume my-first-job
```

### Resuming Jobs

Resume a paused or failed job:

```bash
mozart resume my-first-job
```

Mozart continues from where it left off.

## Common Patterns

### Pattern 1: Code Review Sheets

```yaml
name: "code-review"
description: "Review PRs in sheets"

sheet:
  size: 5
  total_items: 25

prompt:
  template: |
    Review PRs {{ start_item }} to {{ end_item }}.

    For each PR:
    - Check code quality
    - Identify potential bugs
    - Suggest improvements

    Save results to reviews/sheet{{ sheet_num }}.md

validations:
  - type: file_exists
    path: "reviews/sheet{{ sheet_num }}.md"
```

### Pattern 2: Documentation Generation

```yaml
name: "generate-docs"
description: "Generate API documentation"

sheet:
  size: 10
  total_items: 50

prompt:
  template: |
    Document functions {{ start_item }} to {{ end_item }}.

    For each function, generate:
    - Description
    - Parameters
    - Return values
    - Examples

    Output: docs/api-{{ sheet_num }}.md

validations:
  - type: file_exists
    path: "docs/api-{{ sheet_num }}.md"
  - type: content_contains
    path: "docs/api-{{ sheet_num }}.md"
    pattern: "## Parameters"
```

### Pattern 3: Data Processing with Retries

```yaml
name: "process-data"
description: "Process data with robust error handling"

sheet:
  size: 20
  total_items: 100

retry:
  max_retries: 3
  base_delay_seconds: 30
  exponential_base: 2.0

rate_limit:
  wait_minutes: 15
  max_waits: 10

prompt:
  template: |
    Process data sheet {{ sheet_num }}.
    Items: {{ start_item }} to {{ end_item }}.

validations:
  - type: file_modified
    path: "data/processed.json"
```

## Template Variables

Available in prompts:

| Variable | Example | Description |
|----------|---------|-------------|
| `sheet_num` | `1` | Current sheet (1-indexed) |
| `total_sheets` | `3` | Total number of sheets |
| `start_item` | `1` | First item in sheet |
| `end_item` | `10` | Last item in sheet |
| `workspace` | `./workspace` | Workspace path |
| `stakes` | `"Be careful!"` | Custom stakes text |
| `thinking_method` | `"Think step by step"` | Thinking guidance |
| `stage` | `2` | Logical stage number (with fan-out) |
| `instance` | `1` | Instance within fan-out group |
| `fan_count` | `3` | Total instances in stage |
| `total_stages` | `7` | Original stage count |

## Monitoring with Dashboard

Start the web dashboard for real-time monitoring:

```bash
mozart dashboard
```

Access at `http://localhost:8000`:
- View all jobs
- Check progress
- See sheet details
- API docs at `/docs`

Custom port:
```bash
mozart dashboard --port 3000
```

## Next Steps

- [CLI Reference](cli-reference.md) - All commands and options
- [Daemon Guide](daemon-guide.md) - Daemon configuration, systemd integration, and troubleshooting
- [Score Writing Guide](score-writing-guide.md) - How to author Mozart scores
- [Configuration Reference](configuration-reference.md) - Every config field documented
- See `examples/` directory for configuration examples

## Troubleshooting

### Job Won't Start

1. Check config: `mozart validate config.yaml`
2. Verify Claude CLI: `claude --version`
3. Check workspace exists: `ls -la ./workspace`

### Validation Failing

1. Check file paths are correct
2. Verify template variables expand properly
3. Use `--dry-run` to see generated prompts

### Rate Limits

Mozart detects rate limits and waits automatically. Configure wait times:

```yaml
rate_limit:
  wait_minutes: 60
  max_waits: 24
```

### Resume Not Working

1. Check job state: `mozart status <job-id>`
2. Ensure config is available (stored in state or via `--config`)
3. Use `--force` to restart completed jobs
