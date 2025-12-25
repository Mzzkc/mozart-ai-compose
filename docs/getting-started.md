# Getting Started with Mozart AI Compose

Mozart AI Compose is an orchestration tool for running multiple Claude AI sessions with configurable prompts. It handles batch processing, retries, state management, and more.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/mozart-ai-compose.git
cd mozart-ai-compose

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Mozart
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### Verify Installation

```bash
mozart --version
# Mozart AI Compose v0.1.0
```

## Prerequisites

Mozart uses Claude CLI as its default backend. Ensure you have:

1. **Claude CLI installed**: Follow [Claude CLI installation guide](https://docs.anthropic.com/claude-code)
2. **API access configured**: Claude CLI should be authenticated

## Your First Job

### Step 1: Create a Configuration File

Create a file called `my-first-job.yaml`:

```yaml
# my-first-job.yaml
name: "my-first-batch"
description: "Process files in batches of 10"

backend:
  type: claude_cli
  skip_permissions: true

batch:
  size: 10
  total_items: 30

prompt:
  template: |
    Process batch {{ batch_num }} of {{ total_batches }}.

    Items in this batch: {{ start_item }} to {{ end_item }}.

    Please:
    1. Review the items
    2. Generate a summary
    3. Save to output/batch{{ batch_num }}.md

    {{ stakes }}

  stakes: "Take your time and be thorough!"

retry:
  max_retries: 2

validations:
  - type: file_exists
    path: "output/batch{{ batch_num }}.md"
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
  Batches: 3 (10 items each)
  Validations: 1
```

### Step 3: Dry Run

Preview what Mozart will do without executing:

```bash
mozart run my-first-job.yaml --dry-run
```

This shows:
- Job configuration summary
- Batch plan with item ranges
- Prompt that will be sent to Claude

### Step 4: Run the Job

Execute the job:

```bash
mozart run my-first-job.yaml
```

You'll see:
- Progress bar with ETA
- Current batch status
- Validation results

### Step 5: Check Status

While running (or after), check job status:

```bash
# Show all jobs
mozart list

# Show specific job details
mozart status my-first-batch
```

## Handling Interruptions

### Graceful Shutdown

Press `Ctrl+C` during execution for a graceful shutdown:

```
Ctrl+C received. Finishing current batch and saving state...

State saved. Job paused at batch 2/3.

To resume: mozart resume my-first-batch
```

### Resuming Jobs

Resume a paused or failed job:

```bash
mozart resume my-first-batch
```

Mozart continues from where it left off.

## Common Patterns

### Pattern 1: Code Review Batches

```yaml
name: "code-review"
description: "Review PRs in batches"

batch:
  size: 5
  total_items: 25

prompt:
  template: |
    Review PRs {{ start_item }} to {{ end_item }}.

    For each PR:
    - Check code quality
    - Identify potential bugs
    - Suggest improvements

    Save results to reviews/batch{{ batch_num }}.md

validations:
  - type: file_exists
    path: "reviews/batch{{ batch_num }}.md"
```

### Pattern 2: Documentation Generation

```yaml
name: "generate-docs"
description: "Generate API documentation"

batch:
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

    Output: docs/api-{{ batch_num }}.md

validations:
  - type: file_exists
    path: "docs/api-{{ batch_num }}.md"
  - type: content_contains
    path: "docs/api-{{ batch_num }}.md"
    pattern: "## Parameters"
```

### Pattern 3: Data Processing with Retries

```yaml
name: "process-data"
description: "Process data with robust error handling"

batch:
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
    Process data batch {{ batch_num }}.
    Items: {{ start_item }} to {{ end_item }}.

validations:
  - type: file_modified
    path: "data/processed.json"
```

## Template Variables

Available in prompts:

| Variable | Example | Description |
|----------|---------|-------------|
| `batch_num` | `1` | Current batch (1-indexed) |
| `total_batches` | `3` | Total number of batches |
| `start_item` | `1` | First item in batch |
| `end_item` | `10` | Last item in batch |
| `workspace` | `./workspace` | Workspace path |
| `stakes` | `"Be careful!"` | Custom stakes text |
| `thinking_method` | `"Think step by step"` | Thinking guidance |

## Monitoring with Dashboard

Start the web dashboard for real-time monitoring:

```bash
mozart dashboard
```

Access at `http://localhost:8000`:
- View all jobs
- Check progress
- See batch details
- API docs at `/docs`

Custom port:
```bash
mozart dashboard --port 3000
```

## Next Steps

- [CLI Reference](cli-reference.md) - All commands and options
- [Configuration Guide](configuration.md) - Full config options
- [Best Practices](best-practices.md) - Tips for production use

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
