# Mozart AI Compose

**Cognitive orchestration for AI-assisted work**

Mozart orchestrates multi-phase AI workflows with checkpointing, validation gates, and automatic recovery. Whether you're processing code reviews, writing documentation, curating training data, or conducting systematic research, Mozart ensures work completes reliably and correctly.

## Table of Contents

- [What is Mozart?](#what-is-mozart)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Examples](#examples)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## What is Mozart?

Mozart is a general-purpose orchestration system for AI-assisted workflows. It divides complex work into **sheets** (discrete execution units), each with its own prompt template, validation rules, and retry budget. Mozart handles the reliability concerns so you can focus on the work itself.

### Key Capabilities

- **Sheet-based execution**: Divide work into atomic units with independent retry and validation
- **Validation-first**: Exit code 0 does not mean success. Only validation pass means success
- **Automatic recovery**: Checkpoint state enables resume after interruption, rate limits, or failures
- **Self-healing**: Diagnose and fix common issues automatically when retries are exhausted
- **Learning system**: Record outcomes and detect patterns across executions
- **Multiple backends**: Claude CLI, Anthropic API, or Recursive Light

### When to Use Mozart

Mozart is ideal for:

- **Multi-phase workflows**: Code review across many files, batch documentation generation
- **Tasks requiring validation**: Work that must meet quantitative quality gates
- **Long-running operations**: Jobs that may be interrupted and need reliable resume
- **Parallel execution**: Independent work streams that can run concurrently
- **Cross-project learning**: Accumulating patterns across multiple job executions

Mozart is NOT for:

- Single-shot prompts (use Claude CLI directly)
- Interactive conversations (use Claude Code)
- Real-time chat applications (use the API directly)

## Installation

### Prerequisites

- Python 3.11+
- Claude CLI installed and authenticated (for `claude_cli` backend)

### Install from Source

```bash
git clone https://github.com/emzi/mozart-ai-compose.git
cd mozart-ai-compose
pip install -e "."
```

### Verify Installation

```bash
mozart --version
```

## Quick Start

### 1. Create a Configuration

Create a file named `hello-world.yaml`:

```yaml
name: hello-world
description: Simple demonstration of Mozart execution
workspace: ./workspace/hello-world

backend:
  type: claude_cli
  skip_permissions: true
  timeout_seconds: 120

sheet:
  size: 1
  total_items: 3

prompt:
  template: |
    You are executing sheet {{ sheet_num }} of {{ total_sheets }}.

    Task: Write a haiku about the number {{ sheet_num }}.

    Save your haiku to: {{ workspace }}/haiku-{{ sheet_num }}.txt

validations:
  - type: file_exists
    path: "{workspace}/haiku-{sheet_num}.txt"
    description: "Haiku file must exist"
```

### 2. Validate Configuration

```bash
mozart validate hello-world.yaml
```

Expected output:

```
Configuration valid: hello-world.yaml
  3 sheets will be executed
  Workspace: ./workspace/hello-world
```

### 3. Run the Job

```bash
mozart run hello-world.yaml
```

Mozart executes each sheet sequentially, validating output before proceeding to the next.

### 4. Monitor Progress

While the job runs (or after), check status:

```bash
mozart status hello-world --workspace ./workspace/hello-world
```

### 5. Resume if Interrupted

If the job is interrupted (Ctrl+C, rate limit, error), resume from where it left off:

```bash
mozart resume hello-world --workspace ./workspace/hello-world
```

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| Sheet-based execution | Divide work into atomic sheets with independent prompts and validation |
| Validation system | Five validation types: file_exists, file_modified, content_contains, content_regex, command_succeeds |
| Retry logic | Exponential backoff with jitter, completion mode for partial success |
| State management | Atomic checkpoint saves with JSON or SQLite backends |
| Rate limit handling | Automatic detection, wait, and resume |
| Graceful shutdown | Ctrl+C saves state for later resume |

### Advanced Features

| Feature | Description |
|---------|-------------|
| Self-healing | Automatic diagnosis and remediation when retries exhausted (`--self-healing`) |
| Learning system | Outcome recording, pattern detection, cross-workspace learning |
| Parallel execution | DAG-based sheet dependencies for concurrent execution |
| Web dashboard | Real-time monitoring with job control and log streaming |
| Job chaining | Chain jobs via on_success hooks with depth limits |
| Worktree isolation | Git worktree isolation for parallel-safe execution |
| Cost tracking | Per-sheet and per-job cost limits |
| Circuit breaker | Cross-workspace coordination, rate limit sharing |
| Human-in-the-loop | Escalation for low-confidence decisions (`--escalation`) |

## CLI Reference

### Core Commands

| Command | Purpose |
|---------|---------|
| `mozart run <config>` | Execute a job from YAML configuration |
| `mozart resume <job-id>` | Resume a paused or failed job |
| `mozart status [job-id]` | Show job status and progress |
| `mozart validate <config>` | Validate configuration file |
| `mozart list` | List all jobs in workspace |

### Diagnostic Commands

| Command | Purpose |
|---------|---------|
| `mozart logs <job-id>` | View or tail log files |
| `mozart errors <job-id>` | List job errors with details |
| `mozart diagnose <job-id>` | Comprehensive diagnostic report |

### Learning Commands

| Command | Purpose |
|---------|---------|
| `mozart patterns` | View learning patterns |
| `mozart learning-stats` | Learning system statistics |
| `mozart learning-insights` | Actionable insights from patterns |
| `mozart learning-drift` | Detect pattern effectiveness drift |

Additional learning commands are available for pattern management (`pattern-quarantine`, `pattern-validate`, `pattern-show`, `aggregate-patterns`, `recalculate-trust`) and analysis (`patterns-entropy`, `learning-epistemic-drift`, `learning-activity`). Run `mozart --help` for the complete list.

### Dashboard

```bash
mozart dashboard --port 8000
```

Starts the web dashboard for visual monitoring and control.

### Common Options

| Option | Description |
|--------|-------------|
| `--workspace <path>` | Workspace directory for job artifacts |
| `--dry-run` | Validate and show execution plan without running |
| `--self-healing` | Enable automatic diagnosis and remediation |
| `--json` | Output in JSON format |
| `-v, --verbose` | Detailed output |

## Configuration

Mozart jobs are configured with YAML files. Here is a complete example demonstrating key options:

```yaml
name: code-review-batch
description: Review multiple pull requests with validation
workspace: ./workspace/code-review

backend:
  type: claude_cli
  skip_permissions: true
  disable_mcp: true
  timeout_seconds: 1800
  allowed_tools: [Read, Grep, Glob, Bash]

sheet:
  size: 5
  total_items: 50
  start_item: 1

prompt:
  template_file: ./prompts/review.j2
  variables:
    repository: my-project
    review_type: security

retry:
  max_retries: 3
  base_delay_seconds: 10.0
  exponential_base: 2.0
  jitter: true

validations:
  - type: file_exists
    path: "{workspace}/review-{sheet_num}.md"
    description: "Review report must exist"
  - type: content_contains
    path: "{workspace}/review-{sheet_num}.md"
    pattern: "## Summary"
    description: "Report must contain summary section"
  - type: command_succeeds
    command: "grep -c 'CRITICAL\\|HIGH\\|MEDIUM\\|LOW' {workspace}/review-{sheet_num}.md"
    description: "Report must contain severity ratings"

notifications:
  - type: desktop
    on_events: [job_complete, job_failed]
  - type: slack
    on_events: [job_failed]
    config:
      webhook_url: ${SLACK_WEBHOOK_URL}
```

### Configuration Reference

#### Backend Options

| Option | Type | Description |
|--------|------|-------------|
| `type` | string | Backend type: `claude_cli`, `anthropic_api`, or `recursive_light` |
| `skip_permissions` | bool | Skip Claude CLI permission prompts (required for unattended) |
| `disable_mcp` | bool | Disable MCP servers for faster execution |
| `timeout_seconds` | int | Maximum execution time per sheet (default: 1800) |
| `allowed_tools` | list | Restrict available tools |
| `cli_model` | string | Override default model |

#### Validation Types

| Type | Required Fields | Description |
|------|-----------------|-------------|
| `file_exists` | `path` | Check that file exists |
| `file_modified` | `path` | Check that file was modified during execution |
| `content_contains` | `path`, `pattern` | Check file contains literal string |
| `content_regex` | `path`, `pattern` | Check file matches regex pattern |
| `command_succeeds` | `command` | Check command exits with code 0 |

#### Sheet Configuration

| Option | Type | Description |
|--------|------|-------------|
| `size` | int | Items per sheet |
| `total_items` | int | Total items to process |
| `start_item` | int | First item number (1-indexed) |
| `dependencies` | dict | DAG for parallel execution |

## Examples

Mozart includes examples demonstrating various use cases:

### Software Development

| Example | Description |
|---------|-------------|
| [simple-sheet.yaml](examples/simple-sheet.yaml) | Minimal configuration to get started |
| [api-backend.yaml](examples/api-backend.yaml) | Using Anthropic API directly |
| [self-improvement.yaml](examples/self-improvement.yaml) | Incremental codebase improvement with test gates |
| [sheet-review.yaml](examples/sheet-review.yaml) | Multi-agent coordinated code review |
| [worktree-isolation.yaml](examples/worktree-isolation.yaml) | Parallel-safe execution using git worktrees |
| [observability-demo.yaml](examples/observability-demo.yaml) | Demonstration of Mozart's observability features |

### Beyond Coding

| Example | Domain | Description |
|---------|--------|-------------|
| [systematic-literature-review.yaml](examples/systematic-literature-review.yaml) | Research | PRISMA-compliant academic literature review |
| [training-data-curation.yaml](examples/training-data-curation.yaml) | Data | Curate training data with inter-annotator agreement |
| [nonfiction-book.yaml](examples/nonfiction-book.yaml) | Writing | Book manuscript using Snowflake Method |
| [strategic-plan.yaml](examples/strategic-plan.yaml) | Planning | Strategic plan with multi-framework analysis |
| [parallel-research.yaml](examples/parallel-research.yaml) | Research | Parallel source collection with synthesis |

See [examples/README.md](examples/README.md) for detailed documentation of each example.

## Architecture

```
                              +-------------------+
                              |   YAML Config     |
                              +--------+----------+
                                       |
                                       v
+------------------+          +--------+----------+          +------------------+
|   CLI (Typer)    +--------->|  Execution Runner +--------->|  Backend         |
+------------------+          +--------+----------+          |  (Claude CLI /   |
                                       |                     |   Anthropic API) |
                                       v                     +------------------+
                              +--------+----------+
                              |  State Manager    |
                              |  (JSON/SQLite)    |
                              +--------+----------+
                                       |
                                       v
                              +--------+----------+
                              |  Validation       |
                              |  (5 types)        |
                              +--------+----------+
                                       |
                                       v
                              +--------+----------+
                              |  Learning System  |
                              |  (Patterns/Store) |
                              +-------------------+
```

### Key Concepts

- **Sheet**: An atomic unit of work with its own prompt, execution, validation, and retry budget. Sheets are numbered sequentially and can define dependencies for parallel execution.

- **Validation**: Rules that determine whether a sheet completed successfully. Mozart's validation-first philosophy means exit code 0 is insufficient; validations must pass.

- **Backend**: The execution engine that runs prompts. Claude CLI provides subprocess execution with tool access; Anthropic API provides direct API calls; Recursive Light provides HTTP API integration.

- **Checkpoint**: Persistent state saved after each sheet, enabling resume from any point. Atomic writes prevent corruption on interruption.

- **Pattern**: A learned association between context and outcome, used to improve future executions through the learning system.

## Documentation

- [Usage Skill](skills/mozart-usage.md) - Comprehensive reference for Mozart usage
- [Examples](examples/) - Working configurations for various use cases
- [Getting Started](docs/getting-started.md) - Step-by-step introduction
- [CLI Reference](docs/cli-reference.md) - Complete command documentation
- [CLAUDE.md](CLAUDE.md) - Instructions for AI assistants working with Mozart

## Development

### Setup

```bash
git clone https://github.com/emzi/mozart-ai-compose.git
cd mozart-ai-compose
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
```

### Code Style

- All functions have type hints
- Pydantic v2 for all models
- Async throughout for I/O operations
- Protocol-based abstractions for swappability

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Mozart orchestrates the complexity so you can focus on the work.
