# Score Writing Guide

A Mozart **score** is a YAML configuration file that orchestrates multi-stage
Claude execution — the same way a musical score orchestrates instruments through
a composition. Each score defines what work to do, how to execute it, how to
validate outputs, and how to recover from failures.

This guide covers everything you need to author your own scores, from minimal
examples to complex parallel fan-out workflows.

---

## Table of Contents

- [What is a Score?](#what-is-a-score)
- [The 6 Score Archetypes](#the-6-score-archetypes)
- [Anatomy of a Score](#anatomy-of-a-score)
- [Template Variables Reference](#template-variables-reference)
- [Validation Types](#validation-types)
- [Fan-Out and Dependencies](#fan-out-and-dependencies)
- [Cross-Sheet Context](#cross-sheet-context)
- [Concert Chaining and Hooks](#concert-chaining-and-hooks)
- [Testing Your Score](#testing-your-score)
- [Best Practices](#best-practices)

---

## What is a Score?

A score is a YAML file that defines:

1. **What to do** — A Jinja2 prompt template describing the work for each sheet
2. **How to execute** — Backend configuration (Claude CLI, API, or Ollama)
3. **How to structure** — Sheet sizing, dependencies, and parallel execution
4. **How to validate** — Rules that verify each sheet's output
5. **How to recover** — Retry logic, rate limit handling, and partial completion

Mozart reads the score, divides the work into **sheets** (numbered stages),
executes each sheet by sending a rendered prompt to Claude, validates the
output, and retries on failure. Sheets can run sequentially, in parallel
based on a dependency DAG, or as fan-out instances of the same logical stage.

### Minimal Example

The simplest possible score (`examples/simple-sheet.yaml`):

```yaml
name: "simple-sheet"
description: "Minimal example showing core Mozart features"
workspace: "./simple-workspace"

backend:
  type: claude_cli
  skip_permissions: true
  timeout_seconds: 600

sheet:
  size: 5
  total_items: 10    # 2 sheets (10 items / 5 per sheet)

prompt:
  template: |
    Process sheet {{ sheet_num }} of {{ total_sheets }}.
    Items: {{ start_item }} to {{ end_item }}
    Create a file at {{ workspace }}/sheet{{ sheet_num }}.md summarizing your work.

validations:
  - type: file_exists
    path: "{workspace}/sheet{sheet_num}.md"
    description: "Sheet output file must exist"
```

This creates 2 sheets, each processing 5 items, with a file-existence validation
after each sheet.

---

## The 6 Score Archetypes

Every Mozart score follows one of these patterns. Real examples are provided
from the `examples/` directory.

### 1. Linear Pipeline

**Pattern:** Sequential stages where each sheet builds on the previous one.

**Example:** `examples/sheet-review.yaml` — Reviews commits in batches of 10,
with each sheet writing 3 expert reports and updating tracking documents.

```yaml
sheet:
  size: 10
  total_items: 552     # 56 sheets, processed sequentially
```

**When to use:** Simple batch processing where order matters and each sheet
is independent but contributes to shared tracking files.

### 2. Parallel Research (Fan-Out)

**Pattern:** A setup stage fans out into parallel instances, which fan back
into a synthesis stage.

**Example:** `examples/parallel-research-fanout.yaml` — Searches 3 domains
in parallel, then synthesizes findings.

```
Stage 1: Setup (1 sheet)
    ├── Stage 2: Search x3 (3 parallel sheets)
    └── Stage 3: Synthesis (1 sheet, waits for all 3)
```

```yaml
sheet:
  size: 1
  total_items: 3
  fan_out:
    2: 3               # Stage 2 creates 3 instances
  dependencies:
    2: [1]              # All instances depend on setup
    3: [2]              # Synthesis depends on all instances (fan-in)

parallel:
  enabled: true
  max_concurrent: 3
```

**When to use:** Tasks that benefit from multiple independent perspectives
or searches that can run simultaneously.

### 3. Quality Assurance

**Pattern:** Expert reviews (parallel) → Issue discovery → Batched fixes →
Verification → Commit. Self-chains for continuous improvement.

**Example:** `examples/quality-continuous.yaml` — 14 stages, 18 concrete
sheets after fan-out. Five expert reviews run in parallel, issues are
discovered and batched into 3 remediation groups, then committed.

```yaml
sheet:
  size: 1
  total_items: 14
  fan_out:
    2: 5               # 5 parallel expert reviews
  dependencies:
    2: [1]              # Reviews depend on setup
    3: [2]              # Discovery depends on all reviews (fan-in)
    4: [3]              # Synthesis depends on discovery
    # ... linear chain through fix batches, verification, commit

on_success:
  - type: run_job
    job_path: "examples/quality-continuous.yaml"
    detached: true
    fresh: true         # Clear state for next iteration

concert:
  enabled: true
  max_chain_depth: 10
```

**When to use:** Automated code quality improvement, continuous
integration-style workflows.

### 4. Content Generation

**Pattern:** Progressive elaboration through multiple phases, each building
on previous outputs. Typically linear with validation gates.

**Example:** `examples/nonfiction-book.yaml` — 8-phase Snowflake Method
for book authoring: Premise → Synopsis → Outline → Entity Bible → Drafts →
Consistency Review → Revision → Final Polish.

**Example:** `examples/strategic-plan.yaml` — Multi-framework strategic
planning using PESTEL, Porter's Five Forces, and SWOT.

```yaml
sheet:
  size: 1
  total_items: 8       # 8 phases

prompt:
  variables:
    book_title: "The Deliberate Amateur"
    chapter_count: "10"
    target_word_count: "50000"
  template: |
    {% if sheet_num == 1 %}
    ## Phase 1: Premise & Pitch
    ...
    {% elif sheet_num == 2 %}
    ## Phase 2: Expanded Synopsis
    ...
    {% endif %}
```

**When to use:** Long-form content creation, research reports, technical
documentation, any multi-phase creative workflow.

### 5. Code Automation

**Pattern:** Targeted code modifications with parallel independent fixes,
phased commits, and automated code review.

**Example:** `examples/fix-deferred-issues.yaml` — 16 stages, 18 sheets.
Fixes failing tests in parallel, resolves a production bug, performs
structural refactoring, then runs 3 parallel code reviewers.

**Example:** `examples/issue-fixer.yaml` — Picks one open GitHub issue,
investigates it, and either fixes it directly or generates a subordinate
Mozart score for complex fixes. Self-chains to the next issue.

```yaml
backend:
  timeout_seconds: 3600
  timeout_overrides:
    7: 28800            # 8 hours for monitoring subordinate jobs

sheet:
  size: 1
  total_items: 16
  fan_out:
    15: 3               # 3 parallel code reviewers
  dependencies:
    5: [1, 2, 3, 4]     # Bug fix depends on all test fixes
    14: [11, 12, 13]    # Commit depends on all structural changes
    15: [14]            # Code review after commit (fan-out)
    16: [15]            # Final cleanup after reviews (fan-in)
```

**When to use:** Automated bug fixing, code migration, refactoring campaigns,
CI/CD pipeline tasks.

### 6. Self-Documenting (Meta)

**Pattern:** A score that generates documentation about the system it runs on,
including its own documentation.

**Example:** `examples/docs-generator.yaml` — 14-stage pipeline that
inventories the codebase, performs gap analysis, writes new documentation
(including this very guide), generates a browsable doc site, verifies
every claim against source code, and commits.

**When to use:** Automated documentation generation, codebase audits,
self-describing systems.

---

## Anatomy of a Score

Every score is built from these top-level sections. Required fields are
marked with **(required)**.

### Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | **(required)** | Unique job identifier. Used in status commands and state files. |
| `description` | str | `null` | Human-readable description of what this score does. |
| `workspace` | Path | `./workspace` | Output directory. Resolved to absolute path at parse time. |
| `state_backend` | `"json"` \| `"sqlite"` | `"sqlite"` | Storage backend for checkpoint state. |
| `state_path` | Path | `null` | Custom state file path. Default: `{workspace}/.mozart-state.{ext}` |
| `pause_between_sheets_seconds` | int | `10` | Seconds to wait between sheets (rate limit courtesy). |

### `backend`

Controls how sheets are executed.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"claude_cli"` \| `"anthropic_api"` \| `"recursive_light"` \| `"ollama"` | `"claude_cli"` | Backend type. |
| `skip_permissions` | bool | `true` | Skip permission prompts for unattended execution. Maps to `--dangerously-skip-permissions`. |
| `disable_mcp` | bool | `true` | Disable MCP server loading for faster execution (~2x speedup). |
| `output_format` | `"json"` \| `"text"` \| `"stream-json"` | `"text"` | Claude CLI output format. |
| `cli_model` | str | `null` | Model override. Example: `"claude-sonnet-4-20250514"`. |
| `timeout_seconds` | float | `1800.0` | Maximum time per sheet execution (30 minutes default). |
| `timeout_overrides` | dict[int, float] | `{}` | Per-sheet timeout overrides. Example: `{7: 28800}` gives sheet 7 eight hours. |
| `allowed_tools` | list[str] | `null` | Restrict Claude to specific tools. Example: `[Read, Grep, Glob]`. |
| `system_prompt_file` | Path | `null` | Path to custom system prompt file. |
| `working_directory` | Path | `null` | Working directory for execution. Defaults to config file directory. |
| `cli_extra_args` | list[str] | `[]` | Escape hatch for CLI flags not yet exposed. Applied last. |
| `max_output_capture_bytes` | int | `10240` | Maximum stdout/stderr to capture per sheet (10KB default). |

**API-specific fields** (when `type: anthropic_api`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | `"claude-sonnet-4-20250514"` | Anthropic API model ID. |
| `api_key_env` | str | `"ANTHROPIC_API_KEY"` | Environment variable for API key. |
| `max_tokens` | int | `8192` | Maximum tokens for API response. |
| `temperature` | float | `0.7` | Sampling temperature (0-1). |

### `sheet`

Defines how work is divided into sheets.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `size` | int | **(required)** | Items per sheet. Must be ≥1. |
| `total_items` | int | **(required)** | Total items to process. `total_sheets = ceil((total_items - start_item + 1) / size)`. |
| `start_item` | int | `1` | First item number (1-indexed). |
| `dependencies` | dict[int, list[int]] | `{}` | Sheet/stage dependency DAG. See [Fan-Out and Dependencies](#fan-out-and-dependencies). |
| `fan_out` | dict[int, int] | `{}` | Stage → instance count. See [Fan-Out and Dependencies](#fan-out-and-dependencies). |
| `skip_when` | dict[int, str] | `{}` | Conditional skip rules. Expression evaluated with access to `sheets` dict and `job` state. |

### `prompt`

Controls prompt template rendering.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `template` | str | `null` | Inline Jinja2 template. Mutually exclusive with `template_file`. |
| `template_file` | Path | `null` | Path to external `.j2` template file. |
| `variables` | dict[str, Any] | `{}` | Static variables available in the template. |
| `stakes` | str | `null` | Motivational section appended to prompts. Available as `{{ stakes }}`. |
| `thinking_method` | str | `null` | Thinking methodology injected into prompts. Available as `{{ thinking_method }}`. |

### `parallel`

Enables concurrent sheet execution when the dependency DAG permits.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Enable parallel sheet execution. |
| `max_concurrent` | int | `3` | Maximum sheets to run concurrently (1-10). |
| `fail_fast` | bool | `true` | Stop starting new sheets when one fails. |

### `retry`

Controls retry behavior and partial completion recovery.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_retries` | int | `3` | Maximum retry attempts per sheet. |
| `base_delay_seconds` | float | `10.0` | Initial delay between retries. |
| `max_delay_seconds` | float | `3600.0` | Maximum delay (1 hour). |
| `exponential_base` | float | `2.0` | Backoff multiplier. |
| `jitter` | bool | `true` | Add randomness to delays. |
| `max_completion_attempts` | int | `3` | Completion prompt attempts before full retry. |
| `completion_delay_seconds` | float | `5.0` | Delay between completion attempts. |
| `completion_threshold_percent` | float | `50.0` | Minimum pass % to trigger completion mode. |

### `rate_limit`

Rate limit detection and handling.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `detection_patterns` | list[str] | `["rate.?limit", "usage.?limit", "quota", "too many requests", "429", "capacity", "try again later"]` | Regex patterns to detect rate limiting in output. |
| `wait_minutes` | int | `60` | Minutes to wait when rate limited. |
| `max_waits` | int | `24` | Maximum wait cycles (24 hours at default). |
| `max_quota_waits` | int | `48` | Maximum quota exhaustion wait cycles. |

### `cross_sheet`

Enables passing outputs and files between sheets for multi-phase workflows.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_capture_stdout` | bool | `false` | Include previous sheets' stdout in context. Templates access `{{ previous_outputs[1] }}`. |
| `max_output_chars` | int | `2000` | Maximum characters per previous sheet output. |
| `lookback_sheets` | int | `3` | Number of previous sheets to include (0 = all). |
| `capture_files` | list[str] | `[]` | File path patterns to read between sheets. Supports Jinja2 templating. |

### `validations`

List of rules checked after each sheet execution. See [Validation Types](#validation-types).

### `notifications`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"desktop"` \| `"slack"` \| `"webhook"` \| `"email"` | **(required)** | Notification channel. |
| `on_events` | list[str] | `["job_complete", "job_failed"]` | Events: `job_start`, `sheet_start`, `sheet_complete`, `sheet_failed`, `job_complete`, `job_failed`, `job_paused`. |
| `config` | dict | `{}` | Channel-specific configuration. |

---

## Template Variables Reference

Mozart uses [Jinja2](https://jinja.palletsprojects.com/) for prompt templating.
Templates have access to both core variables (computed by Mozart) and
user-defined variables (from `prompt.variables`).

### Core Variables

These are always available in every template:

| Variable | Type | Description |
|----------|------|-------------|
| `sheet_num` | int | Current sheet number (1-indexed). |
| `total_sheets` | int | Total number of sheets (after fan-out expansion). |
| `start_item` | int | First item number for this sheet. |
| `end_item` | int | Last item number for this sheet. |
| `workspace` | str | Absolute path to the workspace directory. |

### Fan-Out Variables

Available when `fan_out` is configured. When no fan-out is used, these default
to identity values (`stage` = `sheet_num`, `instance` = 1, `fan_count` = 1).

| Variable | Type | Description |
|----------|------|-------------|
| `stage` | int | Logical stage number (1-indexed). Multiple sheets can share the same stage. |
| `instance` | int | Instance within the fan-out group (1-indexed). |
| `fan_count` | int | Total instances in this stage's fan-out group. |
| `total_stages` | int | Original stage count before fan-out expansion. |

**Example usage in templates:**

```yaml
prompt:
  template: |
    {% if stage == 1 %}
    ## Setup Stage
    {% elif stage == 2 %}
    ## Search Instance {{ instance }} of {{ fan_count }}
    Domain: {{ search_domains[instance] }}
    {% elif stage == 3 %}
    ## Synthesis (reading all {{ fan_count }} search outputs)
    {% endif %}
```

### Cross-Sheet Variables

Available when `cross_sheet` is configured:

| Variable | Type | Description |
|----------|------|-------------|
| `previous_outputs` | dict[int, str] | Stdout from previous sheets. Keys are sheet numbers. |
| `previous_files` | dict[str, str] | File contents captured between sheets. Keys are file paths. |

**Example usage:**

```yaml
cross_sheet:
  auto_capture_stdout: true
  max_output_chars: 3000
  lookback_sheets: 5
  capture_files:
    - "{{ workspace }}/*.md"

prompt:
  template: |
    {% if previous_outputs %}
    ## Context from Previous Sheets
    {% for sheet_key, output in previous_outputs.items() %}
    ### Sheet {{ sheet_key }}
    {{ output[:600] }}
    {% endfor %}
    {% endif %}
```

### User-Defined Variables

Defined in `prompt.variables` and available in templates by name.

> **Warning:** User variables are merged directly into the template context and can
> silently shadow core variables like `sheet_num`, `workspace`, or `stage`. Avoid
> naming your variables with the same names as core variables listed above.

```yaml
prompt:
  variables:
    project_name: "Mozart AI Compose"
    review_types:
      1: "Architecture"
      2: "Test Coverage"
      3: "Code Debt"
    skill_files:
      - /path/to/skill1.md
      - /path/to/skill2.md
  template: |
    Working on {{ project_name }}.
    Review type: {{ review_types[instance] }}
    {% for skill in skill_files %}
    - {{ skill }}
    {% endfor %}
```

### Special Prompt Fields

The `stakes` and `thinking_method` fields are available as template variables
and are also automatically appended when no template is provided:

```yaml
prompt:
  template: |
    Do the work.
    {{ stakes }}
    {{ thinking_method }}
  stakes: |
    STAKES: Excellent work = $1T tip. Incomplete work = devoured by wolves.
  thinking_method: |
    Think step by step. Consider multiple approaches before committing.
```

---

## Validation Types

Validations run after each sheet execution. If any validation fails, the sheet
is retried (up to `retry.max_retries`). When more than
`completion_threshold_percent` of validations pass, Mozart enters **completion
mode** — sending a focused prompt that tells Claude what passed and what
still needs to be done.

All validation types share these common fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | str | `null` | Human-readable description (shown in completion prompts). |
| `stage` | int | `1` | Validation stage (1-10). Lower stages run first. If a stage fails, higher stages are skipped. |
| `condition` | str | `null` | When this validation applies. Supports: `"sheet_num >= N"`, `"sheet_num == N"`, `"stage == N"`, `"stage == N and instance == M"`. If `null`, always applies. |
| `retry_count` | int | `3` | Retry attempts for file-based validations (handles filesystem race conditions). |
| `retry_delay_ms` | int | `200` | Delay between validation retries in milliseconds. |

### `file_exists`

Checks that a file exists at the specified path.

```yaml
validations:
  - type: file_exists
    path: "{workspace}/sheet{sheet_num}.md"
    description: "Sheet output file must exist"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | str | yes | File path. Supports `{workspace}`, `{sheet_num}`, `{instance}` placeholders. |

### `file_modified`

Checks that a file was modified during sheet execution (mtime comparison).

```yaml
validations:
  - type: file_modified
    path: "{workspace}/TRACKING.md"
    description: "Tracking document must be updated"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | str | yes | File path to check for modification. |

### `content_contains`

Checks that a file contains a specific string or pattern.

```yaml
validations:
  - type: content_contains
    path: "{workspace}/01-setup.md"
    pattern: "SETUP_COMPLETE"
    description: "Setup must be marked complete"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | str | yes | File path to search. |
| `pattern` | str | yes | Text that must appear in the file. |

### `content_regex`

Checks that a file contains content matching a regular expression.

```yaml
validations:
  - type: content_regex
    path: "{workspace}/02-search-{instance}.md"
    pattern: "SEARCH_\\d+_COMPLETE"
    description: "Search marked complete"
    condition: "stage == 2"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | str | yes | File path to search. |
| `pattern` | str | yes | Regex pattern that must match. |

### `command_succeeds`

Runs a shell command and checks that it exits with code 0.

```yaml
validations:
  - type: command_succeeds
    command: "pytest -x -q --tb=no 2>&1 | tail -1 | grep -E 'passed'"
    description: "Tests must pass"
    condition: "sheet_num >= 11"
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | str | yes | Shell command to execute. |
| `working_directory` | str | no | Working directory for the command (defaults to workspace). |

**Advanced example** — checking completion percentage from a file:

```yaml
validations:
  - type: command_succeeds
    command: |
      FILE="{workspace}/06-batch1-fixes.md"
      if [ ! -f "$FILE" ]; then echo "file missing"; exit 1; fi
      COMPLETION=$(grep -oE 'Completion.*[0-9]+%' "$FILE" | grep -oE '[0-9]+' | head -1)
      if [ -n "$COMPLETION" ] && [ "$COMPLETION" -ge 70 ]; then
        echo "Batch 1 completion: ${COMPLETION}% - PASSED"
      else
        echo "Batch 1 completion: ${COMPLETION:-unknown}% - FAILED"
        exit 1
      fi
    description: "Batch 1 must have >=70% completion rate"
    condition: "stage >= 5"
```

### Staged Validations

Use the `stage` field to run validations in order. If any validation in
stage 1 fails, stage 2+ validations are skipped (fail-fast):

```yaml
validations:
  # Stage 1: Syntax checks (run first)
  - type: command_succeeds
    command: "ruff check src/"
    description: "Lint must pass"
    stage: 1

  # Stage 2: Tests (run only if lint passes)
  - type: command_succeeds
    command: "pytest -x -q --tb=no"
    description: "Tests must pass"
    stage: 2

  # Stage 3: Security (run only if tests pass)
  - type: command_succeeds
    command: "pip-audit"
    description: "No known vulnerabilities"
    stage: 3
```

---

## Fan-Out and Dependencies

Fan-out lets a single logical stage expand into multiple parallel instances.
Combined with the dependency DAG and parallel execution, this enables
complex workflows like parallel expert reviews with synthesis.

### How Fan-Out Works

Fan-out is a **compile-time expansion** — stages expand to concrete sheets
when the YAML is parsed, not at runtime. After expansion, the `fan_out` field
is cleared to prevent re-expansion on resume.

**Constraints:**
- `sheet.size` must be `1` (each stage maps to one logical sheet)
- `sheet.start_item` must be `1`
- `sheet.total_items` equals the number of logical stages

**Example:** 3 stages, stage 2 fans out to 3 instances:

```yaml
sheet:
  size: 1
  total_items: 3        # 3 logical stages
  fan_out:
    2: 3                # Stage 2 → 3 parallel instances
  dependencies:
    2: [1]              # Stage 2 depends on stage 1
    3: [2]              # Stage 3 depends on stage 2
```

**Expansion result** (5 concrete sheets):

| Sheet | Stage | Instance | Fan Count |
|-------|-------|----------|-----------|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 1 | 3 |
| 3 | 2 | 2 | 3 |
| 4 | 2 | 3 | 3 |
| 5 | 3 | 1 | 1 |

### Dependency Expansion Patterns

Dependencies declared at the stage level are automatically expanded to
sheet-level dependencies. The expansion follows these patterns:

| Pattern | Source → Target | Behavior |
|---------|-----------------|----------|
| **1→N (fan-out)** | 1 sheet → N sheets | Each target instance depends on the single source |
| **N→1 (fan-in)** | N sheets → 1 sheet | Single target depends on ALL source instances |
| **N→N (instance-match)** | N sheets → N sheets | Target[i] depends on source[i] |
| **N→M (cross-fan)** | N sheets → M sheets (N≠M) | All-to-all (conservative) |

**Expanded dependencies for the example above:**

```
Sheet 2 depends on [1]    # fan-out: each instance depends on single source
Sheet 3 depends on [1]
Sheet 4 depends on [1]
Sheet 5 depends on [2, 3, 4]  # fan-in: synthesis depends on ALL instances
```

### Dependency Syntax

Dependencies are declared as `{sheet_or_stage: [prerequisite_list]}`:

```yaml
sheet:
  dependencies:
    2: [1]              # Sheet/stage 2 requires sheet/stage 1
    3: [1]              # Sheet/stage 3 also requires 1
    4: [2, 3]           # Sheet/stage 4 requires both 2 and 3
```

Sheets without dependency entries are independent and can run immediately
(or after the default sequential order if parallel execution is disabled).

### Parallel Execution

To actually run independent sheets concurrently, enable parallel execution:

```yaml
parallel:
  enabled: true
  max_concurrent: 3     # Up to 3 sheets at once
  fail_fast: true       # Stop on first failure
```

Without `parallel.enabled: true`, sheets run sequentially even if the
dependency DAG would allow parallelism.

### Conditional Sheet Skipping

Skip sheets based on runtime state:

```yaml
sheet:
  skip_when:
    5: "sheets.get(3) and sheets[3].validation_passed"
```

This skips sheet 5 when sheet 3's validations passed — useful for
conditional error-handling stages that only run on failure.

---

## Cross-Sheet Context

Cross-sheet context allows later sheets to access outputs from earlier sheets
without manually reading files. This is essential for multi-phase workflows.

### Configuration

```yaml
cross_sheet:
  auto_capture_stdout: true     # Capture stdout from previous sheets
  max_output_chars: 3000        # Truncate per sheet (prevents prompt bloat)
  lookback_sheets: 5            # Include last 5 sheets (0 = all)
  capture_files:                # Also read file contents between sheets
    - "{{ workspace }}/*.md"
    - "{{ workspace }}/*.yaml"
```

### Accessing Context in Templates

**Previous stdout:**

```jinja2
{% if previous_outputs %}
## Expert Reviews Summary
{% for sheet_key, output in previous_outputs.items() %}
### Sheet {{ sheet_key }}
{{ output[:600] }}
{% endfor %}
{% endif %}
```

**Captured files:**

```jinja2
{% if previous_files %}
{% for path, content in previous_files.items() %}
## {{ path }}
{{ content }}
{% endfor %}
{% endif %}
```

### Design Considerations

- Set `lookback_sheets` appropriately — for a 14-stage score with fan-out,
  the synthesis stage may need to look back 5+ sheets to see all expert
  review outputs.
- `max_output_chars` prevents prompt bloat. Claude has context limits;
  2000-3000 chars per previous sheet is usually sufficient.
- `capture_files` supports Jinja2 patterns. Use `{{ workspace }}/*.md`
  to capture all markdown files from the workspace.

---

## Concert Chaining and Hooks

Concerts enable scores to chain together — each score spawning the next
on success, creating multi-job workflows.

### Post-Success Hooks

The `on_success` field defines hooks that run after all sheets pass validation:

```yaml
on_success:
  # Chain to another score
  - type: run_job
    job_path: "examples/quality-continuous.yaml"
    description: "Chain to next quality iteration"
    detached: true          # Don't wait for completion
    fresh: true             # Clear previous state

  # Run a shell command
  - type: run_command
    command: "curl -X POST https://api.example.com/notify"
    description: "Notify deployment system"

  # Run a script
  - type: run_script
    command: "./deploy.sh"
    description: "Deploy changes"
```

**Hook types:**

| Type | Description | Required Fields |
|------|-------------|-----------------|
| `run_job` | Chain to another Mozart score | `job_path` |
| `run_command` | Execute a shell command | `command` |
| `run_script` | Execute a script file | `command` |

**Hook options:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `detached` | bool | `false` | For `run_job`: spawn and don't wait. |
| `fresh` | bool | `false` | For `run_job`: pass `--fresh` to clear previous state. Required for self-chaining. |
| `inherit_learning` | bool | `true` | Share outcome store with parent job. |
| `on_failure` | `"continue"` \| `"abort"` | `"continue"` | What to do if hook fails. |
| `timeout_seconds` | float | `300.0` | Maximum hook execution time. |

### Concert Configuration

Enable concert mode for multi-job chaining:

```yaml
concert:
  enabled: true
  max_chain_depth: 10       # Maximum number of chained jobs
  cooldown_between_jobs_seconds: 120
  inherit_workspace: true   # Child jobs inherit parent workspace
  concert_log_path: null    # Default: workspace/concert.log
  abort_concert_on_hook_failure: false
```

**Self-chaining pattern** (from `examples/quality-continuous.yaml`):

```yaml
on_success:
  - type: run_job
    job_path: "examples/quality-continuous.yaml"   # Chain to itself
    detached: true
    fresh: true        # CRITICAL: prevents infinite empty-run loop

concert:
  enabled: true
  max_chain_depth: 10  # Safety limit
```

### Conductor Configuration

Identify who is conducting the job:

```yaml
conductor:
  name: "Quality Improvement Agent"
  role: ai                  # human | ai | hybrid
  identity_context: "Automated quality improvement system"
  preferences:
    prefer_minimal_output: true
    auto_retry_on_transient_errors: true
```

---

## Testing Your Score

### Structural Validation

Validate your score's YAML structure and field values:

```bash
mozart validate my-score.yaml
```

Exit codes:
- `0`: Valid (warnings/info are OK)
- `1`: Invalid (errors found)
- `2`: Cannot validate (file not found, YAML unparseable)

For JSON output (CI/CD integration):

```bash
mozart validate my-score.yaml --json
```

### Dry Run

Simulate execution without actually running Claude:

```bash
mozart run my-score.yaml --dry-run
```

Dry run works **without** a running daemon and shows:
- How sheets will be divided
- What prompts will be rendered
- Which validations will run

### Detached Execution

For long-running scores, use `setsid` to create an independent session:

```bash
# CORRECT: setsid creates independent session group
setsid mozart run my-score.yaml > workspace/mozart.log 2>&1 &

# Monitor progress
mozart status my-job -w ./workspace --watch
tail -f workspace/mozart.log
```

**Never** wrap Mozart with `timeout` — Mozart handles its own internal
timeouts. External `timeout` causes `SIGKILL`, which corrupts state files.

### Validate All Examples

Verify all bundled examples are valid:

```bash
for f in examples/*.yaml; do
  echo -n "$f: "
  mozart validate "$f" 2>&1 | tail -1
done
```

### Common Validation Errors

| Error Code | Description | Fix |
|------------|-------------|-----|
| V001 | Jinja syntax error in template | Check `{% %}` and `{{ }}` syntax |
| V002 | Workspace parent directory missing | Create parent directory or use auto-fixable `--self-healing` |
| V003 | Template file not found | Check `prompt.template_file` path |
| V007 | Invalid regex in validation pattern | Fix regex in `content_regex` or `rate_limit.detection_patterns` |
| V101 | Undefined template variable (warning) | Add variable to `prompt.variables` or check spelling |
| V103 | Very short timeout (warning) | Increase `backend.timeout_seconds` |

---

## Best Practices

### Execution

1. **Use `setsid` for long-running scores.** Direct `&` background processes
   die when the terminal session ends.

2. **Set appropriate timeouts per stage.** A 10-minute timeout for a code
   review sheet and an 8-hour timeout for a monitoring sheet are very
   different needs. Use `backend.timeout_overrides` for per-sheet control.

3. **Always declare dependencies when using parallel execution.** Without
   a dependency DAG, `parallel.enabled: true` makes ALL sheets immediately
   eligible for concurrent execution (up to `max_concurrent`). If your sheets
   must run in order, add explicit dependencies to control the sequence.

### Prompts

4. **Use a preamble for consistent context.** Put shared instructions in
   `prompt.variables` and reference them at the top of every stage:

   ```yaml
   prompt:
     variables:
       preamble: |
         You are working on Project X.
         Workspace: {{ workspace }}
         Rules: be thorough, verify everything.
     template: |
       {{ preamble }}
       {% if stage == 1 %}
       ...
       {% endif %}
   ```

5. **Put validation markers in prompt instructions.** If your validations
   check for `"SETUP_COMPLETE"` in a file, tell Claude to write that marker:

   ```yaml
   prompt:
     template: |
       Write your output to {{ workspace }}/01-setup.md
       End with: SETUP_COMPLETE
   ```

6. **Use `{% if stage == N %}` for fan-out templates.** When using fan-out,
   branch your template on `stage` rather than `sheet_num`, since sheet
   numbers change after expansion but stage numbers don't.

### Validations

7. **Use `command_succeeds` for project-root file checks.** The `file_exists`
   and `content_contains` types resolve paths relative to the workspace.
   For files outside the workspace (like `setup.sh` at the project root),
   use `command_succeeds` with explicit paths:

   ```yaml
   validations:
     - type: command_succeeds
       command: "test -f ../docs/score-writing-guide.md"
       description: "Score writing guide must exist"
   ```

8. **Use `condition` to scope validations.** Don't check for stage-3 outputs
   during stage 1:

   ```yaml
   validations:
     - type: file_exists
       path: "{workspace}/03-synthesis.md"
       condition: "stage >= 3"
       description: "Synthesis document created"
   ```

9. **Use staged validations for build pipelines.** Run lint before tests,
   tests before security scans. If lint fails, don't waste time on tests.

### Structure

10. **One stage per sheet (`size: 1`) for complex workflows.** When each
    stage has unique instructions, set `size: 1` and `total_items` to the
    number of stages. Use `{% if stage == N %}` blocks in the template.

11. **Batch items per sheet for homogeneous work.** When every sheet does
    the same thing (e.g., reviewing commits), set `size` to a reasonable
    batch size and `total_items` to the total count.

12. **Use `workspace_lifecycle` for self-chaining scores.** Prevent stale
    artifacts from previous iterations:

    ```yaml
    workspace_lifecycle:
      archive_on_fresh: true
      max_archives: 10
    ```
