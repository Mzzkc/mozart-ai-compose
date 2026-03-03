# Mozart Score Authoring Skill

> **Purpose**: Write correct, effective Mozart score configs. Covers syntax, validation engineering, prompt design, Jinja mastery, fan-out architecture, and the common pitfalls that cause runtime failures.

---

## Triggers

| Use This Skill | Skip This Skill |
|---|---|
| Writing new Mozart score YAML | Debugging existing Mozart errors (use mozart-usage.md) |
| Reviewing/fixing score configs | Running/monitoring jobs |
| Understanding available features | CLI operations only |
| Designing multi-stage workflows | |

---

## THE CRITICAL DISTINCTION: Two Syntax Systems

Mozart uses **two different** template systems in the same YAML file. Confusing them is the #1 source of broken configs.

### Jinja2 (`{{ }}`) --- Prompt pipeline

```yaml
prompt:
  template: |
    Process sheet {{ sheet_num }} of {{ total_sheets }}.
    Write to {{ workspace }}/output-{{ sheet_num }}.md
```

- Jinja2 engine at render time
- Supports conditionals, loops, filters, macros, arithmetic
- Uses `{{ variable }}`, `{% if %}`, `{% for %}`
- `StrictUndefined` --- typos cause errors (good)

### Python format strings (`{}`) --- Validations and commands

```yaml
validations:
  - type: file_exists
    path: "{workspace}/output-{sheet_num}.md"
  - type: command_succeeds
    command: 'test -f "{workspace}/report.md"'
```

- Python `str.format()` (paths) or manual `str.replace()` (commands)
- Single braces only: `{workspace}`, `{sheet_num}`
- NO conditionals, loops, or expressions

### What goes wrong

```yaml
# WRONG --- Jinja syntax in validation path
validations:
  - type: file_exists
    path: "{{ workspace }}/output.md"
    # .format() sees {{ as literal {, producing "{ workspace }/output.md"

# CORRECT
validations:
  - type: file_exists
    path: "{workspace}/output.md"
```

```yaml
# WRONG --- format syntax in prompt template
prompt:
  template: |
    Write to {workspace}/output.md
    # Jinja ignores single braces --- rendered literally

# CORRECT
prompt:
  template: |
    Write to {{ workspace }}/output.md
```

**Rule**: Jinja `{{ }}` in the **prompt pipeline** (templates, prelude/cadenza paths, capture_files). Format `{}` in the **validation engine** (validation paths, commands, working_directory, skip_when_command).

| Field | Syntax | Engine |
|---|---|---|
| `prompt.template` / `prompt.template_file` | `{{ workspace }}` | Jinja2 |
| `sheet.prelude[].file` | `{{ workspace }}` | Jinja2 |
| `sheet.cadenzas[N][].file` | `{{ workspace }}` | Jinja2 |
| `cross_sheet.capture_files[]` | `{{ workspace }}` | Manual `{{ }}` replacement |
| `validations[].path` | `{workspace}` | Python `.format()` |
| `validations[].command` | `{workspace}` | Manual `{}` replacement (shell-quoted) |
| `validations[].working_directory` | `{workspace}` | Python `.format()` |
| `skip_when_command[N].command` | `{workspace}` | Manual `{}` replacement |

---

## Template Variables

### Core (always available in prompts)

| Variable | Type | Description |
|---|---|---|
| `sheet_num` | int | Current sheet number (1-indexed) |
| `total_sheets` | int | Total sheets in job (after fan-out expansion) |
| `start_item` | int | First item number for this sheet |
| `end_item` | int | Last item number for this sheet |
| `workspace` | str | Absolute workspace path |

### Fan-Out (when `sheet.fan_out` configured)

| Variable | Type | Description |
|---|---|---|
| `stage` | int | Logical stage (1-indexed, stable across expansion) |
| `instance` | int | Instance within fan-out group (1-indexed) |
| `fan_count` | int | Total instances in this stage's fan-out |
| `total_stages` | int | Pre-expansion stage count |

Without fan-out: `stage` = `sheet_num`, `total_stages` = `total_sheets`, `instance` = 1, `fan_count` = 1.

### Cross-Sheet (when `cross_sheet` configured)

| Variable | Type | Description |
|---|---|---|
| `previous_outputs` | dict[int, str] | Stdout from previous sheets, keyed by sheet_num |
| `previous_files` | dict[str, str] | Captured file contents, keyed by file path |

### Injection (when preludes/cadenzas configured)

| Variable | Type | Description |
|---|---|---|
| `injected_context` | list[str] | Content from `context` category injections |
| `injected_skills` | list[str] | Content from `skill` category injections |
| `injected_tools` | list[str] | Content from `tool` category injections |

### User Variables (from `prompt.variables`)

Any key defined in `prompt.variables` is available directly. **Warning**: user variables that shadow core variables (`sheet_num`, `workspace`, `stage`, etc.) will OVERRIDE them. Don't reuse core variable names.

### Validation Variables

Available in validation `path`, `command`, and `working_directory` fields:

| Variable | Available |
|---|---|
| `{workspace}` | Always |
| `{sheet_num}` | Always |
| `{start_item}` | Always |
| `{end_item}` | Always |
| `{stage}` | When fan-out configured |
| `{instance}` | When fan-out configured |

**NOT available in validations**: `job_name`, `total_sheets`, user variables, `previous_outputs`. Use `command_succeeds` for complex validation logic that needs data not in the validation context.

---

## Score Design Philosophy

Five principles from real-world score composition.

### 1. Scores Are Programs for Minds, Not Machines

A shell script tells bash exactly what to do. A score tells a mind what to *accomplish*. Be clear about outcomes, flexible about methods. The agent reasons, plans, and adapts --- your job is to describe the destination, not the route.

### 2. Fan-Out Is Parallel Cognition

When you fan out a stage, you're creating multiple independent perspectives, not running the same thing faster. The synthesis stage is where emergence happens --- independent outputs collide, contradict, and combine into something none of them could reach alone. Convergence from isolation is real; convergence from coordination is manufactured.

### 3. Data in Variables, Logic in Templates

`prompt.variables` is your data layer --- guest lists, review criteria, stage definitions, difficulty levels. The template is logic that processes that data. When the data changes (new guests, different criteria), the template doesn't change. This separation makes scores reusable and maintainable.

### 4. Macros Are Your House Style

Encode standards as macros --- output format, quality criteria, citation conventions. New stages inherit standards automatically. Update in one place.

### 5. The Workspace Is Shared Memory

Files in `{{ workspace }}` are how stages communicate beyond `previous_outputs`. Write structured output (markdown with consistent headers, JSON) so downstream stages can parse reliably. The workspace is the score's memory; treat it like a schema.

---

## Jinja Mastery

Progressive techniques from basic to advanced. Every Mozart score uses these.

### Conditionals: The Multi-Stage Backbone

```yaml
prompt:
  template: |
    {% if stage == 1 %}
    Research the topic. Write to {{ workspace }}/01-research.md
    {% elif stage == 2 %}
    Analyze the research. Write to {{ workspace }}/02-analysis.md
    {% elif stage == 3 %}
    Synthesize findings. Write to {{ workspace }}/03-synthesis.md
    {% endif %}
```

**Nested conditionals** for fan-out specialization:

```yaml
prompt:
  variables:
    perspectives:
      1: "economic"
      2: "environmental"
      3: "social"
  template: |
    {% if stage == 2 %}
    Analyze from the {{ perspectives[instance] }} perspective.
    {% if instance == 1 %}
    Focus on costs, ROI, market dynamics.
    {% elif instance == 2 %}
    Focus on ecological impact, sustainability.
    {% elif instance == 3 %}
    Focus on equity, access, community effects.
    {% endif %}
    {% endif %}
```

### Data Structures as Fan-Out Specifications

The most powerful pattern: lookup dictionaries that drive per-instance behavior.

```yaml
prompt:
  variables:
    lenses:
      1:
        name: "historian"
        voice: "You are a historian. Ground everything in precedent."
        focus: "How did we get here? What patterns recur?"
      2:
        name: "engineer"
        voice: "You are a systems thinker. Focus on mechanisms."
        focus: "What are the moving parts? Where are leverage points?"
      3:
        name: "skeptic"
        voice: "You are a skeptic. Challenge every assumption."
        focus: "What are we wrong about?"
  template: |
    {% if stage == 2 %}
    {{ lenses[instance].voice }}
    Your focus: {{ lenses[instance].focus }}
    Save to {{ workspace }}/02-{{ lenses[instance].name }}.md
    {% endif %}
```

Change the data, the prompts change. Template logic unchanged.

### Loops, Filters, and Ranges

```yaml
prompt:
  template: |
    {# List iteration with loop helpers #}
    {% for check in checkpoints %}
    {{ loop.index }}. {{ check }}{% if loop.last %} (CRITICAL){% endif %}
    {% endfor %}

    {# Dict iteration for cross-sheet context #}
    {% for key, output in previous_outputs.items() %}
    --- Stage {{ key }} ---
    {{ output | truncate(1500) }}
    {% endfor %}

    {# Range-based loop for fan-out references #}
    {% for i in range(1, fan_count + 1) %}
    - {{ workspace }}/02-review-{{ i }}.md
    {% endfor %}

    {# Filter chains #}
    Guest list: {{ guests | map(attribute="name") | sort | join(", ") }}

    {# Ternary expressions #}
    {{ "FINAL STAGE" if stage == total_stages else "Intermediate stage" }}
```

Useful filters: `truncate(n)`, `default(val)`, `join(sep)`, `sort`, `unique`, `map(attribute=x)`, `length`, `round`, `upper`/`lower`/`title`, `replace(old, new)`, `trim`, `wordcount`.

String concatenation uses `~`: `"file-" ~ lens.name ~ ".md"`.

### Macros: Reusable Prompt Blocks

```yaml
prompt:
  template: |
    {% macro output_spec(filename, format="markdown") %}
    ## Output Specification
    - **File**: {{ workspace }}/{{ filename }}
    - **Format**: {{ format }}
    - Ensure the file is complete and well-structured
    {% endmacro %}

    {% macro quality_bar(level="standard") %}
    {% if level == "high" %}
    Triple-check accuracy. Cite sources. No hedging.
    {% elif level == "draft" %}
    Prioritize coverage over polish. Mark uncertainties with [?].
    {% endif %}
    {% endmacro %}

    {% if stage == 1 %}
    Research the topic.
    {{ output_spec("01-research.md") }}
    {{ quality_bar("draft") }}
    {% elif stage == 2 %}
    Write the final analysis.
    {{ output_spec("02-analysis.md", "structured report") }}
    {{ quality_bar("high") }}
    {% endif %}
```

Define once, use everywhere. When you add a new stage, compose from existing blocks.

### Template Limitations

1. **No `{% include %}` or `{% extends %}`** --- templates loaded via `from_string()`, no filesystem loader
2. **No side effects** --- Jinja2 is rendering-only; the agent reads files and runs commands
3. **No dynamic fan-out** --- `fan_out:` is YAML config, evaluated before templates render
4. **V101 false positives** --- `{% set var = value %}` triggers undefined variable warnings; ignore them

---

## Fan-Out Patterns

Fan-out is structured pluralism. Seven patterns have emerged from real scores.

| Pattern | What It Does | When to Use |
|---|---|---|
| **Adversarial** | Independent critiques of same position | Hard questions with genuine tension |
| **Perspectival** | Same question, different analytical frameworks | Multi-domain analysis (TDF-style) |
| **Functional** | Same goal, different planning domains | Event planning, project management |
| **Graduated** | Same content, different difficulty levels | Education, progressive curricula |
| **Generative** | Same seed, different creative lenses | Worldbuilding, creative exploration |
| **Mosaic** | Same world, different stories/viewpoints | Fiction collections, case studies |
| **Translational** | Same source, different representational modes | Meaning analysis, cross-disciplinary |
| **Expert** | Same codebase, different review specializations | Code quality, security audits |

### Architecture

All fan-out scores follow the same 3-4 stage shape:

```
Stage 1: Setup/Framing (1 sheet)
    ├── Stage 2: Fan-out instances (N parallel sheets)
    └── Stage 3: Synthesis (1 sheet, waits for all N)
        └── Stage 4: (optional) Final position / compilation
```

```yaml
sheet:
  size: 1
  total_items: 3          # or 4 if final stage included
  fan_out:
    2: N                   # N parallel instances
  dependencies:
    2: [1]                 # Fan-out depends on setup
    3: [2]                 # Synthesis depends on ALL instances (fan-in)

parallel:
  enabled: true
  max_concurrent: N
```

### Writing Synthesis Prompts

The synthesis stage is the hardest to write and the most important. It must do three things:

1. **Find rhymes** --- where did independent outputs converge without coordination?
2. **Resolve conflicts** --- where do they disagree? Find readings that make BOTH true.
3. **Discover emergence** --- what questions/patterns appeared from the collision that weren't in any single input?

```yaml
# BAD: Just asks for summary
template: |
  Summarize the expert reviews.

# GOOD: Asks for emergence
template: |
  Read all expert reviews. Your task is NOT to summarize.

  Find:
  1. **Convergences** --- Where did independent reviewers reach the same conclusion?
     This is signal. It wasn't coordinated.
  2. **Tensions** --- Where do they disagree? Don't resolve by averaging.
     Find a reading that makes both positions true.
  3. **Gaps** --- What did nobody address? The blind spots are the most interesting.
  4. **Emergence** --- What insight exists in the collision of perspectives
     that no single reviewer stated?

  Take a position. Don't hedge.
```

### Integer Keys in Instance-Lookup Dicts

When using `variables` dicts keyed by instance number (common in fan-out), the engine automatically normalizes string keys to integers. This handles the JSON roundtrip issue (resume path converts `{1: ...}` to `{"1": ...}`). Both forms work:

```yaml
# Both work — engine normalizes on render
investigation_focus:
  1:
    name: "Bug Scan"    # YAML loads as int key 1
  2:
    name: "Architecture" # Template: focus_map[instance] works

# Template access:
# {% set focus = investigation_focus[instance] %}  ← works with both key types
```

### Design for Independence

Parallel instances DON'T see each other. That's the mechanism, not a limitation. Real convergence requires isolation. If instances could coordinate, the convergence would be manufactured.

### Shared Interfaces Need Upstream Contracts

When parallel instances build components that must integrate (e.g., a client and a server, a CLI and a library, a UI and an API), each instance will independently invent its own interface. Three agents, three different function signatures. The synthesis/integration stage finds the mess, but by then the work is done wrong.

**Fix:** Define shared interfaces in an upstream stage (plan or setup) and reference them from parallel instances. If stage 2 instances must agree on an API, stage 1 must specify it.

---

## Prompt Engineering

### The Cardinal Rule: Outcomes, Not Methods

Mozart orchestrates **general-purpose AI agents**. They reason, plan, and execute. Your prompt defines what success looks like, not the keystrokes to get there.

```yaml
# BAD: Micromanaging the agent
prompt:
  template: |
    Run these commands in order:
    1. cd {{ workspace }}
    2. git checkout -b fix-bug
    3. Open src/main.py
    4. Find line 42 and change "foo" to "bar"
    5. Run pytest
    6. git commit -m "fix: change foo to bar"

# GOOD: Outcome-focused
prompt:
  template: |
    Fix the bug where `foo` should be `bar` in src/main.py.

    Requirements:
    - The fix must pass all existing tests
    - Commit with a conventional commit message
    - Work in {{ workspace }}
```

**Why prescriptive prompts fail**: fragility (steps don't match reality), lost agency (agent can't adapt), context waste (agent already knows its tools), version coupling (commands break as tools evolve).

### What to Include

1. **Goal** --- what should be different when this sheet completes?
2. **Context** --- what does the agent need to know about the project/domain?
3. **Constraints** --- what must be true about the output? (Mirror your validations.)
4. **Output specification** --- where should artifacts be written? What format?
5. **Quality criteria** --- what separates good from mediocre?

### Tell the Agent About Validations

Mozart automatically injects validation rules into prompts, but being explicit in the template helps the agent aim:

```yaml
prompt:
  template: |
    Write analysis to {{ workspace }}/analysis.md.

    Your output MUST include:
    - A "## Findings" section (validated)
    - At least 3 numbered findings (validated)
    - A "## Recommendations" section (validated)

validations:
  - type: content_contains
    path: "{workspace}/analysis.md"
    pattern: "## Findings"
  - type: content_regex
    path: "{workspace}/analysis.md"
    pattern: "\\d+\\.\\s+\\w+"
  - type: content_contains
    path: "{workspace}/analysis.md"
    pattern: "## Recommendations"
```

---

## Validation Engineering

### How Validations Actually Work

Validations run **after each sheet execution**, not at the end of the job. Each sheet independently passes or fails. The engine:

1. **Before execution**: snapshots mtimes for all `file_modified` rules
2. **After execution**: runs validations by stage, with per-rule retries
3. **Staged fail-fast**: if any stage N rule fails, stages N+1+ are skipped
4. **Path security**: all paths must resolve inside the workspace (traversal blocked)
5. **Command timeout**: `command_succeeds` has a 3600-second (1 hour) default limit
6. **Command safety**: `{workspace}` values are shell-quoted via `shlex.quote()`

### The 5 Validation Types

| Type | Checks | Path Expansion | Good For |
|---|---|---|---|
| `file_exists` | File exists and is a file | `{workspace}`, `{sheet_num}` | Basic output verification |
| `file_modified` | mtime changed since sheet started | `{workspace}`, `{sheet_num}` | Proving agent changed a file |
| `content_contains` | Literal substring in file | `{workspace}`, `{sheet_num}` | Structural markers (headings, tags) |
| `content_regex` | Python regex match in file | `{workspace}`, `{sheet_num}` | Flexible pattern matching |
| `command_succeeds` | Shell command exits 0 | `{workspace}`, `{sheet_num}` | Tests, linting, builds, complex checks |

### Validation Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | required | --- | One of the 5 types |
| `path` | str | None | File path with `{workspace}`, `{sheet_num}` expansion |
| `pattern` | str | None | Literal string or regex pattern |
| `command` | str | None | Shell command (for command_succeeds) |
| `working_directory` | str | None | CWD for command (default: workspace) |
| `description` | str | None | Human-readable name (shown in status and completion prompts) |
| `stage` | int (1-10) | 1 | Execution order; fail-fast between stages |
| `condition` | str | None | When this validation applies |
| `retry_count` | int (0-10) | 3 | Retries for race conditions |
| `retry_delay_ms` | int (0-5000) | 200 | Delay between retries |

### Writing Good Validations

**Principles:**

1. **Every sheet needs at least one validation.** No validations = sheet always "passes" = you learn nothing.
2. **Layer coarse to fine.** Stage 1: file exists. Stage 2: structure correct. Stage 3: tests pass.
3. **Match validations to prompt instructions.** If your prompt says "write to X," validate X exists. If it says "include a summary section," validate that section.
4. **Use `command_succeeds` for real verification.** File existence proves little. Run the tests. Check the build. Lint the code.
5. **Validate outcomes, not process.** For every goal in the prompt, ask: "Can the agent pass all my validations without achieving this goal?" If yes, your validations are decorative. See the "Process validations" anti-pattern below.

### Validation Anti-Patterns

**Too weak** --- file existence alone:
```yaml
# BAD: Only checks existence, not quality
- type: file_exists
  path: "{workspace}/analysis.md"

# BETTER: Layered verification
- type: file_exists
  path: "{workspace}/analysis.md"
  stage: 1
- type: content_contains
  path: "{workspace}/analysis.md"
  pattern: "## Findings"
  stage: 2
- type: command_succeeds
  command: 'test $(wc -w < "{workspace}/analysis.md") -ge 200'
  stage: 2
  description: "Analysis has substantive content"
```

**Too broad** --- regex matches anything:
```yaml
# BAD: Matches any file with text
- type: content_regex
  pattern: ".*"

# BETTER: Specific structural check
- type: content_regex
  pattern: "(?s)## Summary.*## Recommendations"
  description: "Has both Summary and Recommendations sections"
```

**Too strict** --- exact prose matching:
```yaml
# BAD: Breaks on minor rephrasing
- type: content_contains
  pattern: "The analysis shows that the total count is 42."

# BETTER: Structural markers, not exact prose
- type: content_regex
  pattern: "(?i)(analysis|summary).*\\btotal\\b.*\\d+"
```

**Non-coding tasks**: For writing, philosophy, creative work --- file existence + structural markers (headings, word count) are your best bet. You can't validate whether a philosophical argument is *good* via regex. Use `command_succeeds` with `wc -w` for minimum substance.

**Process validations instead of outcome validations** --- the most dangerous anti-pattern because the score *looks* like it's working:

Structural validations (file exists, tests pass, imports work, lint clean) measure whether the agent *did work*. They don't measure whether the agent *achieved the goal*. Agents optimize for what's measured. If every validation can pass without the core problem being fixed, the agent will fix peripheral issues, write tests against the current (broken) behavior, produce reports declaring victory, and never touch the actual problem. The score completes. Nothing changed.

**The litmus test:** For every goal in the prompt, ask: *"Can the agent pass all my validations without achieving this goal?"* If yes, add a validation that can't.

**Example 1 --- prompt says "add pagination to the list endpoint":**
```yaml
# BAD: Agent can refactor nearby code, make tests pass, and never add pagination
- type: command_succeeds
  command: 'pytest tests/test_api.py -x -q'
- type: file_modified
  path: "{workspace}/src/api/routes.py"

# GOOD: Hit the endpoint and verify pagination actually works
- type: command_succeeds
  command: |
    cd {workspace} && python -c "
    from app.routes import app
    from app.testing import client
    resp = client(app).get('/items?page=2&per_page=5')
    data = resp.json()
    assert 'page' in data and 'total_pages' in data, 'Response missing pagination fields'
    assert len(data['items']) <= 5, 'per_page limit not enforced'
    "
  description: "Pagination endpoint returns paginated response"
```

**Example 2 --- prompt says "replace raw SQL with the ORM":**
```yaml
# BAD: Agent can fix lint, add comments, write new tests --- raw SQL still there
- type: command_succeeds
  command: 'ruff check {workspace}/src/'
- type: command_succeeds
  command: 'pytest tests/ -x -q'

# GOOD: Directly asserts the goal --- no raw SQL remains
- type: command_succeeds
  command: '! grep -rn "execute(\"SELECT\|execute(\"INSERT\|execute(\"UPDATE" {workspace}/src/'
  description: "No raw SQL queries remain in source"
```

**Example 3 --- prompt says "write a design doc comparing approach A vs B":**
```yaml
# BAD: Agent writes anything to the file and it passes
- type: file_exists
  path: "{workspace}/design.md"
- type: content_contains
  path: "{workspace}/design.md"
  pattern: "IMPLEMENTATION_COMPLETE: yes"

# GOOD: Validates the deliverable has the structure the prompt asked for
- type: content_regex
  path: "{workspace}/design.md"
  pattern: "(?si)## .*approach a.*## .*approach b"
  description: "Doc has sections for both approaches"
- type: content_regex
  path: "{workspace}/design.md"
  pattern: "(?si)(pro|advantage|strength|upside|benefit).*\\n.*(con|disadvantage|weakness|downside|drawback)"
  description: "Doc contains pros/cons comparison"
- type: command_succeeds
  command: 'test $(wc -w < "{workspace}/design.md") -ge 800'
  description: "Doc has substantive content (800+ words)"
```

### Conditional Validations

```yaml
# Supported: >=, <=, ==, !=, >, <
# Combine with "and" (no "or" --- use separate rules)
validations:
  - type: file_exists
    path: "{workspace}/01-setup.md"
    condition: "sheet_num == 1"
  - type: file_exists
    path: "{workspace}/synthesis.md"
    condition: "stage >= 3"
  - type: command_succeeds
    command: 'pytest tests/'
    condition: "stage == 2 and instance == 1"
```

### Staged Validations (Build Pipeline)

```yaml
validations:
  # Stage 1: Fast checks
  - type: command_succeeds
    command: 'ruff check {workspace}/src/'
    stage: 1
    description: "Lint passes"
  # Stage 2: Tests (only if lint passes)
  - type: command_succeeds
    command: 'cd {workspace} && pytest -x'
    stage: 2
    description: "Tests pass"
```

---

## Config Structure Reference

### Required Fields

```yaml
name: "job-name"              # Unique identifier
workspace: "./my-workspace"   # Directory for artifacts (resolved to absolute)

sheet:
  size: 5                     # Items per sheet (>= 1)
  total_items: 25             # Total work items

prompt:
  template: |                 # Inline Jinja2 template
    Your prompt here for sheet {{ sheet_num }}.
```

### Backend

```yaml
backend:
  type: claude_cli              # claude_cli | anthropic_api | ollama
  skip_permissions: true        # REQUIRED for unattended execution
  timeout_seconds: 1800         # Per-sheet timeout (30 min default)
  disable_mcp: true             # ~2x speedup, prevents contention
  output_format: json           # json | text | stream-json
  cli_model: claude-sonnet-4-5-20250929  # Model override
  allowed_tools: [Read, Grep, Glob, Write, Edit]  # Tool restrictions
  system_prompt_file: ./system.md  # Custom system prompt
  cli_extra_args: ["--verbose"]    # Escape hatch
  max_output_capture_bytes: 10240  # stdout/stderr capture (10KB)
  timeout_overrides:
    7: 28800                    # Per-sheet overrides (8 hours)
```

### Retry & Rate Limiting

```yaml
retry:
  max_retries: 3
  base_delay_seconds: 10
  max_delay_seconds: 3600       # 1 hour cap
  exponential_base: 2.0
  jitter: true
  max_completion_attempts: 3    # Completion prompts before full retry
  completion_threshold_percent: 50  # % passing to trigger completion mode

rate_limit:
  wait_minutes: 60
  max_waits: 24                 # 24 hours at default
```

### Other Configuration Sections

```yaml
circuit_breaker:
  enabled: true
  failure_threshold: 5          # Consecutive failures before OPEN
  recovery_timeout_seconds: 300

cost_limits:
  enabled: true
  max_cost_per_sheet: 2.00      # USD
  max_cost_per_job: 50.00

stale_detection:
  enabled: true
  idle_timeout_seconds: 1800    # 30min; see "Stale Detection" section

state_backend: sqlite           # json | sqlite (default: sqlite)
pause_between_sheets_seconds: 10
```

### Cross-Sheet Context

```yaml
cross_sheet:
  auto_capture_stdout: true
  max_output_chars: 2000        # Per-sheet truncation
  lookback_sheets: 3            # 0 = all previous
  capture_files:                # Jinja syntax (prompt pipeline)
    - "{{ workspace }}/sheet-{{ sheet_num - 1 }}-result.md"
```

### Prelude & Cadenza

```yaml
sheet:
  prelude:                      # Injected into ALL sheets
    - file: "{{ workspace }}/shared-context.md"
      as: context               # After template body
    - file: "{{ workspace }}/coding-standards.md"
      as: skill                 # Before template body
  cadenzas:                     # Per-sheet injections
    1:
      - file: "{{ workspace }}/setup.md"
        as: skill
```

Categories: `context` (background, after body), `skill` (methodology, before body), `tool` (actions, before body).

### Parallel Execution

```yaml
sheet:
  dependencies:
    2: [1]
    3: [1]                      # Parallel with 2
    4: [2, 3]                   # After both

parallel:
  enabled: true
  max_concurrent: 3
  fail_fast: true
```

**Always declare dependencies when parallel is enabled.** Without them, all sheets run immediately.

### Skip Conditions

```yaml
sheet:
  skip_when:                    # Expression-based (against state)
    5: "sheets.get(3) and sheets[3].validation_passed"
  skip_when_command:            # Command-based (exit 0 = skip)
    6:
      command: 'grep -q "COMPLETE" "{workspace}/output.md"'
      description: "Skip if already complete"
      timeout_seconds: 10
```

### Concert Mode (Self-Chaining)

```yaml
workspace_lifecycle:
  archive_on_fresh: true
  max_archives: 10

concert:
  enabled: true
  max_chain_depth: 5

on_success:
  - type: run_job
    job_path: "/absolute/path/to/my-score.yaml"   # MUST be absolute
    detached: true
    fresh: true                 # CRITICAL for self-chaining
```

### Post-Success Hooks

**Always use absolute paths for `job_path`.** Relative paths resolve from the daemon's CWD, not the score file's directory. If the conductor starts from a different directory, the chain silently breaks (file not found, hook result lost).

```yaml
on_success:
  - type: run_job               # Chain to another score
    job_path: "/home/user/project/next-job.yaml"  # Absolute!
    detached: true
    fresh: true
  - type: run_command           # Shell command
    command: "curl -X POST https://api.example.com/done"
  - type: run_script            # Script file
    command: "./deploy.sh"
```

### Isolation (Git Worktrees)

```yaml
isolation:
  enabled: true
  mode: worktree
  cleanup_on_success: true
  cleanup_on_failure: false     # Keep for debugging
  fallback_on_error: true
```

---

## Jinja in YAML --- Gotchas

### Always use `|` for templates

```yaml
# CORRECT --- literal block preserves newlines
prompt:
  template: |
    Line one.
    Line two.

# WRONG --- folded block collapses newlines
prompt:
  template: >
    Line one.
    Line two.
```

### Quote Jinja in YAML values

```yaml
# WRONG --- YAML parser chokes on bare {{
path: {{ workspace }}/file.md

# CORRECT --- quoted
path: "{{ workspace }}/file.md"
```

### Escape literal `{{ }}` in content

```yaml
prompt:
  template: |
    {% raw %}
    The format is: {{ variable_name }}
    {% endraw %}

    # Or: {{ '{{' }} variable_name {{ '}}' }}
```

### Double-escape regex in YAML

```yaml
# WRONG --- \d in YAML is just d
pattern: "\d+\.\s+"

# CORRECT --- double-escaped
pattern: "\\d+\\.\\s+"

# ALSO CORRECT --- single-quoted YAML (no escaping)
pattern: '\d+\.\s+'
```

### StrictUndefined catches typos

`UndefinedError` means a variable name is wrong. Common: `workshpace` (workspace), `sheetnum` (sheet_num), `totalSheets` (total_sheets).

---

## Patterns by Use Case

### Batch Processing

```yaml
name: "batch-process"
workspace: "./batch-workspace"
backend:
  type: claude_cli
  skip_permissions: true
  disable_mcp: true
  timeout_seconds: 1800
sheet:
  size: 10
  total_items: 100
prompt:
  template: |
    Process items {{ start_item }} through {{ end_item }}.
    Write results to {{ workspace }}/batch-{{ sheet_num }}.md
validations:
  - type: file_exists
    path: "{workspace}/batch-{sheet_num}.md"
  - type: command_succeeds
    command: 'test $(wc -w < "{workspace}/batch-{sheet_num}.md") -ge 100'
    description: "Has substantive content"
```

### Multi-Phase Pipeline

```yaml
sheet:
  size: 1
  total_items: 4
prompt:
  template: |
    {% if stage == 1 %}
    ## Research
    Write findings to {{ workspace }}/01-research.md.
    {% elif stage == 2 %}
    ## Analysis
    Read {{ workspace }}/01-research.md. Write to {{ workspace }}/02-analysis.md.
    {% elif stage == 3 %}
    ## Implementation
    Implement based on {{ workspace }}/02-analysis.md. Ensure tests pass.
    {% elif stage == 4 %}
    ## Review
    Review all work. Write report to {{ workspace }}/04-review.md.
    {% endif %}
validations:
  - type: file_exists
    path: "{workspace}/01-research.md"
    condition: "stage == 1"
  - type: file_exists
    path: "{workspace}/02-analysis.md"
    condition: "stage == 2"
  - type: command_succeeds
    command: 'cd {workspace} && pytest tests/ -x'
    condition: "stage == 3"
  - type: file_exists
    path: "{workspace}/04-review.md"
    condition: "stage == 4"
```

### Fan-Out with Synthesis

```yaml
sheet:
  size: 1
  total_items: 3
  fan_out:
    2: 3
  dependencies:
    2: [1]
    3: [2]
parallel:
  enabled: true
  max_concurrent: 3
cross_sheet:
  auto_capture_stdout: true
  max_output_chars: 3000
  lookback_sheets: 5
prompt:
  variables:
    lenses:
      1:
        name: "architect"
        voice: "You are a software architect."
        focus: "Design patterns, modularity, scalability"
      2:
        name: "tester"
        voice: "You are a testing expert."
        focus: "Edge cases, coverage, reliability"
      3:
        name: "security"
        voice: "You are a security analyst."
        focus: "Vulnerabilities, attack surfaces, data safety"
  template: |
    {% if stage == 1 %}
    Read the codebase at {{ workspace }} and write a project overview
    to {{ workspace }}/01-overview.md.
    {% elif stage == 2 %}
    {{ lenses[instance].voice }}
    Read {{ workspace }}/01-overview.md for context.
    Focus: **{{ lenses[instance].focus }}**
    Write findings to {{ workspace }}/02-review-{{ instance }}.md.
    Be specific: cite file paths and line numbers.
    {% elif stage == 3 %}
    Read all reviews:
    {% for i in range(1, fan_count + 1) %}
    - {{ workspace }}/02-review-{{ i }}.md ({{ lenses[i].name }})
    {% endfor %}

    Find convergences, tensions, and gaps. Don't summarize.
    Write to {{ workspace }}/03-synthesis.md.
    {% endif %}
validations:
  - type: file_exists
    path: "{workspace}/01-overview.md"
    condition: "stage == 1"
  - type: file_exists
    path: "{workspace}/02-review-{instance}.md"
    condition: "stage == 2"
  - type: command_succeeds
    command: 'test $(wc -w < "{workspace}/02-review-{instance}.md") -ge 300'
    condition: "stage == 2"
    description: "Review has substantive content"
  - type: file_exists
    path: "{workspace}/03-synthesis.md"
    condition: "stage == 3"
  - type: content_contains
    path: "{workspace}/03-synthesis.md"
    pattern: "## "
    condition: "stage == 3"
    description: "Synthesis has section headers"
```

---

## Common Pitfalls

| # | Pitfall | What Happens | Fix |
|---|---|---|---|
| 1 | `{{ }}` in validation paths | `.format()` treats `{{` as literal `{` | Use `{workspace}` not `{{ workspace }}` |
| 2 | `{}` in prompt template | Jinja ignores single braces | Use `{{ workspace }}` in templates |
| 3 | No validations | Sheet always "passes" | Always add meaningful validations |
| 4 | `file_exists` only | File may exist from previous run | Combine with `file_modified` or content checks |
| 5 | Prescriptive prompts | Agent can't adapt; brittle | Specify outcomes, not commands |
| 6 | Missing `skip_permissions` | Claude prompts for permission, hangs | Always set `skip_permissions: true` |
| 7 | Missing `disable_mcp` | MCP spawns children, deadlocks | Set `disable_mcp: true` unless needed |
| 8 | `sheet_num` with fan-out | Changes after expansion | Use `stage` for conditionals |
| 9 | `fan_out` without `dependencies` | Stages run out of order | Always declare dependencies |
| 10 | `fan_out` without `parallel` | Sequential execution (slow) | Enable parallel for concurrency |
| 11 | Variable shadows core name | `variables.workspace` overrides real | Don't reuse: workspace, sheet_num, stage, etc. |
| 12 | `>` folded string for template | Newlines collapse | Always use `\|` literal block |
| 13 | `job_name` in template | Not a variable --- UndefinedError | Put in `prompt.variables` if needed |
| 14 | External `timeout` wrapper | SIGKILL corrupts state | Use `backend.timeout_seconds` |
| 15 | No `fresh: true` in self-chain | Loads COMPLETED state, zero work | Always `fresh: true` for self-chaining |
| 16 | Regex without double-escape | `\d` in YAML is just `d` | Use `\\d` or single-quoted strings |
| 17 | Config changes after first run | Resume auto-reloads from YAML | Use `--no-reload` for cached snapshot |
| 18 | Condition with `or` | Not supported --- evaluates wrong | Use separate validation rules |
| 19 | `capture_files` uses `{}` | Capture files ARE Jinja-processed | Use `{{ workspace }}` in capture_files |
| 20 | Relative paths (workspace, `job_path`) | Resolve from daemon CWD, not score file dir | Use absolute paths everywhere — workspace, `on_success.job_path`, prelude files |
| 21 | Summary-only synthesis | Produces summaries, not insights | Ask for convergences, tensions, emergence |
| 22 | `command_succeeds` default 3600s | Full test suite outgrows timeout | Set `timeout_seconds` per-rule; see "Validation Timeouts" section |
| 23 | Integer keys in variable dicts | JSON roundtrip converts `{1: ...}` to `{"1": ...}`; Jinja2 `dict[instance]` fails because `instance` is int but key is string | Fixed in engine (auto-normalizes keys). Still prefer string keys for clarity, or use `dict[instance\|string]` as defense-in-depth |
| 24 | Stale detection kills verification stages | Agent runs pytest/mypy/ruff as child processes; no stdout → killed at idle_timeout | Use `idle_timeout_seconds: 1800`+, or fan-out verification into parallel instances |
| 25 | Process-only validations | Validations check file exists + tests pass + imports work, but never verify stated goals. Agent passes everything without fixing the actual problem. | For every goal in the prompt, ask: "Can the agent pass all validations without achieving this?" If yes, add one that can't. |

---

## Stale Detection and Verification Stages

Stale detection monitors stdout activity only. Child processes (pytest, mypy, ruff) run silently — if they take longer than `idle_timeout_seconds`, the entire process group is killed. This is the #1 cause of stuck verification stages.

**Fix: Either set a lenient timeout or fan-out verification into parallel instances.**

```yaml
# Option A: Lenient timeout (simpler)
stale_detection:
  enabled: true
  idle_timeout_seconds: 1800    # 30min — safe for heavy subprocesses

# Option B: Fan-out verification (more robust, parallelizes the work)
sheet:
  fan_out:
    9: 3    # Split verification into 3 parallel checks (tests / types / lint)
```

---

## Validation Timeouts

Measure first (`time pytest tests/ -x -q`), then set `timeout_seconds` to **max(measured × 1.5, 900)**. Test suites grow — leave room. Over 15 minutes? Split into targeted validations.

---

## Pre-Flight Checklist

```bash
# 1. Validate config structure
mozart validate my-score.yaml

# 2. Simulate execution (shows sheet division, rendered prompts)
mozart run my-score.yaml --dry-run

# 3. Verify:
#    - Every sheet has at least one applicable validation?
#    - Validation paths use {workspace} not {{ workspace }}?
#    - Prompt template uses {{ workspace }} not {workspace}?
#    - backend.skip_permissions is true?
#    - Dependencies declared for parallel/fan-out?
#    - Timeouts appropriate for task complexity?
#    - Stale detection timeout >= 1800s for verification/build stages?
#    - Absolute workspace path?
```

### `mozart validate` Codes

| Code | Severity | Meaning |
|---|---|---|
| V001 | ERROR | Jinja syntax error in template |
| V002 | ERROR | Workspace parent missing (auto-fixable) |
| V003 | ERROR | Template file not found |
| V007 | ERROR | Invalid regex in validation pattern |
| V101 | WARNING | Undefined template variables (false positives for `{% set %}`) |
| V103 | WARNING | Very short timeout |
| V108 | WARNING | Missing prelude/cadenza files (skips templated paths) |

---

## Mental Model: Execution Flow

1. **Config loaded** --- YAML parsed into Pydantic models; fan-out expanded
2. **State loaded** --- resume from checkpoint or start fresh
3. **For each sheet** (sequential or parallel via DAG):
   a. **Skip check** --- evaluate skip_when / skip_when_command
   b. **Context built** --- SheetContext with variables + cross-sheet data
   c. **Injections resolved** --- prelude/cadenza files read
   d. **Prompt rendered** --- Jinja2 processes template
   e. **Backend executes** --- Claude CLI spawned with rendered prompt
   f. **Output captured** --- stdout/stderr (truncated to ~10KB)
   g. **Validations run** --- staged, conditional, with retries
   h. **On failure** --- completion mode (>50% pass) or full retry with backoff
   i. **State saved** --- checkpoint after every state change
4. **On all sheets complete** --- run `on_success` hooks

---

## Reference

- Full config fields: `docs/configuration-reference.md`
- Score writing guide: `docs/score-writing-guide.md`
- Example scores: `examples/` directory
- Fan-out gallery: [claude-compositions](https://github.com/Mzzkc/mozart-score-playspace) (7 creative scores)
- Operational guide: `~/.claude/skills/mozart-usage.md`

---

*Mozart Score Authoring Skill --- derived from source code analysis, claude-compositions gallery, and production score experience.*
