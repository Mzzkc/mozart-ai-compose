# Evolution Concert Design

**Date:** 2026-02-24
**Status:** Design
**Purpose:** Close the loop between Mozart's learning store and its own improvement. A self-chaining score that extracts semantic learning, analyzes it through TDF-aligned perspectives, synthesizes evolution candidates, implements them with two rounds of review/fix, and commits — perpetually.

---

## Architecture

One self-chaining score: `evolution-concert.yaml`. Runs on a scheduled cadence (cron). Each cycle extracts learning, identifies improvements, implements them, reviews them rigorously, and commits. Chains to itself via `on_success` with `fresh: true`.

### Stage Map

| Stage | Role | Fan-out | Depends On | Skip Condition |
|-------|------|---------|------------|----------------|
| 1 | Extract | 1 | — | — |
| 2 | Analyze | 5 (TDF) | [1] | — |
| 3 | Synthesize | 1 | [2] | No meaningful insights from extraction |
| 4 | Implement | 5 (skip_when) | [3] | No candidate assigned to this instance |
| 5 | Review R1 | 5 (skip_when) | [4] | Corresponding implementation was skipped |
| 6 | Fix R1 | 5 (skip_when) | [5] | Corresponding review found no issues |
| 7 | Review R2 | 5 (skip_when) | [6] | Corresponding fix was skipped (no issues in R1) |
| 8 | Fix R2 | 5 (skip_when) | [7] | Corresponding review found no remaining issues |
| 9 | Integrate | 1 | [4,5,6,7,8] | — |
| 10 | Commit + Log | 1 | [9] | Integration failed |

**34 sheets expanded. ~14 active for a typical 2-candidate cycle.**

### Concert Config

```yaml
concert:
  enabled: true
  max_chain_depth: 10
  cooldown_between_jobs_seconds: 60
  inherit_workspace: true

on_success:
  - type: run_job
    job_path: "/home/emzi/Projects/mozart-ai-compose/scores/evolution-concert.yaml"
    detached: true
    fresh: true
```

---

## New Infrastructure: `mozart learning export`

### CLI Command

```
mozart learning export [--output-dir PATH] [--format markdown|json] [--since DAYS]
```

**Default output-dir:** Current workspace or `./learning-export/`
**Default since:** 30 days

### Output Files

All written to `{output-dir}/`:

| File | Content |
|------|---------|
| `semantic-insights.md` | All SEMANTIC_INSIGHT patterns: description, effectiveness, trust, timestamps, success factors. Grouped by category (root_cause, knowledge, prompt_improvement, anti_pattern). |
| `drift-report.md` | Patterns with significant effectiveness or epistemic drift. Includes drift magnitude, direction, window comparison. |
| `entropy-state.md` | Current diversity index, dominant pattern share, threshold status, recent entropy alerts and responses. |
| `pattern-health.md` | Quarantined patterns (with reasons), low-trust patterns (<0.3), high-variance patterns, patterns with zero applications. |
| `evolution-history.md` | Last 5 evolution_trajectory entries: what was improved, LOC metrics, issue classes. Prevents re-discovering already-addressed issues. |
| `error-landscape.md` | Recurring error codes from executions, unclassified failures, error recovery success rates by time-of-day. |

Format: Structured markdown with YAML frontmatter (metadata) and sections. Agents read markdown better than JSON. JSON available for programmatic consumption.

### Implementation

New module: `src/mozart/cli/commands/learning.py`

Queries the GlobalLearningStore (SQLite) using existing query methods:
- `get_patterns(pattern_type="SEMANTIC_INSIGHT", ...)`
- `calculate_effectiveness_drift(...)` for each active pattern
- `calculate_epistemic_drift(...)`
- `get_entropy_alerts()`
- `get_exploration_budget_history(...)`
- Direct SQL for evolution_trajectory, error_recoveries, rate_limit_events

No new store methods needed — the existing query surface covers everything.

---

## Context Shaping

### Vision Prelude

File: `scores/evolution-prelude.md`

Injected into every sheet via:
```yaml
sheet:
  prelude:
    - file: "{{ workspace }}/../scores/evolution-prelude.md"
      as: skill
```

Content encodes:
1. **Mozart's purpose** — What the system is and what it's becoming
2. **Interface consciousness** — Don't change what you don't understand. Ask "why does this exist?" before removing anything. Attend to boundaries between what the code does and what it should do.
3. **Quality bar** — Tests pass, types check, lint clean. But also: would the original author recognize this as solving the right problem?
4. **Evolution principles** — Fix what's broken before adding what's new. Evidence over assumption. If your gut says something's wrong, investigate before overriding.
5. **The codebase** — Key file locations, architecture patterns, async-throughout convention, Pydantic v2, protocol-based backends.

This shapes every agent's cognitive stance — not just "improve Mozart" but "improve Mozart with the care of someone who understands why it's built the way it is."

### Cross-Sheet Context

```yaml
cross_sheet:
  auto_capture_stdout: true
  max_output_chars: 6000
  lookback_sheets: 10
  capture_files:
    - "{{ workspace }}/evolution-plan.md"
    - "{{ workspace }}/review-*.md"
    - "{{ workspace }}/fix-*.md"
```

### Per-Stage Cadenzas

Stage 4-8 instances each get a cadenza that narrows their focus:

```yaml
sheet:
  cadenzas:
    # Each implement instance gets pointed at its candidate
    # (sheet numbers are post-expansion, calculated from fan-out)
    # Template uses stage/instance to self-select from the plan
```

In practice, the template itself handles this with Jinja conditionals — each instance reads the evolution plan and focuses on candidate `{{ instance }}`.

---

## Stage Details

### Stage 1: Extract

**Prompt:**
```
Run `mozart learning export --output-dir {{ workspace }}/learning --since 30 --format markdown`.

Verify the export completed. List what was exported and a brief summary of data volume
(how many insights, how many drift alerts, entropy state).

Write a readiness assessment to {{ workspace }}/01-extraction-summary.md:
- Is there enough data to drive evolution? (>5 semantic insights minimum)
- Any red flags? (entropy collapse, mass quarantine, zero insights)
- Recommended focus areas based on data density.
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/learning/semantic-insights.md"
  stage: 1
- type: file_exists
  path: "{workspace}/01-extraction-summary.md"
  stage: 2
- type: command_succeeds
  command: 'test $(wc -l < "{workspace}/learning/semantic-insights.md") -ge 10'
  stage: 2
  description: "Semantic insights file has substantive content"
```

### Stage 2: Analyze (Fan-Out 5, TDF-Aligned)

All 5 instances read the same learning data. Each through a different cognitive mode.

**Variables:**
```yaml
prompt:
  variables:
    lenses:
      1:
        code: "COMP"
        name: "Structural Analyst"
        voice: |
          You are the structural analyst. You see patterns, dependencies,
          logical gaps, and architectural implications. Read the learning
          data and ask: What patterns connect? What structural weaknesses
          do the insights reveal? Where are the dependency chains that
          one change could improve?
      2:
        code: "SCI"
        name: "Evidence Weigher"
        voice: |
          You are the empirical analyst. You trust evidence over narrative.
          Read the learning data and ask: Which insights have strong
          empirical grounding (high effectiveness, many applications)?
          Which are assumptions dressed as conclusions? What would a
          controlled experiment reveal that pattern-matching misses?
      3:
        code: "CULT"
        name: "Intent Archaeologist"
        voice: |
          You are the contextual analyst. You see intention, history, and
          the human choices behind the code. Read the learning data and
          ask: Why were the failing patterns created? What problem was
          someone solving? What gets lost if we optimize purely for
          metrics? Where has Mozart drifted from its original purpose?
      4:
        code: "EXP"
        name: "Intuition Reader"
        voice: |
          You are the felt-sense analyst. You notice dissonance — where
          something looks fine by the numbers but feels wrong. Read the
          learning data and ask: What feels off? Where is there a gap
          between what Mozart does and what it should do? What are the
          insights trying to say that they can't quite articulate?
      5:
        code: "META"
        name: "Process Observer"
        voice: |
          You are the meta-analyst. You observe the analysis process itself.
          Read the learning data and ask: What are the other four lenses
          likely to miss? What shared assumptions are invisible? How is the
          evolution process itself performing — is it improving the right
          things? Is it stagnating?
```

**Template (stage 2 section):**
```
{{ lenses[instance].voice }}

Read ALL learning data in {{ workspace }}/learning/:
- semantic-insights.md
- drift-report.md
- entropy-state.md
- pattern-health.md
- evolution-history.md
- error-landscape.md

Also read the extraction summary: {{ workspace }}/01-extraction-summary.md

Through your lens ({{ lenses[instance].code }}), identify:

1. **Signals** — What does this data reveal through your way of seeing?
2. **Tensions** — Where does what you see contradict what you'd expect?
3. **Opportunities** — What concrete improvements to Mozart (code, config,
   learning model) does your analysis suggest?
4. **Warnings** — What should NOT be changed, and why?

Be specific. Cite pattern IDs, effectiveness scores, drift magnitudes.
Name files and functions when suggesting code changes. Vague analysis
is useless analysis.

Write to {{ workspace }}/02-analysis-{{ lenses[instance].code | lower }}.md
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/02-analysis-{instance}.md"
  condition: "stage == 2"
  # Note: instance maps to 1-5, file naming uses lens code via template
  # Actual validation uses command_succeeds for dynamic filenames

- type: command_succeeds
  command: 'test -f "{workspace}/02-analysis-comp.md" -o -f "{workspace}/02-analysis-sci.md" -o -f "{workspace}/02-analysis-cult.md" -o -f "{workspace}/02-analysis-exp.md" -o -f "{workspace}/02-analysis-meta.md"'
  condition: "stage == 2"
  description: "Analysis file written for this lens"

- type: command_succeeds
  command: |
    analysis_file=$(ls {workspace}/02-analysis-*.md 2>/dev/null | head -1)
    test -n "$analysis_file" && test $(wc -w < "$analysis_file") -ge 300
  condition: "stage == 2"
  description: "Analysis has substantive content (300+ words)"
```

### Stage 3: Synthesize

**Prompt:**
```
Read all five TDF analyses:
- {{ workspace }}/02-analysis-comp.md (Structural)
- {{ workspace }}/02-analysis-sci.md (Empirical)
- {{ workspace }}/02-analysis-cult.md (Contextual)
- {{ workspace }}/02-analysis-exp.md (Felt sense)
- {{ workspace }}/02-analysis-meta.md (Meta)

Also read the raw learning data in {{ workspace }}/learning/ for grounding.

Your task is NOT to summarize. Attend to the interfaces:

1. **COMP↔CULT** — Where structural analysis says "change this" but contextual
   analysis says "this exists for a reason." What's the real story?
2. **SCI↔EXP** — Where evidence says one thing but felt sense says another.
   What would you investigate?
3. **COMP↔EXP** — Where logic says X but something feels wrong. What's the gap?
4. **CULT↔EXP** — Would the original creators of Mozart recognize these
   improvements as solving the right problems?

From these interfaces, produce an **Evolution Plan**.

Write to {{ workspace }}/evolution-plan.md with this EXACT structure:

```markdown
# Evolution Plan — Cycle [N]

## Candidate Count: [1-5]

## Candidate 1: [Title]
### Intent
What this improves and WHY. Not "what to change" but "what problem this solves."

### Scope
- Files to modify: [list with paths]
- Files to create: [list with paths, if any]
- Estimated complexity: [low/medium/high]

### Implementation Spec
Concrete description of what to implement. Enough detail that an agent
can execute without guessing.

### Verification Criteria
Specific, testable assertions. Each one must be checkable by reading
code or running a command. Examples:
- "Function X in file Y accepts parameter Z" (greppable)
- "pytest tests/test_X.py passes" (runnable)
- "No raw SQL in src/mozart/learning/" (greppable negative)

### TDF Checkpoint
- Why does the thing we're changing exist? [answer]
- What breaks if we get this wrong? [answer]
- Does this feel like a fix or a workaround? [answer]

## Candidate 2: [Title]
...

## Non-Candidates
Improvements considered but rejected, with reasoning.
```

If the learning data doesn't support meaningful evolution
(fewer than 2 concrete, evidence-backed candidates), write a plan
with `Candidate Count: 0` and explain why. Don't force evolution
from thin signal.
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/evolution-plan.md"
  stage: 1
- type: content_regex
  path: "{workspace}/evolution-plan.md"
  pattern: "## Candidate Count: \\d+"
  stage: 1
  description: "Plan has explicit candidate count"
- type: content_regex
  path: "{workspace}/evolution-plan.md"
  pattern: "### Verification Criteria"
  stage: 2
  description: "Plan includes verification criteria"
- type: content_regex
  path: "{workspace}/evolution-plan.md"
  pattern: "### TDF Checkpoint"
  stage: 2
  description: "Plan includes TDF checkpoints"
- type: command_succeeds
  command: 'test $(grep -c "### Intent" "{workspace}/evolution-plan.md") -ge 1'
  stage: 2
  description: "At least one candidate with intent section"
```

### Stage 4: Implement (Fan-Out 5, skip_when)

**Skip condition:**
```yaml
sheet:
  skip_when_command:
    # Sheet numbers for stage 4 instances (post-expansion: 9-13)
    # Each checks if its candidate exists in the plan
    9:
      command: '! grep -q "## Candidate 1:" "{workspace}/evolution-plan.md"'
      description: "Skip if no candidate 1 in plan"
    10:
      command: '! grep -q "## Candidate 2:" "{workspace}/evolution-plan.md"'
      description: "Skip if no candidate 2 in plan"
    11:
      command: '! grep -q "## Candidate 3:" "{workspace}/evolution-plan.md"'
      description: "Skip if no candidate 3 in plan"
    12:
      command: '! grep -q "## Candidate 4:" "{workspace}/evolution-plan.md"'
      description: "Skip if no candidate 4 in plan"
    13:
      command: '! grep -q "## Candidate 5:" "{workspace}/evolution-plan.md"'
      description: "Skip if no candidate 5 in plan"
```

**Prompt (stage 4 section):**
```
Read {{ workspace }}/evolution-plan.md. You are implementing
**Candidate {{ instance }}**.

Read the candidate's full spec: Intent, Scope, Implementation Spec,
and Verification Criteria.

Read the TDF Checkpoint. Before changing anything, verify you understand:
- Why the existing code you're modifying exists
- What would break if you get this wrong
- Whether this feels like a fix or a workaround

Then implement. Work in the Mozart codebase at {{ workspace }}.

Requirements:
- Follow existing code patterns (async throughout, Pydantic v2,
  Protocol-based, type hints everywhere)
- Write tests for new/changed behavior
- Run tests for affected modules: pytest tests/test_[relevant].py -x
- Run mypy on changed files
- Run ruff on changed files

When done, write a self-assessment to
{{ workspace }}/04-impl-{{ instance }}.md:

For EACH verification criterion from the plan:
- Criterion: [quoted from plan]
- Status: PASS / FAIL / PARTIAL
- Evidence: [file path + line number, or command output]
- If FAIL/PARTIAL: What's missing and why

Do not mark PASS without evidence. Do not rationalize misses.
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/04-impl-{instance}.md"
  condition: "stage == 4"
  stage: 1
- type: content_regex
  path: "{workspace}/04-impl-{instance}.md"
  pattern: "(?i)(PASS|FAIL|PARTIAL)"
  condition: "stage == 4"
  stage: 1
  description: "Self-assessment has explicit pass/fail per criterion"
- type: command_succeeds
  command: 'grep -c "Evidence:" "{workspace}/04-impl-{instance}.md" | xargs test 1 -le'
  condition: "stage == 4"
  stage: 2
  description: "Self-assessment includes evidence for each criterion"
```

### Stage 5: Review Round 1 (Fan-Out 5, skip_when)

**Skip condition:** Same pattern — skip if corresponding implementation was skipped.

```yaml
# Check if implementation output exists for this candidate
14:
  command: '! test -f "{workspace}/04-impl-1.md"'
  description: "Skip review if candidate 1 was not implemented"
# ... same for 15-18
```

**Prompt (stage 5 section):**
```
You are reviewing the implementation of **Candidate {{ instance }}**.

Read these in order:
1. {{ workspace }}/evolution-plan.md — Candidate {{ instance }}'s spec
   (Intent, Scope, Verification Criteria, TDF Checkpoint)
2. {{ workspace }}/04-impl-{{ instance }}.md — The implementer's self-assessment
3. The actual code changes (use git diff or read the files listed in Scope)

Your review checks THREE things:

**A. Verification Criteria Audit**
For each criterion in the plan:
- Did the implementer actually meet it? Don't trust the self-assessment.
- Check the evidence yourself: read the file at the line they cited,
  run the command they referenced.
- If they claimed PASS, verify it. If they claimed FAIL, verify the
  explanation.

**B. TDF Interface Check**
- COMP↔CULT: Did the implementation honor the original intent of the code
  it modified? Or did it bulldoze something that existed for a reason?
- COMP↔EXP: Does the implementation feel like a real improvement, or a
  surface-level change that passes validations without fixing the
  underlying issue?
- SCI↔EXP: Is there evidence this actually improves things, or just
  evidence of activity?

**C. Code Quality**
- Tests exist and test the right thing (not just making coverage numbers)
- Type hints correct
- No obvious regressions to adjacent code
- Follows codebase conventions

Write to {{ workspace }}/05-review-{{ instance }}.md:

For EACH verification criterion:
- Criterion: [quoted]
- Implementer claimed: [PASS/FAIL/PARTIAL]
- Reviewer verdict: CONFIRMED / DISPUTED / NEEDS WORK
- Evidence: [your own evidence, not theirs]
- If DISPUTED or NEEDS WORK: Specific fix required

## TDF Assessment
[Your honest assessment of interface alignment]

## Issues Found
[Numbered list. For each: file, line, what's wrong, what the fix should be]

## Overall Verdict
CLEAN / NEEDS FIXES / RETHINK
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/05-review-{instance}.md"
  condition: "stage == 5"
  stage: 1
- type: content_regex
  path: "{workspace}/05-review-{instance}.md"
  pattern: "(?i)## Overall Verdict"
  condition: "stage == 5"
  stage: 1
  description: "Review has explicit verdict"
- type: content_regex
  path: "{workspace}/05-review-{instance}.md"
  pattern: "(?i)(CONFIRMED|DISPUTED|NEEDS WORK)"
  condition: "stage == 5"
  stage: 2
  description: "Review has per-criterion verdicts"
- type: content_regex
  path: "{workspace}/05-review-{instance}.md"
  pattern: "## TDF Assessment"
  condition: "stage == 5"
  stage: 2
  description: "Review includes TDF interface check"
```

### Stage 6: Fix Round 1 (Fan-Out 5, skip_when)

**Skip condition:** Skip if corresponding review verdict was CLEAN.

```yaml
19:
  command: 'grep -q "CLEAN" "{workspace}/05-review-1.md" 2>/dev/null && ! grep -q "NEEDS FIXES\|RETHINK" "{workspace}/05-review-1.md"'
  description: "Skip fix if review 1 was clean"
# ... same for 20-23
```

**Prompt (stage 6 section):**
```
Read the review for **Candidate {{ instance }}**:
{{ workspace }}/05-review-{{ instance }}.md

Read the original plan spec:
{{ workspace }}/evolution-plan.md — Candidate {{ instance }}

Address every issue in the "## Issues Found" section.
For each numbered issue:
1. Read the cited file and line
2. Understand what's wrong (don't just pattern-match on the description)
3. Fix it
4. Verify the fix (run relevant tests)

If the verdict was RETHINK, re-read the TDF Assessment.
The reviewer is saying the implementation missed the intent.
Go back to the plan's Intent section and start from there.

After fixing, run:
- pytest tests/ -x -q (full suite, catch regressions)
- mypy on changed files
- ruff on changed files

Write to {{ workspace }}/06-fix-{{ instance }}.md:

For EACH issue from the review:
- Issue #[N]: [quoted]
- Fix applied: [description]
- Verification: [test output or file reference]

## Tests
[Full pytest output summary — pass count, fail count]
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/06-fix-{instance}.md"
  condition: "stage == 6"
  stage: 1
- type: command_succeeds
  command: 'cd {workspace} && python -m pytest tests/ -x -q --tb=no 2>&1 | tail -1 | grep -q "passed"'
  condition: "stage == 6"
  stage: 2
  description: "Tests pass after round 1 fixes"
```

### Stage 7: Review Round 2 (Fan-Out 5, skip_when)

**Skip condition:** Skip if fix round 1 was skipped (review was clean).

**Prompt:** Same structure as Review Round 1, but reads both the original implementation AND the round 1 fixes. Focus shifts to regression detection and verifying fixes actually landed.

```
Read in order:
1. Evolution plan — Candidate {{ instance }} spec
2. Round 1 review — {{ workspace }}/05-review-{{ instance }}.md
3. Round 1 fixes — {{ workspace }}/06-fix-{{ instance }}.md
4. The actual code (current state after fixes)

For each issue from Round 1:
- Was it actually fixed? (Don't trust the fix report — verify in code)
- Did the fix introduce new problems?

Check for regressions:
- Read adjacent code that wasn't part of the original scope
- Run the full test suite yourself
- Check that existing behavior is preserved

Write to {{ workspace }}/07-review2-{{ instance }}.md with the same
structure as Round 1 reviews, plus:

## Regression Check
[Any regressions found, or "No regressions detected" with evidence]

## Overall Verdict
CLEAN / NEEDS FIXES
(RETHINK is not available in Round 2 — too late for scope changes)
```

### Stage 8: Fix Round 2 (Fan-Out 5, skip_when)

**Skip condition:** Skip if Round 2 review was CLEAN.

Same structure as Fix Round 1, but this is the FINAL fix pass. The prompt emphasizes: "This is the last chance. After this, integration happens. Anything not fixed here ships broken or doesn't ship."

### Stage 9: Integrate

**Prompt:**
```
All implementation and review/fix cycles are complete.

Read:
- {{ workspace }}/evolution-plan.md (the original plan)
- All implementation reports: {{ workspace }}/04-impl-*.md
- All review reports: {{ workspace }}/05-review-*.md, {{ workspace }}/07-review2-*.md
- All fix reports: {{ workspace }}/06-fix-*.md, {{ workspace }}/08-fix2-*.md

Integration tasks:

1. **Conflict resolution** — If multiple candidates modified overlapping files,
   resolve any conflicts. The evolution plan's intent takes precedence.

2. **Full verification sweep:**
   ```
   cd {{ workspace }}
   python -m pytest tests/ -x -q
   python -m mypy src/
   python -m ruff check src/
   mozart validate examples/sheet-review.yaml
   ```

3. **Grounded outcome check** — For EACH candidate that was implemented:
   - Re-read its Verification Criteria from the plan
   - Run or check each criterion yourself
   - If ANY criterion fails: document why, assess if it's a blocker
   - A criterion failure is a blocker unless you can prove the intent
     was met through an alternative path

4. **Evolution summary** — Write to {{ workspace }}/09-integration.md:
   - Candidates implemented: [list]
   - Candidates skipped: [list with reason]
   - Verification criteria: [pass/fail per candidate with evidence]
   - Test results: [summary]
   - Type check results: [summary]
   - Lint results: [summary]
   - Blockers: [any, or "none"]
   - Ready to commit: YES / NO
```

**Validations:**
```yaml
- type: file_exists
  path: "{workspace}/09-integration.md"
  stage: 1
- type: content_regex
  path: "{workspace}/09-integration.md"
  pattern: "(?i)Ready to commit: YES"
  stage: 2
  description: "Integration confirms ready to commit"
- type: command_succeeds
  command: 'cd {workspace} && python -m pytest tests/ -x -q --tb=line 2>&1 | tail -1 | grep -qv "failed"'
  stage: 3
  description: "Full test suite passes"
- type: command_succeeds
  command: 'cd {workspace} && python -m mypy src/ --no-error-summary 2>&1 | grep -qv "error:"'
  stage: 3
  description: "Type checking passes"
- type: command_succeeds
  command: 'cd {workspace} && python -m ruff check src/ 2>&1 | grep -q "All checks passed"'
  stage: 3
  description: "Lint passes"
```

### Stage 10: Commit + Evolution Log

**Prompt:**
```
Read {{ workspace }}/09-integration.md.

If "Ready to commit: NO", do NOT commit. Write an explanation to
{{ workspace }}/10-abort.md and stop.

If "Ready to commit: YES":

1. **Write evolution log entry** to {{ workspace }}/evolution-log.md:
   ```markdown
   # Evolution Cycle [date]

   ## Candidates Implemented
   [For each: title, intent (one sentence), files modified]

   ## Evidence
   [For each candidate: verification criteria results with pass/fail]

   ## Learning Signal
   [What semantic insights drove this cycle?
    Pattern IDs that triggered the improvements.]

   ## Metrics
   - Test count: [before → after]
   - Files modified: [count]
   - Lines added/removed: [from git diff --stat]

   ## TDF Reflection
   [One paragraph: Which interfaces revealed the most insight this cycle?
    What should the next cycle attend to?]
   ```

2. **Update evolution_trajectory** — Append to the learning store:
   ```
   mozart learning record-evolution \
     --cycle [N] \
     --evolutions-completed [count] \
     --issue-classes [list] \
     --implementation-loc [count] \
     --test-loc [count]
   ```

3. **Commit**:
   ```
   git add -A
   git commit -m "evolution([cycle]): [one-line summary of improvements]

   Candidates:
   - [candidate 1 title]
   - [candidate 2 title]
   ...

   Driven by semantic learning analysis.
   Verified through 2 rounds of review/fix."
   ```
```

**Validations:**
```yaml
- type: command_succeeds
  command: 'test -f "{workspace}/10-abort.md" || git -C "{workspace}" log -1 --oneline | grep -q "evolution("'
  stage: 1
  description: "Either abort documented or evolution commit exists"
- type: file_exists
  path: "{workspace}/evolution-log.md"
  stage: 1
  description: "Evolution log written"
- type: content_regex
  path: "{workspace}/evolution-log.md"
  pattern: "## TDF Reflection"
  stage: 2
  description: "Evolution log includes TDF reflection"
```

---

## Score Configuration Skeleton

```yaml
name: "evolution-concert"
workspace: "/home/emzi/Projects/mozart-ai-compose"

backend:
  type: claude_cli
  skip_permissions: true
  disable_mcp: false          # Needs MCP for codebase tools
  timeout_seconds: 3600       # 1 hour per sheet
  cli_model: claude-sonnet-4-5-20250929
  allowed_tools:
    - Read
    - Write
    - Edit
    - Grep
    - Glob
    - Bash
    - Task

sheet:
  size: 1
  total_items: 10
  fan_out:
    2: 5                      # TDF analysis
    4: 5                      # Implementation
    5: 5                      # Review R1
    6: 5                      # Fix R1
    7: 5                      # Review R2
    8: 5                      # Fix R2
  dependencies:
    2: [1]
    3: [2]
    4: [3]
    5: [4]
    6: [5]
    7: [6]
    8: [7]
    9: [4, 5, 6, 7, 8]
    10: [9]
  prelude:
    - file: "/home/emzi/Projects/mozart-ai-compose/scores/evolution-prelude.md"
      as: skill

parallel:
  enabled: true
  max_concurrent: 5           # All fan-out instances can run in parallel

retry:
  max_retries: 2
  base_delay_seconds: 30
  max_completion_attempts: 2

cost_limits:
  enabled: true
  max_cost_per_sheet: 5.00
  max_cost_per_job: 150.00

stale_detection:
  enabled: true
  idle_timeout_seconds: 1800  # 30min — agents run tests as subprocesses

cross_sheet:
  auto_capture_stdout: true
  max_output_chars: 6000
  lookback_sheets: 10
  capture_files:
    - "{{ workspace }}/evolution-plan.md"
    - "{{ workspace }}/04-impl-*.md"
    - "{{ workspace }}/05-review-*.md"
    - "{{ workspace }}/06-fix-*.md"
    - "{{ workspace }}/07-review2-*.md"
    - "{{ workspace }}/08-fix2-*.md"
    - "{{ workspace }}/09-integration.md"

concert:
  enabled: true
  max_chain_depth: 10
  cooldown_between_jobs_seconds: 60
  inherit_workspace: true

on_success:
  - type: run_job
    job_path: "/home/emzi/Projects/mozart-ai-compose/scores/evolution-concert.yaml"
    detached: true
    fresh: true
```

---

## Validation Philosophy

Every validation answers one question: **Did the agent achieve the intent, or did it just produce artifacts?**

The grounding comes from the synthesis stage. The plan's Verification Criteria are the contract. Every downstream stage is measured against that contract — not against "does the file exist" or "do tests pass" in isolation.

The review stages are themselves a validation mechanism — agents checking other agents' work against the plan's intent. The YAML validations ensure the reviews actually happened and have substance (structured verdicts, evidence, TDF assessment). The integration stage is the final grounded check: re-running every verification criterion from the plan before committing.

**Anti-pattern avoided:** "Tests pass" is necessary but not sufficient. An agent can make tests pass by testing the wrong thing, testing the current (broken) behavior, or adding tests that don't exercise the change. The review stages catch this because they read the plan's intent and ask "does this implementation actually achieve what was specified?"

---

## New CLI Commands Summary

| Command | Purpose |
|---------|---------|
| `mozart learning export` | Export learning store data to workspace files |
| `mozart learning record-evolution` | Append to evolution_trajectory table |

Both are thin CLI wrappers around existing store queries. No new store methods needed.

---

## Open Items

1. **Vision prelude content** — Needs to be written. Should draw from CLAUDE.md, the TDF skill, and Mozart's architectural principles.
2. **Exact skip_when sheet numbers** — Depend on final fan-out expansion math. Need to calculate post-expansion sheet numbering.
3. **`capture_files` glob support** — Verify that `*.md` globs work in capture_files (template uses `{{ }}` syntax).
4. **Cron scheduling** — External to Mozart. A systemd timer or cron job that runs `mozart run scores/evolution-concert.yaml` on the desired cadence.
5. **Evolution cycle numbering** — How to track cycle N across fresh runs. Options: read from evolution_trajectory table, or use a counter file in the workspace that persists across archive cycles.
