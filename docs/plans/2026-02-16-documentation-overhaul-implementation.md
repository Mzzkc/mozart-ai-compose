# Documentation Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Augment Mozart's existing documentation with Jinja2 primer content, cross-repo links, a real docs index, and cleanup of outdated/internal files.

**Architecture:** In-place edits to 5 user-facing docs, move 2 research docs to docs/research/, update 3 memory-bank files. No new user-facing files created. Content pulled from /home/emzi/Projects/claude-compositions/primer.md and README.md, adapted for Mozart context.

**Tech Stack:** Markdown, YAML examples, git

**Key URLs:**
- Mozart repo: `https://github.com/Mzzkc/mozart-ai-compose`
- Compositions repo: `https://github.com/Mzzkc/mozart-score-playspace`

---

### Task 1: Augment score-writing-guide.md — Expressive Templates

**Files:**
- Modify: `docs/score-writing-guide.md` (insert after line 499, which ends the "Template Variables Reference" section, before "Validation Types")

**Context:** The current score-writing-guide covers score anatomy and config fields but doesn't teach Jinja2 expressiveness. The primer at `/home/emzi/Projects/claude-compositions/primer.md` has excellent progressive content (levels 1-9) that needs to be adapted and inserted.

**Step 1: Read the primer source**

Read `/home/emzi/Projects/claude-compositions/primer.md` for the full content to adapt.

**Step 2: Insert "Expressive Templates" section**

Insert a new `## Expressive Templates` section after line 499 (end of Template Variables Reference) and before line 500 (`## Validation Types`). This section should contain the following subsections adapted from the primer:

```markdown
## Expressive Templates

Mozart uses [Jinja2](https://jinja.palletsprojects.com/) for prompt templating.
While basic variable substitution (`{{ sheet_num }}`) covers simple scores,
Jinja2's full power — conditionals, loops, filters, macros — turns scores into
programs that generate instructions for minds.

This section walks through progressively expressive patterns. For runnable
examples that use these techniques, see the
[Mozart Score Playspace](https://github.com/Mzzkc/mozart-score-playspace).

### Arithmetic and Inline Expressions

Jinja2 evaluates expressions inside `{{ }}`:

[Content adapted from primer Level 2 — computed ranges, ternary expressions, percentage calculations. Use the primer's examples but keep them concise.]

### Conditionals (The Multi-Stage Backbone)

[Content adapted from primer Level 3 — if/elif/endif for stage branching, nested conditionals for fan-out specialization with perspectives dict]

### Custom Variables as Data Structures

[Content adapted from primer Level 4 — lists, nested dicts, lookup tables in prompt.variables. The dinner-party guests example is perfect.]

### Loops

[Content adapted from primer Level 5 — iterating lists (loop.index, loop.last), iterating dicts (previous_outputs.items()), range-based loops with ~ concatenation]

### Filters

[Content adapted from primer Level 6 — filter table (upper, lower, title, trim, truncate, default, replace, join, length, round, int/float, first/last, sort, unique, reject/select, map, batch, wordcount), chaining examples]

### Macros (Reusable Prompt Blocks)

[Content adapted from primer Level 7 — output_spec and quality_bar macros, parameterized macros with defaults. Emphasize: "define once, use everywhere"]

### Fan-Out + Jinja2

[Content adapted from primer Level 8 — the 4-lens example (historian, engineer, poet, skeptic) with full YAML showing sheet config, variables, and template. This is the expressive power duo.]

### Advanced Patterns

[Content adapted from primer Level 9 — progressive difficulty, conditional validation hints, cross-sheet memory with selective recall, self-documenting stages]

### Template Limitations

A few things that won't work in Mozart templates:

1. **`{% include %}`** — Templates are loaded via `from_string()`, not from a filesystem loader
2. **`{% extends %}`** — No template inheritance for the same reason
3. **Modifying validation paths** — Validation paths use `{single_brace}` Python format strings, not Jinja2
4. **Side effects** — Jinja2 can't make HTTP calls, read files, or execute commands
5. **Dynamic fan-out** — `fan_out:` is YAML config, evaluated before templates render
```

**Step 3: Insert "Fan-Out Patterns" section**

After "Template Limitations", before "Validation Types", add:

```markdown
## Fan-Out Patterns

Fan-out isn't just parallelism — it's structured pluralism. Different fan-out
patterns produce different kinds of emergence in the synthesis stage.

| Pattern | What It Does | Example |
|---------|-------------|---------|
| **Adversarial** | Independent critiques of the same position | `examples/parallel-research.yaml`, [dialectic.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/dialectic.yaml) |
| **Perspectival** | Same question, different analytical frameworks | [thinking-lab.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/thinking-lab.yaml) |
| **Functional** | Same goal, different planning domains | [dinner-party.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/dinner-party.yaml) |
| **Graduated** | Same content, different difficulty levels | [skill-builder.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/skill-builder.yaml) |
| **Generative** | Same seed, different creative lenses | [worldbuilder.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/worldbuilder.yaml) |
| **Expert** | Same codebase, different review specializations | `examples/quality-continuous.yaml` |

The synthesis stage that follows fan-out is where emergence happens. Independent
outputs produce convergences that no single perspective would generate.

For creative examples with real output, see the
[Mozart Score Playspace](https://github.com/Mzzkc/mozart-score-playspace).
```

**Step 4: Insert "Philosophy of Score Design" section**

After "Fan-Out Patterns", before "Validation Types", add content adapted from the primer's philosophy section:

```markdown
## Philosophy of Score Design

**Scores are programs for minds, not machines.** A shell script tells bash exactly
what to do. A score tells a mind what to *accomplish*. Design accordingly — be
clear about outcomes, flexible about methods.

**Fan-out is parallel cognition.** When you fan out a stage, you're creating
multiple independent perspectives. The synthesis stage is where those perspectives
collide, contradict, and combine into something none of them could reach alone.

**Macros are your house style.** Every organization has implicit standards — how
to format output, what quality level to expect. Encode these as macros. New scores
inherit your standards automatically.

**Data in variables, logic in templates.** Keep `prompt.variables` as the source
of truth for domain-specific data. Keep the template as the logic that processes
it. When the data changes, the template doesn't need to.

**The workspace is shared memory.** Files in `{{ workspace }}` are how stages
communicate beyond `previous_outputs`. Write structured output so downstream
stages can parse it reliably.
```

**Step 5: Verify the document reads coherently**

Read the full score-writing-guide.md after edits and verify:
- Table of contents links still work (update TOC if needed)
- No duplicate content between existing sections and new sections
- Section ordering flows logically

**Step 6: Commit**

```bash
git add docs/score-writing-guide.md
git commit -m "docs(score-writing): add expressive templates, fan-out patterns, and design philosophy

Pull Jinja2 primer content from mozart-score-playspace into score-writing-guide.
Add fan-out pattern taxonomy and philosophy of score design section."
```

---

### Task 2: Expand docs/index.md

**Files:**
- Modify: `docs/index.md` (replace entire contents)

**Step 1: Replace index.md with documentation hub**

Replace the 3-line stub with a categorized doc map. Include all docs in docs/ plus links to examples and external repos.

Content structure:
- One-line project description
- "Getting Started" category: getting-started.md, score-writing-guide.md
- "Reference" category: cli-reference.md, configuration-reference.md, limitations.md
- "System Guides" category: daemon-guide.md, MCP-INTEGRATION.md, mozart-reference.md
- "Learning & Internals" category: DISTRIBUTED-LEARNING-ARCHITECTURE.md
- "Examples" category: link to examples/README.md, link to mozart-score-playspace
- "Research" category: link to docs/research/ (after Task 6 moves files there)
- Suggested reading order for new users vs experienced users

**Step 2: Commit**

```bash
git add docs/index.md
git commit -m "docs: expand index.md from stub to documentation hub"
```

---

### Task 3: Augment docs/getting-started.md

**Files:**
- Modify: `docs/getting-started.md`

**Step 1: Add repo URL and project link near top**

After the title and before "Installation", add a brief note:

```markdown
**Repository:** [github.com/Mzzkc/mozart-ai-compose](https://github.com/Mzzkc/mozart-ai-compose)
```

**Step 2: Strengthen daemon requirement in Step 4**

Current Step 4 (lines 129-136) just says "The Mozart daemon is required." Add a callout:

```markdown
> **What if I skip this?** Running `mozart run` without a conductor produces:
> `Error: Conductor not running. Start with: mozart start`
> Only `mozart validate` and `mozart run --dry-run` work without a conductor.
> See the [Daemon Guide](daemon-guide.md) for details.
```

**Step 3: Add fan-out pattern to Common Patterns**

After Pattern 3 (Data Processing, line ~286), add Pattern 4:

```markdown
### Pattern 4: Parallel Expert Reviews (Fan-Out)

[Minimal fan-out example: 3 stages, stage 2 fans to 3 parallel reviewers, stage 3 synthesizes. Show sheet config with fan_out and dependencies, parallel enabled, and a template using {% if stage == N %} with instance-based specialization.]
```

**Step 4: Expand Next Steps section**

Replace the current Next Steps list (lines 344-348) with a progressive learning path:

```markdown
## Next Steps

**Learn more:**
- [Score Writing Guide](score-writing-guide.md) — Archetypes, Jinja2 templates, fan-out patterns, concert chaining
- [CLI Reference](cli-reference.md) — All commands and options
- [Configuration Reference](configuration-reference.md) — Every config field documented

**Explore examples:**
- [Examples](../examples/) — 24 working configurations across software, research, writing, and planning
- [Mozart Score Playspace](https://github.com/Mzzkc/mozart-score-playspace) — Creative showcase: philosophy, worldbuilding, education, and more

**Go deeper:**
- [Daemon Guide](daemon-guide.md) — Conductor architecture and troubleshooting
- [Known Limitations](limitations.md) — Constraints and workarounds
```

**Step 5: Commit**

```bash
git add docs/getting-started.md
git commit -m "docs(getting-started): add repo URL, strengthen daemon guidance, add fan-out pattern"
```

---

### Task 4: Update examples/README.md

**Files:**
- Modify: `examples/README.md`

**Step 1: Add missing examples to tables**

Add to "Software Development Examples" table:
- `issue-solver.yaml` — "17-stage roadmap-driven issue solver with fan-out reviewers and self-chaining" — High
- `fix-deferred-issues.yaml` — "16-stage parallel bug fixing with quality gates" — High
- `fix-observability.yaml` — "13-stage observability improvements with parallel reviewers" — High
- `phase3-wiring.yaml` — "10-stage daemon scheduler/rate-coordinator wiring" — High

**Step 2: Add Creative & Experimental section**

After "Beyond Coding" section, add:

```markdown
### Creative & Experimental

For scores that explore Mozart's expressive capabilities beyond typical workflows, see the [Mozart Score Playspace](https://github.com/Mzzkc/mozart-score-playspace):

| Score | Domain | Fan-Out Pattern | Description |
|-------|--------|----------------|-------------|
| [dialectic.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/dialectic.yaml) | Philosophy | Adversarial | Hegelian dialectic: thesis → 3 independent antitheses → synthesis |
| [thinking-lab.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/thinking-lab.yaml) | Meta-cognition | Perspectival | Multi-perspective analysis through 5 parallel lenses |
| [dinner-party.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/dinner-party.yaml) | Hospitality | Functional | Parallel planning across menu, drinks, ambiance, logistics |
| [skill-builder.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/skill-builder.yaml) | Education | Graduated | Progressive curriculum from guided to mastery |
| [worldbuilder.yaml](https://github.com/Mzzkc/mozart-score-playspace/blob/main/scores/worldbuilder.yaml) | Creative writing | Generative | Build fictional worlds through 5 independent creative lenses |

These scores include real output in their workspace directories.
```

**Step 3: Fix validation syntax error**

Lines 100-106 of the current examples/README.md use `{{ workspace }}` in validation paths — should be `{workspace}`. Fix these.

**Step 4: Add missing examples to validation table**

Add issue-solver, fix-deferred-issues, fix-observability, phase3-wiring to the "Validation Summary" table at the bottom.

**Step 5: Commit**

```bash
git add examples/README.md
git commit -m "docs(examples): add missing examples, link to score playspace, fix validation syntax"
```

---

### Task 5: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Add repo badge/link near top**

After the title line, verify the repo URL is visible. The current README has it in clone commands but not as a standalone link. If not already present as a link, add it.

**Step 2: Add score playspace to Documentation section**

In the Documentation section (lines 470-479), add:

```markdown
- [Score Playspace](https://github.com/Mzzkc/mozart-score-playspace) - Creative showcase: philosophy, worldbuilding, education, and more
```

**Step 3: Add score playspace link to Examples section**

After the "Beyond Coding" examples table (line ~424), before the architecture section, add a brief note:

```markdown
For creative and experimental scores (philosophy, worldbuilding, education), see the [Mozart Score Playspace](https://github.com/Mzzkc/mozart-score-playspace).
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): add score playspace links to documentation and examples sections"
```

---

### Task 6: Move research docs to docs/research/

**Files:**
- Create: `docs/research/` directory
- Move: `docs/OPUS-CONVERGENCE-ANALYSIS.md` → `docs/research/OPUS-CONVERGENCE-ANALYSIS.md`
- Move: `docs/TOKEN-COMPRESSION-STRATEGIES.md` → `docs/research/TOKEN-COMPRESSION-STRATEGIES.md`
- Create: `docs/research/README.md`

**Step 1: Create directory and move files**

```bash
mkdir -p docs/research
git mv docs/OPUS-CONVERGENCE-ANALYSIS.md docs/research/
git mv docs/TOKEN-COMPRESSION-STRATEGIES.md docs/research/
```

**Step 2: Create docs/research/README.md**

```markdown
# Research Documents

Internal research and analysis documents. These are not user-facing
documentation — they record investigations, strategies, and meta-analyses
that informed Mozart's development.

- [Token Compression Strategies](TOKEN-COMPRESSION-STRATEGIES.md) — Research into reducing token costs (not yet implemented)
- [Opus Convergence Analysis](OPUS-CONVERGENCE-ANALYSIS.md) — Meta-analysis comparing Mozart and RLF evolution patterns
```

**Step 3: Update any cross-references**

Check if any other doc links to these files and update paths. Likely candidates: docs/index.md (already updated in Task 2 to point to research/).

**Step 4: Commit**

```bash
git add docs/research/
git commit -m "docs: move research/internal docs to docs/research/"
```

---

### Task 7: Update memory-bank files

**Files:**
- Modify: `memory-bank/projectbrief.md`
- Modify: `memory-bank/progress.md`
- Modify: `memory-bank/context/techContext.md`

**Step 1: Update projectbrief.md architecture**

Read current file. Update the "Architecture" section (currently lists 5 modules from 2025-12-18) to reflect current state:
- Add daemon (conductor, job manager, IPC, scheduler, backpressure)
- Add learning system (global store, patterns, drift, budget, entropy)
- Add MCP integration
- Add isolation (worktree)
- Add healing (self-healing, validation)
- Update "Last Updated" to 2026-02-16
- Keep scope, non-goals, origin, success criteria, technology choices intact

**Step 2: Update progress.md**

Read current file. Add entries after the last entry (Phase 2, 2025-12-23):
- Learning system implementation (2026-01-14)
- Daemon mode (2026-02-11)
- MCP integration
- Example expansion (2026-01-23)
- Self-healing and enhanced validation
- Quality scores running autonomously (2026-02-14)
- Bug fixes and stabilization (2026-02-15)
- Update metrics table with current numbers (3384+ tests, 26+ CLI commands)

**Step 3: Update techContext.md**

Read current file. Update:
- Architecture diagram from 4-layer to 7-layer (CLI, Conductor, Execution, Backends, State, Learning, Validation)
- Add daemon section (conductor, job manager, IPC, event bus)
- Add learning system section (global store, patterns, trust scoring)
- Add MCP section
- Add isolation section (worktree)
- Update dependencies list (add daemon deps: aiohttp, etc.)
- Update "Last Updated" to 2026-02-16

**Step 4: Commit**

```bash
git add memory-bank/
git commit -m "docs(memory-bank): update projectbrief, progress, and techContext to current state"
```

---

### Task 8: Final verification

**Step 1: Verify all internal links work**

Check that relative links between docs still resolve correctly:
- docs/index.md links to all docs
- getting-started.md Next Steps links
- score-writing-guide.md table of contents
- examples/README.md links to example files
- README.md links to docs/

**Step 2: Verify external links are correct**

- `https://github.com/Mzzkc/mozart-ai-compose` — repo URL
- `https://github.com/Mzzkc/mozart-score-playspace` — compositions repo
- Links to specific score files in playspace repo

**Step 3: Read through each edited file for coherence**

Quick read of each edited doc to catch:
- Duplicate content
- Broken flow
- Inconsistent terminology
- Missing context

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "docs: fix links and coherence issues from documentation overhaul"
```
