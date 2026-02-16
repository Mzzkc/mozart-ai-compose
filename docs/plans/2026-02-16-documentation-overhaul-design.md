# Documentation Overhaul Design

**Date:** 2026-02-16
**Approach:** Augment in-place (no new files except research/ directory)
**Audience:** Both new users (onboarding) and score authors (power users)

---

## Goals

1. Pull Jinja2 primer content from claude-compositions into score-writing-guide.md
2. Link to claude-compositions repo as creative showcase throughout docs
3. Add repo URL (https://github.com/Mzzkc/mozart-ai-compose) prominently
4. Fix docs/index.md from 3-line stub to real documentation hub
5. Fill gaps in examples/README.md (missing examples, fan-out patterns)
6. Move research/internal docs to docs/research/
7. Update outdated memory-bank files (projectbrief, progress, techContext)

---

## Files to Edit

### Primary (User-Facing)

#### docs/score-writing-guide.md
**Biggest change.** Add new sections after "Template Variables Reference":

- **Expressive Templates** (from primer.md levels 1-9):
  - Arithmetic and inline expressions
  - Conditionals as multi-stage backbone
  - Custom variables as data structures
  - Loops (lists, dicts, range-based)
  - Filters (chaining, useful-filters table)
  - Macros (reusable prompt blocks)
  - Fan-out + Jinja2 combined
  - Advanced patterns (progressive difficulty, selective recall, self-documenting stages)
  - What won't work (no include/extends, no side effects, no dynamic fan-out)

- **Fan-Out Patterns** (from claude-compositions):
  - Adversarial, Perspectival, Functional, Graduated, Generative
  - Link to claude-compositions for examples with real output

- **Philosophy of Score Design** (from primer.md):
  - Scores are programs for minds
  - Fan-out is parallel cognition
  - Macros are house style
  - Data in variables, logic in templates
  - The workspace is shared memory

#### docs/getting-started.md
- Add repo URL at top
- Strengthen daemon requirement messaging (callout with error message)
- Add fan-out pattern to Common Patterns section
- Expand Next Steps with progressive learning path and claude-compositions link

#### docs/index.md
- Replace 3-line stub with categorized documentation map
- Categories: Getting Started, Reference, System Guides, Learning, Examples, Research

#### examples/README.md
- Add missing examples: issue-solver, fix-observability, phase3-wiring
- Add Creative & Experimental section linking to claude-compositions
- Add fan-out pattern taxonomy table
- Fix validation syntax in example ({{ }} → { })

#### README.md
- Add claude-compositions link in Documentation section
- Ensure repo URL appears prominently beyond just clone commands

### Secondary (Cleanup)

#### docs/research/ (new directory)
- Move OPUS-CONVERGENCE-ANALYSIS.md
- Move TOKEN-COMPRESSION-STRATEGIES.md
- Add brief README.md explaining these are internal research

#### memory-bank/projectbrief.md
- Update architecture to include daemon, learning, MCP, isolation
- Update timestamp

#### memory-bank/progress.md
- Add phases 3+ (daemon, learning, MCP, examples, stabilization)
- Update metrics table

#### memory-bank/context/techContext.md
- Update 4-layer → 7-layer architecture
- Add daemon, learning, MCP, isolation sections
- Update dependencies

### Not Touching
- docs/cli-reference.md, daemon-guide.md, configuration-reference.md
- docs/limitations.md, MCP-INTEGRATION.md, mozart-reference.md
- docs/DISTRIBUTED-LEARNING-ARCHITECTURE.md
- STATUS.md, activeContext.md, CHANGELOG.md, VISION.md

---

## Content Sources

- **primer.md** from /home/emzi/Projects/claude-compositions/primer.md
  - Jinja2 levels 1-9, philosophy of score design
  - Adapt to Mozart context (not copy verbatim)

- **README.md** from /home/emzi/Projects/claude-compositions/README.md
  - Fan-out pattern taxonomy (adversarial, perspectival, functional, graduated, generative)
  - Link as creative showcase

- **Repo URL:** https://github.com/Mzzkc/mozart-ai-compose
- **Compositions URL:** https://github.com/Mzzkc/claude-compositions (verify)

---

## Implementation Order

1. score-writing-guide.md (largest change, most value)
2. docs/index.md (quick win, high impact)
3. getting-started.md (targeted additions)
4. examples/README.md (fill gaps)
5. README.md (light touch)
6. docs/research/ move (cleanup)
7. memory-bank updates (cleanup)
