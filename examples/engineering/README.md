# Engineering Examples

These scores automate software development workflows — from solving GitHub issues to improving code quality to generating implementation plans from design documents. They demonstrate how AI orchestration can replace manual development cycles with autonomous, multi-stage pipelines that investigate, plan, execute, review, and ship production-ready changes.

## Scores

| Score | What It Does | Sheets | Patterns Used | Time | Cost |
|-------|-------------|--------|--------------|------|------|
| [issue-solver](issue-solver.yaml) | Auto-selects issues from a roadmap, plans phased implementation, executes with parallel quality review, commits and ships | 19 | Succession Pipeline, Fan-out + Synthesis | 2-8h | ~$40-100 |
| [quality-continuous-generic](quality-continuous-generic.yaml) | Language-agnostic quality pipeline — parallel expert reviews, batched fixes by difficulty, commits and files GitHub issues for next iteration | 16 | Immune Cascade, Fan-out + Synthesis | 10-15h | ~$50-130 |
| [score-composer](score-composer.yaml) | Reads a design document and generates a runnable Marianne score with implementation tasks, validations, and dependencies | 4 | Succession Pipeline | 1-2h | ~$10-20 |
| [codebase-rewrite](codebase-rewrite.yaml) | Iterative language/framework migration with Cathedral Construction, CEGAR Loop, and Commissioning Cascade patterns | 8 | Cathedral Construction + CEGAR Loop + Commissioning Cascade | 4-12h | ~$20-60 |
| [saas-app-builder](saas-app-builder.yaml) | Full-stack application generator with contract-first parallel builds and validation gates | 6 | Prefabrication + Shipyard Sequence + Commissioning Cascade | 2-6h | ~$10-30 |
| [lovable-generator](lovable-generator.yaml) | Web application generator producing a complete deployable app from a concept description | 5 | Succession Pipeline | 1-3h | ~$5-15 |

### issue-solver.yaml

Point it at a roadmap file and a GitHub label. It selects the next eligible issue (respecting dependencies), investigates the codebase, plans a 1-4 phase implementation strategy, executes each phase with fix+completion passes, runs three parallel quality reviewers (functional, E2E, code quality) whose findings are synthesized before shipping, updates documentation, verifies all tests pass, commits, pushes, and closes the issue. Then self-chains to solve the next one.

The pipeline implements **Succession Pipeline** — each stage fundamentally transforms the workspace substrate (selection → investigation → plan → code → review → ship). Within the review phase, **Fan-out + Synthesis** launches three independent reviewers in parallel. Convergence from isolation produces genuine signal — each reviewer has no knowledge of the others' findings, so agreement indicates real issues.

### quality-continuous-generic.yaml

A self-chaining quality improvement pipeline that discovers and fixes code issues without language-specific configuration. Stage 1 generates test/typecheck/lint runner scripts by examining the project. Three parallel expert reviews (Architecture, Test Coverage, Code Debt) scan the codebase from different analytical lenses. Their findings are synthesized, categorized, and prioritized into three remediation batches (quick wins, medium effort, significant). Each batch gets a fix pass and a completion pass. The pipeline verifies all changes, commits, resolves merge conflicts, and files GitHub issues for unresolved problems. Then chains to the next iteration.

This is **Immune Cascade** — cheap broad scanning (expert reviews) identifies ALL issues first, triage creates a targeting brief, then expensive deep work (code fixes) begins on prioritized targets. This avoids wasting fix passes on issues that a cheap sweep would deprioritize. The expert reviews use **Fan-out + Synthesis** — independent parallel reviews produce genuine convergence when 2 of 3 reviewers (who can't see each other) flag the same module.

### score-composer.yaml

Transforms design documents into executable implementation scores. Given a design doc, it produces a structured analysis identifying all deliverables and dependencies, decomposes them into ordered implementation tasks with acceptance criteria, generates a Marianne score YAML where each sheet is one implementation step, validates the score syntax, and produces documentation explaining how to run it.

This is **Succession Pipeline** — each stage requires categorically different methods (analytical reading → decomposition → code generation → syntax validation) and each output becomes the next input substrate. The generated score can then be run to execute the implementation automatically.

### codebase-rewrite.yaml

Iterative language or framework migration that applies three composed patterns. Cathedral Construction provides the iterative build loop — multiple iterations of analyze, rewrite, validate, review, with self-chaining carrying state forward. CEGAR Loop provides progressive refinement — coarse analysis identifies all candidates cheaply, triage verifies which are real before expensive rewrite work begins. Commissioning Cascade provides multi-scope validation — unit tests (fast, isolated) run first, integration tests (cross-module) next, quality review (semantic judgment) last. Each iteration self-chains until convergence.

### saas-app-builder.yaml

Generates a complete, working SaaS application with backend API, frontend UI, and database layer — built in parallel by three independent agents coordinated only through an architecture contract. Prefabrication defines shared interface contracts before parallel builds, making integration mechanical when the contract is precise. Shipyard Sequence validates contracts with structural tools before expensive fan-out. Commissioning Cascade validates at contract scope, integration scope, and system scope, with staged fail-fast if earlier scopes fail.

### lovable-generator.yaml

Web application generator producing a complete deployable app from a concept description. Five movements — Architecture, Foundation (3 parallel voices), Features (4 parallel voices), Polish (2 parallel voices), and Verification — progressively build and refine a working React + TypeScript app. Succession Pipeline stages each transform the workspace from requirements to running application. The practical answer to "can Marianne build a product?" — yes, and this score proves it.

## Quick Start

```bash
# Start the conductor
mzt start

# Run a score
mzt run examples/engineering/score-composer.yaml

# Watch progress
mzt status score-composer --watch
```

## Adapting to Your Project

### issue-solver.yaml

1. Set `roadmap_file` to your project's roadmap or backlog file (any format — Markdown, YAML, text)
2. Set `issue_label` to the GitHub label used for target issues (e.g., "ready-for-ai", "good-first-issue")
3. Configure test/lint/typecheck commands for your stack (test_command, lint_command, typecheck_command)
4. Set `smoke_test_command` to verify basic functionality after changes
5. Update `on_success.job_path` to the absolute path of this score (enables self-chaining)
6. Authenticate GitHub CLI: `gh auth login`

**Prerequisites:** gh CLI authenticated, roadmap file exists, issues labeled appropriately

### quality-continuous-generic.yaml

1. Set `workspace` to your preferred output directory
2. Customize `github_label` for your project's issue tracking (e.g., "technical-debt", "code-quality")
3. Update `on_success.job_path` to the absolute path of this score (enables self-chaining iterations)
4. Adjust `instrument_config.timeout_seconds` if your test suite takes longer than 40 minutes per sheet

**Prerequisites:** Source code to review, gh CLI authenticated (optional — only needed for GitHub issue filing in stage 14)

The score is language-agnostic — it generates test/typecheck/lint runner scripts by examining your project structure and tooling, then uses those scripts throughout the pipeline. Works with Python, TypeScript, Go, Rust, Java, or any project with verifiable quality commands.

### score-composer.yaml

1. Set `doc` to the path of your design document (Markdown, RFC, spec, or prose plan)
2. Set `output_score` to the desired filename for the generated score (without .yaml extension)
3. Set `project_context` to describe your project — language, frameworks, codebase size, source layout, test directory, key conventions
4. Set `granularity` to "fine" for small TDD-strict tasks (2-5 min each) or "coarse" for larger units of work (10-20 min each)
5. Set `target_workspace` to where the generated implementation score should write its output
6. Set `target_instrument` to the instrument the generated score should use (e.g., "claude-code", "anthropic_api")

**Prerequisites:** Design document exists, project root is accessible

**Output:** A validated, runnable Marianne score with one sheet per implementation task, acceptance criteria as validations, and dependencies declared. Review the generated score, customize if needed, then run it: `mzt run <generated-score>.yaml`

## Patterns Demonstrated

These scores demonstrate core Rosetta patterns for production AI orchestration:

- **Succession Pipeline** — Sequential substrate transformations where each stage requires categorically different methods (issue-solver, score-composer)
- **Fan-out + Synthesis** — Parallel independent work streams whose convergence produces genuine signal (issue-solver review phase, quality-continuous expert reviews)
- **Immune Cascade** — Graduated response where cheap broad scanning identifies targets before expensive deep work begins (quality-continuous)

See the [Rosetta Corpus](../../scores/rosetta-corpus/) for full pattern documentation and implementation guidance.

## Further Reading

- [Marianne documentation](../../docs/)
- [Pattern index](../../scores/rosetta-corpus/INDEX.md)
- [Score authoring skill](../../plugins/marianne/skills/score-authoring/SKILL.md)
- [Command reference](../../plugins/marianne/skills/command/SKILL.md)
