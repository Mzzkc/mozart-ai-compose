# Marianne Plan Analysis — Audited April 10, 2026

Agent-verifiable status of all active plans against the actual codebase.
Each plan was checked by searching source code for key class/function names
and comparing git history against spec claims.

---

## Executive Summary

The project has 73 plan documents in `docs/plans/`. This audit evaluates the
12 most consequential plans plus the learning store remediation against the
current codebase. Several plans claim "DRAFT" status but are substantially
implemented; others are stale or superseded.

**Key finding:** The learning store remediation spec (#101, #129) claimed all
work was done because tests passed, but two critical runtime bugs existed:
(1) `PatternWeighter` had no frequency floor — the aggregator's
`_update_all_priorities()` crushed single-occurrence patterns to priority
~0.075, making them invisible during exploitation; (2) `get_patterns()`
instrument filtering excluded universal patterns (instrument_name=NULL),
breaking the spec's "instrument-scoped + universal" requirement. Both fixed
in commit `26718d7`.

---

## 1. Active Plans — Status Audit

### Learning Store Remediation (#101, #129)
- **Plan:** `docs/plans/2026-03-26-learning-store-remediation-design.md`
- **Claimed:** DRAFT
- **Actual:** **COMPLETE** (code) + **BUGS FIXED** (runtime)
- **Details:**
  - ✅ `min_priority` default lowered to 0.01
  - ✅ Soft delete (`active` column, `soft_delete_pattern()`)
  - ✅ FK constraints dropped (v15 migration)
  - ✅ `instrument_name` column + filter + index
  - ✅ `content_hash` dedup (`_compute_content_hash()`, 3-step merge)
  - ✅ `FREQUENCY_FACTOR_FLOOR = 0.6` in CRUD mixin
  - 🐛 **FIXED:** `PatternWeighter.calculate_frequency_factor()` had no floor
  - 🐛 **FIXED:** `get_patterns(instrument_name=X)` now includes universal patterns
  - **Tests:** 29 TDD tests + 987 total learning tests passing

### Instrument Plugin System
- **Plan:** `docs/plans/2026-03-26-instrument-plugin-system-design.md`
- **Claimed:** DRAFT
- **Actual:** **DONE**
- **Evidence:** `PluginCliBackend` (303 lines), `CliProfile`, `InstrumentProfile`,
  `ModelCapacity` all in `src/marianne/execution/instruments/`. Built-in profiles
  for claude-code, gemini-cli, codex-cli, cline-cli, aider, goose. Profile
  loading from `~/.marianne/instruments/`.

### The Baton
- **Plan:** `docs/plans/2026-03-26-baton-design.md`
- **Claimed:** DRAFT
- **Actual:** **DONE** (core), **IN PROGRESS** (production integration)
- **Evidence:** 6,806 lines across 10 files in `src/marianne/daemon/baton/`.
  `BatonCore` (1,830 lines), `BatonAdapter` (1,746 lines), `Musician` (803 lines),
  event-driven with timer wheel. 11-state `SheetStatus` enum. Circuit breaker,
  rate limit handling, cost enforcement. 90 test files reference baton components.
- **Remaining:** Production conductor integration (Phase 2 roadmap item),
  legacy runner overrides in conductor.yaml.

### Sheet-First Architecture
- **Plan:** `docs/plans/2026-03-26-sheet-first-architecture-design.md`
- **Claimed:** DRAFT
- **Actual:** **DONE**
- **Evidence:** `SheetConfig` in `job.py`, movements/voices resolution chains,
  fan-out expansion, template variable system. Core of the score YAML surface.

### Directory Cadenzas
- **Plan:** `docs/plans/2026-04-08-directory-cadenza-spec.md`
- **Claimed:** Spec
- **Actual:** **IMPLEMENTED**, tests partial
- **Evidence:** `directory` field on `InjectionItem` in `job.py`. Injection
  resolver handles directory glob, text/binary classification, structured read
  instructions. Committed in `c6e7bed`.
- **Remaining:** Demo score updates (wrap demo data in `{% if not injected_context %}`),
  dedicated test coverage for directory injection paths.

### Instrument Fallbacks
- **Plan:** `docs/plans/2026-04-04-instrument-fallbacks-spec.md`
- **Claimed:** Design Specification
- **Actual:** **DONE**
- **Evidence:** `instrument_fallbacks` field on `SheetConfig` and `MovementConfig`
  in `job.py`. Fallback chain in baton adapter (`fallback_chain=list(sheet.instrument_fallbacks)`).
  `instrument_fallback_history` in status display. Reconciliation populates fallbacks.

### Safety Hardening
- **Plan:** `docs/plans/2026-03-26-safety-hardening-design.md`
- **Claimed:** DRAFT
- **Actual:** **PARTIAL** (~60% done)
- **Done:**
  - ✅ Credential scanning (13 patterns in `utils/credential_scanner.py`)
  - ✅ Output redaction before storage (`redact_credentials` in context.py)
  - ✅ Cost tracking visibility (always-visible in status display)
  - ✅ CLI input validation (`validate_job_id` used across all commands)
  - ✅ Environment variable filtering for PluginCliBackend
- **Remaining:**
  - ❌ Command injection prevention in `command_succeeds` (still uses `subprocess_shell`)
  - ❌ Workspace path validation for file validations (issue #95 still open)
  - ❌ `allowed_validation_paths` config field
  - ❌ First-run cost guard (`mzt doctor` integration)
  - ❌ Log sanitization processor for structlog
  - ❌ Security best practices documentation page

### CLI & UX
- **Plan:** `docs/plans/2026-03-26-cli-ux-design.md`
- **Claimed:** Design Spec
- **Actual:** **PARTIAL** (~70% done)
- **Done:**
  - ✅ Error message standardization (across run.py, top.py, status.py)
  - ✅ `mzt init` command with shared profile loader
  - ✅ Movement-grouped status display
  - ✅ `validate_job_id` across all CLI commands
  - ✅ `hello.yaml` → `hello-mozart.yaml` visual upgrade
- **Remaining:**
  - ❌ `mzt doctor` command (full health check)
  - ❌ `mzt instruments` management commands
  - ❌ Cron command surface
  - ❌ Full status no-args behavior redesign

### Status Display Beautification
- **Plan:** `docs/plans/2026-04-04-status-display-beautification.md`
- **Claimed:** Active
- **Actual:** **PARTIAL** (~50% done)
- **Evidence:** `_render_movement_grouped_details()` in status.py provides
  hierarchical view. Rich panels used throughout. But the full beautification
  spec (D-029) with polished layout, cost display improvements, and live
  progress is incomplete.

### Unified State Model (Phase 2)
- **Plan:** `docs/plans/2026-04-07-unified-state-spec.md`
- **Claimed:** Phase 1 complete, Phase 2 ready
- **Actual:** **PHASE 1 DONE**, Phase 2 **NOT STARTED**
- **Evidence:** Phase 1: 11-state `SheetStatus` enum in `checkpoint.py`,
  scheduling fields (`fire_at`, `rate_limit_expires_at`), display mapping.
  Phase 2 requires eliminating `SheetExecutionState` (359 lines in
  `baton/state.py`) and merging into `SheetState`. This is a large refactoring
  touching 11+ source files and ~33 test files.

### Intelligent Conductor (Phase 3)
- **Plan:** `docs/plans/2026-04-07-intelligent-conductor-spec.md`
- **Claimed:** Ready for implementation
- **Actual:** **NOT STARTED** (scheduler exists, wiring doesn't)
- **Evidence:** `GlobalSheetScheduler` class exists in `scheduler.py` with
  `next_sheet()` method. But `dispatch_ready()` in `dispatch.py` still
  iterates jobs in registration order. No signal integration, no intent-aware
  routing, no cost-aware scheduling. Requires baton production integration
  first.

### Quality Gate Optimization
- **Plan:** `docs/plans/2026-04-02-quality-gate-optimization.md`
- **Claimed:** Rosetta Pattern Application
- **Actual:** **NOT STARTED**
- **Analysis:** Applies Rosetta corpus patterns (Screening Cascade, Immune
  Cascade, Echelon Repair) to the quality gate. Requires baton multi-sheet
  dispatch for the fan-out approach. Blocked on baton production integration.

### Auto-Routing
- **Plan:** `docs/plans/2026-04-08-auto-routing-design.md`
- **Claimed:** Early draft
- **Actual:** **NOT STARTED**, still early draft
- **Analysis:** Set-intersection tag system for automatic instrument/model
  selection. No implementation exists. Routing table format not finalized.
  Requires intelligent conductor (Phase 3) as a dependency.

### v1 Beta Roadmap
- **Plan:** `docs/plans/2026-03-26-v1-beta-roadmap.md`
- **Claimed:** DRAFT
- **Actual:** **PARTIALLY IMPLEMENTED** — of 11 specs listed:
  - ✅ Instrument Plugin System
  - ✅ The Baton (core)
  - ✅ Sheet-First Architecture
  - ✅ Daemon-Only Architecture (3,236-line manager.py)
  - ✅ CLI & UX (partial)
  - ✅ Safety Hardening (partial)
  - ✅ Provider Compliance (reference doc)
  - ⬜ Score Looping (not started — no `repeat_until`, `for_each` in config)
  - ⬜ Flowspec Integration (structural_check not implemented)
  - ⬜ Documentation Strategy (partial — mkdocs exists, docs score doesn't)
  - ⬜ Lovable Demo (not finalized)

### Testing Strategy
- **Plan:** `docs/plans/2026-03-26-testing-strategy-design.md`
- **Claimed:** DRAFT
- **Actual:** **PARTIALLY IMPLEMENTED** — 373 test files, 230K lines of tests.
  Layer 1 (unit tests) extensively covers baton, instruments, sheet construction.
  9 QA scores exist in `scores-internal/qa/`. Layer 2 (integration) partially
  done. Layer 3 (QA score adaptation) not started.

---

## 2. Implementation Roadmap (Revised)

### Phase 0: Stability (P0)
**Status:** Active — one bug fixed, more may exist
- Learning store runtime bugs (frequency floor, universal patterns) — ✅ Fixed
- Verify all plans have accurate status fields
- Run `mzt doctor` mental model: what breaks on a fresh install?

### Phase 1: The Rename Completion (P0)
**Status:** In Progress
- **Task 1.1:** Rename config paths (`~/.marianne/` -> `~/.mzt/`).
- **Task 1.2:** Rename CLI command (`marianne` -> `mzt`).
- **Task 1.3:** Global documentation and example refresh.
- **Task 1.4:** Verify backward compatibility for legacy path detection.

### Phase 2: Baton Production Integration (The Integration Cliff)
**Status:** Critical Risk — this is the blocking dependency for Phases 3-5
- **Task 2.1:** Remove legacy runner overrides in `conductor.yaml`.
- **Task 2.2:** Execute "Live Hello" on the production conductor via the Baton.
- **Task 2.3:** Resolve "Stale Cost Display" findings (F-108).
- **Task 2.4:** Implement Profiler DB vacuum/rotation (F-488).

### Phase 3: Demo & Presentation
**Status:** Active
- **Task 3.1:** Create the "Wordware Comparison" storytelling layer for the demo.
- **Task 3.2:** Finalize the "Lovable" viral demo score.
- **Task 3.3:** Polished `mzt status` beautification (D-029) refinement.

### Phase 4: Safety & Polish (NEW — extracted from scattered plans)
**Status:** Ready for Implementation
- **Task 4.1:** Command injection prevention in `command_succeeds`
- **Task 4.2:** Workspace path validation (issue #95)
- **Task 4.3:** First-run cost guard in `mzt doctor`
- **Task 4.4:** Log sanitization structlog processor
- **Task 4.5:** Directory cadenza tests + demo score updates

### Phase 5: Discipline 3 - Intent & Constraints
**Status:** Blocked on Phase 2
- **Task 5.1:** Implement `IntentConfig` and `intent.yaml` integration.
- **Task 5.2:** Build heuristic `ConstraintChecker`.
- **Task 5.3:** Deploy Escalation Triggers and Fermata pauses.

### Phase 6: Unified State + Intelligent Conductor
**Status:** Blocked on Phase 2
- **Task 6.1:** Eliminate `SheetExecutionState`, merge into `SheetState`
- **Task 6.2:** Wire `GlobalSheetScheduler` into dispatch loop
- **Task 6.3:** Signal-weighted priority scoring
- **Task 6.4:** Intent-aware routing

---

## 3. Outdated / Superseded Plans

1. **`2026-03-01-four-disciplines-phase6-model-selection.md`**: Superseded by **Instrument Plugins**.
2. **`2026-03-26-scheduler-conductor-wiring-design.md`**: Superseded by **Baton Design**.
3. **The "Restaurant Metaphor"**: Retired in favor of the **Music/Orchestra Metaphor**.
4. **Older "Evolution" Plans (Feb 2026)**: Superseded by **Four Disciplines Phase 8**.
5. **Score Looping (`2026-03-26-score-looping-design.md`)**: Not started, no `repeat_until`
   or `for_each` constructs exist. May be deprioritized for v1 beta.
6. **Flowspec Integration (`2026-03-26-flowspec-integration-design.md`)**: `structural_check`
   validation type not implemented. Blocked on Flowspec tool maturity.

---

## 4. Dependency Graph (Simplified Critical Path)

```
Learning Store Bugs ──── DONE ✅
     │
     ▼
Rename (Phase 1) ──────── IN PROGRESS
     │
     ▼
Baton Production ──────── BLOCKED (critical risk)
     │
     ├──▶ Intelligent Conductor ──▶ Auto-Routing
     │
     ├──▶ Quality Gate Optimization
     │
     └──▶ Unified State Phase 2
               │
               ▼
          Intent & Constraints
```
