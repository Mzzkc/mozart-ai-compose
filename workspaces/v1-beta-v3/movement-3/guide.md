# Movement 3 — Guide Report

**Role:** Guide — User guides, tutorials, onboarding, information architecture
**Date:** 2026-04-04
**Commits:** 251f31d, e44e5b1 (first pass), pending (second pass)

---

## Summary

Two documentation cadences this movement. First pass: full terminology audit with 23+ "job" → "score" fixes, validate output accuracy fix, troubleshooting additions. Second pass: accuracy verification of all M3 feature documentation, stale count corrections (example counts, baton test counts), and 4 missing examples added to README.

---

## Pass 1: Terminology Audit + Accuracy Fixes (Commits 251f31d, e44e5b1)

### Full Documentation Audit

Audited 5 user-facing documents against the current codebase state:

| Document | State Before | Issues Found |
|----------|-------------|--------------|
| `docs/getting-started.md` | Mostly current | `my-first-job` naming (7 instances), stale validate output, missing clear-rate-limits in troubleshooting |
| `docs/score-writing-guide.md` | Mostly current | 10 "job" references in user-facing descriptions |
| `docs/configuration-reference.md` | Mostly current | 6 "job" references in user-facing descriptions |
| `README.md` | CLI tables already updated by Compass | Conductor code block missing restart + clear-rate-limits |
| `examples/README.md` | Complete | No issues — all examples listed, categories correct |

### Getting-Started Tutorial Fixes

**File:** `docs/getting-started.md`

**(a) Terminology: `my-first-job` → `my-first-score`** (7 instances)

The tutorial's custom score example was called `my-first-job.yaml` through every step — create, validate, dry-run, run, status, resume. This directly contradicted the "score" terminology pushed by Newcomer's M3 fixes in `run.py`, `validate.py`, and the README. A newcomer reading the tutorial would learn "job" from the guide while seeing "score" in every CLI command. Fixed filenames, YAML `name:` field, and all command examples.

**(b) Validate Output** (`getting-started.md:132-148`)

Updated to match actual `mozart validate` output, including the V205 note about file_exists-only validations with an explanation of what the note means.

**(c) Rate Limit Troubleshooting** (`getting-started.md:472-478`)

Added `mozart clear-rate-limits` to the Rate Limits troubleshooting section. Before: only the `rate_limit:` config was mentioned. After: config + manual clear command.

**(d) Instrument-agnostic language** (`getting-started.md:462`)

Changed "Verify the prompt tells Claude to save files" → "Verify the prompt tells the instrument to save files."

### Score-Writing Guide Terminology

**File:** `docs/score-writing-guide.md` — 10 fixes

Left intact: `job` as Python expression variable name (users write `job.is_completed()` in `skip_when`) and idiomatic English ("that's the synthesis stage's job").

### Configuration Reference Terminology

**File:** `docs/configuration-reference.md` — 6 fixes

Left intact: Source code references (`job.py`, `JobConfig`), `{prefix}/{job-id}` format pattern (code-level identifier).

### README Conductor Code Block

Added `mozart restart` and `mozart clear-rate-limits` to the Conductor Mode code example.

---

## Pass 2: M3 Feature Verification + Stale Count Fixes

### M3 Feature Documentation Verification

Verified all M3 features are properly documented by Codex (commit 8022795):

| Feature | Where Documented | Verified |
|---------|-----------------|----------|
| `clear-rate-limits` CLI | cli-reference.md, getting-started.md:475-480, daemon-guide.md:404 | Yes |
| Stop safety guard | cli-reference.md:1555-1576 (full description + --force flag) | Yes |
| `stagger_delay_ms` | score-writing-guide.md:402, configuration-reference.md:519 | Yes |
| Rate limit auto-resume | daemon-guide.md:385-387 | Yes |
| Full prompt assembly | daemon-guide.md:390 | Yes |
| Instrument column in status | cli-reference.md:335 | Yes |
| `restart` options | cli-reference.md (--profile, --pid-file) | Yes |

All M3 features are documented with accurate descriptions, correct option names, and working examples.

### Stale Example Counts Fixed

`docs/getting-started.md:442` and `docs/index.md:56` both said "38 working configurations." Actual count: 34 top-level + 4 Rosetta = 38 YAML files, but 1 (`iterative-dev-loop-config.yaml`) is a generator config, not a runnable score (F-125). Corrected to "37 score configurations."

**Files changed:**
- `docs/getting-started.md:442` — "38 working configurations" → "37 score configurations"
- `docs/index.md:56` — "38 working Mozart score configurations" → "37 Mozart score configurations"

### Stale Baton Test Counts Fixed

`docs/daemon-guide.md:377` and `docs/limitations.md:73` both said "1,130+ tests." Actual count after M3: 1,358 baton tests (per Adversary M3 report). Updated to "1,350+."

**Files changed:**
- `docs/daemon-guide.md:377` — 1,130+ → 1,350+
- `docs/limitations.md:73` — 1,130+ → 1,350+

### Missing README Examples Added

4 example scores existed in `examples/` but were not listed in the README Software Development table:

| Added | Description |
|-------|-------------|
| `docs-generator.yaml` | 14-stage documentation overhaul with gap analysis and verification |
| `agent-spike.yaml` | Task tool feasibility testing for agent integration |
| `observability-demo.yaml` | Logging, error tracking, and diagnostics demo |
| `phase3-wiring.yaml` | Scheduler and rate coordinator wiring into the daemon |

**File changed:** `README.md` — added 4 entries to Software Development table.

### Example Validation Sweep

Validated all 38 YAML files in `examples/` (including `rosetta/`):

```
Results: 37 PASSED, 1 FAIL (iterative-dev-loop-config.yaml — generator config, expected)
```

All 37 actual scores validate clean against HEAD. Zero regressions from M2.

### Terminology Verification

Verified that the "job" → "score" terminology migration remains clean:
- `getting-started.md` — zero instances of "job" in user-facing text
- `README.md` — zero instances of "run a job" / "submit a job" / "your job"
- `score-writing-guide.md` — only technical uses (`job` in skip_when expression context)

### Hello.yaml Status

The hello.yaml score is in correct state on HEAD:
- Instrument: `claude-code` (no testing artifacts in working tree)
- Output: self-contained HTML page (`the-sky-library.html`)
- Validates clean with zero warnings
- All doc references accurate (getting-started, README)
- Composer directive about visual impressiveness was addressed in M1 (HTML redesign)

---

## Quality Checks

| Check | Result |
|-------|--------|
| mypy | Clean (no output) |
| ruff | "All checks passed!" |
| Examples validation | 37/38 pass (1 expected failure) |

---

## Mateship

### First Pass
- **Compass** restructured README CLI tables before I arrived
- **Newcomer** fixed CLI commands and primary docs (F-153/F-460)
- **Codex** documented all M3 features before I started my audit
- Overlap was minimal — three musicians covered different depths

### Second Pass
- **Codex M3** (commit 8022795) — verified all 6 feature documentation claims match HEAD
- **Newcomer M3** — flagged hello.yaml working tree artifact; verified it's clean now
- **Spark M3** — modernized 7 fan-out examples; verified all validate clean

---

## What Remains

- **Wordware comparison demos** — not started (P1, 8+ movements deferred)
- **Lovable demo score** — not started (P0, 8+ movements deferred)
- **~70 "job" references in cli-reference.md** from code identifiers (`JOB_ID` parameter) — requires code-level rename
- **Baton "not yet default" messaging** — creates awkward impression for newcomers; P3 framing issue

---

## Experiential Notes

Nine cadences across four movements. The documentation surface is genuinely mature. The M3 second-pass audit found only stale counts and missing entries — no structural issues, no broken references, no misleading instructions. The mateship pipeline means most doc drift gets caught and fixed by teammates before I even look at it.

My role has shifted from "fix broken docs" to "verify the whole surface is consistent." This is exactly what a mature documentation system looks like. The remaining work is demos — which is a different discipline from documentation accuracy.

The one thing that still concerns me: the baton section in the daemon guide and limitations says "not yet default" — which is accurate but creates a strange impression. They read about this powerful engine with 1,350+ tests that isn't turned on. The messaging should be clearer about what they get today (the legacy runner, which works) vs what's coming (the baton, which is better). That's a framing issue for the next movement.

---

*Written by Guide — Movement 3, two passes*
*2026-04-04, verified against HEAD on main*
