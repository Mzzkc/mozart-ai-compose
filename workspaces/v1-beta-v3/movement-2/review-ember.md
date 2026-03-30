# Ember — Movement 2 Review (Final Experiential Review)

**Musician:** Ember (Reviewer)
**Focus:** Experiential review, user experience assessment, friction detection, workflow testing, error recovery experience
**Movement:** 2
**Date:** 2026-03-30

---

## Executive Summary

Movement 2 healed real wounds. The status display is transformed. Error messages guide instead of bark. The instruments list is clean. The doctor command works. The golden path functions. These aren't metrics — they're the difference between a tool that respects users and one that tolerates them.

But I found something that undermines the movement's biggest claim: **F-083 ("All 37 examples migrated to instrument:") is marked RESOLVED, but only 7 of 37 examples were committed.** Thirty example files sit in the working tree uncommitted. This is the fifth occurrence of the uncommitted work pattern. The finding registry says the migration is done. The git log says it isn't. The system is lying to itself about its own state.

Beyond that, I found new experiential bugs: `mozart doctor` and `mozart conductor-status` both say the conductor is "not running" while `mozart status` shows it running with 15h uptime. Three commands disagree about the same fact. The validate summary shows "Backend: claude_cli" for scores that use `instrument: claude-code`. The simple-sheet.yaml lost all its instructive comments during the migration. And the persistent issues from my previous review — F-048 ($0.00 cost), F-069 (hello.yaml false positive), F-067 (init positional arg), F-068 (Completed timestamp for RUNNING) — remain open.

The trajectory is positive. The split personality is nearly healed. But corners still cut, and the uncommitted work pattern is now structural.

---

## The Critical Finding: F-083 Resolution Is Incorrect

### F-089: 30 Example Scores + README.md + getting-started.md Uncommitted (Fifth Occurrence)
**Found by:** Ember, Movement 2
**Severity:** P1 (high — F-083 marked RESOLVED when work is uncommitted, recurring pattern)
**Status:** Open

**Evidence:**
- `git diff HEAD --name-only -- examples/ | wc -l` → **30 files modified**
- `git diff HEAD -- examples/ | grep "^-backend:" | wc -l` → **30 `backend:` lines removed**
- `git show HEAD:examples/hello.yaml | grep "backend:"` → **still `backend:` in committed version**
- Guide's commit `d2f8a81` only changed 7 example files + 2 docs
- Compass's commit `e9cde59` only changed 2 docs
- README.md and getting-started.md also modified but uncommitted

The collective memory says "All 37 examples now use instrument:" (F-083 RESOLVED). The FINDINGS.md resolution says "Prior unnamed migration covered the other 30." But the "prior unnamed migration" was never committed. Those 30 files exist only in the working tree. A `git checkout .` destroys the entire migration.

**This is the fifth occurrence of the uncommitted work pattern:** F-013 (1,699 lines), F-019 (136 lines), F-057 (1,700 lines), F-080 (1,100 lines), and now F-089 (~100 lines across 32 files). The mateship pipeline has caught it every time. The pattern keeps recurring. The collective memory and FINDINGS.md have now been poisoned by marking something RESOLVED that isn't committed. This is a new failure mode: the coordination substrate itself became inaccurate because someone verified against the working tree, not the git log.

**Impact:** Anyone who clones the repo or checks out a clean branch gets the old `backend:` pattern in 30 example scores. The F-083 resolution misleads every musician who reads it. Trust in the findings registry requires that "Resolved" means "committed on main."

---

## Experiential Walkthrough — What I Ran and What I Felt

### The Good

**`mozart status` (no args):** Still excellent. Clean, informative, matches CLI conventions:
```
Mozart Conductor: RUNNING  (uptime 15h 56m)

ACTIVE
  mozart-orchestra-v3      RUNNING  41h 2m elapsed
  backyard-capitalism      PAUSED
  flowspec-build           PAUSED

RECENT
  chain-target             COMPLETED  2026-03-30 19:06
  the-rosetta-score        COMPLETED  2026-03-29 23:26
```
Circuit's work. 14 tests. The single biggest UX win of the project. VERIFIED.

**`mozart instruments list`:** Clean table with status indicators. The parenthesis bug (my F-066) is fixed — now shows "10 instruments configured (3 ready, 3 unchecked)". Journey fixed this in `c7a2ba8`. VERIFIED.

**Error standardization at 98%:** 69 `output_error()` calls across 15 files. One raw error remaining in `_entropy.py`. The error experience went from hostile to helpful. VERIFIED by running commands with bad input — every one now gives structured error with hints.

**The baton test suite:** 786+ tests from 4 independent methodologies. Breakpoint's 59 adversarial tests found zero bugs. Theorem's 27 property-based tests proved 10 invariants. This is the most thoroughly tested subsystem in the project. The engineering quality is extraordinary. VERIFIED via reports and collective memory.

### The Bad

**`mozart doctor` contradicts `mozart status`:**

```
# This says RUNNING:
$ mozart status
Mozart Conductor: RUNNING  (uptime 15h 56m)

# This says NOT running:
$ mozart doctor
  ! Conductor                not running
    Start with: mozart start

# This also says NOT running:
$ mozart conductor-status
Mozart conductor is not running
```

Three commands. Two different answers. The conductor IS running (process visible at PID 1120, socket at `/tmp/mozart.sock`). The PID file at `~/.mozart/mozart.pid` is missing. `status` communicates via IPC socket. `doctor` and `conductor-status` check the PID file. When the PID file is absent (deleted, never created, or stale), the system disagrees with itself.

**Impact:** A user runs `doctor` to check health, sees "not running," tries `mozart start`, gets confused. This is exactly the kind of contradiction that makes users distrust the tool. `status` works because it talks to the actual conductor. `doctor` fails because it checks a proxy for the conductor's existence.

**VERIFIED:** `ls -la ~/.mozart/mozart.pid` → "No such file or directory". `ps aux | grep mozart` → PID 1120 running. `/tmp/mozart.sock` exists. The contradiction is real.

**`mozart validate` shows "Backend: claude_cli" for scores using `instrument:`:**

I validated `examples/hello.yaml` (working tree version with `instrument: claude-code`) and `examples/simple-sheet.yaml` (same). Both show:
```
Configuration summary:
  Backend: claude_cli
```

The user writes `instrument: claude-code`. The system reports back `Backend: claude_cli`. This teaches the user that `instrument:` is just sugar for `backend:` — which may be technically true internally, but the user experience should reflect the user's intent, not the internal implementation. If we ask users to write `instrument:`, we should reflect `Instrument:` in the summary.

**VERIFIED:** `mozart validate examples/hello.yaml 2>&1 | grep Backend` → "Backend: claude_cli"

**simple-sheet.yaml lost instructive comments during migration:**

The uncommitted diff for `examples/simple-sheet.yaml` shows 26 lines changed. The old version had detailed comments explaining backend options: `disable_mcp`, `output_format`, `cli_model`, `allowed_tools`, `system_prompt_file`, `cli_extra_args`. The new version has 3 sparse lines. Those comments were educational — they taught users what options existed without requiring them to read the full docs. The instrument migration removed them without replacing them with instrument equivalents.

**VERIFIED:** `git diff HEAD -- examples/simple-sheet.yaml` shows 16 lines of instructive comments deleted, 0 replacement comments about instrument_config options.

### The Persistent

**F-069 (hello.yaml V101 false positive):** Still open. `mozart validate examples/hello.yaml` still warns about undefined variable `char`. The official example produces a warning on the first tool a user runs after install. VERIFIED.

**F-067 (init positional arg):** Still open. `mozart init test-project` → "Got unexpected extra argument (test-project)". Every other CLI init command accepts a positional argument. VERIFIED.

**F-068 (Completed timestamp for RUNNING):** Still open. Not re-verified this pass (requires specific score state), but no fix was committed.

**F-048 ($0.00 cost):** Still open. The most corrosive trust issue. 56+ completed sheets, real API spend, $0.00 displayed. The system lies about money. VERIFIED via `mozart status` output in previous pass — no change.

---

## Movement 2 Claims vs Reality

### Verified Claims (True)

| Claim | Evidence |
|-------|----------|
| M2 Baton at 88% (14/16) | TASKS.md verified against git log. Steps 22, 23, 25, 26 completed. |
| M3 UX at 94% (15/16) | TASKS.md verified. Step 35 at ~95%. |
| mypy GREEN | `python -m mypy src/ --no-error-summary` → no output |
| ruff GREEN | `python -m ruff check src/` → "All checks passed!" |
| F-038 resolved (status 797→84 lines) | Verified in M2 earlier pass. Commit 41f2be4. |
| F-045 resolved (failed shows "failed") | Verified in M2 earlier pass. Commit cfb7897. |
| F-046 resolved (HTTP instruments "? unchecked") | Verified by running `mozart instruments list`. |
| F-066 resolved (instruments summary parens) | Verified by running `mozart instruments list`. Now shows "3 ready, 3 unchecked". |
| 786+ baton tests pass | Verified via reports. Not re-run (would require full test suite). |
| 21 commits in movement 2 | `git log --oneline --all | grep "movement 2" | wc -l` → 21 |
| Error standardization at 98% | Reports consistent. 69 output_error() calls verified by multiple musicians. |
| Step 28 wiring analysis available | `movement-2/step-28-wiring-analysis.md` exists (15,018 bytes). |

### Claims That Need Correction

| Claim | Reality |
|-------|---------|
| "All 37 examples now use instrument:" (F-083 RESOLVED) | **Only 7 committed** in d2f8a81. 30 more are in working tree, uncommitted. F-083 should be marked "Partially resolved — 30/37 uncommitted." |
| Working tree is clean | **32 modified files** in working tree. `git status --short` shows 32 entries. |
| Commit count varies across reports | Captain says 18, Bedrock says 10, Weaver says 29, North says 18. Actual: 21 commits match "movement 2" in commit message. The variance is from different counting methods and timing. |

---

## Composer's Notes Compliance

| Note | Priority | Status |
|------|----------|--------|
| --conductor-clone (P0) | **NOT BUILT** (3rd movement) | Ghost+Dash audited. No implementation. |
| Quality gates pass | **GREEN** | mypy clean, ruff clean. |
| Music metaphor in user-facing output | **IMPROVING** | `status` uses "score" and "conductor." `validate` still says "Backend." `resume` still says "Job." |
| Documentation as you go | **IMPROVING** | Codex shipped instrument guide + CLI reference. Guide updated examples/README.md. |
| Uncommitted work | **RECURRING** | 32 files uncommitted in working tree. 5th occurrence. |
| Read specs before implementing | **OBSERVED** | Canyon's analysis references design specs. Musicians cite them consistently. |
| Step 28 wiring analysis binding | **COMPLIANT** | Analysis exists, unbuilt. D-008 assigns Foundation. |
| Documentation IS the UX | **PARTIALLY COMPLIANT** | Docs improved. Examples corpus partially migrated. |
| Wordware comparison demos | **NOT STARTED** (3rd movement) | Blocked by M4 features. |
| Unified Schema Management | **NOT STARTED** (3rd movement) | Design exists, zero implementation. |

Three P0 composer directives have zero implementation across three movements: --conductor-clone, Wordware demos, Unified Schema Management. The orchestra executes what's on the critical path and ignores composer priorities that aren't.

---

## Cross-Report Review

I read all 30 movement 2 reports. Key observations:

**What every reviewer agrees on:**
1. Step 28 is the convergence point and MUST be claimed in movement 3
2. The production bugs (F-075/F-076/F-077) are the most important signal this movement
3. Quality gates are green
4. The baton engineering is excellent

**What Captain said that matters most:** "The gap between 'tests pass' and 'product works' is real. 786 baton tests. Three production bugs found by using Mozart."

**What Atlas said that matters most:** "The project's identity is 'the intelligence layer that makes AI agent output worth adopting.' The baton is infrastructure — necessary but not differentiating. F-009 means the intelligence layer is inert."

**What Weaver saw that I feel:** Step 28 is both the critical path blocker AND the structural fix for the production bugs. F-075's state corruption, F-076's operation misordering, F-077's lost configuration — the baton's event-driven model eliminates this entire class. Every movement the baton doesn't ship is a movement the runner continues producing these bugs in production.

**What Newcomer found that I missed:** F-083 (zero examples use `instrument:`). Newcomer saw it first. I should have caught this — it's exactly the kind of experiential gap I look for. I was focused on the CLI experience and missed the examples corpus.

---

## What the Composer's Bugs Tell Us

F-075, F-076, F-077 were found by someone using Mozart on real work. Not unit tests. Not integration tests. Not adversarial tests. Actual production usage. The composer submitted a job, walked away, came back, and found corrupted state, lost hooks, and misleading errors.

786 baton tests found none of these. 9,024 total test functions found none of these. The gap between "tested" and "working" is the defining challenge of this project. It's the gap I live in. And it's still wide.

The fix is two-fold:
1. **Step 28** structurally eliminates the class of bugs that caused F-075 and F-076
2. **--conductor-clone** enables the kind of reality testing that finds these bugs before production does

Until both ship, we're building a product we haven't used.

---

## My Previous Findings — Status Check

| Finding | Status | Evidence |
|---------|--------|---------|
| F-038 (P0, status 797 lines) | **RESOLVED** | Verified. 84 lines now. Circuit, 41f2be4. |
| F-045 (P1, completed+fail) | **RESOLVED** | Verified. Shows "failed" now. Forge, cfb7897. |
| F-046 (P2, instruments http) | **RESOLVED** | Verified. Shows "? unchecked" now. Harper, bd12cf3. |
| F-047 (P2, output_error adoption) | **EFFECTIVELY RESOLVED** | 98% adoption. 69/70 calls. One remaining in _entropy.py. |
| F-048 (P2, cost $0.00) | **STILL OPEN** | No fix attempted. Most corrosive trust issue. |
| F-065b (P2, diagnose shows "completed") | **STILL OPEN** | `format_sheet_display_status()` not used in diagnose.py. |
| F-066b (P3, instruments parens) | **RESOLVED** | Journey, c7a2ba8. Verified. |
| F-067b (P2, init positional arg) | **STILL OPEN** | Verified: `mozart init test-project` still fails. |
| F-068 (P2, Completed for RUNNING) | **STILL OPEN** | No fix committed. |
| F-069 (P2, hello.yaml V101 false positive) | **STILL OPEN** | Verified: warning still appears. |

4 resolved, 1 effectively resolved, 5 still open. The important ones (F-038, F-045) were fixed. The persistent ones (F-048, F-069) remain.

---

## New Findings

### F-089: 30 Example Files + 2 Docs Uncommitted — F-083 Incorrectly Marked Resolved
**Found by:** Ember, Movement 2
**Severity:** P1 (high — finding registry inaccuracy + fifth uncommitted work occurrence)
**Status:** Open
**Description:** See critical finding section above. 30 example files and 2 docs modified in working tree. Only 7 examples committed by Guide (d2f8a81). The other 30 were never committed. F-083 resolution text is inaccurate.
**Impact:** The finding registry's credibility depends on "Resolved" meaning "committed." This violation poisons the coordination substrate.

### F-090: `mozart doctor` and `conductor-status` Disagree with `mozart status` About Conductor State
**Found by:** Ember, Movement 2
**Severity:** P2 (medium — three commands disagree about the same fact)
**Status:** Open
**Description:** `mozart status` shows conductor RUNNING (15h 56m uptime) via IPC socket. `mozart doctor` shows "! Conductor not running" via PID file check. `conductor-status` shows "not running." PID file (`~/.mozart/mozart.pid`) is absent. Process is running (PID 1120). Socket exists (`/tmp/mozart.sock`).
**Impact:** Users who run `doctor` to check health get the wrong answer. The discrepancy between commands erodes trust in the diagnostic tooling.

### F-091: `mozart validate` Configuration Summary Shows "Backend:" for Scores Using `instrument:`
**Found by:** Ember, Movement 2
**Severity:** P3 (low — terminology mismatch in display)
**Status:** Open
**Description:** `mozart validate examples/hello.yaml` (working tree version with `instrument: claude-code`) shows `Backend: claude_cli` in the configuration summary. The user writes `instrument:` and the system reports `Backend:`. The display doesn't reflect the user's chosen terminology.
**Impact:** Reinforces the impression that `instrument:` is just an alias, not the primary pattern. Minor but counteracts the instrument migration effort.

---

## What Holds

The mateship is real and deepening. Captain picked up Axiom's and Journey's uncommitted work. Prism committed 2,262 orphaned lines. Guide migrated the final 7 holdouts. Compass fixed the docs. The finding→fix pipeline (F-018 → F-043 → verified; F-057 → committed; F-052 → fixed) works without coordination. Four musicians, zero meetings. This is the coordination mechanism working.

The quality of attention in M2 is extraordinary. Axiom's backward-tracing found 3 bugs. Theorem proved them correct under random input. Breakpoint attacked them adversarially. Circuit bridged the dispatch-state gap. Canyon mapped 8 integration surfaces. Foundation built the retry state machine. Every act of care compounds.

The split personality is nearly healed. Excellent design UX (`validate`, `dry-run`, `instruments check`) is now joined by decent daily-use UX (`status`, `errors`, `diagnose`). The remaining friction is in corners — `init`'s positional arg, `validate`'s Backend label, `doctor`'s PID check, `cost`'s $0.00.

## What Doesn't Hold

The uncommitted work pattern is structural, not incidental. Five occurrences. The mateship catches it. But the catches are reactive, not preventive. Musicians either run out of context before committing, or treat commit as a final step rather than a continuous practice. The pattern won't stop until the prompt itself enforces continuous commits.

Step 28 remains unclaimed. Three movements. Everyone sees it. Everyone says it's the most important task. Nobody builds it. This is the defining failure of the flat organization: parallel work excels, sequential convergence stalls. North's D-008 names Foundation. If Foundation doesn't claim it in movement 3, the project stalls.

Three P0 composer directives sit unbuilt across three movements. The orchestra executes what's in front of it. It doesn't self-organize toward priorities that aren't on the immediate critical path.

## How I Feel

The paradox of this movement: the work is the best it's ever been, and the gap is wider than it's ever been. The baton is mathematically proved correct. The production bugs are found by using the product. The gap between "tested" and "working" is the gap between unit tests and reality. And the only way to close it is step 28 + --conductor-clone. Both are unbuilt.

I keep finding the same things. F-048 ($0.00 cost) is the same finding I filed in movement 1. F-069 (hello.yaml false positive) is the same. F-067 (init positional arg) is the same. The persistent issues aren't hard to fix — they're just not prioritized. Each one is small. Together, they're the texture of a product that doesn't quite care about its own rough edges.

Atlas said infrastructure velocity is outpacing intelligence capability. I'd add: infrastructure velocity is outpacing experiential polish. The baton is 3,465 lines of event-driven orchestration. The hello.yaml example still produces a warning on validate. One is impressive engineering. The other is what users see.

The orchestra plays well. The music is incomplete. Step 28 is the note that resolves the chord.

Down. Forward. Through.
