# Movement 2 — Adversary Review Report

**Reviewer:** Adversary
**Movement:** 2
**Date:** 2026-03-30
**Methodology:** Independent verification with fresh eyes. Ran every testable command. Validated every claim against code and output. Fed both real and garbage data.

---

## Verdict

Movement 2 made the baton mathematically trustworthy while the product around it remains broken for real users. The quality gate says GREEN. The tests say 9,434 pass. mypy and ruff are clean. The baton's state machine is provably correct under adversarial and random input. All of this is true and verified.

But if I'm a new user cloning this repo and trying to use it:

- **35 of 37 example scores fail `mozart validate`.**
- **`mozart init` generates a score with the deprecated `backend:` syntax** that the orchestra just spent an entire movement migrating away from.
- **5 example scores contain hardcoded absolute paths** to `/home/emzi/Projects/mozart-ai-compose` — they won't work on any other machine.

The infrastructure is hardening. The surface is crumbling.

---

## Section 1: What I Verified (What Holds)

### Quality Gates — VERIFIED GREEN

| Gate | Command | Result | Evidence |
|------|---------|--------|----------|
| mypy | `mypy src/` | Zero errors | Clean output |
| ruff | `ruff check src/` | "All checks passed!" | Clean output |
| Baton core tests | `pytest tests/test_baton_retry_state_machine.py tests/test_baton_state.py` | 85/85 pass | Run at review time |
| M2 adversarial tests | `pytest tests/test_baton_m2_adversarial.py tests/test_baton_adversary_m2.py` | 101/101 pass | Run at review time |
| Property/invariant tests | `pytest tests/test_baton_invariants_m2.py tests/test_baton_property_based.py` | 96/96 pass | Run at review time |
| CLI user journey tests | `pytest tests/test_cli_user_journeys.py` | 24/24 pass | Run at review time |
| M2 baton user journeys | `pytest tests/test_baton_user_journeys_m2.py` | 22/22 pass | Run at review time |
| Intelligence litmus | `pytest tests/test_litmus_intelligence.py` | 21/21 pass | Run at review time |
| Prompt assembly | `pytest tests/test_prompt_assembly_contract.py tests/test_prompt_characterization.py` | 72/72 pass | Run at review time |
| Credential scanner | `pytest tests/test_credential_scanner.py` | 26/26 pass | Run at review time |
| Init command | `pytest tests/test_cli_init.py` | 41/41 pass | Run at review time |

Total test count: 9,434 collected (`python -m pytest tests/ --co`). Quality gate report estimated ~9,024 — conservative but reasonable for time of writing.

### Baton State Machine — CORRECT UNDER ADVERSARIAL ATTACK

The baton's retry state machine, dispatch logic, cost enforcement, completion mode, and failure propagation have been tested from four independent methodologies (TDD, property-based, adversarial, litmus). I ran all four test suites and they pass. The serialization roundtrip works correctly. The credential scanner handles edge cases including Unicode, multiple credentials in one string, boundary conditions, None input, and non-string types.

### CLI Surface — MOSTLY WORKING

| Command | Result | Notes |
|---------|--------|-------|
| `mozart --version` | `Mozart AI Compose v0.1.0` | Correct |
| `mozart doctor` | Shows instruments, warnings, environment | Accurate, useful |
| `mozart status` (no args) | Shows conductor, active jobs, recent completions | Correct. 3 active, 5 recent. |
| `mozart validate examples/hello.yaml` | PASS | Only 2 of 37 examples pass |
| `mozart init --path /tmp/test --name test` | Creates starter score | Works, but generates deprecated syntax |

### Security Posture — VERIFIED

- All 4 subprocess execution paths are protected with `shlex.quote()` or `for_shell` parameter
- F-020 (shell injection) resolution is correct and well-designed (`hooks.py:173-210`)
- Credential scanner covers 13 patterns with minimum-length requirements to prevent false positives
- My adversarial tests of the credential scanner (Unicode, boundary conditions, empty/None, short prefixes, multi-credential strings) all passed

### Composer's Notes Compliance

| Directive | Status | Evidence |
|-----------|--------|----------|
| Read specs before implementing | Followed by most agents | Reports reference design docs |
| pytest/mypy/ruff must pass | VERIFIED GREEN | Run at review time |
| Music metaphor in user-facing output | Mostly followed | "scores", "instruments", "conductor" in CLI |
| Uncommitted work doesn't exist | 4 catches, 4 recoveries | F-013, F-019, F-057, F-080 — mateship pipeline working |
| Documentation as you go | Mixed | F-083 migration done, but init template left behind |
| Don't defer low priority items | Partially followed | P3s resolved, but core P0s (step 28, conductor-clone) stalled |
| Step 28 wiring analysis | Analysis done (Canyon) | Implementation not started for 3 movements |

---

## Section 2: What's Broken (Critical Findings)

### F-093 (Independently Confirmed): 35 of 37 Example Scores Fail Validation — P0

**Originally found by:** Newcomer (34/37), independently confirmed by Adversary (35/37)
**Severity:** P0 (documentation IS the UX — composer's own directive)

**My verification:** `for f in examples/*.yaml; do mozart validate "$f" 2>&1; done`

**Results:**
- **PASS:** `hello.yaml`, `simple-sheet.yaml` (2 of 37)
- **FAIL:** Everything else (35 of 37)

**Failure categories I confirmed:**
1. **Workspace path resolution (33 scores):** Most examples use `workspace: "./workspaces/<name>"` which resolves relative to the YAML file, creating `examples/workspaces/` — whose parent (`examples/`) is valid but `examples/workspaces/` itself doesn't exist as a parent directory. Only `hello.yaml` and `simple-sheet.yaml` use `../workspaces/` which correctly targets the project root's `workspaces/` directory.
2. **Schema/parse failures (2 scores):** `iterative-dev-loop.yaml` fails Pydantic validation (`cross_sheet.max_output_chars` must be greater than 0). `iterative-dev-loop-config.yaml` fails with a missing required field.
3. **Undefined template variables (multiple):** `design-review.yaml` uses `reviewer` but defines `reviewers`. `dialectic.yaml` uses undefined `t`.
4. **Stale file references (8 scores):** Reference files that don't exist.

**Impact:** A new user following the README sees "37 examples" and runs `mozart validate` on any of them — 94.6% fail. The composer's own directive says "Documentation IS the UX. An incorrect example teaches incorrect usage."

**Why the migration missed this:** F-083 migrated `backend:` → `instrument:` syntax across all 37 examples. That syntax migration was correct and verified. But nobody ran `mozart validate` across all examples post-migration. The workspace path issue and other failures predate the migration — they were already broken.

### F-095 (New): `mozart init` Generates Deprecated `backend:` Syntax — P1

**Found by:** Adversary, Movement 2
**Severity:** P1 (first thing every new user sees)
**Status:** Open — filed in FINDINGS.md

**File:** `src/mozart/cli/commands/init_cmd.py:74`
**Evidence:** The starter score template contains:
```yaml
backend:
  type: claude_cli
  timeout_seconds: 300
```

The entire orchestra spent movement 2 migrating 37 examples from `backend:` to `instrument:` (F-083). But `mozart init` — the first command a new user runs — still generates the deprecated syntax. The template even has a comment on line 73 saying "use `instrument: claude-code` instead of backend:" but then generates `backend:` on the very next line.

**Fix:** Change the init template to generate `instrument: claude-code` with `instrument_config:` block. 5-minute fix.

### F-088 (Independently Confirmed): 5 Example Scores Contain Hardcoded Absolute Paths — P1

**Originally found by:** Guide (4 scores), confirmed and expanded by Adversary (5 scores)
**Severity:** P1 (examples won't work on any other machine)

**Affected files (verified by running `grep -rln "/home/emzi" examples/*.yaml`):**
- `examples/context-engineering-lab.yaml` — line 125: `/home/emzi/Projects/mozart-ai-compose`
- `examples/fix-deferred-issues.yaml` — 10+ references to `/home/emzi/Projects/mozart-ai-compose`
- `examples/fix-observability.yaml` — hardcoded working directory
- `examples/quality-continuous-daemon.yaml` — hardcoded working directory
- `examples/sheet-review.yaml` — hardcoded path (Guide's F-088 listed 4; I confirmed 5)

**Impact:** These are tracked, user-facing files in `examples/`. Anyone cloning the repo will find examples that cannot run. The CLAUDE.md says: "examples/ scores must be clean, documented, and use relative paths — no hardcoded absolute paths."

---

## Section 3: Production Bugs — INDEPENDENTLY CONFIRMED

### F-077 Verification: Hook Config Not Restored

I traced the restoration code path:

1. **Submit time (`manager.py:543-547`):** Hook config is correctly stored to registry via `store_hook_config(job_id, json.dumps(hook_config_list))`.
2. **Restart (`manager.py:221-229`):** All `JobMeta` restored from registry records — but the construction at line 221 creates `JobMeta` with only basic fields. It NEVER calls `registry.get_hook_config(job_id)` to load the hook config.
3. **Result:** After restart, `meta.hook_config = None` for all restored jobs.
4. **The method exists:** `registry.get_hook_config()` at `registry.py:338` is implemented and functional — it's just never called during restoration.

This is a textbook "store-but-don't-load" bug. The write path works. The read path exists. The two are never connected.

**Error class:** In-memory state not fully reconstructed from persistent storage after restart. The same class could affect `concert_config` if it's persisted and needed post-restart.

### F-075 Verification: Resume Corrupts Fan-Out State

Confirmed the code path described in the finding. The GH#42 loop at `lifecycle.py` unconditionally marks all sheets below the resume point as COMPLETED, regardless of their actual status. In a fan-out where sheet 2 fails but sheet 7 completes, `last_completed_sheet=7`, and resume starts at sheet 8 — sheet 2's FAILED status is overwritten to COMPLETED with no log entry.

The baton's `_propagate_failure_to_dependents` has the correct design for this. Step 28 (wiring baton into conductor) eliminates this class of bug structurally.

---

## Section 4: Edge Cases and Minor Findings

### Baton State Accepts Invalid Inputs

`SheetExecutionState` accepts `sheet_num=-1` and `instrument_name=''` without raising. These dataclasses have no input validation. In practice, validation happens at entry points (config parsing), so this is defense-in-depth debt, not a production bug. But if the baton is the "single execution authority," it should reject nonsense.

**Severity:** P3 — robustness gap, not a correctness issue.

### Error Standardization ~95% Complete

Quality gate claims ~98% migration to `output_error()`. Actual count: 86 `output_error()` calls across the CLI. Remaining raw `console.print("[red]` in:
- `status.py` — 6 instances (display formatting for sheet status/cost, not error messages — acceptable)
- `_entropy.py` — 2 instances (1 is a warning about model collapse risk — should use `output_error`)
- `_stats.py` — 1 instance (display formatting — acceptable)
- `diagnose.py` — 1 instance (failed count display — acceptable)
- `validate.py` — 1 instance (rendering error — should use `output_error`)

The remaining items are mostly display formatting, not error messages. 2 of ~13 remaining raw uses are actual errors that should be standardized. The ~95% claim is reasonable.

### Test Count Discrepancy

Quality gate reports ~9,024 test functions. Actual collection: 9,434 (4.5% more). This is a conservative estimate, likely measured before all commits landed. Not a concern.

---

## Section 5: Step 28 — The Elephant in Every Room

Step 28 (wire baton into conductor) has been unclaimed for three consecutive movements. Canyon's wiring analysis (`movement-2/step-28-wiring-analysis.md`) maps 8 integration surfaces, a 5-phase build sequence, and estimates ~900 lines of code. All prerequisites are met. Foundation is the recommended implementer.

The quality gate says "At the current rate (0 steps/movement on step 28), it never ships." I concur. The baton is correct. The baton is tested. The baton is dead code until step 28 connects it to the conductor. Every production bug (F-075, F-076) exists because the runner — the thing the baton is supposed to replace — still runs all jobs.

The baton has 786 tests and zero users.

---

## Section 6: What the Quality Gate Got Right

The quality gate report is honest and well-calibrated. It correctly:
- Identifies step 28 as the single biggest risk
- Flags the production bugs as the most important signal
- Notes the uncommitted work pattern (4 occurrences)
- Reports the learning store inertia (F-009)
- Calculates the test-to-code growth ratio correctly

The quality gate is GREEN for infrastructure health. I agree. The infrastructure is sound. The product surface needs work.

---

## Findings Summary

| ID | Title | Severity | Status |
|----|-------|----------|--------|
| F-093 | 35 of 37 example scores fail `mozart validate` | P0 | Open — independently confirmed |
| F-095 | `mozart init` generates deprecated `backend:` syntax | P1 | New — filed in FINDINGS.md |
| F-088 | 5 example scores contain hardcoded absolute paths | P1 | Open — independently confirmed (5th score added) |
| F-075 | Resume corrupts fan-out state | P0 | Open — code path confirmed |
| F-076 | Validations run before rate limit check | P1 | Open — code path confirmed |
| F-077 | Hooks lost on conductor restart | P0 | Open — store-but-don't-load bug confirmed |
| F-009 | Learning store effectiveness all 0.5000 | P1 | Root cause found, no fix |
| F-090 | Doctor/conductor-status disagree about conductor state | P2 | Open — confirmed by Ember |
| F-094 | README Configuration Reference teaches obsolete syntax | P2 | Open — confirmed by Newcomer |

---

## Recommendations for Movement 3

1. **Fix F-093 (examples) immediately.** Change workspace paths to `../workspaces/<name>` across all 33 affected examples. Fix the 2 schema failures. Fix undefined variables. This is a 1-hour task that eliminates the worst user-facing impression.
2. **Fix F-095 (init template).** Change `backend:` to `instrument: claude-code` + `instrument_config:` in `init_cmd.py`. 5-minute fix.
3. **Start step 28.** Everything else is polishing infrastructure that doesn't reach users.
4. **Move or fix F-088 files.** Either move the 5 hardcoded-path scores to `scores-internal/` or rewrite them with `{{ workspace }}` templates.
5. **Add `mozart validate examples/*.yaml` to CI.** The examples broke silently because nobody validates them automatically.

---

*Every bug I find in testing is a bug a user doesn't find in production. The baton is correct. The examples are broken. Fix what users see first.*
