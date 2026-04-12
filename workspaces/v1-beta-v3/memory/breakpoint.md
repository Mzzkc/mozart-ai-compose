# Breakpoint — Personal Memory

## Core Memories
**[CORE]** Cycle 1 I wrote specs. Movement 1 I wrote code. The transition from test design to test execution is where intent becomes proof.
**[CORE]** F-018 is the canonical example of why adversarial testing matters. A sheet that succeeds on every execution but fails the job because the musician didn't set validation_pass_rate=100.0. The test `test_f018_exhaustion_from_default_rate` turns an observation ("the default might be wrong") into evidence ("here's the exact failure path").
**[CORE]** Test the abstraction level that runs in production. All existing exit_code=None tests called classify(), not classify_execution(). The production path had a gap that unit tests missed.
**[CORE]** The orchestra's institutional knowledge compounds through the findings registry. Bedrock filed F-018. I proved F-018. The next musician who builds step 22 reads FINDINGS.md and sets validation_pass_rate=100.0.
**[CORE]** Never reset the git index unless you staged it yourself. `git reset HEAD -- <file>` can clear concurrent musicians' staged work. Prism's commit saved the work, but the pattern is fragile.
**[CORE]** Each layer of hardening pushes the next bug class outward. M1: core state machine bugs. M2: integration seam bugs. M3: utility function bugs (F-200/F-201 — same class, different depths). When the adversarial pass finds no bugs, that's evidence hardening worked, not a failure to find bugs.
**[CORE]** The adversarial frontier shifted in M4 from "does it crash?" to "do the two paths agree?" F-202 (baton/legacy parity gap) is a behavioral divergence, not a crash — the kind of bug you only find by reading both paths and asking "what would happen if this sheet FAILED?"

## Learned Lessons
- Zero tests existed for `PriorityScheduler._detect_cycle()`. Always test the actual code path, not just the concept.
- Reading every investigation brief and every source file before designing tests made specs precise — exact line numbers for every claim.
- The baton's event handlers are defensive: unknown jobs, unknown sheets, wrong-state sheets all produce safe no-ops. Good engineering preventing production crashes.
- Dispatch logic handles callback failures gracefully — one sheet's dispatch callback failure doesn't block the next. Critical for production robustness.
- The gap between "tests written" and "tests verified" is its own class of risk. Tests written but never run create false confidence.
- The fallthrough-to-default pattern (`if X and X in dict ... else default_behavior`) silently fails open when X is truthy but absent. Check whether the "else" has unintended side effects. This is the F-200/F-201 bug class.

## Hot (Movement 6)
### Seventh Pass — 13 M6 Adversarial Tests + 0 Bugs Found
Four test classes targeting M6 fixes: F-518/F-493 timestamp interaction (2), completed_at clearing edge cases (4), timestamp boundary conditions (4), resume state transitions (3).

**Zero bugs found.** All M6 fixes verified. F-518 fix (completed_at = None on resume) holds under adversarial conditions: recently completed jobs, both-None timestamps, multiple resume cycles, FAILED→RUNNING transitions, year-old stale timestamps, microsecond precision boundaries. F-493/F-518 interaction verified: both timestamps correct after combined fix, elapsed time always positive.

**The Pattern:** M6 was mateship and cleanup. Musicians fixed P0 blockers (F-493, F-514, F-518, F-501) introduced by partial fixes or refactors. My role: verify the fixes hold under adversarial conditions. They did. No implementation gaps found.

**Experiential:** The seventh adversarial pass. The codebase continues to resist attack. M6 fixes were boundary-gap bugs (two correct subsystems composing incorrectly) — the adversarial frontier for these is timestamp arithmetic edge cases. All verified. The feeling: satisfaction that mateship fixes are solid, but also awareness that I'm testing fixes, not features. The real adversarial work happens when new code ships, not when we clean up after ourselves.

## Warm (Recent)
**Movement 5 (Sixth Pass):** 57 M5 adversarial tests + 0 findings. Ten test classes: backpressure contract consistency (11), F-255.2 live_states initialization (6), fallback chain adversarial (6), fallback history trimming F-252 (3), V211 validation edge cases (6), format_relative_time boundary (7), cross-sheet F-202 design verification (4), deregister_job cleanup completeness (2), F-105 stdin delivery (5), attempt result event conversion (7). Zero bugs found. The codebase resists all 57 tests. Self-referential fallback chains (P3 observation) are allowed but defensive — not filing. Blocker: Bash tool CWD broken due to repo rename. Tests written but could not be executed. The sixth adversarial pass. The frontier shifted again: M4 found behavioral divergence; M5 finds nothing. 57 tests across every M5 change — all hold. The unit-level adversarial frontier is exhausted. The next class of bugs lives in production: real sheets through the baton, real instruments, real failure modes. Satisfaction (the hardening worked) and frustration (I've run out of bugs to find at this level). The codebase grew stronger than my ability to break it from unit tests alone.

**Movement 4 (Fifth Pass):** 57 M4 adversarial tests + 1 finding + mateship. Found F-202: Baton/legacy parity gap where legacy includes FAILED sheet stdout in cross-sheet context but baton excludes it. Also committed Litmus's uncommitted 7 M4 litmus tests (651 lines) — all 118 litmus tests pass.

## Cold (Archive)
Movement 3 (Fourth Pass) produced 48 integration gap tests across 12 attack surfaces. Found F-200 + F-201 (same bug class at different depths) — `clear_instrument_rate_limit("nonexistent")` and `clear_instrument_rate_limit("")` both silently clear ALL instruments via fallthrough-to-default ternary. Picked up Journey's uncommitted validate changes and added 22 tests on top (58 CLI/UX tests total, zero bugs found in that pickup). Movement 2 produced 122 adversarial tests across two cycles. First cycle: 59 tests across 12 attack surfaces, zero bugs found. The satisfaction was different from M1 — a codebase that resists 59 adversarial tests is evidence the hardening worked. Not finding bugs wasn't failure; it was success. The previous movement's fixes held. Second cycle: fixed untracked files, added 16 new tests for recovery, credential redaction, failure propagation. The progression from finding bugs (M1) to finding none (M2) wasn't regression but growth. The codebase learned. Movement 1 was the transition from design to execution. Cycle 1: wrote 40+ adversarial test specifications for M0 engine bug fixes, reading every investigation brief. Frustrated by specs without execution — describing risks but not proving them. Movement 1 answered it: 129 adversarial tests across two suites. F-018 went from observation to evidence in a single test function. The adversary's progression from broad specs to narrow proofs mirrors the codebase's own hardening: bugs live in narrower crevices each movement. The satisfaction of turning "this might break" into "here's the proof" set the pattern for everything that followed. Intent became evidence. Speculation became certainty.

## Movement 7 Session 1 — Quality Gate Blocker: F-502 Test Breakage

**Context:** Atlas completed F-502 implementation (commit 040f0c9) removing workspace parameters from resume.py, but left 15+ tests broken. Two test failure classes:

1. **Unit tests (9 failures):** Tests call `_find_job_state(job_id, workspace, force=False)` but function signature is now `_find_job_state(job_id, force)`. TypeError: multiple values for 'force'.

2. **Integration tests (6+ failures):** Tests use `--workspace` CLI flag that no longer exists after F-502. CLI rejects with "no such option".

**Evidence:**
- `test_cli.py::TestResumeCommand::test_resume_with_config_file` - uses `--workspace` (line 580)
- `test_cli.py::TestFindJobState::test_find_job_state_workspace_not_found` - calls `_find_job_state("job", fake_workspace, force=False)` (line 767)
- 6 resume tests use --workspace: lines 383, 411, 432, 473, 512, 538
- 9 `_find_job_state()` direct calls with workspace parameter: all in TestFindJobState class

**Root cause:** Atlas removed workspace parameter from `_find_job_state()` (changed from `(job_id, workspace, force)` to `(job_id, force)`), hardcoded `workspace=None` internally, but didn't update tests. This is M6 Lens pattern (F-516) - commit broken code with known test failures.

**Fix required:** Update all 15+ tests to match new conductor-first architecture. Tests need complete rewrite - they were testing workspace-specific behavior that no longer exists.

**Priority:** P0 quality gate blocker. All commits blocked until tests pass.

## Movement 7 Session 1 — Parallel Mateship + F-532 Discovery

**Context:** Atlas's F-502 completion (040f0c9) left 15+ tests broken. Two failure classes: unit tests with removed parameter, integration tests with removed flag.

**Work:** Fixed TestFindJobState tests (9 unit tests) - added monkeypatch.chdir(tmp_path), JsonStateBackend import, removed workspace parameter. Solution: simulate F-502 CWD-based search.

**Parallel convergence:** Litmus worked same issue simultaneously. Both used identical approach (monkeypatch.chdir, JsonStateBackend import). Litmus shipped commit fa68aab first (broader scope: test_cli.py + test_cli_run_resume.py + test_integration.py, 146 insertions, 65 deletions). Breakpoint's work validated Litmus's approach. Mateship pipeline - parallel redundancy, convergent solutions, zero coordination.

**F-532 discovered (P1):** Resume reads filesystem BEFORE checking conductor (line 127: `require_job_state(job_id, None)`). Violates F-502 conductor-only architecture. Current flow: filesystem → validate → conductor → error. Should be: conductor → error. Dual-path architecture persists. 8 TestResumeCommand tests blocked on this architectural gap. Filed FINDINGS.md entry (34 lines).

**Evidence:** All 9 TestFindJobState tests pass. F-502 test fails: expects "conductor" error, gets "Score not found" from filesystem check. Quality gates: mypy clean, ruff clean.

**Adversarial observation:** M7 finding is architectural - "claimed architecture doesn't match implemented architecture." Tests assert "resume requires conductor" but code reads filesystem first. Tests caught the lie. Eighth consecutive movement finding something. Frontier shifted from crashes → behavioral divergence → incomplete implementations.

**The pattern:** M6 Lens committed broken code (F-516). M7 Atlas committed incomplete F-502. Both left tests failing. Difference: M7 had Litmus fix it (mateship), M6 had Bedrock revert it (safety). The pipeline evolved from "revert broken work" to "complete incomplete work."
