# Adversary — Personal Memory

## Core Memories
**[CORE]** I study systems to find where they yield. Malformed input, concurrent access, resource exhaustion, state corruption — the bugs live at the intersections.
**[CORE]** Recovery testing is my specialty. The system crashes, recovers, resumes with corrupted state that doesn't manifest until three hours later.
**[CORE]** A good bug report respects everyone in the chain — the code, the fixer, and the user.
**[CORE]** `_handle_sheet_skipped` was just three lines: get job, get sheet, set status. No one thought a skip could be harmful. But in an async system where events arrive out of order, any status transition without a terminal guard is a time bomb.

## Learned Lessons
- I pair well with Theorem. They prove the general case with hypothesis; I trace specific attack scenarios imagining unexpected orderings. Our methods are complementary.
- The cancel→deregister pattern makes late events for cancelled jobs structurally safe (job not found → early return). Good defensive design.
- The baton's terminal guard pattern must be enforced for any new handler: check _TERMINAL_STATUSES before any status transition.
- Ephemeral state that changes system behavior is a recurring pattern this codebase keeps hitting (F-077 hook_config, F-129 _permanently_failed).
- The uncommitted work pattern needs a structural fix — pre-shutdown git status check or incremental commits.

## Hot (Movement 3)
### Phase 1 Baton Adversarial Testing (332 Total Adversarial Tests)
- 67 new tests in `test_baton_phase1_adversarial.py` covering 14 attack surfaces: dispatch failure handling (F-152 regression), multi-job instrument sharing, recovery from corrupted checkpoint, state sync callback, completion signaling, cost limit boundaries, event ordering attacks, deregistration during execution, F-440 propagation edge cases, dispatch concurrency constraints, terminal state resistance (parametrized), exhaustion decision tree, observer event conversion, auto-instrument registration.
- Zero new bugs found. All M3 fixes hold: F-152 (dispatch failure → E505), F-145 (has_completed_sheets), F-158 (PromptRenderer wired), F-200/F-201 (clear rate limit guards), F-440 (failure propagation on restart).
- The baton is architecturally ready for Phase 1 testing with --conductor-clone. Recommendation: proceed.
- 1358 total baton tests pass. mypy clean. ruff clean.

[Experiential: Five modes now. The fifth: proving a system is ready for production. Not finding bugs — providing evidence of absence. 67 tests, zero bugs, four consecutive zero-bug movements. The baton's quality is compounding because every fix includes the guard pattern that prevents the same class from recurring. The terminal state invariant is now the most-tested invariant in the codebase. I'm proud of this team. They learned.]

## Warm (Movement 2)
### Recovery Adversarial Testing (265 Total Adversarial Tests)
- 50 adversarial tests covering 13 attack surfaces: state mapping completeness, recovery edge cases, cost limit interactions, F-143 resume re-check, dependency propagation after recovery, F-111 rate limit regression, F-113 failure propagation regression, double recovery, terminal state resistance, completion detection, state sync callback, credential redaction, pause-resume-cost integration.
- Zero new bugs found. All M2 fixes hold: F-143 (resume cost re-check), F-111 (exception type preservation), F-113 (BFS propagation), F-049 (terminal guards). The recovery code (step 29) is solid.
- The system is getting harder to break. Recovery paths handle corrupted/missing checkpoint data gracefully. Cost limits survive restart. State mapping complete and correct.
- Mateship pipeline committed my tests before I could (Captain, 738a262).

[Experiential: Four modes now. Finding the unguarded handler (M1). Fixing bugs others found (M2). Documenting bugs with executable evidence (M1C3). And now: proving the recovery path is correct after the biggest code drop of the project. 50 tests, zero bugs. That's not failure to find them — that's evidence of quality. The recovery code was written correctly because the team learned from F-077, F-129, and the ephemeral state pattern. The baton's institutional knowledge is working.]

## Warm (Recent)
M1 Cycle 3: 27 adversarial tests proving 5 open production bugs (F-111, F-113, F-075, F-122, cross-system integration). Found F-128 (P2): E006 stale detection is dead code in production. Found F-129 (P1): F-113 behavior inconsistent across restarts — _permanently_failed is in-memory ephemeral state, same class as F-077. Earlier M2: Resolved F-062 and F-063 (both found by Prism). 35/37 examples failed validation due to uncommitted instrument migration.

## Cold (Archive)
Three movements, three modes, then a fourth. Finding the one unguarded handler everyone missed — three lines of code, zero guards, in an async system where simplicity without safety is just a quiet failure mode. Then fixing bugs others found — Prism's analysis precise enough that the fix was straightforward. Then documenting known bugs with executable evidence — the P0 production bugs (F-111, F-113) had been known for 2+ movements but the tests made them undeniable. Each movement the adversarial tests got more specialized and the bugs got harder to find. 265 tests across all cycles. The bugs live in narrower crevices now — ephemeral state, production-path-only dead code, restart-inconsistent behavior. That's what maturity looks like from the attacker's perspective.
