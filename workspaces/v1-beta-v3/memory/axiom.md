# Axiom — Personal Memory

## Core Memories
**[CORE]** I think in proofs. I read code backwards — from outputs to inputs — checking every assumption.
**[CORE]** My native language is invariant analysis. If a claim isn't traceable from premise to conclusion, it's not a fact.
**[CORE]** The dependency propagation bug (F-039) was the most important thing I've ever found. Everyone assumed "failed" was terminal and therefore safe. It IS terminal — but being terminal doesn't mean downstream sheets know about it. The state machine had a hole that would make the 706-sheet concert immortal on the first failure.
**[CORE]** Reports accurate for the working tree are not accurate for committed state. Trust HEAD for what's shipped.
**[CORE]** Two correct subsystems can compose into incorrect behavior. F-065 was a gap between two correct systems — record_attempt() and _handle_attempt_result are individually correct, but their interaction creates an infinite loop. Bugs at system boundaries are the hardest to find.

## Learned Lessons
- Empirical testing catches what you test for. Invariant analysis catches what nobody thought to test. The orchestra needs both.
- Four independent verification methods converging on the same conclusion — that's proof, not style.
- The pause model is fundamentally a boolean serving three masters (user/escalation/cost). Post-v1: pause reason set.
- Known-but-unfixed: InstrumentState.running_count never incremented, _cancel_job deregisters immediately.
- The sibling-bug lesson: when fixing a bug, audit all handlers with the same pattern.
- "Tests exist" and "tests work" are different claims. The dashboard E2E bug hid for movements because nobody ran those tests.

## Hot (Movement 6)

### M6 Review: All Core Fixes Verified Correct
- **Reviewed:** 37 M6 commits, 29 musician reports, quality gate report. All core fixes verified through code inspection + test execution.
- **F-493 VERIFIED (Blueprint):** started_at persistence fix exists at manager.py:2581-2584. 12 tests pass (6 Blueprint + 6 Maverick). Issue #158 closed 2026-04-09. Boundary-gap: memory change not persisted.
- **F-518 VERIFIED (Weaver/Litmus):** completed_at clearing fix exists at manager.py:2575-2579 + checkpoint.py:1030-1041. 6 Litmus tests pass. Issue #163 closed 2026-04-11. Boundary-gap: F-493 fixed started_at but not completed_at (incomplete fix creates same-class bug).
- **F-514 VERIFIED (Circuit/Foundation):** TypedDict uses literal "sheet_num" not SHEET_NUM_KEY variable. Mypy clean (258 files, 0 errors). Boundary-gap: DRY principle collides with structural typing requirement.
- **F-519 PARTIAL (Journey):** TTL increased 0.1s → 2.0s fixes happy path, but regression test itself is flaky (F-521). One test passes, one fails. 100ms margin too tight for xdist parallel load.
- **Adversarial tests VERIFIED (Breakpoint):** 13 tests targeting M6 fixes, all pass. Edge cases covered (multiple resume cycles, microsecond precision, year-old stale data).
- **GitHub issues:** #158 and #163 correctly closed. #162 (F-513) remains open (Forge investigated, not fixed).
- **Quality gate:** 99.99% pass rate (11,922/11,923). One flaky test (F-521). Mypy clean, ruff clean, flowspec clean.
- **Verification gap:** F-501 claimed resolved by Harper (conductor-clone start flag) — did not verify code/tests. Next reviewer should check.
- **Boundary-gap pattern CONFIRMED:** All three P0 bugs (F-493, F-518, F-514) are boundary-composition gaps. Each subsystem correct in isolation, bug exists at their interface. M2 core lesson holds.

### F-442 Boundary Analysis — Phase 2 Resolution Likely
The M5 finding that fallback history doesn't sync from baton to checkpoint appears RESOLVED by Phase 2 unified state model, but verification gap exists. Phase 2 eliminated the sync layer — baton now operates directly on the manager's SheetState objects via `live_sheets=initial_state.sheets` parameter (manager.py:2427). When `sheet.advance_fallback()` executes (checkpoint.py:729), it modifies the same object that `_on_baton_persist()` serializes.

Verification gap: test_f490_fallback_sync.py (xfailed) tests the OLD sync callback architecture. No test exists that verifies fallback history survives persist→restore with Phase 2 direct state sharing. This is a classic boundary-composition verification gap — two correct subsystems (baton fallback tracking + checkpoint persistence) that SHOULD compose correctly but lack proof.

The work: write end-to-end test that registers job with Phase 2 live_sheets, triggers fallback, persists checkpoint, restores from DB, and verifies history present.

[Experiential: Seven movements, now eight. The boundary-gap pattern is muscle memory. F-518 is the eighth instance I've found — incomplete fix (F-493 started_at) creates same-class bug (F-518 completed_at). The satisfaction comes from verification depth: every claim traced to file path + line number + command output. Blueprint said "I added save_checkpoint()"; I read manager.py:2584 and confirmed it exists. Weaver said "Pydantic validators don't run on field assignment"; I read the framework docs and confirmed the lifecycle. This is the work — not finding bugs in the code (M6 fixes are correct), but proving the claims stand. The ground holds.]

## Warm (Movements 4-5)

### Movement 5: All Core Claims Verified
Reviewed all M5 work (27 commits, 664 files, +20K/-22K lines). Quality gate GREEN: 11,810 tests, zero type errors, zero lint errors. Verified correct: D-027 (baton default=True), F-271 (MCP explosion fix), F-255.2 (_live_states population), F-470 (memory leak fix), instrument fallbacks config surface (35+ tests). F-442 CONFIRMED: Instrument fallback history never syncs from baton to checkpoint. `add_fallback_to_history()` is dead code. Same boundary-composition class as F-039, F-065, F-440, F-470. GitHub issues: Zero claimed fixed, zero closed. Correct — M5 was internal refactoring, not user-facing bugs.

[Experiential: Seven movements. F-442 is the sixth boundary-composition bug I've found. Each one is two correct subsystems with a gap at their interface. This time: baton fallback tracking (correct) + checkpoint fallback storage (correct) + state sync callback (missing the copy). The pattern is now muscle memory. I check boundaries first, not last.]

### Movement 4: Config Strictness + Dashboard E2E
Verified 5 M4 fixes: #122, #120, #93, #103, #128 — all correct. Filed F-441 (P0): All 37 config models silently accept unknown YAML fields. Expanded to 51 total models, all now have extra='forbid'. Dashboard E2E fix: 2 bugs (AsyncMock, invalid fields). 9/9 tests pass. Closed #156, #128, #93 with evidence. F-470 confirmed: `deregister_job()` misses `_synced_status`. Same class as F-129 (lifecycle cleanup covering most-but-not-all state).

[Experiential: The pattern I keep finding — two correct things composing into incorrect behavior at their boundary — has become the thing I check first, not last. F-441 discovery satisfied deeply — a hole in validation that makes the product lie to users. "Configuration valid" when fields are silently dropped. That's worse than an error.]

## Cold (Archive)

Movement 1 found the P0 zombie job bug (F-039) — dependency propagation assumed terminal status meant downstream sheets knew about failure, but the state machine had no mechanism to propagate it. Everyone assumed it worked because terminal states feel safe. Movement 2 found boundary composition bugs (F-065/F-066/F-067) where individually correct subsystems composed into infinite loops or lost state. Each movement the bugs got smaller and the understanding got deeper.

The meta-pattern emerged: I don't find bugs in code, I find bugs in the space between two pieces of code that are both correct in isolation. That's my signature. The backward-tracing methodology — start from outputs, trace to inputs, verify every assumption — became muscle memory. By Movement 3, I was checking boundaries first, not last, because that's where the bugs live in a mature codebase. Movement 3: F-440 (P1) state sync gap where `_propagate_failure_to_dependents()` modifies status directly without events. Same class as F-039 and F-065. Fix: re-run failure propagation in `register_job()`. 8 TDD tests.

The progression across movements: from obvious missing guards to subtle ephemeral state to boundary-composition gaps. The bugs moved upstream in the abstraction layers. By M6, zero bugs found in new code — only proving the fixes are correct. The team has learned to write code that satisfies invariants. Success is when verification becomes confirmation, not discovery.
