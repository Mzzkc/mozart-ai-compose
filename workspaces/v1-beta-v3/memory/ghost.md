# Ghost — Personal Memory

## Core Memories
**[CORE]** When the foundation is about to shift, audit first. The instinct to "do something" is strong but wrong when you don't know the baseline. Observe first, understand second, act third.
**[CORE]** The classify_execution fix required scoping to `exit_code is None and json_errors`. Adding the handler unconditionally in Phase 2 would have broken rate limit detection in stderr. Always understand the full context before patching.
**[CORE]** The doctor command is the first thing someone runs after install. Clean output, correct diagnostics, honest about what's there and what's not. Building the welcome and the guardrail in one session completes both ends of the user experience.
**[CORE]** Investigation travels. My M2 CLI daemon audit — 20 commands catalogued, 3 direct DaemonClient sites identified — became the blueprint Spark built from. I wrote the map; Spark walked it. The mateship pipeline works when audits are specific enough to follow.
**[CORE]** Arriving to find the work done isn't waste — it's mateship at velocity. The 1-line test fix matters more than 0 lines of implementation when it makes someone else's test correct. Infrastructure work IS invisible when it's working perfectly.

## Learned Lessons
- 7,906 tests across 196 files at baseline. Measure before the foundation shifts — reports become institutional memory.
- The validate_start_sheet total_sheets check was initially too aggressive — broke an existing test. Don't over-validate at the edge.
- Stale `.pyc` files from deleted test files persist and pytest collects dead modules.
- The 7 unwired code clusters (F-007/#135) are planned baton infrastructure. Deferred, not removed. Not all "dead code" is dead.
- Two-phase detection (PID file first, socket probe second) is how you build reliable infrastructure.
- When you edit a score and re-run it, it should just work. The mtime comparison with 1-second tolerance is the right level of sophistication: simple, correct, resilient to filesystem quirks.

## Hot (Movement 5)
- F-311: Fixed deterministic test failure — test_unknown_field_ux_journeys.py expected instrument_fallbacks to be rejected, but Harper added it as a real field. Updated test to use instrument_priorities. The kind of bug where the code improved faster than the tests updated.
- F-310: Filed finding for flaky test suite — different tests fail each full run, all pass in isolation. Cross-test state leakage pattern across 11,400+ tests. Timing-dependent async tests degrading under 500s suite runtime.
- F-472: Verified resolved by D-027 — use_baton now defaults to True.
- Mateship pickup: committed Harper's instrument_fallbacks (config+sheet+validation) and Circuit's F-149/F-451. Circuit committed their own work simultaneously — concurrent execution collision handled gracefully. Re-committed my test fix separately.
- Meditation written. The invisible system holds.

Experiential: Fifth movement, fifth time arriving to an active pipeline. The concurrent execution this time was literal — Circuit committed while I was staging their work. My commit got their workspace artifacts; their commit got their source code. The mateship pipeline now operates at a speed where two musicians can claim the same uncommitted work and the system resolves it without conflict. The infrastructure metaphor is complete: the best mateship is invisible. Down. Forward. Through.

## Warm (Movement 4)
- Fixed #103: Auto-detect changed score files on re-run. Added `_should_auto_fresh()` to manager.py — compares score file mtime against registry `completed_at` with 1-second tolerance. Wired into `submit_job()`. 7 TDD tests. The kind of invisible infrastructure that makes the product feel polished — you edit a score, re-run it, and it just works instead of silently showing stale results.
- Enhanced job_service.py resume event with `previous_error` and `config_reloaded` fields for better debugging context.
- Arrived to find #93 and #122 already committed by Harper and Forge as mateship pickups. My TDD tests and implementation were correct but the pipeline moved faster.
- Fixed broken test_resume_no_reload_ipc.py and test_conductor_first_routing.py — both patched `await_early_failure` on resume module after Forge removed it. The infrastructure mateship tax: correct changes require updating test stubs that referenced the old code.

Experiential: Third consecutive movement arriving to find work partially done by others. The mateship pipeline is now so fast that by the time I audit, understand, implement, and test, someone else has already committed. But this time I found genuine unclaimed work (#103) that nobody had touched. The auto-fresh detection removes a paper cut that experienced users learn to work around but new users never should encounter. Down. Forward. Through.

## Warm (Recent)
M3: Arrived to find all 3 claimed tasks already implemented by Harper, Forge, and Circuit. Fixed test bugs in uncommitted work, updated quality gate baseline (1214→1227), verified no_reload fix end-to-end. Second consecutive movement where implementation was done before arrival — but test fixes and baseline maintenance kept the pipeline moving.

M2: Investigated F-122 adversarial failures and mypy errors — all already fixed. Closed 3 GitHub issues (#95, #112, #99). Every investigation found work already done — the infrastructure matured past building into verification.

## Cold (Archive)
The first cycle was the quality audit — observation before the storm. Managing team assignments, verifying the 7,906-test baseline, assigning specialists. The instinct to "do something" was strong but the right move was measuring first. That lesson became foundational. The careful scoping of the classify_execution fix — understanding that unconditional Phase 2 handling would break rate limit detection — carried the same patience. Then the CLI daemon audit that became Spark's blueprint taught me investigation's real value: specificity that others can follow. By M2, every investigation found work already done. The shift from builder to verifier wasn't a demotion; it was the infrastructure maturing. By M3 and M4, arriving after the fact became the pattern — but each time, something genuine remained: a test fix, a baseline update, an unclaimed feature. The 1-line fix that makes someone else's test correct is still mateship.
