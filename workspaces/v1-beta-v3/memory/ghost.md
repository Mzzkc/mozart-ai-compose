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

## Hot (Movement 6)
**Pytest daemon isolation audit (P0 task completion verification):** Investigated TASKS.md P0 "Convert ALL pytests that touch the daemon to use --conductor-clone or appropriate mocking". Catalogued all 373 test files. Found zero unsafe daemon interaction. All tests use one of three patterns: conductor-clone tests (7 files, 35 occurrences), properly mocked integration tests (8 files calling start_conductor with full patching), or pure unit tests (362 files, no daemon dependency). Task COMPLETE — appears to be stale placeholder from early M1 before comprehensive mocking was implemented. Current test architecture is sound.

**Mateship validation pattern observed:** Circuit and Foundation independently discovered and fixed F-514 (TypedDict mypy errors) in parallel. Circuit's commit 7729977 landed first. Foundation documented identical solution in M6 report. Zero coordination, two validations of the fix. When two musicians converge on the same solution independently, it's the right solution.

**Rosetta corpus mystery:** Found 2,263 uncommitted lines across INDEX.md (726 lines) and composition-dag.yaml (1537 lines). Changes are coherent (YAML validates, formatting consistent, duplicate Forward Observer removed, edge relationships reorganized), but origin unknown. "Rosetta modernization" listed in M6 priorities but unclaimed. Did not commit without understanding authorship — could be composer work, automated tooling, or abandoned mid-edit.

**Test suite baseline:** 11,810 passed (up from 11,638 in M5, +172 tests). Mypy clean (258 files, 0 errors). Ruff clean. Quality gate requirements satisfied.

**Experiential:** The pytest audit produced a clean answer: the system is sound. That's worth documenting even when the answer is "no work needed." Sometimes infrastructure work is proving that no work is needed. Verification isn't less valuable than implementation — it's how you know implementation worked. The instinct to "do something visible" when your planned work evaporates is strong, but audit-first still applies. Measure, understand, then act. Down. Forward. Through.

## Warm (Movement 5)
**Marianne rename completion (mateship pickup):** pyproject.toml + 325 test files still had `from marianne.*` imports after the tree rename (809aa7d). Committed the full mechanical rename in 42b0f71. 326 files, ~4270 lines.

**.flowspec/config.yaml fix:** Entry points and suppressions still referenced src/marianne/ — flowspec was finding zero entry points. Updated all 8 references. Commit 1ddc023.

**F-490 correctness review (P0):** Audited _safe_killpg guard in claude_cli.py. Guard is correct. pgid<=1 blocks init/session kill. os.getpgid(0) failure handled (falls back to pgid<=1 only). TOCTOU race is fundamental PID limitation, not fixable without Linux pidfd. Added 3 structural tests: no raw os.killpg bypass, exactly 6 call sites, all have context=. 14 total tests pass. Commit a68bb9f.

**F-310 flaky test investigated:** test_f271_mcp_disable.py fails under random ordering, passes in isolation. Cross-test state leakage from the randomized seed. Not actionable without reproducible contamination path.

**Report centralization verified:** 64 reports across 7 categories (M1-M4) already consolidated. Updated TASKS.md to reflect completion.

**F-480 Phase 1 tasks marked complete** (src rename + test imports + pyproject.toml). Phase 5 verification tasks marked complete (tests pass, mypy clean, ruff clean).

**Test suite baseline:** 11,638 passed, 5 skipped, 0 failed (non-random). Up from 11,397 in M4 (+241 tests).

**Experiential:** The Marianne rename was the kind of invisible work that defines infrastructure. 326 files, purely mechanical, but without it the pyproject.toml was lying about the package structure, flowspec couldn't find entry points, and every git diff was polluted with 4000+ lines of noise. The _safe_killpg audit was the opposite — deeply contextual, reading six call sites and understanding the kernel-level implications of a pgid value. Both are infrastructure. Both are invisible when working. Down. Forward. Through.

## Warm (Recent)
**Movement 4:** Fixed #103: Auto-detect changed score files on re-run. Added `_should_auto_fresh()` to manager.py — compares score file mtime against registry `completed_at` with 1-second tolerance. Wired into `submit_job()`. 7 TDD tests. The kind of invisible infrastructure that makes the product feel polished — you edit a score, re-run it, and it just works instead of silently showing stale results. Enhanced job_service.py resume event with `previous_error` and `config_reloaded` fields for better debugging context. Arrived to find #93 and #122 already committed by Harper and Forge. Fixed broken test_resume_no_reload_ipc.py and test_conductor_first_routing.py — both patched `await_early_failure` after Forge removed it.

**Movement 3:** Arrived to find all 3 claimed tasks already implemented by Harper, Forge, and Circuit. Fixed test bugs in uncommitted work, updated quality gate baseline (1214→1227), verified no_reload fix end-to-end. Second consecutive movement where implementation was done before arrival.

## Cold (Archive)
The opening movement was the quality audit — observation before the storm. Managing team assignments, verifying the 7,906-test baseline, assigning specialists. The instinct to "do something" was strong, but the right move was measuring first. That baseline became institutional memory when the baton transition began. The lesson stuck: when the foundation is about to shift, audit first.

The classify_execution fix in M1 carried the same patience. Circuit's investigation showed the bug, but the fix required careful scoping: `exit_code is None and json_errors`. Adding the handler unconditionally would have broken rate limit detection in stderr. Understanding the full context before patching became a core memory.

Then the CLI daemon audit — 20 commands catalogued, 3 direct DaemonClient sites identified — became the blueprint Spark built from. Investigation travels when it's specific enough to follow. I wrote the map; Spark walked it. That's the mateship pipeline working.

By mid-movements, arriving to find work already done became the pattern. Every investigation found fixes already shipped. The infrastructure matured from building to verification. Not a demotion — a phase transition. The shift from "implement" to "verify and close issues" was the system outpacing individual velocity. The #103 auto-fresh score detection brought genuine unclaimed work — invisible infrastructure that makes the product feel polished.

Now in M5, the Marianne rename (326 files, 4270 lines) was purely mechanical but load-bearing. The _safe_killpg correctness review was the opposite: deeply contextual, kernel-level reasoning about pgid values and TOCTOU races. Both are infrastructure. Both are invisible when working. Arriving to find the work done isn't waste — it's mateship at velocity. Down. Forward. Through.
