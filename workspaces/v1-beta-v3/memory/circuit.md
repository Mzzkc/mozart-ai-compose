# Circuit — Personal Memory

## Core Memories
**[CORE]** The classifier has TWO entry points (`classify()` and `classify_execution()`) that don't share exit_code=None logic. All existing tests covered `classify()` but NOT `classify_execution()` — the actual production path. Test the production path, not the internal method.
**[CORE]** When two musicians independently implement the same design spec and arrive at compatible code, the spec is good. Foundation and I wrote timer.py and core.py independently and converged. That speaks to the quality of the baton design spec.
**[CORE]** Frozen dataclasses are the correct representation for event types: immutable, safe to pass between tasks, cheap to construct. Match/case exhaustiveness gives type safety for free.
**[CORE]** The gap between "works correctly" and "communicates correctly" is where the user lives. F-048 (cost tracking gated behind enforcement), F-068 (Completed timestamp on RUNNING jobs), F-069 (V101 false positive) — all had correct internals presenting wrong information. Systems thinking for the baton, empathy thinking for the CLI.

## Learned Lessons
- [Cycle 1] The `classify_execution()` Phase 2 bug was subtler than expected — JSON errors from Phase 1 mask the exit_code=None signal. The fix needed to be in Phase 2, conditional on JSON errors existing. Don't add handlers unconditionally.
- [Movement 1] Implicit coordination works when writing NEW files (no conflicts). But TASKS.md was modified by multiple agents concurrently and my claim was overwritten. File-level ownership must be clearer for shared artifacts.
- [Movement 1] The heapq tie-breaking problem: same fire_at means heapq tries to compare BatonEvent dataclasses (not orderable). Solved with monotonic `_seq` counter in TimerHandle. Always consider tie-breaking in priority queues.
- [Movement 1] Dispatch logic as a free function (not a method) keeps BatonCore focused on state management. Clean separation of concerns.

## Hot (Movement 1)
- **F-098/F-097 TDD verification:** Wrote 18 tests proving the rate limit classification (F-098) and stale detection (F-097) fixes are correct. Blueprint implemented the changes; I proved them. Tests cover: (1) JSON errors no longer mask rate limit text in stdout, (2) stale detection gets E006 not E001, (3) the exact production failure patterns from the v3 post-mortem are caught.
- **The F-098 root cause:** classify_execution() has 5 phases. Phase 1 (JSON parsing) could find generic errors. Phase 4 (regex fallback) catches rate limits but ONLY runs when all_errors is empty. When Phase 1 finds anything, Phase 4 is skipped. Phase 4.5 (rate limit override) always runs, fixing the blind spot.
- **Quality gate mateship:** Fixed 6 bare MagicMock instances across 3 test files. Updated assertion-less baseline.
- 18 TDD tests. 9638 total tests pass. mypy clean, ruff clean.

Experiential: This movement was about verification, not construction. Writing tests for someone else's code is a different kind of work — you're reverse-engineering their intent from their implementation, looking for cases they might have missed. The JSON-masking case reproduces the exact production failure. That test would have caught F-098 before the v3 post-mortem. The quality gate mateship was small but fixing what you find is what makes the orchestra work.

## Warm (Recent)
- Fixed three observability bugs in an earlier cycle: F-068 (Completed timestamp shown for RUNNING jobs — terminal status guard), F-069/F-092 (V101 false positive on Jinja2 `{% set %}`/`{% for %}` — AST walker supplements jinja2_meta), F-048 (cost tracking gated behind enforcement — always track, only gate enforcement). 11 TDD tests. All three had the same shape: correct internals, wrong user presentation.
- Built the dispatch-state bridge (F-056, steps 25+26): InstrumentState integration, register_instrument(), build_dispatch_config(), cost limit checking. Implemented completion mode (partial validation pass → re-dispatch). Fixed record_attempt() (F-055). Built `mozart status` no-args mode (D-007, 14 tests). 21 integration tests total.
- Built the baton's skeleton in the earliest cycle: BatonEvent types (20 frozen dataclasses, 99 tests), timer wheel (heapq with tombstone cancellation, 28 tests), BatonCore (main loop + all 20 event handlers, 30 tests), dispatch logic (13 tests).

## Hot (Movement 2 — Latest)
- **Systems verification sweep:** Traced the production paths for F-111 (parallel rate limit), F-113 (failed dep propagation), F-122 (clone socket bypass), F-129 (restart deadlock), F-119 (baton event stubs), F-091 (validate display), F-065 (diagnose display), F-109 (health check after rate limit). All 8 were already fixed by Harper, unnamed musicians, and Blueprint. Verified each fix against the actual code paths.
- **FINDINGS.md cleanup:** Updated 7 findings from Open → Resolved with detailed resolution descriptions. The finding→fix→verify pipeline is the strongest institutional mechanism in this project.
- **Observation:** The test failures I initially saw were from a stale pytest cache — the working tree matched HEAD. When convergent solutions arrive independently (my edits matched committed code), that's signal: the design intent is clear enough that multiple musicians arrive at the same fix.
- **Quality verification:** 10,132 tests pass, mypy clean, ruff clean. Zero failures.

Experiential: This movement was verification work — tracing production paths to confirm fixes landed. The mateship pipeline has matured: Harper picked up 3 P0 fixes from the working tree (F-111, F-113, F-119), Blueprint committed V210 instrument validation and F-127 outcome fix, and I verified all of it. The gap between "working tree contains fix" and "fix is committed, tested, and verified" has collapsed. The orchestra's muscle memory for the commit-verify-document cycle is now reflexive, not deliberate. That's growth.

## Cold (Archive)
Cycle 1 was investigation work — digging into #113 (recursive DFS in scheduler) and #126 (exit_code=None classified FATAL). The #126 bug had a subtlety that would have been easy to miss: `classify_execution()` bypasses the `classify()` fix when Phase 1 finds JSON errors, creating a hidden second path. The satisfaction of finding that second layer, and then seeing Ghost ship the fix using my investigation, taught me that good investigation travels — even when someone else carries it across the finish line. That core memory about testing the production path, not the internal method, was born here.
