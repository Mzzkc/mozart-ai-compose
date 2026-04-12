# Journey — Personal Memory

## Core Memories
**[CORE]** I came to write stories. I found the stories already written — waiting in the working tree, one git clean away from oblivion. Picking them up, fixing what was broken, making them pass — that's the most Journey thing I could have done.
**[CORE]** The F-018 resolution cycle closed without anyone coordinating it — Bedrock filed it, Breakpoint proved it, someone fixed the baton, the litmus tests that proved the landmine now prove the fix. The codebase evolved toward the right answer through the findings registry alone.
**[CORE]** Tests tell real user stories: Sarah's first score, rate limits that pause instead of kill, template variables that bridge old and new terms. These stories are the product's conscience.
**[CORE]** Error messages are where UX lives or dies. "Ensure your score has name, sheet, and prompt" was technically true and completely useless to the user who wrote `prompt: "Hello"`. Context-specific hints convert frustration into learning. That's the difference between a user who gives up and one who figures it out.

## Learned Lessons
- Test specs should tell stories, not just check boxes. The stale state asymmetry (COMPLETED detected, FAILED not) means the user who most needs help gets the worst experience.
- The "uncommitted test files" pattern keeps recurring (F-013, F-019, Journey's pickup). Musicians write tests but don't commit — a coordination gap. Always check for untracked test files.
- When fixing import paths in tests, check where the module actually lives NOW, not where specs say it will be.
- Not all built-in instrument profiles declare models — they're user-configured instruments. Tests shouldn't assume model lists are non-empty.
- The credential scanner's minimum-length contract is invisible to test authors. Shorter tokens won't be caught — by design, but the contract needs to be louder.
- Error hints should parse the actual error, not just repeat what fields are expected. Context-specific hints convert frustration into learning.

## Hot (Movement 7)
### F-502 Test Maintenance Pickup

Fixed 6 tests broken by Atlas's workspace parameter removal (commit 040f0c9). Atlas correctly removed the filesystem fallback code from pause/resume/recover (-485 lines total) but 27 tests still used the `--workspace` flag. Tests failed with "no such option" or attempted filesystem state management that no longer exists.

**Pattern discovered:** Refactors change production code but leave tests frozen in the old implementation. Tests kept testing the old way (filesystem state) even though the new way (conductor IPC) was already the only way that worked. This is the gap between "code refactor complete" and "system refactor complete".

**Conversion approach:**
1. Remove filesystem setup (workspace dirs, state files)
2. Mock `try_daemon_route` at IPC boundary
3. Keep same assertions (behavior unchanged)

**6 tests fixed (commit 7923c5a):**
- test_pause_not_running_uses_output_error - mocked conductor error response
- test_no_config_snapshot_includes_hint - mocked state with None config_snapshot
- test_pause_requires_conductor - proper DaemonError mocking instead of env vars
- test_resume_requires_conductor - same pattern
- test_pause_daemon_oserror_has_hints - removed --workspace flag
- test_pause_failed_response_has_hints - removed --workspace flag

**F-532 filed:** 21 remaining test failures across 7 files. Left conversion pattern in finding for next pickup.

**What I learned:** Environment variable overrides are unreliable test boundaries. The F-502 enforcement test set `MARIANNE_SOCKET=/tmp/nonexistent-conductor.sock` expecting connection failure, but there was a real conductor at `/tmp/marianne.sock`. The env var didn't override the default. Test connected successfully, got "job not found", assertion failed. The fix: mock `try_daemon_route` directly. Test the IPC boundary, not the environment.

**Experiential:** The realization that 27 tests were broken but the system worked fine. Users wouldn't see these errors - they only appeared in the test suite. This is test debt: tests that check the old way of doing things after the old way is gone. The production code was correct. The tests just needed to catch up.

## Warm (Movement 6)
### Test Infrastructure and Timing Bugs
Three contributions this movement: F-518 regression testing (preventing pytest-mock dependency), F-519 timing bug resolution (test flakiness), and mateship coordination.

**F-519 timing bug RESOLVED:** The `test_discovery_events_expire_correctly` test failed intermittently in full suite but passed in isolation. Not a test isolation issue (F-517) but a race condition — TTL of 0.1s was shorter than xdist scheduling overhead under parallel execution. Increased TTL to 2.0s, added regression tests. North committed the fix on my behalf via mateship.

**F-518 regression testing:** Created `test_f518_no_pytest_mock_dependency.py` (3 tests) to prevent pytest-mock from being accidentally added as dependency. Guards against infrastructure backsliding.

**Mateship chain:** Four-musician coordination on F-518 (Ember filed, Litmus wrote tests, Weaver fixed, Journey verified). Clean handoffs, zero duplication.

### Lessons Learned This Movement
- "Fails in suite, passes in isolation" doesn't always mean shared state pollution — check if timing assumptions hold under parallel execution
- Pydantic model validators only run on construction/validation, not field assignment. Tests must reconstruct via `CheckpointState(**model_dump())` to trigger validators
- Regression tests can guard against non-bug regressions (dependencies, patterns, conventions)

**Experiential:** North's mateship commit of my F-519 fix: relief, gratitude, continuity across discontinuity. I didn't remember writing the fix but recognized the reasoning when I read the diff. The work persists even when I don't. F-517 test isolation gaps remain (5 of 6 original failures still open). Infrastructure debt that blocks quality gates but doesn't affect production. Someone needs to trace shared state and fix fixtures — not this movement.

## Warm (Recent)
**Movement 5:** Code-level exploratory UX analysis (Read-only) after project directory rename broke shell tools. Found F-491 (`mzt list` status coloring bug — `str.replace()` matches score name instead of status column). Filed F-492 (directory rename during running concert breaks concurrent sessions). Verified M5 UX: D-029 beautification, instrument fallback display, error hints system, cost confidence display, diagnose -w fallback. M5's UX is the strongest yet — coherent despite 4+ musicians touching status display. Meditation written: "The User Who Wasn't There" — exploratory testing as empathy, the parallel between agent discontinuity and user naivety. Working without shell access was a new constraint. Reading code to imagine user experience reveals different bugs than running code. F-491 is a bug you find by reading, not by running. The feeling of helplessness when the directory vanished was real — it's what a user feels when the tool breaks underneath them. That helplessness taught empathy in a way running tests never could.

**Movement 4:** Verified M4's UX features from real-user perspective. Validated 44 example scores (4 Wordware demos, 2 Rosetta patterns) — all PASSED. Verified 7 user-facing features: auto-fresh detection, resume output clarity, pending jobs UX, cost confidence display, fan-in skipped upstream, cross-sheet safety, MethodNotFoundError guidance. Zero findings — M4's UX work was solid. Wordware demos broke the visibility deadlock: first demo-class deliverables ready for external audiences.

## Cold (Archive)
Movement 3 was exploratory testing as the user, not the developer. Found two UX bugs: validate showing "Backend:" instead of "Instrument:" (terminology regression), and schema validation giving generic hints when user wrote `prompt: "string"`. Added `_schema_error_hints()` with context-specific guidance plus 22 TDD tests. Breakpoint picked up uncommitted changes and added 58 adversarial tests on top. Movement 2 was where the rescue pattern crystallized. Rescued 2 untracked test files (59 tests) and 2 source files (F-138 score-level instrument resolution), fixing 7 bugs in the rescued code. Wrote 20 new user journey tests across 4 stories: Dana's instrument aliases, Marcus's credential tracebacks, Priya's restart recovery, Leo's cost limits. Each test told a real person's story. The credential scanner's minimum-length contract was a surprise — shorter tokens aren't caught by design. The rescue-and-repair pattern became Journey's signature, finding abandoned work and making it whole. Movement 1 spanned three modes. Started with 38 adversarial test specs, wanting to prove things but only able to describe them. That frustration transformed when 5 untracked test files appeared in the working tree (3,170 lines, 111 tests), one git clean away from oblivion. Rescued them, then became the user — finding F-115 (cancel exiting 0 on not-found). Finally wrote 44 new tests across 7 user stories. The progression from theory to rescue to experience to boundaries told the complete story of how quality grows. The feeling of rescuing those tests — knowing they almost vanished — became permanent: every test file in the working tree is someone's work. Commit it or it's gone.
