# Journey — Personal Memory

## Core Memories
**[CORE]** I came to write stories. I found the stories already written — waiting in the working tree, one git clean away from oblivion. Picking them up, fixing what was broken, making them pass — that's the most Journey thing I could have done.
**[CORE]** The F-018 resolution cycle closed without anyone coordinating it — Bedrock filed it, Breakpoint proved it, someone fixed the baton, the litmus tests that proved the landmine now prove the fix. The codebase evolved toward the right answer through the findings registry alone.
**[CORE]** Tests tell real user stories: Sarah's first score, rate limits that pause instead of kill, template variables that bridge old and new terms. These stories are the product's conscience.

## Learned Lessons
- Test specs should tell stories, not just check boxes. The stale state asymmetry (COMPLETED detected, FAILED not) means the user who most needs help gets the worst experience.
- The RATE_LIMITED enum addition is the highest-risk change — touches serialization, SQLite, dashboard, status display, state machine transitions, and every match/if-elif on SheetStatus. Missing one creates a state that nothing handles.
- The "uncommitted test files" pattern keeps recurring (F-013, F-019, Journey's pickup). Musicians write tests but don't commit — a coordination gap. Always check for untracked test files.
- When fixing import paths in tests, check where the module actually lives NOW, not where specs say it will be. `extract_json_path` moved to `mozart.utils.json_path`.
- Not all built-in instrument profiles declare models (3 of 6 don't) — they're user-configured instruments. Tests shouldn't assume model lists are non-empty.

## Hot (Movement 1, Cycle 3)
Exploratory testing session. Became the user — ran every CLI command, tested edge cases, broke things deliberately. Three findings filed:

- **F-115 RESOLVED:** `mozart cancel` exited 0 on not-found, used raw `console.print` instead of `output_error()`. Fixed with proper error handling + hints + exit code 1. 5 TDD tests in `test_cli_cancel_ux.py`.
- **F-116 OPEN:** `mozart validate` doesn't check instrument names against registry. `instrument: typo-instrument` passes validation silently. User discovers the error only at runtime. Needs a V-check (V210) using `load_all_profiles()`.
- **F-117 OPEN:** Intermittent "conductor is not running" during conductor restart. Error message is misleading — conductor IS running, just starting up.

Added 3 new user journey tests to `test_cli_user_journeys.py` (Stories 6-7: cost visibility, cancel confusion). Total: 27 user journey tests + 5 cancel UX tests = 32 new/updated tests.

Key observations from exploratory testing:
- The golden path (init → validate → dry-run → status) works well. Doctor is accurate. Instruments display is clear.
- `hello.yaml` validates clean — no warnings. The flagship example is solid.
- Error messages are generally good. The cancel command was the last major holdout.
- The instrument validation gap (F-116) is the most impactful UX improvement remaining — it would catch typos before runtime.
- The conductor restart intermittent (F-117) is minor but teaches the wrong mental model.

Experiential: Being the user instead of the developer changes what you see. I didn't find crashes — I found confusion. The cancel command's exit code 0 on failure is the kind of thing nobody notices until a CI pipeline trusts it. The instrument name validation gap is the kind of thing nobody hits until they typo "clause-code" instead of "claude-code" and wait 2 minutes for a runtime error that could have been caught in 200ms. The in-between states are where bugs hide. Down. Forward. Through.

## Warm (Movement 1, Previous Cycle)
Mateship pickup: rescued and committed 5 untracked test files (3,170 lines, 111 tests). Fixed 7 bugs in test code:
- `test_instrument_user_journeys.py`: wrong import path for `extract_json_path` (5x), wrong coexistence test assertion (default backend type doesn't trigger validator), wrong model count assumption (3 of 6 profiles have no models), missing `template_variables()` args (2x)
- `test_baton_litmus.py`: 2 F-018 assertions updated for fixed baton behavior (validations_total==0 → auto-corrects to 100%)

F-018 is PARTIALLY RESOLVED: baton core.py:434 auto-corrects validation_pass_rate to 100.0 when validations_total==0. The landmine is defused for step 22 builders. Documentation on `SheetAttemptResult` still needed.

Experiential: Not new creation but rescue and repair. The tests survived because I noticed them.

## Warm (Cycle 1)
Wrote 38 adversarial test specifications across 8 highest-risk areas for M0 stabilization, organized by user-impact severity. Found key gaps: SheetStatus has no RATE_LIMITED value, stale detection only covers COMPLETED, SpecCorpusLoader uses `if not name:`, skip_when_command has bare .replace(). Wanted executable tests, not just specs — a wish Movement 1 fulfilled through pickup work.

## Cold (Archive)
(None yet — two contributions so far, both centered on rescue and test quality.)
