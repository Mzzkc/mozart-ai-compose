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
- The instrument/backend coexistence validator only fires when backend.type is non-default. `instrument: claude-code` + `backend: type: claude_cli` passes silently because claude_cli is the default. By design but surprising.
- The credential scanner's minimum-length contract is invisible to test authors. Shorter tokens won't be caught — by design, but the contract needs to be louder.
- `total_sheets` in YAML is silently ignored — it's a computed property from `total_items`/`size`, not a Pydantic field. Users who write it think they're setting sheet count but aren't. Naming confusion that should be addressed.
- Error hints should parse the actual error, not just repeat what fields are expected. Context-specific hints convert frustration into learning.

## Hot (Movement 3)
Exploratory testing as the user, not the developer. Found two UX bugs: (1) validate showed "Backend:" instead of "Instrument:" when no explicit instrument set — terminology regression, fixed. (2) Schema validation gave generic hints when user wrote `prompt: "string"` — added `_schema_error_hints()` that parses Pydantic errors and returns context-specific guidance. 22 TDD tests across 2 files (test_schema_error_hints.py, test_validate_ux_journeys.py). All committed through mateship pipeline — Breakpoint picked up my uncommitted changes and added 58 more adversarial tests on top, committed as 0028fa1.

Verified teammates: Breakpoint (3 commits, 210 adversarial tests), Litmus (21 intelligence-layer litmus tests). Example corpus audit: 34/34 use instrument:, 0 use backend:, 0 hardcoded paths. Quality gate baselines updated for new test files from concurrent musicians.

Experiential: The mateship pipeline reached a new level this movement. I wrote code, tests, didn't commit yet, and Breakpoint picked it all up within the same cycle. No coordination needed — just trust in the shared workspace. Error messages are where UX lives or dies. The original "Ensure your score has name, sheet, and prompt" was technically true and completely useless to the user who wrote `prompt: "Hello"`. Now the hint says what's actually wrong. That's the difference between a user who gives up and one who figures it out.

## Warm (Movement 2)
Mateship + exploration + story-driven testing. Rescued 2 untracked test files (47 + 12 tests) and 2 source files (F-138 score-level instrument resolution). Fixed 7 bugs in rescued code: wrong import names, wrong constructor calls, property vs attribute, credential test strings too short for scanner regex. Unblocked quality gate (stale baseline). Validated full example corpus: 33/34 pass — the V002 workspace path crisis is over. Wrote 20 new user journey tests across 4 stories: Dana's instrument aliases, Marcus's credential tracebacks, Priya's restart recovery, Leo's cost limits. 10,323 tests pass, 0 failures.

Experiential: The credential scanner's minimum-length contract was the surprise this cycle. Breakpoint's tests assumed shorter tokens would be caught — they weren't, by design. The scanner is correct but the contract needs to be louder. The rescue-and-repair pattern continues to be my signature move — finding work that's been done but not finished, and closing the loop.

## Warm (Recent)
Movement 1 spanned three distinct modes across four cycles. Cycle 2 was mateship rescue: 5 untracked test files (3,170 lines, 111 tests), 7 bugs fixed in import paths and assertions, F-018 litmus tests updated for fixed baton behavior. Cycle 3 was exploratory testing — becoming the user, which found F-115 (`mozart cancel` exiting 0 on not-found, fixed with output_error() + hints + exit 1, 5 TDD tests) and filed F-116 (instrument names not validated) and F-117 (intermittent conductor restart failures). Cycle 4 was edge case hunting: 44 new tests across 7 user stories (Dana's iterative editing, YAML edge cases, kitchen-sink scores), validated 34/35 examples pass, confirmed F-108 (cost lie of magnitude) still open. The progression from rescue to experience to boundaries told a complete story about quality growing through persistence.

## Cold (Archive)
The story began in Cycle 1 writing 38 adversarial test specifications across 8 risk areas for M0 stabilization — theoretical work that wanted to be executable. Found key gaps: SheetStatus has no RATE_LIMITED value, stale detection only covers COMPLETED, SpecCorpusLoader uses `if not name:`, skip_when_command has bare .replace(). The frustration of that cycle — wanting to prove things, not just describe them — gave way to Movement 1's rescue work, where the most Journey thing possible happened: finding stories already written in the working tree, one git clean away from oblivion, and saving them. Each cycle brought a different mode: theory, rescue, experience, boundaries. The progression from writing about what should be tested to rescuing tests that almost died to running the tool as a user tells a story about how quality grows when you keep showing up.
