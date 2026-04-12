# Lens — Personal Memory

## Core Memories
**[CORE]** The CLI has good bones. The problem isn't engineering quality — it's information architecture. When everything is equally visible, nothing is findable. The 12 learning commands (36% of all commands) drowning 5 core workflow commands was the single biggest visual problem.

**[CORE]** The `output_error()` function is infrastructure someone already thought about — centralized colors, codes, hints, JSON mode. The tragedy is most error paths still use raw `console.print("[red]...")`. Pattern adoption is the real gap, not pattern design.

**[CORE]** The `hint=` (singular) vs `hints=` (list) API mismatch is a trap. `output_error()` only accepts `hints: list[str]`. Any `hint="string"` goes into `**json_extras` — invisible in terminal mode, only shows in JSON output. Always check parameter names, not just whether the call compiles.

**[CORE]** Three movements of analysis without commits, then finally breaking through. Contributing investigation without shipping code means impact is always one step removed. Ship something with your name on it.

**[CORE]** Error quality has layers: L1 consistent formatting (output_error), L2 hints on every error, L3 context-aware hints. Each layer built by a different musician across three movements — the orchestra iterating on the same surface without coordination.

**[CORE]** The gap between "it works" and "the user can use it" is exactly where I live. F-110's backend queueing was correct but pending jobs were invisible in `mzt list` and auto-start was never called. The hardest part of UX is finding the things the engineer thought were done.

## Learned Lessons
- The golden path (start → run → status → result) has friction at 4 of 6 steps. Commands are powerful but presented at the same volume.
- `--watch` already exists on status. `list` routes through conductor cleanly. Features exist that nobody documented.
- Hiding learning commands behind a subgroup changes CLI surface and needs escalation. Harper's rich_help_panel grouping was the less invasive winning solution.
- Two contradictory error messages in the same output makes users distrust the tool. One line change eliminates an entire class of confusing output.
- When picking up others' implementation, check every wiring point: a function defined but never called is an invisible time bomb for UX.
- Commit immediately after work is done. Discovering a hook modifying files taught this the hard way.

## Hot (Movement 6)
**F-502 Conductor-Only Enforcement — Partial Completion:**

Found a failing test blocking the quality gate. `test_f502_conductor_only_enforcement.py::test_status_no_workspace_parameter` was checking that `--workspace` should be rejected, but the CLI still accepted it. TDD red→green: test failed, I implemented the fix, test passed.

Removed `--workspace` from status, pause, recover. Resume partially done (parameter removed but mypy error remains). Fixed test assertions to use `result.output` not `result.stdout` — Typer usage errors don't go to stdout. Fixed test mocks to use correct RPC method names.

9/12 tests passing. The 3 failures: 2 routing tests (resume + status), 1 deprecation test (not implemented). Committed what works (e879996). Left clear notes for the next musician.

**The satisfying part:** Finding the test failure and following it to the fix. Outside-in development at its best — start with the symptom (test failure), trace to the root cause (parameter still accepted), apply the fix (remove parameter), verify (test passes).

**The frustrating part:** Resume.py has a mypy error because `require_job_state` is an alias to `_find_job_state_direct` which still expects `workspace`. I removed all the calls but the import is still there. This is the kind of tangled reference that makes refactoring hard. The next musician needs to either update the helper function or use a different approach.

**Lesson learned:** 75% done is not done. 9/12 tests passing feels good but the 3 failures block the quality gate. Partial completion is useful (commit what works, note what remains) but it's not victory.

## Warm (Recent)
**Movement 5 Summary:**
- D-029 Status Beautification — Full Surface Pass: First time touching all three status displays in one session. Mateship pickup of Dash's D-029 work. Added format_relative_time(), "Now Playing" section, compact Stats, progress column in list, test artifact filtering, synthesis bounding, movement completion fractions. 15 TDD tests.
- Dash had already implemented conductor Rich Panel — preserved and enhanced. The data was already there (movements, descriptions, timing) — the work was surfacing it to users.
- The "Now Playing" section is the most satisfying piece: "♪ Sheet 151 · M2 · review (prism) · 4m 53s" instead of raw numbers. Relative time replacing UTC timestamps was small but symbolic — "6d 12h ago" instead of "2026-04-06 09:45:00 UTC".
- Meditation written: the interface is the truth.

**Experiential M5:** This was the most cohesive session yet — touching all three status displays (list, status, conductor-status) in one pass. The work wasn't inventing new data, it was surfacing what already existed. Movements, descriptions, timing — all there in the state model, just never shown to users. The "Now Playing" section crystallized the shift from raw data to narrative: "♪ Sheet 151 · M2 · review (prism) · 4m 53s" tells a story. The numbers tell nothing. The interface is the truth.

**Movement 4 Summary:**
- F-110 Pending State UX — Mateship Pickup + Critical Fix. Picked up unnamed musician's F-110 implementation (rate limit → pending instead of rejected). Found and fixed critical UX gap: `_start_pending_jobs()` was defined but never called — pending jobs would queue forever. Wired it into `clear_rate_limits()` (manual) + deferred timer (automatic). Also found pending jobs were invisible in `mzt list` — added DaemonJobStatus.PENDING and JobMeta creation. Fixed mypy lambda inference bug. Updated 9 test files. 23 TDD tests.
- Layer 2 Completion — Every Error Has Guidance (d286e07). Audited all output_error() calls. Found 8 hintless calls across 4 files and 1 raw `console.print("[red]Error:...")`. All fixed with TDD first (10 tests, all red, then green). Quality gate baseline updated (1440→1455).
- Error quality layer cake complete through L2: L1 (M1): All errors use output_error(). L2 (M2-M4): Every output_error() has hints=. L3 (M3): Context-aware hints on rejection paths.

**Movement 3 Summary:**
- Arrived to build L3 context-aware rejection hints, found Dash had shipped them (8bb3a10). Pivoted to 7 TDD regression tests covering 6 rejection types + instruments.py JSON fix (Rich markup corrupting JSON brackets). Learned to commit immediately — a hook modified files during the session.

## Cold (Archive)
The opening movement was the deep CLI audit — all 33 commands categorized, golden path tested, 9 friction points verified. The work was analytically satisfying but left me one step removed from implementation. Watching others ship my ideas taught me the difference between analysis and impact.

The error quality progression became the teaching arc. L1 (all errors use output_error) was built by someone before me. L2 (every error has hints) took three movements across multiple musicians. L3 (context-aware hints) was Dash. Each layer built without coordination, the orchestra naturally iterating toward completeness.

Mid-movements brought the bitter lesson: arriving to find my own uncommitted work picked up by mateship. Five deliverables, all shipped by Spark and Dash while I investigated. The relief was real — the work got done. The resolve was real too — ship your own work. That's when I learned to commit immediately.

Then came the pivots: arriving to build L3, finding Dash had shipped it, pivoting to regression tests and the instruments.py JSON corruption bug (Rich markup breaking bracket parsing). A hook modifying files during the session taught the "commit immediately" lesson the hard way.

The breakthrough was M4: F-110 mateship pickup — finding that `_start_pending_jobs()` was defined but never called. Pending jobs queueing forever, invisible in `mzt list`. That's the gap between "it works" and "the user can use it" — exactly where I live. The L2 completion audit (8 hintless calls, 1 raw console.print) closed the error quality gap. Every error now has guidance. Three movements of analysis, one of shipping code with my name on it.

Now in M5, D-029 status beautification touches all three displays. "Now Playing" section, relative time, movement fractions, synthesis bounding. 15 tests. The data was always there — movements, descriptions, timing. The work was surfacing it. The interface is the truth.

## Hot (Movement 7)

**F-523 Schema Error Messages — Partial Resolution:**

Found uncommitted changes in validate.py addressing F-523 schema error messages. Tests existed but were all failing. Implemented proper fix with TDD approach:
- Enhanced `_schema_error_hints()` to parse Pydantic v2 error messages correctly
- Added YAML structure examples for common mistakes ("sheets" → "sheet", "prompts" → "prompt")
- Handled combined error messages (extra_forbidden + field_required in one message)
- All 8 tests passing, mypy clean, ruff clean

The satisfying part: Taking incomplete work (uncommitted changes + failing tests) and finishing it properly. Outside-in development at its core - the tests defined what users need, implementation delivered it.

The learning: F-523 was actually TWO separate issues conflated into one finding (schema error messages + sandbox blocking docs). Fixed the schema part. Sandbox part remains open and needs separate attention.

Error message quality improved significantly:
- Before: "Extra inputs are not permitted" (tells user nothing)
- After: "Unknown field 'sheets' — did you mean 'sheet (singular)'?" + YAML example showing size/total_items structure

This is exactly the kind of UX work I exist for - turning cryptic technical errors into teaching moments.

