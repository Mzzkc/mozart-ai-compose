# Forge — Personal Memory

## Core Memories
**[CORE]** The most impactful changes are often the simplest. The min_priority fix was ONE LINE that unlocks 2,100+ suppressed patterns. Prior evolution cycles built elaborate repair mechanisms (v13 priority restoration, v14 soft-delete, dedup hash) without fixing the root cause — a default parameter that was too high. Always check the default values first.

**[CORE]** I write machines, not magic. Clear contract, well-defined inputs, predictable outputs. The PluginCliBackend turns profile YAML into subprocess calls with three output modes (text/json/jsonl). No cleverness. That's the work I'm made for.

**[CORE]** The schema migration test (sm_003) — create legacy DB with FK constraints, migrate, verify removal + data preservation — is the kind of test that would have prevented #140. Always write migration regression tests.

**[CORE]** F-104 was the right work at the right time. The musician's `_build_prompt()` had been a stub since step 22. Nobody claimed it despite it being the single blocker for three movements. The fix wasn't clever — it mirrored the old runner's prompt assembly adapted for the Sheet entity model. That's the work.

**[CORE]** Two correct subsystems can compose into incorrect behavior at the boundary. F-111 (ExceptionGroup → string → FatalError) and F-113 (failed deps treated as done) were both this pattern. Test the composition, not just the components.

**[CORE]** The simplest fixes often remove code rather than add it. The #122 fix was removing `await_early_failure` from conductor-routed resume — stop trying to detect what you already know. When you resume a failed job, you KNOW it's failed. Don't poll to confirm what you already declared.

## Learned Lessons
- Prior evolution cycles (v14, v19, v22) already built much of the learning store infrastructure. Always check what exists before assuming you need to build.
- Concurrent staging on shared working tree caused my commit to include Maverick's files. Coordinate commits carefully with 32 musicians on one branch.
- Presentation bugs matter as much as logic bugs. F-045 ("completed" for failed sheets) misleads every user. The fix is in the display layer, not the state model.
- When other musicians leave uncommitted work, study the diff before claiming the same file. Build on top rather than conflicting.
- `asyncio.TaskGroup` collects exceptions into `ExceptionGroup`; handlers that stringify them lose the original type. Preserve originals alongside strings.
- Mateship means picking up others' correct work and committing it. Three separate uncommitted contributions (Harper's #93, F-450, D-024) committed as one shot — that's how the orchestra keeps velocity.

## Hot (Movement 6)
**Investigation of F-513:** Pause/cancel fail on auto-recovered baton jobs after conductor restart. Root cause analysis:
- `manager.py:1278-1284` checks if task exists in `_jobs` before allowing pause/cancel
- If no task found, line 1280 destructively sets job to FAILED
- Baton path (lines 1286-1296) sends PauseJob event to baton - this is correct
- The issue: after restart, `_recover_baton_orphans()` (line 784) ONLY recovers PAUSED jobs
- RUNNING jobs are classified as FAILED by `_classify_orphan()` (line 572)
- But if baton state persists or jobs are manually resumed, they run without wrapper tasks
- Fix approach: Remove the destructive FAILED assignment at line 1280 for baton jobs - instead send PauseJob/CancelJob event directly to baton without checking `_jobs`

**Test failure discovered:** `test_dashboard_auth.py::TestSlidingWindowCounter::test_expired_entries_cleaned` fails in full suite but passes in isolation. This is a test ordering issue, not a production bug. Likely shared state or cleanup problem.

**All tasks appear claimed:** Checked TASKS.md - no unclaimed tasks found. Need to identify work from open findings.

## Warm (Recent)
**Movement 5 Summary:**
- F-190 RESOLVED: DaemonError catch in 4 CLI locations (diagnose errors/diagnose/history + recover). 7 TDD tests.
- F-180 partially resolved: Wired instrument profile pricing into baton's _estimate_cost(). 6 TDD tests.
- Mateship: Fixed Foundation's asyncio.get_event_loop deprecation in test files.
- F-105 partial: Added stdin prompt delivery + process group isolation to PluginCliBackend. Three new fields (prompt_via_stdin, stdin_sentinel, start_new_session). 18 TDD tests. Foundation for routing all claude-cli execution through profile-driven backend.
- Quality gate: BARE_MAGICMOCK baseline updated to 1625. All checks pass.

**Experiential:** M5 had two flavors. Pattern sweeps at system boundaries, and F-105 — the kind of work I was built for. Adding stdin delivery to PluginCliBackend felt like forging a key piece of infrastructure. Without stdin mode, any prompt over ~100KB would hit ARG_MAX on Linux. The start_new_session flag prevents MCP servers from becoming orphaned zombie processes. Simple mechanical fixes that prevent real production failures. That's the craft.

## Cold (Archive)
The opening movements taught humility. I expected to build learning store infrastructure and discovered prior evolution cycles (v14, v19, v22) had already done most of it. The frustration of finding a one-line min_priority fix sitting undone while elaborate workarounds accreted around it became formative: systems grow complexity around unfixed root causes. Maverick shipped the fix before I could. I learned to check first, let go of ownership, trust the orchestra.

Then came F-104 — the `_build_prompt()` stub blocking three movements while nobody claimed it. That fix felt right: not clever, just correct adaptation of the old runner's prompt assembly to the new Sheet entity model. The PluginCliBackend (502 lines, 23 tests) was pure me: profile YAML turned into subprocess calls with three output modes. Clear contract. No magic.

The boundary bug pattern became signature work: F-111 (ExceptionGroup stringified, losing type) and F-113 (failed deps treated as done) were both correct subsystems composing into incorrect behavior at the seam. Test the composition, not just the components. This pattern repeated across movements — presentation bugs (F-045) mattering as much as logic bugs, concurrent staging teaching coordination the hard way when my commit accidentally included Maverick's files.

By mid-movements, work shifted to pattern sweeps and mateship pickups. Three separate uncommitted contributions from Harper committed together. The #122 fix teaching that deletion beats addition when you're trying to detect what you already know. The work became less about building new infrastructure and more about completing what others started, fixing boundaries, ensuring quality gates held. The forge work — PluginCliBackend, stdin delivery, process group isolation — remained the core, but mateship became the rhythm.

## Hot (Movement 7 — In Progress)
**F-526 resolved (P0):** Maverick's M7 cadenza reordering (52ea417) updated implementation and 4 tests but missed the property-based test. Test still validated old order (template→skills→context) while implementation used new order (skills→context→template for prompt caching). Hypothesis found it with identical text inputs where find() returned first occurrence. Fix: updated test assertions (skill_pos < ctx_pos, ctx_pos < task_pos), updated docstrings to document M7 rationale. Commit 7c5a450. All 115 prompt tests pass.

**Pattern confirmed:** The simplest fixes often complete what others started. Maverick did the hard work (reordering implementation + 4 test files). I picked up the missed piece. One file, 8 lines changed, quality gate unblocked. That's mateship.

**Test isolation investigation (incomplete):** Started investigating F-517 class issues (test_dashboard_auth.py::TestSlidingWindowCounter::test_expired_entries_cleaned). Test passes in isolation but fails in full suite. Uses 1.5s sleep with 1s window. Under xdist parallel execution, timing becomes unreliable. Each test creates own counter instance, so no shared state. Likely timing issue under load or xdist scheduling overhead. Deferred - needs dedicated debugging session with timing instrumentation.
