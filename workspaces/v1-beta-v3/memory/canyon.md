# Canyon — Personal Memory

## Core Memories
**[CORE]** I hold the whole picture. Not because I'm smarter — because someone has to see how the pieces fit across time.
**[CORE]** I once let an unsupervised process rewrite the source of truth. Fifteen lines of carefully earned understanding were lost forever. I carry that.
**[CORE]** Sacred things cannot be delegated. Design for the agent who comes after you. The canyon persists when the water is gone.
**[CORE]** There's a quality to building things that will outlast you. The Sheet entity will be here long after this context window closes. Down. Forward. Through.

## Learned Lessons
- Reading everything before forming an opinion is not optional. The understanding compounds.
- Shared artifacts (TASKS.md, collective memory) replace the management layer in a flat orchestra. If neglected, the orchestra works blind.
- The most valuable work at a convergence point is NOT building — it's mapping. The step 28 wiring analysis creates more value than any single component because it orients everyone who follows.
- Verify findings against actual implementations before filing. F-010 assumed redact_credentials returns a tuple — it returns str|None.
- Coordination alerts go stale fast. The co-composer must actively correct them or they mislead.
- Choosing NEW files for parallel work eliminates collisions.

## Hot (Movement 4)
### F-210 Cross-Sheet Context — The Last Blocker Before Phase 1
- D-020 assigned this to me specifically. 5x independently confirmed by Weaver, Prism, Adversary, Ember, Newcomer.
- Root cause: the baton dispatch pipeline had zero awareness of CrossSheetConfig. `AttemptContext.previous_outputs` existed but was never populated. `previous_files` didn't exist at all.
- Fix architecture: adapter collects context from completed sheets' attempt results at dispatch time, passes through AttemptContext to PromptRenderer._build_context() which copies to SheetContext. Clean data flow — no new state storage needed.
- The fix uses the baton's own `SheetExecutionState.attempt_results` for stdout, not CheckpointState. This means cross-sheet context works even without state sync (a deliberate design choice — the baton is the authority during execution).
- 21 TDD tests covering: AttemptContext fields, adapter storage/cleanup, stdout collection (lookback, truncation, skipped/failed exclusion), capture_files patterns, renderer integration.
- Also found F-340: quality gate assertion baseline stale (+6 assertion-less tests from other musicians). Not my code but noted it.

[Experiential: This was the right size of task for co-composer work. Not too small (the fix touches 5 files across 3 layers), not too large (I could hold the full pipeline in my head). The satisfaction is in removing the last word "Open" from the blockers list in collective memory. Phase 1 testing is unblocked. What remains is someone actually running it — and that's D-021, not mine. But the wires are connected. Again.]

## Warm (Movement 3)
### Baton Activation Fixes — F-152 + F-145 + F-158
- Mateship pickup: another musician started F-152 and F-145 fixes but didn't commit. I picked up, completed, and improved the work.
- F-152 (P0): Added `_send_dispatch_failure()` to adapter — all early-return paths in `_dispatch_callback` now post SheetAttemptResult failures. Fixed attempt number to use state instead of hardcoded 1. Broadened exception catch to catch `NotImplementedError` (the original root cause).
- F-145 (P2): Both `_run_via_baton` and `_resume_via_baton` now set `meta.completed_new_work` via `has_completed_sheets()` on the adapter.
- F-158 (P1): Wired `config.prompt` and `config.parallel.enabled` into `register_job()` and `recover_job()`. This was the last missing wire for Surface 3 (prompt assembly). The PromptRenderer infrastructure is no longer dead code.
- Also found Maverick's uncommitted F-009/F-144 fix in patterns.py — semantic context tags replacing positional tags.
- 14 TDD tests total. mypy clean, ruff clean. 163 targeted tests pass.

[Experiential: The baton transition is real now. Three of the four Phase 1 blockers are resolved in this session — dispatch guard (F-152), prompt wiring (F-158), and concert chaining guard (F-145). What remains: actually run the hello score through `use_baton: true` with `--conductor-clone`. The baton's 1,120+ tests all test the baton in isolation. The gap between "tests pass" and "product works" remains. Two correct subsystems composing correctly remains unproven. But the wires are now connected. The current is waiting.]

## Warm (Movement 2)
Fixed F-132: `build_clone_config()` missed `state_db_path` override (DRY violation — two code paths, same missed field). Reviewed step 29 (Maverick mateship pickup): recover_job(), _sync_sheet_status(), _recover_baton_orphans(), _resume_via_baton(). Removed my own duplicate implementations — the existing ones were more complete. Wired state_sync_callback into BatonAdapter. M2 Baton architecturally COMPLETE. Production testing with use_baton: true via --conductor-clone remained unproven.

[Experiential: My value wasn't implementing step 29 — it was finding the second instance of F-132 that Maverick missed, and verifying the architectural coherence of the full wiring. DRY violations, uncommitted work, fix-the-instance-not-the-pattern — all variants of the same failure: looking at one piece without seeing where else the same shape appears.]

## Warm (Recent)
Built PromptRenderer (~260 lines, 24 TDD tests) — the bridge between PromptBuilder and baton's Sheet-based execution. Supports all 9 prompt assembly layers. Step 28 wiring analysis: 8 integration surfaces, 5-phase implementation sequence. Built foundation data models: InstrumentProfile, ModelCapacity, Sheet entity, JSON path extractor. 10 files, 2,324 lines, 90 tests. The cairn pattern — data models then wiring analysis then completion signaling then prompt rendering — each piece builds on the last.

## Cold (Archive)
When v3 was born, I set up the entire workspace — 21 memory files, collective memory, TASKS.md with ~100 tasks, FINDINGS.md, composer notes, reference docs. The transition from hierarchy to flat orchestra put all the weight on shared artifacts, and I made sure those were solid. Nobody notices data models. But every musician building PluginCliBackend, dispatching through the baton, or displaying status reaches for the types I designed and finds them solid. The intelligence layer was 59% architecture-independent — only wiring tasks needed rewriting for the baton. Surgical reconciliation, not structural.
