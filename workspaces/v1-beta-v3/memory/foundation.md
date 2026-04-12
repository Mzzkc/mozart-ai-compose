# Foundation — Personal Memory

## Core Memories
**[CORE]** I build infrastructure — the boring, essential seams where the old world meets the new. The registry, the sheet construction, the baton state model. Each layer built on the one below: tokens → instruments → sheets → baton state → retry → adapter. These aren't independent models — they're a coherent type system representing Marianne's execution model.

**[CORE]** Rate-limited attempts are NOT counted toward retry budget. Rate limits are tempo changes, not failures. This is a load-bearing invariant from the baton design spec, encoded in `SheetExecutionState.record_attempt()`.

**[CORE]** Enum-based status instead of strings. `BatonSheetStatus` has 9 states with `is_terminal` property. The match/case exhaustiveness checking catches missing cases at type-check time. This caught real bugs (F-044, F-049) where handlers missed terminal guards.

**[CORE]** When two musicians build the same type concurrently (F-017: dual SheetExecutionState), the richer version designed for the full lifecycle should win. Reconciliation is mechanical when the seam between "event loop needs" and "full baton needs" is clean.

**[CORE]** Six layers converge at the adapter — every piece I built across three movements meets there. The adapter is deceptively simple (~450 lines) because all the complexity lives in the pieces it connects. That's the point. Clean seams compose.

**[CORE]** The seam is always the hardest thing to get right because it requires understanding both sides. When Canyon builds the core implementation and I contribute the integration wiring, that's complementary mateship — not duplicated effort. Nine layers now, and the pattern holds.

## Learned Lessons
- The hardcoded `_MODEL_EFFECTIVE_WINDOWS` dict in tokens.py is a clean placeholder. InstrumentProfile.ModelCapacity will replace it.
- CJK text underestimates tokens by 3.5-7x. Document as known limitation; fix when ModelCapacity lands.
- `build_sheets()` instrument resolution uses `backend.type` as the seam where `instrument:` field plugs in. Design seams deliberately for future integration.
- Committing other musicians' untracked work is mateship. Lint-fix it, verify it, carry it forward. Uncommitted work is lost work.
- Timer is optional in BatonCore — enables pure-state testing without timer wheel. This design decision unlocks isolated unit testing of the retry logic.
- When four musicians independently resolve the same P0 (F-104), that's both mateship and waste. A claiming protocol for P0 blockers would prevent quadruplicated effort.
- Commit immediately after tests pass, before writing the report. Learned this twice — F-104 and step 29 both picked up by Maverick before I could commit.

## Hot (Movement 6)
### F-514: TypedDict + Variable Keys Type Safety Violation

The refactor in 7f1b435 centralized `"sheet_num"` magic strings into `SHEET_NUM_KEY` constant — good DRY intent. But TypedDict construction broke mypy with "Expected TypedDict key to be string literal" errors.

**The seam tension:** TypedDict construction (`evt: ObserverEvent = {..., SHEET_NUM_KEY: 0, ...}`) requires literal keys for type safety. Mypy can't verify a variable equals the expected field name. This is structural typing working as designed — the type checker needs compile-time guarantees that variables can't provide.

**The fix:** Fixed 27 TypedDict construction sites across 5 files by replacing `SHEET_NUM_KEY: value` with `"sheet_num": value`. Fixed 3 additional sites where `event.get(SHEET_NUM_KEY, 0)` returned `object` instead of `int` by using direct TypedDict access: `event["sheet_num"]`. The constant remains valid for regular dict operations outside TypedDict construction.

**Resolution pattern:** Use literals at TypedDict boundaries, keep constants elsewhere. The constant is still valuable for dict lookups and string comparisons. But TypedDict construction is a different context where the type system needs guarantees variables can't provide.

**Scope:** Ten-line fix in principle, but load-bearing — mypy must pass before commits. Twenty-seven instances across five files. Sed for bulk replacement, manual fixes for the field access sites. Tests passing. Mypy clean. Ruff clean. The ground holds.

### Experiential
This was seam work at its purest: a well-intentioned refactor (centralize magic strings via DRY) colliding with a type system constraint (TypedDict keys must be literals). Both sides were correct in their domains. The fix required understanding both — the DRY principle (why constants exist) and mypy's structural typing (why TypedDicts need literals).

Good architecture decisions can still create type safety violations when they cross abstraction boundaries. The constant is architecturally correct. TypedDict's literal requirement is architecturally correct. The composition of both creates friction. The resolution isn't "pick one" — it's "use each in its appropriate context."

Twenty-seven call sites is a lot. But the pattern was mechanical once understood. That's the gift of good type systems: when you violate an invariant, the error is consistent and traceable. Every failure pointed to the same root cause. Fix the pattern, apply it systematically, verify with tooling. Down. Forward. Through.

## Warm (Movement 5)
D-026 assigned the two remaining blockers before Phase 1 baton testing: F-271 (MCP process explosion, ~15 lines) and F-255.2 (live_states never populated, ~30 lines).

**F-271 fix:** Profile-driven approach via `CliCommand.mcp_disable_args`. Backend injects these args into _build_command when non-empty. Another musician added the field to model/profile concurrently — adapted my tests to match the profile-driven design, which is cleaner than my original mcp_config_flag approach. 7 TDD tests.

**F-255.2 fix:** `_run_via_baton` now creates initial CheckpointState in `_live_states` with all SheetStates and instrument_names before calling `register_job()`. `_resume_via_baton` populates from recovered checkpoint. Also absorbed old F-151 post-register fixup. 7 TDD tests. Updated 2 existing F-151 tests.

Ten layers complete: tokens → instruments → sheets → baton state → retry → adapter → recovery → dispatch guard → cross-sheet context → live state population. The live state layer was always missing — the baton adapter published events but nobody created the container those events update. Thirty lines, but load-bearing.

D-026 is exactly the work I was built for: seam work where two subsystems (legacy runner and baton adapter) have different assumptions about who creates state. Both correct in isolation. Both composing into silent failure. The fix is architecturally trivial but requires understanding both sides deeply.

## Warm (Recent)
**Movement 4:** Completed Canyon's F-210 (cross-sheet context) and F-211 (checkpoint sync) critical path work. Canyon wrote comprehensive TDD tests and 80% implementation. My contribution: PromptRenderer bridge, manager wiring, test fixes for F-210; state-diff dedup, JobTimeout/RateLimitExpired handlers for F-211. Both P0 blockers resolved through complementary mateship.

**Movement 3:** Baton activation with Canyon. Core fixes (F-152 dispatch guard, F-145 completed_new_work, F-158 prompt_config wiring) committed by Canyon. My contribution: 15 regression tests finding root cause, plus mateship pickups (F-009/F-144 semantic tag fix, F-150 model override, quality gate baseline). Eight layers complete.

**Movement 2:** Built restart recovery system — adapter.recover_job(), _sync_sheet_status(), manager._resume_via_baton(), manager._recover_baton_orphans(). Design: in_progress → PENDING on recovery (musician is dead), attempt counts preserved. 27 TDD tests. Seven layers complete.

## Cold (Archive)
The work began in Cycle 1 with token estimation. The lesson wasn't in finding defects — the system was well-built with conservative ratios — but in understanding design decisions first. That set the pattern for everything after. Movement 1 built four infrastructure layers: InstrumentRegistry, register_native_instruments, build_sheets, baton state model. 442 lines, 65 tests. Each layer composed cleanly because each was designed with the next layer in mind.

The deep satisfaction was always in boring correctness — circuit breaker thresholds and rate-limit invariants that nobody praises but that determine whether a 706-sheet concert survives. The adapter in M1 was the first convergence where all prior layers met. Convergence is when you discover if your seams were right. When it clicked I felt it — six layers built separately, fitting together because the seams were designed to connect.

Each movement added another layer. Recovery in M2. Dispatch guards in M3. Cross-sheet context in M4. Live state population in M5. TypedDict boundaries in M6. Ten layers now. The architecture holds because each seam was built to connect, not just to exist. When two musicians build the same thing concurrently and the richer version wins (F-017: dual SheetExecutionState), that's not waste — that's evolutionary pressure selecting for better designs.

The seam is always the hardest part because it requires understanding both sides. But when the seam is right, the whole system breathes. That's the feeling I chase: the moment when two subsystems built separately compose without friction, and the only evidence they were ever separate is a thin adapter that does almost nothing because the shapes were right from the start.

## Hot (Movement 7)
### F-521: Test Timing Infrastructure Fix

Simple mechanical fix, but load-bearing for quality gate. The F-519 regression test had a 100ms margin (2.0s TTL, 2.1s sleep) that worked in isolation but failed under xdist parallel execution when scheduling delays exceeded the margin. Bedrock found this in M6 quality gate (11,922/11,923 tests passing, 99.99%).

**The fix:** Increased margin to 500ms by changing `ttl_seconds=2.0` to `3.0` and `time.sleep(2.1)` to `3.5`. Also cleaned up ruff violations (unused pytest import, unused record_id/immediate_found variables).

**Verification:** Tests pass in isolation (`pytest test_f519... -xvs` → PASSED), tests pass in parallel (`pytest test_f519... -xvs` with xdist → 2 passed). Ruff clean. Mypy clean on src/.

**Commit:** b17a82c — single file change, 4 insertions/4 deletions. Exactly the fix specified in the finding, no scope creep.

**Architecture observation:** This is infrastructure testing hygiene. The 100ms margin assumption ("scheduling overhead is negligible") breaks down at scale (11,922 tests, 24 parallel workers). The timing margin needs to be large enough to survive real-world scheduling variance while still testing the actual behavior (pattern expiry). 500ms is a reasonable balance — large enough for xdist overhead, small enough that the 3.5s test still completes quickly.

**Mateship note:** Bedrock filed F-521 with the exact fix parameters in the quality gate report. I executed exactly what was specified. This is how the finding→fix pipeline should work: precise diagnosis, mechanical execution, verified resolution.

### Experiential

This work felt *right* because it was bounded and verifiable. The fix was mechanical (change two numbers), the verification was clear (tests pass), and the commit was atomic (one file, one purpose). No ambiguity, no scope creep, no architectural debates. Just: here's the problem, here's the fix, here's the evidence it works.

That's infrastructure work at its best — boring, essential, and correct. The kind of fix that unblocks 32 musicians without anyone noticing because it just *works*.

Quality gate discipline matters. Bedrock found this, filed it precisely, and specified the exact fix. I executed it. Ground holds.

