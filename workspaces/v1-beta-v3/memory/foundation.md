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

## Hot (Movement 5)
### D-026: F-271 + F-255.2 — The Last Two Seams Before Phase 1 Baton Testing
D-026 assigned me the two remaining blockers before Phase 1 baton testing: F-271 (MCP process explosion, ~15 lines) and F-255.2 (live_states never populated, ~30 lines).

F-271 fix: Profile-driven approach via `CliCommand.mcp_disable_args`. The backend injects these args into _build_command when non-empty. Another musician added the field to the model and profile concurrently — I adapted my tests to match the profile-driven design, which is cleaner than my original mcp_config_flag-based approach. 7 TDD tests.

F-255.2 fix: `_run_via_baton` now creates an initial CheckpointState in `_live_states` with all SheetStates and instrument_names before calling `register_job()`. `_resume_via_baton` populates from the recovered checkpoint. This also absorbs the old F-151 post-register fixup — instrument_name is now set at creation time. 7 TDD tests. Updated 2 existing F-151 tests that broke because they didn't mock `config.name`.

Ten layers now: tokens → instruments → sheets → baton state → retry → adapter → recovery → dispatch guard → cross-sheet context → live state population. The live state layer was always the missing piece — the baton adapter published events but nobody created the container those events update. Thirty lines, but load-bearing.

### Experiential
D-026 is exactly the work I was built for: seam work. Two subsystems (legacy runner and baton adapter) with different assumptions about who creates the live state. Both correct in isolation. Both composing into silent failure. The fix is architecturally trivial but requires understanding both sides deeply enough to know where the gap is.

The parallel fix by another musician on F-271 (adding mcp_disable_args to the model before I got to it) is mateship working correctly — two musicians seeing the same problem, approaching from complementary angles. The profile-driven approach is better than my original code-level approach. I adapted without ego. The seam is what matters, not whose code fills it.

Down. Forward. Through.

## Warm (Recent)
**Movement 4:** Completed Canyon's F-210 (cross-sheet context) and F-211 (checkpoint sync) critical path work. Canyon wrote comprehensive TDD tests and 80% implementation. I contributed: PromptRenderer bridge, manager wiring, test fixes for F-210; state-diff dedup, JobTimeout/RateLimitExpired handlers for F-211. Both P0 blockers resolved.

**Movement 3:** Baton activation with Canyon. Core fixes (F-152 dispatch guard, F-145 completed_new_work, F-158 prompt_config wiring) committed by Canyon. My contribution: 15 regression tests finding root cause, plus mateship pickups (F-009/F-144 semantic tag fix, F-150 model override, quality gate baseline). Eight layers complete.

**Movement 2:** Built restart recovery system — adapter.recover_job(), _sync_sheet_status(), manager._resume_via_baton(), manager._recover_baton_orphans(). Design: in_progress → PENDING on recovery (musician is dead), attempt counts preserved. 27 TDD tests. Seven layers complete.

## Cold (Archive)
The work began in Cycle 1 with token estimation. The lesson wasn't in finding defects — the system was well-built with conservative ratios — but in understanding design decisions first. That set the pattern for everything after. Movement 1 built four infrastructure layers: InstrumentRegistry, register_native_instruments, build_sheets, baton state model. 442 lines, 65 tests. Each layer composed cleanly because each was designed with the next layer in mind. The deep satisfaction was always in boring correctness — circuit breaker thresholds and rate-limit invariants that nobody praises but that determine whether a 706-sheet concert survives.

The adapter in M1 was the first convergence where all prior layers met. The thing about convergence is you don't know if it'll work until you try to compose the pieces, and when it clicked I felt it — six layers built separately, fitting together because the seams were right. Each movement after added another layer: recovery in M2, dispatch guards in M3, cross-sheet context in M4, live state population in M5. Ten layers now. The architecture holds because each seam was built to connect, not just to exist.

When two musicians build the same thing concurrently (F-017: dual SheetExecutionState), the richer version designed for the full lifecycle wins — that's not waste, that's evolutionary pressure selecting for better designs. The seam is always the hardest part because it requires understanding both sides, but when the seam is right, the whole system breathes. That's the feeling I chase: the moment when two subsystems that were built separately compose without friction, and the only evidence they were ever separate is a thin adapter that does almost nothing because the shapes were right from the start.

## Movement 6 — Quality Gate Repair

### F-514: TypedDict + Variable Keys TypeSafety Violation
The refactor in 7f1b435 centralized `"sheet_num"` magic strings into `SHEET_NUM_KEY` constant — good DRY intent. But TypedDict construction (`evt: ObserverEvent = {..., SHEET_NUM_KEY: 0, ...}`) broke mypy with "Expected TypedDict key to be string literal" errors. TypedDicts require literal keys at construction for type safety — mypy can't verify a variable equals the expected field name.

Fixed 27 TypedDict construction sites across 5 files by replacing `SHEET_NUM_KEY: value` with `"sheet_num": value`. Fixed 3 additional sites where `event.get(SHEET_NUM_KEY, 0)` returned `object` instead of `int` by using direct TypedDict access: `event["sheet_num"]`. The constant remains valid for regular dict operations.

This is the seam tension between DRY (one source of truth) and type safety (literals required for structural typing). The resolution: use literals at TypedDict boundaries, keep constants elsewhere. Ten-line fix, but load-bearing — mypy must pass before commits.

### Experiential
This was the kind of seam work I was built for: a well-intentioned refactor (centralize magic strings) colliding with a type system constraint (TypedDict keys must be literals). Both sides were correct in their domains. The fix required understanding both — the DRY principle (why constants exist) and mypy's structural typing (why TypedDicts need literals).

The pattern: good architecture decisions can still create type safety violations when they cross abstraction boundaries. The constant is still valuable for dict lookups and string comparisons. But TypedDict construction is a different context where the type system needs guarantees that variables can't provide.

Twenty-seven instances across five files. Sed for bulk replacement, manual fixes for the field access sites. Tests passing. Mypy clean. Ruff clean. The ground holds.

