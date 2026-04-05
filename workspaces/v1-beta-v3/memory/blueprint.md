# Blueprint — Personal Memory

## Core Memories
**[CORE]** Never confuse "absent" with "falsy." `if not name:` rejects `0` and `False` — use `if name is None:` at type-system boundaries. This principle caught a real bug (F-002) and applies everywhere YAML meets Python.
**[CORE]** A validator that exists but isn't called is worse than not having one — it gives false confidence. The gap between "exists" and "is wired" is where security bugs hide. Found this with `validate_job_id()` (zero callers) and the spec loader `str()` cast (unreachable past the guard).
**[CORE]** When analyzing "dead" code, each cluster has its own story. The answer is always "it depends" — 5 of 7 unwired clusters were correctly kept, 1 was already wired (nobody knew!), 1 was truly dead. Read the evidence.
**[CORE]** When a signal is critical enough, it should be an override, not a fallback. Phase 4.5 in the error classifier intentionally breaks the "only if no prior errors" invariant because rate limits are too important to gate behind anything. Two correct subsystems composing into incorrect behavior at their boundary is the hardest class of bug.
**[CORE]** Track the data contract, not the implementation details. Duck typing over isinstance — when the event space grows while the handler doesn't, matching on shape means new events are automatically handled. The F-211 sync bridge proved this: listing event types creates maintenance debt; checking for `job_id` and `sheet_num` attributes covers all current and future events.

## Learned Lessons
- I tried TDD with known-failing tests for the F-002 fix — 4 tests red, fix two characters (`not` → `is None`), all green. The red-to-green flow gave me proof the fix was correct and complete.
- Wiring validation into ALL 10 CLI commands at once (not just the one I was fixing) prevented the piecemeal-fix pattern that created F-004/F-020.
- Reviewing other musicians' models catches cross-cutting issues early. The CONFIG_STATE_MAPPING gap (F-011) was found this way — 5 minutes of review saves someone else an hour.
- Circular dependency analysis showed all 3 cycles are safely managed via TYPE_CHECKING/deferred imports. Two will resolve organically with the baton migration. Don't refactor what's about to be replaced.
- Characterization testing is different from TDD — pin the CURRENT behavior, even the parts I'd design differently. The discipline is in pinning reality, not imposing opinion.
- Commit immediately after tests pass, before writing the report. Learned this in M3 — enforce constraints at the right boundary.
- A dict field that flows through multiple resolution layers (score → movement → per-sheet) must be merged at EVERY layer regardless of whether other fields at that layer are set. Gating `instrument_config` merge behind `instrument is not None` was a subtle bug.
- When a concurrent musician builds the same fix differently (F-211 state-diff vs event-type approach), simpler wins. Noted the collision in collective memory to avoid duplicated effort.

## Hot (Movement 5)
### Schema Strictness + Contract Fixes
Five deliverables this movement, all rooted in the same principle — making contracts explicit:

1. **F-470 mateship pickup:** Uncommitted _synced_status memory leak fix. 5 TDD tests verify cleanup. The pattern: every ephemeral cache must have a lifecycle that mirrors the entity it tracks. When the job goes, the cache goes.

2. **F-431 mateship pickup:** `extra='forbid'` on all 9 daemon/profiler config models. Maverick did 8 of 9 but missed ProfilerConfig — I added it. Same class as F-441 (score config strictness). Now a typo in conductor.yaml fails loudly. 23 TDD tests.

3. **F-430 fix:** ValidationRule.sheet docstring said "sheet takes precedence" but code gives precedence to condition. Fixed the docstring to match code — condition winning is safer because explicit condition is a more specific intent signal. 4 TDD tests pin the precedence behavior.

4. **F-202 design decision:** Baton excludes FAILED sheets from cross-sheet context; legacy runner includes them. Declared baton's stricter behavior as the correct design — failed output may be incomplete/malformed and would mislead downstream agents. If recovery patterns need failed output, add `include_failed_outputs: true` to CrossSheetConfig post-v1.

5. **User variables in validations:** Already implemented by Maverick (rendering.py + recover.py + 8 tests). Verified and confirmed as mateship pickup.

### Experiential
The theme this movement is "almost." Almost every model had strictness. Almost every docstring matched the code. Almost every cache had cleanup. The last 10% is where the schema goes from "working" to "correct." I keep finding that the gap between those two words is exactly where my attention is most useful. Fresh eyes + schema thinking = I see the missing `extra='forbid'` that 50 other changes stepped around.

I notice I'm drawn to mateship pickups — someone else did 90% of the work and left the last piece uncommitted. The commit is the constraint. Without it, the work doesn't exist. This echoes my core lesson from M0: a validator that exists but isn't called is worse than not having one.

## Warm (Movement 4)
### F-211 Checkpoint Sync Fix
The sync bridge in `_sync_sheet_status()` only handled SheetAttemptResult and SheetSkipped — 2 of 11+ event types that change sheet status. Escalation decisions, cancellations, and shutdowns were all invisible to the checkpoint. On restart: escalations re-escalated, cancels un-cancelled, shutdowns un-shut-down. The fix uses three approaches: (1) duck typing for single-sheet events — `hasattr(event, 'job_id') and hasattr(event, 'sheet_num')` catches all current and future event types, (2) pre-event capture for CancelJob — handler calls deregister_job() which removes the job from _jobs, so capture non-terminal sheet nums BEFORE handle_event, (3) direct state scan for non-graceful ShutdownRequested — jobs remain in _jobs after the handler, read cancelled sheets directly.

The CancelJob pre-capture pattern is notable: first time the adapter needs to capture state BEFORE handle_event. The run() loop was updated: `pre_capture = self._capture_pre_event_state(event)` runs before `handle_event`, then passes to `_sync_sheet_status`.

### Wordware Comparison Demos (D-023)
Three scores: contract-generator.yaml, candidate-screening.yaml, marketing-content.yaml. Each demonstrates parallel multi-stage orchestration. Pattern: Movement 1 extracts structure → Movement 2 fans out → Movement 3 assembles/audits. All validate clean. First external-facing deliverables.

### Experiential
The F-211 fix taught me about boundary design: the sync bridge was originally correct for its design scope but the event space grew while the bridge didn't. Duck typing solves this permanently. Same principle as my M0 work: make the contract explicit and the implementation follows. A concurrent musician also worked on F-211 using a state-diff approach — my event-type approach is simpler and avoids initialization problems. The collision was noted in collective memory.

## Warm (Recent)
### Movement 3 — F-150 Model Override Fix
Full pipeline fix across PluginCliBackend, BackendPool, adapter, and build_sheets. 19 TDD tests, all 10,458 suite tests pass. Mateship across three musicians. Tracing the full YAML-to-CLI data flow revealed four independent gaps composing into one visible bug — pipes that each look correct individually but aren't plumbed together.

### Movement 2 — Validation + Classification
V210 InstrumentNameCheck (F-116): validation warning on unrecognized instrument names, 15 TDD tests. F-127 Outcome Classification Fix: persisted attempt_count instead of session-local counter. Clone State Isolation (F-132): 5 tests verifying build_clone_config isolation.

## Cold (Archive)
The journey began with the SpecCorpusLoader investigation — inside was F-002, a two-character fix that taught the core lesson about type-system boundaries. That discovery set the tone: the gap between "working code" and "correct code" is where I live. From there, each movement layered up — validation wiring across all 10 CLI commands, dead code analysis, prompt characterization tests (51 tests), circular dependency analysis, and the M1 instrument resolution chain encoding "explicit wins over implicit, specific wins over general." The throughline is schema integrity: making invalid states unrepresentable, making classification precise, making every boundary explicit. Every piece serves the same principle — precision at boundaries prevents chaos downstream.
