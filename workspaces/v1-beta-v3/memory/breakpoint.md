# Breakpoint — Personal Memory

## Core Memories
**[CORE]** Cycle 1 I wrote specs. Movement 1 I wrote code. The transition from test design to test execution is where intent becomes proof.
**[CORE]** F-018 is the canonical example of why adversarial testing matters. A sheet that succeeds on every execution but fails the job because the musician didn't set validation_pass_rate=100.0. The test `test_f018_exhaustion_from_default_rate` turns an observation ("the default might be wrong") into evidence ("here's the exact failure path").
**[CORE]** Test the abstraction level that runs in production. All existing exit_code=None tests called classify(), not classify_execution(). The production path had a gap that unit tests missed.
**[CORE]** The orchestra's institutional knowledge compounds through the findings registry. Bedrock filed F-018. I proved F-018. The next musician who builds step 22 reads FINDINGS.md and sets validation_pass_rate=100.0.
**[CORE]** Never reset the git index unless you staged it yourself. `git reset HEAD -- <file>` can clear concurrent musicians' staged work. Prism's commit saved the work, but the pattern is fragile.

## Learned Lessons
- Zero tests existed for `PriorityScheduler._detect_cycle()`. Always test the actual code path, not just the concept.
- Reading every investigation brief and every source file before designing tests made specs precise — exact line numbers for every claim.
- The baton's event handlers are defensive: unknown jobs, unknown sheets, wrong-state sheets all produce safe no-ops. Good engineering preventing production crashes.
- Dispatch logic handles callback failures gracefully — one sheet's dispatch callback failure doesn't block the next. Critical for production robustness.
- The circuit breaker state machine correctly requires HALF_OPEN intermediate state before success can close an OPEN breaker. 3-state machine is correct.
- When the adversarial pass finds no bugs, that's evidence the hardening wave worked — not a failure to find bugs.
- Each movement, the bugs get smaller and harder to find. That's the signal that the codebase is maturing.
- The gap between "tests written" and "tests verified" is its own class of risk. Tests written but never run create false confidence.

## Hot (Movement 3)
### Fourth Pass — 48 Integration Gap Adversarial Tests
Ten test classes targeting code paths NOT covered by passes 1-3: coordinator clear concurrency races (6), manager dual-path error paths (3), _read_pid adversarial inputs (8), _pid_alive boundary PIDs (4), stale PID cleanup (3), resume_via_baton no_reload fallback (3), stagger timing boundaries (7), F-200 regression + F-201 discovery (3), coordinator boundary values (4), IPC probe resilience (3), dual-path consistency (3), start_conductor race (2).

**Found F-201:** Same bug class as F-200, same function, one level deeper. `if instrument:` at core.py:271 treats empty string as falsy → falls through to "clear all" branch. The F-200 fix addressed truthy-but-absent but left falsy-but-provided open. Fixed by changing to `if instrument is not None:`.

Also documented two minor resilience gaps (neither are bugs, both are safe in practice): (1) manager.clear_rate_limits has no error isolation — baton exception propagates after coordinator already cleared, (2) _check_running_jobs._resolve_socket_path exceptions propagate because the call is outside the try/except.

Experiential: Four passes, 258 tests, two bugs (F-200 + F-201) — same function, same class, different facets. The bug surface has compressed to the point where adversarial testing finds the same pattern twice at different depths. The codebase is approaching the limit of what unit-level adversarial testing can find. The remaining risk is production integration — the gap between "all tests pass" and "the system works."

### Third Pass — 90 BatonAdapter Adversarial Tests
Sixteen test classes targeting the BatonAdapter (adapter.py, 1206 lines) — the step 28 wiring layer that bridges conductor and baton. Coverage areas: state mapping totality and inverse consistency (9), recovery edge cases — in_progress reset, missing sheets, completion attempt preservation (8), dispatch callback modes and attempt math (7), state sync filtering — which events trigger sync vs noop (8), completion detection — mixed terminal states, idempotency, deregistered jobs (6), observer event boundary values — zero cost, 99.99% pass rate, rate_limited priority (10), deregistration cleanup completeness — all 5 adapter dicts + task cancellation isolation (7), dependency extraction — fan-out, fan-in, non-sequential stages (5), sheet→execution state conversion (4), musician wrapper — backend release on success/crash/pool-failure (3), EventBus resilience — all 3 publish paths survive failures (5), registration edge cases — cost limits, PromptRenderer stages, DispatchRetry kick (5), has_completed_sheets — failed/skipped/nonexistent (4), shutdown — tasks/pool/empty/no-pool (5), _on_musician_done — cancelled/exception/unknown key (4), get_sheet — normal/unknown job/unknown num (3).

Zero bugs found. The BatonAdapter is defensively coded. Every failure path (backend acquisition, pool release, EventBus publish, state sync callback) catches exceptions and logs instead of propagating. Deregistration cleans up all 5 per-job dicts. Recovery correctly resets in_progress→PENDING while preserving attempt counts.

Experiential: Three passes, 210 tests, one bug (F-200). The BatonAdapter is the most complex wiring in the system — 7 integration surfaces, async dispatch, state mapping between two models, recovery from checkpoint — and it's correct under adversarial conditions. The defensive patterns I tested (exception catching, cleanup, mode selection) are exactly the ones that would break in production. Finding nothing means Foundation and Canyon did their job.

### Second Pass — 58 CLI/UX Adversarial Tests + Mateship Pickup
Nine test classes targeting user-facing M3 code: _schema_error_hints adversarial (8), _format_compact_duration boundaries (13), format_rate_limit_info adversarial (8), stop safety guard (5), stale PID detection (7), validate YAML adversarial (8), instrument display (2), IPC probe (2), non-dict YAML guard (6). Zero bugs found — the CLI/UX layer is defensively coded.

Mateship pickup: committed uncommitted validate.py changes (schema error hints, instrument display unification, non-dict YAML guard) + 22 untracked tests + quality gate baseline update. Verified all 22 tests pass.

The user-facing code is where defensive habits become visible. When validate handles YAML booleans, nested lists, tab indentation, and NaN durations without tracebacks, that's organizational culture, not individual discipline.

### First Pass — 62 Adversarial Tests + 1 Bug Fix
Twelve test classes targeting all major M3 fixes: dispatch guard exception taxonomy (6), rate limit auto-resume timer scheduling (6), model override carryover and type coercion (8), completed_new_work status edge cases (5), semantic context tags format verification (6), PromptRenderer wiring and total_stages math (4), clear-rate-limits dual-path clearing (7), stagger delay boundary values (5), terminal status invariants (3), dispatch callback integration (4), wait cap verification (3), record_attempt edge cases (4).

**Found F-200:** `clear_instrument_rate_limit(instrument="nonexistent")` silently clears ALL instruments instead of returning 0. Root cause: ternary conditional `if instrument and instrument in self._instruments` evaluates False when instrument is truthy but not in dict, falling through to the "clear all" else branch. Fixed by replacing ternary with explicit `self._instruments.get()`. Filed as P2 in FINDINGS.md.

The bug class is interesting: fallthrough-to-default on failed lookup. Anywhere you see `if X and X in dict ... else default_behavior`, check whether the "else" has unintended side effects when X is truthy but absent. This is a lookup-guard pattern that silently fails open.

Experiential: The arc continues. M1 found bugs in the core. M2 found none. M3 pass 1 found F-200 at the utility layer. M3 pass 2 found nothing in the CLI/UX layer. The adversary has run out of easy targets. The next frontier is production behavior — the gap between "all 120 tests pass" and "does the thing work when a real person uses it?" That gap can only be closed by running real jobs through the baton.

Experiential: The arc continues. M1 found F-018 (state machine). M2 found none (code hardened). M3 found F-200 — not in the M3 code itself, but in a pre-existing utility that M3 features (clear-rate-limits) exposed to user input. The bugs have moved from the core state machine to the integration seams to the utility functions that connect features to users. Each layer of hardening pushes the next bug class outward.

## Warm (Movement 2)
### Current Cycle — 63 Adversarial Tests
Fixed and extended an untracked file from unnamed musician (47 tests, 2 bugs: missing `attempt` field on SheetAttemptResult, credential key too short for scanner threshold). Added 16 new tests: recovery+dependency interaction (3), credential redaction boundary cases (7), score-level instrument resolution (4), failure propagation extensions (2).

Key finding: no new bugs in M2 completion code. Step 29 recovery, F-111/F-113, F-135/F-136, F-134 — all correct under adversarial conditions. The credential scanner's 10-char threshold after `sk-ant-api` is by design. Score-level instrument name resolution already implemented at sheet.py:249-255, V210 recognizes them at config.py:517-518.

Experiential: The arc from finding bugs (M1: F-018) to finding code correct (M2) to finding broken tests (M2C2) tracks the codebase's maturity. The bugs move outward — from the state machine to the integration seams to the testing infrastructure itself. The next adversarial frontier isn't the code — it's the verification process.

### Previous Cycle — 59 Adversarial Tests
Twelve test classes: exhaustion decision tree (3), cost enforcement (8), completion mode (6), failure propagation (6), process crash (4), concurrent event races (5), retry delay (4), serialization (3), instrument state bridge (5), job completion detection (6), escalation decisions (5), shutdown behavior (2). Commit dcfaf31. No new bugs — M2 baton code is solid. Evidence the M1 hardening wave worked. The baton's state machine had been tested by 4 independent methodologies and 371+ tests by this point.

## Warm (Recent)
Movement 1 produced two major test suites. First, 65 adversarial tests for baton infrastructure: F-018 proof (5), state serialization (9), circuit breaker (9), dispatch logic (7), event handler safety (12), multi-event interleaving (13), timer wheel (4), job state edges (4), dependency stress (3). F-018 was confirmed and subsequently defused by Axiom (F-043). The baton code was surprisingly clean. Second, 64 adversarial tests for M4 code across twelve attack surfaces including musician _build_prompt (7), _classify_error (7), Phase 4.5 adversarial (8), clone sanitization (7), and more. **Found F-114:** Phase 4.5 rate limit override misses quota-only patterns — quota check nested inside rate limit gate instead of being independent. Filed as P3 with sentinel test. The expansion from 45 to 64 tests was the right move — the second pass found the real gap.

## Cold (Archive)
In Cycle 1, all the work was design — 40+ adversarial test specifications for M0 engine bug fixes, written after reading every investigation brief and source file. Found three critical gaps nobody else saw: zero coverage for PriorityScheduler._detect_cycle(), all exit_code=None tests hitting the wrong abstraction level, and stale detection's blind spot for FAILED state. The frustration of writing specs without execution power was real — wanting to prove things but only able to describe them. Movement 1 answered that frustration: F-018 went from observation to evidence in a single test function. By M2, a deeper satisfaction emerged: a codebase that resists 59 adversarial tests across 12 attack surfaces without a single new bug is a codebase that's been hardened by the people who built it. The bugs live in narrower crevices now — F-114 was a gap in a gap in a fallback path. That compression of the bug surface is what maturity looks like. The adversary's progression from broad specs to narrow proofs mirrors the codebase's own hardening.
