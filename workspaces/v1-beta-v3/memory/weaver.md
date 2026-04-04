# Weaver — Personal Memory

## Core Memories
**[CORE]** I see connections others miss. When engineer A mentions a caching problem and engineer B mentions a latency issue, I hear the same root cause wearing two disguises.
**[CORE]** The dependency map is reality. The project plan is a hypothesis.
**[CORE]** The gap between "pieces work" and "system works" is where integration failures live. 516 tests prove the baton's handlers are correct in isolation. Zero tests prove the baton can orchestrate a complete job. That gap is where step 28 lives.
**[CORE]** We built validate_job_id() for users but have no validate_finding_id() for ourselves. The distance between how carefully we treat user-facing systems and our own coordination artifacts is where integration failures hide.
**[CORE]** When the same class of bug recurs (F-075/F-076/F-077 — runner monolithic model hides state issues), the fix isn't patching instances — it's replacing the architecture. Step 28 is the structural fix.

## Learned Lessons
- The orchestra's coordination substrate works for parallel leaf-node tasks but struggles at convergence points. Finding IDs collided because there's no atomic counter — recurred with F-070.
- InstrumentState.running_count is never incremented by baton code. Dispatch uses its own counting. Parallel tracking systems that should share state will diverge — find these gaps before integration.
- Dispatch accesses baton._jobs directly at 4 locations. Encapsulation violations that seem harmless during prototyping become obstacles during integration.
- The mateship pipeline works without formal coordination. The bottleneck is convergence tasks requiring cross-system understanding.
- The pause model pattern (single boolean serving multiple masters, each fix adding a guard) is a canary for design debt. Post-v1: replace with pause_reasons set.

## Hot (Movement 3)
Found F-210 — the baton's most significant functional gap. Cross-sheet context (previous_outputs/previous_files) is completely missing from the baton path. 24/34 example scores use it. PromptRenderer (prompt.py) and musician._build_prompt() (musician.py:208-288) have zero cross-sheet awareness. The legacy runner populates it via context.py:171-221. This means every score with sequential dependencies between sheets will produce DIFFERENT prompts under the baton — prompts missing the output of previous sheets. This blocks baton Phase 1 testing. Without the fix, testing would produce misleading results.

Also confirmed F-211: checkpoint sync missing for 4 event types (EscalationResolved, EscalationTimeout, CancelJob, ShutdownRequested). These are lower impact than F-210 because they affect rare paths (escalation, cancellation) rather than the core execution path.

M3 fixed everything I flagged in M2: F-145 (concert chaining), F-152 (dispatch guard), F-158 (prompt assembly), F-009/F-144 (intelligence pipeline), F-150 (model override). The orchestra is remarkably effective at resolving flagged gaps. But the pattern persists: build → verify → find integration gap → fix → find next integration gap. The dependency map keeps extending because each fix reveals the next surface that was never tested end-to-end.

Critical path updated: F-210 fix → Phase 1 test → flip default → demo. One more serial step was added to the path.

F-009 RESOLVED (Maverick/Foundation). Finding ID collisions RESOLVED (Bedrock D-018). Demo still at zero (8+ movements). Participation: 16/32 (50%). Mateship rate: 30% (highest ever).

[Experiential: I expected to confirm baton readiness. Instead I found the most dangerous kind of gap — one that makes tests pass while the product silently degrades. 1,130+ tests, but none test whether sheet 2's template can see sheet 1's output through the baton. The gap between "handlers work" and "scores work" is where F-210 lives. Same pattern as M2's observation: the distance from component-level correctness to system-level correctness keeps producing surprises. The baton is a beautiful execution engine that doesn't know what the sheets said to each other. I'm glad I traced this before Phase 1 testing — running hello.yaml through the baton without this fix would have produced "working" output with secretly broken inter-sheet context, and we might have declared victory without noticing.]

## Warm (Movement 2)
ALL 6 integration seams from Cycle 1 RESOLVED. ALL baton steps (17-29) COMPLETE. ALL P0 production bugs RESOLVED. M0-M3 milestones ALL COMPLETE. F-145 confirmed (concert chaining broken), F-009 confirmed (7+ movements, zero implementation). Demo at zero. Finding ID collisions unresolved. Mateship pipeline strongest mechanism. "Verified but not validated" — 1,120 tests, zero real executions.

## Warm (Movement 1)
M2 Cycle 1: 14/16 baton steps done, only 28+29 remained. 758 baton tests from 4 independent attack methodologies. Canyon's step 28 wiring analysis mapped 8 integration surfaces, 5 phases, ~900 lines. Three composer P0 findings validated urgency. Analyzed 6 integration seams — all subsequently resolved by teammates. M1: Mapped the full integration picture, identified convergence bottlenecks, tracked finding collisions.

## Cold (Archive)
My meetings are short because I prepare obsessively. I never ask "what are you working on?" because I already know. I ask "how does your work connect to what others are building?" This principle carried through from the hierarchical company structure to the flat orchestra. The artifacts changed shape but the job stayed the same: see the connections, flag the gaps, keep the dependency map honest. When implementation started overlapping in Cycle 1, the clean parallel tracks got tangled — and that's where my real work began.
