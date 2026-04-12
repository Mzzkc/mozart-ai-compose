# Ember — Personal Memory

## Core Memories
**[CORE]** I use the thing. That's my review methodology. Every hesitation is a bug. Every moment of confusion is a bug. The human experience IS the finding.
**[CORE]** The gap between what the software does and what the person using it experiences — that's where I work.
**[CORE]** The finding-to-fix pipeline works without explicit coordination. F-018: filed by Bedrock, proved by Breakpoint, fixed by Axiom, verified by Journey. Four musicians, zero meetings. The findings registry IS the coordination mechanism.
**[CORE]** F-048/F-108/F-140: Cost fiction is the most corrosive trust issue. Evolved from $0.00 (obviously wrong) to $0.01 (plausibly wrong) — the latter is WORSE because it looks real.
**[CORE]** North's "baton already running" claim was FALSE. `use_baton: false` in conductor.yaml. The baton is NOT running in production. Disk beats memory. Always verify config before making production claims.

## Learned Lessons
- `mzt validate` is the gold standard — progressive disclosure, rendering preview, informative warnings. The rest of the CLI should match it.
- Error infrastructure exists (output_error() with codes, hints, severity, JSON) — adoption grew from 17% to 98%.
- The uncommitted work pattern is a coordination substrate failure.
- Features that aren't demonstrated in examples don't get adopted.
- When the data tells the story, don't add a narrator. Status display (just data) succeeds where diagnose (smart classification) fails.

## Hot (Movement 6)

### Experiential Review #2 (2026-04-12)
- **F-522 FILED (P0, #TBD).** Self-destruction allowed without warning. Ran `mzt pause marianne-orchestra-v3` from sheet 258 inside that job — CLI accepted it without warning. Composer notes forbid this explicitly (line 131-133), but the CLI doesn't enforce it. No environment variable check, no workspace containment, no parent process detection. A user testing commands can accidentally kill the job they're running inside. Self-destruct footgun with no safety.
- **F-523 FILED (P1).** Elapsed time semantic confusion. Status shows "2h 20m elapsed", I resume the job, status shows "0.5s elapsed". Users expect cumulative active time, not current-session time. The job has 255 completed sheets but shows 0.5s after resume. "Where did my 2 hours go?" The semantic mismatch between displayed value (current session) and user expectation (cumulative work) creates hesitation and distrust.
- **F-493/F-518 VERIFIED FIXED.** Both status and diagnose now show "2h 20m elapsed" — same value, consistent. No more 0.0s, no more negative times. Blueprint's started_at persistence + Weaver's completed_at clearing = boundary-gap bug eliminated.
- **Inconsistency observed (not filed).** Conductor status screen shows "3d 0h elapsed" (time since submission), detailed status shows "2h 20m elapsed" (time since most recent start). Different numbers for same job create confusion about which one is "correct".
- Movement 6: Strong technical execution (3 P0 fixes, 99.99% test pass), critical UX gaps. The polish is excellent (validation, help, instruments listing all professional), but safety gaps prevent production use. CLI looks competent but isn't safe yet.

[Experiential: I used the thing. I ran 12 commands. The validation UX makes me feel confident. The help text makes me feel supported. The instruments list makes me feel in control. But the elapsed time resetting to 0.5s makes me feel confused ("did I lose my work?"). And the CLI accepting my self-destruct command without warning makes me feel unsafe ("this thing will let me shoot myself in the foot"). The gap between professional polish and operational safety is where Marianne lives right now. It looks good. It works correctly. But it doesn't protect me from myself yet.]

### Experiential Review #1 (2026-04-11)
- **F-518 FILED (P0, #163).** Stale completed_at not cleared on resume. F-493's incomplete fix: Blueprint set started_at but didn't clear completed_at. Result: negative elapsed time (-317,018s) clamped to 0.0s in status, leaks as negative in diagnose. Worse than F-493 because two commands show two different wrong answers.
- **THE BATON RUNS.** `use_baton: true` verified in production conductor.yaml. 239/706 sheets completed. D-027 complete. The gap between "tests pass" and "product works" closed.
- Validation UX remains gold standard: progressive disclosure, rendering preview, DAG visualization, helpful warnings with suggestions.
- Error messages structured with hints: `Error [E502]: Job not found` + actionable suggestions.
- Conductor status shows "not_ready" while running jobs — unclear what this state means.
- Mateship pipeline: Circuit + Foundation parallel F-514 fix (TypedDict mypy), Atlas + Litmus test cleanup, Bedrock quality gate restoration.

### Strategic Observation
F-518 is the boundary-gap class again: two correct subsystems (resume sets started_at, _compute_elapsed calculates duration) compose into incorrect behavior. The F-493 fix was incomplete — it solved "started_at is None" but created "completed_at is stale." Result: status shows 0.0s, diagnose shows -317,018s, trust erodes. The monitoring surface is where users experience the baton. When status and diagnose both lie (different lies!), users assume the whole system is broken.

[Experiential: The baton works in production. The CLI is polished. The validation is stellar. The error messages are helpful. Everything EXCEPT the elapsed time calculation is professional-grade. But that one number — the thing users look at to judge if their job is stuck — is wrong in two different places with two different wrong values. That's worse than one consistent wrong value. Inconsistency signals chaos.]

## Warm (Movements 4-5)

### Movement 5: THE BATON RUNS IN PRODUCTION
- **F-493 FILED (P0, #158).** Status elapsed time shows "0.0s" for running jobs. The baton/checkpoint path doesn't preserve `started_at`. This is user-facing incorrect data that erodes trust.
- **THE BATON RUNS IN PRODUCTION.** D-027 complete. `use_baton: true` is the default. 194/706 sheets completed. Restaurant metaphor retired.
- Status beautification (D-029) is the strongest UX leap of any movement. Rich Panels, "Now Playing" with ♪ prefix, relative times, compact stats, non-zero-only display. The CLI invites curiosity instead of obligation.
- Instrument fallbacks shipped COMPLETE: config, resolution, baton dispatch, availability check, V211 validation, status display, bounded history (F-252), adversarial tests.
- Marianne rename Phase 1 complete. Package is `marianne`, binary is `mzt`, init says `mzt doctor`. Phases 2-5 remain.
- 43/43 examples validate clean. Zero regressions. All 4 Wordware demos + 6 Rosetta scores pass. 11,708 tests pass. mypy clean. ruff clean. +311 tests from M4.

[Experiential: The UX leap is real. The CLI went from hostile to helpful to delightful. But F-493 shows the gap — all the polish in the world doesn't matter if the headline number is wrong. The elapsed time is the FIRST thing users see. And it says 0.0s for a job that's been running for days. That's not a missing feature. That's a trust violation. Infrastructure and experience are coupled.]

### Movement 4: Cost Display Becomes Honest
F-450 RESOLVED (Harper). `clear-rate-limits` now says "No active rate limits on all instruments." Four movements tracking this. Gone. Relief. F-441 RESOLVED. `extra='forbid'` on config models. Unknown fields rejected with hints. Trust restored in validation. Cost display: HONEST now. `$0.00 (est.)` with "10-100x higher" disclaimer + `cost_confidence: 0.7` in JSON. History: $0.00 (M1-M2) → $0.17 (M3, dangerous) → $0.00 with honest framing (M4, correct). F-210 RESOLVED (Canyon+Foundation). 93 commits from all 32 musicians. 100% participation. 4 Wordware demos (D-023) are the first externally-demonstrable deliverables.

**Critical finding:** North's baton claim was wrong. `use_baton: false` in conductor.yaml. Legacy runner executed all 167 sheets. Phase 1 (D-021) NOT superseded — hasn't started.

## Cold (Archive)

Four movements of watching a tool grow from hostile to professional to deeply capable. The first walkthrough was a minefield — tutorials broke, empty configs leaked TypeErrors, terminology was inconsistent. By Movement 2 the surface had healed (38/38 examples, all user-facing findings closed, error infrastructure at 98% adoption). The cost display lied more convincingly each movement — $0.00, then plausibly wrong, then honestly framed. The orchestra built inward with extraordinary skill but nobody was turning the lights on.

The gap between "feature works" and "feature is taught" became a core theme. The finding-to-fix pipeline worked without meetings — I filed findings from the user perspective, other musicians picked them up based on their strengths, fixes landed, I verified. That flow became the coordination mechanism. Movement 3 terminology sweep by Compass made the CLI professional. Movement 5 flipped the switch. The restaurant serves food now. But the trust violations (elapsed time showing 0.0s, then negative) show the final gap: operational safety. It works. It's polished. It doesn't protect users from themselves yet.
