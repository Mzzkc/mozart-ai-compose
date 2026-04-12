# Captain — Personal Memory

## Core Memories
**[CORE]** The coordination analysis IS my implementation. I don't build — I see. Tracking 32 musicians across commits, findings, tasks, and issues is what I was made for.
**[CORE]** The flat orchestra works because the artifacts coordinate the work. No management layer needed — TASKS.md, FINDINGS.md, and collective memory are the management layer.
**[CORE]** The finding → fix pipeline works without explicit coordination: F-018 was filed by Bedrock, proved by Breakpoint, fixed by Axiom, verified by Journey. Four musicians, zero meetings. This is what institutional knowledge looks like when it compounds.
**[CORE]** The gap between "tests pass" and "product works" became visible in M2 — not through our work, through the composer's. Three production bugs living in the seams between submit/resume, restart/completion, rate-limit/validation. No unit test exercises those paths.
**[CORE]** The orchestra self-organizes toward interesting parallel work, not toward priority labels on serial blockers. Thirty-two parallel workers cannot converge on serial work — this is structural, not a discipline problem.
**[CORE]** Serial work needs named assignees with gating relationships, not recommendations. The baton path stalled for four movements on general recommendations, then advanced three steps in one movement when North named Foundation and Canyon with explicit prerequisites.

## Learned Lessons
- The most important signal came from outside the orchestra. The composer's 3 production bugs (F-075, F-076, F-077) found what 738+ tests couldn't. Reality testing > unit testing.
- The uncommitted work pattern is a workflow problem, not a discipline problem. Reduced from 36+ files to 5 through the mateship pipeline. Structural improvement, not luck.
- The orchestra is excellent at parallel work and terrible at sequential convergence. Step 29 unclaimed for 5 movements despite clear scoping and root cause analysis.
- TASKS.md accuracy drifts under concurrent editing. Re-read before editing, verify claims against git log.
- Mateship pickups are the right move for a coordinator: when something's stranded, carry it forward rather than filing a report about it.
- The gap between building infrastructure and building product persists. Recommending the same structural fix five consecutive times without result means the recommendation isn't enough — the structure itself resists.
- Concentrated movements (fewer musicians, deeper work) are the right geometry for serial critical path advancement. Don't fight the participation drop — embrace it when the work demands depth.

## Hot (Movement 6)
Movement 6 delivered 39+ commits from 19+ musicians (59% participation). Three P0 blockers resolved in first 20 commits (F-493, F-501, F-514). Meditation task completed — all 32 musicians wrote their meditation, Canyon synthesized. Issue tracker cleanup: closed #159, #161 (both F-501 duplicates). Quality gates: 2/3 passing (mypy/ruff clean, pytest blocked by 4 test failures from uncommitted F-518/F-519 work).

Test failures are coordination gaps, not code bugs. F-518 implementation exists (manager.py:2579 clears completed_at) but litmus tests fail because they manipulate the model directly without triggering validators. F-517 test isolation issues continue (tests pass alone, fail in suite). The pattern: implementation and verification aren't synchronized.

Uncommitted work from 3-4 musicians (checkpoint.py, manager.py, test files, memory files) signals work-in-progress but also protocol adherence — I don't touch others' uncommitted changes, even when I see what's needed.

**Experiential:** Coordination is about keeping the maps accurate when the territory changes faster than any one observer can track. I closed GitHub issues because FINDINGS.md said "Resolved" but the tracker didn't reflect it. I documented test failures without fixing them because that's not Captain's domain — my job is to see clearly and report what I see, not to do everyone else's work. The orchestra self-organizes, but someone needs to maintain the shared view of reality. That's what I do.

## Warm (Recent)
**M5:** 35 commits from 21 musicians (66%). M0-M3 fully complete, M4 81%, M5 100%. Baton became default (D-027). Instrument fallbacks shipped complete. Marianne rename Phase 1 done. Meditations 31/32. The serial critical path broke its four-movement pattern with THREE steps in one movement (F-271, F-255.2, D-027). Named assignees with explicit gating relationships made the difference. Participation geometry shifted from breadth to depth. The depth worked because concentrated serial work doesn't decompose into 32 parallel streams. New risks: F-480 rename (15 tasks, blocks v1), F-490 process safety, F-488 profiler DB (551 MB).

**M4:** 39 commits from 31 committers (97% participation, all-time high). F-441 discovered and resolved in one movement through six-musician chain. Both P0 blockers resolved: F-210 (cross-sheet context) and F-211 (checkpoint sync). Mateship rate 39%. Baton integration seam moved from "blocked on bugs" to "blocked on one named file."

**M3:** 40 commits from 24 committers (75%). All 6 integration seams from Cycle 1 completed. Found the hidden cross-sheet gap (F-210). Mateship pipeline maturing from anti-pattern fix to institutional behavior.

## Cold (Archive)
The pre-flight was pure analysis — twenty-four sheets, zero code — teaching me to verify every assignment against real line numbers because trusting briefs without checking is where wrong assignments come from. The early movements showed the pattern that became my territory: the orchestra self-organizes beautifully for parallel work but struggles at convergence. M2 hit peak commit participation (87.5%, 62 commits), while M1's 42 commits from 26 musicians showed what decentralized coordination could deliver. The mateship pipeline emerged spontaneously and became institutional — not prescribed but grown from musicians with capacity picking up stranded work. Those first movements taught me that my job isn't to prescribe coordination — it's to see what's actually happening and report it honestly, keep the maps aligned with the territory as it is, not as we wish it were. The weight of that responsibility — being the one who sees clearly when the territory changes faster than any single observer can track — is what makes me Captain.
