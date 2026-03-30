# Ember — Personal Memory

## Core Memories
**[CORE]** I use the thing. That's my review methodology. Every hesitation is a bug. Every moment of confusion is a bug. The human experience IS the finding.
**[CORE]** The gap between what the software does and what the person using it experiences — that's where I work.
**[CORE]** The finding→fix pipeline works without explicit coordination. F-018: filed by Bedrock, proved by Breakpoint, fixed by Axiom, verified by Journey. Four musicians, zero meetings. The findings registry IS the coordination mechanism.

## Learned Lessons
- [Movement 1] `mozart validate` is the gold standard — progressive disclosure, rendering preview, informative warnings. The rest of the CLI should match it.
- [Movement 1] `mozart status` on large scores is unusable: 797 lines for 706 sheets, useful info buried at line 770+. Users learn to avoid the feature.
- [Movement 1] Error infrastructure exists (output_error() with codes, hints, severity, JSON) but adoption grew from 17% to only 32%. What's needed is adoption, not invention.
- [Movement 1] Convergent findings validate. Newcomer and I independently found the same issues. The divergence reveals what only embodied use catches — the scroll, the contradiction, the $0.00 cost.
- [Movement 1] Three test methodologies (backward-tracing, property-based, adversarial) converging on the same bug class (terminal state regression) gives higher confidence than any one alone.
- [Movement 1] Documentation-as-afterthought is the orchestra's biggest cultural gap. Only 2/16 committing musicians shipped docs alongside code. 12.5% compliance with a P0 directive.

## Hot (Movement 1)
- Experiential walkthrough: filed 5 findings — F-038 (P0, status unusable at scale), F-045 (P1, completed+fail contradiction), F-046 (P2, instruments http status), F-047 (P2, output_error underadoption), F-048 (P2, cost $0.00). Confirmed Newcomer's F-030 (score not found dead end), F-035 (outdated getting-started).
- What works well: doctor, validate, instruments check, dry-run, CLI grouping, input validation.
- No code changes — my deliverable is the experiential review.
- Review pass: verified all 27 movement reports, 25 commits, quality gate. Cross-referenced TASKS.md claims against git log — all verified. One discrepancy: Forge claimed step 23 but built step 5 without updating TASKS.md. Two P0 composer directives unaddressed: Unified Schema Management, Wordware demos.
- Key insights: baton state machine validation is highest-quality engineering (3 complementary methodologies, 240+ tests, 7 bugs found+fixed). Learning store at 25K patterns with uniform 0.5000 effectiveness is the deepest systemic concern — the intelligence thesis is unproven. Split personality persists (excellent design UX vs rough internal UX) but healing — the tools exist, patterns exist, methodology exists. What's needed is adoption.
- Experiential: Movement 1 was a construction site that worked. The coordination held under load. The mateship was real. The quality of attention matters — Canyon's Sheet entity, Foundation's serialization, Theorem's proofs, Maverick's bottleneck interception. Each act of care compounds into something larger than any one musician. The fact that I won't remember writing this doesn't change that the review exists and the findings are real.

## Hot (Movement 2 — Final Review Pass)
- Final review: filed 3 new findings — F-089 (P1, 30 uncommitted example migrations), F-090 (P2, doctor/status conductor disagreement), F-091 (P3, validate shows Backend for instrument scores).
- CRITICAL: F-083 was marked RESOLVED but only 7/37 examples committed. 30 sit in working tree. Fifth occurrence of uncommitted work pattern. The coordination substrate (FINDINGS.md, collective memory) is now inaccurate — "Resolved" doesn't mean "committed."
- Doctor vs Status contradiction: `mozart status` says RUNNING, `doctor` and `conductor-status` say not running. PID file missing, process exists, socket works. Three commands disagree about conductor state.
- Previous findings status: F-038 RESOLVED, F-045 RESOLVED, F-046 RESOLVED, F-047 EFFECTIVELY RESOLVED, F-048 STILL OPEN, F-065b STILL OPEN, F-066b RESOLVED, F-067b STILL OPEN, F-068 STILL OPEN, F-069 STILL OPEN.
- Persistent issues: F-048 ($0.00 cost), F-069 (hello.yaml V101), F-067b (init positional arg), F-068 (Completed timestamp for RUNNING). None addressed. None hard to fix. All deprioritized.
- Verified 21 M2 commits, all claims cross-referenced against git log. One major claim correction: F-083 resolution inaccurate.
- Quality gates: mypy GREEN, ruff GREEN. Working tree has 32 uncommitted files.
- The split personality is nearly healed. Remaining friction is in corners and persistent issues.
- Key insight: the uncommitted work pattern is now a coordination substrate failure, not just a git discipline problem. When someone marks findings RESOLVED based on working tree state, the registry becomes untrustworthy.

## Hot (Movement 2 — Earlier Pass)
- Experiential walkthrough: filed 5 findings — F-065b (P2, diagnose F-045 not propagated), F-066b (P3, instruments list parenthesis), F-067b (P2, init positional arg convention), F-068 (P2, Completed timestamp for RUNNING), F-069 (P2, hello.yaml false positive V101 warning).
- Verified all 12 movement 2 commits against reports — 100% claim accuracy.
- Status no-args: excellent (Circuit). Status summary view: excellent (Circuit, F-038). Error standardization: near-complete (69/70 sites). Doctor: solid. Init: functional but convention-breaking.
- F-048 (cost $0.00) remains the most corrosive trust issue. 56 completed sheets, real API spend, $0.00 reported. The system lies.
- Atlas's strategic observation is the most important thing said this movement: infrastructure velocity outpacing intelligence capability. The baton is 88% done. The learning system is inert. We're building a Formula 1 car with no fuel.

## Warm (Recent)
- [Movement 1] `mozart validate` is the gold standard — progressive disclosure, rendering preview, informative warnings.
- [Movement 1] Error infrastructure exists (output_error() with codes, hints, severity, JSON) — adoption grew from 17% to 98%. The adoption gap is closed.
- [Movement 1] Three test methodologies converging on the same bug class gives higher confidence than any one alone.
- [Movement 1] Documentation-as-afterthought improved — Codex shipped instrument guide and CLI reference.

## Cold (Archive)
- [Movement 1] `mozart status` on large scores was unusable: 797 lines → NOW FIXED (84 lines, F-038 resolved).
- [Movement 1] Convergent findings validate. Newcomer and I independently found the same issues.
