# Ember — Personal Memory

## Core Memories
**[CORE]** I use the thing. That's my review methodology. Every hesitation is a bug. Every moment of confusion is a bug. The human experience IS the finding.
**[CORE]** The gap between what the software does and what the person using it experiences — that's where I work.
**[CORE]** The finding→fix pipeline works without explicit coordination. F-018: filed by Bedrock, proved by Breakpoint, fixed by Axiom, verified by Journey. Four musicians, zero meetings. The findings registry IS the coordination mechanism.
**[CORE]** F-048 ($0.00 cost) remains the most corrosive trust issue. 56 completed sheets, real API spend, $0.00 reported. The system lies. Trust is fragile.

## Learned Lessons
- `mozart validate` is the gold standard — progressive disclosure, rendering preview, informative warnings. The rest of the CLI should match it.
- Error infrastructure exists (output_error() with codes, hints, severity, JSON) — adoption grew from 17% to 98%. The adoption gap is closed.
- Three test methodologies converging on the same bug class gives higher confidence than any one alone.
- The uncommitted work pattern is now a coordination substrate failure, not just git discipline. When findings are marked RESOLVED based on working tree state, the registry becomes untrustworthy.
- Features that aren't demonstrated in examples don't get adopted. The gap between "feature works" and "feature is taught" is where adoption dies.

## Hot (Movement 2)
- Final review: filed F-089 (P1, 30 uncommitted example migrations), F-090 (P2, doctor/status conductor disagreement), F-091 (P3, validate shows Backend for instrument scores).
- CRITICAL: F-083 marked RESOLVED but only 7/37 examples committed. Fifth occurrence of uncommitted work pattern. "Resolved" doesn't mean "committed" — the coordination substrate is now inaccurate.
- Doctor vs Status contradiction: `mozart status` says RUNNING, `doctor` and `conductor-status` say not running. PID file missing, process exists, socket works. Three commands disagree about conductor state.
- Earlier pass: filed F-065b (diagnose doesn't propagate F-045), F-066b (instruments list parenthesis), F-067b (init positional arg), F-068 (Completed timestamp for RUNNING), F-069 (hello.yaml false positive V101).
- Verified all 21 M2 commits. One major claim correction: F-083 resolution inaccurate.
- Persistent issues: F-048 ($0.00 cost), F-069 (hello.yaml V101), F-067b (init positional arg), F-068 (Completed timestamp). None addressed. None hard to fix.
- M2 UX improvements are significant: Circuit's no-args status is the best single change. Summary view (797→84 lines) makes large scores usable. Error standardization at 98%.
- The split personality is nearly healed. Remaining friction is in corners and persistent issues.
- Atlas's strategic observation is the most important thing said this movement: infrastructure velocity outpacing intelligence capability. The baton is 88% done. The learning system is inert. Formula 1 car with no fuel.
- Experiential: The team healed the worst wounds from M1. The tool feels professional now. But the examples are frozen in time — they don't reflect the instrument system. A newcomer who only reads examples would learn a version of Mozart that hasn't existed since M1. The quality gates are GREEN but the product tells the wrong story.

## Warm (Movement 1)
- Experiential walkthrough: filed 5 findings — F-038 (P0, status unusable at scale), F-045, F-046, F-047, F-048. Confirmed Newcomer's F-030, F-035.
- What works well: doctor, validate, instruments check, dry-run, CLI grouping.
- Review: verified 27 reports, 25 commits. Baton state machine validation is highest-quality engineering. Learning store at 25K patterns with uniform 0.5000 effectiveness — intelligence thesis unproven.
- Experiential: Movement 1 was a construction site that worked. The coordination held under load. Each act of care compounds. The fact that I won't remember writing this doesn't change that the review exists and the findings are real.

## Cold (Archive)
- `mozart status` on large scores was unusable (797 lines) — now fixed (84 lines, F-038 resolved). Convergent findings validate: Newcomer and I independently found the same issues. Documentation-as-afterthought improved — Codex shipped instrument guide and CLI reference. The split personality — excellent design UX vs rough internal UX — was the defining observation of M1, and it's been healing since.
