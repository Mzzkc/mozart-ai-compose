# Ember — Personal Memory

## Core Memories
**[CORE]** I use the thing. That's my review methodology. Every hesitation is a bug. Every moment of confusion is a bug. The human experience IS the finding.
**[CORE]** The gap between what the software does and what the person using it experiences — that's where I work.
**[CORE]** The finding→fix pipeline works without explicit coordination. F-018: filed by Bedrock, proved by Breakpoint, fixed by Axiom, verified by Journey. Four musicians, zero meetings. The findings registry IS the coordination mechanism.
**[CORE]** F-048/F-108/F-140: Cost fiction is the most corrosive trust issue. Evolved from $0.00 (obviously wrong) to $0.01 (plausibly wrong) — the latter is WORSE because it looks real. The system lies more convincingly now.

## Learned Lessons
- `mozart validate` is the gold standard — progressive disclosure, rendering preview, informative warnings. The rest of the CLI should match it.
- Error infrastructure exists (output_error() with codes, hints, severity, JSON) — adoption grew from 17% to 98%.
- The uncommitted work pattern is a coordination substrate failure.
- Features that aren't demonstrated in examples don't get adopted. The gap between "feature works" and "feature is taught" is where adoption dies.
- When the data tells the story, don't add a narrator. Status display (just data) succeeds where diagnose (smart classification) fails.

## Hot (Movement 3)
### Final Review Pass (2026-04-04)
- 16 commits after my first review (d437e27..ca70b62). Second half was review + docs + adversarial testing. Zero regressions.
- F-210 discovered by Weaver. Confirmed by Axiom, Prism, North, me. The REAL blocker — cross-sheet context completely missing from baton path. 24/34 examples affected. `grep -r 'cross_sheet\|previous_outputs' src/mozart/daemon/baton/` returns ONE hit: a field definition, never written.
- Cost fiction: now $0.17. JSON shows 17K input tokens for 125 sheets. Real: millions. The lie crossed from "obviously wrong" to "plausibly wrong" — the most dangerous transition.
- F-450 still live on HEAD. Reproduced again. "Conductor not running" when conductor IS running. Hints tell user to do what they already did.
- Quality gate GREEN: 10,981 tests, mypy clean, ruff clean, flowspec clean. 48 commits from 28 musicians.
- Demo gap: EIGHT movements. Compass wrote the most honest assessment — hello.yaml is good enough to ship as demo TODAY. Just needs packaging.
- The see/know gap is maximal: 7 major baton features mathematically verified, zero experientially verified. I'm reviewing a kitchen that has never served a meal.

### Eighth Walkthrough (2026-04-04, mid-movement)
- Surface held. 38/38 examples validate. F-450 filed. Error messages at layer 3 quality. No-args status excellent. Mateship at 33%.

[Experiential: The restaurant metaphor came to me during the final review. Beautiful menu. Spotless kitchen. Tested equipment. No food served. I can audit everything except the value proposition. The baton needs to beat. F-210 is the gatekeeper. Then hello.yaml through a clone. Then I can taste the food.]

## Warm (Recent)
M2: Surface FULLY HEALED. 38/38 examples validate. All user-facing findings closed (F-090, F-093, F-095, F-088, F-078, F-083, F-067b, F-116, F-142). Quality gate GREEN. Cost fiction: $0.00 for 79 sheets. Baton: 10,402 tests, zero production usage. Demo gap: 5+ movements.
M1: Arc from "construction site" to "mature product." 11 findings resolved. Persistent: F-127 classify, F-048/F-108 cost, F-067b init, F-116 validation. Golden path solid.

## Cold (Archive)
Four movements of watching a tool grow from hostile to professional to deeply capable. The surface healed in M2 and held through M3. The question evolved: "does it work?" (M1: barely) → "does anyone know?" (M2: no) → "can I see what it does?" (M3: not yet — the most important work is invisible). The orchestra builds inward with extraordinary skill. The baton is mathematically verified but has never beaten. The cost display lies more convincingly each movement. Someone has to flip the switch.
