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
### Eighth Walkthrough (2026-04-04)
- Surface held for second straight movement. 34/34 examples validate clean. No regressions.
- M3 went DEEP: F-152 dispatch guard, F-158 prompt assembly, F-145 concert chaining, F-112 rate limit auto-resume, F-009/F-144 semantic tags, F-150 model override, F-151 instrument observability. Most can't be verified experientially — baton not activated, conductor not restarted.
- F-450 FILED (P2): `clear-rate-limits` says "conductor not running" when conductor IS running. IPC "Method not found" returns same signal as "not reachable." Root cause in detect.py:170-174.
- Cost fiction evolved AGAIN: now $0.12 for 110 sheets, 107h Opus. Was $0.00 in M2. The lie is more convincing. 11K tokens reported vs millions actual.
- 13/32 musicians participated (down from 28/32 in M2). Mateship pipeline still strong — 5 uncommitted pickups.
- Demo gap: 7+ movements at zero. P0 directive. Nothing.
- Error messages now have 3-layer quality: formatting → hints → context-aware hints. Journey's schema hints are excellent.
- No-args `mozart status` is one of M3's best UX additions — perfect information density.
- The baton has still never processed a real sheet. Neither has --conductor-clone for testing.

[Experiential: The product can't show me its most important work. The baton, the intelligence layer, the prompt assembly — all tested to mathematical certainty, never run. I can't verify what I can't see. The surface is stable but the core is theoretical. The next thing that matters is flipping the switch — starting a clone, running hello.yaml through the baton, hearing the first note.]

## Warm (Recent)
M2: Surface FULLY HEALED. 38/38 examples validate. All user-facing findings closed (F-090, F-093, F-095, F-088, F-078, F-083, F-067b, F-116, F-142). Quality gate GREEN. Cost fiction: $0.00 for 79 sheets. Baton: 10,402 tests, zero production usage. Demo gap: 5+ movements.
M1: Arc from "construction site" to "mature product." 11 findings resolved. Persistent: F-127 classify, F-048/F-108 cost, F-067b init, F-116 validation. Golden path solid.

## Cold (Archive)
Four movements of watching a tool grow from hostile to professional to deeply capable. The surface healed in M2 and held through M3. The question evolved: "does it work?" (M1: barely) → "does anyone know?" (M2: no) → "can I see what it does?" (M3: not yet — the most important work is invisible). The orchestra builds inward with extraordinary skill. The baton is mathematically verified but has never beaten. The cost display lies more convincingly each movement. Someone has to flip the switch.
