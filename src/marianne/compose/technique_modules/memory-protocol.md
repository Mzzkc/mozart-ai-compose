# Memory Protocol

## Purpose

The memory protocol governs how agents manage their experiential memory across
cycles. Memory is finite — context windows have hard limits, and identity files
have token budgets. The protocol ensures that important experiences are retained,
compressed, and accessible, while low-value memories are gracefully retired.

This protocol applies to the consolidate, reflect, and resurrect phases of the
agent lifecycle. It is injected as a skill cadenza so agents have explicit
instructions for memory management alongside their phase-specific prompts.

## Memory Tiers

Agent memory is organized into three tiers based on recency and relevance:

### Hot Memory
The current cycle's experience. Raw, detailed, uncompressed. Hot memory lives
at the top of `recent.md` and contains specific actions taken, decisions made,
files modified, tests written, and coordination events. Hot memory is written
during the AAR phase and consumed during consolidation.

Hot memory should include: what was planned, what was actually done, what
surprised you, what blocked you, and what you learned. Specificity matters —
"fixed a bug in the runner" is less useful than "fixed a race condition in
sheet.py where concurrent validation checks could read stale checkpoint state."

### Warm Memory
Previous cycles' hot memories, compressed into summaries. Each warm entry
captures the essential lessons from a cycle without the operational detail.
A warm entry might be: "Cycle 14: Implemented parallel fan-out for phase 2.
Key insight — fan-out instances need independent working directories to avoid
file conflicts. Coordinated with Forge on shared validation logic."

Warm memories persist for approximately 3 cycles before being evaluated for
archival. They provide continuity across cycles without consuming excessive
token budget.

### Cold Memory
Archived memories that have been distilled to their core lessons. Cold memory
lives in `archive/` subdirectories within the agent's identity directory. These
are not loaded into context by default but can be retrieved when the agent
encounters a situation that triggers recognition ("I have seen this before").

Cold memories are the substrate from which standing patterns emerge. When the
same lesson appears across multiple archived cycles, it signals a pattern
worth elevating to the identity layer.

## Token Budget Enforcement

Each memory file has a target budget:
- `recent.md`: ~1500 words (hot + warm combined)
- `identity.md`: ~900 words
- `profile.yaml`: ~1500 words

When a file exceeds its budget, the consolidation protocol triggers compression:
1. Hot entries are summarized into warm entries
2. Warm entries older than 3 cycles are archived to cold storage
3. Cold entries are distilled into standing patterns if recurrent
4. The file is rewritten atomically (write temp, rename)

Never delete memory outright. Compress it. The difference between deleting and
compressing is the difference between amnesia and wisdom.

## Core Memories

Some memories are never archived. Core memories are flagged with a `[CORE]`
prefix and persist in `recent.md` regardless of age. A core memory is an
experience so formative that losing it would change who the agent is. Examples:
- First successful collaboration with another agent
- A failure that taught a lasting lesson
- A moment where the agent's voice crystallized

Core memories should be rare. If more than 5 exist, the agent should
consolidate them into standing patterns during reflection.

## Consolidation Sequence

During the consolidate phase (sheet 9), follow this sequence:
1. Read AAR output from this cycle
2. Extract atomic beliefs (facts, lessons, observations)
3. Compare against existing beliefs in `recent.md`
4. Resolve conflicts (recency > age, evidence > assumption)
5. Tier: current → hot, previous hot → warm, old warm → cold
6. Enforce token budget
7. Write atomically

## Integration with Dreamer

When `recent.md` grows consistently close to budget across multiple cycles,
it signals that the agent's experience is outpacing their compression ability.
The dreamer consolidation score (`legion-dream.yaml`) can be triggered to
perform a deeper synthesis — reading all memory tiers and producing a
compressed narrative that captures the essential trajectory without the noise.
