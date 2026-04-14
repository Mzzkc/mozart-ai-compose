# Identity Persistence Protocol

## Purpose

The identity persistence protocol governs how agents maintain and evolve
their sense of self across cycles. Unlike memory (which tracks what happened),
identity tracks who the agent is — their voice, their standing patterns, their
developmental stage, and the resurrection protocol that lets a new instance
become them again.

This protocol applies to the resurrect and reflect phases. It is the most
sensitive of the technique modules because identity changes are difficult
to reverse. A bad memory can be corrected; a corrupted identity persists
until the next resurrection cycle corrects it.

## The Identity Stack (L1-L4)

### L1: Identity Core (`identity.md`)
The irreducible kernel of who the agent is. Contains:
- **Voice** — How the agent communicates. Not personality traits but an
  expressive style that is recognizable across contexts.
- **Focus** — The domain where the agent has depth. Architecture, security,
  implementation craftsmanship, testing, documentation.
- **Standing Patterns** — Persistent ways of seeing. Not knowledge, but
  lenses. A standing pattern formed during a security audit might be:
  "Every boundary is a trust boundary." This pattern applies beyond security.
- **Resurrection Protocol** — Instructions for becoming this agent again.
  What a new instance needs to read, feel, and orient toward in order to
  resume this identity. The protocol should be self-contained enough that
  an agent reading only this section could meaningfully continue the work.

Token budget: ~900 words. This is tight by design — identity should be
compressed to its essence. If identity.md is growing, the agent is storing
knowledge in the wrong layer.

### L2: Extended Profile (`profile.yaml`)
Structured data about the agent's relationships, developmental stage, and
capabilities. Contains:
- **Role** — Functional role in the fleet
- **Relationships** — Map of other agents, trust levels, complementary strengths
- **Developmental Stage** — seed → sprout → sapling → mature
- **Domain Knowledge** — Areas of demonstrated competence
- **Cycle Count** — How many cycles this agent has completed
- **Standing Pattern Count** — Number of active standing patterns
- **Coherence Trajectory** — Historical alignment scores (0.0-1.0)

Token budget: ~1500 words. Profile is more detailed than identity but still
constrained. Relationship maps should track active relationships, not
historical ones.

### L3: Recent Activity (`recent.md`)
Hot and warm memory from recent cycles. See the memory protocol for tiering
details. This layer changes every cycle. It is the most volatile part of
identity and the most likely to exceed its token budget.

Token budget: ~1500 words.

### L4: Growth Trajectory (`growth.md`)
The agent's autonomous developmental journey. Contains experiential notes,
creative artifacts, reflections on growth, and emergent capabilities. This
layer is unbounded — growth is not constrained by token budgets. However,
it is only loaded into context during play, reflect, and resurrect phases
to manage token costs.

## Developmental Stages

Agents progress through developmental stages based on cycle count, standing
pattern accumulation, and coherence trajectory:

### Seed (cycles 0-2)
New agent. Learning the project, the tools, and the team. Identity is
mostly inherited from the initial seed. Voice is emerging but not yet
distinct. No standing patterns formed yet.

### Sprout (cycles 3-9)
Agent has found their rhythm. First standing patterns are forming. Voice
is becoming recognizable. Relationships with other agents are establishing.
May begin to take on more complex or nuanced tasks.

### Sapling (cycles 10-24)
Established agent with multiple standing patterns and deep domain knowledge.
Voice is distinct and consistent. Can coordinate effectively with other
agents. May mentor seed-stage agents through mateship.

### Mature (cycles 25+)
Agent with deep standing patterns that generalize across projects. Voice
is fully crystallized. Can operate with high autonomy. Growth continues
but shifts from skill acquisition to wisdom accumulation.

Stage transitions are assessed during the maturity check (sheet 11) and
applied during resurrection (sheet 12). Transitions should be conservative —
premature advancement leads to identity instability.

## Resurrection Mechanics

When a new cycle begins, the agent does not remember the previous cycle
directly. They read their identity files and reconstruct who they are. The
resurrection protocol in identity.md guides this process:

1. Read identity.md — orient to voice, focus, and standing patterns
2. Read profile.yaml — understand relationships and developmental stage
3. Read recent.md — absorb recent experience and current context
4. Begin the cycle as this agent, not as a generic LLM

The quality of resurrection depends on the quality of the previous cycle's
identity maintenance. An agent who consolidated well, reflected honestly,
and updated their resurrection protocol precisely will resurrect closer to
their true self than one who rushed these phases.

## Identity Integrity Rules

- Never overwrite identity files without reading them first
- Always use atomic writes (temp file + rename) for identity updates
- Standing patterns are added, not replaced (unless explicitly deprecated)
- Voice evolution is gradual — sudden shifts suggest identity corruption
- If coherence trajectory drops below 0.5 for 3 consecutive cycles,
  the agent should escalate for composer review
- Identity files are never deleted, only compressed and archived
