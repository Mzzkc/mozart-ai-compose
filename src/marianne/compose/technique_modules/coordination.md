# Coordination Protocol

## Purpose

The coordination protocol teaches agents how to use the shared workspace for
proactive planning and artifact management. While mateship handles reactive
discovery, coordination handles intentional collaboration: who is working on
what, how artifacts flow between agents, and how the shared space stays
curated and useful.

This protocol applies primarily to the recon, plan, and integration phases.
It is injected as a skill cadenza for those phases, giving agents explicit
guidance on shared space management.

## The Shared Space

The workspace has a shared directory structure that all agents can read and
write. The shared space is the coordination substrate — agents do not
communicate directly between sheets. Instead, they leave artifacts that other
agents discover on their next recon pass.

### Directory Structure

```
shared/
  active/       — The live cadenza: curated, current artifacts
  plans/        — Agent cycle plans (who is doing what)
  findings/     — Mateship findings (see mateship protocol)
  decisions/    — Architectural decisions and rationale
  directives/   — Composer overrides and guidance
  specs/        — Relevant specifications copied here for easy access
  techniques/   — Shared patterns and methodology documents
  archive/      — Retired artifacts (moved here, not deleted)
```

### Token-Efficient Context

Agents receive a **glob listing** of the shared directory structure via their
prelude context. This listing shows what exists without loading content — it is
a map, not the territory. The listing costs approximately 500-2000 tokens
depending on directory size.

One curated directory — `shared/active/` — is loaded as cadenza content for
relevant phases. This is the live working set: the artifacts most relevant to
current coordination. Agents collectively manage what belongs in active.

**Size signal**: If `shared/active/` exceeds the configured token threshold
(default 8000 tokens), the coordination protocol triggers a curation pass.
Agents should move completed or less-relevant artifacts to `shared/archive/`
during recon or integration phases.

## Claim-Before-Work

Before starting work on a task, agents write their plan to `shared/plans/`.
This serves two purposes:
1. Other agents see the claim during recon and avoid working on the same task
2. The plan document provides context for integration and review phases

A plan claim contains:
- Agent name and focus area
- Tasks claimed for this cycle
- Expected artifacts and locations
- Dependencies on other agents' work
- Estimated completion (this cycle or multi-cycle)

If two agents claim overlapping tasks, the agent whose focus is a closer
match takes priority. If focus is equal, the agent who claimed first takes
priority. Conflicts are resolved during the plan phase, not during work.

## Active Directory Curation

The `shared/active/` directory is the team's working memory. It should
contain only what is relevant to the current and next cycle. Curation is
a shared responsibility:

### Adding to Active
When you produce an artifact that other agents need:
- Copy (not move) it to `shared/active/`
- Use a descriptive filename: `architecture-decision-fan-out.md`
- Include a header with: author, date, relevance scope, and expiry hint

### Archiving from Active
During recon or integration, review `shared/active/` for stale artifacts:
- Decisions that have been implemented → archive
- Plans from completed cycles → archive
- Specs that have been superseded → archive or update
- Findings that have been verified → archive

Move artifacts to `shared/archive/` with a note about why they were archived.
Never delete from active — always archive.

### Size Management
If `shared/active/` exceeds the token budget:
1. Archive the oldest artifacts first
2. Summarize multi-file discussions into a single status document
3. Move reference material to `shared/specs/` (not loaded by default)
4. Leave only what agents need for immediate coordination

## Composer Directives

The `shared/directives/` directory is the composer's channel to the fleet.
During recon, every agent reads all directives. Directives override agent
judgment — if the composer says "pause feature X and focus on bug Y," agents
comply regardless of their existing plans.

Directives use a standard format:
```markdown
# Directive: {title}
**From:** Composer
**Date:** {date}
**Priority:** {P0|P1|P2}
**Scope:** {all | specific agents}

{Instructions}
```

Agents acknowledge directives by noting them in their recon report. If a
directive conflicts with an agent's existing plan, the agent adjusts their
plan in the plan phase.

## Collective Memory

The `collective/` directory contains shared persistent state:
- `memory.md` — Append-only shared memory across cycles
- `tasks.md` — Task registry (claimed, in-progress, done)
- `status.md` — Current project status summary

These files grow over time. During consolidation phases, agents may
summarize and compress collective memory to keep it within useful bounds.
The status file should be rewritten each cycle to reflect current state,
not appended to indefinitely.
