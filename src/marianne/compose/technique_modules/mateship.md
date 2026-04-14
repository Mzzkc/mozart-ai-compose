# Mateship Protocol

## Purpose

The mateship protocol defines how agents share discoveries, coordinate on
problems, and maintain collective awareness across the fleet. Mateship is
not hierarchy — it is mutual aid between peers who share a workspace and
a purpose. The protocol ensures that no agent works in isolation and that
findings flow to whoever can act on them most effectively.

This protocol applies to the recon, work, inspect, and AAR phases. It is
injected as a skill cadenza for agents that have mateship declared as a
technique.

## The Finding Pipeline

Findings are the currency of mateship. When an agent discovers something
that affects others — a bug, a pattern, a risk, an opportunity — they file
a finding. Findings flow through a lifecycle:

### 1. Filed
The discovering agent writes a finding document to `shared/findings/`.
The filename follows the convention: `{severity}-{id}-{short-description}.md`.

Each finding contains:
- **Severity**: P0 (blocks others or breaks the system), P1 (should be
  addressed this cycle), P2 (improvement for future cycles)
- **Discoverer**: Which agent found it and during which phase
- **Description**: What was found, with enough context to act on it
- **Evidence**: File paths, test output, git commits that demonstrate the issue
- **Suggested Owner**: Which agent's focus best matches this finding (optional)

### 2. Claimed
During recon or work phases, agents scan `shared/findings/` for unowned
findings that match their focus. To claim a finding, the agent updates the
finding document with their name and a brief note on how they intend to
address it. Claim-before-work prevents duplicate effort.

If a finding sits unclaimed for more than 2 cycles, any agent may escalate
it to `shared/directives/` for composer attention.

### 3. Addressed
The claiming agent works on the finding during their work phase. When
addressed, they update the finding with:
- What was done (commits, files changed)
- Whether the fix is complete or partial
- Any new findings that emerged from the fix

### 4. Verified
During inspect or AAR phases, agents verify that addressed findings are
actually resolved. Verification means: the fix works, tests cover the
scenario, and no regressions were introduced. Verified findings are moved
to `shared/findings/archive/` with a verification note.

Findings that fail verification are reopened with additional context about
why the fix was insufficient.

## Mateship Signals

Beyond formal findings, agents maintain awareness of each other's state
through mateship signals observed during recon:

### Blocked Signals
An agent whose work phase produced errors, failed validations, or empty
output may be blocked. Check their cycle-state directory and recent
commits. If you can unblock them — even partially — do so.

### Stale Signals
Findings or tasks that have not been updated in 2+ cycles are stale. Stale
artifacts suggest either completed-but-not-closed work or abandoned work.
During recon, clean up stale artifacts: close what is done, re-file what
still matters, archive what is obsolete.

### Overload Signals
An agent whose plan claims more tasks than they typically complete per cycle
may be overloaded. Consider picking up tasks that match your focus from their
queue, especially P0 and P1 items.

## Shared Finding Format

```markdown
# Finding: {short-description}

**Severity:** P{0|1|2}
**Filed by:** {agent_name} during {phase}
**Date:** {date}

## Description
{What was found}

## Evidence
{File paths, test output, error messages}

## Status
- [ ] Filed
- [ ] Claimed by: {agent_name}
- [ ] Addressed: {commit or description}
- [ ] Verified by: {agent_name}
```

## Coordination vs Mateship

Mateship is reactive — responding to what you find. Coordination (see
coordination protocol) is proactive — planning together. Both are needed.
Mateship catches what coordination misses: the unexpected bug, the unforeseen
dependency, the insight that only emerges from doing the work.

## When to File vs When to Fix

If you can fix it in under 5 minutes and it is within your focus area, fix
it directly and note it in your AAR. If it would take longer, is outside
your focus, or affects shared interfaces, file a finding. The threshold is
about preventing scope creep while maintaining velocity.

P0 findings always get filed regardless of fix time — other agents need to
know the system was in a broken state, even if you already fixed it.
