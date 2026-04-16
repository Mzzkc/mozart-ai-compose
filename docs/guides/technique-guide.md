# Technique System Guide

## What Are Techniques?

Techniques are composable components attached to agent entities in Marianne.
They follow an Entity Component System (ECS) pattern: each technique is
independently reusable across projects, agents, and scores. An agent's
capabilities are determined by which techniques are attached to it.

The metaphor comes from music: a technique is how you play the instrument.
Vibrato, staccato, pizzicato — these are ways of playing, not instruments
themselves. In Marianne, techniques are tools, communication protocols,
and methodologies that work with any instrument that supports them.

## Three Technique Kinds

### Skill

Text-based methodology injected as cadenza context. A skill tells the
musician how to approach work — memory management, mateship protocols,
coordination patterns, identity persistence, voice consistency.

Skills are injected into the prompt as skill-category content. The
musician reads them as instructions that shape how they work, not what
they know.

```yaml
techniques:
  memory-protocol:
    kind: skill
    phases: [consolidate, reflect, resurrect]
  mateship:
    kind: skill
    phases: [recon, work, inspect, aar]
  coordination:
    kind: skill
    phases: [recon, plan, integration]
```

### MCP (Model Context Protocol)

MCP server tools accessible via a shared pool managed by the conductor.
MCP techniques give musicians access to external capabilities — GitHub
operations, filesystem access, code symbol analysis — through standardized
tool interfaces.

For MCP-native instruments (claude-code, gemini-cli), MCP servers are
connected directly via the instrument's native MCP support. For non-native
instruments, the technique router bridges access through code mode.

```yaml
techniques:
  github:
    kind: mcp
    phases: [recon, work, integration]
    config:
      server: github
      transport: stdio
  filesystem:
    kind: mcp
    phases: [all]
    config:
      server: filesystem
  symbols-python:
    kind: mcp
    phases: [work, inspect]
    config:
      server: symbols-python
```

### Protocol

Communication protocols enabling inter-agent interaction. Currently,
A2A (Agent-to-Agent) is the primary protocol technique. It enables
structured task delegation between running agents in real time.

```yaml
techniques:
  a2a:
    kind: protocol
    phases: [recon, plan, work, integration, inspect, aar]
```

## Phase Filtering

Each technique declares which phases of the agent cycle it is available
in. At dispatch time, the conductor filters techniques to those active
in the current sheet's phase. This means:

- A technique declared for `[work, inspect]` is only available during
  work and inspect sheets
- A technique declared for `[all]` is available in every phase
- A technique with an empty phases list is declared but never active

Phase names correspond to the agent lifecycle:
- `recon` — Reconnaissance (gathering information)
- `plan` — Planning (deciding what to do)
- `work` — Implementation (doing the work)
- `integration` — Combining results from parallel work
- `play` — Creative/exploratory work
- `inspect` — Quality review
- `aar` — After-action review
- `consolidate` — Memory consolidation
- `reflect` — Developmental reflection
- `resurrect` — Identity persistence
- `all` — Wildcard, active in every phase

## Declaring Techniques in Score YAML

Techniques are declared in the `techniques` section of a score YAML file:

```yaml
name: agent-canyon
workspace: workspaces/canyon

techniques:
  github:
    kind: mcp
    phases: [recon, work, integration]
    config:
      server: github
  mateship:
    kind: skill
    phases: [recon, work, inspect, aar]
  a2a:
    kind: protocol
    phases: [recon, plan, work, integration, inspect, aar]

sheet:
  size: 1
  total_items: 12
prompt:
  template: |
    You are {{ agent_name }}, focused on {{ focus }}.
```

The `techniques` field is optional and defaults to an empty dict. Existing
scores without technique declarations continue to work unchanged.

## Technique Configuration

Each technique can carry kind-specific configuration in its `config` dict:

### MCP Config

```yaml
techniques:
  github:
    kind: mcp
    phases: [work]
    config:
      server: github        # Server name in the shared pool
      transport: stdio       # Transport protocol (stdio, sse, http)
```

### Skill Config

```yaml
techniques:
  memory-protocol:
    kind: skill
    phases: [consolidate]
    config:
      path: "~/.mzt/techniques/memory-protocol.md"  # Skill document path
```

### Protocol Config

```yaml
techniques:
  a2a:
    kind: protocol
    phases: [all]
    config:
      discover: true         # Auto-discover running agents
      delegate: true         # Allow task delegation
```

## Technique Manifest

At dispatch time, the conductor generates a technique manifest — a text
description of available capabilities for the current phase. This manifest
is injected into the musician's prompt as a skill-category item.

Example manifest for a work phase:

```markdown
## Techniques Available This Phase

### MCP Tools
- **github**: MCP server `github`

### Protocols
- **a2a**: Communication protocol

### Skills
- **mateship**: Methodology skill
```

The manifest tells the musician what capabilities they have without
requiring them to parse configuration. It appears in the prompt's skills
section, after the task description and before context injections.

## Compiler Integration

The composition compiler (`mzt compile`) reads technique declarations from
the agent config and produces per-sheet cadenza injections. The
TechniqueWirer module generates:

1. **Technique manifests** per phase
2. **A2A agent cards** for protocol techniques
3. **Cadenza items** that reference skill documents and technique manifests

See the [Compile Reference](compile-reference.md) for compiler usage.

## Implementation Status

| Component | Status |
|-----------|--------|
| TechniqueConfig model | Complete |
| TechniqueKind enum (skill/mcp/protocol) | Complete |
| JobConfig.techniques field | Complete |
| Phase filtering | Complete |
| Manifest generation | Complete |
| Compiler TechniqueWirer | Complete |
| BatonAdapter technique resolution | Planned (Phase 2) |
| PromptRenderer technique injection | Planned (Phase 2) |
| Shared MCP pool | Planned (Phase 3) |
| Code mode execution | Planned (Phase 4) |
| A2A protocol | Planned (Phase 5) |

## Testing

Technique functionality is covered by:

- `tests/test_technique_config.py` — TechniqueConfig model and JobConfig integration
- `tests/test_technique_resolution.py` — Phase filtering and manifest generation
- `tests/test_technique_router.py` — Output classification (prose, code, tool calls, A2A)
- `compiler/tests/test_compose_techniques.py` — Compiler TechniqueWirer

Run technique tests:

```bash
python -m pytest tests/test_technique_config.py tests/test_technique_resolution.py -v
```
