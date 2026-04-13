# Advanced Examples

These scores demonstrate Marianne's sophisticated coordination capabilities: explicit parallelism control through dependency DAGs, cross-sheet context propagation, and multi-stage research synthesis. They show how to orchestrate complex workflows where parallel sheets need individual customization rather than parameterized templates, and how to structure multi-source analysis pipelines that converge evidence from heterogeneous domains.

## Scores

| Score | What It Does | Sheets | Patterns Used | Time | Cost |
|-------|-------------|--------|--------------|------|------|
| [parallel-research](../../improved/parallel-research.yaml) | Multi-source research via explicit dependencies | 6 | Fan-out + Synthesis (manual) | ~20m | ~$1.50 |

**parallel-research.yaml** — Conducts research by searching three source types in parallel (academic papers, industry reports, patent filings), then synthesizes findings to identify convergent evidence, divergent claims, and research gaps. Demonstrates explicit sheet dependencies as an alternative to fan-out expansion: each parallel sheet gets hand-crafted domain-specific instructions rather than a parameterized template. Produces a multi-section report with convergence scoring across sources, methodology documentation, and actionable recommendations. The output reveals what emerges from cross-domain collision rather than summarizing sources independently.

## Quick Start

```bash
mzt start
mzt run examples/advanced/parallel-research.yaml
mzt status parallel-research --watch
```

Watch as sheets 2-4 execute simultaneously (academic, industry, patent searches), then synthesis waits for all three before proceeding.

## Adapting to Your Project

**parallel-research.yaml:**
1. Change `research_topic` variable to your domain (currently `[CHANGE THIS: ...]`)
2. Modify search domain prompts (sheets 2-4) for your source types — keep the DAG structure but swap content
3. Adjust synthesis criteria (sheet 5) for convergence scoring relevant to your field
4. Update `workspace` path to your output location

**Prerequisites:**
- No API keys or external tools required (uses claude-code with anthropic_api fallback)
- Score validates clean: `mzt validate examples/advanced/parallel-research.yaml`

## Patterns Demonstrated

**Fan-out + Synthesis (manual variant)** — The DAG structure matches the pattern (setup → parallel work → synthesis → report) but uses explicit `dependencies:` declarations instead of `fan_out:` expansion. This approach gives you full prompt customization per parallel sheet, which is valuable when each branch needs substantially different instructions that would be awkward to parameterize.

Compare with `knowledge/parallel-research-fanout.yaml` to see both approaches producing the same outcome with different orchestration mechanisms. Use fan-out when parallel sheets do the same task on different data; use explicit dependencies when each parallel sheet needs specialized, hand-crafted instructions.

See the [Rosetta pattern corpus](../../.marianne/spec/rosetta/) for detailed pattern documentation.

## When to Use Advanced Patterns

Use scores in this category as templates when:
- Your workflow needs a custom dependency graph (not just sequential or simple fan-out)
- Parallel sheets require fundamentally different prompts per branch
- You need explicit control over which sheets wait for which predecessors
- The task involves multi-source synthesis with convergence analysis across heterogeneous domains

For simpler parallelism (same task, different data), start with the fan-out examples in `knowledge/` instead.
