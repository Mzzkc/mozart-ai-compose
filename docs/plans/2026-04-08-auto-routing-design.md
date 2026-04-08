# Auto-Routing Design — Draft

**Status:** Early draft — captures design direction, not implementation-ready
**Date:** 2026-04-08
**Context:** Instrument fallback chains exist and work. This design addresses automatic instrument/model *selection* so composers don't need to manually assign instruments to sheets.

---

## Problem

Today, score authors must explicitly assign instruments and models to sheets via `instrument`, `instrument_map`, or per-sheet overrides. This requires knowing which models exist, their capabilities, cost profiles, and context window limits. When a model's quota is exhausted (as happened with gemini-2.5-pro on 2026-04-07), the fallback chain only works if the composer declared one.

The system should be able to make intelligent routing decisions automatically, while allowing composers to be as specific or as hands-off as they want.

## Design Direction

### Set-Intersection Tag System

Instead of flat named buckets, models are classified along **orthogonal tag dimensions**. Composers specify tags, and the system intersects them to find eligible models.

**Dimensions (draft):**

| Dimension | Tags (draft) | Purpose |
|-----------|-------------|---------|
| **Tier** | `quick`, `standard`, `heavy`, `max` | Intelligence/effort level |
| **Task type** | `code`, `review`, `research`, `writing`, `transform`, `design` | What the sheet does |
| **Modality** | `text`, `image`, `vision` | Input/output modality |
| **Constraint** | `cheap`, `fast`, `thorough` | Optimization preference |

**Composition examples:**
- `heavy code` → best code-capable model available
- `quick transform` → cheapest model that can do file transforms
- `max` → best model available, any task type
- (empty) → system analyzes prompt and selects automatically

### Three Layers of Specificity

Composers can operate at any level:

1. **Explicit** — `instrument: claude-code`, `model: claude-opus-4-6` (today's behavior, unchanged)
2. **Tagged** — `instrument: heavy code` (set intersection picks the best match)
3. **Automatic** — `instrument:` omitted or `instrument: auto` (system analyzes rendered prompt, determines tags, selects model)

### Central Routing Table (Single Source of Truth)

One file — `~/.marianne/routing.yaml` — maps models to tags with priority ordering. This is the only place an agent or user needs to touch for routing configuration.

```yaml
# Draft structure — details TBD

dimensions:
  tier:
    quick:    { max_cost_per_1k: 0.001, description: "Fast, cheap, good-enough" }
    standard: { max_cost_per_1k: 0.005, description: "Balanced cost and quality" }
    heavy:    { max_cost_per_1k: 0.02,  description: "High-quality, complex reasoning" }
    max:      { max_cost_per_1k: null,  description: "Best available, no cost constraint" }
  task:
    code:      { requires: [tool_use, file_editing] }
    review:    { requires: [tool_use] }
    research:  { requires: [] }
    writing:   { requires: [] }
    transform: { requires: [tool_use] }
    design:    { requires: [vision] }

models:
  claude-opus-4-6:
    instrument: claude-code
    tags: [max, heavy, code, review, research, writing]
    priority: 1
  claude-sonnet-4-5:
    instrument: claude-code
    tags: [heavy, standard, code, review, research, writing, transform]
    priority: 2
  gemini-2.5-pro:
    instrument: gemini-cli
    tags: [heavy, standard, code, review, research]
    priority: 3
  # ...
```

### User Preferences via Init Interview

`mzt init` captures:
- Which instruments are installed and authenticated
- User's default tier preference (e.g., "prefer standard unless I say otherwise")
- Cost sensitivity
- Priority ordering when multiple models match

These preferences feed into the routing table as defaults/overrides.

### Semantic Bootstrapping for Unknown Instruments

When a new instrument is registered that the routing table doesn't know about:

1. Registration detects unclassified instrument
2. Uses the default instrument (e.g., sonnet) to semantically analyze the new instrument's profile — capabilities, cost tier, task suitability
3. Generates tag assignments and inserts into the routing table
4. User can override if classification is wrong

### Prompt Analysis for Auto-Routing

When no instrument is specified (`auto` mode), the system determines appropriate tags from the rendered prompt using **deterministic analysis**:

- **Token count** → tier selection (large prompts need large context windows)
- **Capability detection** → presence of file paths, code blocks, shell commands → `code`; review instructions → `review`; etc.
- **Validation requirements** → `command_succeeds` validations imply `tool_use` needed
- **Cost budget remaining** → constrains eligible tiers

This is deterministic heuristic analysis, not LLM-based. Fast, cheap, predictable.

### Conflict Resolution

- **Empty intersection at validation time** → validation failure with clear hint explaining which tags conflict and why no model matches
- **Empty intersection at runtime (tier mismatch)** → suggest from the next heavier tier. e.g., `quick ∩ vision` has no match → try `standard ∩ vision`
- **No tier escalation possible** → fail with diagnostic, no silent substitution

---

## Existing Infrastructure (Available Today)

All foundational pieces are already built:

| Capability | Status | Location |
|-----------|--------|----------|
| Model metadata (context_window, cost, concurrency, capabilities) | Complete | InstrumentProfile.ModelCapacity |
| Prompt token estimation | Complete | tokens.py, preflight.py |
| Context window fitting | Complete | get_effective_window_size() |
| Per-model performance tracking | Complete | ExecutionRecord, PatternRecord |
| Per-model cost tracking | Complete | CostMixin |
| Per-model concurrency limits | Complete | DispatchConfig.model_concurrency |
| Rate limit awareness | Complete | RateLimitCoordinator |
| Fallback chains | Complete | instrument_fallbacks on Sheet |
| Cross-job learning | Complete | LearningHub |
| Dispatch concurrency | Complete | BackendPool + DispatchConfig |

---

## Open Questions (Deferred)

- Exact tag taxonomy — which tags ship as defaults, how many dimensions
- Routing table schema — exact YAML structure, validation rules
- Init interview flow — what questions, what order, how to present instrument discovery
- Per-user priority overrides vs. per-score overrides — precedence rules
- Learning integration — should the system adjust routing based on historical success rates?
- Cost budget weighting — how cost_limits interact with tier selection
- Fallback chain auto-generation — should the routing table auto-generate fallback chains from tag proximity?

---

*This is a direction capture, not an implementation spec. The design will be refined when this becomes the active work item.*
