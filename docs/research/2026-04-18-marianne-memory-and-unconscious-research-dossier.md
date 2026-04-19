# Research Dossier — Marianne Memory Architecture & Unconscious Local Model

**Date:** 2026-04-19
**Purpose:** Synthesize four parallel research streams into a unified dossier that seeds the design drafts for Marianne's memory system (S6 Memory Architecture) and unconscious local model layer (S6 The Unconscious / Smart Conductor).
**Inputs:** `02-research-cam.md`, `02-research-memory-systems.md`, `02-research-local-models.md`, `02-research-integration.md`, `01-prior-art-brief.md`
**Method:** Cross-stream triangulation, tension analysis, and emergent-insight extraction — not concatenation.

---

## Convergences

The four research streams were conducted independently. Where they agree without coordination, the signal is strong enough to treat as design axioms.

### 1. The Memory Substrate Is SQLite + Local Embeddings

All four streams converge on SQLite as the right storage substrate:

- **CAM research** confirms it supports multi-tenant isolation through namespaced database files and identity-scoped queries.
- **Memory systems research** validates that SQLite handles millions of rows with sub-second queries; the existing learning store already operates at schema v15 with WAL mode.
- **Integration research** confirms it meets the offline constraint ("doomsday-prep grade") — no cloud dependency, no external database process.
- **Local models research** implicitly relies on it for ConductorDecision storage and training data extraction.

Local embeddings via `all-MiniLM-L6-v2` (80MB, Apache 2.0) provide adequate semantic retrieval for Marianne's domain-specific content at ~5ms per query on CPU. This is bundled in the wheel. Optional upgrade to `NV-Embed-v2` for GPU users. No cloud embedding API is acceptable.

### 2. The Tiered Memory Model Is Already Proven

Legion's hot/warm/cold/core tiering with dreaming consolidation is the only working memory system in this project — and it has survived dozens of real sessions. Every stream references this model as the baseline to extend, not replace:

- **Memory systems** confirms the pattern aligns with modern production approaches (Letta's memory blocks, Mem0's temporal decay, HippoRAG's incremental graph updates).
- **CAM** maps the tiers directly to RLF's L1-L4 person model layers.
- **Integration** uses the tier model to define retrieval latency tiers (fast/standard/comprehensive).
- **Prior-art brief** identifies the dreaming score (`legion-dream.yaml`) as a working implementation of the consolidation pattern the S6 spec asks about.

### 3. The Decision Source Must Be Pluggable via a DecisionSource Protocol

The unconscious and integration research arrive independently at the same interface: a `DecisionSource` Protocol with `evaluate(context) -> ConductorDecision`. Three implementations: `HeuristicDecisionSource` (zero dependencies, ships immediately), `LocalModelDecisionSource` (llama-cpp-python + cached GGUF), `RemoteDecisionSource` (delegates to a cloud instrument). All three produce the same `ConductorDecision` record for inspectability. The integration research pins the insertion point: after `BatonCore.handle_event()`, before `dispatch_ready()`. The baton's sequential event processing eliminates race conditions between decision evaluations.

### 4. Memory Reads Are Budget-Free; Flow-Control Actions Are Budgeted

All streams agree on the action budget semantics defined in S6: retrieving memory is always free, recording experience is always free, observe-only decisions are free. Only flow-control actions (goto, skip, retry, inject_context) consume the per-job action budget. The integration research adds a critical refinement: **escalation (communication to the composer) should also be budget-free**, because blocking communication when the budget is exhausted defeats the guardrail's purpose. This refinement was not in the S6 spec itself — it emerged from mapping the budget against the fermata system.

### 5. CAM-Readiness Is Architectural Properties, Not API Conformance

The CAM research found no external specification to conform to — "CAM" is a named design intent, not a documented system. All streams agree that CAM-readiness means designing for six architectural properties: multi-tenant isolation, identity-scoped access, associative retrieval, incremental growth, offline operation, and structural alignment with the RLF person model. A `CAMExportProtocol` provides future compatibility without current coupling. The integration research reinforces this by scoping the initial implementation to single-tenant (Marianne Mozart only) while designing the Protocol for multi-tenant growth.

### 6. The Baton Event Loop Is the Integration Spine

The integration research maps exactly where memory and unconscious plug into the conductor's runtime. The insertion point — after `handle_event()`, before `dispatch_ready()` — is validated by the unconscious research's latency measurements (sub-500ms for classification tasks) and the memory systems research's retrieval benchmarks (sub-200ms for multi-signal fusion). The architecture flows linearly: event → state mutation → smart conductor evaluation (memory read + unconscious judgment) → dispatch. No EventBus subscription — a direct callback on BatonCore avoids ordering issues.

---

## Tensions

Where findings conflict, the resolution must make both sides true, not average them.

### 1. Latency Budget vs. Retrieval Richness

The integration research demands sub-100ms memory retrieval for smart conductor decisions. The memory systems research recommends multi-signal retrieval (semantic + keyword + entity matching) for high-quality results — but three parallel scoring passes over a moderate fact store are inherently more expensive than a single embedding lookup.

**Resolution:** Tiered retrieval, not a single retrieval mode. Fast path (embedding-only, sub-50ms) for the unconscious's classification decisions where speed matters more than recall. Standard path (multi-signal fusion, sub-200ms) for reasoned decisions where the unconscious can trade latency for quality. Comprehensive path (full search + re-ranking, seconds) for compose and maestro sessions where interactive latency is acceptable. The `MemoryStore.retrieve_relevant()` method accepts a `latency_tier` parameter. Both the latency budget and retrieval richness demands are satisfied — just at different tiers.

### 2. Offline Bundling vs. Distribution Size

The unconscious research recommends Qwen3-4B (2.5GB GGUF) and the memory systems research recommends bundling `all-MiniLM-L6-v2` (80MB). Together that is 2.6GB+ of additional distribution weight. The offline constraint demands bundling; distribution norms demand reasonable package sizes.

**Resolution:** Separate the two classes of dependency. The embedding model (80MB) is small enough to bundle in the wheel — it is essential for the memory system to function at all, and memory is always-on. The GGUF model (2.5GB) is downloaded on first use via HuggingFace Hub and cached in `~/.marianne/models/`. The `mzt model import` command handles offline provisioning for air-gapped environments. The heuristic fallback requires neither download. This satisfies offline capability (both models can be pre-cached) without bloating the default installation.

### 3. CAM Multi-Tenancy vs. Implementation Simplicity

The CAM research defines six architectural properties that require identity-scoped queries, shared namespaces, and entity-graph access control. The integration research scopes the initial implementation to Marianne Mozart as the sole memory consumer. Building full multi-tenancy from day one adds complexity the system won't exercise.

**Resolution:** Design the Protocol for multi-tenancy; implement for single-tenant. The `MemoryStore` Protocol accepts `agent_id` parameters on every method, but the initial implementation uses a single hard-coded identity (Marianne Mozart). No multi-tenant tests until a second agent actually needs memory access. The Protocol is the contract — it grows into multi-tenancy when CAM demands it, without breaking changes. This is the familiar Marianne pattern: Protocol first, concrete second.

### 4. Learning Store Preservation vs. Clean Separation

The existing learning store has 12 tables, 16 modules, and ~6800 lines of proven telemetry infrastructure. The memory systems research recommends a separate fact store with entity graphs. The integration research shows the learning store's execution telemetry feeds into Marianne's experience. Integrating risks coupling; separating risks duplication.

**Resolution:** The learning store remains as-is. It continues to record execution telemetry with its proven mechanics — pattern broadcasts, trust scores, drift metrics, entropy monitoring. The memory system adds an experiential layer above it. During consolidation (dreaming), a `TelemetryIngester` reads from the learning store and writes distilled facts to the memory store. The two systems share a SQLite runtime but maintain separate databases: `~/.marianne/global-learning.db` (learning store) and `~/.marianne/memory/` (memory system). No schema changes to the learning store. No new dependencies between packages. The bridge is one-way: learning store → memory, never memory → learning store.

### 5. Model Quality vs. Model Availability Across Hardware

The unconscious research finds that Qwen3-4B on CPU-only hardware produces text in 2–3.5 seconds — above the sub-second budget. Classification with GBNF grammars achieves sub-200ms with Qwen3-1.7B on CPU. The recommended model (4B) does not meet latency on all hardware; the model that meets latency everywhere (1.7B or 0.6B) has lower quality.

**Resolution:** Auto-detect hardware at conductor startup and select the appropriate tier. GPU available → Qwen3-4B with full generation capability. CPU-only with sufficient RAM → Qwen3-1.7B (or newer Qwen3.5-2B) in classification-only mode (GBNF-constrained, sub-200ms). Low RAM or no model → heuristics only. The conductor configuration documents these tiers. The `DecisionSource` implementation selects the appropriate model automatically. This is graceful degradation applied to the model itself, not just to the system around it.

---

## Emergent Insights

These findings appear only from the collision of streams — none are visible in any single research note.

### 1. The Dreaming Cycle Is Also the Training Data Pipeline

When the memory systems research (tiered consolidation with ADD-only facts) collides with the unconscious research (training data from ConductorDecision records), a dual-purpose emerges: the dreaming process that compresses memory into tiers is the same process that should extract training examples for the unconscious model. During consolidation, the dreamer reads recent facts, identifies patterns, compresses them into summaries — and can simultaneously extract `(context, decision, outcome)` tuples for fine-tuning. This means consolidation quality directly affects model quality. A poorly tuned dreaming cycle produces both bad memory compression and bad training data. The two systems share a critical dependency that neither research stream identified independently.

**Implication:** The dreaming score must be designed as a multi-output pipeline from the start, not retrofitted with training-data extraction later. The `TelemetryIngester` and training-data extractor are the same component.

### 2. The Entity Graph Is the Unconscious's Fast Path Into Memory

The CAM research defines associative retrieval through entity graphs. The unconscious research shows classification (not generation) is the fast path. When combined: entity-based memory retrieval — "which instruments, patterns, error codes are associated with this event?" — is inherently structured and fast. It is a graph traversal, not a similarity search. Entity neighbors can be pre-computed and cached at consolidation time, making retrieval effectively a hash-table lookup.

This means the entity graph is not just an alternative to embedding search — it is the unconscious model's primary memory access pattern. Embeddings handle the compose and maestro sessions (where the query is unstructured natural language). The entity graph handles the smart conductor (where the query is a structured baton event with known entity types: instrument name, error code, sheet number, venue ID).

**Implication:** The entity graph design should be optimized for the unconscious's access pattern (lookup by entity type + ID, retrieve neighbors) rather than for general-purpose graph traversal. A simple adjacency list with entity-type indexing suffices. HippoRAG's personalized PageRank is overkill for structured queries — reserve it for unstructured retrieval in compose conversations.

### 3. In-Process Colocation Creates a Latency Advantage

The offline constraint forces both the memory system (SQLite) and the unconscious model (GGUF via llama-cpp-python) to run in the conductor's process. This is usually seen as a constraint, but it creates an unexpected advantage: in-process access to both systems eliminates network latency entirely. A remote memory store + remote model would need ~200ms network overhead per round-trip. In-process access means the combined latency budget (memory retrieval + model inference) has near-zero overhead between components.

On a machine with GPU, the combined path is: SQLite entity lookup (~1ms) + model classification (~100ms) = ~101ms total. Even on CPU-only: entity lookup (~1ms) + Qwen3-1.7B classification (~200ms) = ~201ms. Both are well within the 500ms budget. The "doomsday-prep" constraint accidentally optimized the latency path.

**Implication:** Do not introduce an HTTP API between the memory store and the unconscious model. Keep both in-process. The `DecisionSource.evaluate()` method calls the `MemoryStore` directly, not through a service boundary. If a future deployment needs separation (e.g., memory on a different host for the concierge), add the API layer then — not now.

### 4. ConductorDecision Records Are the First CAM Implementation

The CAM research defines collective memory as "shared knowledge across agents." The integration research says ConductorDecision records should be written to the shared namespace, not to Marianne's private memory — because other agents should learn from the conductor's decisions. This means the decision logging infrastructure is itself the first instance of collective memory. Before any musician has memory access, before CAM is formally designed, the shared `decisions` table is a working CAM artifact: knowledge produced by one agent (the conductor) and available to all future agents.

**Implication:** Design the `decisions` table as shared-namespace from the start. Give it the same identity-scoping and access patterns planned for the full CAM entity graph. This makes the decision log the proof-of-concept for multi-agent memory sharing — a concrete test of the CAM architectural properties before they are needed at scale.

### 5. Degradation Tiers Map to Developmental Stages

The unconscious research defines three degradation tiers: Full (memory + model), Heuristic (rules only), Mechanical (no smart conductor). The CAM research maps RLF developmental stages: Recognition → Integration → Generation → Recursion → Transcendence. The mapping is structural:

- **Mechanical** (baton defaults) = Recognition stage — the system follows rules without judgment
- **Heuristic** (rule-based decisions) = Integration stage — the system applies learned rules but cannot reason about novel situations
- **Full** (memory + model) = Generation stage — the system draws on experience and produces novel judgments

This means the degradation tiers are not just a fallback mechanism — they are a developmental progression. A fresh Marianne installation starts at Mechanical (no memory, no model). As the learning store accumulates telemetry, heuristics become effective — Heuristic tier. As ConductorDecision records accumulate and a model is trained, the Full tier activates. Marianne grows into her capabilities over time, just as the RLF person model describes.

**Implication:** Frame the tiers as a growth trajectory in the spec, not just as failure modes. The conductor should report its current tier as part of `mzt status` — "Marianne: Generation stage (memory active, model loaded)" tells the composer not just what is working, but where Marianne is in her development.

### 6. Memory Consolidation and Unconscious Training Share a Feedback Loop

The memory systems research describes consolidation as compressing recent facts into patterns and narratives. The unconscious research describes training as learning from past decisions. These two processes feed each other in a loop not visible in either stream alone:

1. Raw execution telemetry → dreaming consolidation → compressed patterns
2. Compressed patterns → inform the unconscious model's decision context → produce better decisions
3. Better decisions → recorded as ConductorDecision records → fed back into consolidation

This is a reinforcement loop. Good consolidation improves the unconscious model's context, which improves its decisions, which produce better training data for the next consolidation cycle. But it is also a risk: poor consolidation can degrade the model's context, producing worse decisions, which produce worse training data. The feedback loop amplifies both quality and errors.

**Implication:** The consolidation quality must be monitored independently of the unconscious model's accuracy. If decision quality degrades after a consolidation cycle, the system should flag it (not auto-correct — Marianne's judgment may be right and the metric wrong). The dreaming score should produce a consolidation quality report alongside the compressed output.

---

## Implications for Memory Spec

Grouped by the six S6 memory requirements.

### Requirement 1: Fast retrieval for real-time conductor decisions

- **Tiered retrieval interface.** `MemoryStore.retrieve_relevant()` accepts a `latency_tier` parameter: `fast` (<50ms, entity-lookup only), `standard` (<200ms, multi-signal fusion), `comprehensive` (<2s, full search + re-ranking).
- **Entity graph as the fast path.** Entity neighbors are pre-computed during consolidation and cached. The unconscious's structured queries (instrument X, error code Y) resolve via entity lookup, not embedding similarity.
- **In-process access only.** No HTTP boundary between memory and unconscious. SQLite accessed directly in the conductor process.

### Requirement 2: Accumulation over time

- **ADD-only fact storage.** Facts are never deleted or overwritten. New observations append. Tiered consolidation (dreaming) compresses but preserves access to original facts.
- **Three table families.** `facts` (ADD-only, timestamped, entity-linked), `entities` (unique entities with embeddings and relationships), `consolidated` (tiered summaries: core, warm, narrative).
- **Consolidation as dual-output.** Dreaming produces compressed memory AND training data for the unconscious. Single process, two outputs.

### Requirement 3: Per-venue knowledge

- **Namespaced database files.** One `.marianne/memory/venues/{venue_hash}/venue.db` per venue. Cross-venue knowledge (rosetta patterns, general skills) lives in `~/.marianne/memory/global.db`.
- **Retrieval scoping.** The retrieval interface accepts a venue context and searches venue-local first, then falls back to global. Global-only entities (instruments, error taxonomies) are shared.

### Requirement 4: Per-user relationships

- **User relationship store.** `~/.marianne/memory/users/{user_hash}/relationship.db` per user. Tracks interaction history, communication preferences, trust level, expertise areas.
- **Compose and concierge integration.** Surfaces 3, 4, and 6 from the integration research all read user relationship data. The memory spec defines the retrieval interface for each.

### Requirement 5: RLF compatibility

- **Direct tier mapping.** L1 (identity) → `consolidated` with `tier='core'`. L2 (profile) → `consolidated` with `tier='warm'`. L3 (recent) → `facts` table (unconsolidated). L4 (growth) → `consolidated` with `tier='narrative'`.
- **Export interface.** `RLFPersonExport` converts Marianne's memory representation into RLF's format. No live API dependency — the integration path is an exporter/converter.

### Requirement 6: CAM readiness

- **Identity-scoped Protocol.** Every `MemoryStore` method accepts `agent_id`. Initial implementation uses a constant (Marianne Mozart).
- **Shared entities, private facts.** Entity nodes are global. Fact attachment to entities is agent-scoped. ConductorDecision records are the first shared-namespace artifact.
- **CAMExportProtocol.** Serializes memory state as structured data aligned with the RLF person model.

---

## Implications for Unconscious Spec

Grouped by the S6 unconscious / smart conductor guardrails requirements.

### Decision Source Pluggability

- **`DecisionSource` Protocol** with three implementations: `HeuristicDecisionSource` (ships immediately, zero dependencies), `LocalModelDecisionSource` (llama-cpp-python + GGUF), `RemoteDecisionSource` (delegates to cloud instrument via instrument registry).
- **Fallback stack.** The smart conductor holds a prioritized list of decision sources and falls through on failure: local model → heuristics → mechanical execution. Automatic, logged, reversible.
- **Phase 1 ships heuristics only.** The Protocol and ConductorDecision record ship first. The local model implementation ships when the training data pipeline produces 500+ labeled examples.

### Sub-Second Inference

- **Default model:** Qwen3-4B Q4_K_M (2.5GB GGUF). Auto-fallback chain: Qwen3-4B → Qwen3.5-2B → Qwen3.5-0.8B → heuristics-only, based on detected hardware.
- **Runtime:** llama-cpp-python for in-process inference. Optional dependency via `pip install marianne[unconscious]`.
- **GBNF grammars** for constrained output. Classification decisions (5-10 tokens) complete in sub-200ms on any hardware. Reasoned decisions (50 tokens with explanation) require GPU for sub-500ms.
- **Three latency tiers:** Fast (<200ms classification), Medium (<500ms reasoned), Slow (<3s diagnostic). The conductor's configuration selects the tier.

### Inspectable Decisions

- **`ConductorDecision` record** produced for every smart conductor evaluation, regardless of source. Fields: trigger event, decision, reasoning, source, confidence, latency, outcome (filled asynchronously).
- **Placement:** In `src/marianne/core/` alongside `CheckpointState` and `SheetState`. Both specs reference the same type.
- **Storage:** Written to the memory store's shared-namespace `decisions` table (the first CAM artifact). Also persisted in checkpoint state for crash recovery.

### Guardrails

- **Action budget per job.** Default 10. Counts flow-control actions only. Memory reads, memory writes, observe-only decisions, and escalation are budget-free.
- **Authority hierarchy enforcement.** Score YAML > Marianne's judgment > baton defaults. The `DecisionSource` implementation never produces actions that contradict YAML-declared triggers or loops. The smart conductor checks YAML declarations before applying any decision.
- **Graceful degradation.** Three tiers: Full → Heuristic → Mechanical. Automatic downgrade on repeated failure (3 consecutive timeouts or 10 consecutive errors). Manual upgrade via config reload. Tier transitions logged at WARNING level.

### Training Data Pipeline

- **Source:** ConductorDecision records with positive outcomes (the decision led to success).
- **Quality filter:** Only train on decisions where the outcome field confirms the action worked. Exclude neutral (observe_only) and negative outcomes.
- **Pipeline:** Extract from memory store → format as instruction-tuning examples → LoRA fine-tuning via Unsloth → merge into GGUF → deploy to `~/.marianne/models/`.
- **Trigger:** 500+ labeled decisions → offer fine-tuning. The dreaming cycle can initiate this check.

### Conductor Integration

- **Post-event callback in BatonCore.** After `handle_event()`, before `dispatch_ready()`, call `smart_conductor.evaluate(event, state)`. Direct callback, not EventBus subscriber.
- **Event subscription scope.** The unconscious subscribes to `SheetAttemptResult`, `EscalationNeeded`, `ResourceAnomaly`, and `CronTick`. Other events carry no conductor-decision payload.
- **Async boundary.** Model inference runs via `asyncio.to_thread()`. Never blocks the baton's event loop. The `DecisionSource.evaluate()` method is async; the underlying model call is synchronous.
- **Not an instrument.** The unconscious is a conductor-internal component. Instruments are for musicians. The unconscious serves the conductor. Do not register in `InstrumentRegistry`.

---

## Open Questions

### For the Composer

1. **Does CAM have an external specification?** If yes, the memory spec should align directly. If no, the six architectural properties defined in the CAM research are the working definition. The researcher found zero external references.

2. **Hardware constraints for the unconscious.** What GPU/RAM is available on the primary development and deployment machines? This determines whether the default is Qwen3-4B (needs 5GB RAM, GPU preferred) or Qwen3-1.7B (needs 3GB, CPU-viable).

3. **Priority between CAM-readiness and shipping speed.** Multi-tenant isolation adds implementation complexity. Should the initial memory spec implement multi-tenancy, or design the Protocol for it and implement single-tenant?

4. **Decision quality benchmark.** How do we evaluate whether the unconscious model beats heuristics? The learning store has telemetry but not labeled "what should the conductor have done?" ground truth. The composer should define correctness criteria for common failure scenarios.

5. **Consolidation frequency and trigger.** How often should dreaming run? Legion's model triggers at ~1500 words of accumulated memory. The conductor's telemetry grows faster. Options: time-based (hourly), size-based (every N facts), idle-based (when no jobs are running), or event-based (triggered by the conductor when fact count exceeds a threshold). Each has different implications for read consistency, cost, and training data freshness.

### For the Design Specs

6. **ConductorDecision placement.** The integration research recommends `src/marianne/core/`. Both specs reference this type. Define it once, in one location.

7. **Memory file layout.** `~/.marianne/memory/` for user-global identity, `~/.marianne/memory/venues/{hash}/` for per-venue knowledge, `~/.marianne/memory/users/{hash}/` for per-user relationships? Or all in a single database with scoping columns? The file-based approach is inspection-friendly; the database approach is query-friendly. Both work for SQLite.

8. **Embedding model upgrade path.** If `all-MiniLM-L6-v2` proves inadequate for domain-specific orchestration content, how does the system migrate? The schema should store both the embedding vector and the model identifier, enabling gradual re-embedding without a flag-day migration.

9. **Entity extraction method.** During consolidation, who extracts entities from raw facts? Options: the same LLM call that does consolidation (piggybacking — cheapest), a separate extraction pass (most accurate), or the unconscious model itself (fastest but lowest quality). The trade-off is cost vs. quality vs. speed.

10. **Concierge memory topology.** If the concierge runs on a different host (e.g., a cloud server proxying Discord), does it get its own memory copy or query the conductor's memory remotely? S6 says all four modes share memory, but network topology may prevent direct file access. This needs resolution before concierge implementation.

11. **Escalation budget semantics.** The integration research recommends escalation be budget-free (communication should never be blocked). The S6 spec does not distinguish escalation from other actions in the budget. The specs should take a position.

### Research Gaps Neither Stream Resolved

12. **Fine-tuning data volume requirements.** The 500-example threshold for fine-tuning viability is an estimate. Real validation requires benchmarking the base model zero-shot against fine-tuned variants on a Marianne-specific decision set. No such benchmark exists yet.

13. **Entity extraction quality on orchestration-specific content.** All entity extraction research uses natural language documents. Marianne's "documents" are execution events, YAML configs, and error traces. Whether standard NER/OpenIE techniques produce useful entities from this content is untested.

14. **Cross-venue knowledge transfer.** When Marianne learns a pattern in venue A (e.g., "this instrument fails under high fan-out"), should that pattern automatically transfer to venue B? The CAM properties allow it (shared entities), but the venue isolation model discourages it. The boundary between "this is general knowledge" and "this is venue-specific" needs a heuristic or a manual promotion mechanism.

---

## Cross-Stream Dependency Map

```
Memory Spec depends on:
├── CAM research → architectural properties, export protocol, multi-tenant scoping
├── Memory systems → substrate choice, retrieval model, consolidation pattern
├── Integration → six integration surfaces, MemoryStore Protocol methods, latency budgets
└── Unconscious → training data extraction API, entity graph as fast retrieval path

Unconscious Spec depends on:
├── Local models → model selection, runtime, distribution, fine-tuning pipeline
├── Integration → DecisionSource Protocol, baton hook point, action budget
├── Memory → MemorySnapshot type, retrieval interface for decision context
└── CAM → decision records as shared knowledge (first CAM artifact)

Both depend on:
├── ConductorDecision type (shared, placed in src/marianne/core/)
├── ConductorContext type (shared, assembles baton state + memory + learning store)
├── Graceful degradation tiers (defined once, referenced by both)
├── S6 authority hierarchy (Score YAML > Marianne > baton defaults)
└── Dreaming cycle design (dual-output: memory compression + training data)
```

---

## Summary of Recommendations

| Decision | Recommendation | Confidence | Source |
|----------|---------------|------------|--------|
| Memory substrate | SQLite + local embeddings + entity graph | High | All four streams |
| Embedding model | all-MiniLM-L6-v2 bundled (80MB) | High | Memory systems |
| Unconscious model | Qwen3-4B Q4_K_M (2.5GB) with auto-fallback | High | Local models |
| Inference runtime | llama-cpp-python (in-process) | High | Local models |
| Decision interface | DecisionSource Protocol (3 implementations) | High | Integration + Local models |
| Memory access scope | Identity-scoped via Protocol, single-tenant initial impl | Medium | CAM + Integration |
| Consolidation model | ADD-only facts + tiered dreaming (Legion pattern extended) | High | Memory systems + Emergent #1 |
| Retrieval model | Multi-signal fusion (semantic + keyword + entity) | High | Memory systems + Emergent #2 |
| Entity graph purpose | Unconscious fast path + associative retrieval | High | Emergent #2 |
| Baton integration | Post-event callback, not EventBus subscriber | High | Integration |
| CAM readiness | Architectural properties, not API conformance | Medium | CAM |
| Learning store | Keep as-is, bridge via TelemetryIngester | High | Integration + Memory systems |
| Action budget | Flow-control actions only; escalation budget-free | High | Integration + Emergent |
| Decision records | Shared namespace (first CAM artifact) | Medium | CAM + Emergent #4 |
| Degradation framing | Developmental tiers, not just fallback modes | Medium | Emergent #5 |

---

*Research synthesis: 2026-04-19*
*Four streams consolidated: CAM, Memory Systems, Local Models, Integration Surface*
*Prior-art brief used for triangulation anchoring*
*Word count: ~3800*
