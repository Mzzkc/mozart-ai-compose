# Unconscious Local Model Design

**Date:** 2026-04-18 (revised 2026-04-19, merged specification)
**Author:** Synthesized from two independent design runs + adversarial reviews + ground-truth codebase verification
**Dependency:** S6 (Marianne Mozart) from the Baton Primitives spec; Memory System spec (for retrieval interface)
**Research basis:** `03-research-dossier.md` (synthesizing local models, integration surface, CAM, and memory systems research)

---

## Status

status: designed

This spec defines the unconscious — Marianne Mozart's fast judgment layer. Where the memory system provides the data (what Marianne knows), the unconscious provides the inference (what Marianne decides). It is a pluggable decision source that the smart conductor consults in response to baton events, producing structured `ConductorDecision` records that are inspectable regardless of whether the source is a heuristic rule, a local model, or a remote LLM.

The unconscious is not a replacement for the existing baton mechanics. It is an advisory layer that fills gaps in the score's declared behavior. Score YAML always wins. The unconscious suggests; the score dictates.

**Provenance:** This merged specification uses the promoted Run 1 spec (which was revised after a full review cycle to accept-with-major-revisions status) as its backbone, incorporating Run 2's novel contributions: event state snapshots, two-pass context assembly with pre-filter, outcome tracking via decision-ID correlation, shadow mode data integrity with the `applied: bool` field, the complete 10-action DecisionAction enum (LOOP and REORDER added), action budget persistence in BatonJobState checkpoint, dynamic GBNF generation from the DecisionAction enum, burn-in threshold for Tier 2 → Tier 1 promotion, cross-concert ordering via periodic ConductorTick, LOOP action semantics distinguishing ad-hoc from score-declared loops, and AMD GPU detection via `rocm-smi`. All Critical and Important review findings from both runs have been addressed. Factual claims about existing Marianne code have been verified against the codebase. Architectural conflicts have been resolved per the synthesis brief's recommendations.

---

## Goal

Give Marianne Mozart a fast, pluggable decision-making capability that produces sub-second judgments during live execution, degrades gracefully when components are unavailable, learns from accumulated experience, and respects the authority hierarchy (Score YAML > Marianne's judgment > baton defaults) — all while operating fully offline via in-process inference with no external service dependency. The unconscious sees the whole orchestra: it evaluates cross-concert ordering and inter-job semantic dependencies, not just per-sheet decisions. Every decision it makes is inspectable after the fact via `ConductorDecision` records that capture what was observed, what was decided, why, and what happened. This directly satisfies S6's requirement for a "local fine-tuned or distilled model trained on Marianne's accumulated knowledge" that provides "fast feel — the unconscious pattern matching that a human conductor develops over years."

---

## Requirements

### S6 Requirements (Verbatim) with Spec-Level Commitments

**S6 Requirement: "The decision source is pluggable — heuristics today, a local model tomorrow."**
Commitment: `DecisionSource` Protocol (FR-01) with three implementations. Heuristics ship immediately as the quality baseline. Local model ships when training data threshold is met. Remote LLM is a future option for complex diagnostic reasoning.

**S6 Requirement: "Every decision is inspectable after the fact."**
Commitment: `ConductorDecision` frozen dataclass (FR-02) records every evaluation regardless of source. The `outcome` field is filled asynchronously via decision-ID correlation (see Outcome Tracking section). Records persist in the memory store's shared `decisions` table — the first CAM artifact.

**S6 Requirement: "The smart conductor subscribes to the event stream across ALL running work."**
Commitment: EventBus subscriber architecture (FR-03). The smart conductor subscribes to relevant events and evaluates asynchronously. Flow-control events pushed into the baton's inbox. The smart conductor never directly modifies baton state or blocks event handling. Events carry immutable state snapshots (see Event State Snapshots) so the smart conductor evaluates state-as-it-was, not state-as-it-is-now.

**S6 Requirement: "Action budget per job — maximum number of flow-control interventions."**
Commitment: Persisted budget counter (FR-04) in `BatonJobState`. Survives conductor restarts via checkpoint persistence. Escalation is budget-free — blocking communication when budget is exhausted defeats the guardrail's purpose.

**S6 Requirement: "Graceful degradation — smart conductor is advisory, never blocking."**
Commitment: Three-tier degradation state machine (FR-05). Full → Heuristic → Mechanical. Automatic downgrade on consecutive failures. Tier 3 never auto-re-enables. Tier 2 → Tier 1 promotion requires a burn-in threshold of successful heuristic evaluations. Each tier maps to a developmental stage.

**S6 Requirement: "She never overrides YAML-declared behavior."**
Commitment: Authority hierarchy enforcement (FR-06) checked on fast context before any memory or model I/O. Score YAML > Marianne's judgment > baton defaults.

**S6 Requirement: "Marianne can invoke any flow control primitive at will: loop, recover, retry, skip, reorder, inject context, escalate."**
Commitment: Complete `DecisionAction` enum (FR-11) covering all S6 primitives: CONTINUE, RETRY, PAUSE, ESCALATE, SKIP, HEAL, INJECT_CONTEXT, LOOP, REORDER, OBSERVE_ONLY.

**S6 Requirement: "Cross-concert/score ordering — the real magic."**
Commitment: Periodic cross-concert evaluation (FR-12) on a configurable timer (default 5s). Soft priority adjustments, not hard preemption. Semantic awareness of inter-job dependencies.

### Functional Requirements

**FR-01: DecisionSource Protocol.** A `typing.Protocol` defining the interface for all decision sources. Three implementations: `HeuristicDecisionSource` (rules-based, zero dependencies), `LocalModelDecisionSource` (llama-cpp-python + GGUF), `RemoteDecisionSource` (delegates to a cloud API client directly — NOT through the Instrument abstraction, which serves musicians, not the conductor). All three produce the same `ConductorDecision` record.

**FR-02: ConductorDecision record.** Every decision produces a frozen dataclass containing: trigger event type, trigger summary, decision action, reasoning, source identifier, confidence score, latency measurement, context snapshot, optional action payload, an `applied: bool` field distinguishing active decisions from shadow-mode evaluations, and an asynchronously-filled outcome field. Placed in `src/marianne/core/` alongside `CheckpointState` to prevent circular imports between memory and unconscious packages.

**FR-03: EventBus subscriber.** The smart conductor subscribes to relevant events via the existing EventBus. It observes events asynchronously and pushes flow-control events into the baton's inbox. The smart conductor is decoupled from the baton's dispatch cycle — it never blocks event handling or dispatch. Events carry immutable state snapshots (see Event State Snapshots) so the smart conductor evaluates state-as-it-was, not state-as-it-is-now. Events subscribed: `SheetAttemptResult`, `EscalationNeeded`, `LoopCompleted`, `SheetTriggerFired`, `ResourceAnomaly`.

**FR-04: Action budget per job.** A configurable counter (default 10) limiting flow-control interventions per job. Budget-consuming actions: RETRY, PAUSE, SKIP, HEAL, INJECT_CONTEXT, LOOP, REORDER. Budget-free: CONTINUE, OBSERVE_ONLY, ESCALATE. Memory reads, writes, and decision recording are always free. Budget persisted in `BatonJobState` and survives conductor restarts.

**FR-05: Graceful degradation tiers.**
- **Tier 1 (Full):** Memory available, local model loaded. Sub-second decisions with rich context.
- **Tier 2 (Heuristic):** Memory or model unavailable. Rules-based decisions with event-local data.
- **Tier 3 (Mechanical):** Smart conductor disabled. Baton defaults only. Never auto-re-enables.

Promotion from Tier 2 to Tier 1 requires: model loads successfully, memory retrieval succeeds, AND a burn-in threshold of at least 100 successful heuristic evaluations has been met. The burn-in ensures sufficient venue telemetry exists before the model takes over.

**FR-06: Authority hierarchy enforcement.** Before acting, check whether score YAML declares behavior (triggers, loops, skip_when) for this sheet/event. If it does, record decision as `outcome="overridden_by_yaml"` and do not apply. This check runs on fast context (<1ms) — no memory or model I/O needed.

**FR-07: Model auto-detection.** Detect GPU/RAM at startup via stdlib only (with the exception of `rocm-smi` for AMD GPU detection). GPU available → Qwen3-4B. Apple Silicon → Qwen3-4B with Metal acceleration. AMD GPU (ROCm) → Qwen3-4B with Vulkan fallback. CPU with >=6GB available RAM → Qwen3-1.7B. Low RAM → heuristics only. Config override bypasses auto-detection. Non-NVIDIA/Apple/AMD GPUs (e.g., Intel Arc) are unsupported in v1 and default to CPU execution.

**FR-08: GBNF grammar for structured output.** Dynamically generated from `DecisionAction` enum at runtime. Adding a new action automatically includes it in the grammar. Ensures grammar and enum stay synchronized. Zero parse failures — grammar constrains token sampling at each step.

**FR-09: Training data extraction.** Interface for extracting `(context, decision, outcome)` tuples from the memory store's `decisions` table. Rule-based extraction via SQL queries — never depends on LLM availability. Quality filter: positive outcome + confidence >= 0.7 + causal attribution + `applied = true` (shadow decisions are explicitly excluded — see Shadow Mode Data Integrity).

**FR-10: Model download and caching.** GGUF downloaded from HuggingFace Hub on first use, cached in `~/.marianne/models/`. SHA256 verification. Resume-on-interrupt. `mzt model import` for offline/air-gapped provisioning. Graceful fallback to heuristics when download fails.

**FR-11: Complete flow-control vocabulary.** `DecisionAction` enum: CONTINUE, RETRY, PAUSE, ESCALATE, SKIP, HEAL, INJECT_CONTEXT, LOOP, REORDER, OBSERVE_ONLY. Implements the full S6 vocabulary including loop and reorder primitives.

**FR-12: Cross-concert ordering.** Periodic `ConductorTick` (default 5s) evaluates all active jobs. Produces `REORDER` decisions with `action_payload` containing priority adjustments. Soft adjustments — running jobs continue uninterrupted, reorder affects which job gets the next dispatch slot. Not per-event (would bottleneck during fan-out).

**FR-13: Shadow mode.** Local model evaluates events and logs decisions without affecting execution. Both heuristic and model decisions recorded. Shadow decisions carry `applied=false` and are excluded from training data extraction by default. Configured via `conductor.unconscious.shadow_mode: true`.

**FR-14: Event state snapshots.** Flow-control events published to the EventBus carry an immutable state snapshot of the baton's per-sheet and per-job state at the moment the event was emitted. The smart conductor reads this snapshot rather than querying mutable in-memory baton state, eliminating race conditions between event delivery and concurrent state mutations during fan-out.

### Non-Functional Requirements

**NFR-01: Decision latency.** <500ms total end-to-end. Heuristic <1ms. Local model: 50-200ms (GPU), 150-600ms (CPU with 1.7B). Cross-concert ordering on periodic tick, not latency-bound per individual event.

**NFR-02: Memory footprint.** Qwen3-4B: ~5GB total (3.5GB model + 1.5GB conductor overhead). Qwen3-1.7B: ~3.5GB total (2GB model + 1.5GB conductor). Heuristics only: ~1.5GB.

**NFR-03: No external daemon.** In-process via llama-cpp-python. No Ollama, no HTTP server, no separate process. The conductor process owns the model lifecycle.

**NFR-04: Thread safety.** Model inference wrapped in `asyncio.Lock`. `asyncio.to_thread()` wraps synchronous inference. `asyncio.wait_for()` enforces timeout with heuristic fallback. Low contention — sequential event processing — but prevents concurrent inference corruption.

**NFR-05: Deterministic heuristics.** `HeuristicDecisionSource` produces deterministic decisions for identical input. This makes testing straightforward and provides a stable baseline for model comparison.

**NFR-06: Non-blocking decision persistence.** `MemoryStore.record_decision()` is dispatched as a fire-and-forget background task via `asyncio.create_task(asyncio.to_thread(...))` to prevent SQLite disk I/O from stalling the EventBus dispatch loop under heavy orchestration load or fan-out.

---

## Design

### Core Types

**DecisionAction enum.** The complete vocabulary of smart conductor actions:
- `CONTINUE` — No intervention needed. Budget-free.
- `RETRY` — Retry the sheet with current or modified context.
- `PAUSE` — Pause the baton's dispatch cycle. In-flight sheets finish.
- `ESCALATE` — Hand to composer via fermata system. Budget-free.
- `SKIP` — Mark target sheet(s) as SKIPPED.
- `HEAL` — Invoke self-healing infrastructure.
- `INJECT_CONTEXT` — Feed cadenzas or preludes to a sheet.
- `LOOP` — Create ad-hoc loop for a struggling range (where score is silent).
- `REORDER` — Adjust cross-concert scheduling priorities.
- `OBSERVE_ONLY` — Budget exhausted; record observation without acting. Budget-free.

**ConductorDecision dataclass (frozen).** Fields:
- `id: str` — UUID for tracking and outcome correlation
- `job_id: str` — Which job this decision relates to
- `sheet_num: int | None` — Which sheet (None for cross-concert decisions)
- `timestamp: datetime` — When the decision was made
- `trigger_event_type: str` — The event type that triggered evaluation
- `trigger_summary: str` — Human-readable summary of the trigger
- `action: DecisionAction` — The decided action
- `reasoning: str` — Why (rule name for heuristic, response for model, confidence + feature summary for unconscious)
- `decision_source: str` — Which source produced this (e.g., "heuristic", "local_model_qwen3_4b", "shadow_local_model")
- `model_name: str | None` — Model identifier if applicable
- `confidence: float` — 0.0-1.0 confidence in the decision
- `latency_ms: float` — How long evaluation took
- `action_budget_remaining: int` — Budget state after this decision
- `context_snapshot: dict[str, Any]` — Serialized context for reproducibility
- `action_payload: dict[str, Any] | None` — Action-specific data (priority adjustments for REORDER, sheet range for LOOP, etc.)
- `applied: bool` — Whether this decision was actually applied to execution (`false` for shadow-mode and overridden decisions)
- `outcome: str | None` — Filled asynchronously after baton processes consequence
- `outcome_timestamp: datetime | None` — When outcome was recorded

Placed in `src/marianne/core/` alongside `CheckpointState` and `SheetState`. Both the memory system (records decisions) and the unconscious (produces decisions) reference this type.

**ConductorContext dataclass.** Two tiers of context fields:

Fast context (always populated, <1ms from immutable event state snapshot):
- `job_id`, `sheet_num`, `event_type`, `event_data`
- `sheet_state` — current sheet status, attempts, validation details (from event snapshot, not mutable baton state)
- `job_state` — overall job progress, sheet statuses (from event snapshot)
- `cost_accumulated_usd`, `cost_limit_usd`
- `action_budget_remaining`
- `active_jobs_summary` — list of all running jobs (for cross-concert)
- `yaml_declared_triggers` — pre-loaded from job config (for authority check)
- `yaml_declared_loops` — pre-loaded from job config

Rich context (populated only when intervention likely, <200ms):
- `memory_snapshot: MemorySnapshot | None` — relevant memories from entity graph and/or semantic search
- `learning_patterns: list[PatternRecord] | None` — relevant patterns from the learning store

**DecisionSource Protocol:**
```python
class DecisionSource(Protocol):
    @property
    def source_name(self) -> str: ...

    @property
    def is_available(self) -> bool: ...

    async def evaluate(self, context: ConductorContext) -> ConductorDecision: ...

    async def shutdown(self) -> None: ...
```

### Smart Conductor Component

The `SmartConductor` is the glue between the EventBus, the decision sources, memory, and the baton's inbox. It subscribes to events, builds context from event-carried state snapshots, evaluates decisions, and pushes flow-control events.

**Event flow:**
```
BatonCore.handle_event(e)
    |
    +-- [processes event normally]
    |
    +-- EventBus.publish(e, state_snapshot=frozen_snapshot)
            |
            +-- SmartConductor._on_event(e, snapshot)  [async, non-blocking]
                    |
                    +-- Build fast context from snapshot (<1ms)
                    +-- Check authority hierarchy
                    +-- Pre-filter (clean completion? -> CONTINUE, exit)
                    +-- Hydrate rich context (memory + learning, <200ms)
                    +-- Evaluate via DecisionSource (with timeout)
                    +-- Shadow mode parallel eval (if enabled)
                    +-- Budget accounting
                    +-- Record decision (fire-and-forget background task)
                    |
                    +-- baton.push_event(flow_control_event, decision_id=decision.id)
```

**Startup sequence:**
1. Subscribe to `SheetAttemptResult`, `EscalationNeeded`, `LoopCompleted`, `SheetTriggerFired`, `ResourceAnomaly`
2. Start cross-concert evaluation ticker (default 5s)

**Key architectural properties:**
- **Asynchronous observation.** The smart conductor receives events after the baton has processed them. It never blocks `handle_event()` or `dispatch_ready()`.
- **Snapshot-based evaluation.** The smart conductor reads the immutable state snapshot carried by the event, not mutable in-memory baton state. This eliminates the race condition where event delivery delay during fan-out causes the smart conductor to observe future state that doesn't match the triggering event.
- **Inbox-based action.** When the smart conductor decides to intervene, it pushes a flow-control event into the baton's existing inbox. The baton processes this in its next dispatch cycle.
- **Crash isolation.** If the smart conductor crashes, the baton continues with mechanical defaults (Tier 3).
- **Non-blocking persistence.** Decision writes to SQLite are dispatched as fire-and-forget background tasks. The smart conductor does not await disk I/O in the event callback path.
- **No new coupling.** The EventBus and baton inbox already exist. The smart conductor is another subscriber — pluggable, removable, replaceable.

### Event State Snapshots

To prevent race conditions between event delivery and concurrent state mutations during fan-out, the baton attaches an immutable state snapshot to each event before publishing it to the EventBus:

```python
@dataclass(frozen=True)
class EventStateSnapshot:
    """Immutable point-in-time capture of baton state when the event was emitted."""
    sheet_num: int | None
    sheet_status: str | None
    sheet_attempts: int
    validation_pass_rate: float | None
    validation_details: dict[str, Any] | None
    job_progress: dict[str, Any]  # {completed: N, total: M, ...}
    cost_accumulated_usd: float
    cost_limit_usd: float | None
    active_jobs_summary: list[dict[str, Any]]
    timestamp: float
```

The snapshot is frozen (immutable). The smart conductor builds its `ConductorContext` from this snapshot rather than querying `BatonJobState` directly. This means the smart conductor always evaluates the state *as it was when the event occurred*, even if the baton has since processed additional events and mutated its in-memory state.

This design resolves the critical EventBus ordering race condition identified in both runs' reviews (Run 1 C-1, Run 2 C-3): during fan-out, multiple sheets complete concurrently and each emits an event. Without snapshots, a delayed event would cause the smart conductor to read the baton's current (already-mutated) state rather than the state at the time the event fired. Snapshots make each evaluation self-contained and temporally correct.

### Outcome Tracking via Decision-ID Correlation

The `ConductorDecision.outcome` field is filled asynchronously after the baton processes the consequence of the decision. The correlation mechanism works as follows:

1. The smart conductor produces a `ConductorDecision` with a UUID `id` field and pushes a flow-control event into the baton's inbox. The flow-control event carries the `decision_id`.
2. The baton processes the flow-control event (e.g., retries a sheet, pauses dispatch). The `decision_id` is stored on the affected `SheetExecutionState` as `pending_decision_id`.
3. When the consequent event fires (e.g., `SheetAttemptResult` after a retry), the baton includes the `pending_decision_id` in the event payload.
4. The smart conductor observes this consequent event, reads the `pending_decision_id`, and calls `MemoryStore.update_decision_outcome(decision_id, outcome)` to close the loop.

**Outcome values:**
- `"success"` — the subsequent attempt completed successfully
- `"failed"` — the subsequent attempt also failed
- `"improved"` — validation pass rate improved (but not yet completed)
- `"no_change"` — no measurable difference from the intervention
- `"overridden_by_yaml"` — the decision was not applied because YAML declared behavior
- `"not_applied"` — shadow mode decision (never applied)

For decisions where the consequence is not a single subsequent attempt (e.g., PAUSE, REORDER), the outcome is filled when the job reaches a terminal state or a subsequent evaluation observes the effect. REORDER outcomes are assessed on the next `ConductorTick` by comparing job progress against the pre-reorder baseline.

This mechanism ensures the training data pipeline has reliable `(context, decision, outcome)` tuples. Decisions without outcomes (e.g., due to conductor restart before the consequence fires) are excluded from training data extraction by the quality filter.

This section directly resolves Run 2 review C-1 (outcome tracking mechanism undefined) and was absent from Run 1's original draft.

### Shadow Mode Data Integrity

Shadow mode runs the local model alongside heuristics for empirical comparison. Data integrity requires that shadow decisions never contaminate the training pipeline:

1. **`applied` field.** Every `ConductorDecision` carries an `applied: bool` field. Active decisions have `applied=true`. Shadow decisions have `applied=false`.
2. **No outcome updates for shadow decisions.** The outcome correlation mechanism (Decision-ID Correlation above) only fires for applied decisions. Shadow decisions retain `outcome="not_applied"` permanently. Since the heuristic's decision was what actually executed, the shadow model's decision has no causal relationship to the observed outcome.
3. **Training data extraction filter.** The SQL extraction query in FR-09 includes `WHERE applied = true AND decision_source != 'shadow_local_model'`. This is a belt-and-suspenders approach: either filter alone prevents data poisoning; both together make the intent explicit.

Without this protection, shadow decisions would be falsely credited with the heuristic's outcomes, poisoning the training data with examples where the model's proposed action was never tested.

This section directly resolves Run 2 review C-2 (shadow mode data poisoning).

### Two-Pass Context Assembly

Most events need no intervention. Two-pass prevents I/O waste:

**Pass 1 — Fast context (<1ms).** Read from the immutable `EventStateSnapshot` carried by the event, plus the pre-loaded YAML config for authority checks. No mutable state queries, no I/O.

**Pre-filter.** A deterministic heuristic checks if intervention is plausible:
- Clean completion (status="completed", attempts<=1) → CONTINUE immediately, zero I/O
- First attempt (attempts=0) → CONTINUE immediately
- Event carries `skip_smart_conductor=true` flag → CONTINUE immediately (synthetic events, internal resets)
- Any anomaly (failure, high retry count, cost concern, escalation) → proceed to Pass 2

**Pass 2 — Rich context (<200ms).** Only when pre-filter signals possible intervention:
- `MemoryStore.retrieve_relevant()` with fast latency tier (<50ms) — entity-graph lookup, not embedding similarity
- `GlobalLearningStore.get_patterns()` for relevant learned patterns

In a typical job with 10 sheets and 3 retries, ~7 events exit at pre-filter with zero I/O and ~3 events load full context. This is the single best performance optimization identified in the design process — Run 2's contribution. It ensures the smart conductor adds negligible overhead to normal execution.

### HeuristicDecisionSource

Rules-based, zero dependencies, always available. Ships immediately as the quality baseline. The local model must beat this by >=10% to justify its RAM cost.

**Core rules (evaluated in order, first match wins):**
1. **High retry + declining validation** — `attempts > max_retries / 2` AND validation pass rate declining → `ESCALATE` (confidence 0.75). Pattern: the sheet is getting worse, not better.
2. **Validation improving** — validation pass rate increasing across attempts → `RETRY` (confidence 0.80). Pattern: the musician is learning.
3. **Cost budget pressure** — cost >80% of `cost_limit_usd` → `PAUSE` (confidence 0.85). Pattern: running out of budget. **Prerequisite: this rule is gated on the token cost tracking remediation (see Cross-Cutting Concerns in the S6 parent spec). Until cost tracking is reliable, this rule is disabled and the fallback metric is attempt count: >80% of max_retries triggers the same PAUSE logic.**
4. **Rate-limited instrument** — instrument returned rate limit → `CONTINUE` (confidence 0.90). Pattern: baton's built-in backoff handles this.
5. **Execution crashed** — exit code != 0, non-rate-limit error → `HEAL` (confidence 0.65). Pattern: invoke self-healing infrastructure.
6. **Cross-concert dependency** — job whose output unblocks other jobs → `REORDER` with priority boost (confidence 0.60). Pattern: unblock downstream work.
7. **Repeated same error** — same error signature 3+ times → `INJECT_CONTEXT` with error analysis cadenza (confidence 0.70). Pattern: musician needs help understanding the error.

Deterministic for the same input. Rules are evaluated in order; first match wins.

**Cost-based rules (3, and cost-based REORDER deprioritization) depend on accurate token cost tracking. The S6 parent spec explicitly flags current cost tracking as unreliable. These rules are inactive until the cost tracking remediation is complete. Until then, attempt-count and retry-rate serve as proxy metrics for the same decisions.** This gating directly resolves Run 2 review C-4 (dependency on broken cost tracking).

### LocalModelDecisionSource

In-process GGUF inference via llama-cpp-python. Produces structured JSON constrained by GBNF grammar:

```json
{"action": "RETRY", "confidence": 0.82, "reasoning": "Validation rate improving from 40% to 65% across 3 attempts"}
```

Full `ConductorDecision` assembled from model output + metadata (job_id, sheet_num, latency, timestamp, etc.).

**Model selection via hardware auto-detection cascade:**

| Priority | Condition | Model | Expected Latency |
|----------|-----------|-------|-----------------|
| 1 | NVIDIA GPU (`nvidia-smi` found) | Qwen3-4B Q4_K_M (2.5GB) | 50-200ms |
| 2 | Apple Silicon (arm64 + macOS) | Qwen3-4B Q4_K_M with Metal | 80-250ms |
| 3 | AMD GPU (`rocm-smi` found) | Qwen3-4B Q4_K_M with Vulkan | 100-300ms |
| 4 | CPU, >=6GB available RAM | Qwen3-1.7B Q4_K_M (1.4GB) | 150-600ms |
| 5 | CPU, <6GB available RAM | None (heuristics only) | <1ms |
| 6 | Config override | User-specified model | varies |

Hardware detection is stdlib-only (except `rocm-smi` shim):
- `shutil.which("nvidia-smi")` for NVIDIA GPU
- `platform.system() == "Darwin" and platform.machine() == "arm64"` for Apple Silicon
- `shutil.which("rocm-smi")` for AMD ROCm GPU
- `os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")` for available RAM (Linux/macOS)
- Conservative 4GB default for Windows/WSL where `os.sysconf` may not work
- Non-NVIDIA/Apple/AMD GPUs (e.g., Intel Arc) are unsupported in v1 and fall through to CPU execution

**Thread safety:**
- `asyncio.Lock` prevents concurrent inference from corrupting model state
- `asyncio.to_thread()` offloads synchronous inference to the thread pool
- `asyncio.wait_for()` with configurable timeout (default 500ms) falls back to heuristics

**Model configuration:**
- `n_ctx=2048` — sufficient for decision prompts (<500 tokens input + <50 tokens output)
- `n_gpu_layers=-1` — offload all layers to GPU when available
- `temperature=0.1` — near-deterministic classification with minimal exploration
- Model loading takes 2-5 seconds at startup; subsequent inferences are fast

### GBNF Grammar — Dynamic Generation

GBNF grammar is dynamically generated from the `DecisionAction` enum at runtime. This solves the synchronization problem: adding a new action to the enum automatically includes it in the grammar. No manual grammar maintenance.

The grammar constrains token sampling at each generation step. When the grammar says the next token must be `"` (start of string), the sampler masks out all non-matching tokens. The model physically cannot produce invalid JSON. This provides:

1. **Zero parse failures.** Output is guaranteed valid. No try/except with fallback parsing.
2. **Reduced latency.** Grammar pruning eliminates wasted tokens. 30-50% faster than unconstrained generation with post-hoc parsing.
3. **Automatic synchronization.** Grammar matches enum at all times.

Run 1's original spec hardcoded the GBNF grammar as a static string literal. Run 2 identified the synchronization problem this creates when DecisionAction expands (which it did — LOOP and REORDER were added). Dynamic generation is the correct approach.

### RemoteDecisionSource (Future)

Delegates evaluation to a cloud LLM for complex reasoning that exceeds local model capabilities. Reserved for diagnostic analysis and compose conversations. Not implemented in v1 — the `DecisionSource` Protocol ensures it plugs in when needed.

**Clarification on abstraction boundaries:** The `RemoteDecisionSource` uses the raw API client (e.g., Anthropic SDK's `AsyncAnthropic` client) directly. It does NOT route through the `Instrument` abstraction. Instruments serve musicians (executing sheets); the unconscious serves the conductor (making scheduling decisions). These are distinct abstraction levels. The remote source wraps an API client, not an instrument profile. This resolves Run 2 review I-2 (instrument abstraction contradiction).

### Graceful Degradation State Machine

```
                    [Startup: auto-detect hardware]
                              |
                    +---------+-----------+
                    |         |           |
                 GPU/ASi   CPU>=6GB    CPU<6GB
                    |         |           |
              Qwen3-4B   Qwen3-1.7B    No model
                    |         |           |
                    +---------+-----------+
                              |
                    +---------v----------+
                    |   Tier 1: Full     |
                    | (Memory + Model)   |
                    +---------+----------+
                              | 3 consecutive model failures
                    +---------v----------+
                    |  Tier 2: Heuristic |
                    |   (Rules only)     |
                    +---------+----------+
                              | 10 consecutive errors OR config disable
                    +---------v----------+
                    | Tier 3: Mechanical |
                    | (No smart cond.)   |
                    +--------------------+

Tier 3 -> Tier 1: Config reload with smart_conductor: true (explicit human action)
Tier 2 -> Tier 1: Model loads successfully + memory retrieval succeeds
          + burn-in threshold met (>=100 successful heuristic evals)
Tier 3 never auto-re-enables.
```

**Burn-in threshold.** Before promoting from Tier 2 to Tier 1, the system requires at least 100 successful heuristic evaluations for the current venue. This ensures sufficient telemetry exists in venue memory before the model starts making active decisions. The burn-in counter is persisted alongside the tier state and resets if the venue changes. Adopted from Run 2 review suggestion S-2.

**Developmental stage mapping.** These tiers are not just failure modes — they are a growth trajectory. A fresh Marianne installation starts at Mechanical (no telemetry, no model). As execution telemetry accumulates, heuristic rules activate (Integration stage). As decisions accumulate, the burn-in threshold is met, and a model trains, Full activates (Generation stage). Marianne grows into her capabilities over time. `mzt status` reports the current tier as developmental progress, not just a degradation indicator.

### Action Budget Persistence

Stored in `BatonJobState` as `smart_conductor_budget_remaining: int`. Initialized from `conductor.smart_conductor_action_budget_per_job` (default 10) when a job starts. Decremented on flow-control actions (RETRY, PAUSE, SKIP, HEAL, INJECT_CONTEXT, LOOP, REORDER). Persisted in checkpoint saves. Survives conductor restarts.

When budget reaches zero:
- Smart conductor continues to observe and record decisions
- Actions default to `OBSERVE_ONLY` with reasoning noting budget exhaustion
- `ESCALATE` remains available — communication to the composer is never blocked
- `CONTINUE` remains available — acknowledging healthy events costs nothing

An in-memory counter would reset on restart, defeating the guardrail — a job that has used 9 of 10 budget units would suddenly have 10 again. Checkpoint persistence prevents this. This directly resolves Run 1 review C-4 (ephemeral action budgets).

### Cross-Concert Ordering

Periodic `ConductorTick` (default 5s) evaluates all active jobs and produces `REORDER` decisions with priority adjustments. This runs on a timer, not per-event — per-event evaluation would bottleneck during high-throughput fan-out (10+ events/second).

**Heuristic implementation:**
- Jobs whose output unblocks other jobs → priority boost
- Jobs stuck in retry loops → priority decrease (don't starve healthy jobs)
- Jobs approaching cost limits → deprioritize (gated on cost tracking reliability — see HeuristicDecisionSource cost prerequisite note; until then, high-retry-rate jobs are deprioritized instead)

**Local model enhancement:** Once trained, the model learns subtler patterns from historical decision data: "this concert's early stages produce context that makes the next concert's sheets dramatically more successful."

**Priority adjustments are soft.** They modify scheduling weights, not hard-preempt running work. Jobs already dispatched to instruments continue. The reorder affects which job gets the next dispatch slot.

Cross-concert ordering was explicitly excluded from Run 1's design ("phase-2 capability"). Run 1's review (C-2) correctly identified this as an abandonment of S6's "real magic." Run 2's periodic ConductorTick approach — evaluating on a timer rather than per-event — resolves the performance concern that motivated the deferral while delivering the capability S6 requires.

### LOOP Action Semantics

When the smart conductor loops a struggling range, it produces a flow-control event the baton interprets as an ad-hoc loop. Distinct from score-declared loops (S2):

- Score-declared loops have explicit `until` conditions, `max_iterations`, and `index` variables
- Smart conductor loops are ad-hoc: "these sheets are struggling, try them again with injected context"
- Authority check: the smart conductor will NOT loop ranges that already have score-declared loops (YAML wins)
- Ad-hoc loops have a hardcoded `max_iterations=3` safety cap (separate from the action budget)

This distinction is important: score-declared loops are the composer's intent (S2). Smart conductor loops are Marianne's judgment filling gaps where the composer was silent. The authority hierarchy is preserved — YAML-declared loops are never overridden by ad-hoc loops.

### Shadow Mode

The primary quality assurance mechanism for the local model. Configured via `conductor.unconscious.shadow_mode: true`.

- Both heuristic and model evaluate each event in parallel
- Only heuristic decisions are applied (`applied=true`)
- Model decisions recorded with `decision_source="shadow_local_model"` and `applied=false`
- Shadow decisions consume zero action budget (they're never applied)
- Shadow decisions receive `outcome="not_applied"` — they are never updated via the outcome correlation mechanism

Shadow mode enables:
- **Accuracy comparison** — Did the model agree with heuristics? When they disagreed, which was right (per the heuristic's observed outcome)?
- **Latency profiling** — Real production load, not synthetic benchmarks
- **Confidence calibration** — Does the model's 0.8 confidence actually mean 80% accuracy?
- **Regression detection** — After fine-tuning, compare fine-tuned model against base in shadow mode

### Training Data Pipeline

**Phase 1 — Data Collection (ships with v1, no training):**

Every `ConductorDecision` is written to the memory store's shared `decisions` table. The outcome field is filled asynchronously via decision-ID correlation (see Outcome Tracking section). Shadow mode records both heuristic and model decisions, but only heuristic decisions receive outcome updates. This phase requires zero training infrastructure — just data accumulation.

Training data extraction is always rule-based — SQL queries on the `decisions` table with quality filters. It never depends on LLM availability.

**Phase 2 — LoRA Fine-Tuning (after 500+ labeled decisions):**

1. **Extract** decisions with `outcome IN ('success', 'improved')`, `confidence >= 0.7`, and `applied = true` from the decisions table. Shadow decisions (`applied=false`) are explicitly excluded.
2. **Filter** with causal attribution — verify the subsequent attempt actually had different results (the decision changed outcomes, not just noise)
3. **Format** as instruction-tuning examples: `{"instruction": "...", "input": "<ConductorContext>", "output": "<decision JSON>"}`
4. **Train** via Unsloth LoRA (rank 16, alpha 32, 3 epochs, learning rate 2e-4) — approximately 30 minutes on a consumer GPU
5. **Merge** adapter weights into the base model
6. **Quantize** to Q4_K_M GGUF using llama.cpp's `llama-quantize`
7. **Validate** against held-out test set (20% of decisions) — deploy only if >=10% accuracy improvement over heuristic baseline
8. **Deploy** by replacing the GGUF file in `~/.marianne/models/` and restarting the conductor

**Quality gates:**
- First 100 training examples manually reviewed before the first fine-tuning run
- A/B testing via shadow mode before full deployment
- Held-out test set for quantitative comparison
- Rollback path: delete fine-tuned GGUF, conductor falls back to base model or heuristics

**Dual-output consolidation insight (from research dossier):** The dreaming cycle that compresses memory is also the natural pipeline for extracting training data. The `TelemetryIngester` and training-data extractor share a dual purpose. Consolidation quality directly affects model quality — a poorly tuned dreaming cycle produces both bad memory compression AND bad training data. This connection means that improving the memory system's consolidation pipeline has a direct positive effect on the unconscious model's training quality, creating a virtuous cycle rather than two independent optimization problems.

### Model Selection Rationale

**Primary: Qwen3-4B Q4_K_M (2.5GB GGUF, Apache 2.0)**
- Strong quality-to-size ratio for classification tasks (Alibaba reports competitive benchmarks against much larger models, though these claims should be treated as marketing-grade — the 10% accuracy threshold over heuristics is the real quality gate)
- Hybrid thinking mode (toggle chain-of-thought vs. direct response)
- Non-thinking mode produces 5-10 token outputs in sub-200ms on GPU
- First-party GGUF releases on HuggingFace, massive community adoption (9.74M+ downloads)
- Apache 2.0 license with explicit patent grant

**CPU Fallback: Qwen3-1.7B Q4_K_M (1.4GB GGUF, Apache 2.0)**
- Same architecture family — prompt design transfers directly
- 1.4GB Q4_K_M loads in ~2GB RAM total
- Classification-only decisions in 150-600ms on 4-core CPU
- Adequate for the core classification task (action selection + confidence)

**Runtime: llama-cpp-python**
- Only runtime meeting all constraints: in-process, GBNF support, cross-platform, active maintenance
- Pre-built wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), Windows (x86_64)
- Metal GPU acceleration on Apple Silicon, CUDA on NVIDIA, Vulkan fallback
- No external daemon, no HTTP latency, no socket cleanup on crash

**Rejected alternatives:**
- Ollama: External daemon violates NFR-03
- vLLM: Heavyweight, Linux-only (CUDA), requires HTTP server
- MLX: macOS-only
- Phi-4-mini: CPU latency 2-4s exceeds budget; MIT (no patent grant); no 2026 updates

### Conductor Config

```yaml
conductor:
  role: ai
  smart_conductor: true
  smart_conductor_action_budget_per_job: 10
  unconscious:
    enabled: true
    model_repo: "Qwen/Qwen3-4B-GGUF"
    model_file: "qwen3-4b-q4_k_m.gguf"
    cache_dir: "~/.marianne/models"
    max_inference_ms: 500
    fallback: "heuristic"
    auto_detect_hardware: true
    shadow_mode: false
    cross_concert_interval_s: 5.0
    burn_in_threshold: 100
```

All fields have defaults. `smart_conductor: false` disables the entire subsystem (Tier 3 mechanical). `unconscious.enabled: false` with `smart_conductor: true` gives Tier 2 (heuristic-only smart conductor).

### Package Structure

```
src/marianne/
+-- core/
|   +-- conductor_decision.py    # ConductorDecision, DecisionAction, ConductorContext, EventStateSnapshot
+-- conductor/
|   +-- smart.py                 # SmartConductor (EventBus subscriber + evaluator)
|   +-- sources/
|   |   +-- protocol.py          # DecisionSource Protocol
|   |   +-- heuristic.py         # HeuristicDecisionSource
|   |   +-- local_model.py       # LocalModelDecisionSource
|   |   +-- remote.py            # RemoteDecisionSource (future)
|   +-- hardware.py              # Hardware auto-detection (NVIDIA, Apple Silicon, AMD ROCm)
|   +-- grammar.py               # GBNF grammar generation
|   +-- training.py              # Training data extraction
|   +-- outcome.py               # Outcome tracking / decision-ID correlation
```

The `conductor/` package is flat (two levels maximum per P-007). `sources/` is a subpackage because the three implementations are logically grouped.

---

## Open Questions

1. **Decision quality benchmark.** How to evaluate whether the model beats heuristics without labeled ground truth? Shadow mode provides A/B comparison data, but "correct" conductor decisions need defined criteria for common failure scenarios. Options: positive-outcome decisions as proxy, manual labeling of sample, A/B testing via shadow mode. Shadow mode provides (c) automatically; (a) and (b) are complementary.

2. **Fine-tuning data volume threshold.** The 500-example threshold needs empirical validation on a 4B model. LoRA literature generally reports results for 7B+ models. Whether 500 examples produces meaningful improvement on Qwen3-4B is plausible but unverified. Quality gates ensure the system doesn't deploy a worse model.

3. **RLF judgment layer relationship.** If RLF provides its own judgment queries (TDF-based autonomy scoring), does the unconscious become redundant? Resolution: they compose. The unconscious is tactical (per-event, sub-second, local). RLF judgment is strategic (autonomy levels, cross-system). Different time horizons, different scopes.

4. **Model update cadence.** Replace GGUF + restart conductor is the simplest deployment path. Shadow mode provides A/B testing before deployment. More sophisticated hot-swap (load new model, switch atomically) is possible but adds complexity without clear benefit — conductor restarts are infrequent.

5. **Concierge memory topology.** If the concierge runs on a different host (cloud server proxying Discord), it cannot access the same SQLite files. The smart conductor's decisions are relevant to concierge status queries. Resolution needed before concierge implementation — may require a read-only replication or API layer.

---

## Integration

### Integration with EventBus

The smart conductor subscribes to relevant events via the existing EventBus. The integration follows the decoupled architecture mandated by S6:

```
BatonCore.handle_event(e) --publishes--> EventBus(e, snapshot) --delivers--> SmartConductor._on_event(e, snapshot)
                                                                                      |
                                                                                evaluate(e)
                                                                                      |
                                                                                ConductorDecision
                                                                                      |
        baton.push_event(flow_event, decision_id) <----------------------------------+
```

Key properties:
- **Asynchronous observation.** The smart conductor receives events via the EventBus after the baton has processed them. It never blocks `handle_event()` or `dispatch_ready()`.
- **Snapshot-based context.** Events carry immutable `EventStateSnapshot` objects. The smart conductor reads these snapshots — not mutable in-memory baton state — to build `ConductorContext`. This eliminates stale-state bugs during fan-out where multiple events fire concurrently.
- **Inbox-based action.** When the smart conductor decides to intervene, it pushes a flow-control event into the baton's existing inbox, tagged with the `decision_id` for outcome tracking. The baton processes this in its next cycle.
- **No new coupling.** The EventBus and baton inbox already exist. The smart conductor is another subscriber — pluggable, removable, replaceable.
- **Crash isolation.** If the smart conductor crashes, the baton continues with defaults (Tier 3 mechanical).

No changes to EventBus architecture are needed. The `state_snapshot` parameter is an additive change to the EventBus `publish()` method signature.

### Integration with BatonJobState

The smart conductor's action budget is persisted as a field on `BatonJobState`:

```python
smart_conductor_budget_remaining: int = Field(
    default=10,
    description="Remaining flow-control intervention budget for this job",
)
```

Included in checkpoint saves. Survives conductor restarts. The default matches `action_budget_per_job` from conductor config.

### Integration with Memory System

The smart conductor calls `MemoryStore.retrieve_relevant()` to build `ConductorContext.memory_snapshot`. Uses the "fast" latency tier (<50ms) — entity-graph lookup, not embedding similarity search. After every decision, `MemoryStore.record_decision()` persists the `ConductorDecision` to the shared `decisions` table. **Decision persistence is fire-and-forget** — dispatched via `asyncio.create_task(asyncio.to_thread(...))` to prevent SQLite I/O from blocking the EventBus dispatch loop.

The entity graph is the unconscious's primary memory access pattern (emergent insight from research dossier). Structured baton events have known entity types (instruments, error codes, sheet numbers) that map directly to entity-graph queries — inherently fast (<1ms for cached neighbors).

### Integration with Learning Store

Reads relevant patterns from `GlobalLearningStore.get_patterns()` to populate `ConductorContext.learning_patterns`. This is the existing learning store query surface — no changes to learning store schema or API. The learning store is currently at schema version 15 with 13 tables across 16 Python files (~7,279 lines).

### Integration with Smart Conductor Guardrails

All S6 guardrails are implemented:
- **Action budget:** Persisted, checkpoint-safe, escalation-free (see Action Budget Persistence above)
- **Authority hierarchy:** Checked on fast context before any I/O (see FR-06)
- **Decision logging:** Every evaluation produces a `ConductorDecision` record with `applied` field distinguishing active from shadow decisions (see FR-02)
- **Graceful degradation:** Three-tier state machine with explicit recovery paths and burn-in threshold for Tier 2 → Tier 1 promotion (see FR-05)

### Integration with Conductor Config

Smart conductor configuration lives in the conductor/daemon config (global scope), not in score YAML (per-job scope). This aligns with S6: "conductor config only — global scope, entire system."

### Integration with Compose

During `mzt compose`, Marianne conducts the interview and design gate. The smart conductor's accumulated venue knowledge — patterns learned from past executions, instrument reliability profiles, common failure modes — informs the compose conversation. Marianne can advise on score structure based on what she has observed working well in this venue. The compose system reads from the same memory store that the smart conductor writes to; venue knowledge flows naturally from execution experience into composition advice.

### Integration with Maestro

In `mzt maestro` sessions, ConductorDecision records and the current degradation tier are available for Marianne to discuss. When a user asks "how are my jobs?", Marianne can report not just status but her assessment — what decisions she made, why, and what happened. The maestro session reads decision history from the shared `decisions` table.

### Integration with Concierge

Concierge (Discord/Telegram) queries surface the same decision and status data as maestro. The open question is topology: if the concierge runs on a separate host, it cannot access the same SQLite database directly. Resolution is deferred to the concierge implementation spec, but the interface contract is clear: concierge needs read access to `ConductorDecision` records and current degradation tier.

### Integration with `mzt diagnose`

ConductorDecision records surface in diagnostic output. `mzt diagnose <job>` shows:
- What events triggered smart conductor evaluation
- What was decided and why
- What happened after each decision (outcome field, tracked via decision-ID correlation)
- Whether any decisions were overridden by YAML-declared behavior
- Current degradation tier, burn-in progress, and action budget state

### Integration with Fermata/Escalation

ESCALATE decisions use the existing fermata system. Marianne's reasoning (from the `ConductorDecision.reasoning` field) is included in the escalation context, giving the composer the smart conductor's analysis alongside the raw event data.

### Integration with Venue Context

The smart conductor does not receive venue-specific seeded knowledge. Venue understanding accumulates through the memory system as Marianne works with each project. The entity graph stores venue-specific patterns (instrument preferences, common failure modes, score conventions) that the smart conductor retrieves via the fast memory tier.

### Integration with Musicians

The unconscious does not interact with musicians directly. Musicians execute sheets; the smart conductor observes the results and may adjust future sheet execution (retry, skip, inject context). The abstraction boundary is clean: musicians see sheets, instruments, and prompts. The smart conductor sees events, decisions, and the baton's inbox. They never communicate directly — the baton mediates all interaction.

### Not an Instrument

The unconscious is conductor-internal. It is NOT registered in `InstrumentRegistry`. Instruments serve musicians; the unconscious serves the conductor. Registering it as an instrument would create confusion in `mzt instruments list` and conflate two different abstraction levels. The future `RemoteDecisionSource` similarly uses a raw API client — not the Instrument abstraction — to maintain this separation.

---

## Risks and Failure Modes

### Risk 1: Local Model Quality Below Heuristics
**Likelihood:** Medium. **Impact:** RAM cost for no improvement. **Mitigation:** Shadow mode comparison. >=10% accuracy threshold over heuristics required for activation. Heuristic baseline as permanent quality floor. The system is fully functional without the local model.

### Risk 2: CPU-Only Latency Exceeds Budget
**Likelihood:** High on CPU-only laptops. **Impact:** Continuous fallback to heuristics on CPU hardware. **Mitigation:** Auto-detect hardware at startup. CPU → Qwen3-1.7B in classification-only mode (sub-600ms) or heuristics only. The 500ms budget may need to be relaxed to 750ms for CPU-only deployments — this is a configuration knob, not an architectural change.

### Risk 3: Model Corrupts Baton State
**Likelihood:** Low. **Impact:** Incorrect flow-control produces wrong job state. **Mitigation:** Multiple layers of protection: (1) EventBus decoupling — smart conductor never directly modifies baton state, (2) baton validates all incoming events and drops malformed ones, (3) action budget limits total interventions, (4) authority hierarchy prevents overriding YAML-declared behavior.

### Risk 4: Action Budget Exhaustion Blocks Escalation
**Likelihood:** Zero (by design). **Impact:** N/A. **Mitigation:** ESCALATE is explicitly budget-free. This is a firm design decision, not an optimization — blocking communication when budget is exhausted defeats the guardrail's purpose.

### Risk 5: Training Data Quality
**Likelihood:** Medium. **Impact:** Model replicates suboptimal decisions. **Mitigation:** Multi-factor quality filter (positive outcome + high confidence + causal attribution + `applied=true`). Shadow decisions explicitly excluded. Manual review of first 100 examples. The 10% improvement threshold ensures a bad model never activates.

### Risk 6: GBNF Grammar Produces Partial JSON
**Likelihood:** Very low (grammar constrains at sampler level). **Impact:** Parse failure on model output. **Mitigation:** try/except with heuristic fallback. Dynamic grammar stays synchronized with enum. This should be extremely rare since GBNF constraints are enforced at the token sampling level, not post-hoc.

### Risk 7: Model Download Failure
**Likelihood:** Low-medium (network availability dependent). **Impact:** No local model on first startup. **Mitigation:** Graceful heuristic fallback. Retry on next startup. `mzt model import` for offline provisioning. SHA256 verification prevents corrupted downloads. Resume-on-interrupt prevents re-downloading large files.

### Risk 8: Cross-Concert Ordering Misjudgment
**Likelihood:** Medium. **Impact:** Suboptimal ordering (not catastrophic). **Mitigation:** Soft priority adjustments only (no preemption of running work). Action budget applies to REORDER decisions. Shadow mode enables comparison against FIFO baseline.

### Risk 9: In-Process Memory Pressure
**Likelihood:** Medium on 8GB machines. **Impact:** OOM kills of the conductor process. **Mitigation:** Hardware auto-detection avoids loading models that don't fit. Qwen3-4B requires ~5GB total — tight on 8GB but viable. On machines with <6GB available, the system runs heuristics only (~1.5GB). The graceful degradation path handles this automatically.

### Risk 10: Feedback Loop Amplification
**Likelihood:** Low but high impact. **Impact:** Poor consolidation → worse model context → worse decisions → worse training data → worse consolidation. **Mitigation:** Consolidation quality monitored independently of model accuracy. Heuristic baseline provides a quality floor — the local model must beat heuristics by >=10% to activate. If the model's accuracy drops below the heuristic baseline at any point, automatic downgrade to Tier 2 (heuristic).

### Risk 11: EventBus Ordering Under Fan-Out
**Likelihood:** Medium during heavy fan-out. **Impact:** Smart conductor evaluates stale state if events arrive out of order. **Mitigation:** Events carry immutable `EventStateSnapshot` objects captured at emission time. The smart conductor reads these snapshots rather than querying mutable baton state. Even if events arrive out of order, each evaluation is consistent with its triggering event's state. See Event State Snapshots section.

---

## References

### Internal References

- **S6 Specification (Smart Conductor, Guardrails, Memory Interplay, The Unconscious):** `docs/specs/2026-04-17-baton-primitives-and-marianne-mozart-design.md` — parent spec defining the smart conductor's role, action budget, decision authority hierarchy, memory interplay contract, and graceful degradation requirements.
- **Research Dossier:** `docs/research/2026-04-18-marianne-memory-and-unconscious-research-dossier.md` — cross-stream synthesis of four parallel research streams (CAM, memory systems, local models, integration surface). Key emergent insights: entity graph as fast path, dual-output consolidation, developmental tiers, feedback loop amplification.
- **Prior-Art Brief:** `workspaces/marianne-memory-and-unconscious-research-workspace/01-prior-art-brief.md` — inventory of existing Marianne infrastructure relevant to memory and unconscious.
- **Local Models Research:** `workspaces/marianne-memory-and-unconscious-research-workspace/02-research-local-models.md` — model comparison, runtime selection, GBNF grammar, hardware detection, fine-tuning pipeline.
- **Integration Research:** `workspaces/marianne-memory-and-unconscious-research-workspace/02-research-integration.md` — six integration surfaces, two-pass context assembly, action budget semantics, EventBus decoupling.
- **Memory System Spec (companion):** `docs/specs/2026-04-19-marianne-memory-system-design.md` — the companion spec defining the memory substrate this spec's smart conductor reads from.

### Codebase References

- **Baton Core:** `src/marianne/daemon/baton/core.py` — BatonCore event processing loop, dispatch cycle, inbox mechanism
- **Baton Events:** `src/marianne/daemon/baton/events.py` — event types the smart conductor subscribes to
- **Baton State:** `src/marianne/daemon/baton/state.py` — BatonJobState (gains `smart_conductor_budget_remaining`)
- **EventBus:** `src/marianne/daemon/event_bus.py` — pub/sub infrastructure (gains `state_snapshot` parameter on publish)
- **Fermata System:** Distributed across `src/marianne/core/checkpoint.py` (SheetStatus.FERMATA), `src/marianne/daemon/baton/events.py` (EscalationNeeded/Resolved/Timeout), `src/marianne/daemon/baton/core.py` (fermata handling)
- **Learning Store:** `src/marianne/learning/store/` — existing telemetry substrate, schema version 15, 13 tables, 16 Python files, ~7,279 lines
- **Self-Healing:** `src/marianne/healing/` — infrastructure invoked by HEAL actions

### External References

- Qwen3 model family: HuggingFace `Qwen/Qwen3-4B-GGUF` and `Qwen/Qwen3-1.7B`
- Qwen3 technical report: arXiv 2505.09388
- llama.cpp GBNF grammar specification: `ggml-org/llama.cpp` grammars documentation
- llama-cpp-python: `abetlen/llama-cpp-python` on GitHub
- Unsloth fine-tuning framework: unsloth.ai (LoRA fine-tuning for Qwen3)
- HuggingFace Hub: model distribution and caching via `huggingface_hub` Python package

---

*Merged specification composed: 2026-04-19*
*Synthesized from two independent design runs with adversarial review*
*Word count: ~6200*
