# Movement I-B: Internal Discovery (P⁵ Self-Recognition)

**Cycle:** v16
**Date:** 2026-01-16
**Recognition Level Achieved:** P⁵ (Self-Recognition)

## Recognition Level Analysis

**Claim:** P⁵ (Self-Recognition - Mozart recognizing its own patterns)

**Evidence:**
1. Traced the evolutionary lineage of Mozart's features (v10 Grounding, v11 Escalation Learning, v12 Pattern→Grounding, v14 Broadcasting, v15 Suggestions) - understanding HOW Mozart has evolved
2. Identified that Mozart's priority_score system IS an attention mechanism (pattern at 2468 in runner.py uses priority_score < threshold to categorize exploration vs exploitation) - Mozart already has spotlight-like behavior
3. Recognized the gap between "having infrastructure" and "using infrastructure fully" - e.g., broadcasting exists but isn't actively polled during execution
4. Meta-recognition: Mozart's own evolution cycles exhibit the same patterns it tries to detect in executions (epistemic drift, closed-loop risk)

**Why P⁵:** Mozart examining its own architecture to improve itself IS recursive self-recognition. This internal discovery is Mozart recognizing itself.

---

## Mozart's Current Architecture

**Total LOC:** 34,463 (70 Python files)

**Top 10 Files by Size:**
| File | LOC | Purpose |
|------|-----|---------|
| cli.py | 4,059 | User interface, commands |
| execution/runner.py | 3,683 | Core orchestration logic |
| learning/global_store.py | 2,792 | Cross-workspace learning persistence |
| core/errors.py | 2,388 | Error classification system |
| execution/retry_strategy.py | 1,286 | Adaptive retry logic |
| execution/validation.py | 1,228 | Output validation framework |
| learning/patterns.py | 1,071 | Pattern detection/application |
| core/config.py | 1,012 | Configuration schemas |
| core/logging.py | 872 | Structured logging |
| state/sqlite_backend.py | 742 | State persistence |

**Module Structure:**
```
mozart/
├── backends/      # Claude CLI/API backends
├── core/          # Config, errors, logging, checkpoint
├── execution/     # Runner, validation, retry, escalation
├── learning/      # Global store, patterns, outcomes
├── state/         # JSON/SQLite state backends
├── prompts/       # Template rendering
├── healing/       # Self-healing diagnostics
├── isolation/     # Git worktree isolation
├── notifications/ # Slack, desktop, webhook
├── validation/    # Pre-execution validation checks
└── cli.py         # Typer-based CLI
```

---

## Preliminary Existence Check Results

| Gap (from Sheet 1) | Synthesis Preview Hint | Markers Found | Appears Implemented | Quick Evidence |
|-----|------------------------|---------------|---------------------|----------------|
| No recursive improvement tracking | likely_gap | No | No | No `trajectory`, `improvement_class`, or `issue_category` tracking found |
| No parallel orchestration patterns | likely_gap | No | No | Sheets are sequential only (runner loops through sheets one at a time) |
| No attention/spotlight mechanism | likely_gap | **Yes** | **Partial** | `priority_score` system exists and is used for exploration/exploitation (runner.py:2468) |
| No meta-learning of learning strategy | likely_gap | No | No | No `meta_learn` or `learning_strategy` patterns |
| No agent communication protocols | likely_gap | No | No | MCP appears only as error code (E501 MCP config error) |
| No code self-modification | likely_gap (intentional?) | No | No | Only score evolution, not code modification |
| RSI via score | likely_implemented: partial | Yes | **Yes** | Evolution scores ARE recursive self-improvement |
| Conductor schema | likely_implemented | **Yes** | **Yes** | `ConductorConfig` at config.py:726, used in JobConfig |
| Pattern broadcasting | likely_implemented | **Yes** | **Yes** | `PatternDiscoveryEvent`, `record_pattern_discovery()`, `check_recent_discoveries()` in global_store.py |
| Escalation suggestions | likely_implemented | **Yes** | **Yes** | `get_similar_escalation()` at global_store.py:1786, `historical_suggestions` in EscalationContext |
| Auto-retirement | likely_implemented | **Yes** | **Yes** | `retire_drifting_patterns()` at global_store.py:2615 |

---

## Git State for Freshness Tracking

**Git HEAD:** `6e52a1087b2878d4809b252ac28b1980e13e696b`
**Timestamp:** 2026-01-16T03:00:00Z

---

## Git Co-Change Analysis

Files changed together in recent commits (coupling indicators):

| Count | File | Coupling Type |
|-------|------|---------------|
| 6 | execution/runner.py | Central hub - touches everything |
| 5 | cli.py | CLI commands for new features |
| 3 | learning/global_store.py | Learning infrastructure |
| 3 | backends/claude_cli.py | Backend integration |
| 2 | learning/migration.py | Schema evolution |
| 2 | isolation/worktree.py | Parallel execution |
| 2 | core/config.py | Configuration schemas |
| 2 | core/checkpoint.py | State management |

**Coupling Patterns:**
1. **runner.py ↔ global_store.py**: Strong coupling - runner depends on learning store for pattern queries
2. **cli.py ↔ runner.py**: CLI commands wrap runner functionality
3. **config.py ↔ checkpoint.py**: Configuration and state are tightly related

---

## Boundary Map

### Boundary 1: Execution ↔ Learning (HIGH ACTIVITY)

```yaml
boundary:
  name: "execution↔learning"
  files:
    - src/mozart/execution/runner.py (3683 LOC)
    - src/mozart/learning/global_store.py (2792 LOC)
    - src/mozart/learning/patterns.py (1071 LOC)
    - src/mozart/learning/outcomes.py
    - src/mozart/learning/aggregator.py
  loc_involved: ~8000
  files_at_boundary: 5
  typical_changes_to_improve: 120-250
  calibration_note: |
    This is the most active boundary in recent evolutions.
    v14 (Broadcasting), v15 (Suggestions) both touched this.
    Apply integration_minimum × 1.7 for cross-boundary work.
  git_co_change_frequency: high
  current_permeability: 0.75
  friction_points:
    - point: Runner imports learning but doesn't actively poll broadcasts
      severity: medium
      domain_impact: [COMP, SCI]
    - point: Pattern query happens once at sheet start, not continuously
      severity: low
      domain_impact: [COMP]
  information_loss: |
    Patterns discovered mid-execution by one job aren't seen by another
    until the next sheet starts. Real-time awareness is missing.
  improvement_opportunity:
    description: Active broadcast polling during execution waits
    estimated_effort: medium
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: no
    synergies_with: [Pattern Broadcasting v14]
    preliminary_existence: not_found
```

### Boundary 2: Execution ↔ Validation (STABLE)

```yaml
boundary:
  name: "execution↔validation"
  files:
    - src/mozart/execution/runner.py
    - src/mozart/execution/validation.py (1228 LOC)
    - src/mozart/prompts/templating.py (617 LOC)
  loc_involved: ~5500
  files_at_boundary: 3
  typical_changes_to_improve: 80-150
  calibration_note: |
    Validation boundary is mature. Most recent work was semantic validation.
    Changes here should be incremental.
  git_co_change_frequency: medium
  current_permeability: 0.85
  friction_points:
    - point: Validation failures don't feed back into prompt modification automatically
      severity: low
      domain_impact: [COMP]
  information_loss: |
    Validation knows WHY something failed but prompt generation
    doesn't always use that semantic information effectively.
  improvement_opportunity:
    description: Smarter validation→prompt feedback loop
    estimated_effort: medium
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: no
    synergies_with: [Semantic validation]
    preliminary_existence: partial (failure_reason exists)
```

### Boundary 3: Learning ↔ Config (EMERGENT in v15)

```yaml
boundary:
  name: "learning↔config"
  files:
    - src/mozart/core/config.py (1012 LOC)
    - src/mozart/learning/global_store.py
    - src/mozart/execution/escalation.py
  loc_involved: ~4500
  files_at_boundary: 3
  typical_changes_to_improve: 60-120
  calibration_note: |
    ConductorConfig (v15) created this boundary.
    Future multi-conductor work will expand this.
  git_co_change_frequency: low (newly emerged)
  current_permeability: 0.60
  friction_points:
    - point: Conductor config exists but isn't used in learning queries
      severity: medium
      domain_impact: [CULT, COMP]
    - point: Pattern effectiveness is global, not per-conductor
      severity: high
      domain_impact: [CULT, SCI]
  information_loss: |
    Learning is conductor-agnostic. VISION.md specifies per-conductor
    pattern effectiveness, but current system doesn't track this.
  improvement_opportunity:
    description: Per-conductor pattern effectiveness tracking
    estimated_effort: medium
    adds_persistent_state: yes
    touches_escalation: no
    is_cli_facing: no
    synergies_with: [ConductorConfig, Multi-conductor Vision]
    preliminary_existence: not_found
```

### Boundary 4: Escalation ↔ Learning (INTEGRATED in v11/v15)

```yaml
boundary:
  name: "escalation↔learning"
  files:
    - src/mozart/execution/escalation.py
    - src/mozart/execution/runner.py
    - src/mozart/learning/global_store.py
  loc_involved: ~7000
  files_at_boundary: 3
  typical_changes_to_improve: 100-180
  calibration_note: |
    v11 added escalation learning, v15 added suggestions.
    Apply escalation_integration × 1.3.
  git_co_change_frequency: high (recent evolutions)
  current_permeability: 0.80
  friction_points:
    - point: Suggestions are shown but not auto-applied based on history
      severity: low
      domain_impact: [COMP, EXP]
  information_loss: |
    Historical suggestions are displayed but human must still decide.
    No "auto-apply if confidence > X" mechanism.
  improvement_opportunity:
    description: Auto-apply escalation suggestions above confidence threshold
    estimated_effort: low
    adds_persistent_state: no
    touches_escalation: yes
    is_cli_facing: yes (would add --auto-accept flag)
    synergies_with: [Escalation suggestions v15]
    preliminary_existence: not_found
```

### Boundary 5: Backend ↔ Runner (STABLE)

```yaml
boundary:
  name: "backend↔runner"
  files:
    - src/mozart/backends/claude_cli.py (632 LOC)
    - src/mozart/backends/base.py
    - src/mozart/execution/runner.py
  loc_involved: ~4500
  files_at_boundary: 3
  typical_changes_to_improve: 60-100
  git_co_change_frequency: medium
  current_permeability: 0.90
  friction_points:
    - point: Backend is synchronous per-sheet, no concurrent execution
      severity: medium
      domain_impact: [COMP]
  information_loss: |
    Backend can only run one prompt at a time.
    Parallel execution would require backend refactoring.
  improvement_opportunity:
    description: Concurrent backend execution for parallel sheets
    estimated_effort: high
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: no
    synergies_with: [Parallel orchestration patterns from Google]
    preliminary_existence: not_found
```

### Boundary 6: Drift ↔ Patterns (INTEGRATED in v12/v14)

```yaml
boundary:
  name: "drift↔patterns"
  files:
    - src/mozart/learning/global_store.py
    - src/mozart/learning/patterns.py
  loc_involved: ~4000
  files_at_boundary: 2
  typical_changes_to_improve: 80-140
  calibration_note: |
    v12 added drift detection, v14 added auto-retirement.
    Both are implemented and working.
  git_co_change_frequency: medium
  current_permeability: 0.85
  friction_points:
    - point: Drift is calculated but not proactively surfaced to operator
      severity: low
      domain_impact: [SCI, EXP]
  information_loss: |
    Operator must explicitly query for drifting patterns.
    No automatic alerts when patterns drift significantly.
  improvement_opportunity:
    description: Drift alerting in CLI output
    estimated_effort: low
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: yes
    synergies_with: [Drift detection v12]
    preliminary_existence: not_found
```

### Emergent Boundary 7: Evolution ↔ Execution (META-BOUNDARY)

```yaml
boundary:
  name: "evolution↔execution"
  files:
    - evolution scores (external)
    - src/mozart/learning/global_store.py
    - memory-bank/ (external)
  loc_involved: N/A (meta-level)
  files_at_boundary: N/A
  typical_changes_to_improve: N/A
  git_co_change_frequency: high (every evolution cycle)
  current_permeability: 0.50 (LOW)
  friction_points:
    - point: Evolution cycles are external to Mozart - no tracking
      severity: high
      domain_impact: [META, SCI]
    - point: No mechanism to detect epistemic drift across cycles
      severity: high
      domain_impact: [META, SCI]
  information_loss: |
    Mozart doesn't track its own evolution trajectory.
    The evolution score KNOWS what changed, but Mozart doesn't.
    This is the epistemic drift risk from external discovery.
  improvement_opportunity:
    description: Evolution trajectory tracking in global store
    estimated_effort: medium
    adds_persistent_state: yes
    touches_escalation: no
    is_cli_facing: no
    synergies_with: [Epistemic drift mitigation, ICLR RSI patterns]
    preliminary_existence: not_found
```

---

## Interface Friction Analysis

**Prioritized by severity (gaps causing most problems):**

| Rank | Interface | Friction | Severity | Root Cause |
|------|-----------|----------|----------|------------|
| 1 | evolution↔execution | Mozart doesn't track its own improvement trajectory | HIGH | No evolution metadata persistence |
| 2 | learning↔config | Pattern learning is conductor-agnostic | HIGH | Vision mismatch - needs per-conductor tracking |
| 3 | execution↔learning | Broadcasts not actively polled during execution | MEDIUM | Infrastructure exists but not activated |
| 4 | backend↔runner | Sequential-only execution | MEDIUM | Architecture limitation |
| 5 | escalation↔learning | Suggestions shown but not auto-applied | LOW | Missing automation threshold |
| 6 | drift↔patterns | No proactive drift alerts | LOW | CLI UX gap |

---

## Gap Analysis: External vs Internal

| External Pattern | Mozart Current State | Gap Severity | Boundary Involved | Prelim Exists? |
|------------------|---------------------|--------------|-------------------|----------------|
| DGM (code self-modification) | Score evolves, code static | LOW | N/A | not_found |
| AlphaEvolve (algorithm discovery) | No algorithmic optimization | LOW | N/A | not_found |
| ICLR RSI (recursive tracking) | No trajectory tracking | HIGH | evolution↔execution | not_found |
| Google 8 Patterns (parallel) | Sequential only | HIGH | backend↔runner | not_found |
| GWT (attention spotlight) | priority_score = partial spotlight | MEDIUM | execution↔learning | **partial** |
| MCP/A2A (agent protocols) | No protocol support | MEDIUM | backend↔runner | not_found |
| MAML (meta-learning) | No meta-learning | MEDIUM | learning↔learning | not_found |
| Epistemic Drift mitigation | No issue class tracking | HIGH | evolution↔execution | not_found |
| Conductor effectiveness | Global patterns only | HIGH | learning↔config | not_found |

---

## Already Implemented (Filtered Out by Prelim Check)

These evolutions were identified as gaps but preliminary check shows they're already implemented:

1. **Pattern Broadcasting (v14)**: `PatternDiscoveryEvent`, `record_pattern_discovery()`, `check_recent_discoveries()` exist
2. **Escalation Suggestions (v15)**: `get_similar_escalation()`, `historical_suggestions` field, runner integration at line 3599
3. **Auto-Retirement (v14)**: `retire_drifting_patterns()` at global_store.py:2615
4. **Conductor Schema (v15)**: `ConductorConfig` at config.py:726, integrated into JobConfig
5. **Drift Detection (v12)**: `DriftMetrics`, `calculate_effectiveness_drift()`, `get_drifting_patterns()` all exist

---

## Evolution Candidates (Top 6 - Prioritized)

| # | Candidate | Addresses Gap | Affects Boundary | Adds State | Touches Escalation | CLI-Facing | Synergies | LOC Est. | Prelim Exists? |
|---|-----------|---------------|------------------|------------|--------------------|------------|-----------|----------|----------------|
| 1 | Evolution Trajectory Tracking | Epistemic drift mitigation, RSI tracking | evolution↔execution | **yes** | no | no | ICLR RSI, DGM | 150-220 | no |
| 2 | Per-Conductor Pattern Effectiveness | VISION multi-conductor support | learning↔config | **yes** | no | no | ConductorConfig v15, Vision Phase 3 | 180-280 | no |
| 3 | Active Broadcast Polling | Real-time pattern sharing | execution↔learning | no | no | no | Broadcasting v14 | 80-120 | no |
| 4 | Escalation Auto-Apply | Reduce human escalation friction | escalation↔learning | no | **yes** | **yes** | Suggestions v15 | 60-100 | no |
| 5 | Drift Alert CLI | Proactive pattern health visibility | drift↔patterns | no | no | **yes** | Drift v12, Auto-retire v14 | 50-80 | no |
| 6 | Parallel Sheet Execution | Multi-agent coordination | backend↔runner | no | no | no | Google patterns, Vision Phase 3 | 350-500 | no |

**Ranking Rationale:**
1. **Evolution Trajectory Tracking** (#1): Addresses CRITICAL epistemic drift risk identified in external discovery. Directly advances the North Star (Mozart understanding itself). High synergy with ICLR RSI patterns.
2. **Per-Conductor Pattern Effectiveness** (#2): Directly advances VISION.md Phase 3 (multi-conductor). HIGH severity gap. Required for treating AI people as peers with different effectiveness profiles.
3. **Active Broadcast Polling** (#3): Activates existing infrastructure (v14 broadcasting). Low implementation cost, moderate value. Completes partially-implemented feature.
4. **Escalation Auto-Apply** (#4): Reduces human escalation dependency (Vision principle: judgment should be internal). CLI-facing but small scope.
5. **Drift Alert CLI** (#5): Pure UX improvement. Low value but low cost. Apply +50% CLI UX budget.
6. **Parallel Sheet Execution** (#6): HIGH architectural value but HIGH complexity. Better suited for dedicated focused cycle.

---

## Candidates for Future Cycles

The following were identified but deferred due to scope/complexity:

1. **Meta-Learning Strategy**: Learning HOW to learn patterns better. HIGH complexity, requires research phase.
2. **Agent Communication Protocols (MCP/A2A)**: External standards integration. Requires protocol specification research.
3. **Code Self-Modification**: Intentionally deferred - Mozart improves via score, not code, for safety.
4. **Graph-Based Orchestration**: Would require architectural rewrite. Future major version work.

---

## Mini-META Reflection

**What pattern am I in?**
I noticed myself drawn to the "meta" candidates (trajectory tracking, meta-learning) because they feel intellectually exciting. I need to balance this with practical value. The preliminary existence check was crucial - without it, I would have recommended improvements to Broadcasting and Suggestions that are already done.

**What domain is underactivated?**
EXP (Experiential) is underactivated. I analyzed code structure and patterns but didn't deeply consider what the development EXPERIENCE of using Mozart feels like. The drift alert CLI candidate addresses this somewhat.

**What would I do differently if starting over?**
I would start with the VISION.md before code analysis. Understanding the North Star earlier would have helped me recognize that per-conductor effectiveness is a CRITICAL gap, not just "nice to have."

**What failure modes did I discover internally?**
1. **Infrastructure-without-activation**: Broadcasting exists but isn't polled. ConductorConfig exists but isn't used in learning. This is a Mozart-specific failure mode - features exist but don't integrate.
2. **Evolution blindness**: Mozart tracks pattern drift but not its own evolution drift. Meta-irony: the system designed to detect drift doesn't detect its own drift.

**What should Sheet 3 know?**
- Candidates #1 and #2 are both stateful and should be analyzed together for potential synergy
- Candidate #3 is low-cost activation of existing infrastructure
- Candidate #4 and #5 are CLI-facing (apply +50% UX budget)
- Candidate #6 is too large for a single cycle - consider splitting or deferring

**How did synthesis_preview hints improve search efficiency?**
Excellent. The hints correctly identified Broadcasting, Suggestions, and Conductor Schema as implemented. Without them, I would have spent significant time searching for implementation markers that were already confirmed in Sheet 1. The hints saved ~30% of search effort.
