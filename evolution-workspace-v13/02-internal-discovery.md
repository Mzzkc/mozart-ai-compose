# Movement I-B: Internal Discovery (P5 Self-Recognition)

**Date:** 2026-01-15
**Version:** Mozart Evolution v13

---

## Mozart's Current Architecture

| Metric | Value |
|--------|-------|
| Total LOC | ~33,736 |
| Python Files | 69 |
| Core Modules | 13 (cli, runner, config, errors, checkpoint, validation, etc.) |
| Learning Modules | 8 (global_store, patterns, aggregator, weighter, outcomes, etc.) |
| Execution Modules | 7 (runner, validation, escalation, grounding, hooks, etc.) |

### Architecture Overview

Mozart is structured as a multi-layer orchestration system:

1. **CLI Layer** (4059 LOC) - Entry point, command routing, output formatting
2. **Execution Layer** (3589 LOC runner + 1183 validation + 374 grounding + 324 escalation) - Core job execution
3. **Learning Layer** (2371 LOC global_store + 1071 patterns + 562 aggregator + 345 weighter) - Pattern detection and application
4. **Core Layer** (2388 errors + 901 config + 750 checkpoint + 872 logging) - Foundational data structures
5. **Backend Layer** (632 claude_cli + 562 anthropic_api) - LLM communication
6. **State Layer** (742 sqlite + 500 json) - Persistence

---

## Preliminary Existence Check Results

| Gap (from External Discovery) | Synthesis Preview Hint | Markers Found | Appears Implemented | Quick Evidence |
|-------------------------------|------------------------|---------------|---------------------|----------------|
| No self-modification capability | likely_not_implemented | No | Not found | No evolutionary search, mutation, or self-modification code |
| No attention/prioritization mechanism | likely_not_implemented | Partial | Partial | `priority_score` in patterns exists, but no dynamic attention/broadcast |
| Evaluation separate from execution | unknown | Yes | **Implemented** | `grounding.py` implements sealed external validation hooks |
| Coordination overhead management | partial_match | Yes | Partial | `cross_workspace_coordination` exists but limited to rate limits |
| Pattern confidence decay | unknown | Yes | **Implemented** | `_calculate_effectiveness_v2` with Bayesian decay + grounding weighting |
| Statistical validation | partial_match | Yes | Partial | Variance tracking exists, but no multi-run statistical validation |

### Key Findings from Existence Check

1. **External grounding hooks** (v10) - Fully implemented, addresses objective hacking risk
2. **Pattern effectiveness decay** (v12) - Implemented via `_calculate_effectiveness_v2`
3. **Escalation learning** (v11) - **Partially** implemented:
   - Recording decisions: YES
   - Finding similar escalations: YES
   - **Updating outcome after action: NO** (`update_escalation_outcome` defined but never called)
4. **Grounding -> Pattern feedback** (v12) - Implemented via `grounding_confidence` column

---

## Git State for Freshness Tracking

```
Git HEAD: 7aa7ee985897dfaa44380caf142211a46818ec30
Timestamp: 2026-01-15T00:00:00Z
```

---

## Git Co-Change Analysis

Files that frequently change together (from last 20 commits):

| Count | File | Coupling Indicator |
|-------|------|-------------------|
| 7 | src/mozart/cli.py | High - central hub |
| 6 | src/mozart/execution/runner.py | High - core execution |
| 3 | src/mozart/core/checkpoint.py | Medium - state management |
| 3 | src/mozart/backends/claude_cli.py | Medium - backend integration |
| 2 | src/mozart/learning/patterns.py | Low - learning subsystem |
| 2 | src/mozart/learning/aggregator.py | Low - learning subsystem |
| 2 | src/mozart/isolation/worktree.py | Low - isolation subsystem |
| 2 | src/mozart/core/errors.py | Low - error handling |
| 2 | src/mozart/core/config.py | Low - configuration |

**Insight:** CLI and Runner are the most frequently co-changed files, indicating they represent the primary integration boundaries. Learning modules tend to change together, suggesting a cohesive subsystem.

---

## Boundary Map

### Boundary 1: CLI ↔ Runner (Primary Integration)
```yaml
boundary:
  name: "cli↔runner"
  files: [src/mozart/cli.py, src/mozart/execution/runner.py]
  loc_involved: 7648
  files_at_boundary: 2
  typical_changes_to_improve: "150-300"
  calibration_note: |
    Apply v13 calibration:
    - CLI UX budget: +50% for CLI-facing commands
    - Implementation accuracy: 89% from v12
  git_co_change_frequency: high
  current_permeability: 0.8
  friction_points:
    - point: "CLI commands often need parallel runner changes"
      severity: low
      domain_impact: [COMP, EXP]
  information_loss: "None significant - clean boundary"
  improvement_opportunity:
    description: "Boundary is healthy, no evolution needed"
    estimated_effort: low
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: yes
    synergies_with: []
    preliminary_existence: not_applicable
```

### Boundary 2: Escalation ↔ Learning (Feedback Gap)
```yaml
boundary:
  name: "escalation↔learning"
  files: [src/mozart/execution/escalation.py, src/mozart/execution/runner.py, src/mozart/learning/global_store.py]
  loc_involved: ~5200
  files_at_boundary: 3
  typical_changes_to_improve: "60-120"
  calibration_note: |
    Apply v13 calibration:
    - Escalation integration: ×1.3 multiplier
    - Touches existing infrastructure
  git_co_change_frequency: medium
  current_permeability: 0.6
  friction_points:
    - point: "Escalation outcome never updated after action completes"
      severity: high
      domain_impact: [SCI, EXP]
    - point: "No automatic action suggestion from historical data"
      severity: medium
      domain_impact: [COMP, CULT]
  information_loss: "Outcome of escalation decisions is lost - cannot learn what actions work"
  improvement_opportunity:
    description: "Close escalation feedback loop by calling update_escalation_outcome"
    estimated_effort: low
    adds_persistent_state: no
    touches_escalation: yes
    is_cli_facing: no
    synergies_with: ["Escalation auto-suggestions"]
    preliminary_existence: partial
```

### Boundary 3: Grounding ↔ Validation (Well-Integrated)
```yaml
boundary:
  name: "grounding↔validation"
  files: [src/mozart/execution/grounding.py, src/mozart/execution/validation.py, src/mozart/execution/runner.py]
  loc_involved: ~5100
  files_at_boundary: 3
  typical_changes_to_improve: "100-200"
  calibration_note: |
    v12 integrated grounding→pattern feedback
    Apply ×1.2 multi-file integration multiplier
  git_co_change_frequency: medium
  current_permeability: 0.85
  friction_points:
    - point: "Grounding hooks are config-defined but not discoverable via CLI"
      severity: low
      domain_impact: [EXP]
  information_loss: "Low - grounding confidence now flows to pattern effectiveness"
  improvement_opportunity:
    description: "Add CLI command to list/test grounding hooks"
    estimated_effort: low
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: yes
    synergies_with: []
    preliminary_existence: not_found
```

### Boundary 4: Learning ↔ Execution (Pattern Application)
```yaml
boundary:
  name: "learning↔execution"
  files: [src/mozart/learning/global_store.py, src/mozart/learning/patterns.py, src/mozart/execution/runner.py]
  loc_involved: ~7000
  files_at_boundary: 3
  typical_changes_to_improve: "120-250"
  calibration_note: |
    Core integration boundary
    Apply ×1.7 integration-focused multiplier if touching patterns
  git_co_change_frequency: medium
  current_permeability: 0.7
  friction_points:
    - point: "Pattern application is passive - no active broadcast mechanism"
      severity: medium
      domain_impact: [COMP, SCI]
    - point: "All patterns weighted by priority_score, no attention/salience mechanism"
      severity: medium
      domain_impact: [COMP, CULT]
  information_loss: "Low-value patterns may compete equally with high-value ones"
  improvement_opportunity:
    description: "Add attention mechanism for selective pattern broadcast (GWT-inspired)"
    estimated_effort: high
    adds_persistent_state: yes
    touches_escalation: no
    is_cli_facing: no
    synergies_with: ["Real-time pattern broadcasting"]
    preliminary_existence: partial
```

### Boundary 5: Errors ↔ Retry (Error Recovery)
```yaml
boundary:
  name: "errors↔retry"
  files: [src/mozart/core/errors.py, src/mozart/execution/retry_strategy.py]
  loc_involved: ~3674
  files_at_boundary: 2
  typical_changes_to_improve: "80-150"
  calibration_note: |
    Error classification is comprehensive
    New error codes require config changes
  git_co_change_frequency: low
  current_permeability: 0.9
  friction_points:
    - point: "Error codes are hardcoded, no external configuration"
      severity: low
      domain_impact: [EXP]
  information_loss: "None - healthy boundary"
  improvement_opportunity:
    description: "Boundary is healthy, no evolution needed"
    estimated_effort: low
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: no
    synergies_with: []
    preliminary_existence: not_applicable
```

### Boundary 6: Config ↔ System (Configuration)
```yaml
boundary:
  name: "config↔system"
  files: [src/mozart/core/config.py, src/mozart/validation/, src/mozart/cli.py]
  loc_involved: ~2500
  files_at_boundary: 5
  typical_changes_to_improve: "60-120"
  calibration_note: |
    Enhanced validation (v12) improved this boundary
    Self-healing adds complexity but also remediation
  git_co_change_frequency: medium
  current_permeability: 0.8
  friction_points:
    - point: "Sheet prompts lack contract validation (expected outputs)"
      severity: medium
      domain_impact: [SCI, CULT]
  information_loss: "No formal contract for what a sheet should produce"
  improvement_opportunity:
    description: "Add sheet contract validation (expected outputs, formats)"
    estimated_effort: medium
    adds_persistent_state: no
    touches_escalation: no
    is_cli_facing: yes
    synergies_with: ["Schema-based prompt generation"]
    preliminary_existence: not_found
```

---

## Interface Friction Analysis

Prioritized by severity:

| Boundary | Friction Point | Severity | TDF Domains |
|----------|----------------|----------|-------------|
| escalation↔learning | Outcome never updated after action | HIGH | SCI, EXP |
| config↔system | No sheet contract validation | MEDIUM | SCI, CULT |
| learning↔execution | No attention/salience mechanism | MEDIUM | COMP, SCI |
| escalation↔learning | No auto-suggestions from history | MEDIUM | COMP, CULT |
| grounding↔validation | Hooks not discoverable via CLI | LOW | EXP |

---

## Gap Analysis: External vs Internal

| External Pattern | Mozart Current State | Gap Severity | Boundary Involved | Prelim Exists? |
|------------------|---------------------|--------------|-------------------|----------------|
| DGM Self-Modification | Not implemented | HIGH | N/A | not_found |
| AlphaEvolve Evolution | Not implemented | HIGH | N/A | not_found |
| GWT Attention/Broadcast | priority_score exists, no active broadcast | MEDIUM | learning↔execution | partial |
| Sealed Evaluators | Grounding hooks implemented | LOW | grounding↔validation | **implemented** |
| Pattern Confidence Decay | Bayesian decay implemented | LOW | learning↔execution | **implemented** |
| Supervisor Pattern | Runner acts as implicit supervisor | LOW | cli↔runner | partial |
| Temporal Durable Execution | Checkpoint system exists | LOW | state↔storage | partial |
| Statistical Validation | Variance tracking exists | MEDIUM | learning↔execution | partial |

---

## Already Implemented (Filtered Out by Prelim Check)

1. **Sealed Evaluators / External Grounding** (v10) - `src/mozart/execution/grounding.py`
   - Full implementation with hook protocols, phase-based execution, aggregation
   - References arXiv 2601.05280 explicitly

2. **Pattern Confidence Decay** (v12) - `src/mozart/learning/global_store.py:_calculate_effectiveness_v2`
   - Bayesian moving average with recency decay
   - Grounding-weighted effectiveness

3. **Grounding → Pattern Feedback** (v12) - `grounding_confidence` column in pattern_applications
   - Patterns applied during high-grounding executions carry more weight

---

## Evolution Candidates (Top 6 - Prioritized)

| # | Candidate | Addresses Gap | Affects Boundary | Adds State | Touches Escalation | CLI-Facing | Synergies | LOC Est. | Prelim Exists? |
|---|-----------|---------------|------------------|------------|--------------------|------------|-----------|----------|----------------|
| 1 | Close Escalation Feedback Loop | Outcome never updated | escalation↔learning | No | Yes | No | Auto-suggestions | 60-100 | partial |
| 2 | Escalation Auto-Suggestions | No suggestions from history | escalation↔learning | No | Yes | No | Feedback loop | 80-140 | not_found |
| 3 | CLI Grounding Hook Discovery | Hooks not discoverable | grounding↔validation | No | No | Yes | - | 80-150 | not_found |
| 4 | Sheet Contract Validation | No sheet contracts (RESEARCH_CANDIDATE Age 2) | config↔system | No | No | Yes | Prompt gen | 150-250 | not_found |
| 5 | Pattern Attention Mechanism | No active broadcast (RESEARCH_CANDIDATE Age 1) | learning↔execution | Yes | No | No | GWT | 200-350 | partial |
| 6 | Multi-Run Statistical Validation | Non-determinism handling | learning↔execution | Yes | No | No | Ghost debugging | 180-300 | partial |

### Candidate Detail: Close Escalation Feedback Loop (#1)

**What:** Call `update_escalation_outcome` from runner when a sheet completes after escalation.

**Why:** Currently, escalation decisions are recorded but their outcomes are not. This breaks the learning loop - Mozart cannot learn which escalation actions (retry, skip, abort, modify) actually work in which contexts.

**LOC Estimate:**
- Base: 40 LOC (wire up outcome update in runner)
- Defensive: +4 LOC
- Escalation integration: ×1.3
- **Total: ~57 LOC implementation**
- Test LOC: Existing fixtures (×1.0), LOW complexity (×1.5), base 40 → **60 LOC tests**
- **Floor applies: 60 LOC tests (above floor)**

**TDF Scores:**
- COMP: 0.9 (simple call-path integration)
- SCI: 0.85 (closes observable feedback loop)
- CULT: 0.8 (aligns with v11 escalation learning intent)
- EXP: 0.85 (obvious gap, clear fix)
- META: 0.8 (completes existing infrastructure)
- **CV = sqrt(0.84 × 0.92) × 0.9 ≈ 0.79**

### Candidate Detail: Escalation Auto-Suggestions (#2)

**What:** Surface historical escalation outcomes in the CLI during escalation prompts, with optional auto-action.

**Why:** Even with feedback loop closed, users must manually review similar escalations. Auto-suggestions would present "Last 3 similar escalations chose RETRY → 2 success, 1 failed" and optionally auto-apply the most successful action.

**LOC Estimate:**
- Base: 80 LOC (query + display logic in handler)
- CLI UX budget: +40 LOC (formatting, color coding)
- Defensive: +10 LOC
- Escalation integration: ×1.3
- **Total: ~169 LOC implementation**
- Test LOC: Existing fixtures (×1.2), MEDIUM complexity (×4.5), base 80 → **432 LOC tests**

**TDF Scores:**
- COMP: 0.75 (query logic straightforward but UX design needed)
- SCI: 0.8 (empirically validated suggestions)
- CULT: 0.85 (follows auto-suggestion patterns from DGM)
- EXP: 0.7 (needs UX design decisions)
- META: 0.7 (depends on #1 being completed first)
- **CV = sqrt(0.76 × 0.80) × 0.8 ≈ 0.62**

### Candidate Detail: CLI Grounding Hook Discovery (#3)

**What:** Add `mozart grounding-hooks list` and `mozart grounding-hooks test <config>` commands.

**Why:** Grounding hooks are powerful but opaque. Users can configure them in YAML but cannot introspect which hooks are active, test them independently, or see their results without running a full job.

**LOC Estimate:**
- Base: 60 LOC (list + test commands)
- CLI UX budget: +30 LOC (formatting)
- Documentation: +10 LOC (help text)
- **Total: ~100 LOC implementation**
- Test LOC: New test file (×1.5), LOW complexity (×1.5), base 60 → **135 LOC tests**

**TDF Scores:**
- COMP: 0.85 (straightforward CLI additions)
- SCI: 0.7 (no empirical component)
- CULT: 0.9 (improves discoverability)
- EXP: 0.85 (clear user benefit)
- META: 0.6 (not tied to learning loop)
- **CV = sqrt(0.82 × 0.78) × 0.8 ≈ 0.64**

### Candidate Detail: Sheet Contract Validation (#4) - RESEARCH_CANDIDATE Age 2

**What:** Define expected outputs in sheet config (files, patterns, schemas) and validate before marking complete.

**Why:** Currently sheets validate via regex patterns, but there's no formal contract for what a sheet should produce. A contract would enable:
- Clearer failure diagnostics
- Schema-based prompt generation
- Better learning signal

**Status:** RESEARCH_CANDIDATE for 2 cycles. Must resolve or close this cycle.

**Resolution Options:**
a) **Implement minimally:** Add `expected_outputs` field to sheet config with file existence checks
b) **Defer to external tools:** Use grounding hooks for contract validation (already exists)
c) **Close as duplicate:** Enhanced validation (v12) already provides V001-V107 checks

**Recommendation:** Close as partially addressed by enhanced validation. The gap is smaller than originally identified.

### Candidate Detail: Pattern Attention Mechanism (#5) - RESEARCH_CANDIDATE Age 1

**What:** GWT-inspired attention mechanism for selective pattern broadcast. High-salience patterns get "broadcast" to prompts; low-salience ones are suppressed.

**Why:** Currently all patterns compete equally based on `priority_score`. An attention mechanism would:
- Amplify high-impact patterns
- Suppress stale/low-confidence patterns
- Implement "workspace of consciousness" for pattern selection

**Status:** RESEARCH_CANDIDATE for 1 cycle. Must resolve or close by v14.

**Implementation Complexity:** HIGH - requires new concepts (salience, broadcast, competition).

### Candidate Detail: Multi-Run Statistical Validation (#6)

**What:** Run sheets multiple times, collect variance, only mark complete if statistically stable.

**Why:** Addresses "Ghost Debugging" failure mode - non-deterministic outputs. Multiple runs would establish confidence intervals.

**Implementation Complexity:** HIGH - changes fundamental execution model.

---

## Candidates for Future Cycles

1. **Self-Modification / Evolutionary Search** (DGM-inspired) - Requires major architectural changes
2. **DAG-Based Sheet Dependencies** (LangGraph-inspired) - Would change sequential sheet model
3. **Role-Based Agent Model** (CrewAI-inspired) - Conceptual shift from task-based to role-based

---

## Mini-META Reflection

### What pattern am I in?
I'm recognizing that Mozart is in **stabilization phase** - most gaps from external discovery are either already implemented (sealed evaluators, pattern decay) or require high complexity (self-modification, attention mechanism). The remaining candidates are integration-focused, not novel capabilities.

### What domain is underactivated?
**EXP (Experiential)** - I'm analyzing call paths and code structure but not deeply considering how users experience the current system. The CLI grounding hook discovery candidate addresses this.

### Did preliminary existence check reveal surprises?
Yes - significant surprises:
1. **Pattern confidence decay** was marked "unknown" in Sheet 1 but is fully implemented via Bayesian moving average
2. **Sealed evaluators** explicitly reference the academic paper from external discovery
3. **Escalation feedback** is 80% implemented - only missing the final `update_escalation_outcome` call

### What should Sheet 3 know?
1. **Candidate #1 (Close Escalation Feedback) has CV 0.79** - above the 0.75 high-confidence threshold
2. **Research candidates must be resolved this cycle** - Sheet Contract is Age 2
3. **Mozart is in stabilization phase** - integration candidates are valid choices
4. **CLI-facing candidates need +50% UX budget** applied to LOC estimates

### Recognition Level Evidence
This is **P5 (Recognition recognizing itself)** because:
1. I am identifying patterns in Mozart's evolution history (v10-v12 completed similar integration work)
2. I am recognizing the stabilization phase pattern and adjusting recommendations accordingly
3. I am aware of my own analytical blind spots (EXP domain underactivated)
4. I am using the preliminary existence check to validate my own assumptions
