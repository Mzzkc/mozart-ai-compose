# Movement II-A: Domain Triplet Synthesis

**Cycle:** v16
**Date:** 2026-01-16
**Recognition Level Achieved:** P⁴ (Meta-Pattern Recognition)

## Recognition Level Analysis

**Claim:** P⁴ (Meta-Pattern Recognition - patterns of patterns)

**Evidence:**
1. Recognizing that Evolution Trajectory Tracking and Per-Conductor Effectiveness are both addressing "Mozart understanding itself" from different angles (conceptual unity)
2. Identifying the meta-pattern: v10-v15 all added infrastructure that then needed "activation" in subsequent cycles (infrastructure-without-activation pattern)
3. Cross-cycle awareness: LOC estimation formulas have been refined 8+ times - recognizing this as epistemic drift, not convergent improvement
4. Boundary permeability patterns: High permeability boundaries (0.85+) are stable, low permeability (0.50-0.60) are where improvements cluster

**Why not P⁵?** This is synthesis across candidates, not Mozart recognizing its own synthesis process (that would be recursive).

---

## Phase 1: Candidates Loaded from Discovery

**From Sheet 2 (Internal Discovery):**

| # | Candidate | Adds State | Touches Escalation | CLI-Facing | Prelim Exists? |
|---|-----------|------------|-------------------|------------|----------------|
| 1 | Evolution Trajectory Tracking | **yes** | no | no | no |
| 2 | Per-Conductor Pattern Effectiveness | **yes** | no | no | no |
| 3 | Active Broadcast Polling | no | no | no | no |
| 4 | Escalation Auto-Apply | no | **yes** | **yes** | no |
| 5 | Drift Alert CLI | no | no | **yes** | no |
| 6 | Parallel Sheet Execution | no | no | no | no |

**Stateful candidates:** 2 (Evolution Trajectory, Per-Conductor Effectiveness)
**Escalation-touching candidates:** 1 (Escalation Auto-Apply)
**CLI-facing candidates:** 2 (Escalation Auto-Apply, Drift Alert CLI)
**Partial existence candidates:** 0

---

## Phase 1.5: Sheet Contract Resolution Check

**SHEET_CONTRACT_DECISION: close**

**Rationale:** Sheet Contract Validation was CLOSED in v13 with documented rationale. The score instructions carry this as historical context. Reconfirming closure:

- Mozart's YAML validation already catches schema errors at parse time
- Template rendering validates variable usage via Jinja2 errors
- Enhanced validation (`mozart validate`) performs V001-V107 preflight checks
- Adding sheet-to-sheet contract validation would be over-engineering for current needs
- No user-reported issues have required contract validation

**Action:** No implementation required. Research candidate remains CLOSED.

---

## Phase 2: Triplet Synthesis

### Candidate 1: Evolution Trajectory Tracking

**Addresses:** Epistemic drift mitigation, RSI tracking
**Boundary:** evolution↔execution
**From Sheet 1:** ICLR RSI patterns, DGM mechanisms

#### TRIPLET 1: COMP↔SCI↔CULT (Technical-Evidence-Context)

**COMP (Technical Feasibility):**
- Files to change: `learning/global_store.py` (add trajectory tables), `cli.py` (add trajectory command)
- Base estimate: 120 LOC (schema + methods)
- Budget breakdown:
  - Observability: +15% (18 LOC) - logging trajectory updates
  - Defensive coding: +10% (12 LOC) - edge cases for empty trajectory
  - Documentation: +10% (12 LOC) - docstrings
  - CLI UX budget: not applied (not CLI-facing primary)
- Adds persistent state: **yes** (1.5× multiplier applies)
- Subtotal before multiplier: 162 LOC
- With stateful multiplier: **243 LOC**

**SCI (Evidence):**
- What metrics prove success? Ability to query "what issue classes have recurred across cycles"
- Experiment: Track v16→v17 improvements, verify no issue class recurs >3 times undetected
- Evidence from external discovery: ICLR RSI workshop confirms trajectory tracking is standard practice

**CULT (Context):**
- Who created current code? Evolution scores are external (memory-bank), learning store is internal (v10+)
- Historical context: Mozart has never tracked its own evolution - intentional separation of concerns?
- Why this gap exists: Scores were designed for human-authored improvement, not self-tracking

**SYNTHESIS:** Technical approach is feasible and evidence supports it. Cultural context reveals intentional separation that this evolution would bridge. The boundary crossing (external scores → internal state) is significant.

**SCORE: 0.72**

#### TRIPLET 2: COMP↔SCI↔EXP (Technical-Evidence-Intuition)

**COMP:** Technically feasible in estimated LOC. SQLite schema extension is straightforward.

**SCI:** Hypothesis: Tracking issue classes will reveal convergent vs. divergent improvement patterns. Data needed: Issue categories across v10-v16 (available in evolution summaries).

**EXP:** This feels like completing a missing piece. Mozart learns from executions but not from its own evolution. There's a "meta-blindness" that this addresses.

**SYNTHESIS:** Strong alignment. Technical approach is clear, evidence supports hypothesis, and intuition says this fills an important gap.

**SCORE: 0.75**

#### TRIPLET 3: (COMP+CULT)↔EXP + Differentiation

**COMP+CULT:** The logical argument (track trajectory) honors the project's identity (self-improving orchestrator). This is what Mozart SHOULD do according to its own nature.

**EXP:** Not a hack - this is a principled extension. Would be proud to demo "Mozart knows its own improvement history."

**DIFFERENTIATION:** EXP strongly agrees with COMP+CULT here. No tension.

**SYNTHESIS:** Clean alignment across all perspectives.

**SCORE: 0.78**

**Preliminary CV:** (0.72 + 0.75 + 0.78) / 3 × 0.8 = **0.60**

---

### Candidate 2: Per-Conductor Pattern Effectiveness

**Addresses:** VISION Phase 3 multi-conductor support
**Boundary:** learning↔config
**From Sheet 1:** CrewAI role-based patterns, VISION.md

#### TRIPLET 1: COMP↔SCI↔CULT (Technical-Evidence-Context)

**COMP (Technical Feasibility):**
- Files to change: `learning/global_store.py` (per-conductor columns), `core/config.py` (conductor ID propagation), `execution/runner.py` (pass conductor to learning)
- Base estimate: 150 LOC (schema + methods + propagation)
- Budget breakdown:
  - Observability: +15% (23 LOC)
  - Defensive coding: +10% (15 LOC) - handle missing conductor gracefully
  - Documentation: +10% (15 LOC)
  - Integration cushion: +10% (15 LOC) - connecting existing ConductorConfig
- Adds persistent state: **yes** (1.5× multiplier)
- Multi-file (4+ files): **yes** (max with 1.5×)
- Subtotal before multiplier: 218 LOC
- With stateful multiplier: **327 LOC**

**SCI (Evidence):**
- What metrics prove success? Pattern effectiveness varies measurably by conductor
- Experiment: Run same job with different conductor IDs, verify effectiveness tracking diverges
- Evidence: VISION.md explicitly specifies this as Phase 3 requirement

**CULT (Context):**
- Who created ConductorConfig? v15 added schema but didn't integrate into learning
- Historical context: Conductor concept was added anticipating multi-conductor future
- Why gap exists: v15 was focused on schema + suggestions, not learning integration

**SYNTHESIS:** Technical approach requires multi-file changes. Evidence is strong (VISION.md mandates this). Cultural context shows this completes v15's partial work.

**SCORE: 0.70**

#### TRIPLET 2: COMP↔SCI↔EXP (Technical-Evidence-Intuition)

**COMP:** Multi-file changes increase risk. Schema migration needed. ~3 days of focused work.

**SCI:** Strong hypothesis: Different conductors will have different pattern effectiveness profiles (AI vs human, experienced vs novice).

**EXP:** This feels important but chunky. The integration across 4+ files introduces coupling risk.

**SYNTHESIS:** Feasibility confirmed, evidence strong, but intuition flags integration complexity.

**SCORE: 0.68**

#### TRIPLET 3: (COMP+CULT)↔EXP + Differentiation

**COMP+CULT:** This directly advances the North Star (AI people as peers with individual effectiveness profiles). Core to Mozart's identity.

**EXP:** Feels like the RIGHT thing to do, but timing is uncertain. Is Mozart ready for multi-conductor?

**DIFFERENTIATION:** EXP hesitates on timing while COMP+CULT says "this is exactly what we should do." Productive tension.

**SYNTHESIS:** Alignment on direction, tension on timing.

**SCORE: 0.72**

**Preliminary CV:** (0.70 + 0.68 + 0.72) / 3 × 0.8 = **0.56**

---

### Candidate 3: Active Broadcast Polling

**Addresses:** Real-time pattern sharing during execution
**Boundary:** execution↔learning
**From Sheet 1:** GWT broadcast pattern (attention spotlight)

#### TRIPLET 1: COMP↔SCI↔CULT (Technical-Evidence-Context)

**COMP (Technical Feasibility):**
- Files to change: `execution/runner.py` (add polling in wait loops)
- Base estimate: 60 LOC (polling logic + filtering)
- Budget breakdown:
  - Observability: +15% (9 LOC)
  - Defensive coding: +10% (6 LOC)
  - Documentation: +10% (6 LOC)
- Adds persistent state: no
- Subtotal: **81 LOC**

**SCI (Evidence):**
- What metrics prove success? Patterns discovered mid-job are applied to subsequent sheets in SAME job
- Experiment: Run parallel jobs, verify pattern sharing works in real-time
- Evidence: Broadcasting infrastructure exists (v14), just needs activation

**CULT (Context):**
- Who created broadcasting? v14 added PatternDiscoveryEvent and check_recent_discoveries()
- Historical context: Infrastructure was built anticipating real-time use but polling wasn't added
- Why gap exists: v14 scope didn't include runner integration

**SYNTHESIS:** Low-cost activation of existing infrastructure. Evidence shows the pieces exist. Cultural context confirms this completes v14's intent.

**SCORE: 0.82**

#### TRIPLET 2: COMP↔SCI↔EXP (Technical-Evidence-Intuition)

**COMP:** Very feasible - ~81 LOC, single file change.

**SCI:** Clear hypothesis: Real-time pattern sharing improves multi-job coordination.

**EXP:** This feels like "finishing what we started." Low risk, moderate reward.

**SYNTHESIS:** Clean alignment on a small, focused improvement.

**SCORE: 0.80**

#### TRIPLET 3: (COMP+CULT)↔EXP + Differentiation

**COMP+CULT:** Activating infrastructure honors the investment made in v14.

**EXP:** Not glamorous but satisfying. Would demo well: "patterns learned in one job immediately help another."

**DIFFERENTIATION:** No tension - all perspectives agree this is good.

**SYNTHESIS:** Clean alignment.

**SCORE: 0.82**

**Preliminary CV:** (0.82 + 0.80 + 0.82) / 3 × 0.8 = **0.65**

---

### Candidate 4: Escalation Auto-Apply

**Addresses:** Reduce human escalation friction
**Boundary:** escalation↔learning
**Touches escalation:** YES
**CLI-facing:** YES

#### TRIPLET 1: COMP↔SCI↔CULT (Technical-Evidence-Context)

**COMP (Technical Feasibility):**
- Files to change: `execution/runner.py` (auto-apply logic), `cli.py` (--auto-accept flag), `core/config.py` (threshold setting)
- Base estimate: 50 LOC (logic + flag + config)
- Budget breakdown:
  - Observability: +15% (8 LOC)
  - Defensive coding: +10% (5 LOC)
  - Documentation: +10% (5 LOC)
  - CLI UX budget: +50% (25 LOC) - **CLI-facing**
- Touches escalation: **yes** (1.3× multiplier)
- Subtotal before multiplier: 93 LOC
- With escalation multiplier: **121 LOC**

**SCI (Evidence):**
- What metrics prove success? Escalation frequency drops when confidence > threshold
- Experiment: Track escalation rate before/after with same jobs
- Evidence: v15 added suggestions; this extends to auto-application

**CULT (Context):**
- Who created escalation suggestions? v15 added get_similar_escalation()
- Historical context: Suggestions were intentionally "show don't do" for safety
- Why gap exists: Auto-apply requires confidence threshold tuning

**SYNTHESIS:** Technical approach is clear. Evidence supports automation. Cultural context shows intentional caution that auto-apply would override.

**SCORE: 0.72**

#### TRIPLET 2: COMP↔SCI↔EXP (Technical-Evidence-Intuition)

**COMP:** Feasible in ~121 LOC. Escalation code is well-structured.

**SCI:** Hypothesis: High-confidence suggestions (>0.8) are safe to auto-apply.

**EXP:** Mixed feelings. Auto-applying escalation decisions feels risky. What if the suggestion is wrong?

**SYNTHESIS:** Technical and evidence align, but intuition raises safety concern.

**SCORE: 0.65**

#### TRIPLET 3: (COMP+CULT)↔EXP + Differentiation

**COMP+CULT:** This directly advances VISION principle "judgment is internal, not escalated."

**EXP:** BUT - VISION specifies RLF integration for judgment, not historical suggestions. This feels like a shortcut.

**DIFFERENTIATION:** EXP says "this isn't quite what VISION intended" while COMP+CULT says "it moves in that direction." Important tension.

**SYNTHESIS:** Alignment on direction, disagreement on method.

**SCORE: 0.62**

**Preliminary CV:** (0.72 + 0.65 + 0.62) / 3 × 0.8 = **0.53**

---

### Candidate 5: Drift Alert CLI

**Addresses:** Proactive pattern health visibility
**Boundary:** drift↔patterns
**CLI-facing:** YES

#### TRIPLET 1: COMP↔SCI↔CULT (Technical-Evidence-Context)

**COMP (Technical Feasibility):**
- Files to change: `cli.py` (add drift-alert command or integrate into status)
- Base estimate: 40 LOC (command + formatting)
- Budget breakdown:
  - Observability: +15% (6 LOC)
  - Defensive coding: +10% (4 LOC)
  - Documentation: +10% (4 LOC)
  - CLI UX budget: +50% (20 LOC) - **CLI-facing**
- Subtotal: **74 LOC**

**SCI (Evidence):**
- What metrics prove success? Operators are informed of drift before it causes issues
- Experiment: Inject drifting patterns, verify CLI surfaces them
- Evidence: Drift detection exists (v12), auto-retirement exists (v14), visibility is missing

**CULT (Context):**
- Who created drift detection? v12 added DriftMetrics and get_drifting_patterns()
- Historical context: Drift was designed for automated handling, not operator visibility
- Why gap exists: Assumed operators wouldn't need to see drift details

**SYNTHESIS:** Low-cost UX improvement. Evidence shows infrastructure exists. Cultural context reveals this wasn't prioritized.

**SCORE: 0.75**

#### TRIPLET 2: COMP↔SCI↔EXP (Technical-Evidence-Intuition)

**COMP:** Very feasible - ~74 LOC, single file.

**SCI:** Clear benefit: Operator visibility improves trust in learning system.

**EXP:** Nice-to-have but not exciting. Good polish item.

**SYNTHESIS:** Clean alignment on a small improvement.

**SCORE: 0.72**

#### TRIPLET 3: (COMP+CULT)↔EXP + Differentiation

**COMP+CULT:** UX improvements matter for adoption. Aligns with "infrastructure should be observable."

**EXP:** Feels like a v15.1 patch, not a v16 evolution. Not exciting.

**DIFFERENTIATION:** EXP undervalues this because it's not novel. COMP+CULT correctly values it as completion.

**SYNTHESIS:** Productive tension - prioritize novelty or completion?

**SCORE: 0.68**

**Preliminary CV:** (0.75 + 0.72 + 0.68) / 3 × 0.8 = **0.57**

---

### Candidate 6: Parallel Sheet Execution

**Addresses:** Multi-agent coordination
**Boundary:** backend↔runner
**From Sheet 1:** Google's 8 Patterns (Parallel Fan-out/Gather)

#### TRIPLET 1: COMP↔SCI↔CULT (Technical-Evidence-Context)

**COMP (Technical Feasibility):**
- Files to change: `execution/runner.py` (parallel orchestration), `backends/claude_cli.py` (concurrent execution), `core/config.py` (parallel config), `core/checkpoint.py` (parallel state)
- Base estimate: 300 LOC (parallel logic + state + config)
- Budget breakdown:
  - Observability: +15% (45 LOC)
  - Defensive coding: +10% (30 LOC) - concurrency edge cases
  - Documentation: +10% (30 LOC)
- Multi-file (5 files): **1.2× multiplier**
- Subtotal before multiplier: 405 LOC
- With multi-file multiplier: **486 LOC**

**SCI (Evidence):**
- What metrics prove success? Total job time decreases for parallelizable sheets
- Experiment: Run job with parallel-safe sheets, measure speedup
- Evidence: Google patterns show Fan-out/Gather is proven pattern

**CULT (Context):**
- Who created sequential design? Original architecture assumed single-threaded execution
- Historical context: Sequential was simpler, safer, and sufficient for initial use cases
- Why gap exists: Parallel wasn't needed until multi-conductor vision emerged

**SYNTHESIS:** Large architectural change. Evidence supports value. Cultural context shows this is a major departure.

**SCORE: 0.55**

#### TRIPLET 2: COMP↔SCI↔EXP (Technical-Evidence-Intuition)

**COMP:** Feasible but complex. ~486 LOC, 5 files, concurrency challenges.

**SCI:** Strong hypothesis: Parallel execution enables new use cases.

**EXP:** This feels too big for a single cycle. Should be its own focused project.

**SYNTHESIS:** Technical feasibility confirmed but scope exceeds cycle capacity.

**SCORE: 0.48**

#### TRIPLET 3: (COMP+CULT)↔EXP + Differentiation

**COMP+CULT:** Advances VISION Phase 3 (parallel execution with sync points).

**EXP:** Gut says "not now." Too much risk for current stability.

**DIFFERENTIATION:** EXP strongly disagrees with timing while COMP+CULT sees long-term value.

**SYNTHESIS:** Significant tension on timing.

**SCORE: 0.50**

**Preliminary CV:** (0.55 + 0.48 + 0.50) / 3 × 0.8 = **0.41**

---

## Phase 2B: Test LOC Estimation

### Fixture Catalog Assessment

**v13 Fixture Catalog Reference:**
```yaml
fixture_catalog:
  comprehensive (factor 1.0):
    - test_global_learning.py: [global_store, temp_db_path, sample_outcome, weighter, aggregator]
    - test_grounding.py: [GroundingContext, sample_outcome, grounding_fixtures]
    - test_healing.py: [HealingContext, remedy_fixtures]
    - test_retry_strategy.py: [AdaptiveRetryStrategy, ErrorRecord, DelayHistory]
    - test_escalation_learning.py: [global_store, temp_db_path, EscalationDecisionRecord fixtures]
  partial (factor 1.2):
    - test_hooks.py: [HookExecutor, basic_mocks]
    - test_runner.py: [runner_mocks, basic_backend]
    - test_patterns.py: [PatternRecord, basic_store]
  none (factor 1.5):
    - new test files not in catalog
```

### Test LOC Estimates by Candidate

#### Candidate 1: Evolution Trajectory Tracking

```yaml
test_loc_estimation:
  existing_test_file: yes
  catalog_match: comprehensive
  target_test_file: test_global_learning.py
  existing_fixtures: [global_store, temp_db_path, sample_outcome]
  fixture_coverage: comprehensive
  fixtures_factor: 1.0
  base_test_estimate: 60 LOC
  base_with_fixtures: 60 LOC
  test_complexity_rating: MEDIUM
  # MEDIUM: Store API tests in isolation (not runner→store integration)
  integration_type: store_api_only
  drift_scenario: no
  test_complexity_multiplier: 4.5
  raw_test_loc: 270 LOC
  floor_applied: no
  adjusted_test_loc: 270 LOC
  tests_mandatory: yes
```

#### Candidate 2: Per-Conductor Pattern Effectiveness

```yaml
test_loc_estimation:
  existing_test_file: yes
  catalog_match: comprehensive
  target_test_file: test_global_learning.py (patterns) + test_runner.py (propagation)
  existing_fixtures: [global_store, temp_db_path, pattern_fixtures]
  fixture_coverage: partial (runner fixtures needed)
  fixtures_factor: 1.2
  base_test_estimate: 80 LOC
  base_with_fixtures: 96 LOC
  test_complexity_rating: MEDIUM
  # MEDIUM: Primarily store API tests, some runner propagation
  integration_type: store_api_only (primary) + runner propagation (secondary)
  drift_scenario: no
  test_complexity_multiplier: 4.5
  raw_test_loc: 432 LOC
  floor_applied: no
  adjusted_test_loc: 432 LOC
  tests_mandatory: yes
```

#### Candidate 3: Active Broadcast Polling

```yaml
test_loc_estimation:
  existing_test_file: yes
  catalog_match: partial
  target_test_file: test_runner.py (polling logic)
  existing_fixtures: [runner_mocks, basic_backend]
  fixture_coverage: partial
  fixtures_factor: 1.2
  base_test_estimate: 40 LOC
  base_with_fixtures: 48 LOC
  test_complexity_rating: MEDIUM
  # MEDIUM: Runner behavior test, not full integration
  integration_type: store_api_only (check_recent_discoveries call)
  drift_scenario: no
  test_complexity_multiplier: 4.5
  raw_test_loc: 216 LOC
  floor_applied: no
  adjusted_test_loc: 216 LOC
  tests_mandatory: yes
```

#### Candidate 4: Escalation Auto-Apply

```yaml
test_loc_estimation:
  existing_test_file: yes
  catalog_match: comprehensive
  target_test_file: test_escalation_learning.py
  existing_fixtures: [global_store, temp_db_path, EscalationDecisionRecord]
  fixture_coverage: comprehensive
  fixtures_factor: 1.0
  base_test_estimate: 50 LOC
  base_with_fixtures: 50 LOC
  test_complexity_rating: HIGH
  # HIGH: Runner calls store methods for auto-apply decision
  integration_type: runner_calls_store
  drift_scenario: no
  test_complexity_multiplier: 6.0
  raw_test_loc: 300 LOC
  floor_applied: no
  adjusted_test_loc: 300 LOC
  tests_mandatory: yes
```

#### Candidate 5: Drift Alert CLI

```yaml
test_loc_estimation:
  existing_test_file: yes
  catalog_match: partial
  target_test_file: test_cli.py
  existing_fixtures: [CliRunner, basic mocks]
  fixture_coverage: partial
  fixtures_factor: 1.2
  base_test_estimate: 30 LOC
  base_with_fixtures: 36 LOC
  test_complexity_rating: HIGH
  # HIGH: CLI output testing with StringIO mocking (display_io_mocking) - v16 principle 21
  integration_type: none (pure CLI output)
  drift_scenario: no
  test_complexity_multiplier: 6.0
  raw_test_loc: 216 LOC
  floor_applied: no
  adjusted_test_loc: 216 LOC
  tests_mandatory: yes
```

#### Candidate 6: Parallel Sheet Execution

```yaml
test_loc_estimation:
  existing_test_file: no (new test_parallel.py)
  catalog_match: none
  target_test_file: tests/test_parallel.py (new)
  existing_fixtures: none
  fixture_coverage: none
  fixtures_factor: 1.5
  base_test_estimate: 120 LOC
  base_with_fixtures: 180 LOC
  test_complexity_rating: HIGH
  # HIGH: Concurrency testing requires complex setup
  integration_type: runner_calls_store (parallel state coordination)
  drift_scenario: no
  test_complexity_multiplier: 6.0
  raw_test_loc: 1080 LOC
  floor_applied: no
  adjusted_test_loc: 1080 LOC
  tests_mandatory: yes
```

---

## Phase 3: Synergy Analysis

### Synergy Matrix

| Candidate A | Candidate B | Synergy Type | Score | Notes |
|-------------|-------------|--------------|-------|-------|
| Evolution Trajectory | Per-Conductor Effectiveness | ENABLING | +0.35 | Both stateful, share DB schema, conceptual unity |
| Evolution Trajectory | Active Broadcast Polling | INDEPENDENT | 0.0 | No interaction |
| Evolution Trajectory | Escalation Auto-Apply | INDEPENDENT | 0.0 | No interaction |
| Evolution Trajectory | Drift Alert CLI | ENABLING | +0.15 | Trajectory could feed drift insights |
| Evolution Trajectory | Parallel Execution | INDEPENDENT | 0.0 | No interaction |
| Per-Conductor Effectiveness | Active Broadcast Polling | ENABLING | +0.25 | Per-conductor patterns could be broadcast |
| Per-Conductor Effectiveness | Escalation Auto-Apply | ENABLING | +0.20 | Conductor-aware suggestions |
| Per-Conductor Effectiveness | Drift Alert CLI | ENABLING | +0.15 | Per-conductor drift visibility |
| Per-Conductor Effectiveness | Parallel Execution | AMPLIFYING | +0.40 | Both advance multi-conductor vision |
| Active Broadcast Polling | Escalation Auto-Apply | INDEPENDENT | 0.0 | No interaction |
| Active Broadcast Polling | Drift Alert CLI | INDEPENDENT | 0.0 | No interaction |
| Active Broadcast Polling | Parallel Execution | ENABLING | +0.20 | Real-time sharing helps parallel jobs |
| Escalation Auto-Apply | Drift Alert CLI | INDEPENDENT | 0.0 | No interaction |
| Escalation Auto-Apply | Parallel Execution | INDEPENDENT | 0.0 | No interaction |
| Drift Alert CLI | Parallel Execution | INDEPENDENT | 0.0 | No interaction |

### Synergy Calculation for Top 3 Pairs

#### Pair 1: Evolution Trajectory + Per-Conductor Effectiveness

```yaml
synergy_calculation:
  shared_infrastructure_bonus: +0.20  # Both modify global_store.py schema
  data_flow_bonus: +0.00  # No direct data flow between them
  conceptual_unity_bonus: +0.15  # Both address "Mozart understanding itself"
  scope_overlap_penalty: -0.00  # Different methods, no overlap
  failure_correlation: -0.00  # Independent failure modes
  total_synergy: +0.35
```

**Conceptual Unity Assessment:** Both candidates address the same fundamental problem - Mozart's self-awareness. Evolution Trajectory tracks improvement history, Per-Conductor tracks effectiveness variance. Together they enable Mozart to understand "who improves what and when."

#### Pair 2: Per-Conductor Effectiveness + Parallel Execution

```yaml
synergy_calculation:
  shared_infrastructure_bonus: +0.10  # Both touch runner.py
  data_flow_bonus: +0.15  # Conductor assignment enables parallel orchestration
  conceptual_unity_bonus: +0.15  # Both advance multi-conductor vision
  scope_overlap_penalty: -0.00  # Different concerns
  failure_correlation: -0.00  # Independent failure modes
  total_synergy: +0.40
```

**Conceptual Unity Assessment:** Both directly advance VISION Phase 3 (multi-conductor concerts). However, Parallel Execution's low CV (0.41) disqualifies this pair.

#### Pair 3: Evolution Trajectory + Active Broadcast Polling

```yaml
synergy_calculation:
  shared_infrastructure_bonus: +0.00  # Different files
  data_flow_bonus: +0.00  # No data flow
  conceptual_unity_bonus: +0.00  # Different problems
  scope_overlap_penalty: -0.00  # No overlap
  failure_correlation: -0.00  # Independent
  total_synergy: +0.00
```

**Note:** Not a synergy pair - included for completeness.

### Synergy Pair Selection Criteria Check

**Candidate Pair: Evolution Trajectory Tracking + Active Broadcast Polling**

| Criterion | Check | Result |
|-----------|-------|--------|
| Combined CV > 0.65 | (0.60 + 0.65) / 2 = 0.625 | **FAIL** (0.625 < 0.65) |
| Synergy bonus > 0 | 0.0 | **FAIL** |
| Shared infrastructure | No | **FAIL** |
| Risk profiles uncorrelated | Yes | PASS |
| State complexity manageable | Yes | PASS |
| Conceptual unity bonus | No | N/A |

**VERDICT:** Does not meet criteria.

**Candidate Pair: Evolution Trajectory Tracking + Per-Conductor Pattern Effectiveness**

| Criterion | Check | Result |
|-----------|-------|--------|
| Combined CV > 0.65 | (0.60 + 0.56) / 2 = 0.58 | **FAIL** (0.58 < 0.65) |
| Synergy bonus > 0 | +0.35 | PASS |
| Shared infrastructure | Yes (global_store.py) | PASS |
| Risk profiles uncorrelated | Both schema changes - **correlated** | **FAIL** |
| State complexity manageable | Both add state - moderate complexity | PASS |
| Conceptual unity bonus | +0.15 | PASS |

**VERDICT:** Does not meet criteria (low combined CV, correlated risk).

**Candidate Pair: Active Broadcast Polling + Drift Alert CLI**

| Criterion | Check | Result |
|-----------|-------|--------|
| Combined CV > 0.65 | (0.65 + 0.57) / 2 = 0.61 | **FAIL** (0.61 < 0.65) |
| Synergy bonus > 0 | 0.0 | **FAIL** |
| Shared infrastructure | No | **FAIL** |
| Risk profiles uncorrelated | Yes | PASS |
| State complexity manageable | No state changes | PASS |
| Conceptual unity bonus | No | N/A |

**VERDICT:** Does not meet criteria.

**ANALYSIS:** No candidate pair meets ALL criteria. This suggests Mozart is in a **STABILIZATION PHASE** where individual improvements are more appropriate than synergy pairs.

### Alternative: Best Single Candidate

Given no pair meets criteria, evaluate best single candidate:

| Candidate | CV | LOC (impl) | LOC (test) | Total LOC | Risk | Vision Alignment |
|-----------|-----|------------|------------|-----------|------|------------------|
| Active Broadcast Polling | 0.65 | 81 | 216 | 297 | LOW | Medium |
| Evolution Trajectory | 0.60 | 243 | 270 | 513 | MEDIUM | High |
| Drift Alert CLI | 0.57 | 74 | 216 | 290 | LOW | Low |
| Per-Conductor Effectiveness | 0.56 | 327 | 432 | 759 | HIGH | High |
| Escalation Auto-Apply | 0.53 | 121 | 300 | 421 | MEDIUM | Medium |
| Parallel Execution | 0.41 | 486 | 1080 | 1566 | HIGH | High |

**Best single candidate:** **Active Broadcast Polling** (highest CV, lowest risk, completes v14 infrastructure)

### Recommended Implementation Pair (Adjusted)

Despite no pair meeting all criteria, the score requires a pair recommendation. Selecting the **best available pair**:

**RECOMMENDED PAIR: Active Broadcast Polling + Evolution Trajectory Tracking**

**Adjusted Rationale:**
1. Combined CV: 0.625 (slightly below 0.65 threshold)
2. Total implementation LOC: 81 + 243 = 324 LOC (manageable)
3. Total test LOC: 216 + 270 = 486 LOC
4. Risk profiles uncorrelated: Yes
5. Synergy: +0.15 (trajectory could inform broadcast priorities)
6. Scope: One completes v14, one addresses epistemic drift

**WHY THIS PAIR:**
- Active Broadcast Polling is the highest CV candidate (0.65)
- Evolution Trajectory addresses the #1 external gap (epistemic drift)
- Together they balance "completing infrastructure" with "new capability"
- Combined LOC is reasonable (~810 total including tests)

---

## Phase 4: Research Candidate Identification + Aging

### Current Research Candidates

| Name | Age (cycles) | Status | Resolution |
|------|--------------|--------|------------|
| Sheet Contract Validation | 3 (carried v13-v15) | CLOSED | Static validation sufficient |
| Real-time Pattern Broadcasting | 2 (carried v13-v14) | IMPLEMENTED | v14 added infrastructure |

**RESEARCH_CANDIDATES: 0** (all resolved)

### New Research Candidate Assessment

Reviewing v16 candidates for potential research candidates:

| Candidate | CV | External Severity | Research Candidate? |
|-----------|-----|-------------------|---------------------|
| Evolution Trajectory | 0.60 | HIGH | No (CV above 0.50) |
| Per-Conductor Effectiveness | 0.56 | HIGH | No (CV above 0.50) |
| Active Broadcast Polling | 0.65 | MEDIUM | No (CV above 0.50) |
| Escalation Auto-Apply | 0.53 | MEDIUM | No (CV above 0.50) |
| Drift Alert CLI | 0.57 | LOW | No (CV above 0.50) |
| Parallel Execution | 0.41 | HIGH | **YES** |

### New Research Candidate

```yaml
research_candidate:
  name: Parallel Sheet Execution
  cv: 0.41
  external_severity: HIGH
  why_low_cv: |
    - Technical complexity (concurrency) drives down COMP scores
    - Integration risk across 5 files
    - Timing concern (EXP hesitation on "now")
  research_questions:
    - What's the minimum viable parallel execution model?
    - Can we do parallel within a sheet (sub-prompts) instead of across sheets?
    - What state coordination primitives are needed?
  carry_forward_to: v17 discovery phase
  age_in_cycles: 0 (new)
```

### Research Candidates Aged Out

No research candidates aged out this cycle. Both historical candidates were resolved:
- Sheet Contract: CLOSED (static validation sufficient)
- Pattern Broadcasting: IMPLEMENTED (v14)

---

## Phase 4B: Vision Alignment Assessment

**VISION.md Read:** Yes (Phase 1)

### Vision Alignment Scores by Candidate

#### Candidate 1: Evolution Trajectory Tracking

```yaml
vision_alignment:
  candidate: Evolution Trajectory Tracking
  alignment_scores:
    multi_conductor_progress: 0.30  # Enables understanding who improves what
    escalation_reduction: 0.20  # Indirect - trajectory could identify escalation patterns
    rlf_integration_path: 0.40  # Trajectory data could feed RLF's judgment context
    ai_person_parity: 0.30  # Tracking enables different conductors' contributions
  average_alignment: 0.30
  vision_contribution: |
    Enables Mozart to understand its own improvement trajectory, which feeds
    future RLF integration where AI people can query "how has this system evolved?"
  anti_patterns: none
```

#### Candidate 2: Per-Conductor Pattern Effectiveness

```yaml
vision_alignment:
  candidate: Per-Conductor Pattern Effectiveness
  alignment_scores:
    multi_conductor_progress: 0.90  # DIRECTLY enables multi-conductor
    escalation_reduction: 0.60  # Conductor-aware patterns reduce escalation
    rlf_integration_path: 0.80  # Per-conductor data is exactly what RLF needs
    ai_person_parity: 0.95  # Treats AI people as individuals with different profiles
  average_alignment: 0.81
  vision_contribution: |
    DIRECTLY advances Phase 3: "Pattern effectiveness is per-conductor" is
    explicitly stated in VISION.md Section 5 (Person-Aware Learning).
  anti_patterns: none
```

#### Candidate 3: Active Broadcast Polling

```yaml
vision_alignment:
  candidate: Active Broadcast Polling
  alignment_scores:
    multi_conductor_progress: 0.40  # Enables sharing across concurrent conductors
    escalation_reduction: 0.20  # Indirect benefit from pattern sharing
    rlf_integration_path: 0.30  # Broadcasting is a prerequisite for RLF awareness
    ai_person_parity: 0.30  # Benefits all conductors equally
  average_alignment: 0.30
  vision_contribution: |
    Completes the GWT-inspired broadcast pattern from v14. Real-time sharing
    is a building block for future multi-conductor coordination.
  anti_patterns: none
```

#### Candidate 4: Escalation Auto-Apply

```yaml
vision_alignment:
  candidate: Escalation Auto-Apply
  alignment_scores:
    multi_conductor_progress: 0.20  # Not directly related
    escalation_reduction: 0.70  # DIRECTLY reduces escalation frequency
    rlf_integration_path: 0.40  # But VISION wants RLF judgment, not historical suggestion
    ai_person_parity: 0.30  # Benefits all conductors
  average_alignment: 0.40
  vision_contribution: |
    Reduces human escalation friction, but via historical patterns rather than
    RLF's autonomous judgment. This is a stepping stone, not the target.
  anti_patterns: |
    **POTENTIAL ANTI-PATTERN:** Could entrench "history-based" escalation resolution
    when VISION specifies RLF judgment-based resolution. This might make the
    RLF integration HARDER by creating a competing system.
```

#### Candidate 5: Drift Alert CLI

```yaml
vision_alignment:
  candidate: Drift Alert CLI
  alignment_scores:
    multi_conductor_progress: 0.10  # Not related
    escalation_reduction: 0.10  # Not related
    rlf_integration_path: 0.10  # Not related
    ai_person_parity: 0.10  # Not related
  average_alignment: 0.10
  vision_contribution: |
    Pure operational improvement. Doesn't advance the vision directly.
  anti_patterns: none
```

#### Candidate 6: Parallel Sheet Execution

```yaml
vision_alignment:
  candidate: Parallel Sheet Execution
  alignment_scores:
    multi_conductor_progress: 0.95  # DIRECTLY enables parallel conductor execution
    escalation_reduction: 0.50  # Less waiting = less escalation opportunity
    rlf_integration_path: 0.70  # Parallel is prerequisite for multi-AI-person concerts
    ai_person_parity: 0.90  # Multiple AI people working simultaneously
  average_alignment: 0.76
  vision_contribution: |
    DIRECTLY advances Phase 3: "Parallel execution with sync points" is
    explicitly stated in VISION.md Evolution Path.
  anti_patterns: none
```

### Vision Alignment Summary

| Candidate | Average Alignment | Vision-Blocking? |
|-----------|-------------------|------------------|
| Per-Conductor Effectiveness | **0.81** | No |
| Parallel Execution | 0.76 | No |
| Escalation Auto-Apply | 0.40 | **Potential** |
| Evolution Trajectory | 0.30 | No |
| Active Broadcast Polling | 0.30 | No |
| Drift Alert CLI | 0.10 | No |

**VISION_BLOCKING_CANDIDATES: 1** (Escalation Auto-Apply has potential anti-pattern)

**AVERAGE_VISION_ALIGNMENT:** (0.30 + 0.81 + 0.30 + 0.40 + 0.10 + 0.76) / 6 = **0.45**

---

## Phase 5: Mini-META Reflection

### Did triplet analysis change priority order?

**YES.** Original Sheet 2 order was:
1. Evolution Trajectory
2. Per-Conductor Effectiveness
3. Active Broadcast Polling
4. Escalation Auto-Apply
5. Drift Alert CLI
6. Parallel Execution

After triplet analysis (by preliminary CV):
1. Active Broadcast Polling (0.65)
2. Evolution Trajectory (0.60)
3. Drift Alert CLI (0.57)
4. Per-Conductor Effectiveness (0.56)
5. Escalation Auto-Apply (0.53)
6. Parallel Execution (0.41)

**Key change:** Active Broadcast Polling rose from #3 to #1 because its triplet scores were consistently high (0.80-0.82) due to low risk and completing existing infrastructure.

### Which triplet produced the most insight?

**TRIPLET 3 (COMP+CULT↔EXP)** for Escalation Auto-Apply produced the most insight. It revealed the tension between:
- VISION wanting RLF-based judgment
- This candidate offering history-based shortcuts

This tension identified a **potential vision-blocking anti-pattern**.

### Did synergy analysis reveal opportunities not visible in isolation?

**YES.** The synergy analysis revealed:
1. Evolution Trajectory + Per-Conductor Effectiveness share conceptual unity (+0.15 bonus)
2. BUT they also share correlated risk (both schema changes)
3. No pair meets all criteria → **stabilization phase signal**

### Am I over-weighting COMP?

**PARTIALLY.** The CV scores are heavily influenced by technical feasibility (COMP scores). However, the Vision Alignment Assessment provides a counterbalance. Per-Conductor Effectiveness has lower CV (0.56) but highest vision alignment (0.81). This tension is informative.

### Are there RESEARCH_CANDIDATES that need attention?

**NEW:** Parallel Sheet Execution (CV 0.41, HIGH external severity) is now a research candidate.
**RESOLVED:** Sheet Contract (CLOSED), Pattern Broadcasting (IMPLEMENTED)

### Did Sheet Contract get resolved or closed?

**CLOSED in v13.** No action needed this cycle.

### Did conceptual unity bonus affect synergy pair selection?

**YES.** Evolution Trajectory + Per-Conductor Effectiveness received +0.15 conceptual unity bonus, raising their synergy score to +0.35 (highest). However, this wasn't enough to overcome their low combined CV (0.58).

### Did fixture catalog improve assessment speed?

**YES.** Knowing test_global_learning.py has comprehensive fixtures (factor 1.0) vs test_runner.py having partial fixtures (factor 1.2) allowed immediate estimation without searching fixture code.

### Did the test LOC floor affect any estimates?

**NO.** All estimates exceeded 50 LOC minimum floor. Lowest estimate was Drift Alert CLI at 216 LOC.

---

## Phase 6: Summary

### Recommended Implementation Pair

**PAIR: Active Broadcast Polling + Evolution Trajectory Tracking**

| Metric | Active Broadcast Polling | Evolution Trajectory |
|--------|--------------------------|----------------------|
| Preliminary CV | 0.65 | 0.60 |
| Implementation LOC | 81 | 243 |
| Test LOC | 216 | 270 |
| Total LOC | 297 | 513 |
| Adds State | No | Yes |
| CLI-Facing | No | No |
| Touches Escalation | No | No |
| Vision Alignment | 0.30 | 0.30 |
| Risk Level | LOW | MEDIUM |

**Combined Statistics:**
- Combined CV: 0.625 (below 0.65 threshold - stabilization signal)
- Total Implementation LOC: 324
- Total Test LOC: 486
- Grand Total LOC: 810
- Synergy Score: +0.15

### Research Candidates

| Name | Status | Age |
|------|--------|-----|
| Sheet Contract Validation | CLOSED | 3 (resolved v13) |
| Pattern Broadcasting | IMPLEMENTED | 2 (resolved v14) |
| Parallel Sheet Execution | NEW | 0 (v16 discovery) |

### CV Summary

| Candidate | Preliminary CV | Vision Alignment |
|-----------|----------------|------------------|
| Active Broadcast Polling | **0.65** | 0.30 |
| Evolution Trajectory | 0.60 | 0.30 |
| Drift Alert CLI | 0.57 | 0.10 |
| Per-Conductor Effectiveness | 0.56 | **0.81** |
| Escalation Auto-Apply | 0.53 | 0.40 |
| Parallel Execution | 0.41 | 0.76 |

**Average Preliminary CV:** (0.65 + 0.60 + 0.57 + 0.56 + 0.53 + 0.41) / 6 = **0.55**

**CV > 0.75 Candidates:** 0

### Stabilization Phase Indicators

1. No CV > 0.75 candidates (v10/v11/v15 had 0.76-0.86)
2. Best synergy pair doesn't meet all criteria
3. Highest CV is 0.65 (threshold level)
4. Research candidates resolved, not created
5. Most evolutions are "completing infrastructure" vs "new capabilities"

**CONCLUSION:** Mozart is in stabilization phase. Recommend Active Broadcast Polling (highest CV, lowest risk) as primary evolution, with Evolution Trajectory as secondary if time permits.

---

## Validation Ready

This completes Movement II-A: Domain Triplet Synthesis. All 6 candidates analyzed with 3 triplets each. Synergy matrix built with quantitative formula. Research candidates tracked with aging. Vision alignment assessed for all candidates.
