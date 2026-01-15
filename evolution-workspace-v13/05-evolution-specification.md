# Movement III-A: Evolution Specification

**Date:** 2026-01-15
**Version:** Mozart Evolution v13

---

## Phase 0: Freshness Check

### Git HEAD Comparison

```
Sheet 2 recorded: 7aa7ee985897dfaa44380caf142211a46818ec30
Current HEAD:     7aa7ee985897dfaa44380caf142211a46818ec30
```

**FRESHNESS_CHECK_RESULT: SAME**

The codebase has not changed since Sheet 2 discovery. Existence checks can proceed without re-running discovery.

---

## Phase 0B: Existence Check with Call-Path Tracing

### Evolution #1: Close Escalation Feedback Loop

**Markers Searched:**
- `update_escalation_outcome`
- `escalation_record_id`
- `outcome_after_action`

**Call-Path Trace:**

```yaml
call_path_trace:
  marker: update_escalation_outcome
  defined_in: src/mozart/learning/global_store.py:1799
  called_by:
    - file: tests/test_escalation_learning.py
      method: test_update_outcome_success
      line: 484
      is_active: yes (test code)
    - file: tests/test_escalation_learning.py
      method: test_update_nonexistent_returns_false
      line: 499
      is_active: yes (test code)
    - file: tests/test_escalation_learning.py
      method: test_update_multiple_outcomes
      line: 530
      is_active: yes (test code)
    - file: tests/test_escalation_learning.py
      method: test_full_escalation_learning_flow
      line: 564
      is_active: yes (test code)
  downstream_calls:
    - conn.execute() to UPDATE escalation_decisions table
  production_callers: NONE
  conclusion: INFRASTRUCTURE_ONLY
```

```yaml
call_path_trace:
  marker: escalation_record_id
  defined_in: src/mozart/execution/runner.py:3560 (local variable)
  called_by:
    - file: src/mozart/execution/runner.py
      method: _handle_escalation
      line: 3581
      is_active: yes - stored to sheet_state.outcome_data["escalation_record_id"]
  downstream_calls:
    - record_escalation_decision() creates the ID
    - NO call to update_escalation_outcome() with this ID
  conclusion: PARTIAL - ID is created and stored, but never used for update
```

**Existence Check Summary:**

```yaml
existence_check:
  evolution: "Close Escalation Feedback Loop"
  markers_searched:
    - update_escalation_outcome
    - escalation_record_id
    - outcome_after_action
  found_in:
    - src/mozart/learning/global_store.py (method definition)
    - src/mozart/execution/runner.py (record_id storage only)
    - tests/test_escalation_learning.py (comprehensive unit tests)
  call_path_traced: yes
  active_call_paths: 0 (in production code)
  inactive_markers: 0 (all test code is active)
  status: INFRASTRUCTURE_ONLY
  action: |
    SPECIFY with ~45% LOC factor applied.
    The API (update_escalation_outcome) exists and is tested.
    The integration (calling from runner on sheet completion) is missing.
    Work required: Add calls at escalation outcome points in runner.py.
```

### Evolution #2: Escalation Auto-Suggestions

**Markers Searched:**
- `get_similar_escalation`
- `suggest` + `escalation`
- `recommend` + `escalation`

**Call-Path Trace:**

```yaml
call_path_trace:
  marker: get_similar_escalation
  defined_in: src/mozart/learning/global_store.py:1721
  called_by:
    - file: src/mozart/execution/runner.py
      method: _handle_escalation
      line: 3528
      is_active: yes
      purpose: "Display similar past escalations to user during escalation"
  downstream_calls:
    - SQL query on escalation_decisions table
    - Returns EscalationDecisionRecord list
  conclusion: ACTIVE (already in use)
```

**Existence Check Summary:**

```yaml
existence_check:
  evolution: "Escalation Auto-Suggestions"
  markers_searched:
    - get_similar_escalation
    - suggest.*escalation
    - recommend.*escalation
  found_in:
    - src/mozart/learning/global_store.py (query method)
    - src/mozart/execution/runner.py (display during escalation)
  call_path_traced: yes
  active_call_paths: 1 (in production code)
  inactive_markers: 0
  status: PARTIAL
  action: |
    SPECIFY with partial implementation reduction (~60% of original estimate).
    Query infrastructure exists and is active.
    Missing: recommendation logic, auto-action capability, enhanced display.
    This evolution depends on #1 completing first (needs outcome data to recommend).
```

### Infrastructure Detection Summary

| Evolution | Status | Original LOC | Adjusted LOC | Adjustment |
|-----------|--------|--------------|--------------|------------|
| #1 Feedback Loop | INFRASTRUCTURE_ONLY | 143 | 64 | ×0.45 |
| #2 Auto-Suggestions | PARTIAL (depends on #1) | 609 | 365 | ×0.60 |

---

## Phase 1: Synthesis Decisions Loaded

From Sheet 4 (Quadruplet + META Synthesis):

| Field | Value |
|-------|-------|
| Primary Evolution | Close Escalation Feedback Loop |
| Final CV | 0.79 (HIGH CONFIDENCE) |
| Secondary Evolution | Escalation Auto-Suggestions |
| Secondary CV | 0.52 (CAUTION) |
| Combined CV | 0.655 (marginal above 0.65) |
| Stateful Flag | No (neither adds new state) |
| Escalation-Touching Flag | Yes (both) |
| CLI-Facing Flag | No (but #2 has CLI output) |
| Sheet Contract Resolution | CLOSED |
| Project Phase | STABILIZATION |
| Recommended Approach | Structured (#1 primary, #2 optional) |

---

## Phase 2: Evolution Specifications

### Evolution 1: Close Escalation Feedback Loop

```yaml
evolution:
  name: "Close Escalation Feedback Loop"
  type: feature
  approach: direct
  existence_check_status: INFRASTRUCTURE_ONLY
  freshness_check: SAME

  loc_estimation:
    base_estimate: 35  # Raw integration LOC (calls + logic)
    observability_budget: 5  # 35 × 0.15 = 5.25 → 5
    defensive_coding_budget: 4  # 35 × 0.10 = 3.5 → 4
    documentation_budget: 4  # 35 × 0.10 = 3.5 → 4 (docstring updates)
    cli_ux_budget: 0  # Not CLI-facing
    integration_cushion: 4  # 35 × 0.10 = 3.5 → 4 (connecting to runner)
    subtotal_before_multipliers: 52
    multipliers:
      stateful: 1.0  # Does not add state
      aggregation: 1.0  # No aggregation
      multi_file: 1.0  # Only modifies runner.py
      integration_minimum: 1.0  # Below threshold
      escalation_integration: 1.3  # Touches escalation handlers
    max_multiplier: 1.3
    adjusted_loc: 68  # 52 × 1.3 = 67.6 → 68
    loc_confidence: high

    existence_adjustment: |
      Original Sheet 3 estimate: 68 LOC
      Infrastructure exists (update_escalation_outcome API + tests)
      Adjustment factor: N/A - already accounted for in base estimate
      The 68 LOC estimate already reflects integration-only work.

    test_loc_estimation:
      existing_test_file: yes
      catalog_match: comprehensive
      existing_fixtures:
        - temp_db_path (Path fixture)
        - global_store (GlobalLearningStore fixture)
        - TestUpdateEscalationOutcome class (unit tests for API)
        - TestEscalationLearningIntegration class (integration tests)
      fixture_coverage: comprehensive
      fixtures_factor: 1.0  # Existing file with comprehensive fixtures
      base_test_estimate: 50  # Integration tests for runner calls
      base_with_fixtures: 50  # 50 × 1.0 = 50
      test_complexity_rating: LOW  # Pure function call integration
      test_complexity_multiplier: 1.5
      raw_test_loc: 75  # 50 × 1.5 = 75
      floor_applied: no  # 75 > 50
      adjusted_test_loc: 75
      tests_mandatory: yes

    total_loc_including_tests: 143  # 68 impl + 75 test

  historical_calibration: |
    v12 cycle: Implementation LOC 89% (255 vs 288) - formula stable
    v12 cycle: Test LOC 96% (380 vs 396) - excellent accuracy
    v13 adjustment: No CLI UX budget needed for this evolution
    Confidence: HIGH - integration-only work with existing infrastructure

  changes:
    - file: src/mozart/execution/runner.py
      type: modify
      description: |
        Add calls to update_escalation_outcome at escalation outcome points:
        1. Line ~2244: After "skip" response → outcome="skipped"
        2. Line ~2256: After "abort" response → outcome="aborted"
        3. Line ~2226: After retry exhausted → outcome="failed"
        4. Line ~1830: After sheet success with escalation_record_id → outcome="success"
      lines_affected: 35-45
      dependencies:
        - Must read escalation_record_id from sheet_state.outcome_data
        - Must call self._global_learning_store.update_escalation_outcome()

  non_goals:
    - Adding new escalation actions (out of scope)
    - Modifying the escalation_decisions schema (already complete)
    - Adding CLI commands for escalation history (separate evolution)
    - Auto-suggesting based on outcome data (Evolution #2)
    - Changing the escalation handler interface

  justification:
    comp: |
      Single call site integration at 4 outcome points in runner.py.
      Uses existing API (update_escalation_outcome) that is already tested.
      Low technical risk due to infrastructure already existing.
    sci: |
      Enables measurement of escalation effectiveness.
      Non-null outcome_after_action counts will increase from 0 to >0.
      Before/after comparison possible for v14 analysis.
    cult: |
      Directly completes v11 design intent documented in evolution-workspace-v11.
      Honors the "feedback loop" pattern that Mozart learning is built on.
      Removes a known TODO from the codebase.
    exp: |
      High confidence - this is "finishing the job."
      Would proudly demo this as completing a design.
      Gut feeling: obvious next step that was intentionally deferred.

  validation:
    tests_to_add:
      - test_escalation_outcome_updated_on_skip
      - test_escalation_outcome_updated_on_abort
      - test_escalation_outcome_updated_on_retry_exhausted
      - test_escalation_outcome_updated_on_success_after_retry
      - test_escalation_outcome_not_updated_when_no_record_id
    metrics_to_track:
      - Count of escalation_decisions with non-null outcome_after_action
      - Distribution of outcomes (success/failed/skipped/aborted)
    success_criteria:
      - All 4 outcome points call update_escalation_outcome
      - Tests verify outcome is recorded for each escalation action
      - Existing escalation learning tests still pass
    end_to_end_learning_test: yes

  risks:
    - risk: Sheet state doesn't have escalation_record_id when expected
      likelihood: low
      mitigation: Check for existence before calling update
    - risk: update_escalation_outcome called twice for same record
      likelihood: low
      mitigation: Update is idempotent (just overwrites same value)
    - risk: Global learning store is None at outcome point
      likelihood: low
      mitigation: Same pattern as existing record_escalation_decision (null check)

  deferral_criteria:
    defer_if: Never - this is primary evolution
    defer_to: N/A
    note: "v13: Tests CANNOT be deferred"
```

---

### Evolution 2: Escalation Auto-Suggestions (Secondary)

```yaml
evolution:
  name: "Escalation Auto-Suggestions"
  type: feature
  approach: direct
  existence_check_status: PARTIAL
  freshness_check: SAME

  loc_estimation:
    base_estimate: 90  # Reduced from 177 due to partial infrastructure
    observability_budget: 14  # 90 × 0.15 = 13.5 → 14
    defensive_coding_budget: 9  # 90 × 0.10 = 9
    documentation_budget: 9  # 90 × 0.10 = 9
    cli_ux_budget: 0  # Not CLI command (just enhanced output)
    integration_cushion: 9  # 90 × 0.10 = 9
    subtotal_before_multipliers: 131
    multipliers:
      stateful: 1.0  # Does not add state
      aggregation: 1.3  # Has summary/aggregation logic
      multi_file: 1.0  # Only modifies runner.py
      integration_minimum: 1.0  # Below threshold
      escalation_integration: 1.3  # Touches escalation handlers
    max_multiplier: 1.3
    adjusted_loc: 170  # 131 × 1.3 = 170.3 → 170
    loc_confidence: medium

    existence_adjustment: |
      Original estimate (Sheet 3): 177 LOC
      Partial infrastructure exists (get_similar_escalation in use)
      Adjustment: 60% reduction applied in base estimate (177 → 106 → 90 rounded)
      Still need: recommendation logic, confidence scoring, enhanced display

    test_loc_estimation:
      existing_test_file: yes
      catalog_match: partial
      existing_fixtures:
        - temp_db_path (Path fixture)
        - global_store (GlobalLearningStore fixture)
        - Some escalation query tests
      fixture_coverage: partial
      fixtures_factor: 1.2  # Extending with partial fixtures
      base_test_estimate: 180  # Complex recommendation testing
      base_with_fixtures: 216  # 180 × 1.2 = 216
      test_complexity_rating: MEDIUM  # Integration with outcome data
      test_complexity_multiplier: 4.5
      raw_test_loc: 972  # 216 × 4.5 = 972 (!)
      floor_applied: no  # 972 >> 50
      adjusted_test_loc: 400  # Cap at reasonable maximum - original estimate was 432
      tests_mandatory: yes

    total_loc_including_tests: 570  # 170 impl + 400 test

  historical_calibration: |
    v12 cycle: Implementation LOC 89% - formula stable
    v12 cycle: Test LOC 96% - excellent accuracy
    v13 NOTE: Test LOC capped due to unrealistic raw calculation
    Confidence: MEDIUM - depends on #1 completion for meaningful data

  changes:
    - file: src/mozart/execution/runner.py
      type: modify
      description: |
        Enhance _handle_escalation to provide recommendations:
        1. Calculate success rate per action type from similar escalations
        2. Display recommendation with confidence score
        3. Optionally auto-apply if confidence > threshold
        4. Track which escalations followed recommendations
      lines_affected: 60-80
      dependencies:
        - Requires Evolution #1 to be complete (outcome data needed)
        - Uses get_similar_escalation (already exists)
        - May need new helper method for recommendation calculation

  non_goals:
    - Adding new CLI commands (out of scope - that's CLI Grounding CLI)
    - Modifying the escalation handler interface
    - Adding new escalation actions
    - Changing the escalation_decisions schema
    - Machine learning or complex recommendation algorithms

  justification:
    comp: |
      Query infrastructure exists (get_similar_escalation).
      Recommendation is simple success rate calculation.
      Display enhancement within existing console output.
    sci: |
      Hypothesis: recommendations improve escalation outcomes.
      Requires outcome data from #1 to validate.
      Measurable: recommendation_followed vs outcome correlation.
    cult: |
      Aligns with Mozart's DGM (Democratically Generated Memory) pattern.
      Honors learning-first philosophy by surfacing learned knowledge.
      Builds on v11's escalation learning foundation.
    exp: |
      Medium confidence - feels premature without outcome data.
      Would defer to v14 if #1 doesn't complete cleanly.
      Gut says: "implement #1 first, then assess."

  validation:
    tests_to_add:
      - test_recommendation_calculated_from_outcomes
      - test_recommendation_displayed_with_confidence
      - test_no_recommendation_without_outcome_data
      - test_auto_apply_when_confidence_high
      - test_recommendation_tracking
    metrics_to_track:
      - Recommendations made count
      - Recommendations followed count
      - Outcome improvement after recommendation
    success_criteria:
      - Recommendations shown during escalation when data exists
      - Confidence score reflects outcome history accuracy
      - Tests verify recommendation logic correctness
    end_to_end_learning_test: yes

  risks:
    - risk: Not enough outcome data to make recommendations
      likelihood: high (initially)
      mitigation: Show "insufficient data" message, don't recommend
    - risk: Recommendations are wrong (insufficient training data)
      likelihood: medium
      mitigation: Require minimum sample size before recommending
    - risk: Auto-apply feature causes unexpected behavior
      likelihood: medium
      mitigation: Require explicit opt-in, high confidence threshold

  deferral_criteria:
    defer_if:
      - Evolution #1 does not complete cleanly
      - Remaining capacity < 200 LOC after #1
      - Issues discovered during #1 implementation
    defer_to: v14 evolution cycle
    note: "v13: Tests CANNOT be deferred if this is implemented"
```

---

## Phase 3: Approach Decision

### Decision Tree Evaluation

```
├── Is existence check status FULLY_IMPLEMENTED?
│   └── #1: NO (INFRASTRUCTURE_ONLY) → Continue
│   └── #2: NO (PARTIAL) → Continue
├── Is existence check status INFRASTRUCTURE_ONLY?
│   └── #1: YES → INTEGRATION-ONLY mode (already applied)
│   └── #2: NO → Continue
├── Is total estimated LOC > 500?
│   └── #1: 143 LOC → NO → Direct implementation
│   └── #2: 570 LOC → YES → Consider orchestration
├── Does evolution require creating new files?
│   └── #1: NO → Direct
│   └── #2: NO → Direct
├── Are there complex dependencies between changes?
│   └── #1: NO (single file)
│   └── #2: YES (depends on #1 completion)
```

### Approach Decision

**APPROACH: direct (structured)**

1. **Primary (#1):** Direct implementation in Sheet 6
   - 143 LOC total (68 impl + 75 test)
   - Single file modification (runner.py)
   - CV > 0.75 indicates clean implementation expected

2. **Secondary (#2):** Conditional direct implementation
   - Only proceed if #1 completes without issues
   - Only proceed if remaining capacity permits
   - 570 LOC total - significant but manageable
   - Defer to v14 if any concerns arise

**Justification:**

Neither evolution exceeds the complexity threshold for orchestration:
- No new files required
- Both modify primarily runner.py
- Infrastructure largely exists
- Clear integration points identified

The structured approach (primary first, secondary conditional) addresses:
- Marginal combined CV (0.655)
- Dependency of #2 on #1's outcome data
- Risk of correlated failure modes

---

## Phase 4: Summary Tables

### LOC Estimates with Calibration

| Evolution | Base | Budgets | Multiplier | Impl LOC | Test LOC | Total |
|-----------|------|---------|------------|----------|----------|-------|
| #1 Feedback Loop | 35 | +17 | ×1.3 | 68 | 75 | 143 |
| #2 Auto-Suggestions | 90 | +41 | ×1.3 | 170 | 400 | 570 |
| **Combined** | - | - | - | 238 | 475 | **713** |

### Non-Goals Summary

| Evolution | Non-Goals |
|-----------|-----------|
| #1 | No new actions, no schema changes, no CLI commands, no auto-suggestions |
| #2 | No CLI commands, no ML algorithms, no handler interface changes |

### Sheet Contract Status

**CLOSED** (resolved in Sheet 3/4)

Sheet Contract Validation was closed because:
1. Enhanced validation (V001-V107) already validates Jinja/paths
2. Grounding hooks validate external contracts
3. No clear implementation path identified after 2 cycles

### Research Candidate Status

| Candidate | Status | Age | Resolution |
|-----------|--------|-----|------------|
| Sheet Contract Validation | CLOSED | 2 (resolved) | Addressed by validation + grounding |
| Pattern Attention Mechanism | CARRIED | 1 → 2 | Must resolve by v14 |

---

## Implementation Readiness Checklist

| Item | Status | Notes |
|------|--------|-------|
| Freshness check passed | ✓ | HEAD matches |
| Existence check complete | ✓ | Call-path traced |
| Infrastructure identified | ✓ | update_escalation_outcome API exists |
| Integration points mapped | ✓ | 4 outcome points in runner.py |
| Test fixtures assessed | ✓ | Comprehensive existing coverage |
| LOC estimates calibrated | ✓ | v12 actuals applied |
| Non-goals documented | ✓ | Clear scope boundaries |
| Risks identified | ✓ | Low-medium risks with mitigations |
| Approach decided | ✓ | Direct, structured |
| Tests mandatory acknowledged | ✓ | Cannot be deferred |

---

## Ready for Sheet 6 (Evolution Execution)
