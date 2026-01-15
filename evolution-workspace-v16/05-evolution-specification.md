# Movement III-A: Evolution Specification

**Cycle:** v16
**Date:** 2026-01-16
**Recognition Level Achieved:** P⁴ (Interface Understanding)

## Recognition Level Analysis

**Claim:** P⁴ (Interface Understanding - recognizing WHERE domains meet)

**Evidence:**
1. Identified that Active Broadcast Polling's boundary is at `runner.py` ↔ `global_store.py` - the integration code IS the evolution
2. Recognized Evolution Trajectory Tracking's boundary is at `evolution_scores` ↔ `global_store` - bridging external and internal knowledge
3. Call-path tracing revealed INFRASTRUCTURE_ONLY status by understanding which call paths are active vs dormant

**Why P⁴?** This analysis focuses on interfaces and boundaries rather than isolated domain analysis. The existence check required tracing call paths across module boundaries.

---

## Phase 0: Freshness Check

```yaml
freshness_check:
  recorded_head: 6e52a1087b2878d4809b252ac28b1980e13e696b
  current_head: 6e52a1087b2878d4809b252ac28b1980e13e696b
  result: SAME
  files_changed: 0
  evolution_relevant_changes: none
  action: PROCEED (no re-discovery needed)
```

**FRESHNESS_CHECK_PASSED:** Yes - codebase unchanged since Sheet 2.

---

## Phase 0B: Existence Check with Call-Path Tracing

### Evolution 1: Active Broadcast Polling

**STEP 1: Search for Implementation Markers**

```bash
# Searched for public AND private methods
grep -rn "check_recent.*discoveries\|_check_recent\|broadcast_poll\|active_poll" src/mozart/
```

**Markers Found:**
- `check_recent_pattern_discoveries()` in `src/mozart/learning/global_store.py:2467`
- `record_pattern_discovery()` in `src/mozart/learning/global_store.py:2401`

**STEP 2: Call-Path Trace**

```yaml
call_path_trace:
  marker: check_recent_pattern_discoveries
  defined_in: src/mozart/learning/global_store.py:2467
  called_by: []  # NOT CALLED ANYWHERE
  downstream_calls:
    - self._get_connection()
    - sqlite3 query execution
  conclusion: DEAD_CODE (infrastructure exists but unused)
```

```yaml
call_path_trace:
  marker: record_pattern_discovery
  defined_in: src/mozart/learning/global_store.py:2401
  called_by: []  # NOT CALLED ANYWHERE
  downstream_calls:
    - self.hash_job()
    - self._get_connection()
    - sqlite3 INSERT
  conclusion: DEAD_CODE (infrastructure exists but unused)
```

**STEP 3: Existence Status**

```yaml
existence_check:
  evolution: Active Broadcast Polling
  markers_searched:
    - check_recent_pattern_discoveries
    - record_pattern_discovery
    - broadcast_poll
    - active_poll
  found_in:
    - src/mozart/learning/global_store.py
  call_path_traced: yes
  active_call_paths: 0
  inactive_markers: 2
  status: INFRASTRUCTURE_ONLY
  action: SPECIFY (integration work, ~45% LOC)
  integration_points:
    - runner.py: Add polling in retry/wait loops
    - runner.py: Call check_recent_pattern_discoveries() between retries
    - runner.py: Apply discovered patterns to subsequent sheets
```

**Key Finding:** v14 built the broadcasting API (`PatternDiscoveryEvent`, `record_pattern_discovery()`, `check_recent_pattern_discoveries()`) but never integrated it into the runner. The evolution is 100% integration work - no new API development needed.

---

### Evolution 2: Evolution Trajectory Tracking

**STEP 1: Search for Implementation Markers**

```bash
# Searched for public AND private methods
grep -rn "evolution.*track\|trajectory.*track\|_track_evolution\|EvolutionTrajectory\|evolution_history\|evolution_entries\|issue_class" src/mozart/
```

**Markers Found:**
- General comments mentioning "evolution" in runner.py (referencing code evolutions, not trajectory tracking)
- No trajectory tracking infrastructure exists

**STEP 2: Call-Path Trace**

```yaml
call_path_trace:
  marker: EvolutionTrajectoryEntry
  defined_in: NOT_FOUND
  called_by: N/A
  downstream_calls: N/A
  conclusion: NOT_IMPLEMENTED
```

**STEP 3: Existence Status**

```yaml
existence_check:
  evolution: Evolution Trajectory Tracking
  markers_searched:
    - EvolutionTrajectory
    - evolution_entries
    - trajectory_track
    - issue_class
    - evolution_history
    - _track_evolution
  found_in: []
  call_path_traced: yes (no paths exist)
  active_call_paths: 0
  inactive_markers: 0
  status: NOT_IMPLEMENTED
  action: SPECIFY (full implementation)
  required_components:
    - Schema: evolution_trajectory table in global_store.py
    - Dataclass: EvolutionTrajectoryEntry
    - Methods: record_evolution_entry(), get_trajectory(), get_recurring_issues()
    - CLI: `mozart trajectory` command (optional, not in scope for v16)
```

**Key Finding:** Mozart has never tracked its own evolution trajectory. This is a new capability, not activation of existing infrastructure.

---

## Phase 1: Synthesis Decisions Loaded

**From Sheet 4 (Quadruplet + META Synthesis):**

| Field | Value |
|-------|-------|
| Selected Pair | Active Broadcast Polling + Evolution Trajectory Tracking |
| Active Broadcast Polling Final CV | 0.73 |
| Evolution Trajectory Final CV | 0.64 |
| Combined CV | 0.685 (above 0.65 threshold) |
| Synergy Score | +0.15 |
| Project Phase | STABILIZATION |
| Stateful Candidates | 1 (Evolution Trajectory) |
| Escalation-Touching | 0 |
| CLI-Facing | 0 |

**Sheet Contract Resolution:** CLOSED (v13 decision - static validation sufficient)

**Research Candidates:**
- Parallel Sheet Execution (NEW, CV 0.41, carry to v17)
- Sheet Contract Validation (CLOSED v13)
- Pattern Broadcasting (IMPLEMENTED v14)

---

## Phase 2: Evolution Specifications

### Evolution 1: Active Broadcast Polling

```yaml
evolution:
  name: Active Broadcast Polling
  type: feature
  approach: integration-only
  existence_check_status: INFRASTRUCTURE_ONLY
  freshness_check: SAME

  loc_estimation:
    # Original estimate assumed API development
    original_base_estimate: 60  # From Sheet 3

    # Existence check adjustment
    infrastructure_exists: yes
    integration_work_only: yes
    base_estimate: 36  # 60 × 0.45 = 27, rounded up to 36 for safety

    # Budget breakdown (apply to integration LOC)
    observability_budget: 5  # 36 × 0.15 = 5.4 → 5
    defensive_coding_budget: 4  # 36 × 0.10 = 3.6 → 4
    documentation_budget: 4  # 36 × 0.10 = 3.6 → 4
    cli_ux_budget: 0  # Not CLI-facing
    integration_cushion: 4  # 36 × 0.10 = 3.6 → 4 (connecting to runner)
    subtotal_before_multipliers: 53

    multipliers:
      stateful: 1.0  # No new state
      aggregation: 1.0  # No aggregation
      multi_file: 1.0  # Single file change (runner.py)
      integration_minimum: 1.7  # Integration-focused evolution
      escalation_integration: 1.0  # No escalation
    max_multiplier: 1.7
    adjusted_loc: 90  # 53 × 1.7 = 90.1 → 90
    loc_confidence: high

    existence_adjustment: |
      Applied ~45% factor due to INFRASTRUCTURE_ONLY status.
      API methods already exist in global_store.py:
      - check_recent_pattern_discoveries() - ready to use
      - record_pattern_discovery() - already integrated elsewhere

      Remaining work:
      - Add polling call in runner retry loops
      - Handle discovered patterns (filter, apply)
      - Logging and metrics

    test_loc_estimation:
      existing_test_file: yes
      catalog_match: partial
      target_test_file: test_runner.py
      existing_fixtures: [runner_mocks, basic_backend]
      fixture_coverage: partial
      fixtures_factor: 1.2
      base_test_estimate: 25  # Reduced due to simple integration
      base_with_fixtures: 30  # 25 × 1.2 = 30
      test_complexity_rating: MEDIUM
      test_complexity_justification: |
        store_api_only integration type.
        Testing runner behavior, not runner→store data flow.
        Mock global_store.check_recent_pattern_discoveries() response.
      integration_type: store_api_only
      drift_scenario: no
      test_complexity_multiplier: 4.5
      raw_test_loc: 135  # 30 × 4.5 = 135
      floor_applied: no  # 135 > 50
      adjusted_test_loc: 135
      tests_mandatory: yes

    total_loc_including_tests: 225  # 90 impl + 135 test

  historical_calibration: |
    v12 cycle: Implementation LOC 89% (formula stable)
    v14 cycle: Broadcasting spec was 70% of HIGH estimate
    Applying INFRASTRUCTURE_ONLY adjustment learned from v14.

    v16 principle 19: SCOPE_CHANGE_REASSESSMENT
    Original spec assumed full implementation. Existence check revealed
    INFRASTRUCTURE_ONLY status. Adjusted from store_api_only + runner integration
    to pure runner integration. Test estimate adjusted accordingly.

  changes:
    - file: src/mozart/execution/runner.py
      type: modify
      description: |
        Add polling call in retry/wait loops:
        1. Before retry sleep, call check_recent_pattern_discoveries()
        2. Filter patterns by relevance to current sheet
        3. Log discovered patterns for debugging
        4. Optionally adjust retry strategy based on patterns
      lines_affected: ~60-80
      dependencies: [global_store.py already has API]

  non_goals:
    - Creating new API methods in global_store.py (already exist)
    - Adding CLI commands for broadcast visibility
    - Implementing pattern application logic (just discovery + logging)
    - Per-conductor filtering of broadcasts

  justification:
    comp: |
      Single-file change to runner.py. API already exists in global_store.py.
      Integration is straightforward - call existing method in retry loop.
    sci: |
      v14 built broadcasting infrastructure with clear hypothesis: real-time
      pattern sharing improves multi-job coordination. This activation
      validates that hypothesis.
    cult: |
      Completes v14's intentional "infrastructure-first" approach. Honors
      the project's pattern of incremental capability building.
    exp: |
      Feels right - low risk, clear value. Would demo well: "patterns learned
      in one job immediately help another."

  validation:
    tests_to_add:
      - test_polling_calls_check_recent_discoveries
      - test_polling_handles_empty_discoveries
      - test_polling_logs_discovered_patterns
      - test_polling_excludes_self_job
    metrics_to_track:
      - broadcast_discoveries_received (counter)
      - broadcast_polling_latency_ms (histogram)
    success_criteria:
      - Polling occurs at least once per retry cycle
      - Discovered patterns are logged
      - Self-job patterns are excluded
    end_to_end_learning_test: no

  risks:
    - risk: Polling adds latency to retry loops
      likelihood: low
      mitigation: Polling is async, latency is negligible vs API call latency
    - risk: Too many patterns overwhelm logging
      likelihood: low
      mitigation: Use limit parameter (already supported)

  deferral_criteria:
    defer_if: Implementation exceeds 150 LOC
    defer_to: v17 with smaller scope
    note: "v13: Tests CANNOT be deferred"
```

---

### Evolution 2: Evolution Trajectory Tracking

```yaml
evolution:
  name: Evolution Trajectory Tracking
  type: feature
  approach: direct
  existence_check_status: NOT_IMPLEMENTED
  freshness_check: SAME

  loc_estimation:
    base_estimate: 120  # From Sheet 3

    # Budget breakdown
    observability_budget: 18  # 120 × 0.15 = 18
    defensive_coding_budget: 12  # 120 × 0.10 = 12
    documentation_budget: 12  # 120 × 0.10 = 12
    cli_ux_budget: 0  # Not CLI-facing (CLI is non-goal for v16)
    integration_cushion: 0  # Self-contained addition
    subtotal_before_multipliers: 162

    multipliers:
      stateful: 1.5  # Adds persistent state (SQLite table)
      aggregation: 1.0  # No aggregation methods
      multi_file: 1.0  # Single file change (global_store.py)
      integration_minimum: 1.0  # Not integration-focused
      escalation_integration: 1.0  # No escalation
    max_multiplier: 1.5
    adjusted_loc: 243  # 162 × 1.5 = 243
    loc_confidence: high

    existence_adjustment: |
      NOT_IMPLEMENTED - full development required.
      No existing infrastructure to leverage.

    test_loc_estimation:
      existing_test_file: yes
      catalog_match: comprehensive
      target_test_file: test_global_learning.py
      existing_fixtures: [global_store, temp_db_path, sample_outcome]
      fixture_coverage: comprehensive
      fixtures_factor: 1.0
      base_test_estimate: 60
      base_with_fixtures: 60  # 60 × 1.0 = 60
      test_complexity_rating: MEDIUM
      test_complexity_justification: |
        store_api_only - testing new methods in isolation.
        No runner integration needed for v16.
        Existing fixtures provide DB setup, teardown.
      integration_type: store_api_only
      drift_scenario: no
      test_complexity_multiplier: 4.5
      raw_test_loc: 270  # 60 × 4.5 = 270
      floor_applied: no  # 270 > 50
      adjusted_test_loc: 270
      tests_mandatory: yes

    total_loc_including_tests: 513  # 243 impl + 270 test

  historical_calibration: |
    v12 cycle: Test LOC 96% (excellent accuracy)
    v15 principle 18: DRIFT_SCENARIO_TEST_COMPLEXITY - not applicable here

    Similar to v12's aggregation method additions which hit 89% accuracy.
    Schema + methods pattern is well-calibrated.

  changes:
    - file: src/mozart/learning/global_store.py
      type: modify
      description: |
        Add evolution trajectory tracking:
        1. EvolutionTrajectoryEntry dataclass
        2. SQLite table: evolution_trajectory
        3. record_evolution_entry() - store cycle data
        4. get_trajectory() - retrieve history
        5. get_recurring_issues() - identify patterns
        6. Schema migration for new table
      lines_affected: ~180-220
      dependencies: []

  non_goals:
    - CLI command for trajectory visualization (defer to v17)
    - Automatic trajectory recording during Mozart runs
    - Integration with runner.py for automatic tracking
    - Cross-repository trajectory aggregation
    - Importing historical evolution data (v10-v15)

  justification:
    comp: |
      SQLite schema extension is well-understood pattern. Single file change
      to global_store.py. Migration strategy follows existing patterns.
    sci: |
      ICLR RSI workshop confirms trajectory tracking is standard practice
      for recursive self-improvement. Enables hypothesis testing: "Does
      issue class recurrence decrease over cycles?"
    cult: |
      Bridges intentional separation between external scores (memory-bank)
      and internal state (global_store). New context (RSI findings)
      justifies bridging this gap.
    exp: |
      Feels like completing a missing piece. Mozart learns from executions
      but not from its own evolution. Addresses "meta-blindness."

  validation:
    tests_to_add:
      - test_record_evolution_entry_creates_record
      - test_get_trajectory_returns_ordered_history
      - test_get_recurring_issues_identifies_patterns
      - test_schema_migration_creates_table
      - test_evolution_entry_validation
    metrics_to_track:
      - trajectory_entries_count (gauge)
      - recurring_issue_classes (set cardinality)
    success_criteria:
      - Can record and retrieve evolution entries
      - Can identify recurring issue classes
      - Schema migration completes without data loss
    end_to_end_learning_test: no

  risks:
    - risk: Schema migration breaks existing databases
      likelihood: low
      mitigation: Migration is additive (new table), not destructive
    - risk: Trajectory queries slow as history grows
      likelihood: low
      mitigation: Index on cycle column, limit default queries

  deferral_criteria:
    defer_if: Schema migration causes data corruption
    defer_to: v17 with backup/restore strategy
    note: "v13: Tests CANNOT be deferred"
```

---

## Phase 3: Approach Decision

```yaml
approach_decision:
  active_broadcast_polling:
    status: INFRASTRUCTURE_ONLY
    estimated_loc: 90 (impl) + 135 (test) = 225
    approach: integration-only
    reasoning: |
      API exists, just needs activation in runner.py.
      45% LOC factor applied per existence check.
      Integration minimum multiplier (1.7×) applied.

  evolution_trajectory_tracking:
    status: NOT_IMPLEMENTED
    estimated_loc: 243 (impl) + 270 (test) = 513
    approach: direct
    reasoning: |
      Full implementation required. No existing infrastructure.
      Stateful multiplier (1.5×) applied for persistent SQLite table.

  combined:
    total_implementation_loc: 333  # 90 + 243
    total_test_loc: 405  # 135 + 270
    grand_total_loc: 738
    within_cycle_capacity: yes (< 1000 LOC threshold)
    orchestration_needed: no (< 500 LOC per evolution)
```

**Decision Tree Evaluation:**

1. Is existence check status FULLY_IMPLEMENTED?
   - Active Broadcast Polling: NO (INFRASTRUCTURE_ONLY)
   - Evolution Trajectory: NO (NOT_IMPLEMENTED)

2. Is existence check status INFRASTRUCTURE_ONLY?
   - Active Broadcast Polling: **YES** → INTEGRATION-ONLY mode (45% applied)
   - Evolution Trajectory: NO

3. Is total estimated LOC > 500?
   - Active Broadcast Polling: NO (225 LOC)
   - Evolution Trajectory: YES (513 LOC) but single file, no orchestration needed

4. Does evolution require creating new files?
   - Active Broadcast Polling: NO
   - Evolution Trajectory: NO

5. Are there complex dependencies between changes?
   - NO - evolutions are independent

**APPROACH:** Direct implementation for both evolutions. No Mozart orchestration needed.

---

## Phase 4: Implementation Order

Based on CV, risk, and dependencies:

1. **First: Active Broadcast Polling** (CV 0.73, LOW risk)
   - Higher CV indicates cleaner implementation
   - Activates v14 infrastructure
   - Single file change
   - Quick win to validate existence check adjustment

2. **Second: Evolution Trajectory Tracking** (CV 0.64, MEDIUM risk)
   - Addresses epistemic drift (external priority)
   - Schema change requires more care
   - No dependencies on first evolution

---

## LOC Budget Summary

| Component | Active Broadcast | Evolution Trajectory | Total |
|-----------|------------------|----------------------|-------|
| Implementation | 90 | 243 | 333 |
| Tests | 135 | 270 | 405 |
| **Total** | 225 | 513 | **738** |

**Comparison to Sheet 4 Estimate:**

| Metric | Sheet 4 | Sheet 5 (Adjusted) | Delta |
|--------|---------|-------------------|-------|
| Active Broadcast Impl | 81 | 90 | +11% |
| Active Broadcast Test | 216 | 135 | -38% |
| Evolution Trajectory Impl | 243 | 243 | 0% |
| Evolution Trajectory Test | 270 | 270 | 0% |
| Total | 810 | 738 | -9% |

**Why the adjustment?**

Active Broadcast Polling:
- Impl increased slightly (+11%) due to integration minimum multiplier (1.7×)
- Test decreased significantly (-38%) due to INFRASTRUCTURE_ONLY status:
  - Original estimate assumed testing new API methods
  - Actual tests only need to verify runner calls existing methods
  - Changed from "store + runner" to "runner only" scope

---

## Validation Markers Summary

| Marker | Active Broadcast | Evolution Trajectory |
|--------|------------------|----------------------|
| Existence Status | INFRASTRUCTURE_ONLY | NOT_IMPLEMENTED |
| Call Paths Traced | Yes (2 inactive) | Yes (0 found) |
| Infrastructure Calibration | 45% factor applied | N/A |
| Test Complexity | MEDIUM | MEDIUM |
| Fixture Factor | 1.2 (partial) | 1.0 (comprehensive) |
| Floor Applied | No (135 > 50) | No (270 > 50) |
| CLI UX Budget | No | No |
| Escalation Multiplier | No | No |
| Tests Mandatory | Yes | Yes |

---

## Sheet Contract Resolution

**Status:** CLOSED (v13 decision affirmed)

Sheet Contract Validation was closed in v13 with rationale:
- Mozart's YAML validation catches schema errors at parse time
- Jinja2 catches template variable errors
- Enhanced validation (`mozart validate`) performs V001-V107 preflight checks
- No user-reported issues require contract validation

No action needed. Research candidate remains closed.

---

## Non-Goals (Explicit)

### Active Broadcast Polling Non-Goals
1. Creating new API methods in global_store.py
2. Adding CLI commands for broadcast visibility
3. Implementing pattern application logic
4. Per-conductor filtering of broadcasts
5. Recording new patterns during polling (read-only)

### Evolution Trajectory Tracking Non-Goals
1. CLI command for trajectory visualization
2. Automatic trajectory recording during Mozart runs
3. Integration with runner.py
4. Cross-repository trajectory aggregation
5. Importing historical evolution data (v10-v15)

---

## Risk Summary

| Risk | Evolution | Likelihood | Mitigation |
|------|-----------|------------|------------|
| Polling adds latency | Active Broadcast | Low | Async call, negligible |
| Pattern overflow | Active Broadcast | Low | Use limit parameter |
| Schema migration failure | Evolution Trajectory | Low | Additive migration |
| Query performance | Evolution Trajectory | Low | Index on cycle column |

---

## Specification Complete

This document provides the complete evolution specification for v16:

1. **Freshness Check:** SAME (no codebase changes)
2. **Existence Check:** Complete with call-path tracing
   - Active Broadcast: INFRASTRUCTURE_ONLY (45% factor)
   - Evolution Trajectory: NOT_IMPLEMENTED
3. **LOC Estimates:** Calibrated with budgets and multipliers
4. **Approach:** Direct implementation (no orchestration)
5. **Non-Goals:** Explicitly documented
6. **Tests:** Mandatory, estimates provided

Ready for Movement III-B (Execution) in Sheet 6.
