# Mozart Global Learning Architecture

**Status:** IMPLEMENTED (v1.0)
**Created:** 2025-12-27
**Implemented:** 2026-01-14
**Vision:** Every Mozart instance learns from execution, aggregates patterns across all workspaces, and improves over time.

---

## Overview

Mozart's global learning system aggregates execution outcomes and patterns across all workspaces, enabling Mozart to learn from every job and improve retry strategies, error handling, and pattern detection over time.

### What Was Built

| Component | File | LOC | Purpose |
|-----------|------|-----|---------|
| Global Store | `src/mozart/learning/global_store.py` | 1196 | SQLite persistence |
| Aggregator | `src/mozart/learning/aggregator.py` | 348 | Pattern merging |
| Weighter | `src/mozart/learning/weighter.py` | 340 | Priority calculation |
| Error Hooks | `src/mozart/learning/error_hooks.py` | 283 | Error learning |
| Migration | `src/mozart/learning/migration.py` | 283 | Workspace import |
| Tests | `tests/test_global_learning.py` | 569 | 29 test cases |

**Total:** ~3300 LOC implementation + tests

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Mozart Instance                                │
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────────┐ │
│  │   Runner     │───▶│   Pattern    │───▶│   Global Learning Store    │ │
│  │  (executes)  │    │  Aggregator  │    │  (~/.mozart/global-...)    │ │
│  └──────────────┘    └──────────────┘    └────────────────────────────┘ │
│         │                   │                         │                  │
│         ▼                   ▼                         ▼                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   Error Learning Hooks                             │  │
│  │  - Records error classifications and recoveries                    │  │
│  │  - Learns adaptive wait times from recovery success               │  │
│  │  - Shares learned delays across workspaces                         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## CLI Commands

```bash
# View global patterns
mozart patterns [--type failure|success|recovery] [--limit N]

# Aggregate from workspaces
mozart aggregate-patterns [--from-workspaces] [--workspace PATH]

# View learning statistics
mozart learning-stats
```

---

## Key Decisions (TDF-Validated)

| Decision | Choice | CV |
|----------|--------|-----|
| Global Store Location | SQLite (~/.mozart/global-learning.db) | 0.82 |
| Aggregation Trigger | Immediate on job completion | 0.83 |
| Pattern Weighting | Combined recency + effectiveness | 0.80 |
| Error Learning | Extend ErrorClassifier with hooks | 0.82 |

---

## Future Work (v2+)

- Preflight Learning (query patterns before execution)
- Mid-Run Adaptation (modify prompts during retry)
- GitHub Contribution Pipeline (anonymize and share patterns)
- Centralized Learning Store (PostgreSQL for multi-user)

---

*Implemented 2026-01-14 via Mozart Global Learning Orchestration*

---
---

# Original Design Document (Reference)

The sections below contain the original design document for reference.

---

## First Principles

### What is Mozart learning?

Every execution generates knowledge at multiple levels:

```
Level 0: Raw Events
├── Command executed, exit code, duration
├── Stdout/stderr content
├── Validation results (pass/fail, confidence)
└── Resource usage (tokens, time, retries)

Level 1: Patterns
├── Failure patterns ("validation markers missing")
├── Success patterns ("explicit format in code block")
├── Recovery patterns ("retry with clarification works")
└── Timing patterns ("sheet 4 consistently slow")

Level 2: Insights
├── Prompt structure effectiveness
├── Config structure effectiveness
├── Validation reliability scores
├── Error recoverability classifications

Level 3: Improvements
├── Prompt template changes
├── Validation logic changes
├── Default config changes
├── Error handling changes
├── Documentation updates
└── New helper utilities
```

### What can be improved?

| Category | Example | Contribution Type |
|----------|---------|-------------------|
| **Prompts** | "Add explicit validation marker format" | Template change |
| **Validations** | "file_exists + size > 0 more reliable than content_contains" | Code change |
| **Defaults** | "completion_threshold_percent: 60 works better than 50" | Config change |
| **Retry Logic** | "Rate limit needs exponential backoff floor of 30s" | Code change |
| **Error Messages** | "Suggest checking X when Y error occurs" | Code/docs change |
| **New Validators** | "Add yaml_valid validator for config generation" | Code addition |

---

## Architecture

### Local Learning Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Mozart Instance                          │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Runner     │───▶│   Learning   │───▶│   Local DB   │  │
│  │  (executes)  │    │   Extractor  │    │  (SQLite)    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                    │          │
│         │                   ▼                    │          │
│         │           ┌──────────────┐             │          │
│         │           │   Pattern    │             │          │
│         │           │   Analyzer   │             │          │
│         │           └──────────────┘             │          │
│         │                   │                    │          │
│         ▼                   ▼                    ▼          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Learning Application Layer               │  │
│  │  - Preflight checks (before execution)               │  │
│  │  - Mid-run adaptation (during execution)             │  │
│  │  - Post-run analysis (after execution)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Contribution Preparation                 │  │
│  │  - Anonymization                                      │  │
│  │  - Deduplication                                      │  │
│  │  - Quality filtering                                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      GitHub Contribution      │
              │  (PR to mozart-ai-compose)    │
              └──────────────────────────────┘
```

### Database Schema (Local SQLite)

```sql
-- Core execution data
CREATE TABLE executions (
    id TEXT PRIMARY KEY,
    job_hash TEXT NOT NULL,           -- Hash of job name (anonymized)
    sheet_num INTEGER NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,
    status TEXT,                       -- completed, failed, etc.
    retry_count INTEGER DEFAULT 0,
    first_attempt_success BOOLEAN,

    -- Aggregated metrics (no PII)
    prompt_token_estimate INTEGER,
    validation_count INTEGER,
    validation_pass_rate REAL,
    confidence_score REAL
);

-- Learned patterns
CREATE TABLE patterns (
    id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,        -- failure, success, recovery, timing
    pattern_name TEXT NOT NULL,        -- e.g., "validation_markers_missing"
    description TEXT,

    -- Statistics
    occurrence_count INTEGER DEFAULT 1,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,

    -- Effectiveness
    led_to_success_count INTEGER DEFAULT 0,
    led_to_failure_count INTEGER DEFAULT 0,

    -- Suggested action
    suggested_action TEXT,             -- e.g., "add explicit format to prompt"
    action_effectiveness REAL          -- 0.0-1.0 based on outcomes
);

-- Pattern-execution linkage
CREATE TABLE execution_patterns (
    execution_id TEXT REFERENCES executions(id),
    pattern_id TEXT REFERENCES patterns(id),
    PRIMARY KEY (execution_id, pattern_id)
);

-- Improvement suggestions (ready for contribution)
CREATE TABLE improvements (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,            -- prompt, validation, config, code, docs
    target_file TEXT,                  -- Relative path in repo
    description TEXT NOT NULL,

    -- The actual change
    change_type TEXT,                  -- add, modify, delete
    change_content TEXT,               -- Diff or new content

    -- Evidence
    supporting_pattern_ids TEXT,       -- JSON array of pattern IDs
    confidence REAL,                   -- 0.0-1.0
    sample_size INTEGER,               -- How many executions support this

    -- Contribution status
    status TEXT DEFAULT 'pending',     -- pending, contributed, rejected, merged
    contributed_at TIMESTAMP,
    pr_url TEXT
);

-- Contribution history
CREATE TABLE contributions (
    id TEXT PRIMARY KEY,
    contributed_at TIMESTAMP,
    pr_url TEXT,
    status TEXT,                       -- open, merged, closed
    improvements_included TEXT,        -- JSON array of improvement IDs

    -- Anonymized instance identifier (rotates periodically)
    instance_hash TEXT
);
```

### Anonymization Layer

**Must Strip:**
```python
STRIP_FIELDS = [
    "pid",
    "user_id",
    "username",
    "home_directory",
    "absolute_paths",      # Convert to relative
    "api_keys",
    "environment_variables",
    "hostnames",
    "ip_addresses",
    "timestamps",          # Normalize to relative times
    "stdout_content",      # May contain sensitive data
    "stderr_content",      # May contain sensitive data
]
```

**Must Hash (for correlation without exposure):**
```python
HASH_FIELDS = [
    "job_name",            # Hash to allow pattern matching
    "workspace_path",      # Hash to correlate executions
    "project_path",        # Hash to identify project-specific patterns
]
```

**Keep As-Is:**
```python
KEEP_FIELDS = [
    "pattern_names",       # Generic, no PII
    "validation_types",    # Generic
    "success_rates",       # Aggregate metrics
    "retry_counts",        # Numeric
    "duration_buckets",    # Bucketed, not exact
    "config_structure",    # Structure only, no values
    "prompt_structure",    # Structure only, no content
]
```

---

## Three Modes of Learning Application

### Mode 1: Preflight Learning (Before Execution)

```yaml
# In job config
learning:
  enabled: true
  preflight:
    enabled: true
    check_similar_failures: true
    apply_learned_patterns: true
    warn_on_risky_patterns: true
```

**What it does:**
1. Before sheet execution, queries local DB for similar past executions
2. Identifies patterns that led to failures
3. Applies learned mitigations (e.g., modify prompt, adjust timeout)
4. Warns user of risky patterns detected in config

**Example preflight output:**
```
Sheet 3 preflight:
  ⚠ Similar sheet failed 2/3 times historically
  ⚠ Pattern detected: "validation_markers_missing" (80% failure rate)
  ✓ Applied mitigation: Added explicit marker format to prompt
  ✓ Adjusted timeout: 1800s → 2400s (based on timing patterns)
```

### Mode 2: Mid-Run Adaptation (During Execution)

```yaml
learning:
  enabled: true
  mid_run:
    enabled: true
    adapt_on_failure: true
    max_adaptations: 2
```

**What it does:**
1. On validation failure, queries patterns for this failure type
2. Applies learned recovery strategies
3. Modifies prompt/approach for retry based on what worked before

**Decision flow:**
```
Validation Failed
      │
      ▼
Query: "What worked when this pattern occurred before?"
      │
      ▼
Found: "Adding 'IMPORTANT: Include exact format' worked 73% of the time"
      │
      ▼
Apply: Modify prompt with learned enhancement
      │
      ▼
Retry with adaptation
```

### Mode 3: Post-Run Analysis (After Execution)

```yaml
learning:
  enabled: true
  post_run:
    enabled: true
    extract_patterns: true
    suggest_improvements: true
    queue_for_contribution: true
```

**What it does:**
1. Analyzes completed job for new patterns
2. Updates pattern statistics (success/failure counts)
3. Generates improvement suggestions
4. Queues significant improvements for contribution

---

## Contribution Flow

### Local → GitHub PR Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. Improvement Detected                                    │
│     Pattern X led to failure, but modification Y fixed it   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Quality Gate                                            │
│     - Minimum sample size (e.g., 5 occurrences)            │
│     - Minimum confidence (e.g., 0.7)                        │
│     - Not already contributed                               │
│     - Improvement category allowed                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Anonymization                                           │
│     - Strip all PII                                         │
│     - Hash identifiers                                      │
│     - Normalize paths                                       │
│     - Remove content, keep structure                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Contribution Preparation                                │
│     - Generate diff/patch                                   │
│     - Write evidence summary                                │
│     - Create PR description                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. GitHub PR Creation                                      │
│     - Fork if needed                                        │
│     - Create branch: learning/pattern-name-hash             │
│     - Commit changes                                        │
│     - Open PR with evidence                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Human Review (maintainer)                               │
│     - Review anonymization                                  │
│     - Validate improvement                                  │
│     - Merge or close with feedback                          │
└─────────────────────────────────────────────────────────────┘
```

### Contribution Categories

| Category | Auto-Contribute | Requires Review | Example |
|----------|-----------------|-----------------|---------|
| **Pattern Definition** | Yes | Light | New failure pattern identified |
| **Default Config Values** | Yes | Light | Better default timeout |
| **Prompt Structure** | Yes | Moderate | Add explicit format section |
| **Validation Logic** | No | Heavy | New validator implementation |
| **Error Handling** | No | Heavy | New error recovery code |
| **Core Algorithm** | No | Heavy | Retry strategy changes |

### PR Template for Contributions

```markdown
## Mozart Learning Contribution

**Type:** [Pattern | Config | Prompt | Validation | Code]
**Confidence:** 0.XX
**Sample Size:** N executions

### Evidence Summary

This improvement was learned from N executions across M distinct jobs.

**Pattern Observed:**
- [Description of failure pattern]
- Occurrence rate: X%
- Failure rate without mitigation: Y%

**Improvement Applied:**
- [Description of improvement]
- Success rate with improvement: Z%

### Changes

[Diff of proposed changes]

### Anonymization Verification

- [ ] No absolute paths
- [ ] No usernames or user IDs
- [ ] No API keys or secrets
- [ ] No project-specific content
- [ ] Instance identifier is hashed

---
*Automatically generated by Mozart Learning System*
*Instance: [hashed-id] | Contribution: [contribution-id]*
```

---

## Centralized Learning Store (Future)

### Option A: Shared PostgreSQL

```
┌─────────────────────────────────────────────────────────────┐
│                  Central Learning Database                   │
│                     (PostgreSQL)                             │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Patterns  │  │ Statistics │  │Improvements│            │
│  │  (global)  │  │ (aggregate)│  │  (queued)  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ▲                ▲                ▲
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │Instance │      │Instance │      │Instance │
    │    A    │      │    B    │      │    C    │
    └─────────┘      └─────────┘      └─────────┘
```

**Pros:**
- Real-time learning sharing
- Immediate benefit from others' learnings
- Centralized statistics

**Cons:**
- Requires network access
- Privacy concerns (even with anonymization)
- Single point of failure
- Hosting cost

### Option B: Git-Based Federation (Recommended for Start)

```
┌─────────────────────────────────────────────────────────────┐
│                GitHub: mozart-ai-compose                     │
│                                                              │
│  /learnings/                                                 │
│  ├── patterns/                                               │
│  │   ├── validation-markers-missing.yaml                    │
│  │   ├── rate-limit-exponential-backoff.yaml                │
│  │   └── ...                                                 │
│  ├── defaults/                                               │
│  │   ├── recommended-timeouts.yaml                          │
│  │   └── recommended-thresholds.yaml                        │
│  └── statistics/                                             │
│      └── aggregate-patterns.yaml  (updated by CI)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         ▲                ▲                ▲
         │ PR             │ PR             │ PR
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │Instance │      │Instance │      │Instance │
    │    A    │      │    B    │      │    C    │
    │ (local) │      │ (local) │      │ (local) │
    └─────────┘      └─────────┘      └─────────┘
         │                │                │
         └────────────────┴────────────────┘
                          │
                    git pull
                   (sync learnings)
```

**Pros:**
- Works offline
- No central infrastructure needed
- Full transparency (all learnings in git)
- Human review before merge
- Natural deduplication via git

**Cons:**
- Async (not real-time)
- Requires git/GitHub access for contribution
- PR review bottleneck

### Option C: Hybrid (Future Scale)

```
Local SQLite → Periodic sync → Central PostgreSQL → Materialized to Git
                                      │
                                      ▼
                              Real-time API for
                              high-value lookups
```

---

## Implementation Phases

### Phase 1: Local Learning Foundation
**Timeline:** 1-2 weeks

1. Migrate from JSON outcome store to SQLite
2. Implement pattern extraction from executions
3. Add pattern table and linkage
4. Basic preflight pattern checking
5. Configuration options for enabling/disabling

**Deliverables:**
- `src/mozart/learning/database.py` - SQLite schema and operations
- `src/mozart/learning/patterns.py` - Pattern extraction and matching
- `src/mozart/learning/preflight.py` - Preflight checks
- Updated `JobConfig` with learning options

### Phase 2: Mid-Run Adaptation
**Timeline:** 1 week

1. Query patterns on validation failure
2. Apply learned mitigations to prompts
3. Track adaptation effectiveness
4. Update pattern statistics after run

**Deliverables:**
- `src/mozart/learning/adaptation.py` - Mid-run adaptation logic
- Updated runner integration

### Phase 3: Improvement Detection
**Timeline:** 1-2 weeks

1. Post-run analysis for improvement opportunities
2. Improvement suggestion generation
3. Quality gate filtering
4. Local improvement queue

**Deliverables:**
- `src/mozart/learning/improvements.py` - Improvement detection
- `src/mozart/learning/quality.py` - Quality gates

### Phase 4: Anonymization Layer
**Timeline:** 1 week

1. Comprehensive field stripping
2. Identifier hashing
3. Content sanitization
4. Verification tests

**Deliverables:**
- `src/mozart/learning/anonymize.py` - Anonymization logic
- Extensive test coverage for PII detection

### Phase 5: GitHub Contribution Pipeline
**Timeline:** 2 weeks

1. PR generation from improvements
2. Evidence summary formatting
3. Automated branch/commit/PR creation
4. Contribution tracking

**Deliverables:**
- `src/mozart/learning/contribute.py` - GitHub contribution logic
- `src/mozart/cli.py` additions for manual contribution trigger
- CI workflow for contribution validation

### Phase 6: Learning Sync
**Timeline:** 1 week

1. Pull merged learnings from GitHub
2. Apply to local database
3. Periodic sync mechanism

**Deliverables:**
- `src/mozart/learning/sync.py` - Learning synchronization
- CLI command: `mozart learning sync`

---

## CLI Commands

```bash
# View local learning statistics
mozart learning stats

# Show patterns detected
mozart learning patterns [--type failure|success|recovery]

# Show queued improvements
mozart learning improvements [--status pending|contributed|merged]

# Trigger contribution (with review)
mozart learning contribute [--dry-run]

# Sync learnings from upstream
mozart learning sync

# Export anonymized learnings (for manual review)
mozart learning export --output learnings.yaml

# Run self-improvement analysis
mozart learning analyze [workspace]
```

---

## Configuration

```yaml
# In mozart config or ~/.config/mozart/config.yaml

learning:
  # Master switch
  enabled: true

  # Local storage
  database_path: ~/.mozart/learnings.db

  # Preflight checks
  preflight:
    enabled: true
    check_similar_failures: true
    apply_learned_patterns: true
    warn_threshold: 0.5  # Warn if failure rate above this

  # Mid-run adaptation
  adaptation:
    enabled: true
    max_adaptations_per_sheet: 2
    min_confidence: 0.6

  # Post-run analysis
  post_run:
    enabled: true
    extract_patterns: true
    suggest_improvements: true

  # Contribution
  contribute:
    enabled: true  # Queue improvements for contribution
    auto_contribute: false  # Require manual trigger
    min_sample_size: 5
    min_confidence: 0.7
    categories:
      - patterns
      - defaults
      - prompts
    # Exclude categories that need heavy review
    exclude_categories:
      - validation_code
      - core_code

  # Sync
  sync:
    enabled: true
    auto_sync: true  # Pull on startup
    sync_interval_hours: 24
```

---

## Security Considerations

### Privacy Guarantees

1. **No content leaves local machine without explicit action**
   - Patterns are structural, not content-based
   - Improvements describe changes, don't include user data

2. **Anonymization is verified before contribution**
   - Automated scanning for PII patterns
   - Manual review step available

3. **Hashed identifiers rotate**
   - Instance hash rotates monthly
   - Cannot track individual users over time

4. **Opt-out is always available**
   - `learning.contribute.enabled: false` prevents any contribution
   - `learning.enabled: false` disables all learning

### What Never Leaves

- API keys or credentials
- File contents (only structure)
- Absolute paths
- Usernames or user IDs
- Stdout/stderr content
- Environment variables
- Exact timestamps (only relative durations)

---

## Success Metrics

### Local Learning
- Reduction in first-attempt failures
- Reduction in total retry count
- Improvement in average confidence scores

### Global Learning
- Number of patterns in central repository
- Number of improvements merged
- Adoption rate of learned defaults

### Quality
- False positive rate on pattern detection
- Effectiveness of applied mitigations
- PII leak incidents (target: 0)

---

## Open Questions

1. **Should code changes ever be auto-contributed?**
   - Current design: No, always requires PR review
   - Alternative: Auto-merge for trivial changes with high confidence

2. **How to handle conflicting learnings?**
   - Different instances learn opposite things
   - Need conflict resolution strategy

3. **Rate limiting contributions?**
   - Prevent spam from misconfigured instances
   - Quality over quantity

4. **Versioning learnings?**
   - Patterns may become obsolete as Mozart evolves
   - Need versioning or deprecation mechanism

---

*This document outlines the vision for Mozart's distributed learning system. Implementation should be phased, with local learning as the foundation before any contribution mechanisms are built.*
