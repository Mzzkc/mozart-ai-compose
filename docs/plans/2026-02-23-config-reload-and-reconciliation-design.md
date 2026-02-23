# Config Auto-Reload, Reconciliation, and Timeout Defaults

**Date:** 2026-02-23
**Status:** Design
**Scope:** CLI, daemon, runner lifecycle, docs, skills, audit score

---

## Problem Statement

Three related issues with Mozart's config and validation lifecycle:

1. **Validation command timeout default is 300s (5 min)** — too low for real test suites. Sheet 12 of a production concert hit this ceiling 10 times running `pytest`. The constant `VALIDATION_COMMAND_TIMEOUT_SECONDS` in `src/mozart/core/constants.py:77` is the culprit.

2. **Config is cached on first run; resume ignores YAML changes** — Mozart snapshots the full `JobConfig` into `CheckpointState.config_snapshot` at first execution. Subsequent `resume` uses the snapshot, not the YAML file on disk. Users must pass `--reload-config` explicitly. This is a footgun: edit YAML, resume, wonder why nothing changed.

3. **No config reconciliation on reload** — when config IS reloaded, only the snapshot gets replaced. Derived state in the daemon registry (`JobRecord.config_path`, `JobRecord.workspace`), checkpoint-level accumulators (cost tracking, rate limit counters), and other config-derived fields remain stale. Removed config sections leave ghost state.

Additionally: **confidence that workspace contains no status remnants** — the daemon should be the sole source of truth for job state. An audit score should verify this systematically.

---

## Work Items

### 1. Validation Command Timeout: 300s to 3600s

**Change:** Single constant update + documentation sweep.

**Files:**

| File | Change |
|------|--------|
| `src/mozart/core/constants.py:77` | `VALIDATION_COMMAND_TIMEOUT_SECONDS = 300` to `3600`, update docstring |
| `src/mozart/core/config/execution.py:388` | Update `ValidationRule.timeout_seconds` description referencing default |
| `docs/score-writing-guide.md` | Update "300-second hard limit" references |
| `docs/configuration-reference.md` | Update timeout table row |
| `/home/emzi/.claude/skills/mozart-score-authoring.md` | Update line referencing 300s hard limit |

**Behavior:** Validation commands get up to 1 hour by default. Per-rule `timeout_seconds` override still works. `asyncio.wait_for()` returns immediately when command finishes early — no wasted time.

---

### 2. Auto-Reload Config on Resume (Remove `--reload-config`)

**Behavior change:** Resuming a job automatically reloads from the original YAML file when it exists on disk. Cached `config_snapshot` is a fallback, not the default. The `--reload-config` / `-r` flag is fully removed (dead code).

#### New `_reconstruct_config()` Priority

```
1. Explicit --config file.yaml          (unchanged — always wins)
2. Auto-reload from stored config_path   (NEW default, if file exists)
3. Cached config_snapshot               (fallback when file is gone)
4. Error                                (nothing available)
```

#### New Flag: `--no-reload`

For users who explicitly want the cached snapshot (deterministic replay). Skips auto-reload even when the YAML file exists.

#### Output

- Always shown: `"Config reloaded from {path}"` or `"Using cached config ({path} not found on disk)"`
- When config differs from cached snapshot: `"Config changed: N fields updated"`
- With `--verbose`: field-level diff showing what changed

#### Full Removal Scope

**Source code — logic changes:**

| File | Change |
|------|--------|
| `src/mozart/cli/commands/resume.py` | Remove `--reload-config` flag, rewrite `_reconstruct_config()` priority, add `--no-reload`, add diff logic |
| `src/mozart/cli/commands/pause.py` | Update `modify` to not pass `reload_config=True` (auto-reload is default) |
| `src/mozart/daemon/job_service.py` | Mirror new priority in `_reconstruct_config()`, remove `reload_config` parameter |

**Tests — rewrite:**

| File | Change |
|------|--------|
| `tests/test_cli.py` | Rewrite `_reconstruct_config` tests for new priority order |
| `tests/test_cli_run_resume.py` | Rewrite `_reconstruct_config` tests, remove `reload_config` param tests |
| `tests/test_daemon_job_service.py` | Remove `test_reload_config_with_no_path_raises`, add auto-reload tests |

**Documentation — remove all `--reload-config` references:**

| File | References |
|------|------------|
| `docs/cli-reference.md` | Flag in resume options table, example command |
| `CHANGELOG.md` | Feature announcement |
| `examples/README.md` | Example command |
| `CLAUDE.md` | Resume example, modify example |

**Skills — remove all `--reload-config` references:**

| File | References (count) |
|------|--------------------|
| `/home/emzi/.claude/skills/mozart-usage.md` | 12+ references across tables, examples, pitfalls |
| `/home/emzi/.claude/skills/mozart-score-authoring.md` | Pitfall #17 |

#### Modify Command

`modify` continues to work as: pause, reload config, reconcile, resume. The only change is it no longer needs to explicitly pass `reload_config=True` because auto-reload is the default. If `modify` provides `--config new.yaml`, that takes priority (rule 1 above). The user-facing behavior is identical.

---

### 3. Config Reconciliation on Reload

**Problem:** Reloading config replaces the snapshot but leaves derived state stale. This is a structural problem — patching individual fields is fragile and future additions will be missed.

#### Architecture: `reconcile_config()` as a Lifecycle Event

New module: `src/mozart/execution/reconciliation.py`

Called automatically from `_reconstruct_config()` in both `resume.py` and `job_service.py` whenever a config reload occurs (auto or explicit).

#### Layer 1: Registry and Metadata

Update daemon-level metadata to reflect the new config:

- `JobRecord.config_path` and `JobRecord.workspace` in SQLite registry
- `JobMeta.config_path` and `JobMeta.workspace` in manager's in-memory map

This ensures `mozart status`, `mozart resume`, and other operations target the correct paths.

#### Layer 2: Checkpoint State Reconciliation

A declarative mapping from config sections to the checkpoint state fields they influence:

```python
CONFIG_STATE_MAPPING: dict[str, list[str]] = {
    "cost_limits": ["total_cost", "sheet_costs", "cost_warnings_issued"],
    "rate_limit": ["rate_limit_waits", "quota_waits"],
    "circuit_breaker": ["circuit_breaker_state", "circuit_breaker_trips"],
    "retry": [],  # per-sheet retry counts are already reset on new execution
    "backend": [],  # runner is recreated, no checkpoint state to reset
    "prompt": [],  # runner is recreated with new prompt builder
    "learning": [],  # learning store is re-initialized
    "parallel": [],  # parallel executor is recreated
    "stale_detection": [],  # read fresh from config each execution
    "sheet": [],  # sheet definitions are re-read, DAG rebuilt
    "hooks": [],  # hooks are re-read from config
    "notifications": [],  # notifications re-initialized
    "isolation": [],  # worktree manager recreated
    "grounding": [],  # grounding engine recreated
}
```

**Reconciliation logic:**

1. Diff old config snapshot vs new config at section level
2. For each section that changed or was removed: reset the mapped checkpoint fields to defaults
3. For sections with empty mapping (`[]`): no checkpoint state to reset (runner recreation handles it)
4. Log what was reset

**Sections removed entirely from new config:** Their mapped checkpoint fields are reset to defaults. No ghost state.

#### Layer 3: Structural Test (Future-Proofing)

A test that introspects `JobConfig` model fields and asserts every top-level section has an entry in `CONFIG_STATE_MAPPING`:

```python
def test_config_state_mapping_completeness():
    """Every JobConfig section must have a CONFIG_STATE_MAPPING entry."""
    config_sections = set(JobConfig.model_fields.keys())
    mapped_sections = set(CONFIG_STATE_MAPPING.keys())

    # Exclude non-reconcilable metadata fields
    METADATA_FIELDS = {"name", "workspace", "state_backend", "state_path", ...}
    reconcilable = config_sections - METADATA_FIELDS

    unmapped = reconcilable - mapped_sections
    assert not unmapped, (
        f"Config sections {unmapped} have no CONFIG_STATE_MAPPING entry. "
        "Add an entry to define what checkpoint state should be reset "
        "when this section changes. Use [] if the runner recreates it."
    )
```

This is the structural guarantee: add a new config section without a mapping entry and CI fails with a clear message telling you exactly what to do.

#### Layer 4: ReconciliationReport

`reconcile_config()` returns a `ReconciliationReport` dataclass:

```python
@dataclass
class ReconciliationReport:
    registry_updated: bool
    sections_changed: list[str]
    sections_removed: list[str]
    fields_reset: dict[str, list[str]]  # section -> list of reset field names
    config_diff: dict[str, tuple[Any, Any]]  # field -> (old, new) for verbose output
```

This drives both logging output and the `--verbose` diff display.

---

### 4. Workspace Status Audit Score

A fan-out Mozart score that systematically audits every code path for workspace-resident status tracking. Each fan-out instance examines a different subsystem from a different perspective.

#### Fan-Out Instances

| Instance | Scope | Question |
|----------|-------|----------|
| 1 | `src/mozart/cli/commands/` | Do any CLI commands write status/state files to the workspace? |
| 2 | `src/mozart/daemon/` | Does the daemon write any job state to workspace directories? |
| 3 | `src/mozart/execution/runner/` | Does the runner write anything beyond logs and artifacts to workspace? |
| 4 | `src/mozart/state/` | Do state backends ever use workspace-relative paths in daemon mode? |
| 5 | `src/mozart/execution/validation/` | Does the validation engine write state to workspace? |
| 6 | All of `src/mozart/` | Grep for workspace path writes, file creation in workspace, `.mozart-*` patterns |

#### Synthesis (Fan-In)

A final sheet cross-references all instance findings and produces a verdict:

- **CLEAN**: No workspace status remnants found (only legitimate files: logs, artifacts, pause signals)
- **REMNANTS FOUND**: List of specific code paths that write status to workspace, with file:line references

#### Validation

Each instance produces a structured JSON report. The synthesis sheet validates that all instances completed and produces a final summary with actionable findings.

---

## Files Affected (Complete List)

### Source Code

| File | Work Item | Change |
|------|-----------|--------|
| `src/mozart/core/constants.py` | 1 | Timeout `300` to `3600` |
| `src/mozart/core/config/execution.py` | 1 | Update field description |
| `src/mozart/cli/commands/resume.py` | 2 | Remove `--reload-config`, add `--no-reload`, rewrite `_reconstruct_config()` |
| `src/mozart/cli/commands/pause.py` | 2 | Update `modify` to not pass `reload_config` |
| `src/mozart/daemon/job_service.py` | 2, 3 | Remove `reload_config` param, mirror new priority, call `reconcile_config()` |
| `src/mozart/daemon/manager.py` | 3 | Expose method to update `JobMeta` on reconciliation |
| `src/mozart/daemon/registry.py` | 3 | Expose method to update `JobRecord` on reconciliation |
| `src/mozart/execution/reconciliation.py` | 3 | **NEW** — `reconcile_config()`, `CONFIG_STATE_MAPPING`, `ReconciliationReport` |

### Tests

| File | Work Item | Change |
|------|-----------|--------|
| `tests/test_cli.py` | 2 | Rewrite `_reconstruct_config` tests |
| `tests/test_cli_run_resume.py` | 2 | Rewrite `_reconstruct_config` tests |
| `tests/test_daemon_job_service.py` | 2 | Remove reload_config test, add auto-reload tests |
| `tests/test_reconciliation.py` | 3 | **NEW** — reconciliation logic tests, structural completeness test |

### Documentation

| File | Work Item | Change |
|------|-----------|--------|
| `docs/score-writing-guide.md` | 1, 2 | Update timeout reference, remove reload-config references |
| `docs/configuration-reference.md` | 1 | Update timeout table |
| `docs/cli-reference.md` | 2 | Remove `--reload-config`, add `--no-reload` |
| `CHANGELOG.md` | 1, 2, 3 | Update feature description |
| `examples/README.md` | 2 | Update resume example |
| `CLAUDE.md` | 1, 2 | Update resume examples, remove reload-config mentions |

### Skills

| File | Work Item | Change |
|------|-----------|--------|
| `/home/emzi/.claude/skills/mozart-usage.md` | 2 | Remove 12+ `--reload-config` references, update pitfalls |
| `/home/emzi/.claude/skills/mozart-score-authoring.md` | 1, 2 | Update timeout reference, remove pitfall #17 |

### Score

| File | Work Item |
|------|-----------|
| `scores/workspace-status-audit.yaml` | 4 | **NEW** — fan-out audit score |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Auto-reload breaks deterministic replay | `--no-reload` flag preserves old behavior |
| Config file deleted mid-execution | Fallback to cached snapshot with clear log message |
| Reconciliation resets state user wanted to keep | Log everything that gets reset; `--no-reload` skips reconciliation |
| `CONFIG_STATE_MAPPING` incomplete at launch | Structural test catches it immediately |
| Future config sections missing mapping | Structural test fails CI with actionable message |
| Modify command breaks | Modify is pause + resume; auto-reload is default, so modify just works without explicit reload flag |

---

## Non-Goals

- Hot-reloading config for in-flight sheet execution (runner is mid-process; can only apply on next resume)
- Automatic config reload on SIGHUP for individual jobs (daemon SIGHUP already reloads daemon-level config; job-level config reloads on resume)
- Migration of existing cached snapshots (they continue to work as fallbacks)
