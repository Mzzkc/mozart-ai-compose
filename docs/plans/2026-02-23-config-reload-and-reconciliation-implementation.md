# Config Auto-Reload, Reconciliation, and Timeout Defaults — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Mozart auto-reload config on resume, reconcile stale state structurally, and fix the 300s validation timeout default.

**Architecture:** Four independent work items: (1) constant change for timeout, (2) rewrite `_reconstruct_config()` in CLI and daemon to auto-reload with `--no-reload` escape hatch, (3) new `reconciliation.py` module with declarative `CONFIG_STATE_MAPPING` and structural test, (4) fan-out audit score. Items 1 and 4 are independent. Items 2 and 3 are sequential (reconciliation depends on the new config priority logic).

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, aiosqlite, pytest, typer, Rich

**Design doc:** `docs/plans/2026-02-23-config-reload-and-reconciliation-design.md`

---

## Task 1: Validation Command Timeout Default (300s to 3600s)

**Files:**
- Modify: `src/mozart/core/constants.py:77`
- Modify: `src/mozart/core/config/execution.py:384-390`

**Step 1: Update the constant**

In `src/mozart/core/constants.py`, change line 77:

```python
# Before:
VALIDATION_COMMAND_TIMEOUT_SECONDS = 300
"""Timeout for user-defined validation commands (5 minutes)."""

# After:
VALIDATION_COMMAND_TIMEOUT_SECONDS = 3600
"""Timeout for user-defined validation commands (1 hour)."""
```

**Step 2: Update the field description**

In `src/mozart/core/config/execution.py`, update the `timeout_seconds` field description (lines 384-390):

```python
    timeout_seconds: float | None = Field(
        default=None,
        gt=0,
        description="Timeout for command_succeeds validations (seconds). "
        "Overrides the global VALIDATION_COMMAND_TIMEOUT_SECONDS default (3600s / 1 hour). "
        "Use for validations that need a shorter or longer ceiling.",
    )
```

**Step 3: Run existing tests to verify no breakage**

Run: `pytest tests/test_validation_checks.py -v -x`
Expected: All tests pass (timeout constant is only used at runtime, not in test assertions)

**Step 4: Commit**

```bash
git add src/mozart/core/constants.py src/mozart/core/config/execution.py
git commit -m "fix: increase validation command timeout default from 300s to 3600s

300s (5 min) was too low for real test suites. Production concerts
hit this ceiling repeatedly running pytest. 3600s (1 hour) is generous
enough for any validation command while still catching hung processes."
```

---

## Task 2: Update Timeout References in Documentation

**Files:**
- Modify: `docs/score-writing-guide.md` (search for "300" near "command" or "timeout")
- Modify: `docs/configuration-reference.md` (timeout table)
- Modify: `/home/emzi/.claude/skills/mozart-score-authoring.md` (line referencing 300s hard limit)

**Step 1: Update score-writing-guide.md**

Search for all references to "300" in the context of command validation timeout and update to "3600" / "1 hour". The key reference is near line 1161-1165 describing `command_succeeds` having a "300s hard limit".

**Step 2: Update configuration-reference.md**

Find the timeout table row for "Command validation" showing "300s" and update to "3600s (1 hour)".

**Step 3: Update mozart-score-authoring.md skill**

Find the reference to "300-second hard limit" for `command_succeeds` validation (near line 497) and update to "3600-second" / "1 hour".

**Step 4: Commit**

```bash
git add docs/score-writing-guide.md docs/configuration-reference.md
git commit -m "docs: update validation command timeout references to 3600s"
```

Then separately:

```bash
git add /home/emzi/.claude/skills/mozart-score-authoring.md
git commit -m "docs: update score authoring skill timeout reference to 3600s"
```

---

## Task 3: Rewrite `_reconstruct_config()` in CLI (resume.py)

This is the core behavior change. New priority: (1) explicit `--config`, (2) auto-reload from `config_path` if file exists, (3) cached `config_snapshot` fallback, (4) error.

**Files:**
- Modify: `src/mozart/cli/commands/resume.py`
- Test: `tests/test_cli_run_resume.py`

**Step 1: Write failing tests for new priority order**

In `tests/test_cli_run_resume.py`, rewrite `TestReconstructConfig` to test the new behavior. Replace the existing class entirely:

```python
class TestReconstructConfig:
    """Tests for _reconstruct_config auto-reload priority fallback."""

    def test_priority_1_explicit_config_file(self, sample_yaml_config: Path) -> None:
        """Provided --config file should take highest priority."""
        from mozart.cli.commands.resume import _reconstruct_config

        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot={"name": "snapshot-config"},
            config_path="/some/other/path.yaml",
        )
        config, was_reloaded = _reconstruct_config(
            state, config_file=sample_yaml_config, no_reload=False,
        )
        assert config.name == "test-job"
        assert was_reloaded is True

    def test_priority_2_auto_reload_from_config_path(self, sample_yaml_config: Path) -> None:
        """Should auto-reload from stored config_path when file exists."""
        from mozart.cli.commands.resume import _reconstruct_config

        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot={"name": "old-snapshot"},
            config_path=str(sample_yaml_config),
        )
        config, was_reloaded = _reconstruct_config(
            state, config_file=None, no_reload=False,
        )
        assert config.name == "test-job"  # from YAML, not snapshot
        assert was_reloaded is True

    def test_priority_3_snapshot_fallback_when_file_missing(self, tmp_path: Path) -> None:
        """Should fall back to snapshot when config_path file doesn't exist."""
        from mozart.cli.commands.resume import _reconstruct_config

        snapshot = {
            "name": "snapshot-job",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 15},
            "prompt": {"template": "Test {{ sheet_num }}"},
        }
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot=snapshot,
            config_path=str(tmp_path / "deleted.yaml"),
        )
        config, was_reloaded = _reconstruct_config(
            state, config_file=None, no_reload=False,
        )
        assert config.name == "snapshot-job"
        assert was_reloaded is False

    def test_priority_3_snapshot_fallback_when_no_config_path(self) -> None:
        """Should fall back to snapshot when no config_path stored."""
        from mozart.cli.commands.resume import _reconstruct_config

        snapshot = {
            "name": "snapshot-job",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 15},
            "prompt": {"template": "Test {{ sheet_num }}"},
        }
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot=snapshot,
            config_path=None,
        )
        config, was_reloaded = _reconstruct_config(
            state, config_file=None, no_reload=False,
        )
        assert config.name == "snapshot-job"
        assert was_reloaded is False

    def test_no_reload_flag_skips_auto_reload(self, sample_yaml_config: Path) -> None:
        """--no-reload should skip auto-reload and use snapshot."""
        from mozart.cli.commands.resume import _reconstruct_config

        snapshot = {
            "name": "snapshot-job",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 15},
            "prompt": {"template": "Test {{ sheet_num }}"},
        }
        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot=snapshot,
            config_path=str(sample_yaml_config),
        )
        config, was_reloaded = _reconstruct_config(
            state, config_file=None, no_reload=True,
        )
        assert config.name == "snapshot-job"  # snapshot, NOT yaml file
        assert was_reloaded is False

    def test_no_config_available_raises(self) -> None:
        """Should raise typer.Exit when no config source is available."""
        import typer
        from mozart.cli.commands.resume import _reconstruct_config

        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot=None, config_path=None,
        )
        with pytest.raises(typer.Exit):
            _reconstruct_config(state, config_file=None, no_reload=False)

    def test_explicit_config_wins_over_no_reload(self, sample_yaml_config: Path) -> None:
        """--config should win even when --no-reload is set."""
        from mozart.cli.commands.resume import _reconstruct_config

        state = CheckpointState(
            job_id="test", job_name="Test", total_sheets=3,
            config_snapshot={"name": "snapshot"},
        )
        config, was_reloaded = _reconstruct_config(
            state, config_file=sample_yaml_config, no_reload=True,
        )
        assert config.name == "test-job"  # explicit file wins
        assert was_reloaded is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_run_resume.py::TestReconstructConfig -v -x`
Expected: FAIL — `_reconstruct_config` signature doesn't match yet

**Step 3: Rewrite `_reconstruct_config()` in resume.py**

Replace the function (lines 185-282) and update the `ResumeContext` dataclass, CLI flags, and callers:

```python
@dataclass
class ResumeContext:
    """Bundled parameters for direct resume execution."""

    job_id: str
    config_file: Path | None
    workspace: Path | None
    force: bool
    escalation: bool = False
    no_reload: bool = False
    self_healing: bool = False
    auto_confirm: bool = False
```

Remove the `--reload-config` / `-r` flag from the `resume()` function. Add `--no-reload`:

```python
    no_reload: bool = typer.Option(
        False,
        "--no-reload",
        help="Use cached config snapshot instead of auto-reloading from YAML file. "
        "By default, Mozart reloads from the original config path if the file exists.",
    ),
```

Update `_resume_job()` signature: replace `reload_config` param with `no_reload`.

Rewrite `_reconstruct_config()`:

```python
def _reconstruct_config(
    found_state: CheckpointState,
    config_file: Path | None,
    no_reload: bool,
) -> tuple[JobConfig, bool]:
    """Reconstruct JobConfig using priority fallback with auto-reload.

    Priority order:
    1. Provided --config file (always wins)
    2. Auto-reload from stored config_path (default, if file exists)
    3. Cached config_snapshot (fallback when file gone or --no-reload)
    4. Error (nothing available)

    Args:
        found_state: Job checkpoint state with config_snapshot/config_path.
        config_file: Optional explicit config file path.
        no_reload: If True, skip auto-reload and use cached snapshot.

    Returns:
        Tuple of (config, was_reloaded).

    Raises:
        typer.Exit: If no config source available or loading fails.
    """
    # Priority 1: Use provided config file (always takes precedence)
    if config_file:
        try:
            config = JobConfig.from_yaml(config_file)
            console.print(f"[dim]Using config from: {config_file}[/dim]")
            return config, True
        except Exception as e:
            console.print(f"[red]Error loading config file:[/red] {e}")
            raise typer.Exit(1) from None

    # Priority 2: Auto-reload from stored config_path (unless --no-reload)
    if not no_reload and found_state.config_path:
        config_path = Path(found_state.config_path)
        if config_path.exists():
            try:
                config = JobConfig.from_yaml(config_path)
                console.print(
                    f"[cyan]Config reloaded from:[/cyan] {config_path}"
                )
                # Report changes if snapshot exists for comparison
                if found_state.config_snapshot:
                    _report_config_changes(found_state.config_snapshot, config)
                return config, True
            except Exception as e:
                console.print(f"[red]Error reloading config:[/red] {e}")
                raise typer.Exit(1) from None
        else:
            # File doesn't exist — fall through to snapshot
            console.print(
                f"[dim]Config file not found on disk: {config_path}[/dim]"
            )

    # Priority 3: Cached config_snapshot (fallback)
    if found_state.config_snapshot:
        try:
            config = JobConfig.model_validate(found_state.config_snapshot)
            if no_reload:
                console.print("[dim]Using cached config snapshot (--no-reload)[/dim]")
            else:
                console.print("[dim]Using cached config snapshot[/dim]")
            return config, False
        except Exception as e:
            console.print(f"[red]Error reconstructing config from snapshot:[/red] {e}")
            console.print(
                "[dim]Hint: Provide a config file with --config flag.[/dim]"
            )
            raise typer.Exit(1) from None

    console.print(
        "[red]Cannot resume: No config available.[/red]\n"
        "The job state doesn't contain a config snapshot.\n"
        "Please provide a config file with --config flag."
    )
    raise typer.Exit(1)


def _report_config_changes(
    old_snapshot: dict[str, Any],
    new_config: JobConfig,
) -> None:
    """Report config changes between cached snapshot and reloaded config."""
    new_snapshot = new_config.model_dump(mode="json")
    changed_sections: list[str] = []
    for key in set(old_snapshot) | set(new_snapshot):
        if old_snapshot.get(key) != new_snapshot.get(key):
            changed_sections.append(key)
    if changed_sections:
        console.print(
            f"[dim]Config changed: {len(changed_sections)} section(s) updated "
            f"({', '.join(sorted(changed_sections))})[/dim]"
        )
```

Also add `from typing import Any` to imports if not already present.

**Step 4: Update all callers in resume.py**

- `resume()` function: remove `reload_config` param, add `no_reload`, update `asyncio.run()` call
- `_resume_job()`: replace `reload_config` param with `no_reload`
- `_resume_job_direct()`: update `_reconstruct_config()` call to use `ctx.no_reload`

Update docstrings and examples in the `resume()` function:

```python
    """Resume a paused or failed job.

    Loads the job state and continues execution from where it left off.
    By default, Mozart auto-reloads config from the original YAML file
    if it still exists on disk. Use --no-reload to use the cached snapshot.

    Examples:
        mozart resume my-job
        mozart resume my-job --config job.yaml
        mozart resume my-job --no-reload  # Use cached config snapshot
    """
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli_run_resume.py::TestReconstructConfig -v -x`
Expected: All PASS

**Step 6: Run full test suite to check for regressions**

Run: `pytest tests/ -x --timeout=60`
Expected: All pass (or note pre-existing failures)

**Step 7: Commit**

```bash
git add src/mozart/cli/commands/resume.py tests/test_cli_run_resume.py
git commit -m "feat: auto-reload config on resume, remove --reload-config flag

Config is now auto-reloaded from the original YAML path on resume
when the file exists. Cached snapshot is the fallback when the file
is gone. New --no-reload flag for deterministic replay from cache.

The --reload-config / -r flag is removed entirely (dead code)."
```

---

## Task 4: Update `_reconstruct_config()` in Daemon (job_service.py)

Mirror the same auto-reload priority in the daemon's version.

**Files:**
- Modify: `src/mozart/daemon/job_service.py`
- Test: `tests/test_daemon_job_service.py`

**Step 1: Write failing test for daemon auto-reload**

In `tests/test_daemon_job_service.py`, find the existing `test_reload_config_with_no_path_raises` and replace it with auto-reload tests:

```python
def test_auto_reload_from_config_path(self, job_service: JobService, tmp_path: Path):
    """Config should auto-reload from stored config_path when file exists."""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(
        "name: reloaded-job\n"
        "backend:\n  type: claude_cli\n"
        "sheet:\n  size: 3\n  total_items: 9\n"
        "prompt:\n  template: 'Test {{ sheet_num }}'\n"
    )
    mock_state = MagicMock()
    mock_state.config_path = str(config_file)
    mock_state.config_snapshot = {"name": "old-snapshot"}

    result = job_service._reconstruct_config(mock_state)
    assert result.name == "reloaded-job"

def test_snapshot_fallback_when_file_missing(self, job_service: JobService, tmp_path: Path):
    """Should fall back to snapshot when config file doesn't exist."""
    mock_state = MagicMock()
    mock_state.config_path = str(tmp_path / "deleted.yaml")
    mock_state.config_snapshot = {
        "name": "snapshot-job",
        "backend": {"type": "claude_cli"},
        "sheet": {"size": 3, "total_items": 9},
        "prompt": {"template": "Test {{ sheet_num }}"},
    }

    result = job_service._reconstruct_config(mock_state)
    assert result.name == "snapshot-job"
```

**Step 2: Rewrite `_reconstruct_config()` in job_service.py**

Replace lines 818-884 with the new priority:

```python
def _reconstruct_config(
    self,
    state: CheckpointState,
    *,
    config: JobConfig | None = None,
    config_path: Path | None = None,
    no_reload: bool = False,
) -> JobConfig:
    """Reconstruct JobConfig for resume using auto-reload priority.

    Mirrors cli/commands/resume.py::_reconstruct_config() but raises
    exceptions instead of calling typer.Exit().

    Priority order:
    1. Provided config object (explicit override)
    2. Auto-reload from config_path or stored path (default)
    3. Cached config_snapshot (fallback)
    4. Error

    Returns:
        Reconstructed JobConfig.

    Raises:
        JobSubmissionError: If no config source is available.
    """
    from mozart.core.config import JobConfig as JC

    # Priority 1: Explicit config
    if config is not None:
        return config

    # Priority 2: Auto-reload from file (unless no_reload)
    if not no_reload:
        path = config_path or (Path(state.config_path) if state.config_path else None)
        if path and path.exists():
            try:
                return JC.from_yaml(path)
            except Exception as e:
                raise JobSubmissionError(
                    f"Error reloading config from {path}: {e}"
                ) from e

    # Priority 3: Config snapshot from state
    if state.config_snapshot:
        try:
            return JC.model_validate(state.config_snapshot)
        except Exception as e:
            raise JobSubmissionError(
                f"Error reconstructing config from snapshot: {e}"
            ) from e

    raise JobSubmissionError(
        "Cannot resume: no config available. "
        "Provide a config object or ensure state has a config_snapshot."
    )
```

Also update `resume_job()` method: remove `reload_config` parameter, update the call to `_reconstruct_config()`. Change the snapshot update logic (lines 312-317):

```python
# Always update config snapshot when config was reloaded
resolved_config = self._reconstruct_config(
    found_state, config=config, config_path=config_path,
)

# Update snapshot and reset stale cost state
if config is not None or (found_state.config_path and Path(found_state.config_path).exists()):
    found_state.config_snapshot = resolved_config.model_dump(mode="json")
    found_state.cost_limit_reached = False
```

**Step 3: Run tests**

Run: `pytest tests/test_daemon_job_service.py -v -x -k reconstruct`
Expected: PASS

**Step 4: Commit**

```bash
git add src/mozart/daemon/job_service.py tests/test_daemon_job_service.py
git commit -m "feat: auto-reload config in daemon job_service, remove reload_config param

Mirrors CLI auto-reload priority in daemon's _reconstruct_config().
Config auto-reloads from stored path when file exists on disk.
Snapshot is fallback when file is gone."
```

---

## Task 5: Update Modify Command (pause.py)

**Files:**
- Modify: `src/mozart/cli/commands/pause.py`

**Step 1: Remove `reload_config=True` from modify's resume call**

In `src/mozart/cli/commands/pause.py`, find the `_resume_job` call around line 652-661. Since auto-reload is now default, `modify` no longer needs to explicitly pass `reload_config=True`. It passes `--config` which is priority 1 and always wins.

Change:

```python
# Before (line 649-661):
        # Call resume with reload_config
        await _resume_job(
            job_id=job_id,
            config_file=config_file,
            workspace=workspace,
            force=False,
            escalation=False,
            reload_config=True,
            self_healing=False,
            auto_confirm=False,
        )

# After:
        # Resume with new config (auto-reload is default)
        await _resume_job(
            job_id=job_id,
            config_file=config_file,
            workspace=workspace,
            force=False,
            escalation=False,
            no_reload=False,
            self_healing=False,
            auto_confirm=False,
        )
```

Also update the hint text around line 676-682 that references `mozart resume {job_id} -r --config`:

```python
# Before:
            console.print(
                f"  [bold]mozart resume {job_id} -r --config {config_file}[/bold]"
            )

# After:
            console.print(
                f"  [bold]mozart resume {job_id} --config {config_file}[/bold]"
            )
```

**Step 2: Run modify-related tests**

Run: `pytest tests/ -v -x -k "modify or pause"`
Expected: PASS

**Step 3: Commit**

```bash
git add src/mozart/cli/commands/pause.py
git commit -m "refactor: update modify command for auto-reload default

Modify no longer passes reload_config=True since auto-reload is
the new default behavior. --config still takes priority 1."
```

---

## Task 6: Update `test_cli.py` Reconstruct Config Tests

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Find and update the reconstruct config tests**

The tests around lines 796-990 in `test_cli.py` test `_reconstruct_config` with the old `reload_config` parameter. Update all occurrences:

- Replace `reload_config=False` with `no_reload=False`
- Replace `reload_config=True` with `no_reload=False` (auto-reload is default now, so the old "reload" tests become normal auto-reload tests)
- Remove tests that specifically tested the `--reload-config` flag behavior
- Add tests for `no_reload=True` (snapshot preference)

Update the test names and docstrings to reflect new semantics. The key behavioral change: what was previously "Priority 2: --reload-config" is now the default "Priority 2: auto-reload".

**Step 2: Run tests**

Run: `pytest tests/test_cli.py -v -x -k "reconstruct"`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: update test_cli.py reconstruct config tests for auto-reload"
```

---

## Task 7: Config Reconciliation Module

**Files:**
- Create: `src/mozart/execution/reconciliation.py`
- Test: `tests/test_reconciliation.py`

**Step 1: Write failing tests for reconciliation**

Create `tests/test_reconciliation.py`:

```python
"""Tests for config reconciliation on reload."""
from __future__ import annotations

import pytest

from mozart.core.checkpoint import CheckpointState
from mozart.core.config import JobConfig


class TestConfigStateMapping:
    """Tests for CONFIG_STATE_MAPPING completeness."""

    def test_mapping_covers_all_config_sections(self) -> None:
        """Every reconcilable JobConfig section must have a mapping entry."""
        from mozart.execution.reconciliation import (
            CONFIG_STATE_MAPPING,
            METADATA_FIELDS,
        )

        config_sections = set(JobConfig.model_fields.keys())
        mapped_sections = set(CONFIG_STATE_MAPPING.keys())
        reconcilable = config_sections - METADATA_FIELDS

        unmapped = reconcilable - mapped_sections
        assert not unmapped, (
            f"Config sections {unmapped} have no CONFIG_STATE_MAPPING entry. "
            "Add an entry to define what checkpoint state should be reset "
            "when this section changes. Use [] if the runner recreates it."
        )

    def test_mapped_fields_exist_on_checkpoint_state(self) -> None:
        """All fields referenced in mapping must exist on CheckpointState."""
        from mozart.execution.reconciliation import CONFIG_STATE_MAPPING

        checkpoint_fields = set(CheckpointState.model_fields.keys())
        for section, fields in CONFIG_STATE_MAPPING.items():
            for field_name in fields:
                assert field_name in checkpoint_fields, (
                    f"CONFIG_STATE_MAPPING['{section}'] references "
                    f"'{field_name}' which doesn't exist on CheckpointState"
                )


class TestReconcileConfig:
    """Tests for reconcile_config() logic."""

    def _make_state(self, **overrides: object) -> CheckpointState:
        defaults = {
            "job_id": "test",
            "job_name": "Test",
            "total_sheets": 3,
        }
        defaults.update(overrides)
        return CheckpointState(**defaults)

    def _make_snapshot(self, **overrides: object) -> dict:
        base = {
            "name": "test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 3, "total_items": 9},
            "prompt": {"template": "Test {{ sheet_num }}"},
        }
        base.update(overrides)
        return base

    def test_no_changes_returns_empty_report(self) -> None:
        """Identical configs should produce empty report."""
        from mozart.execution.reconciliation import reconcile_config

        snapshot = self._make_snapshot()
        config = JobConfig.model_validate(snapshot)
        state = self._make_state(config_snapshot=snapshot)

        report = reconcile_config(state, config)
        assert report.sections_changed == []
        assert report.fields_reset == {}

    def test_cost_limits_change_resets_cost_state(self) -> None:
        """Changing cost_limits should reset cost tracking fields."""
        from mozart.execution.reconciliation import reconcile_config

        old_snapshot = self._make_snapshot(
            cost_limits={"max_cost_per_job": 10.0}
        )
        new_snapshot = self._make_snapshot(
            cost_limits={"max_cost_per_job": 50.0}
        )
        new_config = JobConfig.model_validate(new_snapshot)
        state = self._make_state(
            config_snapshot=old_snapshot,
            total_estimated_cost=8.5,
            cost_limit_reached=True,
        )

        report = reconcile_config(state, new_config)
        assert "cost_limits" in report.sections_changed
        # Verify state was reset
        assert state.total_estimated_cost == 0.0
        assert state.cost_limit_reached is False

    def test_unchanged_section_not_reset(self) -> None:
        """Sections that didn't change should not reset state."""
        from mozart.execution.reconciliation import reconcile_config

        snapshot = self._make_snapshot()
        config = JobConfig.model_validate(snapshot)
        state = self._make_state(
            config_snapshot=snapshot,
            total_estimated_cost=5.0,
            rate_limit_waits=3,
        )

        report = reconcile_config(state, config)
        assert state.total_estimated_cost == 5.0  # untouched
        assert state.rate_limit_waits == 3  # untouched
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reconciliation.py -v -x`
Expected: FAIL — `reconciliation` module doesn't exist yet

**Step 3: Implement reconciliation.py**

Create `src/mozart/execution/reconciliation.py`:

```python
"""Config reconciliation on reload.

When a job config is reloaded (auto or explicit), derived state in the
checkpoint may be stale. This module provides a declarative mapping from
config sections to checkpoint fields, and a reconcile function that resets
stale fields when their source config section changed.

The structural test in test_reconciliation.py ensures every new config
section gets a mapping entry — preventing future staleness bugs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mozart.core.checkpoint import CheckpointState
from mozart.core.config import JobConfig

_logger = logging.getLogger(__name__)

# Config sections that are metadata / non-reconcilable (don't map to
# checkpoint state that needs resetting on change).
METADATA_FIELDS: frozenset[str] = frozenset({
    "name",
    "description",
    "workspace",
    "state_backend",
    "state_path",
    "pause_between_sheets_seconds",
})

# Declarative mapping: config section -> checkpoint fields to reset.
# When a config section changes, the listed checkpoint fields are reset
# to their Pydantic defaults. Empty list means the runner recreates
# the relevant state from scratch (no checkpoint reset needed).
#
# IMPORTANT: Adding a new top-level field to JobConfig requires adding
# an entry here. The structural test enforces this.
CONFIG_STATE_MAPPING: dict[str, list[str]] = {
    # Sections with checkpoint state that must be reset
    "cost_limits": [
        "total_estimated_cost",
        "total_input_tokens",
        "total_output_tokens",
        "cost_limit_reached",
    ],
    "rate_limit": [
        "rate_limit_waits",
        "quota_waits",
    ],
    "circuit_breaker": [
        "circuit_breaker_history",
    ],
    # Sections where runner recreation handles the reset (no checkpoint state)
    "backend": [],
    "sheet": [],
    "prompt": [],
    "retry": [],
    "learning": [],
    "grounding": [],
    "ai_review": [],
    "logging": [],
    "workspace_lifecycle": [],
    "isolation": [],
    "conductor": [],
    "parallel": [],
    "stale_detection": [],
    "checkpoints": [],
    "bridge": [],
    "cross_sheet": [],
    "feedback": [],
    "validations": [],
    "notifications": [],
    "on_success": [],
    "concert": [],
}


@dataclass
class ReconciliationReport:
    """Report of what was reconciled during config reload."""

    sections_changed: list[str] = field(default_factory=list)
    sections_removed: list[str] = field(default_factory=list)
    fields_reset: dict[str, list[str]] = field(default_factory=dict)
    config_diff: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.sections_changed or self.sections_removed)

    def summary(self) -> str:
        """Human-readable summary of changes."""
        if not self.has_changes:
            return "No config changes detected"
        parts: list[str] = []
        if self.sections_changed:
            parts.append(
                f"{len(self.sections_changed)} section(s) changed: "
                f"{', '.join(sorted(self.sections_changed))}"
            )
        if self.sections_removed:
            parts.append(
                f"{len(self.sections_removed)} section(s) removed: "
                f"{', '.join(sorted(self.sections_removed))}"
            )
        reset_count = sum(len(v) for v in self.fields_reset.values())
        if reset_count:
            parts.append(f"{reset_count} checkpoint field(s) reset")
        return "; ".join(parts)


def reconcile_config(
    state: CheckpointState,
    new_config: JobConfig,
) -> ReconciliationReport:
    """Reconcile checkpoint state after config reload.

    Compares the old config snapshot in state with the new config,
    identifies changed sections, and resets stale checkpoint fields
    according to CONFIG_STATE_MAPPING.

    Args:
        state: Current checkpoint state (mutated in place).
        new_config: The newly loaded config.

    Returns:
        ReconciliationReport describing what changed and what was reset.
    """
    report = ReconciliationReport()

    old_snapshot = state.config_snapshot or {}
    new_snapshot = new_config.model_dump(mode="json")

    # Find changed and removed sections
    all_keys = set(old_snapshot) | set(new_snapshot)
    for key in all_keys:
        if key in METADATA_FIELDS:
            continue
        old_val = old_snapshot.get(key)
        new_val = new_snapshot.get(key)
        if old_val != new_val:
            if key not in new_snapshot:
                report.sections_removed.append(key)
            else:
                report.sections_changed.append(key)
            report.config_diff[key] = (old_val, new_val)

    # Reset checkpoint fields for changed/removed sections
    for section in report.sections_changed + report.sections_removed:
        fields_to_reset = CONFIG_STATE_MAPPING.get(section, [])
        if not fields_to_reset:
            continue

        reset_fields: list[str] = []
        for field_name in fields_to_reset:
            if field_name in CheckpointState.model_fields:
                default = CheckpointState.model_fields[field_name].default
                current = getattr(state, field_name, None)
                if current != default:
                    setattr(state, field_name, default)
                    reset_fields.append(field_name)

        if reset_fields:
            report.fields_reset[section] = reset_fields
            _logger.info(
                "reconciliation.fields_reset",
                section=section,
                fields=reset_fields,
            )

    return report
```

**Step 4: Run tests**

Run: `pytest tests/test_reconciliation.py -v -x`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/mozart/execution/reconciliation.py tests/test_reconciliation.py
git commit -m "feat: add config reconciliation module with structural test

CONFIG_STATE_MAPPING declaratively maps config sections to checkpoint
fields that must be reset on change. Structural test ensures every
new JobConfig section gets a mapping entry."
```

---

## Task 8: Wire Reconciliation Into Resume Paths

**Files:**
- Modify: `src/mozart/cli/commands/resume.py`
- Modify: `src/mozart/daemon/job_service.py`

**Step 1: Wire reconciliation into CLI resume**

In `_resume_job_direct()` in `resume.py`, after the `_reconstruct_config()` call and before saving state, add:

```python
    # Phase 2: Reconstruct config
    config, config_was_reloaded = _reconstruct_config(
        found_state, ctx.config_file, ctx.no_reload
    )

    # Reconcile stale state if config was reloaded
    if config_was_reloaded:
        from mozart.execution.reconciliation import reconcile_config

        report = reconcile_config(found_state, config)
        found_state.config_snapshot = config.model_dump(mode="json")
        if report.has_changes:
            console.print(f"[dim]{report.summary()}[/dim]")
```

**Step 2: Wire reconciliation into daemon job_service**

In `resume_job()` in `job_service.py`, after `_reconstruct_config()`, replace the existing snapshot update logic:

```python
    # Reconcile stale state if config was reloaded
    from mozart.execution.reconciliation import reconcile_config

    report = reconcile_config(found_state, resolved_config)
    # Always update snapshot with latest config
    found_state.config_snapshot = resolved_config.model_dump(mode="json")
    if report.has_changes:
        self._output.job_event(runtime_id, "config_reconciled", {
            "sections_changed": report.sections_changed,
            "fields_reset": report.fields_reset,
        })
```

Remove the old `cost_limit_reached = False` line — reconciliation handles it now via `CONFIG_STATE_MAPPING["cost_limits"]`.

**Step 3: Run tests**

Run: `pytest tests/ -x --timeout=60 -k "resume or reconcil or reconstruct"`
Expected: PASS

**Step 4: Commit**

```bash
git add src/mozart/cli/commands/resume.py src/mozart/daemon/job_service.py
git commit -m "feat: wire config reconciliation into resume paths

reconcile_config() now runs automatically on every config reload
in both CLI and daemon resume paths."
```

---

## Task 9: Registry and Manager Metadata Update

**Files:**
- Modify: `src/mozart/daemon/registry.py`
- Modify: `src/mozart/daemon/manager.py`

**Step 1: Add `update_config_metadata()` to JobRegistry**

In `registry.py`, add a method after `update_status()`:

```python
async def update_config_metadata(
    self,
    job_id: str,
    *,
    config_path: str | None = None,
    workspace: str | None = None,
) -> None:
    """Update config-derived metadata for a job.

    Called during config reconciliation to keep registry in sync
    with the reloaded config.
    """
    updates: list[str] = []
    params: list[Any] = []

    if config_path is not None:
        updates.append("config_path = ?")
        params.append(config_path)
    if workspace is not None:
        updates.append("workspace = ?")
        params.append(workspace)

    if not updates:
        return

    params.append(job_id)
    await self._db.execute(
        f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?",
        params,
    )
    await self._db.commit()
```

**Step 2: Add metadata update to JobManager**

In `manager.py`, add a method to `JobManager` that updates `JobMeta` in-memory state:

```python
def update_job_config_metadata(
    self,
    job_id: str,
    *,
    config_path: Path | None = None,
    workspace: Path | None = None,
) -> None:
    """Update config-derived metadata in the in-memory job map."""
    meta = self._jobs.get(job_id)
    if meta is None:
        return
    if config_path is not None:
        meta.config_path = config_path
    if workspace is not None:
        meta.workspace = workspace
```

**Step 3: Run existing tests**

Run: `pytest tests/ -x -k "registry or manager" --timeout=60`
Expected: PASS (additive changes only)

**Step 4: Commit**

```bash
git add src/mozart/daemon/registry.py src/mozart/daemon/manager.py
git commit -m "feat: add config metadata update methods to registry and manager

Enables config reconciliation to update stale workspace/config_path
in both SQLite registry and in-memory JobMeta."
```

---

## Task 10: Documentation Sweep — Remove `--reload-config`

**Files:**
- Modify: `docs/cli-reference.md`
- Modify: `docs/score-writing-guide.md`
- Modify: `CHANGELOG.md`
- Modify: `examples/README.md`
- Modify: `CLAUDE.md`

**Step 1: Update each file**

For each file, search for `reload-config` and `reload_config` and update:

- **`docs/cli-reference.md`**: Remove `--reload-config` from the resume options table. Add `--no-reload` flag. Update the example command.
- **`docs/score-writing-guide.md`**: Remove references to "config caching pitfall" and `--reload-config`. Add note about auto-reload being default.
- **`CHANGELOG.md`**: Update the `--reload-config on resume` entry to reflect auto-reload. Or add a new entry.
- **`examples/README.md`**: Update the resume example from `--reload-config -c updated.yaml` to just `--config updated.yaml`.
- **`CLAUDE.md`**: Update the resume example. Remove `--reload-config` from the modify/resume section. Update the config caching pitfall note.

**Step 2: Commit**

```bash
git add docs/cli-reference.md docs/score-writing-guide.md CHANGELOG.md examples/README.md CLAUDE.md
git commit -m "docs: remove --reload-config references, document auto-reload default"
```

---

## Task 11: Skills Sweep — Remove `--reload-config`

**Files:**
- Modify: `/home/emzi/.claude/skills/mozart-usage.md` (12+ references)
- Modify: `/home/emzi/.claude/skills/mozart-score-authoring.md` (pitfall #17)

**Step 1: Update mozart-usage.md**

This file has 12+ references. For each:
- Remove `--reload-config` from example commands (e.g., `mozart resume my-job --reload-config` becomes `mozart resume my-job`)
- Remove the `-r` short flag from tables
- Update the pitfall about "config changes ignored" — note that auto-reload is now default
- Update the `modify` examples
- Remove the "Option 2: Use `--reload-config`" section
- Update command reference tables

**Step 2: Update mozart-score-authoring.md**

Find pitfall #17 (line ~1012) about "Config changes after first run" and update:
- Old: "Resume uses cached config | Use `mozart resume --reload-config`"
- New: "Resume auto-reloads from YAML | Use `--no-reload` for cached snapshot"

**Step 3: Commit**

```bash
git add /home/emzi/.claude/skills/mozart-usage.md /home/emzi/.claude/skills/mozart-score-authoring.md
git commit -m "docs: update skills to reflect auto-reload default, remove --reload-config"
```

---

## Task 12: Workspace Status Audit Score

**Files:**
- Create: `scores/workspace-status-audit.yaml`

**Step 1: Read the mozart-score-authoring skill for reference**

Invoke the `mozart-score-authoring` skill (or read `/home/emzi/.claude/skills/mozart-score-authoring.md`) to ensure correct score syntax.

**Step 2: Write the fan-out audit score**

Create `scores/workspace-status-audit.yaml` with a fan-out stage that audits all subsystems, plus a synthesis sheet:

The score should:
- Use fan-out with 6 instances (CLI, daemon, runner, state backends, validation, cross-reference grep)
- Each instance examines its subsystem for workspace-resident state writes
- Instance prompts should be specific: "Read every file in src/mozart/cli/commands/ and identify any code path that writes files to the workspace directory for purposes of tracking job status, state, or control"
- Synthesis sheet cross-references all findings
- Validation: `content_contains` on the synthesis output file looking for "VERDICT:" to ensure completion

**Step 3: Validate the score**

Run: `mozart validate scores/workspace-status-audit.yaml`
Expected: Valid (0 errors)

**Step 4: Commit**

```bash
git add scores/workspace-status-audit.yaml
git commit -m "feat: add workspace status audit score for daemon-only verification

Fan-out score examines all code subsystems for workspace-resident
status tracking. Validates that daemon is sole source of truth."
```

---

## Task 13: Final Integration Test

**Step 1: Run the full test suite**

Run: `pytest tests/ -x --timeout=120`
Expected: All pass

**Step 2: Run type checking**

Run: `mypy src/mozart/execution/reconciliation.py src/mozart/cli/commands/resume.py src/mozart/daemon/job_service.py`
Expected: No errors

**Step 3: Run linting**

Run: `ruff check src/mozart/execution/reconciliation.py src/mozart/cli/commands/resume.py src/mozart/daemon/job_service.py`
Expected: Clean

**Step 4: Validate the audit score**

Run: `mozart validate scores/workspace-status-audit.yaml`
Expected: Valid

**Step 5: Final commit if any fixes needed**

Only commit if integration testing revealed issues that needed fixing.
