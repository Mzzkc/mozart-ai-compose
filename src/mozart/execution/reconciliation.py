"""Config reconciliation on reload.

When a job config is reloaded (auto or explicit), derived state in the
checkpoint may be stale. This module provides a declarative mapping from
config sections to checkpoint fields, and a reconcile function that resets
stale fields when their source config section changed.

The structural test in test_reconciliation.py ensures every new config
section gets a mapping entry -- preventing future staleness bugs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mozart.core.checkpoint import CheckpointState
from mozart.core.config import JobConfig
from mozart.core.logging import get_logger

_logger = get_logger("execution.reconciliation")

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
    "spec": [
        "spec_corpus_hash",
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

    # Build a fresh instance to read Pydantic defaults reliably
    # (avoids poking at FieldInfo.default_factory internals).
    _defaults = CheckpointState(
        job_id="", job_name="", total_sheets=1,
    )

    # Reset checkpoint fields for changed/removed sections
    for section in report.sections_changed + report.sections_removed:
        fields_to_reset = CONFIG_STATE_MAPPING.get(section, [])
        if not fields_to_reset:
            continue

        reset_fields: list[str] = []
        for field_name in fields_to_reset:
            if field_name in CheckpointState.model_fields:
                default = getattr(_defaults, field_name)
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
