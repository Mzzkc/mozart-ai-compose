"""Fan-out expansion for parameterized sheet instantiation.

Expands stage-level fan_out declarations into concrete sheet numbers at
config parse time. All downstream consumers (DAG, parallel executor, state,
validation) see only expanded int sheet numbers — zero changes needed.

Expansion rules:
    Stage 1 (fan_out: 1) → Sheet 1   (stage=1, instance=1, fan_count=1)
    Stage 2 (fan_out: 3) → Sheet 2   (stage=2, instance=1, fan_count=3)
                         → Sheet 3   (stage=2, instance=2, fan_count=3)
                         → Sheet 4   (stage=2, instance=3, fan_count=3)
    Stage 3 (fan_out: 1) → Sheet 5   (stage=3, instance=1, fan_count=1)

Dependency expansion patterns:
    1→N (fan-out):        Each instance depends on the single source
    N→1 (fan-in):         Single target depends on ALL instances
    N→N (instance-match): Instance i depends on instance i
    N→M (cross-fan):      All-to-all (conservative)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FanOutMetadata:
    """Per-sheet metadata tracking which logical stage and instance it represents.

    Attributes:
        stage: Logical stage number (1-indexed).
        instance: Instance within the fan-out group (1-indexed).
        fan_count: Total instances in this stage's fan-out group.
    """

    stage: int
    instance: int
    fan_count: int


@dataclass(frozen=True)
class FanOutExpansion:
    """Result of expanding fan_out declarations into concrete sheet numbers.

    Attributes:
        total_sheets: Total concrete sheet count after expansion.
        total_stages: Original stage count before expansion.
        sheet_metadata: Map of sheet_num → FanOutMetadata.
        expanded_dependencies: Map of sheet_num → list of prerequisite sheet nums.
        stage_sheets: Map of stage → list of sheet nums belonging to that stage.
    """

    total_sheets: int
    total_stages: int
    sheet_metadata: dict[int, FanOutMetadata] = field(default_factory=dict)
    expanded_dependencies: dict[int, list[int]] = field(default_factory=dict)
    stage_sheets: dict[int, list[int]] = field(default_factory=dict)


def expand_fan_out(
    total_stages: int,
    fan_out: dict[int, int],
    stage_dependencies: dict[int, list[int]],
) -> FanOutExpansion:
    """Expand stage-level fan_out into concrete sheet-level assignments.

    Pure function — no side effects, no imports beyond stdlib.

    Args:
        total_stages: Number of logical stages (1-indexed).
        fan_out: Map of stage → instance count. Stages not listed default to 1.
        stage_dependencies: Map of stage → list of prerequisite stages.

    Returns:
        FanOutExpansion with concrete sheet assignments, metadata, and deps.

    Raises:
        ValueError: If fan_out references invalid stages or has invalid counts.
    """
    _validate_inputs(total_stages, fan_out, stage_dependencies)

    # Phase 1: Assign concrete sheet numbers sequentially
    sheet_metadata: dict[int, FanOutMetadata] = {}
    stage_sheets: dict[int, list[int]] = {}
    current_sheet = 1

    for stage in range(1, total_stages + 1):
        count = fan_out.get(stage, 1)
        sheets_for_stage: list[int] = []

        for instance in range(1, count + 1):
            sheet_metadata[current_sheet] = FanOutMetadata(
                stage=stage,
                instance=instance,
                fan_count=count,
            )
            sheets_for_stage.append(current_sheet)
            current_sheet += 1

        stage_sheets[stage] = sheets_for_stage

    total_sheets = current_sheet - 1

    # Phase 2: Expand stage-level dependencies to sheet-level
    expanded_deps: dict[int, list[int]] = {}

    for stage, dep_stages in stage_dependencies.items():
        target_sheets = stage_sheets[stage]
        target_count = len(target_sheets)

        for dep_stage in dep_stages:
            source_sheets = stage_sheets[dep_stage]
            source_count = len(source_sheets)

            _expand_dependency_pair(
                expanded_deps,
                target_sheets,
                target_count,
                source_sheets,
                source_count,
            )

    return FanOutExpansion(
        total_sheets=total_sheets,
        total_stages=total_stages,
        sheet_metadata=sheet_metadata,
        expanded_dependencies=expanded_deps,
        stage_sheets=stage_sheets,
    )


def _validate_inputs(
    total_stages: int,
    fan_out: dict[int, int],
    stage_dependencies: dict[int, list[int]],
) -> None:
    """Validate fan_out and dependency inputs.

    Raises:
        ValueError: On invalid stage references or counts.
    """
    if total_stages < 1:
        raise ValueError(f"total_stages must be >= 1, got {total_stages}")

    for stage, count in fan_out.items():
        if stage < 1 or stage > total_stages:
            raise ValueError(
                f"fan_out references stage {stage}, "
                f"but total_stages is {total_stages} (valid: 1-{total_stages})"
            )
        if count < 1:
            raise ValueError(
                f"fan_out count for stage {stage} must be >= 1, got {count}"
            )

    for stage, deps in stage_dependencies.items():
        if stage < 1 or stage > total_stages:
            raise ValueError(
                f"dependency references stage {stage}, "
                f"but total_stages is {total_stages} (valid: 1-{total_stages})"
            )
        for dep in deps:
            if dep < 1 or dep > total_stages:
                raise ValueError(
                    f"stage {stage} depends on stage {dep}, "
                    f"but total_stages is {total_stages} (valid: 1-{total_stages})"
                )


def _expand_dependency_pair(
    expanded_deps: dict[int, list[int]],
    target_sheets: list[int],
    target_count: int,
    source_sheets: list[int],
    source_count: int,
) -> None:
    """Expand a single stage→stage dependency into sheet→sheet dependencies.

    Patterns:
        1→N (fan-out from single): Each target depends on the single source.
        N→1 (fan-in to single): Single target depends on all sources.
        N→N (instance-matched): Target[i] depends on source[i].
        N→M (cross-fan, N≠M, both >1): All-to-all (conservative).
    """
    if source_count == 1 and target_count == 1:
        # 1→1: simple
        _add_dep(expanded_deps, target_sheets[0], source_sheets[0])

    elif source_count == 1:
        # 1→N (fan-out): each target instance depends on the single source
        for target in target_sheets:
            _add_dep(expanded_deps, target, source_sheets[0])

    elif target_count == 1:
        # N→1 (fan-in): single target depends on ALL source instances
        for source in source_sheets:
            _add_dep(expanded_deps, target_sheets[0], source)

    elif source_count == target_count:
        # N→N (instance-matched): target[i] depends on source[i]
        for target, source in zip(target_sheets, source_sheets):
            _add_dep(expanded_deps, target, source)

    else:
        # N→M (cross-fan): all-to-all (conservative)
        for target in target_sheets:
            for source in source_sheets:
                _add_dep(expanded_deps, target, source)


def _add_dep(deps: dict[int, list[int]], target: int, source: int) -> None:
    """Add a dependency edge, avoiding duplicates."""
    dep_list = deps.setdefault(target, [])
    if source not in dep_list:
        dep_list.append(source)
