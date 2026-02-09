"""Tests for fan-out expansion logic.

Tests cover:
- Pure expansion function (expand_fan_out)
- All dependency expansion patterns (1→N, N→1, N→N, N→M)
- Edge cases (identity, noop, first/last stage, adjacent fan-outs)
- Input validation (bad stage refs, zero counts)
- Config integration (SheetConfig expansion, serialization round-trip)
- Template integration (stage/instance/fan_count rendering)
- Validation condition integration (stage-aware conditions)
- DAG integration (expanded deps create valid DAG, parallel groups)
"""

from pathlib import Path

import pytest

from mozart.core.config import SheetConfig
from mozart.execution.dag import CycleDetectedError, DependencyDAG, build_dag_from_config
from mozart.execution.fan_out import FanOutMetadata, expand_fan_out
from mozart.prompts.templating import PromptBuilder, SheetContext


# ─── Unit tests: expand_fan_out() ────────────────────────────────────────


class TestNoFanOut:
    """When fan_out is empty, expansion is identity."""

    def test_no_fan_out_identity(self):
        result = expand_fan_out(
            total_stages=3,
            fan_out={},
            stage_dependencies={2: [1], 3: [2]},
        )
        assert result.total_sheets == 3
        assert result.total_stages == 3
        # Each stage maps to exactly one sheet
        assert result.stage_sheets == {1: [1], 2: [2], 3: [3]}
        # Metadata: stage == sheet_num, instance == 1, fan_count == 1
        for s in range(1, 4):
            meta = result.sheet_metadata[s]
            assert meta.stage == s
            assert meta.instance == 1
            assert meta.fan_count == 1
        # Dependencies pass through unchanged
        assert result.expanded_dependencies == {2: [1], 3: [2]}


class TestBasicFanOut:
    """Single stage with fan_out creates correct sheets and metadata."""

    def test_basic_fan_out(self):
        result = expand_fan_out(
            total_stages=3,
            fan_out={2: 3},
            stage_dependencies={2: [1], 3: [2]},
        )
        # Stage 1 → sheet 1, Stage 2 → sheets 2,3,4, Stage 3 → sheet 5
        assert result.total_sheets == 5
        assert result.stage_sheets == {1: [1], 2: [2, 3, 4], 3: [5]}

        # Check metadata for fanned stage
        assert result.sheet_metadata[2] == FanOutMetadata(stage=2, instance=1, fan_count=3)
        assert result.sheet_metadata[3] == FanOutMetadata(stage=2, instance=2, fan_count=3)
        assert result.sheet_metadata[4] == FanOutMetadata(stage=2, instance=3, fan_count=3)

        # Non-fanned stages
        assert result.sheet_metadata[1] == FanOutMetadata(stage=1, instance=1, fan_count=1)
        assert result.sheet_metadata[5] == FanOutMetadata(stage=3, instance=1, fan_count=1)


class TestFanOutFromSingle:
    """1→N: each instance depends on the single source."""

    def test_fan_out_from_single(self):
        result = expand_fan_out(
            total_stages=2,
            fan_out={2: 3},
            stage_dependencies={2: [1]},
        )
        # Stage 1 → sheet 1, Stage 2 → sheets 2,3,4
        # Each of 2,3,4 depends on sheet 1
        assert sorted(result.expanded_dependencies[2]) == [1]
        assert sorted(result.expanded_dependencies[3]) == [1]
        assert sorted(result.expanded_dependencies[4]) == [1]


class TestFanIn:
    """N→1: single target depends on ALL instances."""

    def test_fan_in(self):
        result = expand_fan_out(
            total_stages=3,
            fan_out={2: 3},
            stage_dependencies={2: [1], 3: [2]},
        )
        # Stage 3 (sheet 5) depends on stage 2 (sheets 2,3,4) → fan-in
        assert sorted(result.expanded_dependencies[5]) == [2, 3, 4]


class TestInstanceMatched:
    """N→N (same count): instance i depends on instance i."""

    def test_instance_matched(self):
        result = expand_fan_out(
            total_stages=3,
            fan_out={2: 3, 3: 3},
            stage_dependencies={2: [1], 3: [2]},
        )
        # Stage 2 → sheets 2,3,4; Stage 3 → sheets 5,6,7
        # Instance-matched: 5←2, 6←3, 7←4
        assert result.expanded_dependencies[5] == [2]
        assert result.expanded_dependencies[6] == [3]
        assert result.expanded_dependencies[7] == [4]


class TestCrossFanDifferentCounts:
    """N→M (N≠M, both >1): conservative all-to-all."""

    def test_cross_fan_different_counts(self):
        result = expand_fan_out(
            total_stages=2,
            fan_out={1: 2, 2: 3},
            stage_dependencies={2: [1]},
        )
        # Stage 1 → sheets 1,2; Stage 2 → sheets 3,4,5
        # Cross-fan: each of 3,4,5 depends on both 1 and 2
        assert sorted(result.expanded_dependencies[3]) == [1, 2]
        assert sorted(result.expanded_dependencies[4]) == [1, 2]
        assert sorted(result.expanded_dependencies[5]) == [1, 2]


class TestFullWorkflow7Stages:
    """The exact 7-stage issue-fixer-parallel example from the plan."""

    def test_full_workflow_7_stages(self):
        result = expand_fan_out(
            total_stages=7,
            fan_out={2: 3, 4: 3, 5: 3},
            stage_dependencies={
                2: [1],     # investigate depends on survey
                3: [2],     # adversarial waits for ALL investigations (fan-in)
                4: [3],     # execute depends on adversarial
                5: [4],     # review.i depends on execute.i (instance-matched)
                6: [5],     # finalize waits for ALL reviews (fan-in)
                7: [6],     # commit depends on finalize
            },
        )

        # Total: 1 + 3 + 1 + 3 + 3 + 1 + 1 = 13
        assert result.total_sheets == 13
        assert result.total_stages == 7

        # Stage→sheet mapping
        assert result.stage_sheets[1] == [1]
        assert result.stage_sheets[2] == [2, 3, 4]
        assert result.stage_sheets[3] == [5]
        assert result.stage_sheets[4] == [6, 7, 8]
        assert result.stage_sheets[5] == [9, 10, 11]
        assert result.stage_sheets[6] == [12]
        assert result.stage_sheets[7] == [13]

        # Stage 2 (fan-out from 1): each of 2,3,4 depends on 1
        for s in [2, 3, 4]:
            assert result.expanded_dependencies[s] == [1]

        # Stage 3 (fan-in from 2): sheet 5 depends on 2,3,4
        assert sorted(result.expanded_dependencies[5]) == [2, 3, 4]

        # Stage 4 (fan-out from 3): each of 6,7,8 depends on 5
        for s in [6, 7, 8]:
            assert result.expanded_dependencies[s] == [5]

        # Stage 5 (instance-matched from 4): 9←6, 10←7, 11←8
        assert result.expanded_dependencies[9] == [6]
        assert result.expanded_dependencies[10] == [7]
        assert result.expanded_dependencies[11] == [8]

        # Stage 6 (fan-in from 5): sheet 12 depends on 9,10,11
        assert sorted(result.expanded_dependencies[12]) == [9, 10, 11]

        # Stage 7: sheet 13 depends on 12
        assert result.expanded_dependencies[13] == [12]


class TestEdgeCases:
    """Edge cases for fan_out expansion."""

    def test_fan_out_one_is_noop(self):
        """fan_out: {2: 1} produces identity for that stage."""
        result = expand_fan_out(
            total_stages=3,
            fan_out={2: 1},
            stage_dependencies={2: [1], 3: [2]},
        )
        assert result.total_sheets == 3
        assert result.stage_sheets == {1: [1], 2: [2], 3: [3]}
        meta = result.sheet_metadata[2]
        assert meta.stage == 2
        assert meta.instance == 1
        assert meta.fan_count == 1

    def test_fan_out_first_stage(self):
        """Fan-out on stage 1 with no deps — all instances start immediately."""
        result = expand_fan_out(
            total_stages=2,
            fan_out={1: 3},
            stage_dependencies={2: [1]},
        )
        assert result.total_sheets == 4
        assert result.stage_sheets[1] == [1, 2, 3]
        # Sheets 1,2,3 have no dependencies
        assert 1 not in result.expanded_dependencies
        assert 2 not in result.expanded_dependencies
        assert 3 not in result.expanded_dependencies
        # Sheet 4 (stage 2) depends on all of stage 1
        assert sorted(result.expanded_dependencies[4]) == [1, 2, 3]

    def test_fan_out_last_stage(self):
        """Fan-out on final stage — valid, no convergence needed."""
        result = expand_fan_out(
            total_stages=2,
            fan_out={2: 3},
            stage_dependencies={2: [1]},
        )
        assert result.total_sheets == 4
        assert result.stage_sheets[2] == [2, 3, 4]
        # Each of 2,3,4 depends on 1
        for s in [2, 3, 4]:
            assert result.expanded_dependencies[s] == [1]

    def test_multiple_fan_outs_adjacent(self):
        """Consecutive fanned stages — tests cumulative sheet numbering."""
        result = expand_fan_out(
            total_stages=3,
            fan_out={1: 2, 2: 2, 3: 2},
            stage_dependencies={2: [1], 3: [2]},
        )
        # 2 + 2 + 2 = 6 sheets
        assert result.total_sheets == 6
        assert result.stage_sheets == {1: [1, 2], 2: [3, 4], 3: [5, 6]}
        # Instance-matched: 3←1, 4←2, 5←3, 6←4
        assert result.expanded_dependencies[3] == [1]
        assert result.expanded_dependencies[4] == [2]
        assert result.expanded_dependencies[5] == [3]
        assert result.expanded_dependencies[6] == [4]

    def test_no_dependencies(self):
        """Fan-out with no deps — all sheets start immediately."""
        result = expand_fan_out(
            total_stages=3,
            fan_out={2: 3},
            stage_dependencies={},
        )
        assert result.total_sheets == 5
        assert result.expanded_dependencies == {}

    def test_stage_with_multiple_deps(self):
        """A stage depending on multiple other stages."""
        result = expand_fan_out(
            total_stages=3,
            fan_out={1: 2, 2: 2},
            stage_dependencies={3: [1, 2]},
        )
        # Stage 1 → sheets 1,2; Stage 2 → sheets 3,4; Stage 3 → sheet 5
        # Fan-in from both: sheet 5 depends on all of stage 1 and stage 2
        assert sorted(result.expanded_dependencies[5]) == [1, 2, 3, 4]


class TestInputValidation:
    """Invalid inputs raise ValueError."""

    def test_invalid_stage_reference_in_fan_out(self):
        with pytest.raises(ValueError, match="fan_out references stage 5"):
            expand_fan_out(
                total_stages=3,
                fan_out={5: 3},
                stage_dependencies={},
            )

    def test_invalid_count_zero(self):
        with pytest.raises(ValueError, match="must be >= 1, got 0"):
            expand_fan_out(
                total_stages=3,
                fan_out={2: 0},
                stage_dependencies={},
            )

    def test_invalid_count_negative(self):
        with pytest.raises(ValueError, match="must be >= 1, got -1"):
            expand_fan_out(
                total_stages=3,
                fan_out={2: -1},
                stage_dependencies={},
            )

    def test_invalid_dependency_stage(self):
        with pytest.raises(ValueError, match="depends on stage 10"):
            expand_fan_out(
                total_stages=3,
                fan_out={},
                stage_dependencies={2: [10]},
            )

    def test_invalid_dependency_source_stage(self):
        with pytest.raises(ValueError, match="dependency references stage 5"):
            expand_fan_out(
                total_stages=3,
                fan_out={},
                stage_dependencies={5: [1]},
            )

    def test_invalid_stage_zero_in_fan_out(self):
        with pytest.raises(ValueError, match="fan_out references stage 0"):
            expand_fan_out(
                total_stages=3,
                fan_out={0: 2},
                stage_dependencies={},
            )


# ─── Config integration tests ───────────────────────────────────────────


class TestSheetConfigFanOut:
    """SheetConfig model_validator expansion tests."""

    def test_sheet_config_expands_fan_out(self):
        """SheetConfig with fan_out expands total_items and dependencies."""
        config = SheetConfig(
            size=1,
            total_items=3,
            fan_out={2: 3},
            dependencies={2: [1], 3: [2]},
        )
        # Expanded: stage 1→sheet 1, stage 2→sheets 2,3,4, stage 3→sheet 5
        assert config.total_sheets == 5
        assert config.total_items == 5
        # fan_out cleared after expansion
        assert config.fan_out == {}
        # Dependencies expanded to sheet-level
        assert sorted(config.dependencies.get(5, [])) == [2, 3, 4]  # fan-in
        for s in [2, 3, 4]:
            assert config.dependencies.get(s) == [1]  # fan-out from 1

    def test_sheet_config_without_fan_out_unchanged(self):
        """No fan_out → identity behavior, backward compatible."""
        config = SheetConfig(
            size=1,
            total_items=3,
            dependencies={2: [1], 3: [2]},
        )
        assert config.total_sheets == 3
        assert config.dependencies == {2: [1], 3: [2]}
        assert config.fan_out == {}
        assert config.fan_out_stage_map is None

    def test_fan_out_with_size_gt_1_rejected(self):
        """size > 1 with fan_out → ValueError."""
        with pytest.raises(ValueError, match="fan_out requires size=1"):
            SheetConfig(
                size=2,
                total_items=3,
                fan_out={2: 3},
                dependencies={},
            )

    def test_fan_out_with_start_item_gt_1_rejected(self):
        """start_item > 1 with fan_out → ValueError."""
        with pytest.raises(ValueError, match="fan_out requires start_item=1"):
            SheetConfig(
                size=1,
                total_items=3,
                start_item=5,
                fan_out={2: 3},
                dependencies={},
            )

    def test_get_fan_out_metadata(self):
        """get_fan_out_metadata returns correct stage/instance/fan_count."""
        config = SheetConfig(
            size=1,
            total_items=3,
            fan_out={2: 3},
            dependencies={2: [1], 3: [2]},
        )
        # Sheet 1 = stage 1, instance 1
        meta1 = config.get_fan_out_metadata(1)
        assert meta1.stage == 1
        assert meta1.instance == 1
        assert meta1.fan_count == 1

        # Sheet 3 = stage 2, instance 2
        meta3 = config.get_fan_out_metadata(3)
        assert meta3.stage == 2
        assert meta3.instance == 2
        assert meta3.fan_count == 3

        # Sheet 5 = stage 3, instance 1
        meta5 = config.get_fan_out_metadata(5)
        assert meta5.stage == 3
        assert meta5.instance == 1
        assert meta5.fan_count == 1

    def test_get_fan_out_metadata_without_fan_out(self):
        """Without fan_out, metadata is identity (stage=sheet_num)."""
        config = SheetConfig(size=1, total_items=3)
        meta = config.get_fan_out_metadata(2)
        assert meta.stage == 2
        assert meta.instance == 1
        assert meta.fan_count == 1

    def test_total_stages_property(self):
        """total_stages returns original stage count after expansion."""
        config = SheetConfig(
            size=1,
            total_items=3,
            fan_out={2: 3},
            dependencies={2: [1], 3: [2]},
        )
        assert config.total_stages == 3  # Original 3 stages
        assert config.total_sheets == 5  # Expanded 5 sheets

    def test_total_stages_without_fan_out(self):
        """Without fan_out, total_stages == total_sheets."""
        config = SheetConfig(size=1, total_items=5)
        assert config.total_stages == 5
        assert config.total_sheets == 5

    def test_config_serialization_round_trip(self):
        """model_dump() → model_validate() preserves state without re-expansion."""
        config = SheetConfig(
            size=1,
            total_items=3,
            fan_out={2: 3},
            dependencies={2: [1], 3: [2]},
        )
        # Serialize (simulates what lifecycle.py does)
        data = config.model_dump(mode="json")

        # Verify serialized state
        assert data["fan_out"] == {}  # Cleared
        assert data["total_items"] == 5  # Expanded
        assert data["fan_out_stage_map"] is not None

        # Deserialize (simulates resume)
        restored = SheetConfig.model_validate(data)
        assert restored.total_sheets == 5
        assert restored.total_items == 5
        assert restored.fan_out == {}
        # Metadata preserved
        meta = restored.get_fan_out_metadata(3)
        assert meta.stage == 2
        assert meta.instance == 2
        assert meta.fan_count == 3
        # total_stages preserved
        assert restored.total_stages == 3

    def test_fan_out_field_validator_rejects_zero_count(self):
        """Field validator catches count < 1 before model_validator runs."""
        with pytest.raises(ValueError, match="must be >= 1"):
            SheetConfig(
                size=1,
                total_items=3,
                fan_out={2: 0},
            )

    def test_fan_out_field_validator_rejects_negative_stage(self):
        """Field validator catches stage < 1."""
        with pytest.raises(ValueError, match="positive integer"):
            SheetConfig(
                size=1,
                total_items=3,
                fan_out={-1: 2},
            )


# ─── Template integration tests ─────────────────────────────────────────


class TestTemplateIntegration:
    """Fan-out variables in Jinja templates."""

    def test_stage_instance_fan_count_in_template(self):
        """Jinja template renders stage, instance, fan_count correctly."""
        from mozart.core.config import PromptConfig

        prompt_config = PromptConfig(
            template=(
                "Stage {{ stage }} instance {{ instance }}/{{ fan_count }} "
                "of {{ total_stages }} total stages"
            ),
        )
        builder = PromptBuilder(prompt_config)
        context = SheetContext(
            sheet_num=3,
            total_sheets=5,
            start_item=3,
            end_item=3,
            workspace=Path("/tmp"),
            stage=2,
            instance=2,
            fan_count=3,
            total_stages=3,
        )
        prompt = builder.build_sheet_prompt(context)
        assert "Stage 2 instance 2/3 of 3 total stages" == prompt

    def test_backwards_compat_sheet_num_still_works(self):
        """{{ sheet_num }} still returns concrete expanded number."""
        from mozart.core.config import PromptConfig

        prompt_config = PromptConfig(
            template="Sheet {{ sheet_num }} of {{ total_sheets }}",
        )
        builder = PromptBuilder(prompt_config)
        context = SheetContext(
            sheet_num=3,
            total_sheets=5,
            start_item=3,
            end_item=3,
            workspace=Path("/tmp"),
            stage=2,
            instance=2,
            fan_count=3,
        )
        prompt = builder.build_sheet_prompt(context)
        assert "Sheet 3 of 5" == prompt

    def test_stage_defaults_to_sheet_num_without_fan_out(self):
        """When no fan_out, {{ stage }} == {{ sheet_num }}."""
        from mozart.core.config import PromptConfig

        prompt_config = PromptConfig(
            template="stage={{ stage }} sheet={{ sheet_num }}",
        )
        builder = PromptBuilder(prompt_config)
        context = SheetContext(
            sheet_num=2,
            total_sheets=3,
            start_item=2,
            end_item=2,
            workspace=Path("/tmp"),
            # stage=0 triggers fallback to sheet_num
        )
        prompt = builder.build_sheet_prompt(context)
        assert "stage=2 sheet=2" == prompt

    def test_total_stages_defaults_to_total_sheets_without_fan_out(self):
        """When no fan_out, {{ total_stages }} == {{ total_sheets }}."""
        from mozart.core.config import PromptConfig

        prompt_config = PromptConfig(
            template="stages={{ total_stages }} sheets={{ total_sheets }}",
        )
        builder = PromptBuilder(prompt_config)
        context = SheetContext(
            sheet_num=1,
            total_sheets=5,
            start_item=1,
            end_item=1,
            workspace=Path("/tmp"),
            # total_stages=0 triggers fallback to total_sheets
        )
        prompt = builder.build_sheet_prompt(context)
        assert "stages=5 sheets=5" == prompt

    def test_conditional_template_with_stage(self):
        """Jinja {% if stage == N %} branching works."""
        from mozart.core.config import PromptConfig

        prompt_config = PromptConfig(
            template=(
                "{% if stage == 1 %}Survey{% elif stage == 2 %}"
                "Investigate {{ instance }}{% else %}Review{% endif %}"
            ),
        )
        builder = PromptBuilder(prompt_config)

        # Stage 1
        ctx1 = SheetContext(
            sheet_num=1, total_sheets=5, start_item=1, end_item=1,
            workspace=Path("/tmp"), stage=1, instance=1, fan_count=1,
        )
        assert builder.build_sheet_prompt(ctx1) == "Survey"

        # Stage 2, instance 2
        ctx2 = SheetContext(
            sheet_num=3, total_sheets=5, start_item=3, end_item=3,
            workspace=Path("/tmp"), stage=2, instance=2, fan_count=3,
        )
        assert builder.build_sheet_prompt(ctx2) == "Investigate 2"

        # Stage 3
        ctx3 = SheetContext(
            sheet_num=5, total_sheets=5, start_item=5, end_item=5,
            workspace=Path("/tmp"), stage=3, instance=1, fan_count=1,
        )
        assert builder.build_sheet_prompt(ctx3) == "Review"


# ─── Validation condition tests ──────────────────────────────────────────


class TestValidationConditions:
    """Validation conditions using fan-out variables."""

    def _check(self, condition: str, **context_vars: int) -> bool:
        """Helper to test _check_condition with a sheet_context dict."""
        from mozart.execution.validation import ValidationEngine

        sheet_context = {"sheet_num": context_vars.get("sheet_num", 1)}
        sheet_context.update(context_vars)
        engine = ValidationEngine.__new__(ValidationEngine)
        engine.sheet_context = sheet_context
        return engine._check_condition(condition)

    def test_condition_stage_equals(self):
        assert self._check("stage == 2", stage=2) is True
        assert self._check("stage == 2", stage=1) is False

    def test_condition_instance_gte(self):
        assert self._check("instance >= 2", instance=2) is True
        assert self._check("instance >= 2", instance=3) is True
        assert self._check("instance >= 2", instance=1) is False

    def test_condition_fan_count(self):
        assert self._check("fan_count == 3", fan_count=3) is True
        assert self._check("fan_count == 3", fan_count=1) is False

    def test_condition_compound(self):
        assert self._check(
            "stage == 2 and instance == 1", stage=2, instance=1
        ) is True
        assert self._check(
            "stage == 2 and instance == 1", stage=2, instance=2
        ) is False
        assert self._check(
            "stage == 2 and instance == 1", stage=1, instance=1
        ) is False


# ─── DAG integration tests ──────────────────────────────────────────────


class TestDAGIntegration:
    """Expanded dependencies work with DependencyDAG."""

    def test_expanded_deps_create_valid_dag(self):
        """Expanded dependencies pass DependencyDAG validation."""
        config = SheetConfig(
            size=1,
            total_items=3,
            fan_out={2: 3},
            dependencies={2: [1], 3: [2]},
        )
        dag = build_dag_from_config(
            total_sheets=config.total_sheets,
            sheet_dependencies=config.dependencies,
        )
        assert dag.validated is True
        order = dag.get_execution_order()
        assert len(order) == 5

    def test_parallel_groups_match_expected(self):
        """get_parallel_groups() returns correct groups for issue-fixer."""
        config = SheetConfig(
            size=1,
            total_items=7,
            fan_out={2: 3, 4: 3, 5: 3},
            dependencies={
                2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6],
            },
        )
        dag = build_dag_from_config(
            total_sheets=config.total_sheets,
            sheet_dependencies=config.dependencies,
        )
        groups = dag.get_parallel_groups()
        assert groups == [
            [1],           # survey
            [2, 3, 4],     # investigate x3
            [5],           # adversarial
            [6, 7, 8],     # execute x3
            [9, 10, 11],   # review x3
            [12],          # finalize
            [13],          # commit
        ]

    def test_cycle_in_stage_deps_caught_by_dag(self):
        """Cyclic stage dependencies → DAG cycle detection catches it."""
        # Create expanded deps with a cycle: stage 2 depends on 3, stage 3 depends on 2
        # expand_fan_out doesn't check cycles (that's DAG's job)
        expansion = expand_fan_out(
            total_stages=3,
            fan_out={},
            stage_dependencies={2: [3], 3: [2]},
        )
        with pytest.raises(CycleDetectedError):
            DependencyDAG.from_dependencies(
                total_sheets=expansion.total_sheets,
                dependencies=expansion.expanded_dependencies,
            )

    def test_fan_out_dag_ready_sheets(self):
        """DAG.get_ready_sheets works with expanded fan-out deps."""
        config = SheetConfig(
            size=1,
            total_items=3,
            fan_out={2: 3},
            dependencies={2: [1], 3: [2]},
        )
        dag = build_dag_from_config(
            total_sheets=config.total_sheets,
            sheet_dependencies=config.dependencies,
        )
        # Initially only sheet 1 is ready
        assert dag.get_ready_sheets(set()) == [1]
        # After sheet 1, sheets 2,3,4 are ready (fan-out)
        assert dag.get_ready_sheets({1}) == [2, 3, 4]
        # After 2,3,4, sheet 5 is ready (fan-in)
        assert dag.get_ready_sheets({1, 2, 3, 4}) == [5]
        # After all of 2,3 but not 4, sheet 5 is NOT ready
        assert dag.get_ready_sheets({1, 2, 3}) == [4]
