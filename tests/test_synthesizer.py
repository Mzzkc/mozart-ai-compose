"""Tests for Result Synthesizer Pattern (v18 evolution).

Comprehensive tests for the ResultSynthesizer and synthesis integration:
- Unit tests for SynthesisResult and SynthesisConfig
- Integration tests with mock parallel results
- Strategy tests (merge, summarize, pass_through)
- Error handling tests
- CheckpointState synthesis storage tests
- CLI display tests

Note: These tests follow patterns from test_parallel.py for consistency.
"""

import json
from datetime import datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from mozart.core.checkpoint import CheckpointState, JobStatus, SheetState, SheetStatus
from mozart.execution.parallel import ParallelBatchResult
from mozart.execution.synthesizer import (
    ResultSynthesizer,
    SynthesisConfig,
    SynthesisResult,
    SynthesisStrategy,
    synthesize_batch,
)


# =============================================================================
# Fixtures (adapted from test_parallel.py patterns)
# =============================================================================


@pytest.fixture
def mock_state():
    """Create a mock CheckpointState for testing."""
    state = MagicMock(spec=CheckpointState)
    state.job_id = "test-job"
    state.total_sheets = 5
    state.sheets = {}
    state.synthesis_results = {}
    return state


@pytest.fixture
def synthesis_config():
    """Create default synthesis config."""
    return SynthesisConfig()


@pytest.fixture
def sample_outputs():
    """Create sample sheet outputs for testing."""
    return {
        1: "Output from sheet 1\nWith multiple lines",
        2: "Output from sheet 2\nAlso multiline",
        3: "Output from sheet 3\nFinal content here",
    }


@pytest.fixture
def batch_result_success():
    """Create a successful batch result."""
    return ParallelBatchResult(
        sheets=[1, 2, 3],
        completed=[1, 2, 3],
        failed=[],
        skipped=[],
        duration_seconds=10.5,
    )


@pytest.fixture
def batch_result_partial():
    """Create a partial success batch result."""
    return ParallelBatchResult(
        sheets=[1, 2, 3],
        completed=[1, 2],
        failed=[3],
        error_details={3: "Sheet 3 failed"},
        duration_seconds=15.0,
    )


# =============================================================================
# SynthesisStrategy Tests
# =============================================================================


class TestSynthesisStrategy:
    """Tests for SynthesisStrategy enum."""

    def test_strategy_values(self) -> None:
        """Strategy enum has expected values."""
        assert SynthesisStrategy.MERGE.value == "merge"
        assert SynthesisStrategy.SUMMARIZE.value == "summarize"
        assert SynthesisStrategy.PASS_THROUGH.value == "pass_through"

    def test_strategy_from_string(self) -> None:
        """Strategy can be created from string."""
        assert SynthesisStrategy("merge") == SynthesisStrategy.MERGE
        assert SynthesisStrategy("summarize") == SynthesisStrategy.SUMMARIZE
        assert SynthesisStrategy("pass_through") == SynthesisStrategy.PASS_THROUGH


# =============================================================================
# SynthesisConfig Tests
# =============================================================================


class TestSynthesisConfig:
    """Tests for SynthesisConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has expected values."""
        config = SynthesisConfig()

        assert config.strategy == SynthesisStrategy.MERGE
        assert config.include_metadata is True
        assert config.max_content_bytes == 1024 * 1024  # 1MB
        assert config.fail_on_partial is False

    def test_custom_values(self) -> None:
        """Custom config values are stored correctly."""
        config = SynthesisConfig(
            strategy=SynthesisStrategy.SUMMARIZE,
            include_metadata=False,
            max_content_bytes=5000,
            fail_on_partial=True,
        )

        assert config.strategy == SynthesisStrategy.SUMMARIZE
        assert config.include_metadata is False
        assert config.max_content_bytes == 5000
        assert config.fail_on_partial is True


# =============================================================================
# SynthesisResult Tests
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_default_values(self) -> None:
        """Result has expected defaults."""
        result = SynthesisResult(batch_id="test_batch")

        assert result.batch_id == "test_batch"
        assert result.sheets == []
        assert result.strategy == SynthesisStrategy.MERGE
        assert result.status == "pending"
        assert result.sheet_outputs == {}
        assert result.synthesized_content is None
        assert result.error_message is None
        assert result.metadata == {}

    def test_is_complete_property(self) -> None:
        """is_complete returns True for done or failed."""
        result = SynthesisResult(batch_id="test")

        result.status = "pending"
        assert result.is_complete is False

        result.status = "ready"
        assert result.is_complete is False

        result.status = "done"
        assert result.is_complete is True

        result.status = "failed"
        assert result.is_complete is True

    def test_is_success_property(self) -> None:
        """is_success returns True only for done status."""
        result = SynthesisResult(batch_id="test")

        result.status = "pending"
        assert result.is_success is False

        result.status = "done"
        assert result.is_success is True

        result.status = "failed"
        assert result.is_success is False

    def test_to_dict(self) -> None:
        """to_dict serializes all fields correctly."""
        result = SynthesisResult(
            batch_id="batch_123",
            sheets=[1, 2, 3],
            strategy=SynthesisStrategy.SUMMARIZE,
            status="done",
            sheet_outputs={1: "out1", 2: "out2"},
            synthesized_content="merged content",
            metadata={"key": "value"},
        )

        data = result.to_dict()

        assert data["batch_id"] == "batch_123"
        assert data["sheets"] == [1, 2, 3]
        assert data["strategy"] == "summarize"
        assert data["status"] == "done"
        assert data["sheet_outputs"] == {1: "out1", 2: "out2"}
        assert data["synthesized_content"] == "merged content"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data

    def test_from_dict(self) -> None:
        """from_dict deserializes correctly."""
        data = {
            "batch_id": "batch_456",
            "sheets": [4, 5],
            "strategy": "pass_through",
            "status": "ready",
            "created_at": "2025-01-15T10:30:00",
            "completed_at": None,
            "sheet_outputs": {4: "x", 5: "y"},
            "synthesized_content": None,
            "error_message": None,
            "metadata": {},
        }

        result = SynthesisResult.from_dict(data)

        assert result.batch_id == "batch_456"
        assert result.sheets == [4, 5]
        assert result.strategy == SynthesisStrategy.PASS_THROUGH
        assert result.status == "ready"
        assert result.sheet_outputs == {4: "x", 5: "y"}

    def test_roundtrip_serialization(self) -> None:
        """to_dict -> from_dict preserves data."""
        original = SynthesisResult(
            batch_id="roundtrip",
            sheets=[1, 2, 3],
            strategy=SynthesisStrategy.MERGE,
            status="done",
            sheet_outputs={1: "a", 2: "b", 3: "c"},
            synthesized_content="merged",
            metadata={"foo": "bar"},
        )

        data = original.to_dict()
        restored = SynthesisResult.from_dict(data)

        assert restored.batch_id == original.batch_id
        assert restored.sheets == original.sheets
        assert restored.strategy == original.strategy
        assert restored.status == original.status
        assert restored.sheet_outputs == original.sheet_outputs
        assert restored.synthesized_content == original.synthesized_content
        assert restored.metadata == original.metadata


# =============================================================================
# ResultSynthesizer Tests - Preparation
# =============================================================================


class TestResultSynthesizerPrepare:
    """Tests for ResultSynthesizer.prepare_synthesis()."""

    def test_prepare_with_all_completed(self, sample_outputs):
        """Prepare synthesis with all sheets completed."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2, 3],
            failed_sheets=[],
            sheet_outputs=sample_outputs,
        )

        assert result.status == "ready"
        assert result.sheets == [1, 2, 3]
        assert len(result.sheet_outputs) == 3
        assert result.error_message is None

    def test_prepare_with_partial_completion(self, sample_outputs):
        """Prepare synthesis with some sheets failed."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2],
            failed_sheets=[3],
            sheet_outputs={k: v for k, v in sample_outputs.items() if k != 3},
        )

        assert result.status == "ready"
        assert len(result.sheet_outputs) == 2
        assert 3 not in result.sheet_outputs

    def test_prepare_fail_on_partial(self, sample_outputs):
        """Prepare fails if fail_on_partial is True and sheets failed."""
        config = SynthesisConfig(fail_on_partial=True)
        synthesizer = ResultSynthesizer(config)

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2],
            failed_sheets=[3],
            sheet_outputs={k: v for k, v in sample_outputs.items() if k != 3},
        )

        assert result.status == "failed"
        assert "requires all sheets" in result.error_message

    def test_prepare_with_no_outputs(self):
        """Prepare fails with no sheet outputs."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2, 3],
            failed_sheets=[],
            sheet_outputs={},
        )

        assert result.status == "failed"
        assert "No sheet outputs" in result.error_message

    def test_prepare_generates_batch_id(self, sample_outputs):
        """Prepare generates unique batch ID."""
        synthesizer = ResultSynthesizer()

        result1 = synthesizer.prepare_synthesis(
            batch_sheets=[1],
            completed_sheets=[1],
            failed_sheets=[],
            sheet_outputs={1: sample_outputs[1]},
        )

        result2 = synthesizer.prepare_synthesis(
            batch_sheets=[2],
            completed_sheets=[2],
            failed_sheets=[],
            sheet_outputs={2: sample_outputs[2]},
        )

        assert result1.batch_id != result2.batch_id
        assert result1.batch_id.startswith("batch_")
        assert result2.batch_id.startswith("batch_")

    def test_prepare_records_metadata(self, sample_outputs):
        """Prepare records batch metadata."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2],
            failed_sheets=[3],
            sheet_outputs={1: sample_outputs[1], 2: sample_outputs[2]},
        )

        assert result.metadata["batch_size"] == 3
        assert result.metadata["completed_count"] == 2
        assert result.metadata["failed_count"] == 1
        assert result.metadata["outputs_captured"] == 2


# =============================================================================
# ResultSynthesizer Tests - Execution
# =============================================================================


class TestResultSynthesizerExecute:
    """Tests for ResultSynthesizer.execute_synthesis()."""

    def test_execute_merge_strategy(self, sample_outputs):
        """Execute merge strategy combines outputs."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2],
            completed_sheets=[1, 2],
            failed_sheets=[],
            sheet_outputs={1: sample_outputs[1], 2: sample_outputs[2]},
        )

        result = synthesizer.execute_synthesis(result)

        assert result.status == "done"
        assert result.synthesized_content is not None
        assert "Sheet 1" in result.synthesized_content
        assert "Sheet 2" in result.synthesized_content
        assert sample_outputs[1] in result.synthesized_content
        assert sample_outputs[2] in result.synthesized_content

    def test_execute_merge_without_metadata(self, sample_outputs):
        """Execute merge without metadata separators."""
        config = SynthesisConfig(include_metadata=False)
        synthesizer = ResultSynthesizer(config)

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2],
            completed_sheets=[1, 2],
            failed_sheets=[],
            sheet_outputs={1: sample_outputs[1], 2: sample_outputs[2]},
        )

        result = synthesizer.execute_synthesis(result)

        assert result.status == "done"
        assert "--- Sheet" not in result.synthesized_content

    def test_execute_summarize_strategy(self, sample_outputs):
        """Execute summarize strategy creates summary."""
        config = SynthesisConfig(strategy=SynthesisStrategy.SUMMARIZE)
        synthesizer = ResultSynthesizer(config)

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2, 3],
            failed_sheets=[],
            sheet_outputs=sample_outputs,
        )

        result = synthesizer.execute_synthesis(result)

        assert result.status == "done"
        assert "Summary" in result.synthesized_content
        assert "3 sheets" in result.synthesized_content

    def test_execute_pass_through_strategy(self, sample_outputs):
        """Execute pass_through returns JSON."""
        config = SynthesisConfig(strategy=SynthesisStrategy.PASS_THROUGH)
        synthesizer = ResultSynthesizer(config)

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2],
            completed_sheets=[1, 2],
            failed_sheets=[],
            sheet_outputs={1: sample_outputs[1], 2: sample_outputs[2]},
        )

        result = synthesizer.execute_synthesis(result)

        assert result.status == "done"
        # Should be valid JSON
        parsed = json.loads(result.synthesized_content)
        assert "1" in parsed
        assert "2" in parsed

    def test_execute_size_limit_exceeded(self):
        """Execute fails if synthesized content exceeds limit."""
        config = SynthesisConfig(max_content_bytes=50)
        synthesizer = ResultSynthesizer(config)

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1],
            completed_sheets=[1],
            failed_sheets=[],
            sheet_outputs={1: "A" * 100},  # Exceeds 50 byte limit
        )

        result = synthesizer.execute_synthesis(result)

        assert result.status == "failed"
        assert "exceeds limit" in result.error_message

    def test_execute_on_non_ready_status(self, sample_outputs):
        """Execute does nothing if status is not ready."""
        synthesizer = ResultSynthesizer()

        result = SynthesisResult(batch_id="test", status="failed")
        result = synthesizer.execute_synthesis(result)

        assert result.status == "failed"
        assert result.synthesized_content is None

    def test_execute_sets_completed_at(self, sample_outputs):
        """Execute sets completed_at timestamp."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1],
            completed_sheets=[1],
            failed_sheets=[],
            sheet_outputs={1: sample_outputs[1]},
        )

        assert result.completed_at is None

        result = synthesizer.execute_synthesis(result)

        assert result.completed_at is not None
        assert isinstance(result.completed_at, datetime)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestSynthesizeBatch:
    """Tests for synthesize_batch convenience function."""

    def test_synthesize_batch_success(self, sample_outputs):
        """synthesize_batch completes full workflow."""
        result = synthesize_batch(
            batch_sheets=[1, 2, 3],
            completed_sheets=[1, 2, 3],
            failed_sheets=[],
            sheet_outputs=sample_outputs,
        )

        assert result.status == "done"
        assert result.synthesized_content is not None

    def test_synthesize_batch_with_config(self, sample_outputs):
        """synthesize_batch accepts custom config."""
        config = SynthesisConfig(strategy=SynthesisStrategy.PASS_THROUGH)

        result = synthesize_batch(
            batch_sheets=[1],
            completed_sheets=[1],
            failed_sheets=[],
            sheet_outputs={1: sample_outputs[1]},
            config=config,
        )

        assert result.status == "done"
        # Pass-through produces JSON
        parsed = json.loads(result.synthesized_content)
        assert "1" in parsed

    def test_synthesize_batch_empty_outputs(self):
        """synthesize_batch fails with empty outputs."""
        result = synthesize_batch(
            batch_sheets=[1, 2],
            completed_sheets=[1, 2],
            failed_sheets=[],
            sheet_outputs={},
        )

        assert result.status == "failed"


# =============================================================================
# CheckpointState Synthesis Integration Tests
# =============================================================================


class TestCheckpointStateSynthesis:
    """Tests for CheckpointState synthesis methods."""

    def test_add_synthesis(self):
        """add_synthesis stores result correctly."""
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
        )

        result_data = {"batch_id": "batch_1", "status": "done"}
        state.add_synthesis("batch_1", result_data)

        assert "batch_1" in state.synthesis_results
        assert state.synthesis_results["batch_1"]["status"] == "done"

    def test_get_synthesis(self):
        """get_synthesis retrieves result correctly."""
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
        )

        result_data = {"batch_id": "batch_2", "status": "ready"}
        state.synthesis_results["batch_2"] = result_data

        retrieved = state.get_synthesis("batch_2")
        assert retrieved is not None
        assert retrieved["status"] == "ready"

        missing = state.get_synthesis("nonexistent")
        assert missing is None

    def test_clear_synthesis_specific(self):
        """clear_synthesis removes specific batch."""
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
        )

        state.synthesis_results = {
            "batch_1": {"status": "done"},
            "batch_2": {"status": "done"},
        }

        state.clear_synthesis("batch_1")

        assert "batch_1" not in state.synthesis_results
        assert "batch_2" in state.synthesis_results

    def test_clear_synthesis_all(self):
        """clear_synthesis removes all batches."""
        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
        )

        state.synthesis_results = {
            "batch_1": {"status": "done"},
            "batch_2": {"status": "done"},
        }

        state.clear_synthesis()

        assert len(state.synthesis_results) == 0


# =============================================================================
# ParallelBatchResult Synthesis Fields Tests
# =============================================================================


class TestParallelBatchResultSynthesis:
    """Tests for ParallelBatchResult synthesis-related fields."""

    def test_default_synthesis_fields(self):
        """New fields have correct defaults."""
        result = ParallelBatchResult()

        assert result.sheet_outputs == {}
        assert result.synthesis_ready is False

    def test_to_dict_includes_synthesis_fields(self):
        """to_dict includes synthesis fields."""
        result = ParallelBatchResult(
            sheets=[1, 2],
            completed=[1, 2],
            sheet_outputs={1: "out1", 2: "out2"},
            synthesis_ready=True,
        )

        data = result.to_dict()

        assert "sheet_outputs" in data
        assert data["sheet_outputs"] == {1: "out1", 2: "out2"}
        assert data["synthesis_ready"] is True

    def test_synthesis_fields_preserved(self):
        """Synthesis fields can be set and retrieved."""
        result = ParallelBatchResult(
            sheets=[1, 2, 3],
            completed=[1, 2],
            failed=[3],
        )

        result.sheet_outputs = {1: "content1", 2: "content2"}
        result.synthesis_ready = True

        assert len(result.sheet_outputs) == 2
        assert result.synthesis_ready is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestSynthesizerErrorHandling:
    """Tests for synthesizer error handling."""

    def test_empty_batch(self):
        """Empty batch is handled gracefully."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[],
            completed_sheets=[],
            failed_sheets=[],
            sheet_outputs={},
        )

        assert result.status == "failed"

    def test_all_sheets_failed(self):
        """All sheets failed is handled."""
        synthesizer = ResultSynthesizer()

        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2, 3],
            completed_sheets=[],
            failed_sheets=[1, 2, 3],
            sheet_outputs={},
        )

        assert result.status == "failed"

    def test_missing_output_for_completed_sheet(self):
        """Missing output for completed sheet is handled."""
        synthesizer = ResultSynthesizer()

        # Sheet 2 completed but no output provided
        result = synthesizer.prepare_synthesis(
            batch_sheets=[1, 2],
            completed_sheets=[1, 2],
            failed_sheets=[],
            sheet_outputs={1: "output1"},  # Missing output for sheet 2
        )

        # Should still succeed with available outputs
        assert result.status == "ready"
        assert 2 not in result.sheet_outputs


# =============================================================================
# Integration Tests with Mock Runner
# =============================================================================


class TestSynthesizerRunnerIntegration:
    """Tests for synthesizer integration with runner."""

    @pytest.mark.asyncio
    async def test_runner_synthesize_batch_outputs(self):
        """Test _synthesize_batch_outputs method integration."""
        from mozart.execution.runner import JobRunner

        # This is a smoke test to ensure the method exists and can be called
        # Full integration requires backend setup

        # Create minimal mocks
        mock_backend = MagicMock()
        mock_state_backend = MagicMock()
        mock_config = MagicMock()
        mock_config.workspace = "."
        mock_config.parallel = MagicMock()
        mock_config.parallel.enabled = False
        mock_config.isolation = MagicMock()
        mock_config.isolation.enabled = False

        # Verify method exists
        assert hasattr(JobRunner, "_synthesize_batch_outputs")


# =============================================================================
# CLI Display Tests
# =============================================================================


class TestCLISynthesisDisplay:
    """Tests for CLI synthesis display."""

    def test_status_rich_shows_synthesis(self):
        """_output_status_rich includes synthesis section."""
        from mozart.cli import _output_status_rich

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
        )

        state.synthesis_results = {
            "batch_123": {
                "sheets": [1, 2, 3],
                "strategy": "merge",
                "status": "done",
            },
        }

        # Capture output
        with patch("mozart.cli.console") as mock_console:
            _output_status_rich(state)

            # Find synthesis section in calls
            call_args_list = mock_console.print.call_args_list
            synthesis_found = False
            for call in call_args_list:
                if call.args and "Synthesis Results" in str(call.args[0]):
                    synthesis_found = True
                    break

            assert synthesis_found, "Synthesis Results section not found in output"

    def test_status_rich_skips_empty_synthesis(self):
        """_output_status_rich skips synthesis when empty."""
        from mozart.cli import _output_status_rich

        state = CheckpointState(
            job_id="test-job",
            job_name="Test Job",
            total_sheets=5,
        )

        # Empty synthesis_results
        state.synthesis_results = {}

        with patch("mozart.cli.console") as mock_console:
            _output_status_rich(state)

            call_args_list = mock_console.print.call_args_list
            synthesis_found = False
            for call in call_args_list:
                if call.args and "Synthesis Results" in str(call.args[0]):
                    synthesis_found = True
                    break

            assert not synthesis_found, "Synthesis section should be skipped when empty"
