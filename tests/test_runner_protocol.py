"""Tests for RunnerProtocol structural contract.

Verifies that the RunnerProtocol correctly declares the attributes and
methods that the concrete JobRunner class provides, ensuring the protocol
stays in sync with the implementation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from mozart.backends.base import ExecutionResult
from mozart.core.config import JobConfig
from mozart.execution.runner import JobRunner
from mozart.execution.runner.protocol import RunnerProtocol


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> JobConfig:
    """Create a minimal job configuration for protocol testing."""
    return JobConfig.model_validate({
        "name": "test-protocol-job",
        "description": "Test job for protocol tests",
        "backend": {
            "type": "claude_cli",
            "skip_permissions": True,
            "cli_model": "claude-sonnet-4-20250514",
        },
        "sheet": {
            "size": 10,
            "total_items": 30,
        },
        "prompt": {
            "template": "Process sheet {{ sheet_num }}.",
        },
    })


@pytest.fixture
def runner(sample_config: JobConfig) -> JobRunner:
    """Create a JobRunner instance for protocol verification."""
    mock_backend = AsyncMock()
    mock_backend.execute = AsyncMock(
        return_value=ExecutionResult(
            success=True, stdout="ok", stderr="", exit_code=0, duration_seconds=1.0,
        )
    )
    mock_backend.health_check = AsyncMock(return_value=True)

    mock_state_backend = AsyncMock()
    mock_state_backend.load = AsyncMock(return_value=None)
    mock_state_backend.save = AsyncMock()
    mock_state_backend.list_jobs = AsyncMock(return_value=[])

    return JobRunner(
        config=sample_config,
        backend=mock_backend,
        state_backend=mock_state_backend,
    )


# =============================================================================
# Tests
# =============================================================================


class TestRunnerProtocolStructure:
    """Verify RunnerProtocol is a valid structural type for JobRunner."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """RunnerProtocol must be decorated with @runtime_checkable."""
        assert isinstance(RunnerProtocol, type)
        # runtime_checkable protocols support isinstance checks
        assert hasattr(RunnerProtocol, "__protocol_attrs__") or hasattr(
            RunnerProtocol, "__abstractmethods__"
        )

    def test_jobrunner_satisfies_protocol_isinstance(self, runner: JobRunner) -> None:
        """Concrete JobRunner instance must satisfy RunnerProtocol isinstance check."""
        assert isinstance(runner, RunnerProtocol)

    def test_jobrunner_class_is_not_subclass_of_data_protocol(self) -> None:
        """Data protocols (with non-method attrs) don't support issubclass.

        Verify isinstance works but issubclass raises TypeError as expected.
        """
        with pytest.raises(TypeError, match="non-method members"):
            issubclass(JobRunner, RunnerProtocol)


class TestProtocolAttributes:
    """Verify all protocol-declared attributes exist on the concrete runner."""

    # Protocol attribute groups from protocol.py
    CORE_ATTRS = ["config", "backend", "state_backend", "console"]

    LEARNING_ATTRS = [
        "outcome_store", "escalation_handler",
        "checkpoint_handler", "judgment_client",
    ]

    CALLBACK_ATTRS = [
        "progress_callback", "execution_progress_callback",
        "rate_limit_callback",
    ]

    INFRA_ATTRS = ["prompt_builder", "error_classifier", "preflight_checker"]

    PRIVATE_ATTRS = [
        "_global_learning_store", "_grounding_engine",
        "_current_sheet_patterns", "_applied_pattern_ids",
        "_exploration_pattern_ids", "_exploitation_pattern_ids",
        "_state_lock", "_shutdown_requested", "_current_state",
        "_sheet_times", "_pause_requested", "_paused_at_sheet",
        "_summary", "_run_start_time", "_current_sheet_num",
        "_execution_progress_snapshots",
        "_logger", "_execution_context",
        "_circuit_breaker", "_dependency_dag", "_retry_strategy",
        "_self_healing_enabled", "_self_healing_auto_confirm",
        "_healing_coordinator", "_parallel_executor",
    ]

    ALL_ATTRS = CORE_ATTRS + LEARNING_ATTRS + CALLBACK_ATTRS + INFRA_ATTRS + PRIVATE_ATTRS

    @pytest.mark.parametrize("attr", ALL_ATTRS)
    def test_attribute_exists_on_runner(self, runner: JobRunner, attr: str) -> None:
        """Every protocol-declared attribute must exist on a constructed JobRunner."""
        assert hasattr(runner, attr), (
            f"RunnerProtocol declares '{attr}' but JobRunner instance lacks it"
        )


class TestProtocolMethods:
    """Verify all protocol-declared methods exist and are callable on the runner."""

    SYNC_METHODS = [
        "_install_signal_handlers",
        "_remove_signal_handlers",
        "_signal_handler",
        "_check_pause_signal",
        "_clear_pause_signal",
        "_update_progress",
        "_handle_execution_progress",
        "_classify_execution",
        "_classify_error",
        "_get_retry_delay",
        "_query_relevant_patterns",
        "_finalize_summary",
        "_get_config_summary",
        "_get_next_sheet_dag_aware",
        "_get_completed_sheets",
        "_check_cost_limits",
    ]

    ASYNC_METHODS = [
        "_handle_graceful_shutdown",
        "_interruptible_sleep",
        "_handle_pause_request",
        "_setup_isolation",
        "_cleanup_isolation",
        "_execute_sheet_with_recovery",
        "_try_self_healing",
        "_handle_rate_limit",
        "_poll_broadcast_discoveries",
        "_record_pattern_feedback",
        "_track_cost",
        "_initialize_state",
        "_execute_post_success_hooks",
        "_execute_sequential_mode",
        "_execute_parallel_mode",
        "_synthesize_batch_outputs",
        "_aggregate_to_global_store",
    ]

    ALL_METHODS = SYNC_METHODS + ASYNC_METHODS

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_method_exists_and_callable(self, runner: JobRunner, method: str) -> None:
        """Every protocol-declared method must exist and be callable on JobRunner."""
        assert hasattr(runner, method), (
            f"RunnerProtocol declares method '{method}' but JobRunner lacks it"
        )
        assert callable(getattr(runner, method)), (
            f"RunnerProtocol declares '{method}' as a method but it's not callable"
        )

    @pytest.mark.parametrize("method", ASYNC_METHODS)
    def test_async_methods_are_coroutines(self, runner: JobRunner, method: str) -> None:
        """Protocol async methods must be actual coroutine functions on JobRunner."""
        func = getattr(runner, method)
        assert asyncio.iscoroutinefunction(func), (
            f"RunnerProtocol declares '{method}' as async but JobRunner's implementation is not"
        )


class TestProtocolProperty:
    """Verify protocol properties exist on the runner."""

    def test_dependency_dag_property(self, runner: JobRunner) -> None:
        """The dependency_dag property must be accessible."""
        # Should not raise — may return None for simple configs
        result = runner.dependency_dag
        # It's a property that can return DependencyDAG | None
        assert result is None or hasattr(result, "get_ready_sheets")
