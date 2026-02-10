"""Tests for Parallel Sheet Execution (v17 evolution).

Comprehensive tests for the ParallelExecutor and parallel execution mode:
- Parallel executor construction and configuration
- Batch execution
- DAG integration
- Error handling
- Config schema validation
- Runner integration

Note: These are unit tests that mock the actual sheet execution.
Integration tests would require a full backend setup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mozart.core.checkpoint import CheckpointState, JobStatus, SheetStatus
from mozart.execution.dag import DependencyDAG
from mozart.execution.parallel import (
    ParallelBatchResult,
    ParallelExecutionConfig,
    ParallelExecutionError,
    ParallelExecutor,
    execute_sheets_parallel,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_runner():
    """Create a mock JobRunner for testing."""
    runner = MagicMock()
    runner._execute_sheet_with_recovery = AsyncMock()
    runner._dependency_dag = None
    runner.dependency_dag = None
    return runner


@pytest.fixture
def parallel_config():
    """Create default parallel config."""
    return ParallelExecutionConfig(
        enabled=True,
        max_concurrent=3,
        fail_fast=True,
    )


@pytest.fixture
def mock_state():
    """Create a mock CheckpointState."""
    state = MagicMock(spec=CheckpointState)
    state.total_sheets = 5
    state.sheets = {}
    state.current_sheet = None
    state.status = JobStatus.RUNNING
    state.get_next_sheet = MagicMock(return_value=1)
    return state


@pytest.fixture
def diamond_dag():
    """Create a diamond DAG for testing: 1 -> 2,3 -> 4."""
    return DependencyDAG.from_dependencies(
        total_sheets=4,
        dependencies={2: [1], 3: [1], 4: [2, 3]},
    )


# =============================================================================
# ParallelExecutionConfig Tests
# =============================================================================


class TestParallelExecutionConfig:
    """Tests for ParallelExecutionConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config has parallel disabled."""
        config = ParallelExecutionConfig()

        assert config.enabled is False
        assert config.max_concurrent == 3
        assert config.fail_fast is True
        assert config.budget_per_sheet is None

    def test_custom_values(self) -> None:
        """Custom config values are stored correctly."""
        config = ParallelExecutionConfig(
            enabled=True,
            max_concurrent=5,
            fail_fast=False,
            budget_per_sheet=10.0,
        )

        assert config.enabled is True
        assert config.max_concurrent == 5
        assert config.fail_fast is False
        assert config.budget_per_sheet == 10.0


# =============================================================================
# ParallelBatchResult Tests
# =============================================================================


class TestParallelBatchResult:
    """Tests for ParallelBatchResult dataclass."""

    def test_empty_result(self) -> None:
        """Empty result is considered successful."""
        result = ParallelBatchResult()

        assert result.success is True
        assert result.partial_success is False

    def test_all_completed(self) -> None:
        """All sheets completed is success."""
        result = ParallelBatchResult(
            sheets=[1, 2, 3],
            completed=[1, 2, 3],
        )

        assert result.success is True
        assert result.partial_success is True

    def test_partial_completion(self) -> None:
        """Some sheets failed is not success but partial success."""
        result = ParallelBatchResult(
            sheets=[1, 2, 3],
            completed=[1, 2],
            failed=[3],
        )

        assert result.success is False
        assert result.partial_success is True

    def test_all_failed(self) -> None:
        """All sheets failed is not success or partial success."""
        result = ParallelBatchResult(
            sheets=[1, 2, 3],
            failed=[1, 2, 3],
        )

        assert result.success is False
        assert result.partial_success is False

    def test_to_dict(self) -> None:
        """Result serializes to dictionary."""
        result = ParallelBatchResult(
            sheets=[1, 2],
            completed=[1],
            failed=[2],
            error_details={2: "Test error"},
            duration_seconds=5.5,
        )

        data = result.to_dict()

        assert data["sheets"] == [1, 2]
        assert data["completed"] == [1]
        assert data["failed"] == [2]
        assert data["error_details"] == {2: "Test error"}
        assert data["duration_seconds"] == 5.5
        assert data["success"] is False


# =============================================================================
# ParallelExecutor Tests
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor class."""

    def test_init(self, mock_runner, parallel_config) -> None:
        """Executor initializes correctly."""
        executor = ParallelExecutor(mock_runner, parallel_config)

        assert executor.runner is mock_runner
        assert executor.config is parallel_config

    def test_dag_property(self, mock_runner, parallel_config) -> None:
        """DAG property returns runner's DAG."""
        mock_runner.dependency_dag = DependencyDAG.from_dependencies(
            total_sheets=3,
            dependencies=None,
        )

        executor = ParallelExecutor(mock_runner, parallel_config)

        assert executor.dag is mock_runner.dependency_dag

    def test_estimate_parallel_groups_no_dag(
        self, mock_runner, parallel_config
    ) -> None:
        """No DAG returns empty groups."""
        mock_runner.dependency_dag = None
        executor = ParallelExecutor(mock_runner, parallel_config)

        groups = executor.estimate_parallel_groups()

        assert groups == []

    def test_estimate_parallel_groups_with_dag(
        self, mock_runner, parallel_config, diamond_dag
    ) -> None:
        """With DAG returns correct parallel groups."""
        mock_runner.dependency_dag = diamond_dag
        executor = ParallelExecutor(mock_runner, parallel_config)

        groups = executor.estimate_parallel_groups()

        assert groups == [[1], [2, 3], [4]]


# =============================================================================
# Batch Execution Tests
# =============================================================================


class TestBatchExecution:
    """Tests for parallel batch execution."""

    @pytest.mark.asyncio
    async def test_empty_batch(self, mock_runner, parallel_config, mock_state) -> None:
        """Empty batch returns empty result."""
        executor = ParallelExecutor(mock_runner, parallel_config)

        result = await executor.execute_batch([], mock_state)

        assert result.sheets == []
        assert result.completed == []
        assert result.failed == []

    @pytest.mark.asyncio
    async def test_single_sheet_success(
        self, mock_runner, parallel_config, mock_state
    ) -> None:
        """Single sheet execution succeeds."""
        executor = ParallelExecutor(mock_runner, parallel_config)

        result = await executor.execute_batch([1], mock_state)

        assert result.sheets == [1]
        assert result.completed == [1]
        assert result.failed == []
        assert result.success is True
        mock_runner._execute_sheet_with_recovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_sheets_parallel(
        self, mock_runner, parallel_config, mock_state
    ) -> None:
        """Multiple sheets execute in parallel."""
        executor = ParallelExecutor(mock_runner, parallel_config)

        result = await executor.execute_batch([1, 2, 3], mock_state)

        assert result.sheets == [1, 2, 3]
        assert sorted(result.completed) == [1, 2, 3]
        assert result.failed == []
        assert mock_runner._execute_sheet_with_recovery.call_count == 3

    @pytest.mark.asyncio
    async def test_max_concurrent_limit(
        self, mock_runner, parallel_config, mock_state
    ) -> None:
        """Batch respects max_concurrent limit."""
        parallel_config.max_concurrent = 2
        executor = ParallelExecutor(mock_runner, parallel_config)

        result = await executor.execute_batch([1, 2, 3, 4, 5], mock_state)

        # Only first 2 should run, rest skipped for this batch
        assert len(result.completed) + len(result.skipped) == 5
        assert result.skipped == [3, 4, 5]
        assert mock_runner._execute_sheet_with_recovery.call_count == 2

    @pytest.mark.asyncio
    async def test_sheet_failure(
        self, mock_runner, parallel_config, mock_state
    ) -> None:
        """Failed sheet is recorded in result."""
        # Make sheet 2 fail
        async def mock_execute(state, sheet_num):
            if sheet_num == 2:
                raise RuntimeError("Test failure")

        mock_runner._execute_sheet_with_recovery = AsyncMock(side_effect=mock_execute)
        executor = ParallelExecutor(mock_runner, parallel_config)

        result = await executor.execute_batch([1, 2, 3], mock_state)

        # Note: With TaskGroup, all tasks run to completion even if one fails
        # The failure is captured in the result
        assert 2 in result.failed
        assert result.success is False


# =============================================================================
# Get Next Parallel Batch Tests
# =============================================================================


class TestGetNextParallelBatch:
    """Tests for get_next_parallel_batch method."""

    def test_no_dag_returns_single_sheet(
        self, mock_runner, parallel_config, mock_state
    ) -> None:
        """Without DAG, returns single next sheet."""
        mock_runner.dependency_dag = None
        executor = ParallelExecutor(mock_runner, parallel_config)

        batch = executor.get_next_parallel_batch(mock_state)

        assert batch == [1]  # From mock get_next_sheet

    def test_with_dag_returns_ready_sheets(
        self, mock_runner, parallel_config, diamond_dag
    ) -> None:
        """With DAG, returns all ready sheets."""
        mock_runner.dependency_dag = diamond_dag

        # Create state with no sheets completed
        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 4
        state.sheets = {}

        executor = ParallelExecutor(mock_runner, parallel_config)

        batch = executor.get_next_parallel_batch(state)

        # Only sheet 1 is ready initially (no dependencies)
        assert batch == [1]

    def test_with_dag_after_completion(
        self, mock_runner, parallel_config, diamond_dag
    ) -> None:
        """After root completes, dependents become ready."""
        mock_runner.dependency_dag = diamond_dag

        # Create state with sheet 1 completed
        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 4
        state.sheets = {
            1: MagicMock(status=SheetStatus.COMPLETED),
        }

        executor = ParallelExecutor(mock_runner, parallel_config)

        batch = executor.get_next_parallel_batch(state)

        # Sheets 2 and 3 should be ready (depend only on 1)
        assert batch == [2, 3]

    def test_respects_max_concurrent(
        self, mock_runner, parallel_config
    ) -> None:
        """Batch size limited by max_concurrent."""
        # DAG where all sheets can run in parallel
        mock_runner.dependency_dag = DependencyDAG.from_dependencies(
            total_sheets=10,
            dependencies=None,
        )
        parallel_config.max_concurrent = 3

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 10
        state.sheets = {}

        executor = ParallelExecutor(mock_runner, parallel_config)

        batch = executor.get_next_parallel_batch(state)

        assert len(batch) == 3
        assert batch == [1, 2, 3]


# =============================================================================
# Config Schema Tests
# =============================================================================


class TestParallelConfigSchema:
    """Tests for ParallelConfig in JobConfig schema."""

    def test_job_config_with_parallel_disabled(self) -> None:
        """JobConfig defaults to parallel disabled."""
        from mozart.core.config import JobConfig

        config = JobConfig.from_yaml_string("""
name: test-job
sheet:
  size: 1
  total_items: 3
prompt:
  template: "Test"
""")

        assert config.parallel.enabled is False
        assert config.parallel.max_concurrent == 3

    def test_job_config_with_parallel_enabled(self) -> None:
        """JobConfig accepts parallel config."""
        from mozart.core.config import JobConfig

        config = JobConfig.from_yaml_string("""
name: test-job
sheet:
  size: 1
  total_items: 3
parallel:
  enabled: true
  max_concurrent: 5
  fail_fast: false
prompt:
  template: "Test"
""")

        assert config.parallel.enabled is True
        assert config.parallel.max_concurrent == 5
        assert config.parallel.fail_fast is False

    def test_parallel_config_validation(self) -> None:
        """ParallelConfig validates max_concurrent range."""
        from pydantic import ValidationError

        from mozart.core.config import ParallelConfig

        # Valid range
        config = ParallelConfig(max_concurrent=1)
        assert config.max_concurrent == 1

        config = ParallelConfig(max_concurrent=10)
        assert config.max_concurrent == 10

        # Invalid: 0
        with pytest.raises(ValidationError):
            ParallelConfig(max_concurrent=0)

        # Invalid: > 10
        with pytest.raises(ValidationError):
            ParallelConfig(max_concurrent=11)


# =============================================================================
# Checkpoint State Tests
# =============================================================================


class TestCheckpointStateParallel:
    """Tests for parallel-related checkpoint state fields."""

    def test_default_parallel_fields(self) -> None:
        """Default state has parallel disabled."""
        state = CheckpointState(
            job_id="test",
            job_name="test",
            total_sheets=3,
        )

        assert state.parallel_enabled is False
        assert state.parallel_max_concurrent == 1
        assert state.parallel_batches_executed == 0
        assert state.sheets_in_progress == []

    def test_parallel_fields_serialization(self) -> None:
        """Parallel fields serialize correctly."""
        state = CheckpointState(
            job_id="test",
            job_name="test",
            total_sheets=3,
            parallel_enabled=True,
            parallel_max_concurrent=3,
            parallel_batches_executed=5,
            sheets_in_progress=[2, 3],
        )

        data = state.model_dump()

        assert data["parallel_enabled"] is True
        assert data["parallel_max_concurrent"] == 3
        assert data["parallel_batches_executed"] == 5
        assert data["sheets_in_progress"] == [2, 3]


# =============================================================================
# Runner Integration Tests
# =============================================================================


class TestRunnerParallelIntegration:
    """Tests for parallel execution integration with JobRunner."""

    def test_runner_creates_parallel_executor(self) -> None:
        """Runner creates parallel executor when enabled."""
        from unittest.mock import MagicMock

        from mozart.core.config import JobConfig
        from mozart.execution.runner import JobRunner

        config = JobConfig.from_yaml_string("""
name: test-job
sheet:
  size: 1
  total_items: 3
parallel:
  enabled: true
  max_concurrent: 2
prompt:
  template: "Test"
""")

        backend = MagicMock()
        state_backend = MagicMock()

        runner = JobRunner(config, backend, state_backend)

        assert runner._parallel_executor is not None
        assert runner._parallel_executor.config.enabled is True
        assert runner._parallel_executor.config.max_concurrent == 2

    def test_runner_no_executor_when_disabled(self) -> None:
        """Runner doesn't create executor when parallel disabled."""
        from unittest.mock import MagicMock

        from mozart.core.config import JobConfig
        from mozart.execution.runner import JobRunner

        config = JobConfig.from_yaml_string("""
name: test-job
sheet:
  size: 1
  total_items: 3
parallel:
  enabled: false
prompt:
  template: "Test"
""")

        backend = MagicMock()
        state_backend = MagicMock()

        runner = JobRunner(config, backend, state_backend)

        assert runner._parallel_executor is None


# =============================================================================
# Execute Sheets Parallel Function Tests
# =============================================================================


class TestExecuteSheetsParallel:
    """Behavioral tests for the execute_sheets_parallel entry point.

    These tests verify the iterative batch execution loop, fail-fast
    behavior, and state backend locking/restoration that happens during
    parallel execution.
    """

    @pytest.fixture
    def parallel_runner(self):
        """Create a mock runner with state_backend and _state_lock."""
        runner = MagicMock()
        runner._execute_sheet_with_recovery = AsyncMock()
        runner._state_lock = asyncio.Lock()
        runner.state_backend = MagicMock()
        runner.state_backend.save = AsyncMock()
        runner.state_backend.load = AsyncMock(return_value=None)
        return runner

    @pytest.mark.asyncio
    async def test_all_sheets_complete_returns_true(self, parallel_runner) -> None:
        """execute_sheets_parallel returns True when all sheets complete."""
        dag = DependencyDAG.from_dependencies(
            total_sheets=3,
            dependencies={2: [1], 3: [2]},
        )
        parallel_runner.dependency_dag = dag

        # Build state that tracks completion as sheets run
        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 3
        state.sheets = {}

        # Simulate sheet completion: each call marks the sheet as done
        async def mock_execute(st, sheet_num):
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(enabled=True, max_concurrent=3, fail_fast=True)
        result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is True
        # All 3 sheets should have been executed
        assert parallel_runner._execute_sheet_with_recovery.call_count == 3

    @pytest.mark.asyncio
    async def test_fail_fast_stops_on_failure(self, parallel_runner) -> None:
        """execute_sheets_parallel returns False immediately on fail_fast failure."""
        # All sheets independent (can all run in first batch)
        dag = DependencyDAG.from_dependencies(total_sheets=3, dependencies=None)
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 3
        state.sheets = {}

        # Make sheet 2 fail
        async def mock_execute(st, sheet_num):
            if sheet_num == 2:
                raise RuntimeError("Sheet 2 boom")
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(enabled=True, max_concurrent=3, fail_fast=True)
        result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is False

    @pytest.mark.asyncio
    async def test_fail_fast_disabled_continues_remaining_sheets(
        self, parallel_runner
    ) -> None:
        """Without fail_fast, remaining sheets execute even if one fails.

        When a sheet fails (retries exhausted inside _execute_sheet_with_recovery),
        it's marked permanently failed and not retried. Other sheets continue.
        The overall result is False because not all sheets succeeded.
        """
        # Sheet 2 depends on sheet 1; sheet 3 is independent
        dag = DependencyDAG.from_dependencies(
            total_sheets=3,
            dependencies={2: [1]},
        )
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 3
        state.sheets = {}

        # Sheet 1 permanently fails, sheet 3 succeeds independently
        async def mock_execute(st, sheet_num):
            if sheet_num == 1:
                raise RuntimeError("Permanent failure")
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(
            enabled=True, max_concurrent=3, fail_fast=False
        )
        result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is False
        # Sheet 1 and 3 ran in first batch (independent from DAG perspective),
        # sheet 2 becomes unblocked (dep 1 treated as done) and runs in second batch
        assert state.sheets[3].status == SheetStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_state_backend_restored_after_execution(
        self, parallel_runner
    ) -> None:
        """State backend is restored to original after batch execution."""
        original_backend = parallel_runner.state_backend
        dag = DependencyDAG.from_dependencies(total_sheets=1, dependencies=None)
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 1
        state.sheets = {}

        async def mock_execute(st, sheet_num):
            # During execution, state_backend should be a _LockingStateBackend
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(enabled=True, max_concurrent=3, fail_fast=True)
        await execute_sheets_parallel(parallel_runner, state, config)

        # After execution, original backend should be restored
        assert parallel_runner.state_backend is original_backend

    @pytest.mark.asyncio
    async def test_diamond_dag_executes_in_correct_order(
        self, parallel_runner
    ) -> None:
        """Diamond DAG (1 -> 2,3 -> 4) executes sheets in dependency order."""
        dag = DependencyDAG.from_dependencies(
            total_sheets=4,
            dependencies={2: [1], 3: [1], 4: [2, 3]},
        )
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 4
        state.sheets = {}

        execution_order: list[int] = []

        async def mock_execute(st, sheet_num):
            execution_order.append(sheet_num)
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(enabled=True, max_concurrent=3, fail_fast=True)
        result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is True
        # Sheet 1 must run before 2 and 3; 4 must run after 2 and 3
        assert execution_order.index(1) < execution_order.index(2)
        assert execution_order.index(1) < execution_order.index(3)
        assert execution_order.index(2) < execution_order.index(4)
        assert execution_order.index(3) < execution_order.index(4)
        assert len(execution_order) == 4

    @pytest.mark.asyncio
    async def test_empty_state_returns_immediately(self, parallel_runner) -> None:
        """No sheets to run returns True immediately."""
        dag = DependencyDAG.from_dependencies(total_sheets=2, dependencies=None)
        parallel_runner.dependency_dag = dag

        # All sheets already completed
        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 2
        state.sheets = {
            1: MagicMock(status=SheetStatus.COMPLETED),
            2: MagicMock(status=SheetStatus.COMPLETED),
        }

        config = ParallelExecutionConfig(enabled=True, max_concurrent=3, fail_fast=True)
        result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is True
        parallel_runner._execute_sheet_with_recovery.assert_not_called()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestParallelErrorHandling:
    """Tests for error handling in parallel execution."""

    def test_parallel_execution_error_attributes(self) -> None:
        """ParallelExecutionError captures all details."""
        error = ParallelExecutionError(
            failed_sheet=3,
            error=RuntimeError("Test error"),
            completed_sheets=[1, 2],
            cancelled_sheets=[4, 5],
        )

        assert error.failed_sheet == 3
        assert isinstance(error.error, RuntimeError)
        assert error.completed_sheets == [1, 2]
        assert error.cancelled_sheets == [4, 5]
        assert "sheet 3" in str(error)
        assert "Test error" in str(error)


# =============================================================================
# Edge Cases
# =============================================================================


class TestParallelEdgeCases:
    """Tests for edge cases in parallel execution."""

    def test_single_sheet_job(self) -> None:
        """Single-sheet job works with parallel enabled."""
        from unittest.mock import MagicMock

        from mozart.core.config import JobConfig
        from mozart.execution.runner import JobRunner

        config = JobConfig.from_yaml_string("""
name: test-job
sheet:
  size: 1
  total_items: 1
parallel:
  enabled: true
  max_concurrent: 3
prompt:
  template: "Test"
""")

        backend = MagicMock()
        state_backend = MagicMock()

        runner = JobRunner(config, backend, state_backend)

        # Should work fine - parallel executor handles single-sheet case
        assert runner._parallel_executor is not None

    def test_linear_dependency_chain(
        self, mock_runner, parallel_config
    ) -> None:
        """Linear chain has no parallelism (each batch has one sheet)."""
        mock_runner.dependency_dag = DependencyDAG.from_dependencies(
            total_sheets=5,
            dependencies={2: [1], 3: [2], 4: [3], 5: [4]},
        )

        executor = ParallelExecutor(mock_runner, parallel_config)
        groups = executor.estimate_parallel_groups()

        # Each group should have exactly one sheet
        for group in groups:
            assert len(group) == 1

    def test_all_independent_sheets(
        self, mock_runner, parallel_config
    ) -> None:
        """All independent sheets form one parallel group."""
        mock_runner.dependency_dag = DependencyDAG.from_dependencies(
            total_sheets=5,
            dependencies=None,
        )

        executor = ParallelExecutor(mock_runner, parallel_config)
        groups = executor.estimate_parallel_groups()

        # All sheets in one group
        assert len(groups) == 1
        assert groups[0] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_batch_duration_tracking(
        self, mock_runner, parallel_config, mock_state
    ) -> None:
        """Batch records duration."""
        # Add delay to execution
        async def slow_execute(state, sheet_num):
            await asyncio.sleep(0.01)  # 10ms

        mock_runner._execute_sheet_with_recovery = AsyncMock(side_effect=slow_execute)
        executor = ParallelExecutor(mock_runner, parallel_config)

        result = await executor.execute_batch([1, 2], mock_state)

        # Duration should be > 0
        assert result.duration_seconds > 0


class TestLockingStateBackend:
    """Tests for _LockingStateBackend concurrent access.

    B2-13: Verifies that mark_sheet_status calls are serialized under the lock,
    preventing interleaved state writes during parallel execution.
    """

    @pytest.mark.asyncio
    async def test_concurrent_mark_sheet_status_serialized(self) -> None:
        """Concurrent mark_sheet_status calls should not interleave."""
        from mozart.execution.parallel import _LockingStateBackend

        call_order: list[tuple[str, int]] = []

        inner = AsyncMock()

        async def tracked_mark_sheet_status(
            job_id: str,
            sheet_num: int,
            status: SheetStatus,
            error_message: str | None = None,
        ) -> None:
            call_order.append(("enter", sheet_num))
            # Simulate I/O delay — if not locked, calls would interleave
            await asyncio.sleep(0.01)
            call_order.append(("exit", sheet_num))

        inner.mark_sheet_status = AsyncMock(side_effect=tracked_mark_sheet_status)

        lock = asyncio.Lock()
        locking_backend = _LockingStateBackend(inner, lock)

        # Launch concurrent calls
        await asyncio.gather(
            locking_backend.mark_sheet_status("job", 1, SheetStatus.COMPLETED),
            locking_backend.mark_sheet_status("job", 2, SheetStatus.COMPLETED),
            locking_backend.mark_sheet_status("job", 3, SheetStatus.COMPLETED),
        )

        # Verify calls were serialized: each enter/exit pair should be adjacent
        # (no interleaving of enter-1, enter-2, exit-1, exit-2)
        for i in range(0, len(call_order), 2):
            enter_event = call_order[i]
            exit_event = call_order[i + 1]
            assert enter_event[0] == "enter"
            assert exit_event[0] == "exit"
            assert enter_event[1] == exit_event[1], (
                f"Events interleaved: {call_order}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_save_and_load_serialized(self) -> None:
        """Concurrent save() and load() should not interleave (TOCTOU prevention)."""
        from mozart.execution.parallel import _LockingStateBackend

        held_lock = False
        overlap_detected = False

        inner = AsyncMock()

        async def slow_save(state: "CheckpointState") -> None:
            nonlocal held_lock, overlap_detected
            if held_lock:
                overlap_detected = True
            held_lock = True
            await asyncio.sleep(0.01)
            held_lock = False

        async def slow_load(job_id: str = "") -> "CheckpointState | None":
            nonlocal held_lock, overlap_detected
            if held_lock:
                overlap_detected = True
            held_lock = True
            await asyncio.sleep(0.01)
            held_lock = False
            return None

        inner.save = AsyncMock(side_effect=slow_save)
        inner.load = AsyncMock(side_effect=slow_load)

        lock = asyncio.Lock()
        locking_backend = _LockingStateBackend(inner, lock)

        mock_state = MagicMock(spec=CheckpointState)

        await asyncio.gather(
            locking_backend.save(mock_state),
            locking_backend.load("job"),
            locking_backend.save(mock_state),
        )

        assert not overlap_detected, "Lock should prevent concurrent access"


# =============================================================================
# Infinite Loop Regression Tests
# =============================================================================


class TestParallelNoInfiniteLoop:
    """Regression tests for the infinite-loop bug when fail_fast=False.

    Previously, when a sheet permanently failed (exhausted all retries) with
    fail_fast=False, get_next_parallel_batch() would return the failed sheet
    again indefinitely because it checked only COMPLETED status, not FAILED.
    """

    @pytest.fixture
    def parallel_runner(self):
        """Create a mock runner with state_backend and _state_lock."""
        runner = MagicMock()
        runner._execute_sheet_with_recovery = AsyncMock()
        runner._state_lock = asyncio.Lock()
        runner.state_backend = MagicMock()
        runner.state_backend.save = AsyncMock()
        runner.state_backend.load = AsyncMock(return_value=None)
        return runner

    @pytest.mark.asyncio
    async def test_permanent_failure_terminates_no_infinite_loop(
        self, parallel_runner
    ) -> None:
        """Permanent sheet failure with fail_fast=False must not loop infinitely.

        Sets up a 3-sheet DAG (2 depends on 1, 3 independent).
        Sheet 1 permanently fails. Verifies the executor terminates within
        a bounded number of iterations rather than looping forever.
        """
        dag = DependencyDAG.from_dependencies(
            total_sheets=3,
            dependencies={2: [1]},
        )
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 3
        state.sheets = {}

        execution_count: dict[int, int] = {}

        async def mock_execute(st, sheet_num):
            execution_count[sheet_num] = execution_count.get(sheet_num, 0) + 1
            if sheet_num == 1:
                # Permanent failure — retries exhausted
                raise RuntimeError("Sheet 1 permanently failed")
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(
            enabled=True, max_concurrent=3, fail_fast=False
        )

        result = await execute_sheets_parallel(parallel_runner, state, config)

        # Must terminate (not hang)
        assert result is False

        # Sheet 1 should only be attempted ONCE (not retried by the loop)
        assert execution_count.get(1, 0) == 1, (
            f"Sheet 1 was executed {execution_count.get(1, 0)} times — "
            "infinite loop not prevented!"
        )

        # Sheet 3 should have succeeded (independent of sheet 1)
        assert 3 in state.sheets
        assert state.sheets[3].status == SheetStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_downstream_of_failed_sheet_still_runs(
        self, parallel_runner
    ) -> None:
        """Downstream sheets of a failed dependency still get scheduled.

        When sheet 1 permanently fails, sheet 2 (which depends on 1) should
        still be scheduled because the DAG treats failed sheets as 'done'
        for dependency resolution. The downstream sheet gets a chance to run
        (it may handle the missing dependency gracefully or fail on its own).
        """
        dag = DependencyDAG.from_dependencies(
            total_sheets=3,
            dependencies={2: [1], 3: [2]},
        )
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 3
        state.sheets = {}

        execution_order: list[int] = []

        async def mock_execute(st, sheet_num):
            execution_order.append(sheet_num)
            if sheet_num == 1:
                raise RuntimeError("Sheet 1 permanently failed")
            st.sheets[sheet_num] = MagicMock(status=SheetStatus.COMPLETED)

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(
            enabled=True, max_concurrent=3, fail_fast=False
        )

        result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is False
        # Sheet 1 ran first, then 2 was unblocked, then 3
        assert 1 in execution_order
        assert 2 in execution_order
        assert 3 in execution_order
        # Each sheet only ran once
        assert execution_order.count(1) == 1
        assert execution_order.count(2) == 1
        assert execution_order.count(3) == 1

    def test_get_next_batch_filters_permanently_failed(self) -> None:
        """get_next_parallel_batch excludes permanently failed sheets."""
        runner = MagicMock()
        dag = DependencyDAG.from_dependencies(
            total_sheets=3,
            dependencies=None,
        )
        runner.dependency_dag = dag

        config = ParallelExecutionConfig(enabled=True, max_concurrent=3)
        executor = ParallelExecutor(runner, config)

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 3
        state.sheets = {}

        # Initially all 3 sheets should be ready
        batch = executor.get_next_parallel_batch(state)
        assert batch == [1, 2, 3]

        # Mark sheet 2 as permanently failed
        executor._permanently_failed.add(2)

        # Now only sheets 1 and 3 should be returned
        batch = executor.get_next_parallel_batch(state)
        assert batch == [1, 3]

    @pytest.mark.asyncio
    async def test_bounded_iterations_on_all_failures(
        self, parallel_runner
    ) -> None:
        """When ALL sheets fail, the loop must terminate promptly.

        This is the worst-case scenario for the old bug — every sheet fails,
        so without the fix, every sheet would be retried every iteration.
        """
        dag = DependencyDAG.from_dependencies(
            total_sheets=4,
            dependencies=None,
        )
        parallel_runner.dependency_dag = dag

        state = MagicMock(spec=CheckpointState)
        state.total_sheets = 4
        state.sheets = {}

        batch_count = 0

        async def mock_execute(st, sheet_num):
            nonlocal batch_count
            raise RuntimeError(f"Sheet {sheet_num} always fails")

        parallel_runner._execute_sheet_with_recovery = AsyncMock(
            side_effect=mock_execute
        )

        config = ParallelExecutionConfig(
            enabled=True, max_concurrent=4, fail_fast=False
        )

        # Patch to count loop iterations
        original_get_batch = ParallelExecutor.get_next_parallel_batch

        def counting_get_batch(self_exec, st):
            nonlocal batch_count
            batch_count += 1
            return original_get_batch(self_exec, st)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ParallelExecutor,
                "get_next_parallel_batch",
                counting_get_batch,
            )
            result = await execute_sheets_parallel(parallel_runner, state, config)

        assert result is False
        # Should terminate in exactly 2 iterations:
        # 1st: returns [1,2,3,4], all fail, all marked permanently failed
        # 2nd: returns [], loop breaks
        assert batch_count == 2, (
            f"Expected 2 loop iterations but got {batch_count} — "
            "possible infinite loop!"
        )
