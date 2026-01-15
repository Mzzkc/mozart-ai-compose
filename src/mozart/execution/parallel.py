"""Parallel sheet execution for Mozart jobs (v17 evolution).

This module provides parallel execution of sheets within a Mozart job,
using the dependency DAG to determine which sheets can run concurrently.

Key features:
- DAG-aware parallel execution (only runs sheets whose dependencies are satisfied)
- Configurable max concurrency limit
- Cost budget partitioning across parallel branches
- Fail-fast error handling with clean cancellation
- Progress tracking for parallel sheets

The ParallelExecutor works with the existing JobRunner - it provides
a method to execute a batch of sheets concurrently.
"""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mozart.core.logging import get_logger
from mozart.execution.dag import DependencyDAG

if TYPE_CHECKING:
    from mozart.core.checkpoint import CheckpointState
    from mozart.execution.runner import JobRunner

# Module logger
_logger = get_logger("parallel")


class ParallelExecutionError(Exception):
    """Raised when parallel sheet execution encounters an error.

    Attributes:
        failed_sheet: Sheet number that failed.
        error: The underlying error.
        completed_sheets: Sheets that completed before the failure.
        cancelled_sheets: Sheets that were cancelled due to the failure.
    """

    def __init__(
        self,
        failed_sheet: int,
        error: Exception,
        completed_sheets: list[int] | None = None,
        cancelled_sheets: list[int] | None = None,
    ):
        self.failed_sheet = failed_sheet
        self.error = error
        self.completed_sheets = completed_sheets or []
        self.cancelled_sheets = cancelled_sheets or []
        super().__init__(
            f"Parallel execution failed on sheet {failed_sheet}: {error}"
        )


@dataclass
class ParallelBatchResult:
    """Result of executing a parallel batch of sheets.

    Attributes:
        sheets: List of sheet numbers in this batch.
        completed: Sheet numbers that completed successfully.
        failed: Sheet numbers that failed.
        skipped: Sheet numbers that were skipped (cancelled before starting).
        error_details: Map of sheet_num -> error message for failed sheets.
        duration_seconds: Total time to execute the batch.
        sheet_outputs: Map of sheet_num -> output reference (v18: synthesizer support).
        synthesis_ready: Whether outputs are ready for synthesis (v18: synthesizer support).
    """

    sheets: list[int] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    error_details: dict[int, str] = field(default_factory=dict)
    duration_seconds: float = 0.0
    sheet_outputs: dict[int, str] = field(default_factory=dict)
    synthesis_ready: bool = False

    @property
    def success(self) -> bool:
        """True if all sheets in batch completed successfully."""
        return len(self.failed) == 0 and len(self.skipped) == 0

    @property
    def partial_success(self) -> bool:
        """True if at least one sheet completed successfully."""
        return len(self.completed) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sheets": self.sheets,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "error_details": self.error_details,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "sheet_outputs": self.sheet_outputs,
            "synthesis_ready": self.synthesis_ready,
        }


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel execution.

    Attributes:
        enabled: Whether parallel execution is enabled.
        max_concurrent: Maximum sheets to execute concurrently.
        fail_fast: If True, cancel pending sheets when one fails.
        budget_per_sheet: Cost budget limit per sheet (None = equal share).
    """

    enabled: bool = False
    max_concurrent: int = 3
    fail_fast: bool = True
    budget_per_sheet: float | None = None


class ParallelExecutor:
    """Executes sheets in parallel using asyncio.TaskGroup.

    The executor works with the JobRunner to execute sheets concurrently
    when the dependency DAG allows. It handles:
    - Batching sheets into parallel groups based on DAG
    - Respecting max_concurrent limit
    - Error propagation and cancellation
    - Progress tracking

    Example:
        ```python
        executor = ParallelExecutor(runner, config)
        result = await executor.execute_batch([2, 3])  # Run sheets 2 and 3
        ```

    Attributes:
        runner: The JobRunner that executes individual sheets.
        config: Parallel execution configuration.
        dag: Dependency DAG (from runner).
    """

    def __init__(
        self,
        runner: "JobRunner",
        config: ParallelExecutionConfig,
    ):
        """Initialize parallel executor.

        Args:
            runner: JobRunner to execute individual sheets.
            config: Parallel execution configuration.
        """
        self.runner = runner
        self.config = config
        self._logger = _logger

    @property
    def dag(self) -> DependencyDAG | None:
        """Get the dependency DAG from the runner."""
        return self.runner.dependency_dag

    async def execute_batch(
        self,
        sheets: list[int],
        state: "CheckpointState",
    ) -> ParallelBatchResult:
        """Execute a batch of sheets in parallel.

        Args:
            sheets: List of sheet numbers to execute in parallel.
            state: Current job checkpoint state.

        Returns:
            ParallelBatchResult with success/failure details.

        Note:
            All sheets must have their dependencies satisfied before
            calling this method. The executor does not check dependencies.
        """
        import time

        result = ParallelBatchResult(sheets=sheets.copy())
        start_time = time.monotonic()

        if not sheets:
            return result

        # Limit concurrency
        batch = sheets[: self.config.max_concurrent]
        if len(sheets) > self.config.max_concurrent:
            # Sheets beyond limit are skipped for this batch
            # They'll be picked up in the next iteration
            result.skipped = sheets[self.config.max_concurrent :]

        self._logger.info(
            "parallel.batch_start",
            sheets=batch,
            max_concurrent=self.config.max_concurrent,
            skipped=len(result.skipped),
        )

        try:
            # Use TaskGroup for structured concurrency (Python 3.11+)
            async with asyncio.TaskGroup() as tg:
                tasks: dict[int, asyncio.Task[None]] = {}
                for sheet_num in batch:
                    task = tg.create_task(
                        self._execute_single_sheet(sheet_num, state),
                        name=f"sheet-{sheet_num}",
                    )
                    tasks[sheet_num] = task

            # All tasks completed (TaskGroup waits for all)
            # Check results
            for sheet_num, task in tasks.items():
                exc = task.exception()
                if exc is not None:
                    result.failed.append(sheet_num)
                    result.error_details[sheet_num] = str(exc)
                else:
                    result.completed.append(sheet_num)

        except* Exception as eg:
            # TaskGroup raises ExceptionGroup on failure
            # Extract individual exceptions
            for exc in eg.exceptions:
                # Try to find which sheet failed
                # This is a fallback - normally we catch per-task
                error_msg = str(exc)
                self._logger.error(
                    "parallel.task_exception",
                    error=error_msg,
                )

            # Mark any sheets without results as failed
            for sheet_num in batch:
                if sheet_num not in result.completed and sheet_num not in result.failed:
                    result.failed.append(sheet_num)
                    if sheet_num not in result.error_details:
                        result.error_details[sheet_num] = "Task failed in parallel group"

        result.duration_seconds = time.monotonic() - start_time
        result.completed.sort()
        result.failed.sort()

        self._logger.info(
            "parallel.batch_complete",
            completed=result.completed,
            failed=result.failed,
            duration_seconds=round(result.duration_seconds, 2),
        )

        return result

    async def _execute_single_sheet(
        self,
        sheet_num: int,
        state: "CheckpointState",
    ) -> None:
        """Execute a single sheet within the parallel batch.

        Args:
            sheet_num: Sheet number to execute.
            state: Job checkpoint state.

        Raises:
            Exception: Propagates any exception from sheet execution.
        """
        self._logger.debug(
            "parallel.sheet_start",
            sheet_num=sheet_num,
        )

        try:
            # Use runner's internal method for sheet execution
            # This includes retry logic, validation, etc.
            await self.runner._execute_sheet_with_recovery(state, sheet_num)

            self._logger.debug(
                "parallel.sheet_complete",
                sheet_num=sheet_num,
            )
        except Exception as e:
            self._logger.error(
                "parallel.sheet_failed",
                sheet_num=sheet_num,
                error=str(e),
            )
            raise

    def get_next_parallel_batch(
        self,
        state: "CheckpointState",
    ) -> list[int]:
        """Get the next batch of sheets that can run in parallel.

        Uses the DAG to find sheets whose dependencies are satisfied,
        up to max_concurrent.

        Args:
            state: Current job checkpoint state.

        Returns:
            List of sheet numbers ready for parallel execution.
        """
        if self.dag is None:
            # No DAG - fall back to sequential
            next_sheet = state.get_next_sheet()
            return [next_sheet] if next_sheet is not None else []

        # Get completed sheets
        from mozart.core.checkpoint import SheetStatus

        completed: set[int] = set()
        for sheet_num in range(1, state.total_sheets + 1):
            sheet_state = state.sheets.get(sheet_num)
            if sheet_state and sheet_state.status == SheetStatus.COMPLETED:
                completed.add(sheet_num)

        # Get ready sheets from DAG
        ready = self.dag.get_ready_sheets(completed)

        # Filter out already-completed sheets
        pending_ready = [s for s in ready if s not in completed]

        # Limit to max_concurrent
        batch = pending_ready[: self.config.max_concurrent]

        return batch

    def estimate_parallel_groups(self) -> list[list[int]]:
        """Estimate parallel execution groups for the job.

        Returns the theoretical parallel groups based on the DAG.
        Actual execution may differ due to failures.

        Returns:
            List of lists, where each inner list is a parallel group.
        """
        if self.dag is None:
            return []
        return self.dag.get_parallel_groups()


async def execute_sheets_parallel(
    runner: "JobRunner",
    state: "CheckpointState",
    config: ParallelExecutionConfig,
) -> bool:
    """Execute all remaining sheets using parallel execution.

    This is the main entry point for parallel execution mode.
    It iterates through parallel batches until all sheets complete.

    Args:
        runner: JobRunner instance.
        state: Job checkpoint state.
        config: Parallel execution configuration.

    Returns:
        True if all sheets completed successfully, False otherwise.
    """
    executor = ParallelExecutor(runner, config)

    while True:
        batch = executor.get_next_parallel_batch(state)
        if not batch:
            # No more sheets to execute
            break

        result = await executor.execute_batch(batch, state)

        if not result.success and config.fail_fast:
            # Stop on first failure
            return False

    return True
