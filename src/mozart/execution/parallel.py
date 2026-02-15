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

from mozart.core.checkpoint import SheetStatus
from mozart.core.logging import get_logger
from mozart.execution.dag import DependencyDAG

if TYPE_CHECKING:
    from mozart.core.checkpoint import CheckpointState
    from mozart.execution.runner import JobRunner
    from mozart.state.base import StateBackend

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


class _LockingStateBackend:
    """Wraps a StateBackend to serialize save() and load() under an asyncio.Lock.

    Used during parallel execution to prevent concurrent sheets from
    interleaving state mutations and corrupting checkpoint data.

    Both save() and load() are locked to prevent TOCTOU races where a
    concurrent sheet reads partially-written state. All other StateBackend
    methods are explicitly delegated for type safety.
    """

    def __init__(self, inner: "StateBackend", lock: asyncio.Lock) -> None:
        self._inner = inner
        self._lock = lock

    async def save(self, state: "CheckpointState") -> None:
        async with self._lock:
            await self._inner.save(state)

    async def load(self, job_id: str = "") -> "CheckpointState | None":
        async with self._lock:
            return await self._inner.load(job_id)

    async def delete(self, job_id: str) -> bool:
        return await self._inner.delete(job_id)

    async def list_jobs(self) -> list["CheckpointState"]:
        return await self._inner.list_jobs()

    async def get_next_sheet(self, job_id: str) -> int | None:
        async with self._lock:
            return await self._inner.get_next_sheet(job_id)

    async def mark_sheet_status(
        self,
        job_id: str,
        sheet_num: int,
        status: SheetStatus,
        error_message: str | None = None,
    ) -> None:
        async with self._lock:
            await self._inner.mark_sheet_status(job_id, sheet_num, status, error_message)

    async def infer_state_from_artifacts(
        self,
        job_id: str,
        workspace: str,
        artifact_pattern: str,
    ) -> int | None:
        return await self._inner.infer_state_from_artifacts(
            job_id, workspace, artifact_pattern
        )

    async def record_execution(
        self,
        job_id: str,
        sheet_num: int,
        attempt_num: int,
        prompt: str | None = None,
        output: str | None = None,
        exit_code: int | None = None,
        duration_seconds: float | None = None,
    ) -> int | None:
        """Delegate execution recording under lock to prevent data loss."""
        async with self._lock:
            return await self._inner.record_execution(
                job_id, sheet_num, attempt_num,
                prompt=prompt, output=output,
                exit_code=exit_code, duration_seconds=duration_seconds,
            )

    async def close(self) -> None:
        """Delegate close to the inner backend to release connections."""
        await self._inner.close()


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
            runner: The JobRunner that executes individual sheets.
            config: Parallel execution configuration.
        """
        self.runner = runner
        self.config = config
        self._logger = _logger
        self._permanently_failed: set[int] = set()

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

        # Install locking wrapper to serialize state saves across concurrent sheets
        original_backend = self.runner.state_backend
        lock = self.runner._state_lock
        self.runner.state_backend = _LockingStateBackend(original_backend, lock)  # type: ignore[assignment]

        tasks: dict[int, asyncio.Task[None]] = {}
        try:
            # Use TaskGroup for structured concurrency (Python 3.11+)
            async with asyncio.TaskGroup() as tg:
                for sheet_num in batch:
                    task = tg.create_task(
                        self._execute_single_sheet(sheet_num, state),
                        name=f"sheet-{sheet_num}",
                    )
                    tasks[sheet_num] = task

            # All tasks completed successfully (TaskGroup raises ExceptionGroup on failure)
            result.completed.extend(tasks.keys())

        except* Exception as eg:
            # TaskGroup raises ExceptionGroup on failure
            # Extract per-task error details from the tasks dict (still accessible)
            for sheet_num, task in tasks.items():
                if task.done() and task.cancelled():
                    result.failed.append(sheet_num)
                    result.error_details[sheet_num] = "Task cancelled"
                elif task.done() and (task_exc := task.exception()) is not None:
                    result.failed.append(sheet_num)
                    # Preserve exception type and message for diagnostics
                    result.error_details[sheet_num] = (
                        f"{type(task_exc).__name__}: {task_exc}"
                    )
                    self._logger.error(
                        "parallel.sheet_failed_in_group",
                        sheet_num=sheet_num,
                        error_type=type(task_exc).__name__,
                        error=str(task_exc),
                    )
                elif task.done():
                    result.completed.append(sheet_num)

            # Log any ungrouped exceptions that couldn't be mapped to tasks
            # Use object identity (id) instead of str() to avoid collisions
            # when different exceptions have identical string representations.
            # Must check `not t.cancelled()` before `t.exception()` — calling
            # .exception() on a cancelled task re-raises CancelledError.
            mapped_error_ids = {
                id(t.exception())
                for t in tasks.values()
                if t.done() and not t.cancelled() and t.exception() is not None
            }
            for exc in eg.exceptions:
                if id(exc) not in mapped_error_ids:
                    self._logger.error(
                        "parallel.unmapped_exception",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )

        finally:
            # Restore original state backend
            self.runner.state_backend = original_backend

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
            # No DAG - fall back to sequential, but skip permanently failed
            # sheets to prevent infinite loops when fail_fast=False (Q001/#37).
            # get_next_sheet() returns FAILED sheets for retry, but in parallel
            # mode those have already exhausted retries and are in _permanently_failed.
            for sheet_num in range(
                state.last_completed_sheet + 1, state.total_sheets + 1
            ):
                if sheet_num in self._permanently_failed:
                    continue
                sheet_state = state.sheets.get(sheet_num)
                if sheet_state is None or sheet_state.status in (
                    SheetStatus.PENDING,
                    SheetStatus.FAILED,
                ):
                    return [sheet_num]
            return []

        # Get completed and skipped sheets by iterating stored state directly.
        # Skipped sheets count as "done" for dependency resolution so
        # downstream sheets aren't blocked by skipped ancestors (GH#71).
        completed: set[int] = {
            num
            for num, s in state.sheets.items()
            if s.status in (SheetStatus.COMPLETED, SheetStatus.SKIPPED)
        }

        # Treat permanently failed sheets as "done" for dependency resolution
        # so downstream sheets aren't blocked forever waiting for them.
        done_for_dag: set[int] = completed | self._permanently_failed

        # Get ready sheets from DAG (uses done_for_dag so downstream
        # sheets of failed deps can be identified, though they'll be
        # filtered out below if their dep truly failed)
        ready = self.dag.get_ready_sheets(done_for_dag)

        # Filter out completed and permanently failed sheets
        pending_ready = [
            s for s in ready
            if s not in completed and s not in self._permanently_failed
        ]

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

