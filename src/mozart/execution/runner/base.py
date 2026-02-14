"""Base initialization mixin for JobRunner.

Contains the core __init__ method, property accessors, signal handling,
and state management infrastructure. This module provides the foundation
upon which other mixins (lifecycle, sheet, patterns, recovery, cost,
isolation) compose the full JobRunner functionality.

Architecture:
    JobRunner uses mixin composition. This module defines JobRunnerBase,
    which is intended to be mixed with other mixin classes in __init__.py
    to create the complete JobRunner class.

    Mixin Order:
        class JobRunner(
            SheetMixin,      # Sheet execution
            LifecycleMixin,  # run() methods
            RecoveryMixin,   # Error recovery
            PatternsMixin,   # Pattern management
            CostMixin,       # Cost tracking
            IsolationMixin,  # Worktree isolation
            JobRunnerBase,   # Initialization (last = MRO first for super())
        ): pass
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from mozart.execution.grounding import GroundingEngine
    from mozart.execution.parallel import ParallelExecutor
    from mozart.healing.coordinator import SelfHealingCoordinator
    from mozart.learning.global_store import GlobalLearningStore

from mozart.backends.base import Backend
from mozart.core.checkpoint import CheckpointState, JobStatus, ProgressSnapshotDict
from mozart.core.config import JobConfig
from mozart.core.errors import ErrorClassifier
from mozart.core.logging import ExecutionContext, MozartLogger, get_logger
from mozart.execution.circuit_breaker import CircuitBreaker
from mozart.execution.dag import (
    CycleDetectedError,
    DependencyDAG,
    InvalidDependencyError,
    build_dag_from_config,
)
from mozart.execution.escalation import ConsoleCheckpointHandler, ConsoleEscalationHandler
from mozart.execution.preflight import PreflightChecker
from mozart.execution.retry_strategy import AdaptiveRetryStrategy, RetryStrategyConfig
from mozart.learning.judgment import JudgmentClient
from mozart.learning.outcomes import OutcomeStore
from mozart.prompts.templating import PromptBuilder
from mozart.state.base import StateBackend

from .models import (
    GracefulShutdownError,
    RunnerContext,
    RunSummary,
)


class JobRunnerBase:
    """Base mixin providing initialization and infrastructure for JobRunner.

    This class contains:
    - __init__ with full parameter handling and context merging
    - Property accessors for internal state
    - Signal handler setup/teardown for graceful shutdown
    - Pause signal detection for job control
    - Progress tracking infrastructure

    Subclasses/mixins add:
    - LifecycleMixin: run(), resume(), finalize_job()
    - SheetMixin: execute_sheet(), validation
    - PatternsMixin: pattern application and feedback
    - RecoveryMixin: error recovery and self-healing
    - CostMixin: token/cost tracking
    - IsolationMixin: worktree isolation

    Dependency Model:
        Three required dependencies (config, backend, state_backend) are
        positional parameters that must always be provided. All optional
        dependencies are grouped into ``RunnerContext``, passed as the
        keyword-only ``context`` parameter.

        When ``context=None`` (the default), the runner initializes all
        optional components to safe defaults: no learning, no callbacks,
        no self-healing, and a default Rich Console. This allows minimal
        construction for testing and simple use cases::

            runner = JobRunner(config, backend, state_backend)

        For production use with learning, progress reporting, and healing::

            ctx = RunnerContext(
                outcome_store=...,
                global_learning_store=...,
                progress_callback=...,
                self_healing_enabled=True,
            )
            runner = JobRunner(config, backend, state_backend, context=ctx)
    """

    def __init__(
        self,
        config: JobConfig,
        backend: Backend,
        state_backend: StateBackend,
        *,
        context: RunnerContext | None = None,
    ) -> None:
        """Initialize job runner.

        Args:
            config: Job configuration.
            backend: Claude execution backend.
            state_backend: State persistence backend.
            context: Optional RunnerContext grouping all optional dependencies.

        Usage:
            # Minimal setup
            runner = JobRunner(config, backend, state_backend)

            # With learning and progress
            context = RunnerContext(
                outcome_store=outcome_store,
                global_learning_store=get_global_store(),
                progress_callback=my_callback,
            )
            runner = JobRunner(config, backend, state_backend, context=context)
        """
        # Extract optional dependencies from context
        console: Console | None = None
        outcome_store: OutcomeStore | None = None
        escalation_handler: ConsoleEscalationHandler | None = None
        judgment_client: JudgmentClient | None = None
        progress_callback: Callable[[int, int, float | None], None] | None = None
        execution_progress_callback: Callable[[dict[str, Any]], None] | None = None
        global_learning_store: GlobalLearningStore | None = None
        grounding_engine: GroundingEngine | None = None
        rate_limit_callback: Callable[[str, float, str, int], Any] | None = None
        self_healing_enabled = False
        self_healing_auto_confirm = False
        if context is not None:
            console = context.console
            outcome_store = context.outcome_store
            escalation_handler = context.escalation_handler
            judgment_client = context.judgment_client
            progress_callback = context.progress_callback
            execution_progress_callback = context.execution_progress_callback
            global_learning_store = context.global_learning_store
            grounding_engine = context.grounding_engine
            rate_limit_callback = context.rate_limit_callback
            self_healing_enabled = context.self_healing_enabled
            self_healing_auto_confirm = context.self_healing_auto_confirm

        # Core dependencies
        self.config = config
        self.backend = backend
        self.state_backend = state_backend
        self.console = console or Console()

        # Learning components
        self.outcome_store = outcome_store
        self.escalation_handler = escalation_handler
        self.checkpoint_handler: ConsoleCheckpointHandler | None = None  # v21 Evolution
        self.judgment_client = judgment_client

        # Progress callbacks
        self.progress_callback = progress_callback
        self.execution_progress_callback = execution_progress_callback

        # Daemon integration
        self.rate_limit_callback = rate_limit_callback

        # Prompt building and error classification
        self.prompt_builder = PromptBuilder(config.prompt)
        self.error_classifier = ErrorClassifier.from_config(
            config.rate_limit.detection_patterns
        )

        # Preflight checker for workspace validation
        self.preflight_checker = PreflightChecker(
            workspace=config.workspace,
            working_directory=config.backend.working_directory or config.workspace,
        )

        # Global learning store for cross-workspace learning
        # (Evolution #3: Learned Wait Time, #8: Cross-WS Circuit Breaker)
        self._global_learning_store = global_learning_store

        # Grounding engine for external validation of sheet outputs
        # (v8 Evolution: External Grounding Hooks)
        self._grounding_engine = grounding_engine

        # Track applied patterns for feedback loop (Learning Activation)
        self._current_sheet_patterns: list[str] = []
        self._applied_pattern_ids: list[str] = []

        # Pattern Application: Track exploration vs exploitation mode
        # Used to distinguish which patterns were applied via exploration
        # for feedback and effectiveness calculation
        self._exploration_pattern_ids: list[str] = []
        self._exploitation_pattern_ids: list[str] = []

        # Lock for parallel state mutations — used by ParallelExecutor
        # (execution/parallel.py) to wrap state_backend in _LockingStateBackend,
        # serializing concurrent state saves across parallel sheet execution.
        # In sequential mode, asyncio's single-threaded event loop makes this
        # unnecessary, but parallel mode runs multiple sheets concurrently.
        self._state_lock: asyncio.Lock = asyncio.Lock()

        # Graceful shutdown state
        self._shutdown_requested = False
        self._current_state: CheckpointState | None = None
        self._sheet_times: list[float] = []  # Track sheet durations for ETA

        # Pause/resume state tracking (Sheet 12: Job Control Integration)
        self._pause_requested = False
        self._paused_at_sheet: int | None = None

        # Summary tracking for run statistics
        self._summary: RunSummary | None = None
        self._run_start_time: float = 0.0

        # Execution progress tracking (Task 4)
        self._current_sheet_num: int | None = None
        self._execution_progress_snapshots: list[ProgressSnapshotDict] = []
        # Monotonic timestamp of last progress callback (for stale detection)
        self._last_progress_monotonic: float = 0.0

        # Structured logging (Task 8: Logging Integration)
        self._logger: MozartLogger = get_logger("runner")
        self._execution_context: ExecutionContext | None = None

        # Circuit breaker for resilient execution (Task 12)
        self._circuit_breaker: CircuitBreaker | None = None
        if config.circuit_breaker.enabled:
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=config.circuit_breaker.failure_threshold,
                recovery_timeout=config.circuit_breaker.recovery_timeout_seconds,
                name=config.name,
            )

        # Sheet dependency DAG (v17 evolution: Sheet Dependency DAG)
        # Built at initialization to validate dependencies early
        # and enable DAG-aware execution order
        self._dependency_dag: DependencyDAG | None = None
        if config.sheet.dependencies:
            try:
                self._dependency_dag = build_dag_from_config(
                    total_sheets=config.sheet.total_sheets,
                    sheet_dependencies=config.sheet.dependencies,
                )
                self._logger.info(
                    "dag.built",
                    total_sheets=config.sheet.total_sheets,
                    has_dependencies=self._dependency_dag.has_dependencies(),
                    parallelizable=self._dependency_dag.is_parallelizable(),
                )
            except (CycleDetectedError, InvalidDependencyError) as e:
                # Log error and re-raise - don't silently proceed with invalid DAG
                self._logger.error("dag.validation_failed", error=str(e))
                raise

        # Adaptive retry strategy (Task 13)
        # Evolution #3: Pass global learning store for learned wait time injection
        self._retry_strategy = AdaptiveRetryStrategy(
            config=RetryStrategyConfig(
                base_delay=config.retry.base_delay_seconds,
                max_delay=config.retry.max_delay_seconds,
                exponential_base=config.retry.exponential_base,
                jitter_factor=0.25,  # 25% jitter
            ),
            global_learning_store=global_learning_store,
        )

        # Self-healing coordinator (v11 Evolution: Self-Healing)
        # Provides automatic diagnosis and remediation when retries are exhausted
        self._self_healing_enabled = self_healing_enabled
        self._self_healing_auto_confirm = self_healing_auto_confirm
        self._healing_coordinator: SelfHealingCoordinator | None = None
        if self_healing_enabled:
            from mozart.healing.coordinator import SelfHealingCoordinator
            from mozart.healing.registry import create_default_registry

            registry = create_default_registry()
            self._healing_coordinator = SelfHealingCoordinator(
                registry=registry,
                auto_confirm=self_healing_auto_confirm,
                dry_run=False,
            )

        # Parallel executor (v17 evolution: Parallel Sheet Execution)
        # Enables concurrent execution of independent sheets based on DAG
        self._parallel_executor: ParallelExecutor | None = None
        if config.parallel.enabled:
            from mozart.execution.parallel import (
                ParallelExecutionConfig,
                ParallelExecutor,
            )

            parallel_config = ParallelExecutionConfig(
                enabled=True,
                max_concurrent=config.parallel.max_concurrent,
                fail_fast=config.parallel.fail_fast,
            )
            self._parallel_executor = ParallelExecutor(self, parallel_config)  # type: ignore[arg-type]
            self._logger.info(
                "parallel.initialized",
                max_concurrent=config.parallel.max_concurrent,
                fail_fast=config.parallel.fail_fast,
            )

    # ─────────────────────────────────────────────────────────────────────
    # Property Accessors
    # ─────────────────────────────────────────────────────────────────────

    @property
    def dependency_dag(self) -> DependencyDAG | None:
        """Access the dependency DAG (if configured).

        Returns:
            DependencyDAG if sheet dependencies are configured, None otherwise.
        """
        return self._dependency_dag

    # ─────────────────────────────────────────────────────────────────────
    # Signal Handlers for Graceful Shutdown
    # ─────────────────────────────────────────────────────────────────────

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown.

        On Unix, handles SIGINT, SIGTERM, and SIGHUP:
        - SIGINT: User pressed Ctrl+C (interactive)
        - SIGTERM: Standard termination request (kill, systemd, etc.)
        - SIGHUP: Terminal disconnected (SSH drop, terminal close)

        All three trigger the same graceful shutdown: finish the current sheet,
        save state, and exit cleanly. Without SIGTERM/SIGHUP handlers, Mozart
        dies immediately and the job is left in a stale RUNNING state that
        requires manual ``mozart resume`` to recover.

        On Windows, we rely on KeyboardInterrupt.
        """
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    loop.add_signal_handler(sig, self._signal_handler)
            except (RuntimeError, NotImplementedError):
                # Running in a thread or platform doesn't support signals
                pass

    def _remove_signal_handlers(self) -> None:
        """Remove signal handlers."""
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                    loop.remove_signal_handler(sig)
            except (RuntimeError, NotImplementedError, ValueError):
                # Not running or no handler installed
                pass

    def _signal_handler(self) -> None:
        """Handle termination signals by setting shutdown flag.

        Works for SIGINT (Ctrl+C), SIGTERM (kill), and SIGHUP (terminal hangup).
        Sets the shutdown flag so the current sheet finishes and state is saved.
        """
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self.console.print(
                "\n[yellow]Signal received. Finishing current sheet and saving state...[/yellow]"
            )

    async def _handle_graceful_shutdown(self, state: CheckpointState) -> None:
        """Handle graceful shutdown by saving state and raising GracefulShutdown.

        Args:
            state: Current job state to save.

        Raises:
            GracefulShutdownError: Always raised after saving state.
        """
        state.status = JobStatus.PAUSED
        await self.state_backend.save(state)

        # Show resume hint
        self.console.print(
            f"\n[green]State saved.[/green] Job paused at sheet "
            f"{state.last_completed_sheet + 1}/{state.total_sheets}."
        )
        self.console.print(f"\n[bold]To resume:[/bold] mozart resume {state.job_id}")

        raise GracefulShutdownError(f"Job {state.job_id} paused by user request")

    async def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that can be interrupted by shutdown request.

        Args:
            seconds: Time to sleep in seconds.
        """
        # Sleep in small increments to check for shutdown
        increment = min(0.5, seconds)
        elapsed = 0.0

        while elapsed < seconds and not self._shutdown_requested:
            await asyncio.sleep(increment)
            elapsed += increment

    # ─────────────────────────────────────────────────────────────────────
    # Pause Signal Detection (Job Control Integration)
    # ─────────────────────────────────────────────────────────────────────

    def _check_pause_signal(self, state: CheckpointState) -> bool:
        """Check if a pause signal file exists for this job.

        Pause signals are created by the job control service when UI requests
        graceful pause. Uses file-based signaling for cross-process communication.

        Args:
            state: Current job state to get job_id and workspace.

        Returns:
            True if pause signal detected, False otherwise.
        """
        if not state.job_id:
            return False

        # Check for pause signal file in workspace
        workspace_path = Path(self.config.workspace)
        pause_signal_file = workspace_path / f".mozart-pause-{state.job_id}"

        return pause_signal_file.exists()

    def _clear_pause_signal(self, state: CheckpointState) -> None:
        """Clear pause signal file to acknowledge pause handling.

        Args:
            state: Current job state to get job_id and workspace.
        """
        if not state.job_id:
            return

        workspace_path = Path(self.config.workspace)
        pause_signal_file = workspace_path / f".mozart-pause-{state.job_id}"

        try:
            if pause_signal_file.exists():
                pause_signal_file.unlink()
                self._logger.debug(
                    "pause_signal.cleared",
                    job_id=state.job_id,
                    signal_file=str(pause_signal_file),
                )
        except OSError as e:
            self._logger.warning(
                "pause_signal.clear_failed",
                job_id=state.job_id,
                signal_file=str(pause_signal_file),
                error=str(e),
            )

    async def _handle_pause_request(
        self, state: CheckpointState, current_sheet: int
    ) -> None:
        """Handle pause request by saving state and pausing execution.

        Args:
            state: Current job state to update.
            current_sheet: Current sheet number where pause occurred.

        Raises:
            GracefulShutdownError: Always raised after handling pause.
        """
        # Clear the pause signal to acknowledge handling (non-critical)
        try:
            self._clear_pause_signal(state)
        except OSError:
            self._logger.debug("Failed to clear pause signal file", exc_info=True)

        # Update state to paused
        state.mark_job_paused()
        self._paused_at_sheet = current_sheet
        await self.state_backend.save(state)

        # Log pause event
        self._logger.info(
            "job.paused_gracefully",
            job_id=state.job_id,
            paused_at_sheet=current_sheet,
            total_sheets=state.total_sheets,
        )

        # Show pause confirmation to user
        self.console.print(
            f"\n[yellow]Job paused gracefully at sheet "
            f"{current_sheet}/{state.total_sheets}.[/yellow]"
        )
        self.console.print(
            f"[green]State saved.[/green] To resume: [bold]mozart resume {state.job_id}[/bold]"
        )

        raise GracefulShutdownError(f"Job {state.job_id} paused at sheet {current_sheet}")

    # ─────────────────────────────────────────────────────────────────────
    # Progress Tracking
    # ─────────────────────────────────────────────────────────────────────

    def _update_progress(self, state: CheckpointState) -> None:
        """Update progress callback with current state.

        Args:
            state: Current job state.
        """
        if self.progress_callback is None:
            return

        # Calculate ETA based on average sheet time
        eta: float | None = None
        if self._sheet_times:
            avg_time = sum(self._sheet_times) / len(self._sheet_times)
            remaining_sheets = state.total_sheets - state.last_completed_sheet
            eta = avg_time * remaining_sheets

        self.progress_callback(
            state.last_completed_sheet,
            state.total_sheets,
            eta,
        )

    def _handle_execution_progress(self, progress: dict[str, Any]) -> None:
        """Handle execution progress update from backend.

        Called by backend during long-running executions to report
        bytes/lines received and elapsed time.

        Args:
            progress: Dict with bytes_received, lines_received, elapsed_seconds, phase.
        """
        from mozart.utils.time import utc_now

        self._last_progress_monotonic = time.monotonic()

        # Add sheet_num to progress info
        progress_with_sheet = {
            "sheet_num": self._current_sheet_num,
            **progress,
        }

        # Record snapshot for persistence (every 30 seconds of significant change)
        should_snapshot = (
            not self._execution_progress_snapshots
            or progress.get("bytes_received", 0)
            - self._execution_progress_snapshots[-1].get("bytes_received", 0)
            > 1024  # At least 1KB of new data
        )
        if should_snapshot:
            snapshot: dict[str, Any] = {
                **progress_with_sheet,
                "snapshot_at": utc_now().isoformat(),
            }
            self._execution_progress_snapshots.append(
                snapshot  # type: ignore[arg-type]  # dict → ProgressSnapshotDict
            )

        # Forward to CLI callback if set
        if self.execution_progress_callback is not None:
            self.execution_progress_callback(progress_with_sheet)


# Re-export commonly used types for convenience
__all__ = [
    "JobRunnerBase",
]
