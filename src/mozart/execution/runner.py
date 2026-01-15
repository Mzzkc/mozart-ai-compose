"""Job runner with partial completion recovery.

Orchestrates sheet execution with validation, retry logic, and
automatic completion prompt generation for partial failures.
"""

import asyncio
import random
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from mozart.execution.grounding import GroundingEngine, GroundingResult
    from mozart.healing.context import ErrorContext
    from mozart.healing.coordinator import HealingReport, SelfHealingCoordinator
    from mozart.learning.global_store import GlobalLearningStore

from mozart.backends.base import Backend, ExecutionResult
from mozart.core.checkpoint import CheckpointState, JobStatus, SheetState, SheetStatus
from mozart.core.config import IsolationMode, JobConfig
from mozart.core.errors import (
    ClassificationResult,
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
    ErrorCode,
)
from mozart.core.logging import ExecutionContext, MozartLogger, get_logger
from mozart.execution.circuit_breaker import CircuitBreaker
from mozart.execution.escalation import EscalationContext, EscalationHandler, EscalationResponse
from mozart.execution.hooks import HookExecutor, HookResult
from mozart.execution.preflight import PreflightChecker, PreflightResult
from mozart.execution.retry_strategy import (
    AdaptiveRetryStrategy,
    ErrorRecord,
    RetryStrategyConfig,
)
from mozart.execution.validation import (
    FailureHistoryStore,
    HistoricalFailure,
    SheetValidationResult,
    ValidationEngine,
)
from mozart.learning.judgment import JudgmentClient, JudgmentQuery, JudgmentResponse
from mozart.learning.outcomes import OutcomeStore, SheetOutcome
from mozart.prompts.templating import CompletionContext, PromptBuilder, SheetContext
from mozart.state.base import StateBackend
from mozart.utils.time import utc_now


@dataclass
class RunSummary:
    """Summary of a completed job run.

    Tracks key metrics for display at job completion:
    - Total execution time
    - Sheet success/failure counts
    - Validation pass rate
    - Retry statistics
    - Cost tracking
    """

    job_id: str
    job_name: str
    total_sheets: int
    completed_sheets: int = 0
    failed_sheets: int = 0
    skipped_sheets: int = 0
    total_duration_seconds: float = 0.0
    total_retries: int = 0
    total_completion_attempts: int = 0
    rate_limit_waits: int = 0
    validation_pass_count: int = 0
    validation_fail_count: int = 0
    first_attempt_successes: int = 0
    final_status: JobStatus = field(default=JobStatus.PENDING)

    # Cost tracking (v4 evolution: Cost Circuit Breaker)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost: float = 0.0
    cost_limit_hit: bool = False

    # Hook execution results (Concert orchestration)
    hook_results: list[HookResult] = field(default_factory=list)
    hooks_executed: int = 0
    hooks_succeeded: int = 0
    hooks_failed: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate sheet success rate as percentage."""
        if self.total_sheets == 0:
            return 0.0
        return (self.completed_sheets / self.total_sheets) * 100

    @property
    def validation_pass_rate(self) -> float:
        """Calculate validation pass rate as percentage."""
        total = self.validation_pass_count + self.validation_fail_count
        if total == 0:
            return 100.0  # No validations = 100% pass
        return (self.validation_pass_count / total) * 100

    @property
    def first_attempt_rate(self) -> float:
        """Calculate first-attempt success rate as percentage."""
        if self.completed_sheets == 0:
            return 0.0
        return (self.first_attempt_successes / self.completed_sheets) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to dictionary for JSON output."""
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "status": self.final_status.value,
            "duration_seconds": round(self.total_duration_seconds, 2),
            "duration_formatted": self._format_duration(self.total_duration_seconds),
            "sheets": {
                "total": self.total_sheets,
                "completed": self.completed_sheets,
                "failed": self.failed_sheets,
                "skipped": self.skipped_sheets,
                "success_rate": round(self.success_rate, 1),
            },
            "validation": {
                "passed": self.validation_pass_count,
                "failed": self.validation_fail_count,
                "pass_rate": round(self.validation_pass_rate, 1),
            },
            "execution": {
                "total_retries": self.total_retries,
                "completion_attempts": self.total_completion_attempts,
                "rate_limit_waits": self.rate_limit_waits,
                "first_attempt_successes": self.first_attempt_successes,
                "first_attempt_rate": round(self.first_attempt_rate, 1),
            },
        }

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


@dataclass
class GroundingDecisionContext:
    """Context from grounding hooks for completion mode decisions.

    Encapsulates grounding results to inform decision-making about
    whether to retry, complete, or escalate. This integrates external
    validation signals into the adaptive execution flow.

    v10 Evolution: Grounding→Completion Integration
    """

    passed: bool
    """Whether all grounding hooks passed."""

    message: str
    """Human-readable summary of grounding results."""

    confidence: float = 1.0
    """Average confidence across all grounding results (0.0-1.0)."""

    should_escalate: bool = False
    """Whether any grounding hook recommends escalation."""

    recovery_guidance: str | None = None
    """Optional guidance for recovery from grounding failures."""

    hooks_executed: int = 0
    """Number of grounding hooks that were executed."""

    @classmethod
    def from_results(cls, results: list["GroundingResult"]) -> "GroundingDecisionContext":
        """Build context from grounding results list.

        Args:
            results: List of GroundingResult from grounding hooks.

        Returns:
            GroundingDecisionContext summarizing the results.
        """
        if not results:
            return cls(passed=True, message="No grounding hooks executed", hooks_executed=0)

        passed = all(r.passed for r in results)
        confidences = [r.confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        should_escalate = any(r.should_escalate for r in results)

        # Collect recovery guidance from failed hooks
        failed = [r for r in results if not r.passed]
        recovery_guidance = None
        if failed:
            guidance_parts = [r.recovery_guidance for r in failed if r.recovery_guidance]
            if guidance_parts:
                recovery_guidance = "; ".join(guidance_parts)

        # Build message
        if passed:
            message = f"All {len(results)} grounding check(s) passed"
        else:
            failures = ", ".join(f"{r.hook_name}: {r.message}" for r in failed)
            message = f"{len(failed)}/{len(results)} grounding check(s) failed: {failures}"

        return cls(
            passed=passed,
            message=message,
            confidence=avg_confidence,
            should_escalate=should_escalate,
            recovery_guidance=recovery_guidance,
            hooks_executed=len(results),
        )

    @classmethod
    def disabled(cls) -> "GroundingDecisionContext":
        """Create context when grounding is disabled.

        Returns:
            GroundingDecisionContext indicating grounding was not run.
        """
        return cls(passed=True, message="Grounding not enabled", hooks_executed=0)


class SheetExecutionMode(str, Enum):
    """Mode of sheet execution."""

    NORMAL = "normal"
    """Standard first-time execution."""

    COMPLETION = "completion"
    """Completion mode after partial success - focused on missing items."""

    RETRY = "retry"
    """Full retry after completion mode exhausted or minority passed."""

    ESCALATE = "escalate"
    """Escalation mode - low confidence requires external decision."""


class FatalError(Exception):
    """Non-recoverable error that should stop the job."""

    pass


class GracefulShutdownError(Exception):
    """Raised when Ctrl+C is pressed to trigger graceful shutdown.

    This exception is caught by the runner to save state before exiting.
    """

    pass


@dataclass
class RunnerContext:
    """Optional context components for JobRunner.

    Groups optional dependencies that configure learning, escalation,
    and progress reporting. Using a context object reduces the parameter
    count in JobRunner.__init__ from 11 to 5 required parameters plus
    one optional context object.

    Usage:
        # Without context (minimal setup)
        runner = JobRunner(config, backend, state_backend)

        # With learning enabled
        context = RunnerContext(
            outcome_store=JsonOutcomeStore(workspace),
            global_learning_store=get_global_store(),
        )
        runner = JobRunner(config, backend, state_backend, context=context)
    """

    # Learning components
    outcome_store: OutcomeStore | None = None
    """Store for recording sheet outcomes (Phase 1 learning)."""

    escalation_handler: EscalationHandler | None = None
    """Handler for low-confidence decisions (Phase 2 escalation)."""

    judgment_client: JudgmentClient | None = None
    """Client for TDF-aligned execution decisions (Phase 4 Recursive Light)."""

    global_learning_store: "GlobalLearningStore | None" = None
    """Store for cross-workspace learning (Evolution #3, #8)."""

    # External validation
    grounding_engine: "GroundingEngine | None" = None
    """Engine for external validation of sheet outputs (v8 Grounding)."""

    # UI and progress reporting
    console: Console | None = None
    """Rich console for output display."""

    progress_callback: Callable[[int, int, float | None], None] | None = None
    """Callback for progress updates: (completed, total, eta_seconds)."""

    execution_progress_callback: Callable[[dict[str, Any]], None] | None = None
    """Callback for real-time execution progress during sheet runs."""

    # Self-healing configuration (v11 Evolution: Self-Healing)
    self_healing_enabled: bool = False
    """Enable automatic diagnosis and remediation when retries are exhausted."""

    self_healing_auto_confirm: bool = False
    """Auto-confirm suggested fixes (equivalent to --yes flag)."""


class JobRunner:
    """Orchestrates sheet execution with validation and partial recovery.

    The runner implements the following flow:
    1. Execute sheet with standard prompt
    2. Run validations on expected outputs
    3. If all pass: mark complete, continue to next sheet
    4. If majority pass: enter completion mode, generate focused prompt
    5. If minority pass: full retry with original prompt
    6. Track attempts separately for completion vs full retry
    """

    def __init__(
        self,
        config: JobConfig,
        backend: Backend,
        state_backend: StateBackend,
        console: Console | None = None,
        outcome_store: OutcomeStore | None = None,
        escalation_handler: EscalationHandler | None = None,
        judgment_client: JudgmentClient | None = None,
        progress_callback: Callable[[int, int, float | None], None] | None = None,
        execution_progress_callback: Callable[[dict[str, Any]], None] | None = None,
        global_learning_store: "GlobalLearningStore | None" = None,
        grounding_engine: "GroundingEngine | None" = None,
        *,
        context: RunnerContext | None = None,
    ) -> None:
        """Initialize job runner.

        Args:
            config: Job configuration.
            backend: Claude execution backend.
            state_backend: State persistence backend.
            console: Rich console for output (optional, prefer context.console).
            outcome_store: Optional outcome store for learning (prefer context).
            escalation_handler: Optional escalation handler (prefer context).
            judgment_client: Optional judgment client (prefer context).
            progress_callback: Optional progress callback (prefer context).
            execution_progress_callback: Optional execution progress callback (prefer context).
            global_learning_store: Optional global learning store (prefer context).
            grounding_engine: Optional grounding engine (prefer context).
            context: Optional RunnerContext grouping all optional dependencies.
                When provided, context values take precedence over individual params.

        Note:
            The individual optional parameters are preserved for backwards
            compatibility. New code should prefer using the context parameter:

                context = RunnerContext(
                    outcome_store=outcome_store,
                    global_learning_store=get_global_store(),
                )
                runner = JobRunner(config, backend, state_backend, context=context)
        """
        # Merge context with individual params (context takes precedence)
        # Also extract self-healing config from context
        self_healing_enabled = False
        self_healing_auto_confirm = False
        if context is not None:
            console = context.console or console
            outcome_store = context.outcome_store or outcome_store
            escalation_handler = context.escalation_handler or escalation_handler
            judgment_client = context.judgment_client or judgment_client
            progress_callback = context.progress_callback or progress_callback
            execution_progress_callback = context.execution_progress_callback or execution_progress_callback
            global_learning_store = context.global_learning_store or global_learning_store
            grounding_engine = context.grounding_engine or grounding_engine
            self_healing_enabled = context.self_healing_enabled
            self_healing_auto_confirm = context.self_healing_auto_confirm

        self.config = config
        self.backend = backend
        self.state_backend = state_backend
        self.console = console or Console()
        self.outcome_store = outcome_store
        self.escalation_handler = escalation_handler
        self.judgment_client = judgment_client
        self.progress_callback = progress_callback
        self.execution_progress_callback = execution_progress_callback
        self.prompt_builder = PromptBuilder(config.prompt)
        self.error_classifier = ErrorClassifier.from_config(
            config.rate_limit.detection_patterns
        )
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

        # Graceful shutdown state
        self._shutdown_requested = False
        self._current_state: CheckpointState | None = None
        self._sheet_times: list[float] = []  # Track sheet durations for ETA

        # Summary tracking for run statistics
        self._summary: RunSummary | None = None
        self._run_start_time: float = 0.0

        # Execution progress tracking (Task 4)
        self._current_sheet_num: int | None = None
        self._execution_progress_snapshots: list[dict[str, Any]] = []

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
        self._healing_coordinator: "SelfHealingCoordinator | None" = None
        if self_healing_enabled:
            from mozart.healing.coordinator import SelfHealingCoordinator
            from mozart.healing.registry import create_default_registry

            registry = create_default_registry()
            self._healing_coordinator = SelfHealingCoordinator(
                registry=registry,
                auto_confirm=self_healing_auto_confirm,
                dry_run=False,
            )

    def _track_cost(
        self,
        result: ExecutionResult,
        sheet_state: "SheetState",
        state: CheckpointState,
    ) -> tuple[int, int, float, float]:
        """Track token usage and cost from an execution result.

        Records cost in:
        - sheet_state: Per-sheet token/cost tracking
        - state: Cumulative job totals
        - circuit_breaker: Real-time cost enforcement
        - summary: Final run statistics

        Args:
            result: Execution result from backend.
            sheet_state: Current sheet's state object.
            state: Job checkpoint state.

        Returns:
            Tuple of (input_tokens, output_tokens, estimated_cost, confidence).
            confidence is 1.0 for exact API counts, lower for estimates.
        """
        config = self.config.cost_limits

        # Get token counts from result - prefer explicit fields, fall back to estimates
        input_tokens = 0
        output_tokens = 0
        confidence = 1.0

        if result.input_tokens is not None and result.output_tokens is not None:
            # Exact counts from API backend (v4 evolution: precise cost tracking)
            input_tokens = result.input_tokens
            output_tokens = result.output_tokens
            confidence = 1.0  # Exact counts from API
        elif result.tokens_used is not None:
            # Legacy: only total tokens available (deprecated field)
            output_tokens = result.tokens_used
            # Estimate input from output (rough heuristic: input ~= 2x output for prompts)
            input_tokens = output_tokens * 2
            confidence = 0.85  # Reasonable estimate but not exact
        else:
            # Estimate from output length (CLI backend, ~4 chars per token)
            output_chars = len(result.stdout or "") + len(result.stderr or "")
            output_tokens = max(output_chars // 4, 1)  # At least 1 token
            # No way to know input tokens from CLI - estimate from output
            input_tokens = output_tokens * 2  # Rough heuristic
            confidence = 0.7  # Lower confidence for estimates

        # Calculate estimated cost
        estimated_cost = (
            (input_tokens / 1000 * config.cost_per_1k_input_tokens)
            + (output_tokens / 1000 * config.cost_per_1k_output_tokens)
        )

        # Update sheet state
        sheet_state.input_tokens = (sheet_state.input_tokens or 0) + input_tokens
        sheet_state.output_tokens = (sheet_state.output_tokens or 0) + output_tokens
        sheet_state.estimated_cost = (sheet_state.estimated_cost or 0.0) + estimated_cost
        sheet_state.cost_confidence = min(sheet_state.cost_confidence, confidence)

        # Update job state cumulative totals
        state.total_input_tokens += input_tokens
        state.total_output_tokens += output_tokens
        state.total_estimated_cost += estimated_cost

        # Update circuit breaker for real-time tracking
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_cost(input_tokens, output_tokens, estimated_cost)

        # Update summary
        if self._summary is not None:
            self._summary.total_input_tokens += input_tokens
            self._summary.total_output_tokens += output_tokens
            self._summary.total_estimated_cost += estimated_cost

        return input_tokens, output_tokens, estimated_cost, confidence

    def _check_cost_limits(
        self,
        sheet_state: "SheetState",
        state: CheckpointState,
    ) -> tuple[bool, str | None]:
        """Check if any cost limits have been exceeded.

        Args:
            sheet_state: Current sheet's state with cost tracking.
            state: Job checkpoint state with cumulative costs.

        Returns:
            Tuple of (exceeded: bool, reason: str | None).
            If exceeded is True, reason contains the limit that was hit.
        """
        config = self.config.cost_limits
        if not config.enabled:
            return False, None

        # Check per-sheet limit
        if config.max_cost_per_sheet is not None:
            sheet_cost = sheet_state.estimated_cost or 0.0
            if sheet_cost > config.max_cost_per_sheet:
                return True, (
                    f"Sheet cost ${sheet_cost:.4f} exceeded limit "
                    f"${config.max_cost_per_sheet:.2f}"
                )

        # Check per-job limit
        if config.max_cost_per_job is not None:
            job_cost = state.total_estimated_cost
            if job_cost > config.max_cost_per_job:
                return True, (
                    f"Job cost ${job_cost:.4f} exceeded limit "
                    f"${config.max_cost_per_job:.2f}"
                )

            # Emit warning at threshold
            warn_threshold = config.max_cost_per_job * config.warn_at_percent / 100
            if job_cost > warn_threshold and not state.cost_limit_reached:
                self._logger.warning(
                    "cost.warning_threshold",
                    job_cost=round(job_cost, 4),
                    max_cost=config.max_cost_per_job,
                    warn_percent=config.warn_at_percent,
                )
                self.console.print(
                    f"[yellow]Cost warning: ${job_cost:.4f} of "
                    f"${config.max_cost_per_job:.2f} limit "
                    f"({job_cost/config.max_cost_per_job*100:.0f}%)[/yellow]"
                )

        return False, None

    async def run(
        self,
        start_sheet: int | None = None,
        config_path: str | None = None,
    ) -> tuple[CheckpointState, RunSummary]:
        """Run the job from start or resume point.

        Args:
            start_sheet: Optional sheet number to start from (overrides state).
            config_path: Optional path to the original config file (for resume).

        Returns:
            Tuple of (CheckpointState, RunSummary) with final job state and
            execution statistics.

        Raises:
            FatalError: If an unrecoverable error occurs.
            GracefulShutdownError: If Ctrl+C was pressed and job was paused.
        """
        # Initialize timing and summary tracking
        self._run_start_time = time.monotonic()
        state = await self._initialize_state(start_sheet, config_path)
        self._current_state = state

        # Set running PID for zombie detection
        state.set_running_pid()
        await self.state_backend.save(state)

        # Initialize run summary
        self._summary = RunSummary(
            job_id=state.job_id,
            job_name=state.job_name,
            total_sheets=state.total_sheets,
        )

        # Create execution context for correlation (Task 8: Logging Integration)
        self._execution_context = ExecutionContext(
            job_id=state.job_id,
            component="runner",
        )

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

        # Log job start with config summary
        config_summary = self._get_config_summary()
        self._logger.info(
            "job.started",
            job_id=state.job_id,
            total_sheets=state.total_sheets,
            resume_from=start_sheet,
            config=config_summary,
        )

        # Set up worktree isolation if configured (v2 evolution: Worktree Isolation)
        # Store original working directory for restoration if needed
        # Note: working_directory is backend-specific (not in Backend protocol)
        original_working_directory: Path | None = None
        if hasattr(self.backend, "working_directory"):
            original_working_directory = getattr(self.backend, "working_directory", None)
        worktree_path = await self._setup_isolation(state)
        if worktree_path:
            # Override backend working directory to use worktree
            if hasattr(self.backend, "working_directory"):
                setattr(self.backend, "working_directory", worktree_path)
            await self.state_backend.save(state)  # Persist worktree state

        try:
            next_sheet = state.get_next_sheet()
            while next_sheet is not None and next_sheet <= state.total_sheets:
                # Check for shutdown request before starting sheet
                if self._shutdown_requested:
                    await self._handle_graceful_shutdown(state)

                sheet_start = time.monotonic()

                try:
                    await self._execute_sheet_with_recovery(state, next_sheet)
                except FatalError as e:
                    state.mark_job_failed(str(e))
                    await self.state_backend.save(state)
                    # Update summary with failure before raising
                    self._finalize_summary(state)
                    # Log job failure
                    self._logger.error(
                        "job.failed",
                        job_id=state.job_id,
                        sheet_num=next_sheet,
                        error=str(e),
                        duration_seconds=round(time.monotonic() - self._run_start_time, 2),
                        completed_sheets=self._summary.completed_sheets if self._summary else 0,
                    )
                    raise

                # Track sheet timing for ETA calculation
                sheet_duration = time.monotonic() - sheet_start
                self._sheet_times.append(sheet_duration)

                # Update progress callback
                self._update_progress(state)

                # Pause between sheets
                if next_sheet < state.total_sheets:
                    await self._interruptible_sleep(
                        self.config.pause_between_sheets_seconds
                    )

                next_sheet = state.get_next_sheet()

            # Mark job complete if we processed all sheets
            if state.status == JobStatus.RUNNING:
                state.status = JobStatus.COMPLETED
                state.pid = None  # Clear PID on completion
                await self.state_backend.save(state)

            # Finalize summary
            self._finalize_summary(state)

            # Aggregate outcomes to global learning store (Movement IV-B integration)
            await self._aggregate_to_global_store(state)

            # Log job completion with summary
            self._logger.info(
                "job.completed",
                job_id=state.job_id,
                status=state.status.value,
                duration_seconds=round(self._summary.total_duration_seconds, 2),
                completed_sheets=self._summary.completed_sheets,
                failed_sheets=self._summary.failed_sheets,
                success_rate=round(self._summary.success_rate, 1),
                validation_pass_rate=round(self._summary.validation_pass_rate, 1),
                total_retries=self._summary.total_retries,
            )

            # Execute post-success hooks if job completed successfully
            if state.status == JobStatus.COMPLETED and self.config.on_success:
                await self._execute_post_success_hooks(state)

            return state, self._summary
        finally:
            # Clean up worktree isolation if configured (v2 evolution: Worktree Isolation)
            await self._cleanup_isolation(state)

            # Restore original working directory if it was overridden
            if worktree_path and hasattr(self.backend, "working_directory"):
                setattr(self.backend, "working_directory", original_working_directory)

            # Remove signal handlers
            self._remove_signal_handlers()

    def _finalize_summary(self, state: CheckpointState) -> None:
        """Finalize run summary with statistics from completed job.

        Args:
            state: Final job state to extract statistics from.
        """
        if self._summary is None:
            return

        # Calculate total duration
        self._summary.total_duration_seconds = time.monotonic() - self._run_start_time
        self._summary.final_status = state.status

        # Aggregate sheet statistics
        for sheet_state in state.sheets.values():
            if sheet_state.status == SheetStatus.COMPLETED:
                self._summary.completed_sheets += 1
                if sheet_state.first_attempt_success:
                    self._summary.first_attempt_successes += 1
                # Track completion attempts
                if sheet_state.completion_attempts:
                    self._summary.total_completion_attempts += sheet_state.completion_attempts
            elif sheet_state.status == SheetStatus.FAILED:
                self._summary.failed_sheets += 1
            elif sheet_state.status == SheetStatus.SKIPPED:
                self._summary.skipped_sheets += 1

            # Track retries (attempts - 1)
            if sheet_state.attempt_count > 1:
                self._summary.total_retries += sheet_state.attempt_count - 1

            # Track validation results
            if sheet_state.validation_passed is True:
                self._summary.validation_pass_count += 1
            elif sheet_state.validation_passed is False:
                self._summary.validation_fail_count += 1

        # Copy rate limit waits from state
        self._summary.rate_limit_waits = state.rate_limit_waits

    async def _execute_post_success_hooks(self, state: CheckpointState) -> None:
        """Execute post-success hooks after job completion.

        This enables concert orchestration where jobs can chain to other jobs.
        Hooks run in Mozart's Python process, not inside Claude CLI.

        Args:
            state: Final job state (must have COMPLETED status).
        """
        if not self.config.on_success:
            return

        self._logger.info(
            "hooks.executing",
            job_id=state.job_id,
            hook_count=len(self.config.on_success),
            concert_enabled=self.config.concert.enabled,
        )

        # Create hook executor
        executor = HookExecutor(
            config=self.config,
            workspace=self.config.workspace,
            concert_context=None,  # TODO(concert-chaining): Pass concert context from parent job
                                   # to enable job chaining. See ConcertConfig for context structure.
        )

        # Execute hooks
        results = await executor.execute_hooks()

        # Update summary with hook results
        if self._summary:
            self._summary.hook_results = results
            self._summary.hooks_executed = len(results)
            self._summary.hooks_succeeded = sum(1 for r in results if r.success)
            self._summary.hooks_failed = sum(1 for r in results if not r.success)

        # Log summary
        if results:
            self._logger.info(
                "hooks.summary",
                job_id=state.job_id,
                hooks_executed=len(results),
                hooks_succeeded=sum(1 for r in results if r.success),
                hooks_failed=sum(1 for r in results if not r.success),
            )

    def get_summary(self) -> RunSummary | None:
        """Get the current run summary.

        Returns:
            RunSummary if a run has been executed, None otherwise.
        """
        return self._summary

    def _get_config_summary(self) -> dict[str, Any]:
        """Build a safe config summary for logging.

        Returns a dictionary with key configuration values that are safe
        to log (no sensitive data like API keys or tokens).

        Returns:
            Dictionary with non-sensitive configuration summary.
        """
        return {
            "backend_type": self.config.backend.type,
            "sheet_size": self.config.sheet.size,
            "total_items": self.config.sheet.total_items,
            "max_retries": self.config.retry.max_retries,
            "max_completion_attempts": self.config.retry.max_completion_attempts,
            "workspace": str(self.config.workspace),
            "validation_count": len(self.config.validations),
            "rate_limit_wait_minutes": self.config.rate_limit.wait_minutes,
            "learning_enabled": (
                self.config.learning.enabled if hasattr(self.config, "learning") else False
            ),
            "circuit_breaker_enabled": self.config.circuit_breaker.enabled,
            "circuit_breaker_threshold": self.config.circuit_breaker.failure_threshold,
            "isolation_enabled": self.config.isolation.enabled,
            "isolation_mode": self.config.isolation.mode.value if self.config.isolation.enabled else None,
        }

    # =========================================================================
    # Worktree Isolation (v2 evolution: Worktree Isolation)
    # =========================================================================

    async def _setup_isolation(self, state: CheckpointState) -> Path | None:
        """Set up worktree isolation if configured.

        Creates an isolated git worktree for parallel-safe execution when
        isolation.enabled is True. The worktree provides a separate working
        directory, index, and HEAD so multiple jobs can run concurrently.

        Args:
            state: Job checkpoint state for tracking worktree info.

        Returns:
            Path to the worktree if created, None if isolation disabled or failed.

        Raises:
            FatalError: If isolation is required but cannot be established
                and fallback_on_error is False.
        """
        if not self.config.isolation.enabled:
            self._logger.debug("isolation_disabled")
            return None

        if self.config.isolation.mode != IsolationMode.WORKTREE:
            self._logger.warning(
                "unsupported_isolation_mode",
                mode=self.config.isolation.mode.value,
            )
            return None

        # Check for existing worktree from resume
        if state.worktree_path:
            existing_path = Path(state.worktree_path)
            if existing_path.exists():
                self._logger.info(
                    "reusing_existing_worktree",
                    path=str(existing_path),
                )
                return existing_path
            else:
                # Previous worktree was deleted - log warning and create new one
                self._logger.warning(
                    "previous_worktree_missing_creating_new",
                    previous_path=str(existing_path),
                )

        # Import worktree manager (lazy import to avoid circular deps)
        from mozart.isolation.worktree import (
            GitWorktreeManager,
            NotGitRepositoryError,
            WorktreeCreationError,
        )

        # Determine the source path for the git repository
        # Use working_directory if set, otherwise fall back to workspace
        repo_path = self.config.backend.working_directory or self.config.workspace

        manager = GitWorktreeManager(repo_path)

        # Check if in git repo
        if not manager.is_git_repository():
            if self.config.isolation.fallback_on_error:
                self._logger.warning(
                    "not_git_repo_fallback_to_workspace",
                    path=str(repo_path),
                )
                state.isolation_fallback_used = True
                state.isolation_mode = "none"
                return None
            raise FatalError(f"Worktree isolation requires a git repository: {repo_path}")

        # Create worktree (detached HEAD mode for parallel safety)
        try:
            worktree_base = self.config.isolation.get_worktree_base(self.config.workspace)
            result = await manager.create_worktree_detached(
                job_id=state.job_id,
                source_ref=self.config.isolation.source_branch,
                worktree_base=worktree_base,
                lock=self.config.isolation.lock_during_execution,
            )

            if result.success and result.worktree:
                # Update state with worktree info
                state.worktree_path = str(result.worktree.path)
                state.worktree_branch = result.worktree.branch
                state.worktree_locked = result.worktree.locked
                state.worktree_base_commit = result.worktree.commit
                state.isolation_mode = "worktree"

                self._logger.info(
                    "worktree_created",
                    path=str(result.worktree.path),
                    commit=result.worktree.commit,
                    locked=result.worktree.locked,
                )
                return result.worktree.path
            else:
                raise WorktreeCreationError(result.error or "Unknown worktree creation error")

        except (NotGitRepositoryError, WorktreeCreationError) as e:
            if self.config.isolation.fallback_on_error:
                self._logger.warning(
                    "worktree_creation_failed_fallback",
                    error=str(e),
                )
                state.isolation_fallback_used = True
                state.isolation_mode = "none"
                return None
            raise FatalError(f"Failed to create worktree: {e}") from e

    async def _cleanup_isolation(self, state: CheckpointState) -> None:
        """Clean up worktree isolation based on job outcome.

        Handles worktree removal based on job success/failure and cleanup
        configuration. Preserves worktree on failure for debugging by default.

        Args:
            state: Job checkpoint state with worktree info and final status.
        """
        if not state.worktree_path:
            return

        worktree_path = Path(state.worktree_path)
        if not worktree_path.exists():
            self._logger.debug(
                "worktree_already_removed",
                path=str(worktree_path),
            )
            return

        # Determine if we should cleanup based on outcome
        should_cleanup = False
        if state.status == JobStatus.COMPLETED and self.config.isolation.cleanup_on_success:
            should_cleanup = True
        elif state.status == JobStatus.FAILED and self.config.isolation.cleanup_on_failure:
            should_cleanup = True
        elif state.status == JobStatus.PAUSED:
            # Never cleanup on pause - job will resume
            should_cleanup = False

        if not should_cleanup:
            self._logger.info(
                "worktree_preserved",
                path=str(worktree_path),
                status=state.status.value,
                reason="cleanup_disabled_for_outcome",
            )
            return

        # Import worktree manager (lazy import)
        from mozart.isolation.worktree import GitWorktreeManager

        repo_path = self.config.backend.working_directory or self.config.workspace
        manager = GitWorktreeManager(repo_path)

        try:
            # Unlock first if locked
            if state.worktree_locked:
                unlock_result = await manager.unlock_worktree(worktree_path)
                if unlock_result.success:
                    state.worktree_locked = False
                else:
                    self._logger.warning(
                        "worktree_unlock_failed",
                        path=str(worktree_path),
                        error=unlock_result.error,
                    )

            # Remove worktree (force=True to handle dirty state)
            remove_result = await manager.remove_worktree(
                worktree_path,
                force=True,
                delete_branch=False,  # Preserve branch for potential inspection
            )

            if remove_result.success:
                self._logger.info("worktree_removed", path=str(worktree_path))
                # Clear worktree tracking from state
                state.worktree_path = None
                state.worktree_branch = None
                state.worktree_base_commit = None
            else:
                self._logger.warning(
                    "worktree_removal_failed",
                    path=str(worktree_path),
                    error=remove_result.error,
                )
                # Don't fail the job for cleanup issues

        except Exception as e:
            self._logger.warning(
                "worktree_cleanup_exception",
                path=str(worktree_path),
                error=str(e),
            )
            # Don't fail the job for cleanup issues - log and continue

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for graceful shutdown.

        On Unix, uses SIGINT handler. On Windows, we rely on KeyboardInterrupt.
        """
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(
                    signal.SIGINT,
                    self._signal_handler,
                )
            except (RuntimeError, NotImplementedError):
                # Running in a thread or platform doesn't support signals
                pass

    def _remove_signal_handlers(self) -> None:
        """Remove signal handlers."""
        if sys.platform != "win32":
            try:
                loop = asyncio.get_running_loop()
                loop.remove_signal_handler(signal.SIGINT)
            except (RuntimeError, NotImplementedError, ValueError):
                # Not running or no handler installed
                pass

    def _signal_handler(self) -> None:
        """Handle SIGINT by setting shutdown flag."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self.console.print(
                "\n[yellow]Ctrl+C received. Finishing current sheet and saving state...[/yellow]"
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
        self.console.print(
            f"\n[bold]To resume:[/bold] mozart resume {state.job_id}"
        )

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
            snapshot = {
                **progress_with_sheet,
                "snapshot_at": utc_now().isoformat(),
            }
            self._execution_progress_snapshots.append(snapshot)

        # Forward to CLI callback if set
        if self.execution_progress_callback is not None:
            self.execution_progress_callback(progress_with_sheet)

    async def _initialize_state(
        self, start_sheet: int | None,
        config_path: str | None = None,
    ) -> CheckpointState:
        """Initialize or load job state.

        Args:
            start_sheet: Optional override for starting sheet.
            config_path: Optional path to the original config file.

        Returns:
            CheckpointState ready for execution.
        """
        job_id = self.config.name
        state = await self.state_backend.load(job_id)

        if state is None:
            # Serialize JobConfig for resume capability (Task 3: Config Storage)
            config_snapshot = self.config.model_dump(mode="json")

            state = CheckpointState(
                job_id=job_id,
                job_name=self.config.name,
                total_sheets=self.config.sheet.total_sheets,
                config_snapshot=config_snapshot,
                config_path=config_path,
            )
            await self.state_backend.save(state)
            self.console.print("[green]Created new job state[/green]")
        else:
            # Check if another process is already running this job
            if state.status == JobStatus.RUNNING and state.pid is not None:
                import os
                try:
                    os.kill(state.pid, 0)  # Check if process is alive
                    # Process is alive - refuse to start
                    raise FatalError(
                        f"Job is already running (PID {state.pid}). "
                        f"Use 'mozart status {job_id}' to check, or kill the other process first."
                    )
                except ProcessLookupError:
                    # Process is dead - zombie state, recover it
                    state.mark_zombie_detected("Detected on state load - process no longer running")
                    await self.state_backend.save(state)
                except PermissionError:
                    # Can't check (different user) - warn but continue
                    self._logger.warning(
                        "cannot_check_running_process",
                        job_id=job_id,
                        pid=state.pid,
                        reason="permission denied",
                    )

            self.console.print(
                f"[yellow]Resuming from sheet {state.last_completed_sheet + 1}[/yellow]"
            )

        # Override starting sheet if specified
        if start_sheet is not None:
            state.last_completed_sheet = start_sheet - 1

        return state

    async def _try_self_healing(
        self,
        result: ExecutionResult,
        error: ClassifiedError,
        config_path: Path | None,
        sheet_num: int,
        retry_count: int,
        max_retries: int,
    ) -> "HealingReport | None":
        """Attempt self-healing when retries are exhausted.

        Creates an ErrorContext from the execution result and runs the
        healing coordinator to diagnose and potentially fix the issue.

        Args:
            result: The failed execution result.
            error: The classified error from the failure.
            config_path: Path to the config file (if available).
            sheet_num: Current sheet number.
            retry_count: Number of retries attempted.
            max_retries: Maximum retries configured.

        Returns:
            HealingReport if healing was attempted, None if healing is disabled.
        """
        if self._healing_coordinator is None:
            return None

        from mozart.healing.context import ErrorContext

        self._logger.info(
            "sheet.healing_attempt",
            sheet_num=sheet_num,
            error_code=error.error_code.value,
            retry_count=retry_count,
        )

        self.console.print(
            f"\n[yellow]Self-healing: Attempting to diagnose and fix error...[/yellow]"
        )

        # Create error context
        context = ErrorContext.from_execution_result(
            result=result,
            config=self.config,
            config_path=config_path,
            sheet_number=sheet_num,
            error_code=error.error_code.value,
            error_message=error.message,
            error_category=error.category.value,
            retry_count=retry_count,
            max_retries=max_retries,
        )

        # Run healing
        report = await self._healing_coordinator.heal(context)

        # Log the report
        if report.any_remedies_applied:
            self._logger.info(
                "sheet.healing_success",
                sheet_num=sheet_num,
                remedies_applied=[name for name, _ in report.actions_taken],
            )
        else:
            self._logger.warning(
                "sheet.healing_failed",
                sheet_num=sheet_num,
                issues_remaining=report.issues_remaining,
            )

        return report

    async def _execute_sheet_with_recovery(
        self,
        state: CheckpointState,
        sheet_num: int,
    ) -> None:
        """Execute a single sheet with full retry/completion logic.

        Flow:
        1. Execute sheet normally
        2. Run validations
        3. If all pass -> complete
        4. If majority pass -> enter completion mode
        5. If minority pass -> full retry

        Args:
            state: Current job state.
            sheet_num: Sheet number to execute.

        Raises:
            FatalError: If all retries exhausted or fatal error encountered.
        """
        # Track execution timing for learning
        execution_start_time = time.monotonic()

        # Build sheet context
        sheet_context = self._build_sheet_context(sheet_num)
        validation_engine = ValidationEngine(
            self.config.workspace,
            sheet_context.to_dict(),
        )

        # Track attempts
        normal_attempts = 0
        completion_attempts = 0
        max_retries = self.config.retry.max_retries
        max_completion = self.config.retry.max_completion_attempts

        # Track execution history for judgment (Phase 4)
        execution_history: list[ExecutionResult] = []

        # Track error history for adaptive retry (Task 13)
        error_history: list[ErrorRecord] = []

        # Evolution #3: Track pending recovery outcome for global learning store
        # After a retry with wait, we record the outcome (success/failure) to the
        # global store so future jobs can learn from our experience.
        pending_recovery: dict[str, Any] | None = None

        # Query learned patterns before building prompt (Learning Activation)
        # Priority 1: Query from local outcome store (workspace-specific patterns)
        # Priority 2: Query from global learning store (cross-workspace patterns)
        relevant_patterns: list[str] = []
        self._current_sheet_patterns = []
        self._applied_pattern_ids = []

        # Query from local outcome store first (workspace-specific)
        if self.outcome_store is not None:
            try:
                relevant_patterns = await self.outcome_store.get_relevant_patterns(
                    context={
                        "job_id": state.job_id,
                        "sheet_num": sheet_num,
                    },
                    limit=3,
                )
            except Exception as e:
                # Pattern detection failure shouldn't block execution
                self._logger.debug(
                    "patterns.query_local_failed",
                    sheet_num=sheet_num,
                    error=str(e),
                )

        # Query from global learning store (cross-workspace patterns)
        global_patterns, global_pattern_ids = self._query_relevant_patterns(
            job_id=state.job_id,
            sheet_num=sheet_num,
        )

        # Combine patterns (local first, then global, deduplicated)
        if global_patterns:
            # Add global patterns that aren't already covered by local patterns
            for i, gp in enumerate(global_patterns):
                if gp not in relevant_patterns:
                    relevant_patterns.append(gp)
                    self._applied_pattern_ids.append(global_pattern_ids[i])

        # Store applied patterns for feedback tracking
        self._current_sheet_patterns = relevant_patterns.copy()

        if relevant_patterns:
            self._logger.debug(
                "patterns.combined",
                sheet_num=sheet_num,
                local_count=len(relevant_patterns) - len(global_patterns),
                global_count=len(global_patterns),
                total_patterns=len(relevant_patterns),
            )

        # Get applicable validation rules for this sheet
        applicable_rules = validation_engine.get_applicable_rules(self.config.validations)

        # Query historical validation failures for history-aware prompts (Evolution v6)
        # This helps Claude learn from past mistakes and avoid repeating them.
        # Note: Returns empty list for sheet 1 (no prior history to learn from).
        historical_failures: list[HistoricalFailure] = []
        try:
            failure_history_store = FailureHistoryStore(state)
            historical_failures = failure_history_store.query_recent_failures(
                current_sheet=sheet_num,
                lookback_sheets=3,
                limit=3,
            )

            if historical_failures:
                self._logger.debug(
                    "history.failures_found",
                    sheet_num=sheet_num,
                    failure_count=len(historical_failures),
                    sheets=[f.sheet_num for f in historical_failures],
                )
        except Exception as e:
            # Failure history query shouldn't block execution
            self._logger.debug(
                "history.query_failed",
                sheet_num=sheet_num,
                error=str(e),
            )

        # Build original prompt with learned patterns, validation requirements,
        # and historical failures (Evolution v6: History-Aware Prompt Generation)
        original_prompt = self.prompt_builder.build_sheet_prompt(
            sheet_context,
            patterns=relevant_patterns if relevant_patterns else None,
            validation_rules=applicable_rules if applicable_rules else None,
            failure_history=historical_failures if historical_failures else None,
        )
        current_prompt = original_prompt
        current_mode = SheetExecutionMode.NORMAL

        # Run preflight checks before first execution (Task 2: Prompt Metrics)
        preflight_result = self._run_preflight_checks(
            prompt=original_prompt,
            sheet_context=sheet_context.to_dict(),
            sheet_num=sheet_num,
            state=state,
        )

        # Check for fatal preflight errors
        if preflight_result.has_errors:
            error_summary = "; ".join(preflight_result.errors)
            state.mark_sheet_failed(
                sheet_num,
                f"Preflight check failed: {error_summary}",
                "preflight",
            )
            await self.state_backend.save(state)
            # Log preflight failure
            self._logger.error(
                "sheet.preflight_failed",
                sheet_num=sheet_num,
                errors=preflight_result.errors,
            )
            raise FatalError(f"Sheet {sheet_num}: Preflight check failed - {error_summary}")

        # Log sheet start (Task 8: Logging Integration)
        self._logger.info(
            "sheet.started",
            sheet_num=sheet_num,
            execution_mode=current_mode.value,
            prompt_tokens=preflight_result.prompt_metrics.estimated_tokens,
            prompt_lines=preflight_result.prompt_metrics.line_count,
            preflight_warnings=len(preflight_result.warnings),
            patterns_injected=len(relevant_patterns),
        )

        while True:
            # Mark sheet started
            state.mark_sheet_started(sheet_num)
            sheet_state = state.sheets[sheet_num]
            sheet_state.execution_mode = current_mode.value
            await self.state_backend.save(state)

            # Initialize execution progress tracking (Task 4)
            self._current_sheet_num = sheet_num
            self._execution_progress_snapshots.clear()

            # Snapshot mtimes before execution (for file_modified checks)
            validation_engine.snapshot_mtime_files(self.config.validations)

            # Check circuit breaker before execution (Task 12)
            if (
                self._circuit_breaker is not None
                and not self._circuit_breaker.can_execute()
            ):
                wait_time = self._circuit_breaker.time_until_retry()
                cb_state = self._circuit_breaker.get_state()
                self._logger.warning(
                    "circuit_breaker.blocked",
                    sheet_num=sheet_num,
                    state=cb_state.value,
                    wait_seconds=wait_time,
                )
                self.console.print(
                    f"[yellow]Sheet {sheet_num}: Circuit breaker OPEN - "
                    f"waiting {wait_time:.0f}s for recovery[/yellow]"
                )
                if wait_time and wait_time > 0:
                    await self._interruptible_sleep(wait_time)
                continue  # Re-check circuit breaker after wait

            # Evolution #8: Check cross-workspace rate limits before execution
            # If another job has already hit a rate limit, honor it to avoid
            # redundant rate limit hits.
            cross_ws_enabled = (
                self.config.circuit_breaker.cross_workspace_coordination
                and self.config.circuit_breaker.honor_other_jobs_rate_limits
            )
            if cross_ws_enabled and self._global_learning_store is not None:
                try:
                    is_limited, wait_seconds = (
                        self._global_learning_store.is_rate_limited(
                            model=self.config.backend.model,
                        )
                    )
                    if is_limited and wait_seconds and wait_seconds > 0:
                        self._logger.info(
                            "rate_limit.cross_workspace_honored",
                            sheet_num=sheet_num,
                            wait_seconds=round(wait_seconds, 0),
                        )
                        self.console.print(
                            f"[yellow]Sheet {sheet_num}: Honoring cross-workspace rate limit - "
                            f"waiting {wait_seconds:.0f}s[/yellow]"
                        )
                        await self._interruptible_sleep(wait_seconds)
                        # Don't count this as a rate limit wait for this job
                        # since we're just honoring another job's limit
                except Exception as e:
                    # Global store query failure shouldn't block execution
                    self._logger.warning(
                        "rate_limit.cross_workspace_check_failed",
                        sheet_num=sheet_num,
                        error=str(e),
                    )

            # Execute
            self.console.print(
                f"[blue]Sheet {sheet_num}: {current_mode.value} execution[/blue]"
            )
            result = await self.backend.execute(current_prompt)

            # Store execution progress snapshots in sheet state (Task 4)
            if self._execution_progress_snapshots:
                sheet_state.progress_snapshots = self._execution_progress_snapshots.copy()
                sheet_state.last_activity_at = utc_now()

            # Capture raw output for debugging (Task 1: Raw Output Capture)
            sheet_state.capture_output(result.stdout, result.stderr)

            # Track cost (v4 evolution: Cost Circuit Breaker)
            if self.config.cost_limits.enabled:
                self._track_cost(result, sheet_state, state)

                # Check cost limits after tracking
                cost_exceeded, cost_reason = self._check_cost_limits(sheet_state, state)
                if cost_exceeded:
                    state.cost_limit_reached = True
                    if self._summary is not None:
                        self._summary.cost_limit_hit = True
                    self._logger.warning(
                        "cost.limit_exceeded",
                        sheet_num=sheet_num,
                        reason=cost_reason,
                        job_cost=round(state.total_estimated_cost, 4),
                    )
                    self.console.print(f"[red]Cost limit exceeded: {cost_reason}[/red]")

                    # Mark job as paused due to cost limit
                    state.mark_job_paused()
                    state.error_message = f"Cost limit: {cost_reason}"
                    await self.state_backend.save(state)
                    raise GracefulShutdownError(f"Cost limit exceeded: {cost_reason}")

            # Track execution result for judgment (Phase 4)
            execution_history.append(result)

            # ===== VALIDATION-FIRST APPROACH =====
            # Always run validations regardless of exit code.
            # Claude CLI may return exit_code=1 with warnings in JSON even when
            # work was completed successfully. Trust validation results over exit codes.
            # This fixes the "streaming mode" error masking successful execution.
            validation_start = time.monotonic()
            validation_result = validation_engine.run_validations(
                self.config.validations
            )
            validation_duration = time.monotonic() - validation_start

            # Update state with validation details
            self._update_sheet_validation_state(state, sheet_num, validation_result)
            await self.state_backend.save(state)

            if validation_result.all_passed:
                # ===== SUCCESS: All validations passed =====
                # Regardless of exit code, if validations pass, the work was done.
                if not result.success:
                    # Log warning about exit code for debugging, but don't fail
                    self._logger.warning(
                        "sheet.exit_code_ignored_validations_passed",
                        sheet_num=sheet_num,
                        exit_code=result.exit_code,
                        exit_reason=result.exit_reason,
                        validation_count=len(validation_result.results),
                    )
                    self.console.print(
                        f"[yellow]Sheet {sheet_num}: CLI exit {result.exit_code} ignored - "
                        f"all {len(validation_result.results)} validations passed[/yellow]"
                    )

                # ===== GROUNDING: External validation hooks (v8 Evolution) =====
                # Run external grounding hooks to validate output against external
                # sources (APIs, checksums, etc.). This prevents model drift.
                # v10 Evolution: Returns GroundingDecisionContext for completion decisions.
                grounding_ctx = await self._run_grounding_hooks(
                    sheet_num=sheet_num,
                    prompt=current_prompt,
                    output=result.stdout or "",
                    validation_result=validation_result,
                )

                if not grounding_ctx.passed and self.config.grounding.fail_on_grounding_failure:
                    self._logger.warning(
                        "sheet.grounding_failed",
                        sheet_num=sheet_num,
                        message=grounding_ctx.message,
                        confidence=grounding_ctx.confidence,
                    )
                    self.console.print(
                        f"[yellow]Sheet {sheet_num}: Grounding failed - "
                        f"{grounding_ctx.message}[/yellow]"
                    )
                    # v10 Evolution: Use decision methods with grounding context
                    # instead of hard retry. This allows escalation based on
                    # grounding confidence and should_escalate flags.
                    grounding_mode, grounding_reason, _ = await self._decide_with_judgment(
                        sheet_num=sheet_num,
                        validation_result=validation_result,
                        execution_history=execution_history,
                        normal_attempts=normal_attempts,
                        completion_attempts=completion_attempts,
                        grounding_context=grounding_ctx,
                    )
                    self.console.print(
                        f"[dim]Sheet {sheet_num}: Grounding decision: {grounding_mode.value} "
                        f"- {grounding_reason}[/dim]"
                    )
                    if grounding_mode == SheetExecutionMode.ESCALATE:
                        # Escalation requested by grounding - use helper method
                        grounding_error_history: list[str] = (
                            [grounding_ctx.message] if grounding_ctx.message else []
                        )
                        try:
                            response = await self._handle_escalation(
                                state=state,
                                sheet_num=sheet_num,
                                validation_result=validation_result,
                                current_prompt=current_prompt,
                                error_history=grounding_error_history,
                                normal_attempts=normal_attempts,
                            )
                            if response.action == "abort":
                                state.mark_sheet_failed(
                                    sheet_num,
                                    f"Escalation abort: {response.guidance or 'grounding failure'}",
                                    "escalation",
                                )
                                await self.state_backend.save(state)
                                raise FatalError(
                                    f"Sheet {sheet_num} aborted via escalation: "
                                    f"{response.guidance or 'grounding failure'}"
                                )
                            elif response.action == "skip":
                                state.mark_sheet_completed(
                                    sheet_num,
                                    validation_passed=False,
                                    validation_details=validation_result.to_dict_list(),
                                )
                                await self.state_backend.save(state)
                                break  # Skip this sheet
                            # Otherwise continue to retry
                        except FatalError:
                            raise  # Re-raise FatalErrors from escalation
                    # Default: retry on grounding failure
                    continue

                # Record success in circuit breaker (Task 12)
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()

                # Evolution #3: Record recovery outcome if we had a pending retry
                if pending_recovery is not None and self._global_learning_store is not None:
                    try:
                        self._global_learning_store.record_error_recovery(
                            error_code=pending_recovery["error_code"],
                            suggested_wait=pending_recovery["suggested_wait"],
                            actual_wait=pending_recovery["actual_wait"],
                            recovery_success=True,  # Success!
                            model=self.config.backend.model,
                        )
                        self._logger.debug(
                            "learning.recovery_recorded",
                            sheet_num=sheet_num,
                            error_code=pending_recovery["error_code"],
                            actual_wait=pending_recovery["actual_wait"],
                            recovery_success=True,
                        )
                    except Exception as e:
                        self._logger.warning(
                            "learning.recovery_record_failed",
                            sheet_num=sheet_num,
                            error=str(e),
                        )
                    pending_recovery = None

                execution_duration = time.monotonic() - execution_start_time

                # Determine outcome category based on execution path
                first_attempt_success = (normal_attempts == 1 and completion_attempts == 0)
                if first_attempt_success:
                    outcome_category = "success_first_try"
                elif completion_attempts > 0:
                    outcome_category = "success_completion"
                else:
                    outcome_category = "success_retry"

                # Populate SheetState learning fields
                sheet_state = state.sheets[sheet_num]
                sheet_state.first_attempt_success = first_attempt_success
                sheet_state.outcome_category = outcome_category
                sheet_state.confidence_score = validation_result.pass_percentage / 100.0
                # v9 Evolution: Pattern Feedback Loop - save both IDs and descriptions
                sheet_state.applied_pattern_ids = self._applied_pattern_ids.copy()
                sheet_state.applied_pattern_descriptions = self._current_sheet_patterns.copy()
                # v11 Evolution: Grounding→Pattern Integration - capture grounding data
                if grounding_ctx.hooks_executed > 0:
                    sheet_state.grounding_passed = grounding_ctx.passed
                    sheet_state.grounding_confidence = grounding_ctx.confidence
                    sheet_state.grounding_guidance = grounding_ctx.recovery_guidance

                # Record pattern feedback to global store (v9 Evolution)
                await self._record_pattern_feedback(
                    pattern_ids=self._applied_pattern_ids,
                    validation_passed=True,
                    first_attempt_success=first_attempt_success,
                    sheet_num=sheet_num,
                )

                state.mark_sheet_completed(
                    sheet_num,
                    validation_passed=True,
                    validation_details=validation_result.to_dict_list(),
                )

                # Record outcome for learning if store is available
                await self._record_sheet_outcome(
                    sheet_num=sheet_num,
                    job_id=state.job_id,
                    validation_result=validation_result,
                    execution_duration=execution_duration,
                    normal_attempts=normal_attempts,
                    completion_attempts=completion_attempts,
                    first_attempt_success=first_attempt_success,
                    final_status=SheetStatus.COMPLETED,
                )

                await self.state_backend.save(state)
                # Log successful sheet completion
                self._logger.info(
                    "sheet.completed",
                    sheet_num=sheet_num,
                    duration_seconds=round(execution_duration, 2),
                    validation_duration_seconds=round(validation_duration, 3),
                    validation_count=len(validation_result.results),
                    validation_pass_rate=100.0,
                    outcome_category=outcome_category,
                    retry_count=normal_attempts - 1 if normal_attempts > 0 else 0,
                    completion_attempts=completion_attempts,
                    first_attempt_success=first_attempt_success,
                    exit_code_was_nonzero=not result.success,
                )
                self.console.print(
                    f"[green]Sheet {sheet_num}: All {len(validation_result.results)} "
                    f"validations passed[/green]"
                )
                return

            # ===== VALIDATIONS INCOMPLETE =====
            # Some validations failed - decide between retry/completion/failure

            # Show validation summary for user visibility
            passed_count = len(validation_result.get_passed_results())
            failed_count = len(validation_result.get_failed_results())
            pass_pct = validation_result.executed_pass_percentage
            completion_threshold = self.config.retry.completion_threshold_percent

            self.console.print(
                f"[yellow]Sheet {sheet_num}: Validations {passed_count}/{passed_count + failed_count} passed ({pass_pct:.0f}%)[/yellow]"
            )
            for failed in validation_result.get_failed_results():
                self.console.print(
                    f"  [red]✗[/red] {failed.rule.description}"
                )

            # ===== RATE LIMIT CHECK (before completion threshold) =====
            # Rate limits must be handled first, regardless of pass percentage.
            # Otherwise, partial validation success can cause rate limits to be
            # ignored when pass_pct >= completion_threshold.
            if result.rate_limited:
                # Backend detected rate limit - classify and handle
                classification = self._classify_execution(result)
                error = classification.primary

                if error.is_rate_limit:
                    # Clear error history on rate limit
                    error_history.clear()
                    # Handle rate limit with dynamic wait time
                    await self._handle_rate_limit(
                        state,
                        error_code=error.error_code.value,
                        suggested_wait_seconds=error.suggested_wait_seconds,
                    )
                    continue  # Retry same execution

            # Check if pass_pct is high enough for completion mode
            # This applies regardless of exit_code - partial success is still progress
            if pass_pct >= completion_threshold:
                # Jump to judgment/completion logic even if exit_code != 0
                self.console.print(
                    f"[blue]Sheet {sheet_num}: Pass rate ({pass_pct:.0f}%) >= threshold ({completion_threshold}%) - "
                    f"using completion mode[/blue]"
                )
                # Fall through to judgment-based decision below
            elif not result.success:
                # Execution returned non-zero AND some validations failed
                # Use root-cause-aware classification (Evolution v6)
                classification = self._classify_execution(result)
                error = classification.primary

                # Evolution #3: Record recovery outcome as failure if we had a pending retry
                if pending_recovery is not None and self._global_learning_store is not None:
                    try:
                        self._global_learning_store.record_error_recovery(
                            error_code=pending_recovery["error_code"],
                            suggested_wait=pending_recovery["suggested_wait"],
                            actual_wait=pending_recovery["actual_wait"],
                            recovery_success=False,  # Failure
                            model=self.config.backend.model,
                        )
                        self._logger.debug(
                            "learning.recovery_recorded",
                            sheet_num=sheet_num,
                            error_code=pending_recovery["error_code"],
                            actual_wait=pending_recovery["actual_wait"],
                            recovery_success=False,
                        )
                    except Exception as e:
                        self._logger.warning(
                            "learning.recovery_record_failed",
                            sheet_num=sheet_num,
                            error=str(e),
                        )
                    pending_recovery = None

                # Track error for adaptive retry with root cause info (Evolution v6)
                error_record = ErrorRecord.from_classification_result(
                    result=classification,
                    sheet_num=sheet_num,
                    attempt_num=normal_attempts + 1,
                )
                error_history.append(error_record)

                # Record failure in circuit breaker (Task 12)
                # Don't record rate limits as they're handled specially
                if self._circuit_breaker is not None and not error.is_rate_limit:
                    self._circuit_breaker.record_failure()

                if error.is_rate_limit:
                    # Rate limit/quota exhaustion handling uses its own strategy
                    error_history.clear()  # Reset on rate limit
                    # Evolution #8: Pass error code for cross-workspace recording
                    # Pass suggested_wait_seconds for quota exhaustion (E104)
                    await self._handle_rate_limit(
                        state,
                        error_code=error.error_code.value,
                        suggested_wait_seconds=error.suggested_wait_seconds,
                    )
                    continue  # Retry same execution

                if not error.should_retry:
                    # Build informative error message including validation status
                    failed_validations = validation_result.get_failed_results()
                    validation_info = ""
                    if failed_validations:
                        failed_names = [
                            f.rule.description or "unnamed validation"
                            for f in failed_validations
                        ]
                        validation_info = f" (validations failed: {', '.join(failed_names)})"

                    state.mark_sheet_failed(
                        sheet_num,
                        error.message + validation_info,
                        error.category.value,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        execution_duration_seconds=result.duration_seconds,
                    )
                    await self.state_backend.save(state)
                    # Log fatal sheet failure with validation details
                    self._logger.error(
                        "sheet.failed",
                        sheet_num=sheet_num,
                        error_category=error.category.value,
                        error_message=error.message,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        duration_seconds=round(result.duration_seconds or 0, 2),
                        validations_passed=passed_count,
                        validations_failed=failed_count,
                        failed_validation_names=[f.rule.description for f in failed_validations],
                        stdout_tail=result.stdout[-500:] if result.stdout else None,
                    )
                    raise FatalError(
                        f"Sheet {sheet_num}: {error.message}{validation_info}"
                    )

                # Transient error - get adaptive retry recommendation (Task 13)
                retry_recommendation = self._retry_strategy.analyze(
                    error_history=error_history,
                    max_retries=max_retries,
                )

                normal_attempts += 1

                # Check both max retries and adaptive strategy recommendation
                if normal_attempts >= max_retries:
                    # v11 Evolution: Try self-healing before giving up
                    if self._healing_coordinator is not None:
                        healing_report = await self._try_self_healing(
                            result=result,
                            error=error,
                            config_path=None,  # Config path not available in runner
                            sheet_num=sheet_num,
                            retry_count=normal_attempts,
                            max_retries=max_retries,
                        )
                        if healing_report and healing_report.should_retry:
                            # Healing succeeded - reset retry counter and continue
                            normal_attempts = 0
                            self.console.print(healing_report.format())
                            self._logger.info(
                                "sheet.healed",
                                sheet_num=sheet_num,
                                remedies_applied=len(healing_report.actions_taken),
                            )
                            continue

                    # v9 Evolution: Record pattern feedback for failure
                    sheet_state = state.sheets[sheet_num]
                    sheet_state.applied_pattern_ids = self._applied_pattern_ids.copy()
                    sheet_state.applied_pattern_descriptions = self._current_sheet_patterns.copy()
                    await self._record_pattern_feedback(
                        pattern_ids=self._applied_pattern_ids,
                        validation_passed=False,
                        first_attempt_success=False,
                        sheet_num=sheet_num,
                    )

                    state.mark_sheet_failed(
                        sheet_num,
                        f"Failed after {max_retries} retries: {error.message}",
                        error.category.value,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        execution_duration_seconds=result.duration_seconds,
                    )
                    await self.state_backend.save(state)
                    # Log retry exhaustion failure
                    self._logger.error(
                        "sheet.failed",
                        sheet_num=sheet_num,
                        error_category=error.category.value,
                        error_message=f"Retries exhausted: {error.message}",
                        attempt=normal_attempts,
                        max_retries=max_retries,
                        exit_code=result.exit_code,
                        duration_seconds=round(result.duration_seconds or 0, 2),
                    )
                    raise FatalError(
                        f"Sheet {sheet_num} failed after {max_retries} retries"
                    )

                # Check if adaptive strategy recommends stopping early
                if not retry_recommendation.should_retry:
                    state.mark_sheet_failed(
                        sheet_num,
                        f"Adaptive retry aborted: {retry_recommendation.reason}",
                        error.category.value,
                        exit_code=result.exit_code,
                        exit_signal=result.exit_signal,
                        exit_reason=result.exit_reason,
                        execution_duration_seconds=result.duration_seconds,
                    )
                    await self.state_backend.save(state)
                    # Log adaptive strategy abort
                    self._logger.warning(
                        "sheet.adaptive_retry_aborted",
                        sheet_num=sheet_num,
                        error_category=error.category.value,
                        pattern=retry_recommendation.detected_pattern.value,
                        reason=retry_recommendation.reason,
                        confidence=round(retry_recommendation.confidence, 3),
                        strategy=retry_recommendation.strategy_used,
                        attempt=normal_attempts,
                    )
                    raise FatalError(
                        f"Sheet {sheet_num} aborted: {retry_recommendation.reason}"
                    )

                # Log retry attempt with adaptive strategy info
                self._logger.warning(
                    "sheet.retry",
                    sheet_num=sheet_num,
                    attempt=normal_attempts,
                    max_retries=max_retries,
                    error_category=error.category.value,
                    reason=error.message,
                    # Adaptive retry strategy fields (Task 13)
                    retry_delay_seconds=round(retry_recommendation.delay_seconds, 2),
                    retry_confidence=round(retry_recommendation.confidence, 3),
                    retry_pattern=retry_recommendation.detected_pattern.value,
                    retry_strategy=retry_recommendation.strategy_used,
                )
                self.console.print(
                    f"[yellow]Sheet {sheet_num}: {retry_recommendation.detected_pattern.value} - "
                    f"retry {normal_attempts}/{max_retries} "
                    f"(delay: {retry_recommendation.delay_seconds:.1f}s, "
                    f"confidence: {retry_recommendation.confidence:.0%})[/yellow]"
                )

                # Evolution #3: Track pending recovery for learning outcome
                # We'll record whether this retry succeeds or fails in the next iteration
                if self._global_learning_store is not None:
                    pending_recovery = {
                        "error_code": error.error_code.value,
                        "suggested_wait": error.suggested_wait_seconds or 0.0,
                        "actual_wait": retry_recommendation.delay_seconds,
                    }

                await asyncio.sleep(retry_recommendation.delay_seconds)
                continue

            # ===== JUDGMENT/COMPLETION MODE =====
            # Reached when:
            # - pass_pct >= completion_threshold (regardless of exit_code), OR
            # - exit_code == 0 but validations incomplete
            # Use judgment to decide between completion mode, retry, or escalation.

            # Some validations failed - use judgment-based decision logic (Phase 4)
            next_mode, decision_reason, prompt_modifications = await self._decide_with_judgment(
                sheet_num=sheet_num,
                validation_result=validation_result,
                execution_history=execution_history,
                normal_attempts=normal_attempts,
                completion_attempts=completion_attempts,
            )
            # pass_pct already set above

            self.console.print(
                f"[dim]Sheet {sheet_num}: Decision: {next_mode.value} - {decision_reason}[/dim]"
            )

            # Apply prompt modifications from judgment if provided
            if prompt_modifications and next_mode == SheetExecutionMode.RETRY:
                # Join modifications into additional instructions
                modification_text = "\n".join(prompt_modifications)
                current_prompt = (
                    original_prompt + "\n\n---\nJudgment modifications:\n" + modification_text
                )
                self.console.print(
                    f"[blue]Sheet {sheet_num}: Applying {len(prompt_modifications)} "
                    f"prompt modifications from judgment[/blue]"
                )

            if next_mode == SheetExecutionMode.COMPLETION:
                # COMPLETION MODE: High/medium confidence with majority passed
                completion_attempts += 1
                sheet_state.completion_attempts = completion_attempts
                current_mode = SheetExecutionMode.COMPLETION

                # Pass ValidationResult objects (not just rules) to get expanded paths
                completion_ctx = CompletionContext(
                    sheet_num=sheet_num,
                    total_sheets=state.total_sheets,
                    passed_validations=validation_result.get_passed_results(),
                    failed_validations=validation_result.get_failed_results(),
                    completion_attempt=completion_attempts,
                    max_completion_attempts=max_completion,
                    original_prompt=original_prompt,
                    workspace=self.config.workspace,
                )
                # Pass semantic hints (prompt_modifications) to completion prompt
                # These come from either local _decide_next_action or RL judgment
                current_prompt = self.prompt_builder.build_completion_prompt(
                    completion_ctx,
                    semantic_hints=prompt_modifications,
                )

                self.console.print(
                    f"[yellow]Sheet {sheet_num}: Entering completion mode "
                    f"({validation_result.passed_count}/{len(validation_result.results)} passed, "
                    f"{pass_pct:.0f}%). Attempt {completion_attempts}/{max_completion}[/yellow]"
                )

                await asyncio.sleep(self.config.retry.completion_delay_seconds)
                continue

            elif next_mode == SheetExecutionMode.ESCALATE:
                # ESCALATE MODE: Low confidence requires external decision
                # Track error messages for escalation context
                escalation_error_history: list[str] = []
                if sheet_state.error_message:
                    escalation_error_history.append(sheet_state.error_message)

                response = await self._handle_escalation(
                    state=state,
                    sheet_num=sheet_num,
                    validation_result=validation_result,
                    current_prompt=current_prompt,
                    error_history=escalation_error_history,
                    normal_attempts=normal_attempts,
                )

                # Apply escalation response
                if response.action == "retry":
                    normal_attempts += 1
                    if normal_attempts >= max_retries:
                        state.mark_sheet_failed(
                            sheet_num,
                            f"Escalation retry exhausted after {max_retries} attempts",
                            "escalation",
                        )
                        await self.state_backend.save(state)
                        raise FatalError(
                            f"Sheet {sheet_num} exhausted retries after escalation"
                        )
                    current_mode = SheetExecutionMode.RETRY
                    current_prompt = original_prompt
                    await asyncio.sleep(self._get_retry_delay(normal_attempts))
                    continue

                elif response.action == "skip":
                    # Skip this sheet and move to next
                    state.mark_sheet_completed(
                        sheet_num,
                        validation_passed=False,
                        validation_details=validation_result.to_dict_list(),
                    )
                    sheet_state = state.sheets[sheet_num]
                    sheet_state.outcome_category = "skipped_by_escalation"
                    await self.state_backend.save(state)
                    self.console.print(
                        f"[yellow]Sheet {sheet_num}: Skipped via escalation[/yellow]"
                    )
                    return

                elif response.action == "abort":
                    state.mark_sheet_failed(
                        sheet_num,
                        "Aborted via escalation",
                        "escalation",
                    )
                    await self.state_backend.save(state)
                    raise FatalError(
                        f"Sheet {sheet_num}: Job aborted via escalation"
                    )

                elif response.action == "modify_prompt":
                    if response.modified_prompt is None:
                        # Fall back to retry if no modified prompt provided
                        self.console.print(
                            f"[yellow]Sheet {sheet_num}: No modified prompt provided, "
                            f"falling back to retry[/yellow]"
                        )
                        normal_attempts += 1
                        current_mode = SheetExecutionMode.RETRY
                        current_prompt = original_prompt
                    else:
                        current_mode = SheetExecutionMode.RETRY
                        current_prompt = response.modified_prompt
                        self.console.print(
                            f"[blue]Sheet {sheet_num}: Retrying with modified prompt[/blue]"
                        )
                    await asyncio.sleep(self._get_retry_delay(normal_attempts))
                    continue

            else:
                # RETRY MODE: Fall through to full retry
                normal_attempts += 1
                if normal_attempts >= max_retries:
                    # v9 Evolution: Record pattern feedback for validation failure
                    sheet_state = state.sheets[sheet_num]
                    sheet_state.applied_pattern_ids = self._applied_pattern_ids.copy()
                    sheet_state.applied_pattern_descriptions = self._current_sheet_patterns.copy()
                    await self._record_pattern_feedback(
                        pattern_ids=self._applied_pattern_ids,
                        validation_passed=False,
                        first_attempt_success=False,
                        sheet_num=sheet_num,
                    )

                    state.mark_sheet_failed(
                        sheet_num,
                        f"Validation failed after {max_retries} retries and "
                        f"{completion_attempts} completion attempts "
                        f"({validation_result.failed_count} validations still failing)",
                        "validation",
                    )
                    await self.state_backend.save(state)
                    raise FatalError(
                        f"Sheet {sheet_num} exhausted all retry options "
                        f"({validation_result.failed_count} validations failing)"
                    )

                self.console.print(
                    f"[red]Sheet {sheet_num}: {validation_result.failed_count} validations failed "
                    f"({pass_pct:.0f}% passed). Full retry {normal_attempts}/{max_retries}[/red]"
                )

                current_mode = SheetExecutionMode.RETRY
                current_prompt = original_prompt
                await asyncio.sleep(self._get_retry_delay(normal_attempts))

    def _build_sheet_context(self, sheet_num: int) -> SheetContext:
        """Build sheet context for template expansion.

        Args:
            sheet_num: Current sheet number.

        Returns:
            SheetContext with item range and workspace.
        """
        return self.prompt_builder.build_sheet_context(
            sheet_num=sheet_num,
            total_sheets=self.config.sheet.total_sheets,
            sheet_size=self.config.sheet.size,
            total_items=self.config.sheet.total_items,
            start_item=self.config.sheet.start_item,
            workspace=self.config.workspace,
        )

    def _query_relevant_patterns(
        self,
        job_id: str,
        sheet_num: int,
        context_tags: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Query relevant patterns from the global learning store.

        Learning Activation: This method bridges the global learning store's
        pattern knowledge with the prompt injection system. It queries patterns
        that are relevant to the current execution context and formats them
        for injection into sheet prompts.

        Pattern Application v4: Implements epsilon-greedy exploration mode.
        When random() < exploration_rate, also queries lower-priority patterns
        (down to exploration_min_priority) to collect effectiveness data.
        This breaks the cold-start problem where patterns with low effectiveness
        scores never get applied and thus never improve.

        Args:
            job_id: Current job identifier for similar job matching.
            sheet_num: Current sheet number.
            context_tags: Optional tags for context-based filtering.

        Returns:
            Tuple of (pattern_descriptions, pattern_ids).
            - pattern_descriptions: Human-readable strings for prompt injection
            - pattern_ids: IDs for tracking pattern application outcomes
        """
        # Reset exploration tracking for this query
        self._exploration_pattern_ids = []
        self._exploitation_pattern_ids = []

        if self._global_learning_store is None:
            return [], []

        try:
            # Build context tags from execution context for filtering
            # This enables selective pattern retrieval based on current job type
            query_context_tags = context_tags or []
            if not query_context_tags:
                # Auto-generate tags from job context if not provided
                query_context_tags = [
                    f"sheet:{sheet_num}",
                ]
                # Add job name as tag for similar job matching
                if job_id:
                    query_context_tags.append(f"job:{job_id}")

            # Determine exploration vs exploitation mode
            exploration_rate = self.config.learning.exploration_rate
            exploration_min_priority = self.config.learning.exploration_min_priority
            is_exploration_mode = random.random() < exploration_rate

            # Standard exploitation threshold (proven patterns)
            exploitation_threshold = 0.3

            if is_exploration_mode:
                # EXPLORATION MODE: Include lower-priority patterns
                # Use the lower threshold to find untested/low-confidence patterns
                min_priority = exploration_min_priority
                self._logger.info(
                    "patterns.exploration_mode_triggered",
                    job_id=job_id,
                    sheet_num=sheet_num,
                    exploration_rate=exploration_rate,
                    min_priority=min_priority,
                    reason="collecting_effectiveness_data",
                )
            else:
                # EXPLOITATION MODE: Only proven high-priority patterns
                min_priority = exploitation_threshold

            # Query patterns from global store with context filtering
            patterns = self._global_learning_store.get_patterns(
                min_priority=min_priority,
                limit=5,
                context_tags=query_context_tags if query_context_tags else None,
            )

            # If no patterns match with context filtering, fall back to unfiltered
            if not patterns:
                self._logger.debug(
                    "patterns.query_global_fallback",
                    job_id=job_id,
                    sheet_num=sheet_num,
                    context_tags=query_context_tags,
                    reason="no_patterns_matched_context",
                    exploration_mode=is_exploration_mode,
                )
                patterns = self._global_learning_store.get_patterns(
                    min_priority=min_priority,
                    limit=5,
                )

            if not patterns:
                return [], []

            # Format patterns for prompt injection
            descriptions: list[str] = []
            pattern_ids: list[str] = []

            for pattern in patterns:
                # Categorize as exploration vs exploitation for tracking
                if pattern.priority_score < exploitation_threshold:
                    # This pattern would not have been selected without exploration
                    self._exploration_pattern_ids.append(pattern.id)
                    mode_indicator = "🔍"  # Exploration indicator
                else:
                    self._exploitation_pattern_ids.append(pattern.id)
                    mode_indicator = ""

                # Build description based on pattern type and effectiveness
                if pattern.effectiveness_score > 0.7:
                    effectiveness_indicator = "✓"
                elif pattern.effectiveness_score > 0.4:
                    effectiveness_indicator = "○"
                else:
                    effectiveness_indicator = "⚠"

                # Format: [indicator] description (occurrence count)
                # Include mode indicator for exploration patterns
                desc = (
                    f"{mode_indicator}{effectiveness_indicator} "
                    f"{pattern.description or pattern.pattern_name} "
                    f"(seen {pattern.occurrence_count}x, "
                    f"{pattern.effectiveness_score:.0%} effective)"
                )
                descriptions.append(desc)
                pattern_ids.append(pattern.id)

            self._logger.debug(
                "patterns.query_global",
                job_id=job_id,
                sheet_num=sheet_num,
                patterns_found=len(descriptions),
                pattern_ids=pattern_ids,
                context_tags_used=query_context_tags,
                exploration_mode=is_exploration_mode,
                exploration_patterns=len(self._exploration_pattern_ids),
                exploitation_patterns=len(self._exploitation_pattern_ids),
            )

            return descriptions, pattern_ids

        except Exception as e:
            # Pattern query failure shouldn't block execution
            self._logger.warning(
                "patterns.query_global_failed",
                job_id=job_id,
                sheet_num=sheet_num,
                error=str(e),
            )
            return [], []

    def _assess_failure_risk(
        self,
        job_id: str,
        sheet_num: int,
    ) -> dict[str, Any]:
        """Assess failure risk based on historical execution data.

        Learning Activation: Analyzes past executions to assess the risk
        of failure for the current sheet, enabling proactive adjustments
        to retry strategy and confidence thresholds.

        Args:
            job_id: Current job identifier.
            sheet_num: Current sheet number.

        Returns:
            Dict with risk assessment:
            - risk_level: "low", "medium", or "high"
            - confidence: Confidence in assessment (0.0-1.0)
            - factors: List of contributing factors
            - recommended_adjustments: Suggested parameter changes
        """
        if self._global_learning_store is None:
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "factors": ["no global store available"],
                "recommended_adjustments": [],
            }

        try:
            stats = self._global_learning_store.get_execution_stats()
            first_attempt_rate = stats.get("first_attempt_success_rate", 0.0)
            total_executions = stats.get("total_executions", 0)

            # Assess risk based on historical success rate
            if total_executions < 10:
                risk_level = "unknown"
                confidence = 0.2
                factors = [f"insufficient data ({total_executions} executions)"]
            elif first_attempt_rate > 0.7:
                risk_level = "low"
                confidence = min(0.9, total_executions / 100)
                factors = [f"high first-attempt success rate ({first_attempt_rate:.0%})"]
            elif first_attempt_rate > 0.4:
                risk_level = "medium"
                confidence = min(0.8, total_executions / 100)
                factors = [f"moderate first-attempt success rate ({first_attempt_rate:.0%})"]
            else:
                risk_level = "high"
                confidence = min(0.9, total_executions / 100)
                factors = [f"low first-attempt success rate ({first_attempt_rate:.0%})"]

            # Check for active rate limits
            is_limited, wait_time = self._global_learning_store.is_rate_limited()
            if is_limited:
                risk_level = "high"
                factors.append(f"active rate limit (expires in {wait_time:.0f}s)")

            # Build recommendations based on risk level
            recommendations: list[str] = []
            if risk_level == "high":
                recommendations.append("consider increasing retry delays")
                recommendations.append("enable completion mode aggressively")
            elif risk_level == "medium":
                recommendations.append("monitor validation confidence closely")

            return {
                "risk_level": risk_level,
                "confidence": confidence,
                "factors": factors,
                "recommended_adjustments": recommendations,
            }

        except Exception as e:
            self._logger.warning(
                "risk_assessment.failed",
                job_id=job_id,
                sheet_num=sheet_num,
                error=str(e),
            )
            return {
                "risk_level": "unknown",
                "confidence": 0.0,
                "factors": [f"assessment failed: {e}"],
                "recommended_adjustments": [],
            }

    def _run_preflight_checks(
        self,
        prompt: str,
        sheet_context: dict[str, Any],
        sheet_num: int,
        state: CheckpointState,
    ) -> PreflightResult:
        """Run preflight checks on a prompt before execution.

        Analyzes the prompt for potential issues like excessive size,
        missing file references, and invalid working directories.

        Updates SheetState with metrics and any warnings.

        Args:
            prompt: The prompt text to analyze.
            sheet_context: Context dictionary for template expansion.
            sheet_num: Current sheet number.
            state: Current job state to update.

        Returns:
            PreflightResult with metrics and any warnings/errors.
        """
        # Ensure sheet state exists
        if sheet_num not in state.sheets:
            from mozart.core.checkpoint import SheetState
            state.sheets[sheet_num] = SheetState(sheet_num=sheet_num)

        sheet_state = state.sheets[sheet_num]

        # Run preflight checks
        result = self.preflight_checker.check(prompt, sheet_context)

        # Store metrics in sheet state
        sheet_state.prompt_metrics = result.prompt_metrics.to_dict()
        sheet_state.preflight_warnings = result.warnings.copy()

        metrics = result.prompt_metrics

        # Structured logging for preflight results (Task 8)
        if result.has_warnings:
            self._logger.warning(
                "sheet.preflight_warnings",
                sheet_num=sheet_num,
                warnings=result.warnings,
                estimated_tokens=metrics.estimated_tokens,
            )

        # Debug-level log for metrics
        self._logger.debug(
            "sheet.preflight_metrics",
            sheet_num=sheet_num,
            character_count=metrics.character_count,
            estimated_tokens=metrics.estimated_tokens,
            line_count=metrics.line_count,
            word_count=metrics.word_count,
            has_file_references=metrics.has_file_references,
            referenced_path_count=len(metrics.referenced_paths),
        )

        # Console output (preserved for CLI feedback)
        for warning in result.warnings:
            self.console.print(f"[yellow]Sheet {sheet_num} preflight: {warning}[/yellow]")

        for error in result.errors:
            self.console.print(f"[red]Sheet {sheet_num} preflight ERROR: {error}[/red]")

        self.console.print(
            f"[dim]Sheet {sheet_num}: ~{metrics.estimated_tokens:,} tokens, "
            f"{metrics.line_count:,} lines, "
            f"{len(metrics.referenced_paths)} file refs[/dim]"
        )

        return result

    def _update_sheet_validation_state(
        self,
        state: CheckpointState,
        sheet_num: int,
        result: SheetValidationResult,
    ) -> None:
        """Update sheet state with validation tracking.

        Args:
            state: Current job state.
            sheet_num: Sheet number.
            result: Validation results.
        """
        sheet = state.sheets.get(sheet_num)
        if sheet:
            sheet.validation_passed = result.all_passed
            sheet.validation_details = result.to_dict_list()
            sheet.last_pass_percentage = result.pass_percentage

            # Store validation descriptions for display
            sheet.passed_validations = [
                r.rule.description or r.rule.path or "validation"
                for r in result.get_passed_results()
            ]
            sheet.failed_validations = [
                r.rule.description or r.rule.path or "validation"
                for r in result.get_failed_results()
            ]

    async def _run_grounding_hooks(
        self,
        sheet_num: int,
        prompt: str,
        output: str,
        validation_result: SheetValidationResult,
    ) -> GroundingDecisionContext:
        """Run external grounding hooks to validate sheet output.

        Executes registered grounding hooks to validate the sheet output against
        external sources. This addresses the mathematical necessity of external
        validators to prevent model drift (arXiv 2601.05280).

        v10 Evolution: Now returns GroundingDecisionContext with full context
        for completion mode decisions, including confidence and recovery guidance.

        Args:
            sheet_num: Sheet number being validated.
            prompt: The prompt used for execution.
            output: The raw output from execution.
            validation_result: Results from internal validation engine.

        Returns:
            GroundingDecisionContext with pass/fail, confidence, and guidance.
        """
        if self._grounding_engine is None or not self.config.grounding.enabled:
            return GroundingDecisionContext.disabled()

        from mozart.execution.grounding import GroundingContext, GroundingPhase

        # Build grounding context
        context = GroundingContext(
            job_id=self.config.name,
            sheet_num=sheet_num,
            prompt=prompt,
            output=output,
            validation_passed=validation_result.all_passed,
            validation_details=validation_result.to_dict_list(),
            metadata={
                "backend": self.config.backend.type,
                "workspace": str(self.config.workspace),
            },
        )

        # Run post-validation hooks (the most common phase)
        results = await self._grounding_engine.run_hooks(
            context, GroundingPhase.POST_VALIDATION
        )

        # Build decision context from results (v10 Evolution)
        grounding_ctx = GroundingDecisionContext.from_results(results)

        # Log grounding results
        self._logger.info(
            "grounding.hooks_completed",
            sheet_num=sheet_num,
            hooks_run=grounding_ctx.hooks_executed,
            passed=grounding_ctx.passed,
            message=grounding_ctx.message,
            confidence=grounding_ctx.confidence,
        )

        # Handle escalation on grounding failure
        should_log_escalation = (
            not grounding_ctx.passed
            and self.config.grounding.escalate_on_failure
            and grounding_ctx.should_escalate
        )
        if should_log_escalation:
            self._logger.warning(
                "grounding.escalation_triggered",
                sheet_num=sheet_num,
                message=grounding_ctx.message,
            )

        return grounding_ctx

    def _classify_execution(self, result: ExecutionResult) -> ClassificationResult:
        """Classify execution errors using multi-error root cause analysis.

        Uses classify_execution() to detect all errors and identify the root cause.
        Trust the backend's rate_limited flag if set, as the backend sees
        real-time output during execution. The rate limit message may be
        followed by CLI crash stack traces that obscure it in final output.

        Args:
            result: Execution result with error details.

        Returns:
            ClassificationResult with primary (root cause), secondary errors,
            and confidence in root cause identification.
        """
        # When backend detected rate limit, still classify to get specific error
        # (E104 quota exhaustion with dynamic wait vs generic E101 rate limit)
        classification = self.error_classifier.classify_execution(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            exit_signal=result.exit_signal,
            exit_reason=result.exit_reason,
        )

        # Log secondary errors for debugging if any were detected
        if classification.secondary:
            self._logger.debug(
                "runner.secondary_errors_detected",
                primary_code=classification.primary.error_code.value,
                secondary_codes=[e.error_code.value for e in classification.secondary],
                confidence=round(classification.confidence, 3),
            )

        return classification

    def _classify_error(self, result: ExecutionResult) -> ClassifiedError:
        """Classify execution error (backward compatible wrapper).

        This method is maintained for backward compatibility. New code should
        use _classify_execution() to get full root cause analysis.

        Args:
            result: Execution result with error details.

        Returns:
            ClassifiedError (primary error from classification result).
        """
        return self._classify_execution(result).primary

    def _get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-indexed).

        Returns:
            Delay in seconds.
        """
        base = self.config.retry.base_delay_seconds
        exp = self.config.retry.exponential_base
        max_delay = self.config.retry.max_delay_seconds

        delay = min(base * (exp ** (attempt - 1)), max_delay)

        if self.config.retry.jitter:
            # Add 50-100% jitter
            delay *= 0.5 + random.random()

        return delay

    async def _handle_rate_limit(
        self,
        state: CheckpointState,
        error_code: str = "E101",
        suggested_wait_seconds: float | None = None,
    ) -> None:
        """Handle rate limit or quota exhaustion by waiting and health checking.

        Evolution #8: Cross-Workspace Circuit Breaker - Records rate limit events
        to the global learning store so other parallel jobs can avoid hitting
        the same rate limit.

        Args:
            state: Current job state.
            error_code: The error code that triggered the rate limit (default: E101).
            suggested_wait_seconds: Dynamic wait time (for quota exhaustion with
                parsed reset time). If None, uses config.rate_limit.wait_minutes.

        Raises:
            FatalError: If max waits exceeded or health check fails.
        """
        # Determine if this is quota exhaustion (E104) vs regular rate limit
        is_quota_exhaustion = error_code == "E104"

        if suggested_wait_seconds is not None and suggested_wait_seconds > 0:
            # Use dynamic wait time (from parsed reset time)
            wait_seconds = suggested_wait_seconds
            wait_minutes = wait_seconds / 60
        else:
            # Use configured wait time
            wait_minutes = self.config.rate_limit.wait_minutes
            wait_seconds = wait_minutes * 60

        # Track separately for quota exhaustion vs rate limits
        if is_quota_exhaustion:
            state.quota_waits += 1
        else:
            state.rate_limit_waits += 1
        await self.state_backend.save(state)

        # Evolution #8: Record rate limit event to global store for cross-workspace
        # coordination. Other jobs can query this to avoid hitting the same limit.
        cross_ws_enabled = self.config.circuit_breaker.cross_workspace_coordination
        if cross_ws_enabled and self._global_learning_store is not None:
            try:
                self._global_learning_store.record_rate_limit_event(
                    error_code=error_code,
                    duration_seconds=wait_seconds,
                    job_id=state.job_id,
                    model=self.config.backend.model,
                )
                self._logger.info(
                    "rate_limit.cross_workspace_recorded",
                    job_id=state.job_id,
                    error_code=error_code,
                    duration_seconds=wait_seconds,
                )
            except Exception as e:
                # Global store failure shouldn't block rate limit handling
                self._logger.warning(
                    "rate_limit.cross_workspace_record_failed",
                    job_id=state.job_id,
                    error=str(e),
                )

        # Log detection - different event for quota vs rate limit
        if is_quota_exhaustion:
            self._logger.warning(
                "quota_exhausted.detected",
                job_id=state.job_id,
                wait_count=state.quota_waits,
                wait_minutes=wait_minutes,
                wait_seconds=wait_seconds,
            )
            # Quota exhaustion has no max waits - always wait until reset
            self.console.print(
                f"[yellow]Token quota exhausted. Waiting {wait_minutes:.1f} minutes until reset... "
                f"(quota wait #{state.quota_waits})[/yellow]"
            )
        else:
            self._logger.warning(
                "rate_limit.detected",
                job_id=state.job_id,
                wait_count=state.rate_limit_waits,
                max_waits=self.config.rate_limit.max_waits,
                wait_minutes=wait_minutes,
            )
            # Check max waits only for regular rate limits
            if state.rate_limit_waits >= self.config.rate_limit.max_waits:
                self._logger.error(
                    "rate_limit.exhausted",
                    job_id=state.job_id,
                    wait_count=state.rate_limit_waits,
                    max_waits=self.config.rate_limit.max_waits,
                )
                raise FatalError(
                    f"Exceeded maximum rate limit waits ({self.config.rate_limit.max_waits})"
                )
            self.console.print(
                f"[yellow]Rate limited. Waiting {wait_minutes:.0f} minutes... "
                f"(wait {state.rate_limit_waits}/{self.config.rate_limit.max_waits})[/yellow]"
            )

        await asyncio.sleep(wait_seconds)

        # Health check before resuming
        self.console.print("[blue]Running health check...[/blue]")
        if not await self.backend.health_check():
            event_type = "quota_exhausted" if is_quota_exhaustion else "rate_limit"
            self._logger.error(
                f"{event_type}.health_check_failed",
                job_id=state.job_id,
            )
            raise FatalError(f"Backend health check failed after {event_type} wait")

        wait_count = state.quota_waits if is_quota_exhaustion else state.rate_limit_waits
        event_type = "quota_exhausted" if is_quota_exhaustion else "rate_limit"
        self._logger.info(
            f"{event_type}.resumed",
            job_id=state.job_id,
            wait_count=wait_count,
        )
        self.console.print("[green]Health check passed, resuming...[/green]")

    async def _record_sheet_outcome(
        self,
        sheet_num: int,
        job_id: str,
        validation_result: SheetValidationResult,
        execution_duration: float,
        normal_attempts: int,
        completion_attempts: int,
        first_attempt_success: bool,
        final_status: SheetStatus,
    ) -> None:
        """Record sheet outcome for learning if outcome store is available.

        Args:
            sheet_num: Sheet number that completed.
            job_id: Job identifier.
            validation_result: Validation results from the sheet.
            execution_duration: Total execution time in seconds.
            normal_attempts: Number of normal/retry attempts.
            completion_attempts: Number of completion mode attempts.
            first_attempt_success: Whether sheet succeeded on first attempt.
            final_status: Final sheet status.
        """
        if self.outcome_store is None:
            return

        outcome = SheetOutcome(
            sheet_id=f"{job_id}_sheet_{sheet_num}",
            job_id=job_id,
            validation_results=validation_result.to_dict_list(),
            execution_duration=execution_duration,
            retry_count=normal_attempts - 1 if normal_attempts > 0 else 0,
            completion_mode_used=completion_attempts > 0,
            final_status=final_status,
            validation_pass_rate=validation_result.pass_percentage,
            first_attempt_success=first_attempt_success,
            patterns_detected=[],  # Future: pattern detection logic
            timestamp=utc_now(),
        )

        await self.outcome_store.record(outcome)

    async def _record_pattern_feedback(
        self,
        pattern_ids: list[str],
        validation_passed: bool,
        first_attempt_success: bool,
        sheet_num: int,
    ) -> None:
        """Record pattern application feedback to global learning store.

        This closes the pattern feedback loop by recording whether patterns
        that were applied to a sheet execution led to a successful outcome.
        Distinguishes between exploration and exploitation patterns for
        differential effectiveness calculation.

        Args:
            pattern_ids: List of pattern IDs that were applied.
            validation_passed: Whether the sheet validations passed.
            first_attempt_success: Whether the sheet succeeded on first attempt.
            sheet_num: Sheet number for logging context.
        """
        if self._global_learning_store is None or not pattern_ids:
            return

        # Determine outcome improvement:
        # - outcome_improved=True if validation passed AND first_attempt
        # - outcome_improved=False if validation failed (patterns didn't help)
        outcome_improved = validation_passed and first_attempt_success

        for pattern_id in pattern_ids:
            try:
                # Determine application mode based on tracking
                # Exploration patterns were selected via epsilon-greedy below threshold
                if pattern_id in self._exploration_pattern_ids:
                    application_mode = "exploration"
                else:
                    application_mode = "exploitation"

                self._global_learning_store.record_pattern_application(
                    pattern_id=pattern_id,
                    execution_id=f"sheet_{sheet_num}",
                    outcome_improved=outcome_improved,
                    retry_count_before=0,  # We don't track this per-pattern
                    retry_count_after=0 if first_attempt_success else 1,
                    application_mode=application_mode,
                    validation_passed=validation_passed,
                )
                self._logger.debug(
                    "learning.pattern_feedback_recorded",
                    pattern_id=pattern_id,
                    sheet_num=sheet_num,
                    outcome_improved=outcome_improved,
                    validation_passed=validation_passed,
                    first_attempt_success=first_attempt_success,
                    application_mode=application_mode,
                )
            except Exception as e:
                # Pattern feedback recording should not block execution
                self._logger.warning(
                    "learning.pattern_feedback_failed",
                    pattern_id=pattern_id,
                    sheet_num=sheet_num,
                    error=str(e),
                )

    async def _aggregate_to_global_store(self, state: CheckpointState) -> None:
        """Aggregate job outcomes to the global learning store.

        Called on job completion to record all outcomes and detect patterns
        for cross-workspace learning. This is the key integration point for
        Movement IV-B global learning.

        Args:
            state: Final job state with all sheet outcomes.
        """
        if self._global_learning_store is None:
            return

        # Import here to avoid circular imports
        from mozart.learning.aggregator import EnhancedPatternAggregator
        from mozart.learning.outcomes import SheetOutcome

        try:
            # Build SheetOutcome objects from state
            outcomes: list[SheetOutcome] = []
            for sheet_num, sheet_state in state.sheets.items():
                outcome = SheetOutcome(
                    sheet_id=f"{state.job_id}_sheet_{sheet_num}",
                    job_id=state.job_id,
                    validation_results=sheet_state.validation_details or [],
                    execution_duration=sheet_state.execution_duration_seconds or 0.0,
                    retry_count=max(0, sheet_state.attempt_count - 1),
                    completion_mode_used=sheet_state.completion_attempts > 0,
                    final_status=sheet_state.status,
                    validation_pass_rate=(
                        100.0 if sheet_state.validation_passed
                        else 0.0 if sheet_state.validation_passed is False
                        else 50.0  # Unknown
                    ),
                    first_attempt_success=sheet_state.first_attempt_success,
                    timestamp=sheet_state.completed_at or utc_now(),
                    # Output capture for pattern extraction (Evolution: Learning Data Collection)
                    stdout_tail=sheet_state.stdout_tail or "",
                    stderr_tail=sheet_state.stderr_tail or "",
                    # v9 Evolution: Pattern Feedback Loop - pass applied pattern descriptions
                    patterns_applied=sheet_state.applied_pattern_descriptions,
                    # v11 Evolution: Grounding→Pattern Integration - pass grounding context
                    grounding_passed=sheet_state.grounding_passed,
                    grounding_confidence=sheet_state.grounding_confidence,
                    grounding_guidance=sheet_state.grounding_guidance,
                )
                outcomes.append(outcome)

            if not outcomes:
                return

            # Create aggregator and aggregate outcomes (use enhanced for output pattern extraction)
            aggregator = EnhancedPatternAggregator(self._global_learning_store)
            result = aggregator.aggregate_outcomes(
                outcomes=outcomes,
                workspace_path=self.config.workspace,
                model=self.config.backend.model,
            )

            # Log aggregation results
            self._logger.info(
                "learning.global_aggregation",
                job_id=state.job_id,
                outcomes_recorded=result.outcomes_recorded,
                patterns_detected=result.patterns_detected,
                patterns_merged=result.patterns_merged,
                priorities_updated=result.priorities_updated,
            )

            if result.outcomes_recorded > 0:
                self.console.print(
                    f"[dim]Global learning: {result.outcomes_recorded} outcomes recorded, "
                    f"{result.patterns_detected} patterns detected[/dim]"
                )

        except Exception as e:
            # Global learning failures should not block job completion
            self._logger.warning(
                "learning.global_aggregation_failed",
                job_id=state.job_id,
                error=str(e),
            )

    def _decide_next_action(
        self,
        validation_result: SheetValidationResult,
        normal_attempts: int,
        completion_attempts: int,
        grounding_context: GroundingDecisionContext | None = None,
    ) -> tuple[SheetExecutionMode, str, list[str]]:
        """Decide the next action based on confidence, pass percentage, and semantic info.

        This method implements adaptive retry strategy using validation confidence,
        pass percentage, and semantic failure categories to decide between
        COMPLETION, RETRY, or ESCALATE modes.

        Decision logic:
        - If confidence > high_threshold AND pass_pct > threshold: COMPLETION mode
        - If confidence < min_threshold: ESCALATE if handler exists, else RETRY
        - Otherwise: Use existing logic (RETRY or COMPLETION based on pass_pct)

        v10 Evolution: Integrates grounding context into decisions:
        - Grounding confidence adjusts overall confidence (weighted average)
        - Grounding escalation requests trigger escalation
        - Grounding recovery guidance added to semantic hints

        Semantic-aware enhancements:
        - Extracts failure_category to inform retry strategy
        - Returns actionable hints for injection into completion prompts
        - Adjusts decision reason based on dominant failure category

        Args:
            validation_result: Results from validation engine with confidence scores.
            normal_attempts: Number of retry attempts already made.
            completion_attempts: Number of completion mode attempts already made.
            grounding_context: Optional context from grounding hooks (v10 Evolution).

        Returns:
            Tuple of (SheetExecutionMode, reason string, semantic_hints list).
            semantic_hints contains suggested fixes for prompt injection.
        """
        validation_confidence = validation_result.aggregate_confidence
        # Use executed_pass_percentage to exclude skipped validations from staged runs.
        # This prevents cascading stage failures from blocking completion mode.
        # Example: if stage 1 fails (1/2 executed pass), stages 2-4 are skipped,
        # pass_percentage would be 1/5=20%, but executed_pass_percentage is 1/2=50%.
        pass_pct = validation_result.executed_pass_percentage
        completion_threshold = self.config.retry.completion_threshold_percent
        max_completion = self.config.retry.max_completion_attempts

        # Get thresholds from learning config
        high_threshold = self.config.learning.high_confidence_threshold
        min_threshold = self.config.learning.min_confidence_threshold

        # Get semantic information for enhanced decision-making
        semantic_summary = validation_result.get_semantic_summary()
        semantic_hints = validation_result.get_actionable_hints(limit=3)
        dominant_category = semantic_summary.get("dominant_category")

        # v10 Evolution: Factor grounding confidence into decision
        # Use weighted average: validation 70%, grounding 30% (when grounding ran)
        confidence = validation_confidence
        grounding_suffix = ""
        if grounding_context is not None and grounding_context.hooks_executed > 0:
            # Weight validation more heavily as it's the primary signal
            confidence = (validation_confidence * 0.7) + (grounding_context.confidence * 0.3)
            grounding_suffix = f", grounding: {grounding_context.confidence:.2f}"

            # Add grounding recovery guidance to hints if available
            if grounding_context.recovery_guidance:
                semantic_hints = list(semantic_hints)  # Make mutable copy
                semantic_hints.insert(0, f"[Grounding] {grounding_context.recovery_guidance}")

            # Check for grounding-triggered escalation
            if grounding_context.should_escalate:
                escalation_available = (
                    self.config.learning.escalation_enabled
                    and self.escalation_handler is not None
                )
                if escalation_available:
                    return (
                        SheetExecutionMode.ESCALATE,
                        f"grounding hook requests escalation: {grounding_context.message}",
                        semantic_hints,
                    )

        # Build category-specific reason suffix
        category_suffix = ""
        if dominant_category:
            category_suffix = f" (dominant: {dominant_category})"
        category_suffix += grounding_suffix

        # High confidence + majority passed -> completion mode (focused approach)
        if confidence > high_threshold and pass_pct > completion_threshold:
            if completion_attempts < max_completion:
                return (
                    SheetExecutionMode.COMPLETION,
                    f"high confidence ({confidence:.2f}) with {pass_pct:.0f}% passed, "
                    f"attempting focused completion{category_suffix}",
                    semantic_hints,
                )
            else:
                # Completion attempts exhausted, fall back to retry
                return (
                    SheetExecutionMode.RETRY,
                    f"completion attempts exhausted ({completion_attempts}/{max_completion}), "
                    f"falling back to full retry{category_suffix}",
                    semantic_hints,
                )

        # Low confidence -> escalate if enabled and handler available, else retry
        if confidence < min_threshold:
            escalation_available = (
                self.config.learning.escalation_enabled
                and self.escalation_handler is not None
            )
            if escalation_available:
                return (
                    SheetExecutionMode.ESCALATE,
                    f"low confidence ({confidence:.2f}) requires escalation{category_suffix}",
                    semantic_hints,
                )
            else:
                return (
                    SheetExecutionMode.RETRY,
                    f"low confidence ({confidence:.2f}) but escalation not available, "
                    f"attempting retry{category_suffix}",
                    semantic_hints,
                )

        # Medium confidence zone: use pass percentage to decide
        if pass_pct > completion_threshold and completion_attempts < max_completion:
            return (
                SheetExecutionMode.COMPLETION,
                f"medium confidence ({confidence:.2f}) with {pass_pct:.0f}% passed, "
                f"attempting completion mode{category_suffix}",
                semantic_hints,
            )
        else:
            return (
                SheetExecutionMode.RETRY,
                f"medium confidence ({confidence:.2f}) with {pass_pct:.0f}% passed, "
                f"full retry needed{category_suffix}",
                semantic_hints,
            )

    async def _decide_with_judgment(
        self,
        sheet_num: int,
        validation_result: SheetValidationResult,
        execution_history: list[ExecutionResult],
        normal_attempts: int,
        completion_attempts: int,
        grounding_context: GroundingDecisionContext | None = None,
    ) -> tuple[SheetExecutionMode, str, list[str] | None]:
        """Decide next action using Recursive Light judgment if available.

        Consults the judgment_client (Recursive Light) for TDF-aligned decisions.
        Falls back to local _decide_next_action() if no judgment client is configured
        or if the judgment client encounters errors.

        v10 Evolution: Now accepts grounding_context for integrated decisions.

        Args:
            sheet_num: Current sheet number.
            validation_result: Results from validation engine with confidence scores.
            execution_history: List of ExecutionResults from previous attempts.
            normal_attempts: Number of retry attempts already made.
            completion_attempts: Number of completion mode attempts already made.
            grounding_context: Optional context from grounding hooks (v10 Evolution).

        Returns:
            Tuple of (SheetExecutionMode, reason string, optional prompt_modifications).
            The prompt_modifications list is provided if judgment recommends it.
        """
        # Fall back to local decision if no judgment client
        if self.judgment_client is None:
            mode, reason, semantic_hints = self._decide_next_action(
                validation_result, normal_attempts, completion_attempts, grounding_context
            )
            return mode, reason, semantic_hints

        # Build JudgmentQuery from current state
        query = self._build_judgment_query(
            sheet_num=sheet_num,
            validation_result=validation_result,
            execution_history=execution_history,
            normal_attempts=normal_attempts,
        )

        try:
            # Get judgment from Recursive Light
            response = await self.judgment_client.get_judgment(query)

            # Log any patterns learned
            if response.patterns_learned:
                for pattern in response.patterns_learned:
                    self.console.print(
                        f"[dim]Sheet {sheet_num}: Pattern learned: {pattern}[/dim]"
                    )

            # Map JudgmentResponse.recommended_action to SheetExecutionMode
            mode = self._map_judgment_to_mode(response, completion_attempts)
            reason = f"RL judgment ({response.confidence:.2f}): {response.reasoning}"

            return mode, reason, response.prompt_modifications

        except Exception as e:
            # On any error, fall back to local decision
            self.console.print(
                f"[yellow]Sheet {sheet_num}: Judgment client error: {e}, "
                f"falling back to local decision[/yellow]"
            )
            mode, reason, semantic_hints = self._decide_next_action(
                validation_result, normal_attempts, completion_attempts, grounding_context
            )
            return mode, reason + " (judgment fallback)", semantic_hints

    def _build_judgment_query(
        self,
        sheet_num: int,
        validation_result: SheetValidationResult,
        execution_history: list[ExecutionResult],
        normal_attempts: int,
    ) -> JudgmentQuery:
        """Build a JudgmentQuery from current execution state.

        Args:
            sheet_num: Current sheet number.
            validation_result: Validation results with confidence.
            execution_history: Previous execution results for this sheet.
            normal_attempts: Number of retry attempts made.

        Returns:
            JudgmentQuery populated with current state.
        """
        # Extract error patterns from execution history
        error_patterns: list[str] = []
        for result in execution_history:
            if not result.success and result.stderr:
                # Extract first line as pattern identifier
                first_line = result.stderr.split("\n")[0][:100]
                if first_line and first_line not in error_patterns:
                    error_patterns.append(first_line)

        # Serialize execution history for query
        history_dicts: list[dict[str, Any]] = []
        for result in execution_history:
            history_dicts.append({
                "success": result.success,
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds,
                "has_stdout": bool(result.stdout),
                "has_stderr": bool(result.stderr),
            })

        # Get similar outcomes from outcome store if available
        similar_outcomes: list[dict[str, Any]] = []
        # Note: Async call to outcome_store.query_similar() could be added here
        # For now, leave empty - future enhancement

        return JudgmentQuery(
            job_id=self.config.name,
            sheet_num=sheet_num,
            validation_results=validation_result.to_dict_list(),
            execution_history=history_dicts,
            error_patterns=error_patterns,
            retry_count=normal_attempts,
            confidence=validation_result.aggregate_confidence,
            similar_outcomes=similar_outcomes,
        )

    def _map_judgment_to_mode(
        self,
        response: JudgmentResponse,
        completion_attempts: int,
    ) -> SheetExecutionMode:
        """Map JudgmentResponse.recommended_action to SheetExecutionMode.

        Args:
            response: JudgmentResponse from Recursive Light.
            completion_attempts: Number of completion attempts already made.

        Returns:
            SheetExecutionMode corresponding to the recommended action.

        Raises:
            FatalError: If action is 'abort'.
        """
        action = response.recommended_action
        max_completion = self.config.retry.max_completion_attempts

        if action == "proceed":
            # Proceed means validation passed - this shouldn't happen in decision flow
            # but handle gracefully by treating as completion success
            return SheetExecutionMode.NORMAL

        elif action == "retry":
            return SheetExecutionMode.RETRY

        elif action == "completion":
            # Check if we have completion attempts left
            if completion_attempts < max_completion:
                return SheetExecutionMode.COMPLETION
            else:
                # Exhausted completion attempts, fall back to retry
                return SheetExecutionMode.RETRY

        elif action == "escalate":
            return SheetExecutionMode.ESCALATE

        elif action == "abort":
            # Abort raises FatalError to stop the job
            raise FatalError(
                f"RL judgment recommends abort: {response.reasoning}"
            )

        else:
            # Unknown action - fall back to retry
            return SheetExecutionMode.RETRY

    async def _handle_escalation(
        self,
        state: CheckpointState,
        sheet_num: int,
        validation_result: SheetValidationResult,
        current_prompt: str,
        error_history: list[str],
        normal_attempts: int,
    ) -> EscalationResponse:
        """Handle escalation for low-confidence sheet decisions.

        Builds escalation context and invokes the escalation handler
        to get a decision on how to proceed.

        Args:
            state: Current job state.
            sheet_num: Sheet number being processed.
            validation_result: Validation results from the sheet.
            current_prompt: The prompt that was used for execution.
            error_history: List of previous error messages.
            normal_attempts: Number of retry attempts already made.

        Returns:
            EscalationResponse with the action to take.

        Raises:
            FatalError: If no escalation handler is configured.
        """
        if self.escalation_handler is None:
            raise FatalError(
                f"Sheet {sheet_num}: Escalation requested but no handler configured"
            )

        sheet_state = state.sheets.get(sheet_num)
        if sheet_state is None:
            raise FatalError(f"Sheet {sheet_num}: No sheet state found for escalation")

        # Build escalation context
        context = EscalationContext(
            job_id=state.job_id,
            sheet_num=sheet_num,
            validation_results=validation_result.to_dict_list(),
            confidence=validation_result.aggregate_confidence,
            retry_count=normal_attempts,
            error_history=error_history,
            prompt_used=current_prompt,
            output_summary=sheet_state.error_message or "",
        )

        self.console.print(
            f"[yellow]Sheet {sheet_num}: Escalating due to low confidence "
            f"({context.confidence:.1%})[/yellow]"
        )

        # v11 Evolution: Check for similar past escalations for guidance
        if self._global_learning_store is not None:
            try:
                similar = self._global_learning_store.get_similar_escalation(
                    confidence=context.confidence,
                    validation_pass_rate=validation_result.pass_percentage,
                    limit=3,
                )
                if similar:
                    self.console.print(
                        f"[dim]Found {len(similar)} similar past escalation(s):[/dim]"
                    )
                    for past in similar[:2]:
                        outcome = past.outcome_after_action or "unknown"
                        self.console.print(
                            f"[dim]  • {past.action} → {outcome} "
                            f"(conf={past.confidence:.1%})[/dim]"
                        )
            except Exception as e:
                self._logger.debug(
                    "escalation.similar_lookup_failed",
                    sheet_num=sheet_num,
                    error=str(e),
                )

        # Get response from escalation handler
        response = await self.escalation_handler.escalate(context)

        self.console.print(
            f"[blue]Sheet {sheet_num}: Escalation response: {response.action}[/blue]"
        )
        if response.guidance:
            self.console.print(f"[dim]Guidance: {response.guidance}[/dim]")

        # v11 Evolution: Record escalation decision for learning
        escalation_record_id: str | None = None
        if self._global_learning_store is not None:
            try:
                escalation_record_id = self._global_learning_store.record_escalation_decision(
                    job_id=state.job_id,
                    sheet_num=sheet_num,
                    confidence=context.confidence,
                    action=response.action,
                    validation_pass_rate=validation_result.pass_percentage,
                    retry_count=normal_attempts,
                    guidance=response.guidance,
                    model=self.config.backend.model,
                )
                self._logger.info(
                    "escalation.decision_recorded",
                    sheet_num=sheet_num,
                    action=response.action,
                    record_id=escalation_record_id,
                )
                # Store the record ID for outcome update later
                sheet_state.outcome_data = sheet_state.outcome_data or {}
                sheet_state.outcome_data["escalation_record_id"] = escalation_record_id
            except Exception as e:
                self._logger.warning(
                    "escalation.record_failed",
                    sheet_num=sheet_num,
                    error=str(e),
                )

        return response
