"""RunnerProtocol — shared attribute declarations for mixin type safety.

Mixins in the runner package (lifecycle, sheet, patterns, recovery, cost,
isolation) access attributes initialised by ``JobRunnerBase.__init__``.
Because Python's type checkers analyse each class independently, they
report "has no attribute" errors on every cross-mixin access.

This module defines ``RunnerProtocol``, a typing.Protocol that declares
every attribute and method the mixins expect to find on ``self``.  Each
mixin annotates ``self`` in its methods::

    from __future__ import annotations
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .protocol import RunnerProtocol

    class LifecycleMixin:
        async def run(self: RunnerProtocol, ...) -> ...:
            ...

The protocol is **never instantiated** — it exists purely for static
analysis.  At runtime the concrete ``JobRunner`` class (composed via
multiple inheritance) provides all declared members.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from rich.console import Console

    from mozart.backends.base import Backend, ExecutionResult
    from mozart.core.checkpoint import CheckpointState, ProgressSnapshotDict, SheetState
    from mozart.core.config import JobConfig
    from mozart.core.errors import ClassificationResult, ClassifiedError, ErrorClassifier
    from mozart.core.logging import ExecutionContext, MozartLogger
    from mozart.execution.circuit_breaker import CircuitBreaker
    from mozart.execution.dag import DependencyDAG
    from mozart.execution.escalation import ConsoleCheckpointHandler, ConsoleEscalationHandler
    from mozart.execution.grounding import GroundingEngine
    from mozart.execution.parallel import ParallelExecutor
    from mozart.execution.preflight import PreflightChecker
    from mozart.execution.retry_strategy import AdaptiveRetryStrategy
    from mozart.healing.coordinator import HealingReport, SelfHealingCoordinator
    from mozart.learning.global_store import GlobalLearningStore
    from mozart.learning.judgment import JudgmentClient
    from mozart.learning.outcomes import OutcomeStore
    from mozart.prompts.templating import PromptBuilder
    from mozart.state.base import StateBackend

    from .models import RunSummary


@runtime_checkable
class RunnerProtocol(Protocol):
    """Structural type declaring every attribute and helper method
    that runner mixins may access on ``self``.

    The attributes mirror ``JobRunnerBase.__init__`` plus the property
    accessors and internal helpers defined on the base class.  Methods
    from *other* mixins that are called cross-mixin are listed here as
    well so that every ``self.`` reference type-checks.
    """

    # ── Core dependencies (positional args) ──────────────────────────
    config: JobConfig
    backend: Backend
    state_backend: StateBackend
    console: Console

    # ── Learning components ──────────────────────────────────────────
    outcome_store: OutcomeStore | None
    escalation_handler: ConsoleEscalationHandler | None
    checkpoint_handler: ConsoleCheckpointHandler | None
    judgment_client: JudgmentClient | None

    # ── Progress callbacks ───────────────────────────────────────────
    progress_callback: Callable[[int, int, float | None], None] | None
    execution_progress_callback: Callable[[dict[str, Any]], None] | None

    # ── Prompt / error infrastructure ────────────────────────────────
    prompt_builder: PromptBuilder
    error_classifier: ErrorClassifier
    preflight_checker: PreflightChecker

    # ── Global learning store ────────────────────────────────────────
    _global_learning_store: GlobalLearningStore | None
    _grounding_engine: GroundingEngine | None

    # ── Pattern tracking ─────────────────────────────────────────────
    _current_sheet_patterns: list[str]
    _applied_pattern_ids: list[str]
    _exploration_pattern_ids: list[str]
    _exploitation_pattern_ids: list[str]

    # ── State / concurrency ──────────────────────────────────────────
    _state_lock: asyncio.Lock
    _shutdown_requested: bool
    _current_state: CheckpointState | None
    _sheet_times: list[float]
    _pause_requested: bool
    _paused_at_sheet: int | None
    _summary: RunSummary | None
    _run_start_time: float
    _current_sheet_num: int | None
    _execution_progress_snapshots: list[ProgressSnapshotDict]

    # ── Logging ──────────────────────────────────────────────────────
    _logger: MozartLogger
    _execution_context: ExecutionContext | None

    # ── Execution infrastructure ─────────────────────────────────────
    _circuit_breaker: CircuitBreaker | None
    _dependency_dag: DependencyDAG | None
    _retry_strategy: AdaptiveRetryStrategy
    _self_healing_enabled: bool
    _self_healing_auto_confirm: bool
    _healing_coordinator: SelfHealingCoordinator | None
    _parallel_executor: ParallelExecutor | None

    # ── Properties (from JobRunnerBase) ──────────────────────────────
    @property
    def dependency_dag(self) -> DependencyDAG | None: ...

    # ── Methods from JobRunnerBase ───────────────────────────────────
    def _install_signal_handlers(self) -> None: ...
    def _remove_signal_handlers(self) -> None: ...
    def _signal_handler(self) -> None: ...
    async def _handle_graceful_shutdown(self, state: CheckpointState) -> None: ...
    async def _interruptible_sleep(self, seconds: float) -> None: ...
    def _check_pause_signal(self, state: CheckpointState) -> bool: ...
    def _clear_pause_signal(self, state: CheckpointState) -> None: ...
    async def _handle_pause_request(
        self, state: CheckpointState, current_sheet: int
    ) -> None: ...
    def _update_progress(self, state: CheckpointState) -> None: ...
    def _handle_execution_progress(self, progress: dict[str, Any]) -> None: ...

    # ── Methods from IsolationMixin ──────────────────────────────────
    async def _setup_isolation(self, state: CheckpointState) -> Path | None: ...
    async def _cleanup_isolation(self, state: CheckpointState) -> None: ...

    # ── Methods from SheetExecutionMixin ─────────────────────────────
    async def _execute_sheet_with_recovery(
        self, state: CheckpointState, sheet_num: int
    ) -> None: ...

    # ── Methods from RecoveryMixin ───────────────────────────────────
    async def _try_self_healing(
        self,
        result: ExecutionResult,
        error: ClassifiedError,
        config_path: Path | None,
        sheet_num: int,
        retry_count: int,
        max_retries: int,
    ) -> HealingReport | None: ...

    async def _handle_rate_limit(
        self,
        state: CheckpointState,
        error_code: str = ...,
        suggested_wait_seconds: float | None = ...,
    ) -> None: ...

    def _classify_execution(self, result: ExecutionResult) -> ClassificationResult: ...
    def _classify_error(self, result: ExecutionResult) -> ClassifiedError: ...
    def _get_retry_delay(self, attempt: int) -> float: ...

    async def _poll_broadcast_discoveries(
        self, job_id: str, sheet_num: int
    ) -> None: ...

    # ── Methods from PatternsMixin ───────────────────────────────────
    def _query_relevant_patterns(
        self,
        job_id: str,
        sheet_num: int,
        context_tags: list[str] | None = ...,
    ) -> tuple[list[str], list[str]]: ...

    async def _record_pattern_feedback(
        self,
        pattern_ids: list[str],
        ctx: Any,
    ) -> None: ...

    # ── Methods from CostMixin ───────────────────────────────────────
    async def _track_cost(
        self,
        result: ExecutionResult,
        sheet_state: SheetState,
        state: CheckpointState,
    ) -> tuple[int, int, float, float]: ...

    def _check_cost_limits(
        self,
        sheet_state: SheetState,
        state: CheckpointState,
    ) -> tuple[bool, str | None]: ...

    # ── Methods from LifecycleMixin ──────────────────────────────────
    async def _initialize_state(
        self,
        start_sheet: int | None,
        config_path: str | None = ...,
    ) -> CheckpointState: ...

    def _finalize_summary(self, state: CheckpointState) -> None: ...

    async def _execute_post_success_hooks(
        self, state: CheckpointState
    ) -> None: ...

    def _get_config_summary(self) -> dict[str, Any]: ...

    def _get_next_sheet_dag_aware(
        self, state: CheckpointState
    ) -> int | None: ...

    def _get_completed_sheets(
        self, state: CheckpointState
    ) -> set[int]: ...

    async def _execute_sequential_mode(
        self, state: CheckpointState
    ) -> None: ...

    async def _execute_parallel_mode(
        self, state: CheckpointState
    ) -> None: ...

    async def _synthesize_batch_outputs(
        self, result: Any, state: CheckpointState
    ) -> None: ...

    async def _aggregate_to_global_store(
        self, state: CheckpointState
    ) -> None: ...
