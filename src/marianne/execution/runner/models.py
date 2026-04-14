"""Data models for the job runner.

Contains dataclasses, enums, and exceptions used by the JobRunner
and its mixin components. These are extracted to enable clean imports
and avoid circular dependencies during modularization.

Shared types are defined in their canonical locations and re-exported
here for backward compatibility:
- RunSummary / JobCompletionSummary: marianne.core.models
- FatalError, RateLimitExhaustedError, GracefulShutdownError: marianne.core.errors.exceptions
- GroundingDecisionContext, SheetExecutionMode: marianne.core.summary
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from rich.console import Console

if TYPE_CHECKING:
    from collections import deque

    from marianne.backends.base import ExecutionResult
    from marianne.core.checkpoint import CheckpointState
    from marianne.execution.grounding import GroundingEngine
    from marianne.execution.parallel import ResourceChecker
    from marianne.execution.retry_strategy import ErrorRecord
    from marianne.execution.validation import SheetValidationResult
    from marianne.learning.global_store import GlobalLearningStore

# Re-export canonical types for backward compatibility.
# All new code should import from the canonical locations directly.
from marianne.core.errors.exceptions import (  # noqa: F401
    FatalError,
    GracefulShutdownError,
    RateLimitExhaustedError,
)
from marianne.core.models import JobCompletionSummary as RunSummary  # noqa: F401
from marianne.core.summary import (  # noqa: F401
    GroundingDecisionContext,
    SheetExecutionMode,
)
from marianne.execution.escalation import ConsoleCheckpointHandler, ConsoleEscalationHandler
from marianne.learning.judgment import JudgmentClient
from marianne.learning.outcomes import OutcomeStore


@dataclass
class SheetExecutionSetup:
    """Encapsulates the setup phase results for _execute_sheet_with_recovery.

    Groups all initialization state needed before the main execution loop begins.
    This enables cleaner separation between setup and execution phases.
    """

    original_prompt: str
    """The base prompt built from template, patterns, and validation rules."""

    current_prompt: str
    """Active prompt (may be modified by checkpoint system before loop)."""

    current_mode: SheetExecutionMode
    """Initial execution mode (normally NORMAL)."""

    max_retries: int
    """Maximum retry attempts from config."""

    max_completion: int
    """Maximum completion-mode attempts from config."""

    relevant_patterns: list[str] = field(default_factory=list)
    """Learned patterns injected into the prompt."""

    preflight_warnings: int = 0
    """Number of preflight warnings (non-fatal)."""

    preflight_token_estimate: int = 0
    """Estimated token count from preflight metrics."""


@dataclass
class ValidationSuccessContext:
    """Context passed to _handle_validation_success after all validations pass.

    Groups the 11 keyword parameters into a single context object,
    matching the SheetExecutionSetup pattern used for the setup phase.
    """

    state: CheckpointState
    """Current checkpoint state."""

    sheet_num: int
    """Sheet number being executed."""

    result: ExecutionResult
    """ExecutionResult from the backend."""

    validation_result: SheetValidationResult
    """SheetValidationResult with all validation details."""

    validation_duration: float
    """Time spent running validations (seconds)."""

    current_prompt: str
    """The prompt that was executed."""

    normal_attempts: int
    """Number of normal retry attempts used."""

    completion_attempts: int
    """Number of completion-mode attempts used."""

    execution_start_time: float
    """Monotonic timestamp when execution started."""

    execution_history: deque[ExecutionResult]
    """Deque of ExecutionResults from this sheet's attempts."""

    pending_recovery: dict[str, Any] | None
    """Pending self-healing recovery context, if any."""


@dataclass
class ExecutionFailureContext:
    """Context passed to _handle_execution_failure for non-success error handling.

    Groups the parameters needed to classify errors, manage retries,
    and decide between healing, adaptive abort, or normal retry.
    """

    state: CheckpointState
    """Current checkpoint state."""

    sheet_num: int
    """Sheet number being executed."""

    result: ExecutionResult
    """ExecutionResult from the backend."""

    validation_result: SheetValidationResult
    """SheetValidationResult with validation details."""

    passed_count: int
    """Number of validations that passed."""

    failed_count: int
    """Number of validations that failed."""

    error_history: list[ErrorRecord]
    """List of ErrorRecord for adaptive retry analysis."""

    normal_attempts: int
    """Current normal retry attempt count."""

    max_retries: int
    """Maximum retry attempts from config."""

    healing_attempts: int
    """Current healing cycle count."""

    max_healing_cycles: int
    """Maximum healing cycles allowed."""

    pending_recovery: dict[str, Any] | None
    """Pending recovery context from previous retry."""

    grounding_ctx: GroundingDecisionContext
    """GroundingDecisionContext for pattern feedback."""


@dataclass
class FailureHandlingResult:
    """Result from _handle_execution_failure indicating what the caller should do.

    Uses action field for flow control:
    - "continue": Continue the while loop (retry)
    - "fatal": Raise FatalError with the given message
    """

    action: Literal["continue", "fatal"]
    """Either 'continue' or 'fatal'."""

    normal_attempts: int
    """Updated normal attempt counter."""

    healing_attempts: int
    """Updated healing attempt counter."""

    pending_recovery: dict[str, Any] | None
    """Updated pending recovery context."""

    fatal_message: str = ""
    """Error message if action is 'fatal'."""


@dataclass
class ModeDecisionContext:
    """Input context for _apply_mode_decision.

    Groups the 12 parameters of ``_apply_mode_decision`` into a single
    context object, matching the ValidationSuccessContext /
    ExecutionFailureContext pattern used elsewhere.
    """

    state: CheckpointState
    """Current checkpoint state."""

    sheet_num: int
    """Sheet number being executed."""

    validation_result: SheetValidationResult
    """Current validation results."""

    execution_history: deque[ExecutionResult]
    """History of execution results."""

    original_prompt: str
    """Original prompt (for reset on retry)."""

    current_prompt: str
    """Currently active prompt."""

    current_mode: SheetExecutionMode
    """Current execution mode."""

    normal_attempts: int
    """Normal retry attempt count."""

    completion_attempts: int
    """Completion-mode attempt count."""

    max_retries: int
    """Maximum normal retries."""

    max_completion: int
    """Maximum completion attempts."""

    pass_pct: float
    """Current validation pass percentage."""


@dataclass
class ModeDecisionResult:
    """Result from _apply_mode_decision indicating what the caller should do.

    Groups the outcome of judgment/completion/escalation/retry mode selection
    into a single result, reducing control flow complexity in the main loop.

    Flow control:
    - "continue": Re-enter the while loop (completion mode, retry, escalation retry)
    - "return": Exit the function (escalation skip)
    - "fatal": Raise FatalError with the given message
    """

    action: Literal["continue", "return", "fatal"]
    """Flow control action for the caller."""

    current_prompt: str
    """Updated prompt (may be modified by completion/escalation/judgment)."""

    current_mode: SheetExecutionMode
    """Updated execution mode."""

    normal_attempts: int
    """Updated normal attempt counter."""

    completion_attempts: int
    """Updated completion attempt counter."""

    fatal_message: str = ""
    """Error message if action is 'fatal'."""


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

    escalation_handler: ConsoleEscalationHandler | None = None
    """Handler for low-confidence decisions (Phase 2 escalation)."""

    checkpoint_handler: ConsoleCheckpointHandler | None = None
    """Handler for proactive pre-execution checkpoints (v21 Evolution)."""

    judgment_client: JudgmentClient | None = None
    """Client for TDF-aligned execution decisions (Phase 4 Recursive Light)."""

    global_learning_store: GlobalLearningStore | None = None
    """Store for cross-workspace learning (Evolution #3, #8)."""

    # External validation
    grounding_engine: GroundingEngine | None = None
    """Engine for external validation of sheet outputs (v8 Grounding)."""

    # Daemon integration
    daemon_managed: bool = False
    """When True, post-success hooks are suppressed (daemon executes them)."""

    # UI and progress reporting
    console: Console | None = None
    """Rich console for output display."""

    progress_callback: Callable[[int, int, float | None], None] | None = None
    """Callback for progress updates: (completed, total, eta_seconds)."""

    execution_progress_callback: Callable[[dict[str, Any]], None] | None = None
    """Callback for real-time execution progress during sheet runs."""

    # Daemon integration callbacks
    rate_limit_callback: Callable[[str, float, str, int], Any] | None = None
    """Async callback to notify daemon rate coordinator on rate limit hits.

    Signature: (backend_type, wait_seconds, job_id, sheet_num) → Awaitable[None].
    When set, called from _handle_rate_limit() so the daemon's
    RateLimitCoordinator receives live data for cross-job coordination.
    """

    event_callback: Callable[[str, int, str, dict[str, Any] | None], Any] | None = None
    """Async callback to notify daemon of runner lifecycle events.

    Signature: (job_id, sheet_num, event, data) -> Awaitable[None].
    """

    # Preflight thresholds (from daemon config)
    token_warning_threshold: int | None = None
    """Override preflight token warning threshold. None uses module default."""

    token_error_threshold: int | None = None
    """Override preflight token error threshold. None uses module default."""

    # In-process pause signaling (workspace-independent job control)
    pause_event: asyncio.Event | None = None
    """Async event set by the daemon's JobManager to request graceful pause.

    When set, the runner checks this event at sheet boundaries (via
    _check_pause_signal) instead of polling the filesystem for a
    `.marianne-pause-{job_id}` file.  This enables workspace-independent
    job control — the daemon can pause a job without needing to know
    or write to its workspace directory.
    """

    # Self-healing configuration (v11 Evolution: Self-Healing)
    self_healing_enabled: bool = False
    """Enable automatic diagnosis and remediation when retries are exhausted."""

    self_healing_auto_confirm: bool = False
    """Auto-confirm suggested fixes (equivalent to --yes flag)."""

    resource_checker: ResourceChecker | None = field(
        default=None,
        metadata={"description": "Resource checker for parallel execution backpressure."},
    )


# Re-export commonly used types from other modules for convenience
# This allows other runner modules to import from models.py instead
# of managing multiple import sources
__all__ = [
    "ExecutionFailureContext",
    "FailureHandlingResult",
    "FatalError",
    "GracefulShutdownError",
    "GroundingDecisionContext",
    "RateLimitExhaustedError",
    "RunSummary",
    "RunnerContext",
    "SheetExecutionMode",
    "SheetExecutionSetup",
    "ValidationSuccessContext",
]
