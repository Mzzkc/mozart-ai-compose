"""Modularized JobRunner composed from focused mixins.

This package provides the JobRunner class through mixin composition,
splitting the original 4615 LOC runner.py into focused modules:

    - models.py:     Data models (RunSummary, contexts, exceptions)
    - base.py:       Initialization, properties, signal handling
    - lifecycle.py:  run(), resume(), finalize_job()
    - context.py:    Template context construction
    - sheet.py:      Sheet execution state machine
    - patterns.py:   Pattern query and feedback
    - recovery.py:   Error recovery, rate limits, self-healing
    - cost.py:       Token tracking and cost limits
    - isolation.py:  Git worktree isolation

Architecture:
    The JobRunner is composed via multiple inheritance with cooperative
    super() calls. The mixin order matters for Python's MRO:

    class JobRunner(
        SheetExecutionMixin,      # Highest level: sheet execution orchestration
        ContextBuildingMixin,     # Template context construction
        LifecycleMixin,           # Job run lifecycle
        RecoveryMixin,            # Error recovery (used by sheet)
        PatternsMixin,            # Pattern management (used by sheet)
        CostMixin,                # Cost tracking (used by sheet)
        IsolationMixin,           # Worktree isolation (used by lifecycle)
        JobRunnerBase,            # Base initialization (last = first in MRO)
    ): pass

    With this order, JobRunnerBase.__init__ runs first (it's last in the
    class list, so first in MRO when using super()). Other mixins can
    then depend on base attributes being initialized.

Usage:
    from mozart.execution.runner import JobRunner, RunSummary

    runner = JobRunner(config, backend, state_backend)
    summary = await runner.run()
"""

from __future__ import annotations

# Import mixins from modular submodules
from mozart.execution.runner.base import JobRunnerBase
from mozart.execution.runner.context import ContextBuildingMixin
from mozart.execution.runner.cost import CostMixin
from mozart.execution.runner.isolation import IsolationMixin
from mozart.execution.runner.lifecycle import LifecycleMixin
from mozart.execution.runner.models import (
    FatalError,
    GracefulShutdownError,
    GroundingDecisionContext,
    RunnerContext,
    RunSummary,
    SheetExecutionMode,
)
from mozart.execution.runner.patterns import PatternsMixin
from mozart.execution.runner.recovery import RecoveryMixin
from mozart.execution.runner.sheet import SheetExecutionMixin


class JobRunner(
    SheetExecutionMixin,      # Sheet execution orchestration
    ContextBuildingMixin,     # Template context construction
    LifecycleMixin,           # Job run lifecycle (run, resume)
    RecoveryMixin,            # Error recovery and rate limits
    PatternsMixin,            # Pattern query and feedback
    CostMixin,                # Token tracking and cost limits
    IsolationMixin,           # Git worktree isolation
    JobRunnerBase,            # Base initialization (last = first in MRO)
):
    """Orchestrates sheet execution with validation and recovery.

    JobRunner composes multiple mixins to provide:
    - Sheet-by-sheet execution with dependency ordering
    - Automatic retry with exponential backoff
    - Validation integration for quality gates
    - Completion mode for partial success recovery
    - Self-healing for automatic error remediation
    - Cost tracking and limits
    - Git worktree isolation for parallel safety
    - Pattern learning integration

    The class is composed from focused mixins (see module docstring for
    architecture details). This mixin composition pattern enables:
    - Single-file editing for most changes
    - Clear separation of concerns
    - Testable components
    - Easier code navigation

    Example:
        config = JobConfig.from_yaml("job.yaml")
        backend = ClaudeCodeBackend()
        state_backend = JsonStateBackend(workspace)

        runner = JobRunner(config, backend, state_backend)
        summary = await runner.run()

        if summary.final_status == JobStatus.COMPLETED:
            print(f"Success! {summary.completed_sheets}/{summary.total_sheets}")

    See Also:
        - RunSummary: Job completion statistics
        - RunnerContext: Optional context for learning/escalation
        - JobConfig: Job configuration model
    """

    pass  # All functionality provided by mixins


# Public API exports
__all__ = [
    # Main class
    "JobRunner",
    # Data models
    "RunSummary",
    "RunnerContext",
    "SheetExecutionMode",
    "GroundingDecisionContext",
    # Exceptions
    "FatalError",
    "GracefulShutdownError",
    # Mixins (for advanced composition or testing)
    "JobRunnerBase",
    "ContextBuildingMixin",
    "LifecycleMixin",
    "SheetExecutionMixin",
    "RecoveryMixin",
    "PatternsMixin",
    "CostMixin",
    "IsolationMixin",
]
