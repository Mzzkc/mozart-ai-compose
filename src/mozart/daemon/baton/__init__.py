"""The Baton — Mozart's event-driven execution heart.

The baton is the conductor's primary tool. It doesn't decide what to play —
the score does. It doesn't decide how to play — the musicians do. The baton
controls **when** and **how much**: tempo, dynamics, cues, and fermatas.

The baton replaces the current monolithic execution model where
``JobService.start_job()`` runs all sheets sequentially. Instead, the baton
manages sheets across all jobs in a single event-driven loop, dispatching
them to execution when they're ready and the system can handle them.

Package layout::

    events.py       — All BatonEvent types (dataclasses)
    timer.py        — Timer wheel (priority queue of future events)
    state.py        — Baton state models (sheet/instrument/job tracking)
    core.py         — Event inbox, main loop, sheet registry
    musician.py     — Single-attempt sheet execution (play once, report)
    backend_pool.py — Per-instrument backend instance management
    adapter.py      — Wires baton into conductor (step 28)
"""

from mozart.daemon.baton.adapter import BatonAdapter
from mozart.daemon.baton.backend_pool import BackendPool
from mozart.daemon.baton.core import BatonCore
from mozart.daemon.baton.events import (
    BatonEvent,
    CancelJob,
    ConfigReloaded,
    CronTick,
    DispatchRetry,
    EscalationNeeded,
    EscalationResolved,
    EscalationTimeout,
    JobTimeout,
    PacingComplete,
    PauseJob,
    ProcessExited,
    RateLimitExpired,
    RateLimitHit,
    ResourceAnomaly,
    ResumeJob,
    RetryDue,
    SheetAttemptResult,
    SheetSkipped,
    ShutdownRequested,
    StaleCheck,
)
from mozart.daemon.baton.musician import sheet_task
from mozart.daemon.baton.prompt import PromptRenderer, RenderedPrompt
from mozart.daemon.baton.state import (
    AttemptContext,
    AttemptMode,
    BatonJobState,
    BatonSheetStatus,
    CircuitBreakerState,
    InstrumentState,
    SheetExecutionState,
)
from mozart.daemon.baton.timer import TimerHandle, TimerWheel

__all__ = [
    "BatonEvent",
    "CancelJob",
    "ConfigReloaded",
    "CronTick",
    "DispatchRetry",
    "EscalationNeeded",
    "EscalationResolved",
    "EscalationTimeout",
    "JobTimeout",
    "PacingComplete",
    "PauseJob",
    "ProcessExited",
    "RateLimitExpired",
    "RateLimitHit",
    "ResourceAnomaly",
    "ResumeJob",
    "RetryDue",
    "SheetAttemptResult",
    "SheetSkipped",
    "ShutdownRequested",
    "StaleCheck",
    # Timer
    "TimerHandle",
    "TimerWheel",
    # State models
    "AttemptContext",
    "AttemptMode",
    "BatonJobState",
    "BatonSheetStatus",
    "CircuitBreakerState",
    "InstrumentState",
    # Core
    "BackendPool",
    "BatonAdapter",
    "BatonCore",
    "SheetExecutionState",
    # Musician
    "sheet_task",
    # Prompt rendering
    "PromptRenderer",
    "RenderedPrompt",
]
