"""Execution progress tracking for long-running sheet executions.

Provides real-time visibility into sheet execution progress through:
- Byte/line tracking as output streams in
- Phase indicators (starting, executing, validating)
- Periodic progress snapshots for state persistence
- Heartbeat updates to detect stalled executions

This module enables the CLI to show "Still running... 5.2KB received, 3m elapsed"
and provides the data needed for execution history analysis.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mozart.utils.time import utc_now


@dataclass
class ExecutionProgress:
    """Snapshot of execution progress at a point in time.

    Tracks metrics about an ongoing execution to provide visibility
    during long-running sheet operations.

    Attributes:
        started_at: When the execution started.
        last_activity_at: When we last received output or status update.
        bytes_received: Total bytes of output received so far.
        lines_received: Total lines of output received so far.
        phase: Current execution phase.
        elapsed_seconds: Seconds since execution started.
    """

    started_at: datetime
    last_activity_at: datetime
    bytes_received: int = 0
    lines_received: int = 0
    phase: str = "starting"  # "starting", "executing", "validating", "completed"

    @property
    def elapsed_seconds(self) -> float:
        """Calculate seconds elapsed since execution started."""
        delta = utc_now() - self.started_at
        return delta.total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Calculate seconds since last activity."""
        delta = utc_now() - self.last_activity_at
        return delta.total_seconds()

    def format_bytes(self) -> str:
        """Format bytes_received as human-readable string."""
        if self.bytes_received < 1024:
            return f"{self.bytes_received}B"
        elif self.bytes_received < 1024 * 1024:
            return f"{self.bytes_received / 1024:.1f}KB"
        else:
            return f"{self.bytes_received / (1024 * 1024):.1f}MB"

    def format_elapsed(self) -> str:
        """Format elapsed time as human-readable string."""
        seconds = self.elapsed_seconds
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def format_status(self) -> str:
        """Format complete status line for display.

        Returns a string like: "Still running... 5.2KB received, 3m 15s elapsed"
        """
        return (
            f"Still running... {self.format_bytes()} received, "
            f"{self.lines_received} lines, {self.format_elapsed()} elapsed"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with progress metrics and timestamps.
        """
        return {
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "bytes_received": self.bytes_received,
            "lines_received": self.lines_received,
            "phase": self.phase,
            "elapsed_seconds": self.elapsed_seconds,
        }


# Type alias for progress callback function
ProgressCallback = Callable[[ExecutionProgress], None]


@dataclass
class ProgressTracker:
    """Tracks execution progress during a sheet execution.

    Provides methods to update progress metrics as output streams in,
    and optionally notifies a callback for real-time updates.

    Example:
        tracker = ProgressTracker(callback=my_update_fn)
        tracker.set_phase("executing")
        tracker.update(new_bytes=1024, new_lines=10)
        print(tracker.get_progress().format_status())

    Attributes:
        callback: Optional function called on each progress update.
        _progress: Internal ExecutionProgress state.
        _snapshot_interval_seconds: How often to record snapshots.
        _last_snapshot_at: When last snapshot was recorded.
        _snapshots: List of progress snapshots for persistence.
    """

    callback: ProgressCallback | None = None
    _progress: ExecutionProgress = field(init=False)
    _snapshot_interval_seconds: float = 30.0
    _last_snapshot_at: datetime = field(init=False)
    _snapshots: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize progress tracking state."""
        now = utc_now()
        self._progress = ExecutionProgress(
            started_at=now,
            last_activity_at=now,
            phase="starting",
        )
        self._last_snapshot_at = now

    def update(
        self,
        new_bytes: int = 0,
        new_lines: int = 0,
        *,
        force_snapshot: bool = False,
    ) -> None:
        """Update progress with new output received.

        Increments byte/line counters, updates last_activity_at,
        and optionally records a snapshot for persistence.

        Args:
            new_bytes: Number of new bytes received.
            new_lines: Number of new lines received.
            force_snapshot: Force a snapshot even if interval hasn't passed.
        """
        now = utc_now()
        self._progress.bytes_received += new_bytes
        self._progress.lines_received += new_lines
        self._progress.last_activity_at = now

        # Record snapshot if interval has passed
        should_snapshot = force_snapshot or (
            (now - self._last_snapshot_at).total_seconds()
            >= self._snapshot_interval_seconds
        )

        if should_snapshot:
            self._record_snapshot()
            self._last_snapshot_at = now

        # Notify callback
        if self.callback is not None:
            self.callback(self._progress)

    def set_phase(self, phase: str) -> None:
        """Set the current execution phase.

        Records a snapshot when phase changes to track phase transitions.

        Args:
            phase: New phase (starting, executing, validating, completed).
        """
        if self._progress.phase != phase:
            self._progress.phase = phase
            self._progress.last_activity_at = utc_now()
            # Record snapshot on phase change
            self._record_snapshot()
            self._last_snapshot_at = utc_now()

            if self.callback is not None:
                self.callback(self._progress)

    def get_progress(self) -> ExecutionProgress:
        """Get current progress state.

        Returns:
            Copy of current ExecutionProgress.
        """
        return ExecutionProgress(
            started_at=self._progress.started_at,
            last_activity_at=self._progress.last_activity_at,
            bytes_received=self._progress.bytes_received,
            lines_received=self._progress.lines_received,
            phase=self._progress.phase,
        )

    def get_snapshots(self) -> list[dict[str, Any]]:
        """Get all recorded progress snapshots.

        Returns:
            List of snapshot dictionaries for persistence.
        """
        return self._snapshots.copy()

    def _record_snapshot(self) -> None:
        """Record a progress snapshot for persistence."""
        snapshot = self._progress.to_dict()
        snapshot["snapshot_at"] = utc_now().isoformat()
        self._snapshots.append(snapshot)

    def reset(self) -> None:
        """Reset tracker for new execution.

        Clears all counters and snapshots but preserves callback.
        """
        now = utc_now()
        self._progress = ExecutionProgress(
            started_at=now,
            last_activity_at=now,
            phase="starting",
        )
        self._last_snapshot_at = now
        self._snapshots.clear()


class StreamingOutputTracker:
    """Helper for tracking streaming output from subprocess.

    Wraps a ProgressTracker to process raw bytes/lines and update counters.
    Useful for tracking output from asyncio.subprocess streams.

    Example:
        tracker = ProgressTracker()
        stream_tracker = StreamingOutputTracker(tracker)

        async for chunk in stream:
            stream_tracker.process_chunk(chunk)
    """

    def __init__(self, progress_tracker: ProgressTracker) -> None:
        """Initialize streaming output tracker.

        Args:
            progress_tracker: Underlying ProgressTracker to update.
        """
        self._tracker = progress_tracker
        self._partial_line: bytes = b""

    def process_chunk(self, chunk: bytes) -> None:
        """Process a chunk of output bytes.

        Counts complete lines and updates byte/line counters.

        Args:
            chunk: Raw bytes from the output stream.
        """
        if not chunk:
            return

        # Count bytes
        chunk_bytes = len(chunk)

        # Count lines (complete lines only)
        combined = self._partial_line + chunk
        lines = combined.split(b"\n")

        # Last element is partial line (or empty if chunk ended with \n)
        self._partial_line = lines[-1]
        complete_lines = len(lines) - 1

        self._tracker.update(new_bytes=chunk_bytes, new_lines=complete_lines)

    def finish(self) -> None:
        """Finish tracking, counting any remaining partial line."""
        if self._partial_line:
            # Count the partial line as a complete line
            self._tracker.update(new_lines=1)
            self._partial_line = b""
