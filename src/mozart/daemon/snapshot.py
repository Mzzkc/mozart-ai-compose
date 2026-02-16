"""Completion snapshot manager for the Mozart daemon.

Captures durable snapshots of workspace artifacts at job completion
and manages TTL-based cleanup to prevent unbounded storage growth.

Snapshots are stored under ``~/.mozart/snapshots/{job_id}/{timestamp}/``
and include key artifacts: state JSON, logs, and validation results.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from mozart.core.logging import get_logger

_logger = get_logger("daemon.snapshot")

# File patterns to capture in a snapshot (glob patterns relative to workspace).
_CAPTURE_PATTERNS = [
    "*.json",       # State files (e.g., {job_id}.json)
    "mozart.log",   # Job execution log
    "*.log",        # Any additional log files
]


class SnapshotManager:
    """Captures and manages completion snapshots for daemon jobs.

    Snapshots are lightweight copies of key workspace artifacts,
    stored under a persistent directory with TTL-based cleanup.

    Usage::

        manager = SnapshotManager(base_dir=Path("~/.mozart/snapshots"))
        path = manager.capture("my-job", Path("/workspace"))
        manager.cleanup(max_age_hours=168)
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = (base_dir or Path("~/.mozart/snapshots")).expanduser()

    @property
    def base_dir(self) -> Path:
        """Base directory for all snapshots."""
        return self._base_dir

    def capture(self, job_id: str, workspace: Path) -> str | None:
        """Capture a snapshot of workspace artifacts at job completion.

        Args:
            job_id: The job identifier.
            workspace: Path to the job's workspace directory.

        Returns:
            The snapshot directory path as a string, or None if capture
            failed (e.g., workspace doesn't exist or is empty).
        """
        if not workspace.is_dir():
            _logger.warning(
                "snapshot.workspace_missing",
                job_id=job_id,
                workspace=str(workspace),
            )
            return None

        timestamp = str(int(time.time()))
        snapshot_dir = self._base_dir / job_id / timestamp
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for pattern in _CAPTURE_PATTERNS:
            for src in workspace.glob(pattern):
                if not src.is_file():
                    continue
                dst = snapshot_dir / src.name
                # Avoid overwriting if multiple patterns match the same file
                if dst.exists():
                    continue
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except OSError as exc:
                    _logger.warning(
                        "snapshot.copy_failed",
                        job_id=job_id,
                        file=src.name,
                        error=str(exc),
                    )

        if copied == 0:
            # Remove empty snapshot dir if nothing was captured
            try:
                snapshot_dir.rmdir()
            except OSError:
                pass
            _logger.info(
                "snapshot.empty",
                job_id=job_id,
                workspace=str(workspace),
            )
            return None

        _logger.info(
            "snapshot.captured",
            job_id=job_id,
            path=str(snapshot_dir),
            files=copied,
        )
        return str(snapshot_dir)

    def cleanup(self, max_age_hours: int = 168) -> int:
        """Remove snapshots older than the given TTL.

        Args:
            max_age_hours: Maximum age in hours. Snapshots older than
                this are deleted. Defaults to 168 (1 week).

        Returns:
            Number of snapshot directories removed.
        """
        if not self._base_dir.is_dir():
            return 0

        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0

        # Walk: base_dir / {job_id} / {timestamp} /
        for job_dir in self._safe_iterdir(self._base_dir):
            if not job_dir.is_dir():
                continue

            for ts_dir in self._safe_iterdir(job_dir):
                if not ts_dir.is_dir():
                    continue

                # Timestamp is the directory name
                try:
                    ts = int(ts_dir.name)
                except ValueError:
                    continue

                if ts < cutoff:
                    try:
                        shutil.rmtree(ts_dir)
                        removed += 1
                    except OSError as exc:
                        _logger.warning(
                            "snapshot.cleanup_failed",
                            path=str(ts_dir),
                            error=str(exc),
                        )

            # Remove empty job directories
            try:
                if job_dir.is_dir() and not any(job_dir.iterdir()):
                    job_dir.rmdir()
            except OSError:
                pass

        if removed > 0:
            _logger.info("snapshot.cleanup", removed=removed)
        return removed

    def list_snapshots(self, job_id: str) -> list[dict[str, str]]:
        """List all snapshots for a given job.

        Args:
            job_id: The job identifier.

        Returns:
            List of dicts with ``timestamp`` and ``path`` keys,
            sorted newest-first.
        """
        job_dir = self._base_dir / job_id
        if not job_dir.is_dir():
            return []

        snapshots: list[dict[str, str]] = []
        for ts_dir in self._safe_iterdir(job_dir):
            if not ts_dir.is_dir():
                continue
            try:
                int(ts_dir.name)  # Validate timestamp format
            except ValueError:
                continue
            snapshots.append({
                "timestamp": ts_dir.name,
                "path": str(ts_dir),
            })

        snapshots.sort(key=lambda s: s["timestamp"], reverse=True)
        return snapshots

    @staticmethod
    def _safe_iterdir(path: Path) -> list[Path]:
        """Iterate a directory, returning empty list on errors."""
        try:
            return list(path.iterdir())
        except OSError:
            return []
