"""Persistent job registry for the Mozart daemon.

SQLite-backed registry that tracks all jobs submitted to the daemon.
Survives daemon restarts so ``mozart list`` always shows job history.

Separate from the learning store (which tracks patterns across jobs).
This DB tracks operational state: which jobs exist, their workspaces,
PIDs, and statuses.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mozart.core.logging import get_logger

_logger = get_logger("daemon.registry")

@dataclass
class JobRecord:
    """A single job's registry entry."""

    job_id: str
    config_path: str
    workspace: str
    status: str = "queued"
    pid: int | None = None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON-RPC responses."""
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "config_path": self.config_path,
            "workspace": self.workspace,
            "status": self.status,
            "pid": self.pid,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.error_message:
            result["error_message"] = self.error_message
        return result


class JobRegistry:
    """SQLite-backed persistent job registry.

    Thread-safe: SQLite handles concurrency. The daemon is single-threaded
    (asyncio) so contention is minimal, but the DB is safe for external
    readers (e.g. a monitoring tool reading the same file).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        _logger.info("registry.opened", path=str(db_path))

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                config_path TEXT NOT NULL,
                workspace TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                pid INTEGER,
                submitted_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                error_message TEXT
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs (status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_submitted
            ON jobs (submitted_at DESC)
        """)
        self._conn.commit()

    def register_job(
        self,
        job_id: str,
        config_path: Path,
        workspace: Path,
    ) -> None:
        """Register a newly submitted job."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO jobs
                (job_id, config_path, workspace, status, submitted_at)
            VALUES (?, ?, ?, 'queued', ?)
            """,
            (job_id, str(config_path), str(workspace), time.time()),
        )
        self._conn.commit()

    def update_status(
        self,
        job_id: str,
        status: str,
        *,
        pid: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a job's status and optional fields."""
        updates = ["status = ?"]
        params: list[Any] = [status]

        if pid is not None:
            updates.append("pid = ?")
            params.append(pid)

        if status == "running" and pid is not None:
            updates.append("started_at = ?")
            params.append(time.time())

        if status in ("completed", "failed", "cancelled"):
            updates.append("completed_at = ?")
            params.append(time.time())

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        params.append(job_id)
        self._conn.execute(
            f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?",
            params,
        )
        self._conn.commit()

    def get_job(self, job_id: str) -> JobRecord | None:
        """Get a single job by ID."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_jobs(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[JobRecord]:
        """List jobs, most recent first."""
        if status:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY submitted_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def has_active_job(self, job_id: str) -> bool:
        """Check if a job ID exists and is in an active state."""
        row = self._conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ? AND status IN ('queued', 'running')",
            (job_id,),
        ).fetchone()
        return row is not None

    def get_orphaned_jobs(self) -> list[JobRecord]:
        """Find jobs that were running when the daemon last stopped.

        These are jobs with status 'queued' or 'running' — after a daemon
        restart they're orphans since their asyncio tasks no longer exist.
        """
        rows = self._conn.execute(
            "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
            "ORDER BY submitted_at DESC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def mark_orphans_failed(self) -> int:
        """Mark all orphaned jobs as failed on daemon startup.

        Returns the number of jobs marked.
        """
        cursor = self._conn.execute(
            """
            UPDATE jobs SET
                status = 'failed',
                completed_at = ?,
                error_message = 'Daemon restarted while job was active'
            WHERE status IN ('queued', 'running')
            """,
            (time.time(),),
        )
        self._conn.commit()
        count = cursor.rowcount
        if count > 0:
            _logger.warning("registry.orphans_marked_failed", count=count)
        return count

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            config_path=row["config_path"],
            workspace=row["workspace"],
            status=row["status"],
            pid=row["pid"],
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )
