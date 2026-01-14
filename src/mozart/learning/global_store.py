"""Global learning store for cross-workspace pattern persistence.

This module implements the global learning system designed in Movement III:
- SQLite-based storage at ~/.mozart/global-learning.db
- Thread-safe file operations with WAL mode
- Schema as defined in design document
- Integration with existing PatternDetector and outcome storage

The global store enables Mozart to learn across all workspaces, aggregating
patterns and error recoveries to improve over time.
"""

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from mozart.core.logging import get_logger
from mozart.learning.outcomes import SheetOutcome

# Module-level logger for global learning store
_logger = get_logger("learning.global_store")


# Default location following XDG conventions
DEFAULT_GLOBAL_STORE_PATH = Path.home() / ".mozart" / "global-learning.db"


@dataclass
class ExecutionRecord:
    """A record of a sheet execution stored in the global database."""

    id: str
    workspace_hash: str
    job_hash: str
    sheet_num: int
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float
    status: str
    retry_count: int
    first_attempt_success: bool
    validation_pass_rate: float
    confidence_score: float
    model: str | None
    error_codes: list[str] = field(default_factory=list)


@dataclass
class PatternRecord:
    """A pattern record stored in the global database."""

    id: str
    pattern_type: str
    pattern_name: str
    description: str | None
    occurrence_count: int
    first_seen: datetime
    last_seen: datetime
    last_confirmed: datetime
    led_to_success_count: int
    led_to_failure_count: int
    effectiveness_score: float
    variance: float
    suggested_action: str | None
    context_tags: list[str]
    priority_score: float


@dataclass
class ErrorRecoveryRecord:
    """A record of error recovery timing for learning adaptive waits."""

    id: str
    error_code: str
    suggested_wait: float
    actual_wait: float
    recovery_success: bool
    recorded_at: datetime
    model: str | None
    time_of_day: int  # Hour 0-23


@dataclass
class RateLimitEvent:
    """A rate limit event for cross-workspace coordination.

    Evolution #8: Tracks rate limit events across workspaces so that
    parallel jobs can coordinate and avoid hitting the same rate limits.
    """

    id: str
    error_code: str
    model: str | None
    recorded_at: datetime
    expires_at: datetime
    source_job_hash: str
    duration_seconds: float


class GlobalLearningStore:
    """SQLite-based global learning store.

    Provides persistent storage for execution outcomes, detected patterns,
    and error recovery data across all Mozart workspaces. Uses WAL mode
    for safe concurrent access.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    SCHEMA_VERSION = 2  # v2: Added rate_limit_events table

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the global learning store.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to ~/.mozart/global-learning.db
        """
        self.db_path = db_path or DEFAULT_GLOBAL_STORE_PATH
        self._ensure_db_exists()
        self._migrate_if_needed()

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory and file exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper configuration.

        Uses WAL mode for better concurrent access and enables foreign keys.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate_if_needed(self) -> None:
        """Create or migrate the database schema."""
        with self._get_connection() as conn:
            # Check current version
            try:
                cursor = conn.execute(
                    "SELECT version FROM schema_version LIMIT 1"
                )
                row = cursor.fetchone()
                current_version = row["version"] if row else 0
            except sqlite3.OperationalError:
                current_version = 0

            if current_version < self.SCHEMA_VERSION:
                self._create_schema(conn)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create the database schema."""
        # Schema version table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        """)

        # Executions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id TEXT PRIMARY KEY,
                workspace_hash TEXT NOT NULL,
                job_hash TEXT NOT NULL,
                sheet_num INTEGER NOT NULL,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                duration_seconds REAL,
                status TEXT,
                retry_count INTEGER DEFAULT 0,
                first_attempt_success BOOLEAN,
                validation_pass_rate REAL,
                confidence_score REAL,
                model TEXT,
                error_codes TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_workspace_job "
            "ON executions(workspace_hash, job_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_status ON executions(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_exec_started ON executions(started_at)"
        )

        # Patterns table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                description TEXT,
                occurrence_count INTEGER DEFAULT 1,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                last_confirmed TIMESTAMP,
                led_to_success_count INTEGER DEFAULT 0,
                led_to_failure_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.5,
                variance REAL DEFAULT 0.0,
                suggested_action TEXT,
                context_tags TEXT,
                priority_score REAL DEFAULT 0.5
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_priority "
            "ON patterns(priority_score DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_confirmed "
            "ON patterns(last_confirmed)"
        )

        # Pattern applications table (for effectiveness tracking)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_applications (
                id TEXT PRIMARY KEY,
                pattern_id TEXT REFERENCES patterns(id),
                execution_id TEXT REFERENCES executions(id),
                applied_at TIMESTAMP,
                outcome_improved BOOLEAN,
                retry_count_before INTEGER,
                retry_count_after INTEGER
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_app_pattern "
            "ON pattern_applications(pattern_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_app_execution "
            "ON pattern_applications(execution_id)"
        )

        # Error recoveries table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_recoveries (
                id TEXT PRIMARY KEY,
                error_code TEXT NOT NULL,
                suggested_wait REAL,
                actual_wait REAL,
                recovery_success BOOLEAN,
                recorded_at TIMESTAMP,
                model TEXT,
                time_of_day INTEGER
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recovery_code "
            "ON error_recoveries(error_code)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recovery_success "
            "ON error_recoveries(recovery_success)"
        )

        # Workspace clusters table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_clusters (
                workspace_hash TEXT PRIMARY KEY,
                cluster_id TEXT,
                assigned_at TIMESTAMP
            )
        """)

        # Rate limit events table (Evolution #8: Cross-Workspace Circuit Breaker)
        # Tracks rate limit events across workspaces for coordination
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rate_limit_events (
                id TEXT PRIMARY KEY,
                error_code TEXT NOT NULL,
                model TEXT,
                recorded_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                source_job_hash TEXT NOT NULL,
                duration_seconds REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_code "
            "ON rate_limit_events(error_code)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_expires "
            "ON rate_limit_events(expires_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rate_limit_model "
            "ON rate_limit_events(model)"
        )

        # Update schema version
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (self.SCHEMA_VERSION,))

        _logger.info(f"Created global learning schema v{self.SCHEMA_VERSION}")

    @staticmethod
    def hash_workspace(workspace_path: Path) -> str:
        """Generate a stable hash for a workspace path.

        Args:
            workspace_path: The absolute path to the workspace.

        Returns:
            A hex string hash of the workspace path.
        """
        normalized = str(workspace_path.resolve())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def hash_job(job_name: str, config_hash: str | None = None) -> str:
        """Generate a stable hash for a job.

        Args:
            job_name: The job name.
            config_hash: Optional hash of the job config for versioning.

        Returns:
            A hex string hash of the job.
        """
        combined = f"{job_name}:{config_hash or ''}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def record_outcome(
        self,
        outcome: SheetOutcome,
        workspace_path: Path,
        model: str | None = None,
        error_codes: list[str] | None = None,
    ) -> str:
        """Record a sheet outcome to the global store.

        Args:
            outcome: The SheetOutcome to record.
            workspace_path: Path to the workspace for hashing.
            model: Optional model name used for execution.
            error_codes: Optional list of error codes encountered.

        Returns:
            The execution record ID.
        """
        execution_id = str(uuid.uuid4())
        workspace_hash = self.hash_workspace(workspace_path)
        job_hash = self.hash_job(outcome.job_id)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO executions (
                    id, workspace_hash, job_hash, sheet_num,
                    started_at, completed_at, duration_seconds,
                    status, retry_count, first_attempt_success,
                    validation_pass_rate, confidence_score, model, error_codes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    workspace_hash,
                    job_hash,
                    self._extract_sheet_num(outcome.sheet_id),
                    outcome.timestamp.isoformat(),
                    datetime.now().isoformat(),
                    outcome.execution_duration,
                    outcome.final_status.value,
                    outcome.retry_count,
                    outcome.first_attempt_success,
                    outcome.validation_pass_rate,
                    self._calculate_confidence(outcome),
                    model,
                    json.dumps(error_codes or []),
                ),
            )

        _logger.debug(
            f"Recorded outcome {execution_id} for sheet {outcome.sheet_id}"
        )
        return execution_id

    def _extract_sheet_num(self, sheet_id: str) -> int:
        """Extract sheet number from sheet ID.

        Handles various formats:
        - "job-sheet1" -> 1
        - "job-1" -> 1
        - "sheet1" -> 1
        - "1" -> 1
        """
        import re

        # Try to find a number at the end of the string
        match = re.search(r"(\d+)$", sheet_id)
        if match:
            return int(match.group(1))
        return 0

    def _calculate_confidence(self, outcome: SheetOutcome) -> float:
        """Calculate a confidence score for an outcome.

        Uses validation results and retry count to estimate confidence.
        """
        base_confidence = outcome.validation_pass_rate

        # Penalize high retry counts
        retry_penalty = 0.1 * min(outcome.retry_count, 5)
        base_confidence -= retry_penalty

        # Boost first-attempt success
        if outcome.first_attempt_success:
            base_confidence = min(1.0, base_confidence + 0.1)

        return max(0.0, min(1.0, base_confidence))

    def record_pattern(
        self,
        pattern_type: str,
        pattern_name: str,
        description: str | None = None,
        context_tags: list[str] | None = None,
        suggested_action: str | None = None,
    ) -> str:
        """Record or update a pattern in the global store.

        If a pattern with the same type and name exists, increments its count.
        Otherwise, creates a new pattern.

        Args:
            pattern_type: The type of pattern (e.g., 'validation_failure').
            pattern_name: A unique name for this pattern.
            description: Human-readable description.
            context_tags: Tags for matching context.
            suggested_action: Recommended action for this pattern.

        Returns:
            The pattern ID.
        """
        now = datetime.now().isoformat()
        pattern_id = hashlib.sha256(
            f"{pattern_type}:{pattern_name}".encode()
        ).hexdigest()[:16]

        with self._get_connection() as conn:
            # Check if pattern exists
            cursor = conn.execute(
                "SELECT id, occurrence_count FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing pattern
                conn.execute(
                    """
                    UPDATE patterns SET
                        occurrence_count = occurrence_count + 1,
                        last_seen = ?,
                        description = COALESCE(?, description),
                        suggested_action = COALESCE(?, suggested_action),
                        context_tags = ?
                    WHERE id = ?
                    """,
                    (
                        now,
                        description,
                        suggested_action,
                        json.dumps(context_tags or []),
                        pattern_id,
                    ),
                )
            else:
                # Insert new pattern
                conn.execute(
                    """
                    INSERT INTO patterns (
                        id, pattern_type, pattern_name, description,
                        occurrence_count, first_seen, last_seen, last_confirmed,
                        led_to_success_count, led_to_failure_count,
                        effectiveness_score, variance, suggested_action,
                        context_tags, priority_score
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, 0, 0, 0.5, 0.0, ?, ?, 0.5)
                    """,
                    (
                        pattern_id,
                        pattern_type,
                        pattern_name,
                        description,
                        now,
                        now,
                        now,
                        suggested_action,
                        json.dumps(context_tags or []),
                    ),
                )

        return pattern_id

    def record_pattern_application(
        self,
        pattern_id: str,
        execution_id: str,
        outcome_improved: bool,
        retry_count_before: int = 0,
        retry_count_after: int = 0,
    ) -> str:
        """Record that a pattern was applied to an execution.

        This creates the feedback loop for effectiveness tracking.

        Args:
            pattern_id: The pattern that was applied.
            execution_id: The execution it was applied to.
            outcome_improved: Whether the outcome was better than baseline.
            retry_count_before: Retry count before pattern applied.
            retry_count_after: Retry count after pattern applied.

        Returns:
            The application record ID.
        """
        app_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pattern_applications (
                    id, pattern_id, execution_id, applied_at,
                    outcome_improved, retry_count_before, retry_count_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_id,
                    pattern_id,
                    execution_id,
                    now,
                    outcome_improved,
                    retry_count_before,
                    retry_count_after,
                ),
            )

            # Update pattern effectiveness
            if outcome_improved:
                conn.execute(
                    """
                    UPDATE patterns SET
                        led_to_success_count = led_to_success_count + 1,
                        last_confirmed = ?
                    WHERE id = ?
                    """,
                    (now, pattern_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE patterns SET
                        led_to_failure_count = led_to_failure_count + 1
                    WHERE id = ?
                    """,
                    (pattern_id,),
                )

        return app_id

    def record_error_recovery(
        self,
        error_code: str,
        suggested_wait: float,
        actual_wait: float,
        recovery_success: bool,
        model: str | None = None,
    ) -> str:
        """Record an error recovery for learning adaptive wait times.

        Args:
            error_code: The error code (e.g., 'E103').
            suggested_wait: The initially suggested wait time in seconds.
            actual_wait: The actual wait time used in seconds.
            recovery_success: Whether recovery after waiting succeeded.
            model: Optional model name.

        Returns:
            The recovery record ID.
        """
        record_id = str(uuid.uuid4())
        now = datetime.now()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO error_recoveries (
                    id, error_code, suggested_wait, actual_wait,
                    recovery_success, recorded_at, model, time_of_day
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    error_code,
                    suggested_wait,
                    actual_wait,
                    recovery_success,
                    now.isoformat(),
                    model,
                    now.hour,
                ),
            )

        return record_id

    def get_learned_wait_time(
        self,
        error_code: str,
        model: str | None = None,
        min_samples: int = 3,
    ) -> float | None:
        """Get the learned optimal wait time for an error code.

        Analyzes past error recoveries to suggest an adaptive wait time.

        Args:
            error_code: The error code to look up.
            model: Optional model to filter by.
            min_samples: Minimum samples required before learning.

        Returns:
            Suggested wait time in seconds, or None if not enough data.
        """
        with self._get_connection() as conn:
            if model:
                cursor = conn.execute(
                    """
                    SELECT actual_wait, recovery_success
                    FROM error_recoveries
                    WHERE error_code = ? AND model = ? AND recovery_success = 1
                    ORDER BY recorded_at DESC
                    LIMIT 20
                    """,
                    (error_code, model),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT actual_wait, recovery_success
                    FROM error_recoveries
                    WHERE error_code = ? AND recovery_success = 1
                    ORDER BY recorded_at DESC
                    LIMIT 20
                    """,
                    (error_code,),
                )

            rows = cursor.fetchall()
            if len(rows) < min_samples:
                return None

            # Use weighted average favoring shorter successful waits
            waits: list[float] = [float(row["actual_wait"]) for row in rows]
            # Lower waits are better, so we use harmonic mean-like weighting
            avg_wait = sum(waits) / len(waits)
            min_successful = min(waits)

            # Blend toward shorter waits that still work
            return (avg_wait + min_successful) / 2

    def get_learned_wait_time_with_fallback(
        self,
        error_code: str,
        static_delay: float,
        model: str | None = None,
        min_samples: int = 3,
        min_confidence: float = 0.7,
    ) -> tuple[float, float, str]:
        """Get learned wait time with fallback to static delay and confidence.

        This method bridges the global learning store with retry strategies.
        It returns a delay value along with a confidence score indicating
        how much to trust the learned value.

        Evolution #3: Learned Wait Time Injection - provides the bridge between
        global_store's cross-workspace learned delays and retry_strategy's
        blend_historical_delay() method.

        Args:
            error_code: The error code to look up (e.g., 'E101').
            static_delay: Fallback static delay if no learned data available.
            model: Optional model to filter by.
            min_samples: Minimum samples required for learning.
            min_confidence: Minimum confidence threshold for using learned delay.

        Returns:
            Tuple of (delay_seconds, confidence, strategy_name).
            - delay_seconds: The recommended delay (learned or static).
            - confidence: Confidence in the recommendation (0.0-1.0).
              High confidence (>=0.7) means learned delay is reliable.
              Low confidence (<0.7) means learned delay should be blended with static.
            - strategy_name: "global_learned" | "global_learned_blend" | "static_fallback"
        """
        # Query learned wait time from global store
        learned_wait = self.get_learned_wait_time(
            error_code=error_code,
            model=model,
            min_samples=min_samples,
        )

        if learned_wait is None:
            # No learned data - fall back to static
            return static_delay, 0.0, "static_fallback"

        # Calculate confidence based on sample count
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM error_recoveries
                WHERE error_code = ? AND recovery_success = 1
                """,
                (error_code,),
            )
            sample_count = cursor.fetchone()["count"]

        # Confidence scales with sample count: 10 samples = 1.0 confidence
        confidence = min(sample_count / 10.0, 1.0)

        # Apply floor: learned delay shouldn't be less than 50% of static
        # This prevents overly aggressive waits that might fail
        delay_floor = static_delay * 0.5

        if confidence >= min_confidence:
            # High confidence: use learned delay (with floor)
            final_delay = max(learned_wait, delay_floor)
            return final_delay, confidence, "global_learned"
        else:
            # Low confidence: blend learned with static
            # weight = confidence / min_confidence (0.0 to 1.0)
            weight = confidence / min_confidence
            blended = weight * max(learned_wait, delay_floor) + (1 - weight) * static_delay
            return blended, confidence, "global_learned_blend"

    def get_error_recovery_sample_count(self, error_code: str) -> int:
        """Get the number of successful recovery samples for an error code.

        Args:
            error_code: The error code to query.

        Returns:
            Number of successful recovery samples.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM error_recoveries
                WHERE error_code = ? AND recovery_success = 1
                """,
                (error_code,),
            )
            row = cursor.fetchone()
            return int(row["count"]) if row else 0

    def get_patterns(
        self,
        pattern_type: str | None = None,
        min_priority: float = 0.3,
        limit: int = 20,
        context_tags: list[str] | None = None,
    ) -> list[PatternRecord]:
        """Get patterns from the global store.

        Args:
            pattern_type: Optional filter by pattern type.
            min_priority: Minimum priority score to include.
            limit: Maximum number of patterns to return.
            context_tags: Optional list of tags for context-based filtering.
                         Patterns match if ANY of their tags match ANY query tag.
                         If None or empty, no tag filtering is applied.

        Returns:
            List of PatternRecord objects sorted by priority.
        """
        with self._get_connection() as conn:
            # Build query dynamically based on filters
            where_clauses = ["priority_score >= ?"]
            params: list[Any] = [min_priority]

            if pattern_type:
                where_clauses.append("pattern_type = ?")
                params.append(pattern_type)

            # Context tag filtering: match if ANY pattern tag matches ANY query tag
            # Uses json_each() to iterate over the JSON array stored in context_tags
            # Note: Explicitly check for non-empty list to handle [] vs None correctly
            # Safe from SQL injection: placeholders only contain "?" characters,
            # actual tag values are bound via params.extend()
            if context_tags is not None and len(context_tags) > 0:
                tag_placeholders = ", ".join("?" for _ in context_tags)
                where_clauses.append(
                    f"""EXISTS (
                        SELECT 1 FROM json_each(context_tags)
                        WHERE json_each.value IN ({tag_placeholders})
                    )"""
                )
                params.extend(context_tags)

            params.append(limit)
            query = f"""
                SELECT * FROM patterns
                WHERE {" AND ".join(where_clauses)}
                ORDER BY priority_score DESC
                LIMIT ?
            """
            cursor = conn.execute(query, tuple(params))

            records = []
            for row in cursor.fetchall():
                records.append(
                    PatternRecord(
                        id=row["id"],
                        pattern_type=row["pattern_type"],
                        pattern_name=row["pattern_name"],
                        description=row["description"],
                        occurrence_count=row["occurrence_count"],
                        first_seen=datetime.fromisoformat(row["first_seen"]),
                        last_seen=datetime.fromisoformat(row["last_seen"]),
                        last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
                        led_to_success_count=row["led_to_success_count"],
                        led_to_failure_count=row["led_to_failure_count"],
                        effectiveness_score=row["effectiveness_score"],
                        variance=row["variance"],
                        suggested_action=row["suggested_action"],
                        context_tags=json.loads(row["context_tags"] or "[]"),
                        priority_score=row["priority_score"],
                    )
                )

            return records

    def get_execution_stats(self) -> dict[str, Any]:
        """Get aggregate statistics from the global store.

        Returns:
            Dictionary with stats like total_executions, success_rate, etc.
        """
        with self._get_connection() as conn:
            stats: dict[str, Any] = {}

            # Total executions
            cursor = conn.execute("SELECT COUNT(*) as count FROM executions")
            stats["total_executions"] = cursor.fetchone()["count"]

            # First-attempt success rate
            cursor = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN first_attempt_success THEN 1 ELSE 0 END) as successes,
                    COUNT(*) as total
                FROM executions
                """
            )
            row = cursor.fetchone()
            if row["total"] > 0:
                stats["first_attempt_success_rate"] = row["successes"] / row["total"]
            else:
                stats["first_attempt_success_rate"] = 0.0

            # Total patterns
            cursor = conn.execute("SELECT COUNT(*) as count FROM patterns")
            stats["total_patterns"] = cursor.fetchone()["count"]

            # Average pattern effectiveness
            cursor = conn.execute(
                "SELECT AVG(effectiveness_score) as avg FROM patterns"
            )
            row = cursor.fetchone()
            stats["avg_pattern_effectiveness"] = row["avg"] or 0.0

            # Total error recoveries
            cursor = conn.execute("SELECT COUNT(*) as count FROM error_recoveries")
            stats["total_error_recoveries"] = cursor.fetchone()["count"]

            # Unique workspaces
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT workspace_hash) as count FROM executions"
            )
            stats["unique_workspaces"] = cursor.fetchone()["count"]

            return stats

    def get_recent_executions(
        self,
        limit: int = 20,
        workspace_hash: str | None = None,
    ) -> list[ExecutionRecord]:
        """Get recent execution records.

        Args:
            limit: Maximum number of records to return.
            workspace_hash: Optional filter by workspace.

        Returns:
            List of ExecutionRecord objects.
        """
        with self._get_connection() as conn:
            if workspace_hash:
                cursor = conn.execute(
                    """
                    SELECT * FROM executions
                    WHERE workspace_hash = ?
                    ORDER BY completed_at DESC
                    LIMIT ?
                    """,
                    (workspace_hash, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM executions
                    ORDER BY completed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            records = []
            for row in cursor.fetchall():
                records.append(
                    ExecutionRecord(
                        id=row["id"],
                        workspace_hash=row["workspace_hash"],
                        job_hash=row["job_hash"],
                        sheet_num=row["sheet_num"],
                        started_at=datetime.fromisoformat(row["started_at"])
                        if row["started_at"]
                        else None,
                        completed_at=datetime.fromisoformat(row["completed_at"])
                        if row["completed_at"]
                        else None,
                        duration_seconds=row["duration_seconds"] or 0.0,
                        status=row["status"] or "",
                        retry_count=row["retry_count"] or 0,
                        first_attempt_success=bool(row["first_attempt_success"]),
                        validation_pass_rate=row["validation_pass_rate"] or 0.0,
                        confidence_score=row["confidence_score"] or 0.0,
                        model=row["model"],
                        error_codes=json.loads(row["error_codes"] or "[]"),
                    )
                )

            return records

    # =========================================================================
    # Evolution #8: Cross-Workspace Circuit Breaker
    # =========================================================================

    def record_rate_limit_event(
        self,
        error_code: str,
        duration_seconds: float,
        job_id: str,
        model: str | None = None,
    ) -> str:
        """Record a rate limit event for cross-workspace coordination.

        When one job hits a rate limit, it records the event so that other
        parallel jobs can check and avoid hitting the same limit.

        Evolution #8: Cross-Workspace Circuit Breaker - enables jobs running
        in different workspaces to share rate limit awareness.

        Args:
            error_code: The error code (e.g., 'E101', 'E102').
            duration_seconds: Expected rate limit duration in seconds.
            job_id: ID of the job that encountered the rate limit.
            model: Optional model name that triggered the limit.

        Returns:
            The rate limit event record ID.
        """
        from datetime import timedelta

        record_id = str(uuid.uuid4())
        now = datetime.now()
        job_hash = self.hash_job(job_id)

        # Use 80% of expected duration as expiry (conservative TTL)
        # This avoids waiting too long while still being safe
        expires_at = now + timedelta(seconds=duration_seconds * 0.8)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO rate_limit_events (
                    id, error_code, model, recorded_at, expires_at,
                    source_job_hash, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    error_code,
                    model,
                    now.isoformat(),
                    expires_at.isoformat(),
                    job_hash,
                    duration_seconds,
                ),
            )

        _logger.info(
            f"Recorded rate limit event {record_id}: {error_code} "
            f"expires in {duration_seconds * 0.8:.0f}s"
        )
        return record_id

    def is_rate_limited(
        self,
        error_code: str | None = None,
        model: str | None = None,
    ) -> tuple[bool, float | None]:
        """Check if there's an active rate limit from another job.

        Queries the rate_limit_events table to see if any unexpired
        rate limit events exist that would affect this job.

        Evolution #8: Cross-Workspace Circuit Breaker - allows jobs to
        check if another job has already hit a rate limit.

        Args:
            error_code: Optional error code to filter by. If None, checks any.
            model: Optional model to filter by. If None, checks any.

        Returns:
            Tuple of (is_limited: bool, seconds_until_expiry: float | None).
            If is_limited is True, seconds_until_expiry indicates when it clears.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            # Build query based on filters
            query = """
                SELECT expires_at, error_code, model, duration_seconds
                FROM rate_limit_events
                WHERE expires_at > ?
            """
            params: list[str] = [now.isoformat()]

            if error_code is not None:
                query += " AND error_code = ?"
                params.append(error_code)

            if model is not None:
                query += " AND model = ?"
                params.append(model)

            query += " ORDER BY expires_at DESC LIMIT 1"

            cursor = conn.execute(query, params)
            row = cursor.fetchone()

            if row is None:
                return False, None

            # Calculate time until expiry
            expires_at = datetime.fromisoformat(row["expires_at"])
            seconds_until_expiry = (expires_at - now).total_seconds()

            if seconds_until_expiry <= 0:
                return False, None

            _logger.debug(
                f"Rate limit active: {row['error_code']} "
                f"(expires in {seconds_until_expiry:.0f}s)"
            )
            return True, seconds_until_expiry

    def get_active_rate_limits(
        self,
        model: str | None = None,
    ) -> list[RateLimitEvent]:
        """Get all active (unexpired) rate limit events.

        Args:
            model: Optional model to filter by.

        Returns:
            List of RateLimitEvent objects that haven't expired yet.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            if model:
                cursor = conn.execute(
                    """
                    SELECT * FROM rate_limit_events
                    WHERE expires_at > ? AND model = ?
                    ORDER BY expires_at DESC
                    """,
                    (now.isoformat(), model),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM rate_limit_events
                    WHERE expires_at > ?
                    ORDER BY expires_at DESC
                    """,
                    (now.isoformat(),),
                )

            events = []
            for row in cursor.fetchall():
                events.append(
                    RateLimitEvent(
                        id=row["id"],
                        error_code=row["error_code"],
                        model=row["model"],
                        recorded_at=datetime.fromisoformat(row["recorded_at"]),
                        expires_at=datetime.fromisoformat(row["expires_at"]),
                        source_job_hash=row["source_job_hash"],
                        duration_seconds=row["duration_seconds"],
                    )
                )

            return events

    def cleanup_expired_rate_limits(self) -> int:
        """Remove expired rate limit events from the database.

        This is a housekeeping method that can be called periodically
        to prevent the rate_limit_events table from growing unbounded.

        Returns:
            Number of expired records deleted.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM rate_limit_events WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            deleted_count = cursor.rowcount

        if deleted_count > 0:
            _logger.debug(f"Cleaned up {deleted_count} expired rate limit events")

        return deleted_count

    # =========================================================================
    # Learning Activation: Similar Executions and Optimal Timing
    # =========================================================================

    def get_similar_executions(
        self,
        job_hash: str | None = None,
        workspace_hash: str | None = None,
        sheet_num: int | None = None,
        limit: int = 10,
    ) -> list[ExecutionRecord]:
        """Get similar historical executions for learning.

        Learning Activation: Enables querying executions that are similar to
        the current context, supporting pattern-based decision making.

        Args:
            job_hash: Optional job hash to filter by similar jobs.
            workspace_hash: Optional workspace hash to filter by.
            sheet_num: Optional sheet number to filter by.
            limit: Maximum number of records to return.

        Returns:
            List of ExecutionRecord objects matching the criteria.
        """
        with self._get_connection() as conn:
            # Build query dynamically based on provided filters
            conditions: list[str] = []
            params: list[str | int] = []

            if job_hash is not None:
                conditions.append("job_hash = ?")
                params.append(job_hash)
            if workspace_hash is not None:
                conditions.append("workspace_hash = ?")
                params.append(workspace_hash)
            if sheet_num is not None:
                conditions.append("sheet_num = ?")
                params.append(sheet_num)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            cursor = conn.execute(
                f"""
                SELECT * FROM executions
                WHERE {where_clause}
                ORDER BY completed_at DESC
                LIMIT ?
                """,
                (*params, limit),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    ExecutionRecord(
                        id=row["id"],
                        workspace_hash=row["workspace_hash"],
                        job_hash=row["job_hash"],
                        sheet_num=row["sheet_num"],
                        started_at=datetime.fromisoformat(row["started_at"])
                        if row["started_at"]
                        else None,
                        completed_at=datetime.fromisoformat(row["completed_at"])
                        if row["completed_at"]
                        else None,
                        duration_seconds=row["duration_seconds"] or 0.0,
                        status=row["status"] or "",
                        retry_count=row["retry_count"] or 0,
                        first_attempt_success=bool(row["first_attempt_success"]),
                        validation_pass_rate=row["validation_pass_rate"] or 0.0,
                        confidence_score=row["confidence_score"] or 0.0,
                        model=row["model"],
                        error_codes=json.loads(row["error_codes"] or "[]"),
                    )
                )

            return records

    def get_optimal_execution_window(
        self,
        error_code: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Analyze historical data to find optimal execution windows.

        Learning Activation: Identifies times of day when executions are most
        likely to succeed, enabling time-aware scheduling recommendations.

        Args:
            error_code: Optional error code to analyze (e.g., for rate limits).
            model: Optional model to filter by.

        Returns:
            Dict with optimal window analysis:
            - optimal_hours: List of hours (0-23) with best success rates
            - avoid_hours: List of hours with high failure/rate limit rates
            - confidence: Confidence in the recommendation (0.0-1.0)
            - sample_count: Number of samples analyzed
        """
        with self._get_connection() as conn:
            # Query success rate by hour of day from error_recoveries
            if error_code:
                cursor = conn.execute(
                    """
                    SELECT
                        time_of_day,
                        COUNT(*) as total,
                        SUM(CASE WHEN recovery_success THEN 1 ELSE 0 END) as successes
                    FROM error_recoveries
                    WHERE error_code = ?
                    GROUP BY time_of_day
                    ORDER BY time_of_day
                    """,
                    (error_code,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT
                        time_of_day,
                        COUNT(*) as total,
                        SUM(CASE WHEN recovery_success THEN 1 ELSE 0 END) as successes
                    FROM error_recoveries
                    GROUP BY time_of_day
                    ORDER BY time_of_day
                    """
                )

            rows = cursor.fetchall()

            if not rows:
                return {
                    "optimal_hours": [],
                    "avoid_hours": [],
                    "confidence": 0.0,
                    "sample_count": 0,
                }

            # Analyze success rates by hour
            hour_stats: dict[int, tuple[int, int]] = {}  # hour -> (successes, total)
            total_samples = 0

            for row in rows:
                hour = row["time_of_day"]
                total = row["total"]
                successes = row["successes"]
                hour_stats[hour] = (successes, total)
                total_samples += total

            # Find optimal and avoid hours
            optimal_hours: list[int] = []
            avoid_hours: list[int] = []

            for hour, (successes, total) in hour_stats.items():
                if total >= 3:  # Minimum samples for confidence
                    success_rate = successes / total
                    if success_rate >= 0.7:
                        optimal_hours.append(hour)
                    elif success_rate <= 0.3:
                        avoid_hours.append(hour)

            # Calculate confidence based on sample size
            confidence = min(total_samples / 50.0, 1.0)

            return {
                "optimal_hours": sorted(optimal_hours),
                "avoid_hours": sorted(avoid_hours),
                "confidence": confidence,
                "sample_count": total_samples,
            }

    def get_workspace_cluster(
        self,
        workspace_hash: str,
    ) -> str | None:
        """Get the cluster ID for a workspace.

        Learning Activation: Supports workspace similarity by grouping
        workspaces with similar patterns into clusters.

        Args:
            workspace_hash: Hash of the workspace to query.

        Returns:
            Cluster ID if assigned, None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT cluster_id FROM workspace_clusters WHERE workspace_hash = ?",
                (workspace_hash,),
            )
            row = cursor.fetchone()
            return row["cluster_id"] if row else None

    def assign_workspace_cluster(
        self,
        workspace_hash: str,
        cluster_id: str,
    ) -> None:
        """Assign a workspace to a cluster.

        Learning Activation: Groups workspaces with similar execution
        patterns for targeted pattern recommendations.

        Args:
            workspace_hash: Hash of the workspace.
            cluster_id: ID of the cluster to assign to.
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workspace_clusters
                (workspace_hash, cluster_id, assigned_at)
                VALUES (?, ?, ?)
                """,
                (workspace_hash, cluster_id, now),
            )

    def get_similar_workspaces(
        self,
        cluster_id: str,
        limit: int = 10,
    ) -> list[str]:
        """Get workspace hashes in the same cluster.

        Learning Activation: Enables cross-workspace learning by identifying
        workspaces with similar patterns.

        Args:
            cluster_id: Cluster ID to query.
            limit: Maximum number of workspace hashes to return.

        Returns:
            List of workspace hashes in the cluster.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT workspace_hash FROM workspace_clusters
                WHERE cluster_id = ?
                ORDER BY assigned_at DESC
                LIMIT ?
                """,
                (cluster_id, limit),
            )
            return [row["workspace_hash"] for row in cursor.fetchall()]

    def clear_all(self) -> None:
        """Clear all data from the global store.

        WARNING: This is destructive and should only be used for testing.
        """
        with self._get_connection() as conn:
            conn.execute("DELETE FROM pattern_applications")
            conn.execute("DELETE FROM error_recoveries")
            conn.execute("DELETE FROM patterns")
            conn.execute("DELETE FROM executions")
            conn.execute("DELETE FROM workspace_clusters")
            conn.execute("DELETE FROM rate_limit_events")

        _logger.warning("Cleared all data from global learning store")


# Convenience function for getting the singleton store
_global_store: GlobalLearningStore | None = None


def get_global_store(
    db_path: Path | None = None,
) -> GlobalLearningStore:
    """Get or create the global learning store singleton.

    Args:
        db_path: Optional custom path. If None, uses default.

    Returns:
        The GlobalLearningStore instance.
    """
    global _global_store

    if _global_store is None or (
        db_path is not None and _global_store.db_path != db_path
    ):
        _global_store = GlobalLearningStore(db_path)

    return _global_store
