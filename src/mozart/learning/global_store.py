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


@dataclass
class EscalationDecisionRecord:
    """A record of a human/AI escalation decision.

    Evolution v11: Escalation Learning Loop - records escalation decisions
    to learn from feedback over time and potentially suggest actions for
    similar future escalations.
    """

    id: str
    """Unique identifier for this escalation decision."""

    job_hash: str
    """Hash of the job that triggered escalation."""

    sheet_num: int
    """Sheet number that triggered escalation."""

    confidence: float
    """Aggregate confidence score at time of escalation (0.0-1.0)."""

    action: str
    """Action taken: retry, skip, abort, modify_prompt."""

    guidance: str | None
    """Optional guidance/notes from the escalation handler."""

    validation_pass_rate: float
    """Pass percentage of validations at escalation time."""

    retry_count: int
    """Number of retries attempted before escalation."""

    outcome_after_action: str | None = None
    """What happened after the action: success, failed, aborted, skipped."""

    recorded_at: datetime = field(default_factory=datetime.now)
    """When the escalation decision was recorded."""

    model: str | None = None
    """Model used for execution (if relevant)."""


@dataclass
class PatternDiscoveryEvent:
    """A pattern discovery event for cross-job broadcasting.

    v14 Evolution: Real-time Pattern Broadcasting - enables jobs to share
    newly discovered patterns with other concurrent jobs, so knowledge
    propagates across the ecosystem without waiting for aggregation.
    """

    id: str
    """Unique identifier for this discovery event."""

    pattern_id: str
    """ID of the pattern that was discovered."""

    pattern_name: str
    """Human-readable name of the pattern."""

    pattern_type: str
    """Type of pattern (validation_failure, retry_pattern, etc.)."""

    source_job_hash: str
    """Hash of the job that discovered the pattern."""

    recorded_at: datetime
    """When the discovery was recorded."""

    expires_at: datetime
    """When this broadcast expires (TTL-based)."""

    effectiveness_score: float
    """Effectiveness score at time of discovery."""

    context_tags: list[str] = field(default_factory=list)
    """Context tags for pattern matching."""


@dataclass
class DriftMetrics:
    """Metrics for pattern effectiveness drift detection.

    v12 Evolution: Goal Drift Detection - tracks how pattern effectiveness
    changes over time to detect drifting patterns that may need attention.
    """

    pattern_id: str
    """Pattern ID being analyzed."""

    pattern_name: str
    """Human-readable pattern name."""

    window_size: int
    """Number of applications in each comparison window."""

    effectiveness_before: float
    """Effectiveness score in the older window (applications N-2W to N-W)."""

    effectiveness_after: float
    """Effectiveness score in the recent window (applications N-W to N)."""

    grounding_confidence_avg: float
    """Average grounding confidence across all applications in analysis."""

    drift_magnitude: float
    """Absolute magnitude of drift: |effectiveness_after - effectiveness_before|."""

    drift_direction: str
    """Direction of drift: 'positive', 'negative', or 'stable'."""

    applications_analyzed: int
    """Total number of applications analyzed (should be 2 × window_size)."""

    threshold_exceeded: bool = False
    """Whether drift_magnitude exceeds the alert threshold."""


@dataclass
class EvolutionTrajectoryEntry:
    """A record of a single evolution cycle in Mozart's self-improvement trajectory.

    v16 Evolution: Evolution Trajectory Tracking - enables Mozart to track its
    own evolution history, identifying recurring issue classes and measuring
    improvement over time.
    """

    id: str
    """Unique identifier for this trajectory entry."""

    cycle: int
    """Evolution cycle number (e.g., 16 for v16)."""

    recorded_at: datetime
    """When this entry was recorded."""

    evolutions_completed: int
    """Number of evolutions completed in this cycle."""

    evolutions_deferred: int
    """Number of evolutions deferred in this cycle."""

    issue_classes: list[str]
    """Issue classes addressed (e.g., 'infrastructure_activation', 'epistemic_drift')."""

    cv_avg: float
    """Average Consciousness Volume of selected evolutions."""

    implementation_loc: int
    """Total implementation LOC for this cycle."""

    test_loc: int
    """Total test LOC for this cycle."""

    loc_accuracy: float
    """LOC estimation accuracy (actual/estimated as ratio)."""

    research_candidates_resolved: int = 0
    """Number of research candidates resolved in this cycle."""

    research_candidates_created: int = 0
    """Number of new research candidates created in this cycle."""

    notes: str = ""
    """Optional notes about this evolution cycle."""


class GlobalLearningStore:
    """SQLite-based global learning store.

    Provides persistent storage for execution outcomes, detected patterns,
    and error recovery data across all Mozart workspaces. Uses WAL mode
    for safe concurrent access.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    SCHEMA_VERSION = 6  # v6: Added evolution_trajectory table

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
        # Note: execution_id is a string identifier (e.g. "sheet_1"), not a FK to executions
        # The runner passes simple sheet identifiers for pattern tracking purposes
        # v12: Added grounding_confidence column for grounding-weighted effectiveness
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_applications (
                id TEXT PRIMARY KEY,
                pattern_id TEXT REFERENCES patterns(id),
                execution_id TEXT,
                applied_at TIMESTAMP,
                outcome_improved BOOLEAN,
                retry_count_before INTEGER,
                retry_count_after INTEGER,
                grounding_confidence REAL
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

        # Escalation decisions table (Evolution v11: Escalation Learning Loop)
        # Records human/AI escalation decisions for learning
        conn.execute("""
            CREATE TABLE IF NOT EXISTS escalation_decisions (
                id TEXT PRIMARY KEY,
                job_hash TEXT NOT NULL,
                sheet_num INTEGER NOT NULL,
                confidence REAL NOT NULL,
                action TEXT NOT NULL,
                guidance TEXT,
                validation_pass_rate REAL NOT NULL,
                retry_count INTEGER NOT NULL,
                outcome_after_action TEXT,
                recorded_at TIMESTAMP NOT NULL,
                model TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_escalation_job "
            "ON escalation_decisions(job_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_escalation_action "
            "ON escalation_decisions(action)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_escalation_confidence "
            "ON escalation_decisions(confidence)"
        )

        # Pattern discovery events table (Evolution v14: Real-time Pattern Broadcasting)
        # Records pattern discoveries for cross-job sharing with TTL expiry
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_discovery_events (
                id TEXT PRIMARY KEY,
                pattern_id TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                source_job_hash TEXT NOT NULL,
                recorded_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                effectiveness_score REAL NOT NULL,
                context_tags TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_discovery_expires "
            "ON pattern_discovery_events(expires_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_discovery_source "
            "ON pattern_discovery_events(source_job_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pattern_discovery_type "
            "ON pattern_discovery_events(pattern_type)"
        )

        # Evolution trajectory table (Evolution v16: Evolution Trajectory Tracking)
        # Tracks Mozart's own evolution history for recursive self-improvement analysis
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evolution_trajectory (
                id TEXT PRIMARY KEY,
                cycle INTEGER NOT NULL UNIQUE,
                recorded_at TIMESTAMP NOT NULL,
                evolutions_completed INTEGER NOT NULL,
                evolutions_deferred INTEGER NOT NULL,
                issue_classes TEXT NOT NULL,
                cv_avg REAL NOT NULL,
                implementation_loc INTEGER NOT NULL,
                test_loc INTEGER NOT NULL,
                loc_accuracy REAL NOT NULL,
                research_candidates_resolved INTEGER DEFAULT 0,
                research_candidates_created INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trajectory_cycle "
            "ON evolution_trajectory(cycle)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trajectory_recorded "
            "ON evolution_trajectory(recorded_at)"
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
        application_mode: str = "exploitation",
        validation_passed: bool | None = None,
        grounding_confidence: float | None = None,
    ) -> str:
        """Record that a pattern was applied to an execution.

        This creates the feedback loop for effectiveness tracking. After recording
        the application, automatically updates effectiveness_score and priority_score
        for the pattern.

        v12 Evolution: Grounding→Pattern Feedback - grounding_confidence is now
        stored and used to weight effectiveness calculations. Patterns applied
        during high-grounding executions carry more weight.

        Args:
            pattern_id: The pattern that was applied.
            execution_id: The execution it was applied to.
            outcome_improved: Whether the outcome was better than baseline.
            retry_count_before: Retry count before pattern applied.
            retry_count_after: Retry count after pattern applied.
            application_mode: 'exploration' or 'exploitation' - how the pattern was selected.
            validation_passed: Whether validation passed on first attempt.
            grounding_confidence: Grounding confidence (0.0-1.0) from external validation.
                                 None if grounding hooks were not executed.

        Returns:
            The application record ID.
        """
        app_id = str(uuid.uuid4())
        now = datetime.now()
        now_iso = now.isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pattern_applications (
                    id, pattern_id, execution_id, applied_at,
                    outcome_improved, retry_count_before, retry_count_after,
                    grounding_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    app_id,
                    pattern_id,
                    execution_id,
                    now_iso,
                    outcome_improved,
                    retry_count_before,
                    retry_count_after,
                    grounding_confidence,
                ),
            )

            # Update pattern counts
            if outcome_improved:
                conn.execute(
                    """
                    UPDATE patterns SET
                        led_to_success_count = led_to_success_count + 1,
                        last_confirmed = ?
                    WHERE id = ?
                    """,
                    (now_iso, pattern_id),
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

            # Fetch updated pattern data for effectiveness/priority recalculation
            cursor = conn.execute(
                """
                SELECT led_to_success_count, led_to_failure_count, last_confirmed,
                       occurrence_count, variance
                FROM patterns WHERE id = ?
                """,
                (pattern_id,),
            )
            row = cursor.fetchone()

            if row:
                # Calculate new effectiveness score
                new_effectiveness = self._calculate_effectiveness_v2(
                    pattern_id=pattern_id,
                    led_to_success_count=row["led_to_success_count"],
                    led_to_failure_count=row["led_to_failure_count"],
                    last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
                    now=now,
                    conn=conn,
                )

                # Calculate new priority score
                new_priority = self._calculate_priority_score(
                    effectiveness=new_effectiveness,
                    occurrence_count=row["occurrence_count"],
                    variance=row["variance"],
                )

                # Update both scores in the database
                conn.execute(
                    """
                    UPDATE patterns SET
                        effectiveness_score = ?,
                        priority_score = ?
                    WHERE id = ?
                    """,
                    (new_effectiveness, new_priority, pattern_id),
                )

                _logger.debug(
                    f"Updated pattern {pattern_id}: effectiveness={new_effectiveness:.3f}, "
                    f"priority={new_priority:.3f}, mode={application_mode}"
                )

        return app_id

    def _calculate_effectiveness_v2(
        self,
        pattern_id: str,
        led_to_success_count: int,
        led_to_failure_count: int,
        last_confirmed: datetime,
        now: datetime,
        conn: sqlite3.Connection,
        min_applications: int = 3,
    ) -> float:
        """Calculate effectiveness score using Bayesian moving average with decay.

        This is the v2 formula that combines historical and recent performance
        with recency weighting. Patterns that haven't proven themselves recently
        will see their effectiveness decay.

        v12 Evolution: Grounding→Pattern Feedback - effectiveness is now weighted
        by grounding confidence. High-grounding executions carry more trust weight.

        Formula:
            base_effectiveness = (α × recent + (1-α) × historical) × recency_decay
            effectiveness = base_effectiveness × grounding_weight

        Where:
        - α = 0.7 (weight recent outcomes more heavily)
        - recent = success rate in last 5 applications
        - historical = all-time success rate with Laplace smoothing
        - recency_decay = 0.9^(days_since_last_success / 30)
        - grounding_weight = 0.7 + 0.3 × avg_grounding (when grounding data exists)

        Args:
            pattern_id: The pattern being calculated.
            led_to_success_count: Total successes.
            led_to_failure_count: Total failures.
            last_confirmed: When pattern last led to success.
            now: Current timestamp.
            conn: Database connection for recent applications query.
            min_applications: Minimum applications before trusting data.

        Returns:
            Effectiveness score between 0.0 and 1.0.
        """
        total = led_to_success_count + led_to_failure_count

        # Cold start: return optimistic prior to encourage exploration
        if total < min_applications:
            return 0.55

        # Historical effectiveness with Laplace smoothing (prevents 0/1 extremes)
        historical = (led_to_success_count + 0.5) / (total + 1)

        # Query recent applications for recency weighting and grounding confidence
        cursor = conn.execute(
            """
            SELECT outcome_improved, grounding_confidence
            FROM pattern_applications
            WHERE pattern_id = ?
            ORDER BY applied_at DESC
            LIMIT 5
            """,
            (pattern_id,),
        )
        recent_apps = cursor.fetchall()

        if recent_apps:
            recent_successes = sum(1 for app in recent_apps if app["outcome_improved"])
            recent = recent_successes / len(recent_apps)

            # v12: Calculate average grounding confidence from recent applications
            grounding_values = [
                app["grounding_confidence"]
                for app in recent_apps
                if app["grounding_confidence"] is not None
            ]
            if grounding_values:
                avg_grounding = sum(grounding_values) / len(grounding_values)
            else:
                # No grounding data - use neutral multiplier (1.0)
                avg_grounding = 1.0
        else:
            recent = historical
            avg_grounding = 1.0

        # Combine with recency weighting: α = 0.7 prioritizes recent outcomes
        alpha = 0.7
        combined = alpha * recent + (1 - alpha) * historical

        # Apply recency penalty: patterns decay if not recently confirmed
        days_since = (now - last_confirmed).days
        decay = 0.9 ** (days_since / 30.0)

        base_effectiveness = combined * decay

        # v12: Apply grounding weight
        # Formula: grounding_weight = 0.7 + 0.3 × avg_grounding
        # This means:
        # - avg_grounding = 0.0 → weight = 0.70 (30% penalty for ungrounded)
        # - avg_grounding = 0.5 → weight = 0.85 (15% penalty for low grounding)
        # - avg_grounding = 1.0 → weight = 1.00 (full trust for high grounding)
        # When avg_grounding = 1.0 (no grounding data), weight = 1.0 (backward compatible)
        grounding_weight = 0.7 + 0.3 * avg_grounding

        return base_effectiveness * grounding_weight

    def _calculate_priority_score(
        self,
        effectiveness: float,
        occurrence_count: int,
        variance: float,
    ) -> float:
        """Calculate priority score from effectiveness and other factors.

        Priority determines which patterns get selected first during
        exploitation mode. Higher priority = more likely to be applied.

        Formula:
            priority = effectiveness × frequency_factor × (1 - variance)

        Where:
        - frequency_factor = min(1.0, log10(occurrence_count + 1) / 2)
          This gives a boost to frequently-seen patterns (up to 100 occurrences)
        - variance penalizes inconsistent patterns

        Args:
            effectiveness: Calculated effectiveness score (0.0-1.0).
            occurrence_count: How many times pattern has been seen.
            variance: Measure of outcome inconsistency (0.0-1.0).

        Returns:
            Priority score between 0.0 and 1.0.
        """
        import math

        # Frequency factor: log scale to avoid over-weighting very common patterns
        # log10(1) = 0, log10(10) = 1, log10(100) = 2
        # Divided by 2 so patterns need ~100 occurrences to reach max frequency boost
        frequency_factor = min(1.0, math.log10(occurrence_count + 1) / 2.0)

        # Variance penalty: inconsistent patterns should have lower priority
        variance_penalty = 1 - variance

        # Combine factors
        priority = effectiveness * frequency_factor * variance_penalty

        # Ensure bounds
        return max(0.0, min(1.0, priority))

    def update_pattern_effectiveness(
        self,
        pattern_id: str,
    ) -> float | None:
        """Manually recalculate and update a pattern's effectiveness.

        This can be called independently from record_pattern_application()
        for batch updates or manual recalculation.

        Args:
            pattern_id: The pattern to update.

        Returns:
            New effectiveness score, or None if pattern not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT led_to_success_count, led_to_failure_count, last_confirmed,
                       occurrence_count, variance
                FROM patterns WHERE id = ?
                """,
                (pattern_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            now = datetime.now()
            new_effectiveness = self._calculate_effectiveness_v2(
                pattern_id=pattern_id,
                led_to_success_count=row["led_to_success_count"],
                led_to_failure_count=row["led_to_failure_count"],
                last_confirmed=datetime.fromisoformat(row["last_confirmed"]),
                now=now,
                conn=conn,
            )

            new_priority = self._calculate_priority_score(
                effectiveness=new_effectiveness,
                occurrence_count=row["occurrence_count"],
                variance=row["variance"],
            )

            conn.execute(
                """
                UPDATE patterns SET
                    effectiveness_score = ?,
                    priority_score = ?
                WHERE id = ?
                """,
                (new_effectiveness, new_priority, pattern_id),
            )

            _logger.debug(
                f"Manual update pattern {pattern_id}: "
                f"effectiveness={new_effectiveness:.3f}, priority={new_priority:.3f}"
            )
            return new_effectiveness

    def recalculate_all_pattern_priorities(self) -> int:
        """Recalculate priorities for all patterns.

        Use for batch maintenance, e.g., daily recalculation to apply
        recency decay to patterns that haven't been used recently.

        Returns:
            Number of patterns updated.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM patterns"
            )
            pattern_ids = [row["id"] for row in cursor.fetchall()]

        updated = 0
        for pattern_id in pattern_ids:
            result = self.update_pattern_effectiveness(pattern_id)
            if result is not None:
                updated += 1

        _logger.info(f"Recalculated priorities for {updated} patterns")
        return updated

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
    # Evolution v11: Escalation Learning Loop
    # =========================================================================

    def record_escalation_decision(
        self,
        job_id: str,
        sheet_num: int,
        confidence: float,
        action: str,
        validation_pass_rate: float,
        retry_count: int,
        guidance: str | None = None,
        outcome_after_action: str | None = None,
        model: str | None = None,
    ) -> str:
        """Record an escalation decision for learning.

        When a sheet triggers escalation and receives a response from
        a human or AI handler, this method records the decision so that
        Mozart can learn from it and potentially suggest similar actions
        for future escalations with similar contexts.

        Evolution v11: Escalation Learning Loop - closes the loop between
        escalation handlers and learning system.

        Args:
            job_id: ID of the job that triggered escalation.
            sheet_num: Sheet number that triggered escalation.
            confidence: Aggregate confidence score at escalation time (0.0-1.0).
            action: Action taken (retry, skip, abort, modify_prompt).
            validation_pass_rate: Pass percentage at escalation time.
            retry_count: Number of retries before escalation.
            guidance: Optional guidance/notes from the handler.
            outcome_after_action: What happened after (success, failed, etc.).
            model: Optional model name used for execution.

        Returns:
            The escalation decision record ID.
        """
        record_id = str(uuid.uuid4())
        job_hash = self.hash_job(job_id)
        now = datetime.now()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO escalation_decisions (
                    id, job_hash, sheet_num, confidence, action,
                    guidance, validation_pass_rate, retry_count,
                    outcome_after_action, recorded_at, model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    job_hash,
                    sheet_num,
                    confidence,
                    action,
                    guidance,
                    validation_pass_rate,
                    retry_count,
                    outcome_after_action,
                    now.isoformat(),
                    model,
                ),
            )

        _logger.info(
            f"Recorded escalation decision {record_id}: sheet={sheet_num}, "
            f"action={action}, confidence={confidence:.1%}"
        )
        return record_id

    def get_escalation_history(
        self,
        job_id: str | None = None,
        action: str | None = None,
        limit: int = 20,
    ) -> list[EscalationDecisionRecord]:
        """Get historical escalation decisions.

        Retrieves past escalation decisions for analysis or display.
        Can filter by job or action type.

        Args:
            job_id: Optional job ID to filter by.
            action: Optional action type to filter by.
            limit: Maximum number of records to return.

        Returns:
            List of EscalationDecisionRecord objects.
        """
        with self._get_connection() as conn:
            conditions: list[str] = []
            params: list[str | int] = []

            if job_id is not None:
                job_hash = self.hash_job(job_id)
                conditions.append("job_hash = ?")
                params.append(job_hash)

            if action is not None:
                conditions.append("action = ?")
                params.append(action)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            cursor = conn.execute(
                f"""
                SELECT * FROM escalation_decisions
                WHERE {where_clause}
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (*params, limit),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    EscalationDecisionRecord(
                        id=row["id"],
                        job_hash=row["job_hash"],
                        sheet_num=row["sheet_num"],
                        confidence=row["confidence"],
                        action=row["action"],
                        guidance=row["guidance"],
                        validation_pass_rate=row["validation_pass_rate"],
                        retry_count=row["retry_count"],
                        outcome_after_action=row["outcome_after_action"],
                        recorded_at=datetime.fromisoformat(row["recorded_at"]),
                        model=row["model"],
                    )
                )

            return records

    def get_similar_escalation(
        self,
        confidence: float,
        validation_pass_rate: float,
        confidence_tolerance: float = 0.15,
        pass_rate_tolerance: float = 15.0,
        limit: int = 5,
    ) -> list[EscalationDecisionRecord]:
        """Get similar past escalation decisions for guidance.

        Finds historical escalations with similar context (confidence and
        pass rate) to help inform the current escalation decision. Can be
        used to suggest actions or provide guidance to human operators.

        Evolution v11: Escalation Learning Loop - enables pattern-based
        suggestions for similar escalation contexts.

        Args:
            confidence: Current confidence level (0.0-1.0).
            validation_pass_rate: Current validation pass percentage.
            confidence_tolerance: How much confidence can differ (default 0.15).
            pass_rate_tolerance: How much pass rate can differ (default 15%).
            limit: Maximum number of similar records to return.

        Returns:
            List of EscalationDecisionRecord from similar past escalations,
            ordered by outcome success (successful outcomes first).
        """
        with self._get_connection() as conn:
            # Find escalations with similar confidence and pass rate
            # Order by: successful outcomes first, then by how close the match is
            cursor = conn.execute(
                """
                SELECT *,
                       ABS(confidence - ?) as conf_diff,
                       ABS(validation_pass_rate - ?) as rate_diff
                FROM escalation_decisions
                WHERE ABS(confidence - ?) <= ?
                  AND ABS(validation_pass_rate - ?) <= ?
                ORDER BY
                    CASE WHEN outcome_after_action = 'success' THEN 0
                         WHEN outcome_after_action = 'skipped' THEN 1
                         WHEN outcome_after_action IS NULL THEN 2
                         ELSE 3 END,
                    conf_diff + (rate_diff / 100.0)
                LIMIT ?
                """,
                (
                    confidence,
                    validation_pass_rate,
                    confidence,
                    confidence_tolerance,
                    validation_pass_rate,
                    pass_rate_tolerance,
                    limit,
                ),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    EscalationDecisionRecord(
                        id=row["id"],
                        job_hash=row["job_hash"],
                        sheet_num=row["sheet_num"],
                        confidence=row["confidence"],
                        action=row["action"],
                        guidance=row["guidance"],
                        validation_pass_rate=row["validation_pass_rate"],
                        retry_count=row["retry_count"],
                        outcome_after_action=row["outcome_after_action"],
                        recorded_at=datetime.fromisoformat(row["recorded_at"]),
                        model=row["model"],
                    )
                )

            return records

    def update_escalation_outcome(
        self,
        escalation_id: str,
        outcome_after_action: str,
    ) -> bool:
        """Update the outcome of an escalation decision.

        Called after an escalation action is taken and the result is known.
        This closes the feedback loop by recording whether the action led
        to success or failure.

        Args:
            escalation_id: The escalation record ID to update.
            outcome_after_action: What happened (success, failed, aborted, skipped).

        Returns:
            True if the record was updated, False if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE escalation_decisions
                SET outcome_after_action = ?
                WHERE id = ?
                """,
                (outcome_after_action, escalation_id),
            )
            updated = cursor.rowcount > 0

        if updated:
            _logger.debug(
                f"Updated escalation {escalation_id} outcome: {outcome_after_action}"
            )

        return updated

    # =========================================================================
    # v12 Evolution: Goal Drift Detection
    # =========================================================================

    def calculate_effectiveness_drift(
        self,
        pattern_id: str,
        window_size: int = 5,
        drift_threshold: float = 0.2,
    ) -> DriftMetrics | None:
        """Calculate effectiveness drift for a pattern.

        Compares the effectiveness of a pattern in its recent applications
        vs older applications to detect drift. Patterns that were once
        effective but are now declining may need investigation.

        v12 Evolution: Goal Drift Detection - enables proactive pattern
        health monitoring.

        Formula:
            drift = effectiveness_after - effectiveness_before
            drift_magnitude = |drift|
            weighted_drift = drift_magnitude / avg_grounding_confidence

        A positive drift means the pattern is improving, negative means declining.
        The weighted drift amplifies the signal when grounding confidence is low.

        Args:
            pattern_id: Pattern to analyze.
            window_size: Number of applications per window (default 5).
                        Total applications needed = 2 × window_size.
            drift_threshold: Threshold for flagging drift (default 0.2 = 20%).

        Returns:
            DriftMetrics if enough data exists, None otherwise.
        """
        with self._get_connection() as conn:
            # Get pattern name
            cursor = conn.execute(
                "SELECT pattern_name FROM patterns WHERE id = ?",
                (pattern_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            pattern_name = row["pattern_name"]

            # Fetch 2 × window_size recent applications
            # Ordered by applied_at DESC to get most recent first
            cursor = conn.execute(
                """
                SELECT outcome_improved, grounding_confidence, applied_at
                FROM pattern_applications
                WHERE pattern_id = ?
                ORDER BY applied_at DESC
                LIMIT ?
                """,
                (pattern_id, window_size * 2),
            )
            applications = cursor.fetchall()

            # Need at least 2 × window_size applications for comparison
            if len(applications) < window_size * 2:
                _logger.debug(
                    f"Pattern {pattern_id} has {len(applications)} applications, "
                    f"need {window_size * 2} for drift analysis"
                )
                return None

            # Split into recent (first window_size) and older (second window_size)
            recent_apps = applications[:window_size]
            older_apps = applications[window_size : window_size * 2]

            # Calculate effectiveness for each window
            # effectiveness = success_rate with Laplace smoothing
            def calc_effectiveness(apps: list) -> tuple[float, list[float]]:
                successes = sum(1 for a in apps if a["outcome_improved"])
                eff = (successes + 0.5) / (len(apps) + 1)  # Laplace smoothing
                grounding_vals = [
                    a["grounding_confidence"]
                    for a in apps
                    if a["grounding_confidence"] is not None
                ]
                return eff, grounding_vals

            eff_after, grounding_recent = calc_effectiveness(recent_apps)
            eff_before, grounding_older = calc_effectiveness(older_apps)

            # Calculate average grounding confidence across all applications
            all_grounding = grounding_recent + grounding_older
            if all_grounding:
                avg_grounding = sum(all_grounding) / len(all_grounding)
            else:
                avg_grounding = 1.0  # No grounding data - neutral

            # Calculate drift
            drift = eff_after - eff_before
            drift_magnitude = abs(drift)

            # Determine direction
            if drift > 0.05:  # Small threshold to avoid noise
                drift_direction = "positive"
            elif drift < -0.05:
                drift_direction = "negative"
            else:
                drift_direction = "stable"

            # Check if threshold exceeded (weighted by grounding)
            # Lower grounding confidence amplifies the drift signal
            weighted_magnitude = drift_magnitude / max(avg_grounding, 0.5)
            threshold_exceeded = weighted_magnitude > drift_threshold

            return DriftMetrics(
                pattern_id=pattern_id,
                pattern_name=pattern_name,
                window_size=window_size,
                effectiveness_before=eff_before,
                effectiveness_after=eff_after,
                grounding_confidence_avg=avg_grounding,
                drift_magnitude=drift_magnitude,
                drift_direction=drift_direction,
                applications_analyzed=len(applications),
                threshold_exceeded=threshold_exceeded,
            )

    def get_drifting_patterns(
        self,
        drift_threshold: float = 0.2,
        window_size: int = 5,
        limit: int = 20,
    ) -> list[DriftMetrics]:
        """Get all patterns with significant drift.

        Scans all patterns with enough application history and returns
        those that exceed the drift threshold.

        v12 Evolution: Goal Drift Detection - enables CLI display of
        drifting patterns for operator review.

        Args:
            drift_threshold: Minimum drift to include (default 0.2).
            window_size: Applications per window (default 5).
            limit: Maximum patterns to return.

        Returns:
            List of DriftMetrics for drifting patterns, sorted by
            drift_magnitude descending.
        """
        drifting: list[DriftMetrics] = []

        with self._get_connection() as conn:
            # Find patterns with at least 2 × window_size applications
            cursor = conn.execute(
                """
                SELECT pattern_id, COUNT(*) as app_count
                FROM pattern_applications
                GROUP BY pattern_id
                HAVING app_count >= ?
                """,
                (window_size * 2,),
            )
            pattern_ids = [row["pattern_id"] for row in cursor.fetchall()]

        # Calculate drift for each pattern
        for pattern_id in pattern_ids:
            metrics = self.calculate_effectiveness_drift(
                pattern_id=pattern_id,
                window_size=window_size,
                drift_threshold=drift_threshold,
            )
            if metrics and metrics.threshold_exceeded:
                drifting.append(metrics)

        # Sort by drift magnitude descending
        drifting.sort(key=lambda m: m.drift_magnitude, reverse=True)

        return drifting[:limit]

    def get_pattern_drift_summary(self) -> dict[str, Any]:
        """Get a summary of pattern drift across all patterns.

        Provides aggregate statistics for monitoring pattern health.

        v12 Evolution: Goal Drift Detection - supports dashboard/reporting.

        Returns:
            Dict with drift statistics:
            - total_patterns: Total patterns in the store
            - patterns_analyzed: Patterns with enough history for analysis
            - patterns_drifting: Patterns exceeding drift threshold
            - avg_drift_magnitude: Average drift across analyzed patterns
            - most_drifted: ID of pattern with highest drift
        """
        with self._get_connection() as conn:
            # Total patterns
            cursor = conn.execute("SELECT COUNT(*) as count FROM patterns")
            total_patterns = cursor.fetchone()["count"]

            # Patterns with enough applications (10+ for analysis)
            cursor = conn.execute(
                """
                SELECT pattern_id, COUNT(*) as app_count
                FROM pattern_applications
                GROUP BY pattern_id
                HAVING app_count >= 10
                """
            )
            analyzable_patterns = [row["pattern_id"] for row in cursor.fetchall()]

        patterns_analyzed = len(analyzable_patterns)
        if patterns_analyzed == 0:
            return {
                "total_patterns": total_patterns,
                "patterns_analyzed": 0,
                "patterns_drifting": 0,
                "avg_drift_magnitude": 0.0,
                "most_drifted": None,
            }

        # Calculate drift for each
        all_metrics: list[DriftMetrics] = []
        for pattern_id in analyzable_patterns:
            metrics = self.calculate_effectiveness_drift(pattern_id)
            if metrics:
                all_metrics.append(metrics)

        if not all_metrics:
            return {
                "total_patterns": total_patterns,
                "patterns_analyzed": patterns_analyzed,
                "patterns_drifting": 0,
                "avg_drift_magnitude": 0.0,
                "most_drifted": None,
            }

        drifting_count = sum(1 for m in all_metrics if m.threshold_exceeded)
        avg_drift = sum(m.drift_magnitude for m in all_metrics) / len(all_metrics)
        most_drifted = max(all_metrics, key=lambda m: m.drift_magnitude)

        return {
            "total_patterns": total_patterns,
            "patterns_analyzed": len(all_metrics),
            "patterns_drifting": drifting_count,
            "avg_drift_magnitude": avg_drift,
            "most_drifted": most_drifted.pattern_id if most_drifted else None,
        }

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

    # =========================================================================
    # Evolution v14: Real-time Pattern Broadcasting
    # =========================================================================

    def record_pattern_discovery(
        self,
        pattern_id: str,
        pattern_name: str,
        pattern_type: str,
        job_id: str,
        effectiveness_score: float = 1.0,
        context_tags: list[str] | None = None,
        ttl_seconds: float = 300.0,
    ) -> str:
        """Record a pattern discovery for cross-job broadcasting.

        When a job discovers a new pattern, it broadcasts the discovery so
        other concurrent jobs can benefit immediately rather than waiting
        for aggregation.

        v14 Evolution: Real-time Pattern Broadcasting - enables immediate
        pattern sharing across concurrent jobs.

        Args:
            pattern_id: ID of the discovered pattern.
            pattern_name: Human-readable name of the pattern.
            pattern_type: Type of pattern (validation_failure, etc.).
            job_id: ID of the job that discovered the pattern.
            effectiveness_score: Initial effectiveness score (0.0-1.0).
            context_tags: Optional context tags for pattern matching.
            ttl_seconds: Time-to-live in seconds (default 5 minutes).

        Returns:
            The discovery event record ID.
        """
        from datetime import timedelta

        record_id = str(uuid.uuid4())
        now = datetime.now()
        job_hash = self.hash_job(job_id)
        expires_at = now + timedelta(seconds=ttl_seconds)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO pattern_discovery_events (
                    id, pattern_id, pattern_name, pattern_type,
                    source_job_hash, recorded_at, expires_at,
                    effectiveness_score, context_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    pattern_id,
                    pattern_name,
                    pattern_type,
                    job_hash,
                    now.isoformat(),
                    expires_at.isoformat(),
                    effectiveness_score,
                    json.dumps(context_tags or []),
                ),
            )

        _logger.info(
            f"Broadcast pattern discovery '{pattern_name}' (type: {pattern_type}), "
            f"expires in {ttl_seconds:.0f}s"
        )
        return record_id

    def check_recent_pattern_discoveries(
        self,
        exclude_job_id: str | None = None,
        pattern_type: str | None = None,
        min_effectiveness: float = 0.0,
        limit: int = 20,
    ) -> list[PatternDiscoveryEvent]:
        """Check for recent pattern discoveries from other jobs.

        Jobs can poll this to discover patterns found by concurrent jobs,
        enabling immediate adoption of successful patterns.

        v14 Evolution: Real-time Pattern Broadcasting - enables jobs to
        receive pattern broadcasts from other concurrent jobs.

        Args:
            exclude_job_id: Optional job ID to exclude (typically self).
            pattern_type: Optional filter by pattern type.
            min_effectiveness: Minimum effectiveness score to include.
            limit: Maximum number of discoveries to return.

        Returns:
            List of PatternDiscoveryEvent objects from other jobs.
        """
        now = datetime.now()
        exclude_hash = self.hash_job(exclude_job_id) if exclude_job_id else None

        with self._get_connection() as conn:
            # Build query based on filters
            query = """
                SELECT * FROM pattern_discovery_events
                WHERE expires_at > ?
                AND effectiveness_score >= ?
            """
            params: list[str | float] = [now.isoformat(), min_effectiveness]

            if exclude_hash is not None:
                query += " AND source_job_hash != ?"
                params.append(exclude_hash)

            if pattern_type is not None:
                query += " AND pattern_type = ?"
                params.append(pattern_type)

            query += " ORDER BY recorded_at DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            events = []

            for row in cursor.fetchall():
                events.append(
                    PatternDiscoveryEvent(
                        id=row["id"],
                        pattern_id=row["pattern_id"],
                        pattern_name=row["pattern_name"],
                        pattern_type=row["pattern_type"],
                        source_job_hash=row["source_job_hash"],
                        recorded_at=datetime.fromisoformat(row["recorded_at"]),
                        expires_at=datetime.fromisoformat(row["expires_at"]),
                        effectiveness_score=row["effectiveness_score"],
                        context_tags=json.loads(row["context_tags"] or "[]"),
                    )
                )

            return events

    def cleanup_expired_pattern_discoveries(self) -> int:
        """Remove expired pattern discovery events.

        Should be called periodically to prevent the pattern_discovery_events
        table from growing unbounded.

        v14 Evolution: Real-time Pattern Broadcasting - maintains table health.

        Returns:
            Number of expired events removed.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM pattern_discovery_events WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            deleted = cursor.rowcount

        if deleted > 0:
            _logger.debug(f"Cleaned up {deleted} expired pattern discovery events")

        return deleted

    def get_active_pattern_discoveries(
        self,
        pattern_type: str | None = None,
    ) -> list[PatternDiscoveryEvent]:
        """Get all active (unexpired) pattern discovery events.

        Args:
            pattern_type: Optional filter by pattern type.

        Returns:
            List of PatternDiscoveryEvent objects that haven't expired yet.
        """
        now = datetime.now()

        with self._get_connection() as conn:
            if pattern_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM pattern_discovery_events
                    WHERE expires_at > ? AND pattern_type = ?
                    ORDER BY recorded_at DESC
                    """,
                    (now.isoformat(), pattern_type),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM pattern_discovery_events
                    WHERE expires_at > ?
                    ORDER BY recorded_at DESC
                    """,
                    (now.isoformat(),),
                )

            events = []
            for row in cursor.fetchall():
                events.append(
                    PatternDiscoveryEvent(
                        id=row["id"],
                        pattern_id=row["pattern_id"],
                        pattern_name=row["pattern_name"],
                        pattern_type=row["pattern_type"],
                        source_job_hash=row["source_job_hash"],
                        recorded_at=datetime.fromisoformat(row["recorded_at"]),
                        expires_at=datetime.fromisoformat(row["expires_at"]),
                        effectiveness_score=row["effectiveness_score"],
                        context_tags=json.loads(row["context_tags"] or "[]"),
                    )
                )

            return events

    # =========================================================================
    # Evolution v14: Pattern Auto-Retirement
    # =========================================================================

    def retire_drifting_patterns(
        self,
        drift_threshold: float = 0.2,
        window_size: int = 5,
        require_negative_drift: bool = True,
    ) -> list[tuple[str, str, float]]:
        """Retire patterns that are drifting negatively.

        Connects the drift detection infrastructure (DriftMetrics) to action.
        Patterns that have drifted significantly AND in a negative direction
        are retired by setting their priority_score to 0.

        v14 Evolution: Pattern Auto-Retirement - enables automated pattern
        lifecycle management based on empirical effectiveness drift.

        Args:
            drift_threshold: Minimum drift magnitude to consider (default 0.2).
            window_size: Applications per window for drift calculation.
            require_negative_drift: If True, only retire patterns with
                negative drift (getting worse). If False, also retire
                patterns with positive anomalous drift.

        Returns:
            List of (pattern_id, pattern_name, drift_magnitude) tuples for
            patterns that were retired.
        """
        retired: list[tuple[str, str, float]] = []

        # Get all patterns exceeding drift threshold
        drifting = self.get_drifting_patterns(
            drift_threshold=drift_threshold,
            window_size=window_size,
            limit=100,  # Process up to 100 drifting patterns
        )

        if not drifting:
            _logger.debug("No drifting patterns found - nothing to retire")
            return retired

        with self._get_connection() as conn:
            for metrics in drifting:
                # Only retire if negative drift (getting worse)
                if require_negative_drift and metrics.drift_direction != "negative":
                    _logger.debug(
                        f"Skipping {metrics.pattern_name}: drift is {metrics.drift_direction}, "
                        f"not negative"
                    )
                    continue

                # threshold_exceeded should already be True from get_drifting_patterns()
                # but double-check for safety
                if not metrics.threshold_exceeded:
                    continue

                # Retire by setting priority_score to 0
                # Also update suggested_action to document the retirement
                retirement_reason = (
                    f"Auto-retired: drift {metrics.drift_direction} "
                    f"({metrics.drift_magnitude:.2f}), "
                    f"effectiveness {metrics.effectiveness_before:.2f} → "
                    f"{metrics.effectiveness_after:.2f}"
                )

                conn.execute(
                    """
                    UPDATE patterns
                    SET priority_score = 0,
                        suggested_action = ?
                    WHERE id = ?
                    """,
                    (retirement_reason, metrics.pattern_id),
                )

                retired.append((
                    metrics.pattern_id,
                    metrics.pattern_name,
                    metrics.drift_magnitude,
                ))

                _logger.info(
                    f"Retired pattern '{metrics.pattern_name}': {retirement_reason}"
                )

        if retired:
            _logger.info(
                f"Pattern auto-retirement complete: {len(retired)} patterns retired"
            )

        return retired

    def get_retired_patterns(self, limit: int = 50) -> list[PatternRecord]:
        """Get patterns that have been retired (priority_score = 0).

        Returns patterns that were retired through auto-retirement or
        manual deprecation, useful for review and potential recovery.

        Args:
            limit: Maximum number of patterns to return.

        Returns:
            List of PatternRecord objects with priority_score = 0.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM patterns
                WHERE priority_score = 0
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (limit,),
            )

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

    # =========================================================================
    # Evolution v16: Evolution Trajectory Tracking
    # =========================================================================

    def record_evolution_entry(
        self,
        cycle: int,
        evolutions_completed: int,
        evolutions_deferred: int,
        issue_classes: list[str],
        cv_avg: float,
        implementation_loc: int,
        test_loc: int,
        loc_accuracy: float,
        research_candidates_resolved: int = 0,
        research_candidates_created: int = 0,
        notes: str = "",
    ) -> str:
        """Record an evolution cycle entry to the trajectory.

        v16 Evolution: Evolution Trajectory Tracking - enables Mozart to track
        its own evolution history for recursive self-improvement analysis.

        Args:
            cycle: Evolution cycle number (e.g., 16 for v16).
            evolutions_completed: Number of evolutions completed in this cycle.
            evolutions_deferred: Number of evolutions deferred in this cycle.
            issue_classes: Issue classes addressed (e.g., ['infrastructure_activation']).
            cv_avg: Average Consciousness Volume of selected evolutions.
            implementation_loc: Total implementation LOC for this cycle.
            test_loc: Total test LOC for this cycle.
            loc_accuracy: LOC estimation accuracy (actual/estimated as ratio).
            research_candidates_resolved: Number of research candidates resolved.
            research_candidates_created: Number of new research candidates created.
            notes: Optional notes about this evolution cycle.

        Returns:
            The ID of the created trajectory entry.

        Raises:
            sqlite3.IntegrityError: If an entry for this cycle already exists.
        """
        entry_id = str(uuid.uuid4())
        now = datetime.now()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evolution_trajectory (
                    id, cycle, recorded_at, evolutions_completed, evolutions_deferred,
                    issue_classes, cv_avg, implementation_loc, test_loc, loc_accuracy,
                    research_candidates_resolved, research_candidates_created, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    cycle,
                    now.isoformat(),
                    evolutions_completed,
                    evolutions_deferred,
                    json.dumps(issue_classes),
                    cv_avg,
                    implementation_loc,
                    test_loc,
                    loc_accuracy,
                    research_candidates_resolved,
                    research_candidates_created,
                    notes,
                ),
            )

        _logger.info(
            f"Recorded evolution trajectory entry for cycle {cycle}: "
            f"{evolutions_completed} completed, {evolutions_deferred} deferred"
        )

        return entry_id

    def get_trajectory(
        self,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
        limit: int = 50,
    ) -> list[EvolutionTrajectoryEntry]:
        """Retrieve evolution trajectory history.

        v16 Evolution: Evolution Trajectory Tracking - enables analysis of
        Mozart's evolution history over time.

        Args:
            start_cycle: Optional minimum cycle number to include.
            end_cycle: Optional maximum cycle number to include.
            limit: Maximum number of entries to return (default: 50).

        Returns:
            List of EvolutionTrajectoryEntry objects, ordered by cycle descending.
        """
        with self._get_connection() as conn:
            query = "SELECT * FROM evolution_trajectory WHERE 1=1"
            params: list[int] = []

            if start_cycle is not None:
                query += " AND cycle >= ?"
                params.append(start_cycle)

            if end_cycle is not None:
                query += " AND cycle <= ?"
                params.append(end_cycle)

            query += " ORDER BY cycle DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            entries = []

            for row in cursor.fetchall():
                entries.append(
                    EvolutionTrajectoryEntry(
                        id=row["id"],
                        cycle=row["cycle"],
                        recorded_at=datetime.fromisoformat(row["recorded_at"]),
                        evolutions_completed=row["evolutions_completed"],
                        evolutions_deferred=row["evolutions_deferred"],
                        issue_classes=json.loads(row["issue_classes"]),
                        cv_avg=row["cv_avg"],
                        implementation_loc=row["implementation_loc"],
                        test_loc=row["test_loc"],
                        loc_accuracy=row["loc_accuracy"],
                        research_candidates_resolved=row["research_candidates_resolved"] or 0,
                        research_candidates_created=row["research_candidates_created"] or 0,
                        notes=row["notes"] or "",
                    )
                )

            return entries

    def get_recurring_issues(
        self,
        min_occurrences: int = 2,
        window_cycles: int | None = None,
    ) -> dict[str, list[int]]:
        """Identify recurring issue classes across evolution cycles.

        v16 Evolution: Evolution Trajectory Tracking - enables identification
        of patterns in what types of issues Mozart addresses repeatedly.

        Args:
            min_occurrences: Minimum number of occurrences to consider recurring.
            window_cycles: Optional limit to analyze only recent N cycles.

        Returns:
            Dict mapping issue class names to list of cycles where they appeared.
            Only includes issue classes that meet the min_occurrences threshold.
        """
        with self._get_connection() as conn:
            query = "SELECT cycle, issue_classes FROM evolution_trajectory"
            params: list[int] = []

            if window_cycles is not None:
                # Get recent N cycles
                query += " ORDER BY cycle DESC LIMIT ?"
                params.append(window_cycles)
            else:
                query += " ORDER BY cycle DESC"

            cursor = conn.execute(query, params)

            # Count issue class occurrences
            issue_cycles: dict[str, list[int]] = {}

            for row in cursor.fetchall():
                cycle = row["cycle"]
                issues = json.loads(row["issue_classes"])

                for issue in issues:
                    if issue not in issue_cycles:
                        issue_cycles[issue] = []
                    issue_cycles[issue].append(cycle)

            # Filter by min_occurrences
            recurring = {
                issue: sorted(cycles, reverse=True)
                for issue, cycles in issue_cycles.items()
                if len(cycles) >= min_occurrences
            }

            return recurring

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
            conn.execute("DELETE FROM escalation_decisions")
            conn.execute("DELETE FROM pattern_discovery_events")
            conn.execute("DELETE FROM evolution_trajectory")

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
