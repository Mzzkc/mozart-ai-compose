"""Base class for GlobalLearningStore with connection and schema management.

This module provides the foundational `GlobalLearningStoreBase` class that handles:
- SQLite database connection management with WAL mode
- Schema creation and migration
- Static hashing utilities for workspace and job identification

Extracted from global_store.py as part of the modularization effort.
Mixins inherit from this base to add domain-specific functionality.
"""

import hashlib
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from mozart.core.logging import get_logger

# Module-level logger for global learning store
_logger = get_logger("learning.global_store")

# Default location following XDG conventions
DEFAULT_GLOBAL_STORE_PATH = Path.home() / ".mozart" / "global-learning.db"


class GlobalLearningStoreBase:
    """SQLite-based global learning store base class.

    Provides persistent storage infrastructure for execution outcomes, detected patterns,
    and error recovery data across all Mozart workspaces. Uses WAL mode for safe
    concurrent access.

    This base class handles:
    - Database connection lifecycle
    - Schema version management
    - Migration and schema creation
    - Hashing utilities

    Subclasses (via mixins) add domain-specific methods for patterns, executions,
    rate limits, drift detection, escalation, and budget management.

    Attributes:
        db_path: Path to the SQLite database file.
        _logger: Module logger instance for consistent logging.
    """

    # Schema version - increment when schema changes
    # v9: Added exploration_budget and entropy_responses tables for v23 evolutions
    SCHEMA_VERSION = 9

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the global learning store.

        Creates the database directory if needed, establishes the connection,
        and runs any necessary migrations.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to ~/.mozart/global-learning.db
        """
        self.db_path = db_path or DEFAULT_GLOBAL_STORE_PATH
        self._logger = _logger
        self._ensure_db_exists()
        self._migrate_if_needed()

    def _ensure_db_exists(self) -> None:
        """Ensure the database directory and file exist.

        Creates parent directories with parents=True to handle nested paths.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper configuration.

        Uses WAL mode for better concurrent access and enables foreign keys.
        The connection is configured with:
        - WAL journal mode for concurrent reads/writes
        - Foreign keys enabled for referential integrity
        - 30 second busy timeout for lock contention
        - Row factory for dict-like access

        Yields:
            A configured sqlite3.Connection instance.

        Raises:
            sqlite3.Error: If connection or configuration fails.
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

    def close(self) -> None:
        """Close any persistent resources.

        This method exists for API compatibility. The actual connection
        management is handled per-operation via the context manager
        pattern in _get_connection(). This ensures connections are not
        held open between operations, which is safer for concurrent access.
        """
        # Connections are managed per-operation via context manager
        # This method is a no-op but provides a clean API
        pass

    def _migrate_if_needed(self) -> None:
        """Create or migrate the database schema.

        Checks the current schema version and runs migration if needed.
        Migration is idempotent - running it multiple times is safe.
        """
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
        """Create the database schema.

        Creates all tables and indexes needed by the global learning store.
        Uses IF NOT EXISTS for idempotent operation.

        Args:
            conn: Active database connection.
        """
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
                priority_score REAL DEFAULT 0.5,
                quarantine_status TEXT DEFAULT 'pending',
                provenance_job_hash TEXT,
                provenance_sheet_num INTEGER,
                quarantined_at TIMESTAMP,
                validated_at TIMESTAMP,
                quarantine_reason TEXT,
                trust_score REAL DEFAULT 0.5,
                trust_calculation_date TIMESTAMP,
                success_factors TEXT,
                success_factors_updated_at TIMESTAMP
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
        # v19: Quarantine and trust indexes for efficient filtering
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_quarantine "
            "ON patterns(quarantine_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_patterns_trust "
            "ON patterns(trust_score)"
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

        # Exploration budget table (Evolution v23: Exploration Budget Maintenance)
        # Tracks dynamic exploration budget over time
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exploration_budget (
                id TEXT PRIMARY KEY,
                job_hash TEXT NOT NULL,
                recorded_at TIMESTAMP NOT NULL,
                budget_value REAL NOT NULL,
                entropy_at_time REAL,
                adjustment_type TEXT NOT NULL,
                adjustment_reason TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_budget_job "
            "ON exploration_budget(job_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_budget_recorded "
            "ON exploration_budget(recorded_at)"
        )

        # Entropy response table (Evolution v23: Automatic Entropy Response)
        # Records automatic responses to low entropy events
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entropy_responses (
                id TEXT PRIMARY KEY,
                job_hash TEXT NOT NULL,
                recorded_at TIMESTAMP NOT NULL,
                entropy_at_trigger REAL NOT NULL,
                threshold_used REAL NOT NULL,
                actions_taken TEXT NOT NULL,
                budget_boosted INTEGER DEFAULT 0,
                quarantine_revisits INTEGER DEFAULT 0,
                patterns_revisited TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entropy_response_job "
            "ON entropy_responses(job_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entropy_response_recorded "
            "ON entropy_responses(recorded_at)"
        )

        # Update schema version
        conn.execute("DELETE FROM schema_version")
        conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )

        self._logger.info(f"Created global learning schema v{self.SCHEMA_VERSION}")

    @staticmethod
    def hash_workspace(workspace_path: Path) -> str:
        """Generate a stable hash for a workspace path.

        Creates a reproducible 16-character hex hash from the resolved
        absolute path. This allows pattern matching across sessions
        while preserving privacy (paths are not stored directly).

        Args:
            workspace_path: The absolute path to the workspace.

        Returns:
            A hex string hash of the workspace path (16 characters).
        """
        normalized = str(workspace_path.resolve())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def hash_job(job_name: str, config_hash: str | None = None) -> str:
        """Generate a stable hash for a job.

        Creates a reproducible 16-character hex hash from the job name
        and optional config hash. The config hash enables version-awareness:
        the same job with different configs will have different hashes.

        Args:
            job_name: The job name.
            config_hash: Optional hash of the job config for versioning.

        Returns:
            A hex string hash of the job (16 characters).
        """
        combined = f"{job_name}:{config_hash or ''}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

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


# Export public API
__all__ = [
    "GlobalLearningStoreBase",
    "DEFAULT_GLOBAL_STORE_PATH",
    "_logger",
]
