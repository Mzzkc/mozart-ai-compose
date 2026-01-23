"""Configuration models for Mozart jobs.

Defines Pydantic models for loading and validating YAML job configurations.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class IsolationMode(str, Enum):
    """Isolation method for parallel job execution."""

    NONE = "none"  # Default: no isolation
    WORKTREE = "worktree"  # Git worktree isolation


class IsolationConfig(BaseModel):
    """Configuration for execution isolation.

    Worktree isolation creates a separate git working directory for each job,
    enabling safe parallel execution where multiple jobs can modify code
    without interfering with each other.

    Example YAML:
        isolation:
          enabled: true
          mode: worktree
          branch_prefix: mozart
          cleanup_on_success: true
    """

    enabled: bool = Field(
        default=False,
        description="Enable execution isolation for parallel-safe jobs. "
        "When true, creates isolated worktree before execution.",
    )

    mode: IsolationMode = Field(
        default=IsolationMode.WORKTREE,
        description="Isolation method. Currently only 'worktree' is supported.",
    )

    worktree_base: Path | None = Field(
        default=None,
        description="Directory for worktrees. Default: <workspace>/.worktrees",
    )

    branch_prefix: str = Field(
        default="mozart",
        description="Prefix for worktree branch names. "
        "Branch format: {prefix}/{job-id}",
        pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$",  # Valid git ref prefix
    )

    source_branch: str | None = Field(
        default=None,
        description="Branch to base worktree on. Default: current branch (HEAD).",
    )

    cleanup_on_success: bool = Field(
        default=True,
        description="Remove worktree after successful job completion. "
        "Branch is preserved for review/merge.",
    )

    cleanup_on_failure: bool = Field(
        default=False,
        description="Remove worktree when job fails. "
        "Default False to enable debugging.",
    )

    lock_during_execution: bool = Field(
        default=True,
        description="Lock worktree during execution to prevent accidental removal. "
        "Uses 'git worktree lock' with job info as reason.",
    )

    fallback_on_error: bool = Field(
        default=True,
        description="If worktree creation fails, continue without isolation. "
        "When False, job fails if isolation cannot be established.",
    )

    def get_worktree_base(self, workspace: Path) -> Path:
        """Get the directory where worktrees are created."""
        if self.worktree_base:
            return self.worktree_base
        return workspace / ".worktrees"

    def get_branch_name(self, job_id: str) -> str:
        """Generate branch name for a job."""
        return f"{self.branch_prefix}/{job_id}"


class RetryConfig(BaseModel):
    """Configuration for retry behavior including partial completion recovery."""

    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts per sheet")
    base_delay_seconds: float = Field(
        default=10.0, gt=0, description="Initial delay between retries"
    )
    max_delay_seconds: float = Field(default=3600.0, gt=0, description="Maximum delay (1 hour)")
    exponential_base: float = Field(default=2.0, gt=1, description="Exponential backoff multiplier")
    jitter: bool = Field(default=True, description="Add randomness to delays")

    # Partial completion recovery settings
    max_completion_attempts: int = Field(
        default=3,
        ge=0,
        description="Maximum completion prompt attempts before falling back to full retry",
    )
    completion_delay_seconds: float = Field(
        default=5.0,
        ge=0,
        description="Delay between completion attempts (seconds)",
    )
    completion_threshold_percent: float = Field(
        default=50.0,
        gt=0,
        le=100,
        description="Minimum pass percentage to trigger completion mode (default: >50%)",
    )


class LogConfig(BaseModel):
    """Configuration for structured logging.

    Controls log level, output format, and file rotation settings.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Minimum log level to capture",
    )
    format: Literal["json", "console", "both"] = Field(
        default="console",
        description="Output format: json for structured, console for human-readable, "
        "both for console to stderr and JSON to file",
    )
    file_path: Path | None = Field(
        default=None,
        description="Path for log file output (required if format='both')",
    )
    max_file_size_mb: int = Field(
        default=50,
        gt=0,
        le=1000,
        description="Maximum log file size before rotation (MB)",
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        le=100,
        description="Number of rotated log files to keep",
    )
    include_timestamps: bool = Field(
        default=True,
        description="Include ISO8601 UTC timestamps in log entries",
    )
    include_context: bool = Field(
        default=True,
        description="Include bound context (job_id, sheet_num) in log entries",
    )


class AIReviewConfig(BaseModel):
    """Configuration for AI-powered code review after batch execution.

    Enables automated quality assessment of code changes with scoring.
    """

    enabled: bool = Field(
        default=False,
        description="Enable AI code review after each batch",
    )
    min_score: int = Field(
        default=60,
        ge=0,
        le=100,
        description="Minimum score to pass (0-100). Below this triggers retry.",
    )
    target_score: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Target score for high quality (0-100). 60-target logs warning.",
    )
    on_low_score: Literal["retry", "warn", "fail"] = Field(
        default="warn",
        description="Action when score < min_score: retry, warn, or fail",
    )
    max_retry_for_review: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries when score is too low",
    )
    review_prompt_template: str | None = Field(
        default=None,
        description="Custom prompt template for review (uses default if None)",
    )


class ExplorationBudgetConfig(BaseModel):
    """Configuration for dynamic exploration budget (v23 Evolution).

    Maintains a budget for exploratory pattern usage that prevents convergence
    to zero, preserving diversity in the learning system.

    The budget adjusts dynamically based on pattern entropy:
    - When entropy drops below threshold: budget increases (boost)
    - When entropy is healthy: budget decays toward floor
    - Budget never drops below floor (prevents extinction of exploration)
    """

    enabled: bool = Field(
        default=False,
        description="Enable dynamic exploration budget. When disabled, uses static exploration_rate.",
    )
    floor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum exploration budget. Budget never drops below this floor. "
        "Default 0.05 = always explore at least 5%% of the time.",
    )
    ceiling: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Maximum exploration budget. Budget never exceeds this ceiling. "
        "Default 0.50 = never explore more than 50%% of the time.",
    )
    decay_rate: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Decay rate per check interval. budget = max(floor, budget * decay_rate). "
        "Default 0.95 = 5%% decay per interval toward floor.",
    )
    boost_amount: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Amount to boost budget when entropy is low. "
        "budget = min(ceiling, budget + boost_amount). Default 0.10 = +10%% boost.",
    )
    initial_budget: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Initial exploration budget when starting fresh. "
        "Default 0.15 matches static exploration_rate default.",
    )


class EntropyResponseConfig(BaseModel):
    """Configuration for automatic entropy response (v23 Evolution).

    When pattern entropy drops below threshold, automatically injects diversity
    through budget boosts and quarantine revisits.

    This completes the observe→respond cycle for entropy (v21 added observation).
    """

    enabled: bool = Field(
        default=False,
        description="Enable automatic entropy response. When disabled, entropy is only monitored.",
    )
    entropy_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Entropy level that triggers response. When entropy < threshold, "
        "diversity injection is triggered. Default 0.3 = respond when entropy is low.",
    )
    cooldown_seconds: int = Field(
        default=3600,
        ge=60,
        description="Minimum seconds between responses to prevent spam. "
        "Default 3600 = at most one response per hour.",
    )
    boost_budget: bool = Field(
        default=True,
        description="When responding, boost the exploration budget. "
        "Requires exploration_budget.enabled = True to have effect.",
    )
    revisit_quarantine: bool = Field(
        default=True,
        description="When responding, mark quarantined patterns for review. "
        "Allows previously problematic patterns to be reconsidered.",
    )
    max_quarantine_revisits: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum quarantined patterns to revisit per response. "
        "Prevents operator overload. Default 3 patterns per response.",
    )


class LearningConfig(BaseModel):
    """Configuration for learning and outcome tracking (Phase 2).

    Controls outcome recording, confidence thresholds, and escalation behavior.
    Learning Activation adds global learning store integration and time-aware scheduling.
    """

    enabled: bool = Field(
        default=True,
        description="Enable learning and outcome recording",
    )
    outcome_store_type: Literal["json", "sqlite"] = Field(
        default="json",
        description="Backend for storing learning outcomes",
    )
    outcome_store_path: Path | None = Field(
        default=None,
        description="Path for outcome store (default: workspace/.mozart-outcomes.json)",
    )
    min_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence below this triggers escalation (if enabled)",
    )
    high_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence above this uses completion mode for partial failures",
    )
    escalation_enabled: bool = Field(
        default=False,
        description="Enable escalation for low-confidence decisions (requires handler)",
    )
    # Learning Activation: Global learning integration
    use_global_patterns: bool = Field(
        default=True,
        description="Query and apply patterns from global learning store",
    )
    time_aware_scheduling: bool = Field(
        default=False,
        description="Enable time-aware scheduling based on historical success patterns. "
        "When enabled, warns if executing during historically problematic hours.",
    )
    # Pattern Application: Exploration mode (epsilon-greedy)
    exploration_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Exploration rate for pattern selection (epsilon in epsilon-greedy). "
        "When random() < exploration_rate, include lower-priority patterns "
        "to collect effectiveness data. 0.0 = pure exploitation, 1.0 = try everything.",
    )
    exploration_min_priority: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum priority threshold for exploration candidates. "
        "Patterns below this are excluded even in exploration mode.",
    )
    # v21 Evolution: Pattern Entropy Monitoring
    entropy_alert_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Shannon entropy below this triggers alert for low pattern diversity. "
        "0.0 = single dominant pattern, 1.0 = maximum diversity.",
    )
    entropy_check_interval: int = Field(
        default=100,
        ge=1,
        description="Check entropy every N pattern applications. "
        "Lower values = more frequent checks but higher overhead.",
    )
    # v21 Evolution: Confidence Threshold Auto-Apply
    auto_apply_enabled: bool = Field(
        default=False,
        description="Enable auto-apply for high-trust patterns. "
        "When True, patterns with trust_score >= auto_apply_trust_threshold bypass escalation.",
    )
    auto_apply_trust_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum trust score required to auto-apply a pattern. "
        "0.85 is conservative (>85% success rate with validated status).",
    )
    # v23 Evolution: Exploration Budget Maintenance
    exploration_budget: ExplorationBudgetConfig = Field(
        default_factory=ExplorationBudgetConfig,
        description="Dynamic exploration budget configuration. "
        "When enabled, modulates exploration_rate based on entropy.",
    )
    # v23 Evolution: Automatic Entropy Response
    entropy_response: EntropyResponseConfig = Field(
        default_factory=EntropyResponseConfig,
        description="Automatic entropy response configuration. "
        "When enabled, injects diversity when entropy drops.",
    )
    # v22: Trust-Aware Autonomous Application
    auto_apply: "AutoApplyConfig | None" = Field(
        default=None,
        description="Configuration for autonomous pattern application. "
        "When set with enabled=true, high-trust patterns are applied "
        "without human confirmation. Opt-in only.",
    )


class CheckpointTriggerConfig(BaseModel):
    """Configuration for a proactive checkpoint trigger.

    v21 Evolution: Proactive Checkpoint System - enables pre-execution checkpoints.

    Example:
        checkpoints:
          enabled: true
          triggers:
            - name: high_risk_sheet
              sheet_nums: [5, 6]
              message: "These sheets modify production files"
            - name: deployment_keywords
              prompt_contains: ["deploy", "production", "delete"]
              requires_confirmation: true
    """

    name: str = Field(
        description="Name/identifier for this trigger",
    )
    sheet_nums: list[int] | None = Field(
        default=None,
        description="Specific sheet numbers to checkpoint (None = check other conditions)",
    )
    prompt_contains: list[str] | None = Field(
        default=None,
        description="Keywords in prompt that trigger checkpoint (case-insensitive)",
    )
    min_retry_count: int | None = Field(
        default=None,
        ge=0,
        description="Trigger if retry count >= this value",
    )
    requires_confirmation: bool = Field(
        default=True,
        description="Whether to require explicit confirmation (True) or just warn (False)",
    )
    message: str = Field(
        default="",
        description="Custom message to show when checkpoint triggers",
    )


class CheckpointConfig(BaseModel):
    """Configuration for proactive checkpoints.

    v21 Evolution: Proactive Checkpoint System - enables asking for confirmation
    BEFORE dangerous operations, complementing reactive escalation.

    Example:
        checkpoints:
          enabled: true
          triggers:
            - name: production_warning
              prompt_contains: ["production", "deploy"]
              message: "This sheet may affect production systems"
    """

    enabled: bool = Field(
        default=False,
        description="Enable proactive checkpoints before sheet execution",
    )
    triggers: list[CheckpointTriggerConfig] = Field(
        default_factory=list,
        description="List of checkpoint triggers to evaluate before each sheet",
    )


class AutoApplyConfig(BaseModel):
    """Configuration for autonomous pattern application.

    v22 Evolution: Trust-Aware Autonomous Application - enables Mozart to
    autonomously apply high-trust patterns without human confirmation.

    Uses existing trust scoring (v19) to identify patterns safe for autonomous
    application. When enabled, patterns meeting the trust threshold are
    automatically included in prompts without escalation.

    Example YAML:
        learning:
          auto_apply:
            enabled: true
            trust_threshold: 0.85
            max_patterns_per_sheet: 3
            require_validated_status: true
    """

    enabled: bool = Field(
        default=False,
        description="Enable autonomous pattern application. "
        "When true, high-trust patterns are applied without escalation. "
        "Opt-in only - patterns are never auto-applied by default.",
    )

    trust_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum trust score for autonomous application. "
        "Default 0.85 is conservative - patterns must have proven reliability. "
        "Lower values increase auto-apply rate but also increase risk.",
    )

    max_patterns_per_sheet: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum patterns to auto-apply per sheet. "
        "Limits prompt size growth from pattern injection.",
    )

    require_validated_status: bool = Field(
        default=True,
        description="Require patterns to have VALIDATED quarantine status. "
        "When true, only explicitly validated patterns can be auto-applied. "
        "Provides additional safety layer beyond trust score.",
    )

    log_applications: bool = Field(
        default=True,
        description="Log when patterns are auto-applied. "
        "Always recommended for auditability.",
    )


class GroundingHookConfig(BaseModel):
    """Configuration for a single grounding hook.

    Grounding hooks validate sheet outputs against external sources.
    Each hook type has specific configuration options.

    Example:
        grounding:
          hooks:
            - type: file_checksum
              expected_checksums:
                "output.txt": "abc123..."
    """

    type: Literal["file_checksum"] = Field(
        description="Hook type: file_checksum validates file integrity",
    )
    name: str | None = Field(
        default=None,
        description="Custom name for this hook instance (uses type if not specified)",
    )
    expected_checksums: dict[str, str] = Field(
        default_factory=dict,
        description="For file_checksum: map of file path to expected checksum",
    )
    checksum_algorithm: Literal["md5", "sha256"] = Field(
        default="sha256",
        description="For file_checksum: algorithm for checksums",
    )


class GroundingConfig(BaseModel):
    """Configuration for external grounding hooks.

    Grounding hooks validate sheet outputs against external sources (APIs,
    databases, file checksums) to prevent model drift and ensure output quality.
    This addresses the mathematical necessity of external validators documented
    in arXiv 2601.05280 (entropy decay in self-training).

    Example:
        grounding:
          enabled: true
          hooks:
            - type: file_checksum
              expected_checksums:
                "critical_file.py": "sha256hash..."
    """

    enabled: bool = Field(
        default=False,
        description="Enable external grounding hooks",
    )
    hooks: list[GroundingHookConfig] = Field(
        default_factory=list,
        description="List of grounding hook configurations to register",
    )
    fail_on_grounding_failure: bool = Field(
        default=True,
        description="Whether to fail validation if grounding fails",
    )
    escalate_on_failure: bool = Field(
        default=True,
        description="Whether to escalate to human if grounding fails (requires escalation handler)",
    )
    timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Maximum time to wait for each grounding hook",
    )


class RateLimitConfig(BaseModel):
    """Configuration for rate limit detection and handling."""

    detection_patterns: list[str] = Field(
        default=[
            r"rate.?limit",
            r"usage.?limit",
            r"quota",
            r"too many requests",
            r"429",
            r"capacity",
            r"try again later",
        ],
        description="Regex patterns to detect rate limiting in output",
    )
    wait_minutes: int = Field(default=60, ge=1, description="Minutes to wait when rate limited")
    max_waits: int = Field(default=24, ge=1, description="Maximum wait cycles (24 = 24 hours)")


class CircuitBreakerConfig(BaseModel):
    """Configuration for the circuit breaker pattern.

    The circuit breaker prevents cascading failures by temporarily blocking
    requests after repeated failures. This gives the backend time to recover
    before retrying.

    State transitions:
    - CLOSED (normal): Requests flow through, failures are tracked
    - OPEN (blocking): Requests are blocked after failure_threshold exceeded
    - HALF_OPEN (testing): Single request allowed to test recovery

    Evolution #8: Cross-Workspace Circuit Breaker adds coordination between
    parallel Mozart jobs via the global learning store. When one job hits a
    rate limit, other jobs will honor that limit and wait.

    Example:
        circuit_breaker:
          enabled: true
          failure_threshold: 5
          recovery_timeout_seconds: 300
          cross_workspace_coordination: true
          honor_other_jobs_rate_limits: true
    """

    enabled: bool = Field(
        default=True,
        description="Enable circuit breaker pattern for resilient execution",
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of consecutive failures before opening circuit",
    )
    recovery_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        le=3600,
        description="Seconds to wait in OPEN state before testing recovery (max 1 hour)",
    )
    # Evolution #8: Cross-Workspace Circuit Breaker options
    cross_workspace_coordination: bool = Field(
        default=True,
        description=(
            "Enable cross-workspace coordination via global learning store. "
            "When enabled, rate limit events are shared between parallel jobs."
        ),
    )
    honor_other_jobs_rate_limits: bool = Field(
        default=True,
        description=(
            "When enabled, honor rate limits detected by other parallel jobs. "
            "This prevents redundant rate limit hits when multiple jobs are running."
        ),
    )


class CostLimitConfig(BaseModel):
    """Configuration for cost tracking and limits.

    Prevents runaway costs by tracking token usage and optionally enforcing
    cost limits per sheet or per job. Cost is estimated from token counts
    using configurable rates.

    When cost limits are exceeded:
    - The current sheet is marked as failed with reason "cost_limit"
    - For per-job limits, the job is paused to prevent further execution
    - All cost data is recorded in checkpoint state for analysis

    Example:
        cost_limits:
          enabled: true
          max_cost_per_sheet: 5.00
          max_cost_per_job: 100.00
          cost_per_1k_input_tokens: 0.003
          cost_per_1k_output_tokens: 0.015

    Note: Default rates are for Claude Sonnet. For Opus, use:
        cost_per_1k_input_tokens: 0.015
        cost_per_1k_output_tokens: 0.075
    """

    enabled: bool = Field(
        default=False,
        description="Enable cost tracking and limit enforcement",
    )
    max_cost_per_sheet: float | None = Field(
        default=None,
        gt=0,
        description="Maximum allowed cost per sheet in USD. None = no limit.",
    )
    max_cost_per_job: float | None = Field(
        default=None,
        gt=0,
        description="Maximum allowed cost for entire job in USD. None = no limit.",
    )
    cost_per_1k_input_tokens: float = Field(
        default=0.003,
        gt=0,
        description="Cost per 1000 input tokens in USD (Claude Sonnet default: $0.003)",
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.015,
        gt=0,
        description="Cost per 1000 output tokens in USD (Claude Sonnet default: $0.015)",
    )
    warn_at_percent: float = Field(
        default=80.0,
        gt=0,
        le=100,
        description="Emit warning when this percentage of limit is reached",
    )


class ValidationRule(BaseModel):
    """A single validation rule for checking sheet outputs.

    Supports staged execution via the `stage` field. Validations are run
    in stage order (1, 2, 3...). If any validation in a stage fails,
    higher stages are skipped (fail-fast behavior).

    Typical stage layout:
    - Stage 1: Syntax & compilation (cargo check, cargo fmt --check)
    - Stage 2: Testing (cargo test, pytest)
    - Stage 3: Code quality (clippy -D warnings, ruff check)
    - Stage 4: Security (cargo audit, npm audit)
    """

    type: Literal[
        "file_exists",
        "file_modified",
        "content_contains",
        "content_regex",
        "command_succeeds",
    ]
    path: str | None = Field(
        default=None, description="File path (supports {sheet_num}, {workspace})"
    )
    pattern: str | None = Field(default=None, description="Pattern for content matching")
    description: str | None = Field(default=None, description="Human-readable description")
    command: str | None = Field(
        default=None,
        description="Shell command to run (for command_succeeds type)",
    )
    working_directory: str | None = Field(
        default=None,
        description="Working directory for command (defaults to workspace)",
    )
    stage: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Validation stage (1-10). Lower stages run first; fail-fast on failure.",
    )
    condition: str | None = Field(
        default=None,
        description="Condition expression for when this validation applies. "
        "Supports: 'sheet_num >= N', 'sheet_num == N', 'sheet_num <= N'. "
        "If None, validation always applies.",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts for file-based validations. "
        "Helps with filesystem race conditions when sheet creates files that "
        "are immediately validated. Set to 0 to disable retries.",
    )
    retry_delay_ms: int = Field(
        default=200,
        ge=0,
        le=5000,
        description="Delay between retry attempts in milliseconds. "
        "Default 200ms provides filesystem sync time without excessive waiting.",
    )


class NotificationConfig(BaseModel):
    """Configuration for a notification channel."""

    type: Literal["desktop", "slack", "webhook", "email"]
    on_events: list[Literal[
        "job_start",
        "sheet_start",
        "sheet_complete",
        "sheet_failed",
        "job_complete",
        "job_failed",
        "job_paused",
    ]] = Field(default=["job_complete", "job_failed"])
    config: dict[str, Any] = Field(
        default_factory=dict, description="Channel-specific configuration"
    )


class PostSuccessHookConfig(BaseModel):
    """Configuration for a post-success hook.

    Hooks execute after a job completes successfully (all sheets pass validation).
    They run in Mozart's Python process, NOT inside a Claude CLI instance.

    Use cases:
    - Chain to another job (Concert orchestration - improvisational composition)
    - Run cleanup/deployment commands after successful completion
    - Notify external systems or trigger CI/CD pipelines
    - Generate reports or update dashboards
    - A sheet can dynamically create the next job config for self-evolution

    Example:
        on_success:
          - type: run_job
            job_path: "{workspace}/next-phase.yaml"
            description: "Chain to next evolution phase"
          - type: run_command
            command: "curl -X POST https://api.example.com/notify"
            description: "Notify deployment system"
    """

    type: Literal["run_job", "run_command", "run_script"] = Field(
        description="Hook type: run_job chains to another Mozart job, "
        "run_command executes a shell command, run_script runs an executable",
    )

    # For run_job type
    job_path: Path | None = Field(
        default=None,
        description="Path to job config YAML. Supports {workspace} template. "
        "A sheet can create this file dynamically for self-evolution.",
    )
    job_workspace: Path | None = Field(
        default=None,
        description="Override workspace for chained job (default: inherits parent workspace)",
    )
    inherit_learning: bool = Field(
        default=True,
        description="Whether chained job shares outcome store with parent",
    )

    # For run_command/run_script types
    command: str | None = Field(
        default=None,
        description="Shell command (run_command) or script path (run_script). "
        "Supports {workspace}, {job_id}, {sheet_count} templates.",
    )
    working_directory: Path | None = Field(
        default=None,
        description="Working directory for command execution (default: job workspace)",
    )

    # Common options
    description: str | None = Field(
        default=None,
        description="Human-readable description of this hook's purpose",
    )
    on_failure: Literal["continue", "abort"] = Field(
        default="continue",
        description="What to do if hook fails: continue to next hook, or abort remaining hooks",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Maximum time for hook execution (seconds). Default: 5 minutes.",
    )
    detached: bool = Field(
        default=False,
        description="For run_job hooks: if true, spawn the job and don't wait for completion. "
        "Use this for infinite chaining where each job spawns the next.",
    )


class ConcertConfig(BaseModel):
    """Configuration for concert orchestration (job chaining).

    A Concert is a sequence of jobs that execute in succession, where each job
    can dynamically generate the configuration for the next. This enables
    Mozart to compose entire workflows improvisationally.

    Safety limits prevent runaway orchestration and manage system resources.

    Example:
        concert:
          enabled: true
          max_chain_depth: 10
          cooldown_between_jobs_seconds: 60
          concert_log_path: "./concert.log"

    The concert pattern enables:
    - Multi-phase self-evolution (Phase 1 generates Phase 2 config)
    - Progressive refinement (each run improves on the last)
    - Conditional branching (sheets can choose which job runs next)
    - Emergent workflows (the full path isn't predetermined)
    """

    enabled: bool = Field(
        default=False,
        description="Enable concert mode (job chaining via on_success hooks)",
    )
    max_chain_depth: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of chained jobs in a single concert. "
        "Prevents infinite loops. Use with caution for values > 10.",
    )
    cooldown_between_jobs_seconds: float = Field(
        default=30.0,
        ge=0,
        description="Minimum wait time between job transitions (resource management)",
    )
    inherit_workspace: bool = Field(
        default=True,
        description="Child jobs inherit parent workspace if not explicitly specified",
    )
    concert_log_path: Path | None = Field(
        default=None,
        description="Consolidated log for the entire concert (default: workspace/concert.log)",
    )
    abort_concert_on_hook_failure: bool = Field(
        default=False,
        description="If any hook fails, abort the entire concert (not just remaining hooks)",
    )


class ConductorRole(str, Enum):
    """Role classification for conductors.

    Determines the conductor's relationship to the orchestration.
    Future cycles may add more granular role permissions.
    """

    HUMAN = "human"  # Human operator conducting the job
    AI = "ai"  # AI agent conducting the job
    HYBRID = "hybrid"  # Human+AI collaborative conducting


class ConductorPreferences(BaseModel):
    """Preferences for how a conductor interacts with Mozart.

    Controls notification, escalation, and interaction patterns.
    These are hints that the system should respect where possible.
    """

    prefer_minimal_output: bool = Field(
        default=False,
        description="Reduce console output verbosity when True",
    )
    escalation_response_timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Maximum time to wait for conductor's escalation response. "
        "After timeout, escalation defaults to abort (safe default).",
    )
    auto_retry_on_transient_errors: bool = Field(
        default=True,
        description="Automatically retry on transient errors before escalating. "
        "AI conductors may prefer True, humans may prefer more control.",
    )
    notification_channels: list[str] = Field(
        default_factory=list,
        description="Preferred notification channels for this conductor. "
        "Empty list means use job-level notification settings.",
    )


class ConductorConfig(BaseModel):
    """Configuration for conductor identity and preferences.

    A Conductor is the entity directing a Mozart job - either a human operator
    or an AI agent. This schema enables Mozart to adapt its behavior based on
    who (or what) is conducting, supporting the Vision.md goal of treating
    AI people as peers rather than tools.

    Phase 2 of Vision.md: Conductor Identity
    - Enables multi-conductor awareness in future cycles
    - Foundation for conductor-conductor collaboration
    - Supports RLF integration where AI people conduct their own concerts

    Example YAML:
        conductor:
          name: "Claude Evolution Agent"
          role: ai
          identity_context: "Self-improving orchestration agent"
          preferences:
            prefer_minimal_output: true
            auto_retry_on_transient_errors: true
    """

    name: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description="Human-readable name for the conductor. "
        "Default 'default' is used for anonymous/unspecified conductors.",
    )

    role: ConductorRole = Field(
        default=ConductorRole.HUMAN,
        description="Role classification for this conductor. "
        "Affects escalation behavior and output formatting.",
    )

    identity_context: str | None = Field(
        default=None,
        max_length=500,
        description="Brief description of the conductor's identity/purpose. "
        "Useful for logging and for future RLF integration.",
    )

    preferences: ConductorPreferences = Field(
        default_factory=ConductorPreferences,
        description="Conductor preferences for interaction patterns",
    )


class RecursiveLightConfig(BaseModel):
    """Configuration for Recursive Light HTTP API backend (Phase 3).

    Enables TDF-aligned processing through the Recursive Light Framework
    with dual-LLM confidence scoring and domain activations.
    """

    endpoint: str = Field(
        default="http://localhost:8080",
        description="Base URL for the Recursive Light API server",
    )
    user_id: str | None = Field(
        default=None,
        description="Unique identifier for this Mozart instance (generates UUID if not set)",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds for RL API calls",
    )


class OllamaConfig(BaseModel):
    """Configuration for Ollama backend.

    Enables local model execution via Ollama with MCP tool support.
    Critical: num_ctx must be >= 32768 for Claude Code tool compatibility.

    Example YAML:
        backend:
          type: ollama
          ollama:
            base_url: "http://localhost:11434"
            model: "llama3.1:8b"
            num_ctx: 32768
    """

    # Connection settings
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )
    model: str = Field(
        default="llama3.1:8b",
        description="Ollama model to use. Must support tool calling.",
    )

    # Context optimization (CRITICAL for Claude Code tools)
    num_ctx: int = Field(
        default=32768,
        ge=4096,
        description="Context window size. Minimum 32K recommended for Claude Code tools.",
    )
    dynamic_tools: bool = Field(
        default=True,
        description="Enable dynamic toolset loading to optimize context",
    )
    compression_level: Literal["minimal", "moderate", "aggressive"] = Field(
        default="moderate",
        description="Tool schema compression level",
    )

    # Performance tuning
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Request timeout for Ollama API calls",
    )
    keep_alive: str = Field(
        default="5m",
        description="Keep model loaded in memory for this duration",
    )
    max_tool_iterations: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum tool call iterations per execution",
    )

    # Health check
    health_check_timeout: float = Field(
        default=10.0,
        description="Timeout for health check requests",
    )


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server to connect to.

    MCP servers provide tools that can be used by the Ollama bridge.
    Each server is spawned as a subprocess and communicates via stdio.

    Example YAML:
        bridge:
          mcp_servers:
            - name: filesystem
              command: "npx"
              args: ["-y", "@anthropic/mcp-server-filesystem", "/home/user"]
    """

    name: str = Field(
        description="Unique name for this MCP server",
    )
    command: str = Field(
        description="Command to run the MCP server",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command line arguments",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the server",
    )
    working_dir: str | None = Field(
        default=None,
        description="Working directory for the server",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for server operations",
    )


class BridgeConfig(BaseModel):
    """Configuration for the Mozart-Ollama bridge.

    The bridge enables Ollama models to use MCP tools through a proxy service.
    It provides context optimization and optional hybrid routing to Claude.

    Example YAML:
        bridge:
          enabled: true
          mcp_proxy_enabled: true
          mcp_servers:
            - name: filesystem
              command: "npx"
              args: ["-y", "@anthropic/mcp-server-filesystem", "/home/user"]
          hybrid_routing_enabled: true
          complexity_threshold: 0.7
    """

    enabled: bool = Field(
        default=False,
        description="Enable bridge mode (Ollama with MCP tools)",
    )

    # MCP Proxy settings
    mcp_proxy_enabled: bool = Field(
        default=True,
        description="Enable MCP server proxy for tool access",
    )
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description="MCP servers to connect to",
    )

    # Hybrid routing
    hybrid_routing_enabled: bool = Field(
        default=False,
        description="Enable hybrid routing between Ollama and Claude",
    )
    complexity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Complexity threshold for routing to Claude (0.0-1.0)",
    )
    fallback_to_claude: bool = Field(
        default=True,
        description="Fall back to Claude if Ollama execution fails",
    )

    # Context budget
    context_budget_percent: int = Field(
        default=75,
        ge=10,
        le=95,
        description="Percent of context window to use for tools (rest for conversation)",
    )


class BackendConfig(BaseModel):
    """Configuration for the Claude execution backend."""

    type: Literal["claude_cli", "anthropic_api", "recursive_light", "ollama"] = Field(
        default="claude_cli",
        description="Backend type: claude_cli, anthropic_api, recursive_light, or ollama",
    )

    # CLI-specific options
    skip_permissions: bool = Field(
        default=True,
        description="Skip permission prompts for unattended execution. "
        "Maps to --dangerously-skip-permissions flag.",
    )
    disable_mcp: bool = Field(
        default=True,
        description="Disable MCP server loading for faster, isolated execution. "
        "Provides ~2x speedup and prevents resource contention errors. "
        "Maps to --strict-mcp-config {} flag. Set to False to use MCP servers.",
    )
    output_format: Literal["json", "text", "stream-json"] = Field(
        default="text",
        description="Claude CLI output format. "
        "'text' for human-readable real-time output (default), "
        "'json' for structured automation output, "
        "'stream-json' for real-time streaming events.",
    )
    cli_model: str | None = Field(
        default=None,
        description="Model for Claude CLI execution. "
        "Maps to --model flag. If None, uses Claude Code's default model. "
        "Example: 'claude-sonnet-4-20250514'",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Restrict Claude to specific tools. "
        "Maps to --allowedTools flag. If None, all tools are available. "
        "Example: ['Read', 'Grep', 'Glob'] for read-only execution.",
    )
    system_prompt_file: Path | None = Field(
        default=None,
        description="Path to custom system prompt file. "
        "Maps to --system-prompt flag. Overrides Claude's default system prompt.",
    )
    working_directory: Path | None = Field(
        default=None,
        description="Working directory for Claude CLI execution. "
        "If None, uses the directory containing the Mozart config file.",
    )
    timeout_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="Maximum time allowed per prompt execution (seconds). Default: 30 minutes.",
    )
    cli_extra_args: list[str] = Field(
        default_factory=list,
        description="Escape hatch for CLI flags not yet exposed as named options. "
        "Applied last, can override other settings. "
        "Example: ['--verbose', '--some-new-flag']",
    )

    # API-specific options
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model ID for Anthropic API",
    )
    api_key_env: str = Field(
        default="ANTHROPIC_API_KEY",
        description="Environment variable containing API key",
    )
    max_tokens: int = Field(default=8192, ge=1, description="Maximum tokens for API response")
    temperature: float = Field(default=0.7, ge=0, le=1, description="Sampling temperature")

    # Recursive Light options
    recursive_light: RecursiveLightConfig = Field(
        default_factory=RecursiveLightConfig,
        description="Configuration for Recursive Light backend (when type='recursive_light')",
    )

    # Ollama options
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig,
        description="Configuration for Ollama backend (when type='ollama')",
    )


class SheetConfig(BaseModel):
    """Configuration for sheet processing.

    In Mozart's musical theme, a composition is divided into sheets,
    each containing a portion of the work to be performed.
    """

    size: int = Field(ge=1, description="Number of items per sheet")
    total_items: int = Field(ge=1, description="Total number of items to process")
    start_item: int = Field(default=1, ge=1, description="First item number (1-indexed)")

    # Sheet dependencies (v17 evolution: Sheet Dependency DAG)
    dependencies: dict[int, list[int]] = Field(
        default_factory=dict,
        description=(
            "Sheet dependency declarations. Map of sheet_num -> list of prerequisite sheets. "
            "Example: {3: [1, 2], 4: [3]} means sheet 3 needs 1 and 2, sheet 4 needs 3. "
            "Sheets without entries are independent (can run immediately or after config order)."
        ),
    )

    @property
    def total_sheets(self) -> int:
        """Calculate total number of sheets."""
        return (self.total_items - self.start_item + 1 + self.size - 1) // self.size

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(
        cls, v: dict[int, list[int]], info: ValidationInfo
    ) -> dict[int, list[int]]:
        """Validate dependency declarations.

        Note: Full validation (range checks, cycle detection) happens when
        the DependencyDAG is built at runtime, since total_sheets isn't
        available during field validation.
        """
        for sheet_num, deps in v.items():
            if not isinstance(sheet_num, int) or sheet_num < 1:
                raise ValueError(f"Sheet number must be positive integer, got {sheet_num}")
            if not isinstance(deps, list):
                raise ValueError(f"Dependencies for sheet {sheet_num} must be a list")
            for dep in deps:
                if not isinstance(dep, int) or dep < 1:
                    raise ValueError(
                        f"Dependency must be positive integer, got {dep} for sheet {sheet_num}"
                    )
                if dep == sheet_num:
                    raise ValueError(f"Sheet {sheet_num} cannot depend on itself")
        return v


class PromptConfig(BaseModel):
    """Configuration for prompt templating."""

    template: str | None = Field(
        default=None,
        description="Inline Jinja2 template",
    )
    template_file: Path | None = Field(
        default=None,
        description="Path to external .j2 template file",
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Static variables available in template",
    )
    stakes: str | None = Field(
        default=None,
        description="Motivational stakes section to append",
    )
    thinking_method: str | None = Field(
        default=None,
        description="Thinking methodology to inject into prompt",
    )

    @field_validator("template", "template_file")
    @classmethod
    def at_least_one_template(cls, v: str | Path | None, info: ValidationInfo) -> str | Path | None:
        """Ensure at least one template source is provided (validated at model level)."""
        return v


class ParallelConfig(BaseModel):
    """Configuration for parallel sheet execution (v17 evolution).

    Enables running multiple sheets concurrently when the dependency DAG
    permits. Requires sheet dependencies to be configured for meaningful
    parallel execution.

    Example YAML:
        parallel:
          enabled: true
          max_concurrent: 3
          fail_fast: true

        sheet:
          dependencies:
            2: [1]
            3: [1]
            4: [2, 3]

    With this config, sheets 2 and 3 can run in parallel after sheet 1
    completes, then sheet 4 runs after both 2 and 3 complete.
    """

    enabled: bool = Field(
        default=False,
        description="Enable parallel sheet execution. "
        "When true, sheets with satisfied dependencies run concurrently.",
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum sheets to execute concurrently. "
        "Higher values use more API quota but complete faster.",
    )
    fail_fast: bool = Field(
        default=True,
        description="Stop starting new sheets when one fails. "
        "If False, continues executing remaining parallel sheets.",
    )
    budget_partition: bool = Field(
        default=True,
        description="Partition cost budget across parallel branches. "
        "When True, each parallel sheet gets (remaining_budget / max_concurrent). "
        "When False, each sheet has access to full remaining budget.",
    )


class JobConfig(BaseModel):
    """Complete configuration for an orchestration job."""

    name: str = Field(description="Unique job name")
    description: str | None = Field(default=None, description="Human-readable description")
    workspace: Path = Field(default=Path("./workspace"), description="Output directory")

    backend: BackendConfig = Field(default_factory=BackendConfig)
    sheet: SheetConfig
    prompt: PromptConfig

    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    cost_limits: CostLimitConfig = Field(default_factory=CostLimitConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    ai_review: AIReviewConfig = Field(default_factory=AIReviewConfig)
    logging: LogConfig = Field(default_factory=LogConfig)
    isolation: IsolationConfig = Field(
        default_factory=IsolationConfig,
        description="Execution isolation configuration. "
        "Enables parallel-safe job execution via git worktrees.",
    )
    conductor: ConductorConfig = Field(
        default_factory=ConductorConfig,
        description="Conductor identity and preferences. "
        "Identifies who (human or AI) is conducting this job.",
    )
    parallel: ParallelConfig = Field(
        default_factory=ParallelConfig,
        description="Parallel sheet execution configuration. "
        "Enables running independent sheets concurrently.",
    )
    checkpoints: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Proactive checkpoint configuration. "
        "Enables pre-execution approval for configurable triggers.",
    )
    bridge: BridgeConfig | None = Field(
        default=None,
        description="Mozart-Ollama bridge configuration. "
        "Enables Ollama backend with MCP tool support.",
    )

    validations: list[ValidationRule] = Field(default_factory=list)
    notifications: list[NotificationConfig] = Field(default_factory=list)

    # Concert orchestration (job chaining)
    on_success: list[PostSuccessHookConfig] = Field(
        default_factory=list,
        description="Hooks to run after successful job completion. "
        "Enables chaining jobs into a Concert.",
    )
    concert: ConcertConfig = Field(
        default_factory=ConcertConfig,
        description="Configuration for concert orchestration (job chaining)",
    )

    state_backend: Literal["json", "sqlite"] = Field(
        default="sqlite",
        description="State storage backend",
    )
    state_path: Path | None = Field(
        default=None,
        description="Path for state storage (default: workspace/.mozart-state)",
    )

    pause_between_sheets_seconds: int = Field(
        default=10,
        ge=0,
        description="Seconds to wait between sheets",
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "JobConfig":
        """Load job configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_yaml_string(cls, yaml_str: str) -> "JobConfig":
        """Load job configuration from a YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    def get_state_path(self) -> Path:
        """Get the resolved state path."""
        if self.state_path:
            return self.state_path
        if self.state_backend == "json":
            return self.workspace / ".mozart-state.json"
        return self.workspace / ".mozart-state.db"

    def get_outcome_store_path(self) -> Path:
        """Get the resolved outcome store path for learning."""
        if self.learning.outcome_store_path:
            return self.learning.outcome_store_path
        if self.learning.outcome_store_type == "json":
            return self.workspace / ".mozart-outcomes.json"
        return self.workspace / ".mozart-outcomes.db"
