"""Configuration models for Mozart jobs.

Defines Pydantic models for loading and validating YAML job configurations.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator


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


class BackendConfig(BaseModel):
    """Configuration for the Claude execution backend."""

    type: Literal["claude_cli", "anthropic_api", "recursive_light"] = Field(
        default="claude_cli",
        description="Backend type: claude_cli, anthropic_api, or recursive_light",
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


class SheetConfig(BaseModel):
    """Configuration for sheet processing.

    In Mozart's musical theme, a composition is divided into sheets,
    each containing a portion of the work to be performed.
    """

    size: int = Field(ge=1, description="Number of items per sheet")
    total_items: int = Field(ge=1, description="Total number of items to process")
    start_item: int = Field(default=1, ge=1, description="First item number (1-indexed)")

    @property
    def total_sheets(self) -> int:
        """Calculate total number of sheets."""
        return (self.total_items - self.start_item + 1 + self.size - 1) // self.size


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
    ai_review: AIReviewConfig = Field(default_factory=AIReviewConfig)
    logging: LogConfig = Field(default_factory=LogConfig)

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
