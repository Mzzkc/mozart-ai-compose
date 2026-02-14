"""Execution and retry configuration models.

Defines models for retry behavior, rate limiting, circuit breaker,
cost limits, and parallel execution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def _validate_delay_range(self) -> RetryConfig:
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError(
                f"base_delay_seconds ({self.base_delay_seconds}) must not exceed "
                f"max_delay_seconds ({self.max_delay_seconds})"
            )
        return self


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
        "When False, each sheet has access to full remaining budget. "
        "NOTE: Not yet implemented — this field is accepted but not enforced. "
        "Cost checks currently use global total regardless of this setting.",
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

    @model_validator(mode="after")
    def _check_type_specific_fields(self) -> ValidationRule:
        """Validate that type-specific required fields are present."""
        if self.type in ("file_exists", "file_modified", "content_contains", "content_regex"):
            if self.path is None:
                raise ValueError(
                    f"Validation type '{self.type}' requires 'path' field"
                )
        if self.type in ("content_contains", "content_regex"):
            if self.pattern is None:
                raise ValueError(
                    f"Validation type '{self.type}' requires 'pattern' field"
                )
        if self.type == "command_succeeds":
            if self.command is None:
                raise ValueError(
                    "Validation type 'command_succeeds' requires 'command' field"
                )
        return self
