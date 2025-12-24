"""Configuration models for Mozart jobs.

Defines Pydantic models for loading and validating YAML job configurations.
"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationInfo, field_validator


class RetryConfig(BaseModel):
    """Configuration for retry behavior including partial completion recovery."""

    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts per batch")
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


class LearningConfig(BaseModel):
    """Configuration for learning and outcome tracking (Phase 2).

    Controls outcome recording, confidence thresholds, and escalation behavior.
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


class ValidationRule(BaseModel):
    """A single validation rule for checking batch outputs."""

    type: Literal["file_exists", "file_modified", "content_contains", "content_regex"]
    path: str | None = Field(
        default=None, description="File path (supports {batch_num}, {workspace})"
    )
    pattern: str | None = Field(default=None, description="Pattern for content matching")
    description: str | None = Field(default=None, description="Human-readable description")


class NotificationConfig(BaseModel):
    """Configuration for a notification channel."""

    type: Literal["desktop", "slack", "webhook", "email"]
    on_events: list[Literal[
        "job_start",
        "batch_start",
        "batch_complete",
        "batch_failed",
        "job_complete",
        "job_failed",
        "job_paused",
    ]] = Field(default=["job_complete", "job_failed"])
    config: dict[str, Any] = Field(
        default_factory=dict, description="Channel-specific configuration"
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
        description="Pass --dangerously-skip-permissions to claude CLI",
    )
    output_format: Literal["json", "text", "stream-json"] | None = Field(
        default=None,
        description="Output format for claude CLI",
    )
    working_directory: Path | None = Field(
        default=None,
        description="Working directory for claude CLI execution",
    )
    timeout_seconds: float = Field(
        default=1800.0,
        gt=0,
        description="Maximum time allowed per prompt execution (seconds). Default: 30 minutes.",
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


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    size: int = Field(ge=1, description="Number of items per batch")
    total_items: int = Field(ge=1, description="Total number of items to process")
    start_item: int = Field(default=1, ge=1, description="First item number (1-indexed)")

    @property
    def total_batches(self) -> int:
        """Calculate total number of batches."""
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
    batch: BatchConfig
    prompt: PromptConfig

    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)

    validations: list[ValidationRule] = Field(default_factory=list)
    notifications: list[NotificationConfig] = Field(default_factory=list)

    state_backend: Literal["json", "sqlite"] = Field(
        default="sqlite",
        description="State storage backend",
    )
    state_path: Path | None = Field(
        default=None,
        description="Path for state storage (default: workspace/.mozart-state)",
    )

    pause_between_batches_seconds: int = Field(
        default=10,
        ge=0,
        description="Seconds to wait between batches",
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
