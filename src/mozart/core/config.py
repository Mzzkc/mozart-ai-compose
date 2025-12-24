"""Configuration models for Mozart jobs.

Defines Pydantic models for loading and validating YAML job configurations.
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class RetryConfig(BaseModel):
    """Configuration for retry behavior including partial completion recovery."""

    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts per batch")
    base_delay_seconds: float = Field(default=10.0, gt=0, description="Initial delay between retries")
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
    path: Optional[str] = Field(default=None, description="File path (supports {batch_num}, {workspace})")
    pattern: Optional[str] = Field(default=None, description="Pattern for content matching")
    description: Optional[str] = Field(default=None, description="Human-readable description")


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
    config: dict = Field(default_factory=dict, description="Channel-specific configuration")


class BackendConfig(BaseModel):
    """Configuration for the Claude execution backend."""

    type: Literal["claude_cli", "anthropic_api"] = Field(
        default="claude_cli",
        description="Backend type: claude_cli (subprocess) or anthropic_api (direct)",
    )

    # CLI-specific options
    skip_permissions: bool = Field(
        default=True,
        description="Pass --dangerously-skip-permissions to claude CLI",
    )
    output_format: Optional[Literal["json", "text", "stream-json"]] = Field(
        default=None,
        description="Output format for claude CLI",
    )
    working_directory: Optional[Path] = Field(
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

    template: Optional[str] = Field(
        default=None,
        description="Inline Jinja2 template",
    )
    template_file: Optional[Path] = Field(
        default=None,
        description="Path to external .j2 template file",
    )
    variables: dict = Field(
        default_factory=dict,
        description="Static variables available in template",
    )
    stakes: Optional[str] = Field(
        default=None,
        description="Motivational stakes section to append",
    )
    thinking_method: Optional[str] = Field(
        default=None,
        description="Thinking methodology to inject into prompt",
    )

    @field_validator("template", "template_file")
    @classmethod
    def at_least_one_template(cls, v: Optional[str | Path], info) -> Optional[str | Path]:
        """Ensure at least one template source is provided (validated at model level)."""
        return v


class JobConfig(BaseModel):
    """Complete configuration for an orchestration job."""

    name: str = Field(description="Unique job name")
    description: Optional[str] = Field(default=None, description="Human-readable description")
    workspace: Path = Field(default=Path("./workspace"), description="Output directory")

    backend: BackendConfig = Field(default_factory=BackendConfig)
    batch: BatchConfig
    prompt: PromptConfig

    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    validations: list[ValidationRule] = Field(default_factory=list)
    notifications: list[NotificationConfig] = Field(default_factory=list)

    state_backend: Literal["json", "sqlite"] = Field(
        default="sqlite",
        description="State storage backend",
    )
    state_path: Optional[Path] = Field(
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
