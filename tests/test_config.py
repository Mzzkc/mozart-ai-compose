"""Tests for mozart.core.config module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mozart.core.config import (
    BackendConfig,
    SheetConfig,
    JobConfig,
    PromptConfig,
    RateLimitConfig,
    RetryConfig,
    ValidationRule,
)


class TestRetryConfig:
    """Tests for RetryConfig model."""

    def test_defaults(self):
        """Test default values are applied."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_seconds == 10.0
        assert config.max_delay_seconds == 3600.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.max_completion_attempts == 3

    def test_custom_values(self):
        """Test custom values override defaults."""
        config = RetryConfig(
            max_retries=5,
            base_delay_seconds=5.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.base_delay_seconds == 5.0
        assert config.jitter is False

    def test_validation_max_retries_non_negative(self):
        """Test max_retries must be non-negative."""
        with pytest.raises(ValidationError):
            RetryConfig(max_retries=-1)

    def test_validation_base_delay_positive(self):
        """Test base_delay_seconds must be positive."""
        with pytest.raises(ValidationError):
            RetryConfig(base_delay_seconds=0)


class TestSheetConfig:
    """Tests for SheetConfig model."""

    def test_required_fields(self):
        """Test required fields are enforced."""
        with pytest.raises(ValidationError):
            SheetConfig()

    def test_total_sheets_calculation(self):
        """Test total_sheets is calculated correctly."""
        config = SheetConfig(size=10, total_items=35)
        assert config.total_sheets == 4  # ceil(35/10)

        config = SheetConfig(size=10, total_items=30)
        assert config.total_sheets == 3  # exact

    def test_start_item_default(self):
        """Test start_item defaults to 1."""
        config = SheetConfig(size=10, total_items=30)
        assert config.start_item == 1


class TestValidationRule:
    """Tests for ValidationRule model."""

    def test_file_exists_rule(self):
        """Test file_exists validation rule."""
        rule = ValidationRule(
            type="file_exists",
            path="{workspace}/output.txt",
            description="Output file exists",
        )
        assert rule.type == "file_exists"
        assert rule.path == "{workspace}/output.txt"
        assert rule.pattern is None

    def test_content_contains_rule(self):
        """Test content_contains validation rule."""
        rule = ValidationRule(
            type="content_contains",
            path="{workspace}/log.txt",
            pattern="SUCCESS",
            description="Success marker present",
        )
        assert rule.type == "content_contains"
        assert rule.pattern == "SUCCESS"


class TestBackendConfig:
    """Tests for BackendConfig model."""

    def test_claude_cli_default(self):
        """Test claude_cli is the default backend type."""
        config = BackendConfig()
        assert config.type == "claude_cli"
        assert config.skip_permissions is True

    def test_anthropic_api_config(self):
        """Test anthropic_api backend configuration."""
        config = BackendConfig(
            type="anthropic_api",
            model="claude-sonnet-4-20250514",
            api_key_env="ANTHROPIC_API_KEY",
        )
        assert config.type == "anthropic_api"
        assert config.model == "claude-sonnet-4-20250514"


class TestPromptConfig:
    """Tests for PromptConfig model."""

    def test_inline_template(self):
        """Test inline template configuration."""
        config = PromptConfig(template="Hello {{ name }}!")
        assert config.template == "Hello {{ name }}!"
        assert config.template_file is None

    def test_variables(self):
        """Test custom variables in prompt config."""
        config = PromptConfig(
            template="{{ greeting }}",
            variables={"greeting": "Hello!"},
        )
        assert config.variables["greeting"] == "Hello!"


class TestJobConfig:
    """Tests for JobConfig model."""

    def test_from_dict(self, sample_config_dict: dict):
        """Test creating JobConfig from dictionary."""
        config = JobConfig(**sample_config_dict)
        assert config.name == "test-job"
        assert config.sheet.total_items == 30
        assert config.backend.type == "claude_cli"

    def test_from_yaml(self, sample_yaml_config: Path):
        """Test loading JobConfig from YAML file."""
        config = JobConfig.from_yaml(sample_yaml_config)
        assert config.name == "test-job"
        assert config.sheet.size == 10

    def test_workspace_default(self, sample_config_dict: dict):
        """Test workspace defaults to ./workspace."""
        config = JobConfig(**sample_config_dict)
        assert config.workspace == Path("./workspace")

    def test_custom_workspace(self, sample_config_dict: dict):
        """Test custom workspace path."""
        sample_config_dict["workspace"] = "/custom/path"
        config = JobConfig(**sample_config_dict)
        assert config.workspace == Path("/custom/path")
