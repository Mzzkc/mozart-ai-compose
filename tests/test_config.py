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

    def test_disable_mcp_default(self):
        """Test disable_mcp defaults to True for faster batch execution."""
        config = BackendConfig()
        assert config.disable_mcp is True

    def test_disable_mcp_can_be_disabled(self):
        """Test MCP can be re-enabled when needed."""
        config = BackendConfig(disable_mcp=False)
        assert config.disable_mcp is False

    def test_output_format_default(self):
        """Test output_format defaults to json for automation."""
        config = BackendConfig()
        assert config.output_format == "json"

    def test_output_format_json(self):
        """Test output_format can be json."""
        config = BackendConfig(output_format="json")
        assert config.output_format == "json"

    def test_output_format_text(self):
        """Test output_format can be text."""
        config = BackendConfig(output_format="text")
        assert config.output_format == "text"

    def test_output_format_stream_json(self):
        """Test output_format can be stream-json."""
        config = BackendConfig(output_format="stream-json")
        assert config.output_format == "stream-json"

    def test_cli_model_default_none(self):
        """Test cli_model defaults to None (uses Claude Code default)."""
        config = BackendConfig()
        assert config.cli_model is None

    def test_cli_model_override(self):
        """Test cli_model can be set to specific model."""
        config = BackendConfig(cli_model="claude-sonnet-4-20250514")
        assert config.cli_model == "claude-sonnet-4-20250514"

    def test_allowed_tools_default_none(self):
        """Test allowed_tools defaults to None (all tools available)."""
        config = BackendConfig()
        assert config.allowed_tools is None

    def test_allowed_tools_restriction(self):
        """Test allowed_tools can restrict to specific tools."""
        config = BackendConfig(allowed_tools=["Read", "Grep", "Glob"])
        assert config.allowed_tools == ["Read", "Grep", "Glob"]

    def test_allowed_tools_empty_list(self):
        """Test allowed_tools as empty list (very restrictive)."""
        config = BackendConfig(allowed_tools=[])
        assert config.allowed_tools == []

    def test_system_prompt_file_default_none(self):
        """Test system_prompt_file defaults to None (default prompt)."""
        config = BackendConfig()
        assert config.system_prompt_file is None

    def test_system_prompt_file_path(self):
        """Test system_prompt_file accepts Path."""
        config = BackendConfig(system_prompt_file=Path("/custom/prompt.md"))
        assert config.system_prompt_file == Path("/custom/prompt.md")

    def test_timeout_seconds_default(self):
        """Test timeout_seconds default is 30 minutes."""
        config = BackendConfig()
        assert config.timeout_seconds == 1800.0

    def test_cli_extra_args_default_empty(self):
        """Test cli_extra_args defaults to empty list."""
        config = BackendConfig()
        assert config.cli_extra_args == []

    def test_cli_extra_args_escape_hatch(self):
        """Test cli_extra_args allows raw flags as escape hatch."""
        config = BackendConfig(cli_extra_args=["--verbose", "--some-new-flag"])
        assert config.cli_extra_args == ["--verbose", "--some-new-flag"]

    def test_backwards_compatibility_minimal(self):
        """Test minimal config (pre-new-fields) still works."""
        # This is what old configs would look like
        config = BackendConfig(
            type="claude_cli",
            skip_permissions=True,
        )
        assert config.type == "claude_cli"
        assert config.skip_permissions is True
        # New fields should have sensible defaults
        assert config.disable_mcp is True
        assert config.output_format == "json"
        assert config.cli_model is None
        assert config.allowed_tools is None

    def test_backwards_compatibility_with_cli_extra_args(self):
        """Test old configs using cli_extra_args still work."""
        # This simulates configs that used cli_extra_args for MCP disable
        config = BackendConfig(
            type="claude_cli",
            skip_permissions=True,
            cli_extra_args=["--strict-mcp-config", "{}"],
        )
        assert config.cli_extra_args == ["--strict-mcp-config", "{}"]
        # Note: Both disable_mcp=True default AND cli_extra_args will
        # add the flag, but CLI should handle duplicate gracefully

    def test_full_cli_config(self):
        """Test fully specified CLI config with all new options."""
        config = BackendConfig(
            type="claude_cli",
            skip_permissions=True,
            disable_mcp=False,  # Explicitly enable MCP
            output_format="stream-json",
            cli_model="claude-sonnet-4-20250514",
            allowed_tools=["Read", "Edit", "Write"],
            system_prompt_file=Path("./prompts/custom.md"),
            working_directory=Path("/project"),
            timeout_seconds=3600,
            cli_extra_args=["--verbose"],
        )
        assert config.type == "claude_cli"
        assert config.skip_permissions is True
        assert config.disable_mcp is False
        assert config.output_format == "stream-json"
        assert config.cli_model == "claude-sonnet-4-20250514"
        assert config.allowed_tools == ["Read", "Edit", "Write"]
        assert config.system_prompt_file == Path("./prompts/custom.md")
        assert config.working_directory == Path("/project")
        assert config.timeout_seconds == 3600
        assert config.cli_extra_args == ["--verbose"]


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
