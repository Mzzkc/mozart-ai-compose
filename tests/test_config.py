"""Tests for mozart.core.config module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from mozart.core.config import (
    BackendConfig,
    ConductorConfig,
    ConductorPreferences,
    ConductorRole,
    CostLimitConfig,
    IsolationConfig,
    LearningConfig,
    RateLimitConfig,
    RetryConfig,
    SheetConfig,
    JobConfig,
    PromptConfig,
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
        """Test output_format defaults to text for human-readable output."""
        config = BackendConfig()
        assert config.output_format == "text"

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

    def test_timeout_overrides_default_empty(self):
        """Test timeout_overrides defaults to empty dict."""
        config = BackendConfig()
        assert config.timeout_overrides == {}

    def test_timeout_overrides_per_sheet(self):
        """Test per-sheet timeout overrides."""
        config = BackendConfig(timeout_overrides={1: 60.0, 7: 28800.0})
        assert config.timeout_overrides == {1: 60.0, 7: 28800.0}
        assert config.timeout_overrides.get(1) == 60.0
        assert config.timeout_overrides.get(7) == 28800.0
        assert config.timeout_overrides.get(2) is None

    def test_timeout_overrides_does_not_affect_global(self):
        """Test that per-sheet overrides don't change the global timeout."""
        config = BackendConfig(
            timeout_seconds=2400.0,
            timeout_overrides={7: 28800.0},
        )
        assert config.timeout_seconds == 2400.0
        assert config.timeout_overrides[7] == 28800.0

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
        assert config.output_format == "text"
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


class TestConductorPreferences:
    """Tests for ConductorPreferences model."""

    def test_defaults(self):
        """Test default values for conductor preferences."""
        prefs = ConductorPreferences()
        assert prefs.prefer_minimal_output is False
        assert prefs.escalation_response_timeout_seconds == 300.0
        assert prefs.auto_retry_on_transient_errors is True
        assert prefs.notification_channels == []

    def test_custom_values(self):
        """Test custom conductor preferences."""
        prefs = ConductorPreferences(
            prefer_minimal_output=True,
            escalation_response_timeout_seconds=60.0,
            auto_retry_on_transient_errors=False,
            notification_channels=["slack", "desktop"],
        )
        assert prefs.prefer_minimal_output is True
        assert prefs.escalation_response_timeout_seconds == 60.0
        assert prefs.auto_retry_on_transient_errors is False
        assert prefs.notification_channels == ["slack", "desktop"]

    def test_timeout_must_be_positive(self):
        """Test that escalation timeout must be positive."""
        with pytest.raises(ValidationError):
            ConductorPreferences(escalation_response_timeout_seconds=0)
        with pytest.raises(ValidationError):
            ConductorPreferences(escalation_response_timeout_seconds=-10)


class TestConductorConfig:
    """Tests for ConductorConfig model (Vision.md Phase 2)."""

    def test_defaults(self):
        """Test default conductor configuration values."""
        config = ConductorConfig()
        assert config.name == "default"
        assert config.role == ConductorRole.HUMAN
        assert config.identity_context is None
        assert config.preferences is not None
        assert isinstance(config.preferences, ConductorPreferences)

    def test_human_conductor(self):
        """Test human conductor configuration."""
        config = ConductorConfig(
            name="Alice",
            role=ConductorRole.HUMAN,
            identity_context="Lead engineer on the project",
        )
        assert config.name == "Alice"
        assert config.role == ConductorRole.HUMAN
        assert config.identity_context == "Lead engineer on the project"

    def test_ai_conductor(self):
        """Test AI conductor configuration (Vision.md: AI people as peers)."""
        config = ConductorConfig(
            name="Claude Evolution Agent",
            role=ConductorRole.AI,
            identity_context="Self-improving orchestration agent created by RLF",
            preferences=ConductorPreferences(
                prefer_minimal_output=True,
                auto_retry_on_transient_errors=True,
            ),
        )
        assert config.name == "Claude Evolution Agent"
        assert config.role == ConductorRole.AI
        assert config.preferences.prefer_minimal_output is True

    def test_hybrid_conductor(self):
        """Test hybrid conductor configuration (human+AI collaboration)."""
        config = ConductorConfig(
            name="Team Alpha",
            role=ConductorRole.HYBRID,
            identity_context="Human-AI pair conducting evolution",
        )
        assert config.role == ConductorRole.HYBRID

    def test_role_enum_values(self):
        """Test all ConductorRole enum values are valid."""
        assert ConductorRole.HUMAN.value == "human"
        assert ConductorRole.AI.value == "ai"
        assert ConductorRole.HYBRID.value == "hybrid"

    def test_role_from_string(self):
        """Test creating conductor with role as string (YAML loading)."""
        config = ConductorConfig(name="Test", role="ai")
        assert config.role == ConductorRole.AI

    def test_invalid_role(self):
        """Test that invalid role raises ValidationError."""
        with pytest.raises(ValidationError):
            ConductorConfig(role="invalid_role")

    def test_name_validation_min_length(self):
        """Test that name must have at least 1 character."""
        with pytest.raises(ValidationError):
            ConductorConfig(name="")

    def test_name_validation_max_length(self):
        """Test that name is limited to 100 characters."""
        # 100 chars should work
        config = ConductorConfig(name="a" * 100)
        assert len(config.name) == 100
        # 101 chars should fail
        with pytest.raises(ValidationError):
            ConductorConfig(name="a" * 101)

    def test_identity_context_max_length(self):
        """Test that identity_context is limited to 500 characters."""
        # 500 chars should work
        config = ConductorConfig(identity_context="a" * 500)
        assert len(config.identity_context) == 500
        # 501 chars should fail
        with pytest.raises(ValidationError):
            ConductorConfig(identity_context="a" * 501)

    def test_nested_preferences(self):
        """Test that nested preferences are properly parsed."""
        config = ConductorConfig(
            name="Test",
            preferences={
                "prefer_minimal_output": True,
                "escalation_response_timeout_seconds": 120.0,
            },
        )
        assert config.preferences.prefer_minimal_output is True
        assert config.preferences.escalation_response_timeout_seconds == 120.0

    def test_serialization_roundtrip(self):
        """Test that conductor config serializes and deserializes correctly."""
        original = ConductorConfig(
            name="Claude Agent",
            role=ConductorRole.AI,
            identity_context="Test agent",
            preferences=ConductorPreferences(
                prefer_minimal_output=True,
                notification_channels=["slack"],
            ),
        )
        # Serialize to dict
        data = original.model_dump()
        # Deserialize back
        restored = ConductorConfig.model_validate(data)
        assert restored.name == original.name
        assert restored.role == original.role
        assert restored.identity_context == original.identity_context
        assert restored.preferences.prefer_minimal_output == original.preferences.prefer_minimal_output
        assert restored.preferences.notification_channels == original.preferences.notification_channels


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

    def test_conductor_default(self, sample_config_dict: dict):
        """Test conductor has default configuration."""
        config = JobConfig(**sample_config_dict)
        assert config.conductor is not None
        assert config.conductor.name == "default"
        assert config.conductor.role == ConductorRole.HUMAN

    def test_conductor_custom(self, sample_config_dict: dict):
        """Test custom conductor configuration."""
        sample_config_dict["conductor"] = {
            "name": "Evolution Agent",
            "role": "ai",
            "identity_context": "Self-improving orchestration",
            "preferences": {
                "prefer_minimal_output": True,
            },
        }
        config = JobConfig(**sample_config_dict)
        assert config.conductor.name == "Evolution Agent"
        assert config.conductor.role == ConductorRole.AI
        assert config.conductor.identity_context == "Self-improving orchestration"
        assert config.conductor.preferences.prefer_minimal_output is True


class TestIsolationConfig:
    """Tests for IsolationConfig edge cases and defaults."""

    def test_defaults(self):
        config = IsolationConfig()
        assert config.enabled is False
        assert config.mode == "worktree"
        assert config.cleanup_on_success is True
        assert config.cleanup_on_failure is False

    def test_enabled_with_worktree(self):
        config = IsolationConfig(enabled=True, mode="worktree")
        assert config.enabled is True
        assert config.mode == "worktree"

    def test_lock_during_execution_default(self):
        config = IsolationConfig()
        assert config.lock_during_execution is True

    def test_fallback_on_error_default(self):
        config = IsolationConfig()
        assert config.fallback_on_error is True


class TestLearningConfig:
    """Tests for LearningConfig validation and edge cases."""

    def test_defaults(self):
        config = LearningConfig()
        assert config.enabled is True
        assert config.high_confidence_threshold == 0.7
        assert config.min_confidence_threshold == 0.3

    def test_threshold_boundaries(self):
        """Test confidence thresholds at boundary values."""
        config = LearningConfig(
            high_confidence_threshold=1.0,
            min_confidence_threshold=0.0,
        )
        assert config.high_confidence_threshold == 1.0
        assert config.min_confidence_threshold == 0.0

    def test_disabled_learning(self):
        config = LearningConfig(enabled=False)
        assert config.enabled is False


class TestCostLimitConfig:
    """Tests for CostLimitConfig validation and edge cases."""

    def test_defaults(self):
        config = CostLimitConfig()
        assert config.enabled is False
        assert config.max_cost_per_sheet is None
        assert config.max_cost_per_job is None

    def test_enabled_with_limits(self):
        config = CostLimitConfig(
            enabled=True,
            max_cost_per_sheet=1.0,
            max_cost_per_job=10.0,
        )
        assert config.enabled is True
        assert config.max_cost_per_sheet == 1.0
        assert config.max_cost_per_job == 10.0

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            CostLimitConfig(max_cost_per_sheet=-1.0)

    def test_zero_cost_rejected(self):
        with pytest.raises(ValidationError):
            CostLimitConfig(max_cost_per_sheet=0.0)


class TestRateLimitConfig:
    """Tests for RateLimitConfig model."""

    def test_defaults(self):
        config = RateLimitConfig()
        assert config.wait_minutes == 60
        assert config.max_waits == 24

    def test_custom_values(self):
        config = RateLimitConfig(wait_minutes=2, max_waits=3)
        assert config.wait_minutes == 2
        assert config.max_waits == 3

    def test_negative_waits_rejected(self):
        with pytest.raises(ValidationError):
            RateLimitConfig(max_waits=-1)


class TestJobConfigEdgeCases:
    """Edge case tests for JobConfig — nested validation, type coercion, missing fields."""

    def test_minimal_config_from_yaml_dict(self):
        """Minimal YAML-like dict should produce a valid config."""
        config = JobConfig.model_validate({
            "name": "minimal",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 10},
            "prompt": {"template": "{{ sheet_num }}"},
        })
        assert config.name == "minimal"
        assert config.sheet.total_sheets == 2

    def test_missing_name_raises(self):
        with pytest.raises(ValidationError):
            JobConfig.model_validate({
                "backend": {"type": "claude_cli"},
                "sheet": {"size": 5, "total_items": 10},
                "prompt": {"template": "x"},
            })

    def test_string_workspace_coerces_to_path(self):
        """String workspace from YAML should coerce to Path."""
        config = JobConfig.model_validate({
            "name": "test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 10},
            "prompt": {"template": "x"},
            "workspace": "/tmp/test-workspace",
        })
        assert isinstance(config.workspace, Path)
        assert str(config.workspace) == "/tmp/test-workspace"

    def test_nested_validation_rule_types(self):
        """ValidationRules with different types should all parse."""
        config = JobConfig.model_validate({
            "name": "test",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 5, "total_items": 10},
            "prompt": {"template": "x"},
            "validations": [
                {"type": "file_exists", "path": "{workspace}/out.txt", "description": "check"},
                {"type": "command_succeeds", "command": "echo ok", "description": "run"},
            ],
        })
        assert len(config.validations) == 2
        assert config.validations[0].type == "file_exists"
        assert config.validations[1].type == "command_succeeds"

    def test_model_dump_roundtrip(self):
        """Config should survive model_dump -> model_validate roundtrip."""
        original = JobConfig.model_validate({
            "name": "roundtrip",
            "backend": {"type": "claude_cli"},
            "sheet": {"size": 10, "total_items": 50},
            "prompt": {"template": "{{ sheet_num }}"},
            "retry": {"max_retries": 5},
        })
        dumped = original.model_dump()
        restored = JobConfig.model_validate(dumped)
        assert restored.name == original.name
        assert restored.retry.max_retries == 5
        assert restored.sheet.total_sheets == original.sheet.total_sheets
