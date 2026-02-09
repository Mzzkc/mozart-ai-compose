"""Tests for the self-healing module.

Tests cover:
- ErrorContext creation (including from_execution_result)
- Remedy base classes and protocol
- Individual remedy implementations (all 6 built-in remedies)
- RemedyRegistry functionality
- DiagnosisEngine
- SelfHealingCoordinator
- Diagnosis formatting
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mozart.healing.context import ErrorContext
from mozart.healing.coordinator import HealingReport, SelfHealingCoordinator
from mozart.healing.diagnosis import Diagnosis, DiagnosisEngine
from mozart.healing.registry import RemedyRegistry, create_default_registry
from mozart.healing.remedies.base import BaseRemedy, RemedyCategory, RemedyResult, RiskLevel
from mozart.healing.remedies.diagnostics import DiagnoseAuthErrorRemedy, DiagnoseMissingCLIRemedy
from mozart.healing.remedies.jinja import SuggestJinjaFixRemedy
from mozart.healing.remedies.paths import (
    CreateMissingParentDirsRemedy,
    CreateMissingWorkspaceRemedy,
    FixPathSeparatorsRemedy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Create a mock JobConfig."""
    config = MagicMock()
    config.workspace = Path("/tmp/test-workspace")
    config.backend.working_directory = None
    return config


@pytest.fixture
def basic_error_context(mock_config):
    """Create a basic ErrorContext for testing."""
    return ErrorContext(
        error_code="E601",
        error_message="Workspace directory does not exist: /tmp/test-workspace",
        error_category="preflight",
        config=mock_config,
        workspace=Path("/tmp/test-workspace"),
        sheet_number=1,
        working_directory=Path("/tmp/test-workspace"),
    )


@pytest.fixture
def registry():
    """Create an empty registry."""
    return RemedyRegistry()


@pytest.fixture
def default_registry():
    """Create the default registry with all built-in remedies."""
    return create_default_registry()


# =============================================================================
# ErrorContext Tests
# =============================================================================


class TestErrorContext:
    """Tests for ErrorContext creation and methods."""

    def test_basic_creation(self, mock_config):
        """Test basic ErrorContext creation."""
        ctx = ErrorContext(
            error_code="E601",
            error_message="Test error",
            error_category="preflight",
        )
        assert ctx.error_code == "E601"
        assert ctx.error_message == "Test error"
        assert ctx.error_category == "preflight"

    def test_from_preflight_error(self, mock_config):
        """Test creating context from preflight error."""
        ctx = ErrorContext.from_preflight_error(
            config=mock_config,
            config_path=Path("/tmp/config.yaml"),
            error_code="E601",
            error_message="Workspace missing",
            sheet_number=1,
        )
        assert ctx.error_code == "E601"
        assert ctx.error_category == "preflight"
        assert ctx.sheet_number == 1
        assert ctx.workspace == mock_config.workspace

    def test_get_context_summary(self, basic_error_context):
        """Test context summary generation."""
        summary = basic_error_context.get_context_summary()
        assert summary["error_code"] == "E601"
        assert summary["sheet_number"] == 1
        assert summary["has_stdout"] is False


# =============================================================================
# RemedyRegistry Tests
# =============================================================================


class TestRemedyRegistry:
    """Tests for RemedyRegistry."""

    def test_empty_registry(self, registry):
        """Test empty registry."""
        assert registry.count() == 0
        assert registry.all_remedies() == []

    def test_register_remedy(self, registry):
        """Test registering a remedy."""
        remedy = CreateMissingWorkspaceRemedy()
        registry.register(remedy)
        assert registry.count() == 1

    def test_get_by_name(self, registry):
        """Test getting remedy by name."""
        remedy = CreateMissingWorkspaceRemedy()
        registry.register(remedy)

        found = registry.get_by_name("create_missing_workspace")
        assert found is not None
        assert found.name == "create_missing_workspace"

        not_found = registry.get_by_name("nonexistent")
        assert not_found is None

    def test_find_applicable(self, default_registry, basic_error_context):
        """Test finding applicable remedies."""
        applicable = default_registry.find_applicable(basic_error_context)

        # Should find at least the workspace remedy
        assert len(applicable) > 0

        # Results should be sorted by confidence
        confidences = [d.confidence for _, d in applicable]
        assert confidences == sorted(confidences, reverse=True)

    def test_default_registry_has_all_remedies(self, default_registry):
        """Test that default registry has all built-in remedies."""
        names = {r.name for r in default_registry.all_remedies()}

        expected = {
            "create_missing_workspace",
            "create_missing_parent_dirs",
            "fix_path_separators",
            "suggest_jinja_fix",
            "diagnose_auth_error",
            "diagnose_missing_cli",
        }
        assert expected == names


# =============================================================================
# DiagnosisEngine Tests
# =============================================================================


class TestDiagnosisEngine:
    """Tests for DiagnosisEngine."""

    def test_diagnose_empty_registry(self, registry):
        """Test diagnosis with empty registry."""
        engine = DiagnosisEngine(registry)
        ctx = ErrorContext(
            error_code="E601",
            error_message="Test",
            error_category="test",
        )
        diagnoses = engine.diagnose(ctx)
        assert diagnoses == []

    def test_diagnose_finds_applicable(self, default_registry, basic_error_context):
        """Test diagnosis finds applicable remedies."""
        engine = DiagnosisEngine(default_registry)
        diagnoses = engine.diagnose(basic_error_context)

        assert len(diagnoses) > 0
        assert all(isinstance(d, Diagnosis) for d in diagnoses)

    def test_diagnose_sorts_by_confidence(self, default_registry, basic_error_context):
        """Test diagnoses are sorted by confidence."""
        engine = DiagnosisEngine(default_registry)
        diagnoses = engine.diagnose(basic_error_context)

        if len(diagnoses) > 1:
            confidences = [d.confidence for d in diagnoses]
            assert confidences == sorted(confidences, reverse=True)

    def test_get_primary_diagnosis(self, default_registry, basic_error_context):
        """Test getting primary (highest confidence) diagnosis."""
        engine = DiagnosisEngine(default_registry)
        primary = engine.get_primary_diagnosis(basic_error_context)

        assert primary is not None
        assert primary.error_code == basic_error_context.error_code


# =============================================================================
# CreateMissingWorkspaceRemedy Tests
# =============================================================================


class TestCreateMissingWorkspaceRemedy:
    """Tests for CreateMissingWorkspaceRemedy."""

    def test_properties(self):
        """Test remedy properties."""
        remedy = CreateMissingWorkspaceRemedy()
        assert remedy.name == "create_missing_workspace"
        assert remedy.category == RemedyCategory.AUTOMATIC
        assert remedy.risk_level == RiskLevel.LOW

    def test_diagnoses_missing_workspace(self, tmp_path, mock_config):
        """Test diagnosis when workspace doesn't exist."""
        workspace = tmp_path / "missing-workspace"
        mock_config.workspace = workspace

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace directory does not exist: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        remedy = CreateMissingWorkspaceRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert diagnosis.confidence >= 0.9
        assert diagnosis.remedy_name == "create_missing_workspace"

    def test_does_not_diagnose_existing_workspace(self, tmp_path, mock_config):
        """Test no diagnosis when workspace exists."""
        workspace = tmp_path / "existing"
        workspace.mkdir()
        mock_config.workspace = workspace

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace directory does not exist: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        remedy = CreateMissingWorkspaceRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is None

    def test_apply_creates_directory(self, tmp_path, mock_config):
        """Test that apply() creates the workspace directory."""
        workspace = tmp_path / "new-workspace"
        mock_config.workspace = workspace

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace missing: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        remedy = CreateMissingWorkspaceRemedy()
        result = remedy.apply(ctx)

        assert result.success
        assert workspace.exists()
        assert workspace in result.created_paths

    def test_rollback_removes_empty_directory(self, tmp_path, mock_config):
        """Test rollback removes empty directory."""
        workspace = tmp_path / "rollback-test"
        mock_config.workspace = workspace

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace missing: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        remedy = CreateMissingWorkspaceRemedy()
        result = remedy.apply(ctx)
        assert workspace.exists()

        # Rollback
        rolled_back = remedy.rollback(result)
        assert rolled_back, "Rollback should succeed for created workspace"
        assert not workspace.exists()


# =============================================================================
# SuggestJinjaFixRemedy Tests
# =============================================================================


class TestSuggestJinjaFixRemedy:
    """Tests for SuggestJinjaFixRemedy."""

    def test_properties(self):
        """Test remedy properties."""
        remedy = SuggestJinjaFixRemedy()
        assert remedy.name == "suggest_jinja_fix"
        assert remedy.category == RemedyCategory.SUGGESTED
        assert remedy.risk_level == RiskLevel.MEDIUM

    def test_diagnoses_undefined_variable(self, mock_config):
        """Test diagnosis of undefined variable error."""
        ctx = ErrorContext(
            error_code="E304",
            error_message="'shee_num' is undefined",
            error_category="template",
            config=mock_config,
        )

        remedy = SuggestJinjaFixRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert "shee_num" in diagnosis.issue
        # Should suggest 'sheet_num' as correction
        assert "sheet_num" in diagnosis.suggestion

    def test_diagnoses_unclosed_block(self, mock_config):
        """Test diagnosis of unclosed Jinja block."""
        ctx = ErrorContext(
            error_code="E304",
            error_message="Unexpected end of template",
            error_category="template",
            config=mock_config,
        )

        remedy = SuggestJinjaFixRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert "unclosed" in diagnosis.issue.lower() or "end" in diagnosis.suggestion.lower()


# =============================================================================
# DiagnoseAuthErrorRemedy Tests
# =============================================================================


class TestDiagnoseAuthErrorRemedy:
    """Tests for DiagnoseAuthErrorRemedy."""

    def test_properties(self):
        """Test remedy properties."""
        remedy = DiagnoseAuthErrorRemedy()
        assert remedy.name == "diagnose_auth_error"
        assert remedy.category == RemedyCategory.DIAGNOSTIC
        assert remedy.risk_level == RiskLevel.LOW

    def test_diagnoses_rate_limit(self, mock_config):
        """Test diagnosis of rate limit error."""
        ctx = ErrorContext(
            error_code="E101",
            error_message="Rate limit exceeded",
            error_category="rate_limit",
            config=mock_config,
        )

        remedy = DiagnoseAuthErrorRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert "rate" in diagnosis.issue.lower()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=True)
    def test_diagnoses_missing_api_key(self, mock_config):
        """Test diagnosis when API key is missing."""
        ctx = ErrorContext(
            error_code="E102",
            error_message="Authentication failed",
            error_category="auth",
            config=mock_config,
        )

        remedy = DiagnoseAuthErrorRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert "key" in diagnosis.issue.lower() or "key" in diagnosis.suggestion.lower()

    def test_generate_diagnostic(self, mock_config):
        """Test diagnostic message generation."""
        ctx = ErrorContext(
            error_code="E101",
            error_message="Rate limit exceeded",
            error_category="rate_limit",
            config=mock_config,
        )

        remedy = DiagnoseAuthErrorRemedy()
        diagnostic = remedy.generate_diagnostic(ctx)

        assert "rate limit" in diagnostic.lower()
        assert len(diagnostic) > 100  # Should have substantial guidance


# =============================================================================
# SelfHealingCoordinator Tests
# =============================================================================


class TestSelfHealingCoordinator:
    """Tests for SelfHealingCoordinator."""

    @pytest.fixture
    def coordinator(self, default_registry):
        """Create a coordinator with default registry."""
        return SelfHealingCoordinator(default_registry)

    @pytest.mark.asyncio
    async def test_heal_applies_automatic_remedies(self, coordinator, tmp_path, mock_config):
        """Test that automatic remedies are applied without prompting."""
        workspace = tmp_path / "heal-test"
        mock_config.workspace = workspace

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace directory does not exist: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        report = await coordinator.heal(ctx)

        # Should have applied the workspace creation remedy
        assert report.any_remedies_applied
        assert workspace.exists()

    @pytest.mark.asyncio
    async def test_heal_dry_run(self, default_registry, tmp_path, mock_config):
        """Test dry run mode doesn't make changes."""
        workspace = tmp_path / "dry-run-test"
        mock_config.workspace = workspace

        coordinator = SelfHealingCoordinator(
            default_registry,
            dry_run=True,
        )

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace directory does not exist: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        report = await coordinator.heal(ctx)

        # Should have skipped the remedy (dry run)
        assert not report.any_remedies_applied
        assert not workspace.exists()
        # May or may not have skipped remedies depending on diagnosis
        # The important thing is that no changes were made

    @pytest.mark.asyncio
    async def test_heal_disabled_remedies(self, default_registry, tmp_path, mock_config):
        """Test that disabled remedies are skipped."""
        workspace = tmp_path / "disabled-test"
        mock_config.workspace = workspace

        coordinator = SelfHealingCoordinator(
            default_registry,
            disabled_remedies={"create_missing_workspace"},
        )

        ctx = ErrorContext(
            error_code="E601",
            error_message=f"Workspace directory does not exist: {workspace}",
            error_category="preflight",
            workspace=workspace,
            config=mock_config,
        )

        report = await coordinator.heal(ctx)

        # Workspace should not be created (remedy is disabled)
        assert not workspace.exists()
        # If the remedy was diagnosed, it should be in skipped list
        if report.diagnoses:
            skipped_names = [name for name, _ in report.actions_skipped]
            # Might also be skipped for other reasons or have other remedies
            assert not report.any_remedies_applied

    @pytest.mark.asyncio
    async def test_max_healing_attempts(self, default_registry, mock_config):
        """Test that healing stops after max attempts."""
        coordinator = SelfHealingCoordinator(
            default_registry,
            max_healing_attempts=2,
        )

        ctx = ErrorContext(
            error_code="E999",
            error_message="Unknown error",
            error_category="unknown",
            config=mock_config,
        )

        # First two attempts should work (even if no remedies apply)
        await coordinator.heal(ctx)
        await coordinator.heal(ctx)

        # Third attempt should be blocked
        report = await coordinator.heal(ctx)
        assert "Max healing attempts" in str(report.actions_skipped)


# =============================================================================
# HealingReport Tests
# =============================================================================


class TestHealingReport:
    """Tests for HealingReport."""

    def test_any_remedies_applied(self, basic_error_context):
        """Test any_remedies_applied property."""
        report = HealingReport(error_context=basic_error_context)
        assert not report.any_remedies_applied

        report.actions_taken.append((
            "test",
            RemedyResult(success=True, message="OK", action_taken="test"),
        ))
        assert report.any_remedies_applied

    def test_should_retry(self, basic_error_context):
        """Test should_retry property."""
        report = HealingReport(error_context=basic_error_context)
        assert not report.should_retry

        report.actions_taken.append((
            "test",
            RemedyResult(success=True, message="OK", action_taken="test"),
        ))
        assert report.should_retry

    def test_format(self, basic_error_context):
        """Test report formatting."""
        report = HealingReport(error_context=basic_error_context)
        formatted = report.format()

        assert "SELF-HEALING REPORT" in formatted
        assert basic_error_context.error_code in formatted


# =============================================================================
# Integration Tests
# =============================================================================


class TestHealingIntegration:
    """Integration tests for the healing system."""

    @pytest.mark.asyncio
    async def test_full_healing_flow(self, tmp_path):
        """Test complete healing flow from error to fix."""
        # Setup
        workspace = tmp_path / "integration-test"
        mock_config = MagicMock()
        mock_config.workspace = workspace
        mock_config.backend.working_directory = None

        registry = create_default_registry()
        coordinator = SelfHealingCoordinator(registry)

        # Create error context
        ctx = ErrorContext.from_preflight_error(
            config=mock_config,
            config_path=None,
            error_code="E601",
            error_message=f"Workspace directory does not exist: {workspace}",
        )

        # Heal
        report = await coordinator.heal(ctx)

        # Verify
        assert report.should_retry
        assert workspace.exists()
        assert len(report.diagnoses) > 0
        assert report.any_remedies_applied


# =============================================================================
# CreateMissingParentDirsRemedy Tests (NEW - previously untested)
# =============================================================================


class TestCreateMissingParentDirsRemedy:
    """Tests for CreateMissingParentDirsRemedy."""

    def test_properties(self):
        """Test remedy properties."""
        remedy = CreateMissingParentDirsRemedy()
        assert remedy.name == "create_missing_parent_dirs"
        assert remedy.category == RemedyCategory.AUTOMATIC
        assert remedy.risk_level == RiskLevel.LOW

    def test_diagnoses_missing_parent_directory(self, tmp_path, mock_config):
        """Test diagnosis when parent directory is missing."""
        missing = tmp_path / "deep" / "nested" / "output"
        ctx = ErrorContext(
            error_code="E601",
            error_message=f"directory '{missing}' does not exist",
            error_category="preflight",
            config=mock_config,
            workspace=tmp_path,
        )
        remedy = CreateMissingParentDirsRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert diagnosis.confidence == 0.85
        assert diagnosis.remedy_name == "create_missing_parent_dirs"
        assert "paths_to_create" in diagnosis.context

    def test_no_diagnosis_when_directory_exists(self, tmp_path, mock_config):
        """Test no diagnosis when directory already exists."""
        existing = tmp_path / "existing"
        existing.mkdir()
        ctx = ErrorContext(
            error_code="E601",
            error_message=f"directory '{existing}' does not exist",
            error_category="preflight",
            config=mock_config,
            workspace=tmp_path,
        )
        remedy = CreateMissingParentDirsRemedy()
        diagnosis = remedy.diagnose(ctx)
        assert diagnosis is None

    def test_no_diagnosis_for_wrong_error_code(self, mock_config):
        """Test no diagnosis for unrelated error codes."""
        ctx = ErrorContext(
            error_code="E101",
            error_message="directory '/foo/bar' does not exist",
            error_category="rate_limit",
            config=mock_config,
        )
        remedy = CreateMissingParentDirsRemedy()
        assert remedy.diagnose(ctx) is None

    def test_apply_creates_directory_tree(self, tmp_path, mock_config):
        """Test that apply() creates the full directory tree."""
        deep_path = tmp_path / "a" / "b" / "c"
        ctx = ErrorContext(
            error_code="E601",
            error_message=f"directory '{deep_path}' does not exist",
            error_category="preflight",
            config=mock_config,
            workspace=tmp_path,
        )
        remedy = CreateMissingParentDirsRemedy()
        result = remedy.apply(ctx)

        assert result.success
        assert deep_path.exists()
        assert len(result.created_paths) >= 1

    def test_rollback_removes_created_directories(self, tmp_path, mock_config):
        """Test that rollback removes created empty directories."""
        deep_path = tmp_path / "rollback_a" / "rollback_b"
        ctx = ErrorContext(
            error_code="E601",
            error_message=f"directory '{deep_path}' does not exist",
            error_category="preflight",
            config=mock_config,
            workspace=tmp_path,
        )
        remedy = CreateMissingParentDirsRemedy()
        result = remedy.apply(ctx)
        assert deep_path.exists()

        rolled_back = remedy.rollback(result)
        assert rolled_back

    def test_preview_returns_meaningful_text(self, tmp_path, mock_config):
        """Test that preview provides useful info."""
        missing = tmp_path / "preview_dir" / "sub"
        ctx = ErrorContext(
            error_code="E601",
            error_message=f"directory '{missing}' does not exist",
            error_category="preflight",
            config=mock_config,
            workspace=tmp_path,
        )
        remedy = CreateMissingParentDirsRemedy()
        preview = remedy.preview(ctx)
        assert "Create" in preview or "directories" in preview


# =============================================================================
# FixPathSeparatorsRemedy Tests (NEW - previously untested)
# =============================================================================


class TestFixPathSeparatorsRemedy:
    """Tests for FixPathSeparatorsRemedy."""

    def test_properties(self):
        """Test remedy properties."""
        remedy = FixPathSeparatorsRemedy()
        assert remedy.name == "fix_path_separators"
        assert remedy.category == RemedyCategory.AUTOMATIC

    @patch("sys.platform", "linux")
    def test_diagnoses_backslash_paths_on_unix(self, mock_config):
        """Test diagnosis of Windows paths on Unix systems."""
        ctx = ErrorContext(
            error_code="E601",
            error_message=r"File not found: C:\Users\test\output.txt",
            error_category="preflight",
            config=mock_config,
        )
        remedy = FixPathSeparatorsRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert diagnosis.confidence == 0.90
        assert "fixed_path" in diagnosis.context
        assert "/" in diagnosis.context["fixed_path"]

    @patch("sys.platform", "win32")
    def test_no_diagnosis_on_windows(self, mock_config):
        """Test no diagnosis when running on Windows."""
        ctx = ErrorContext(
            error_code="E601",
            error_message=r"File not found: C:\Users\test\output.txt",
            error_category="preflight",
            config=mock_config,
        )
        remedy = FixPathSeparatorsRemedy()
        diagnosis = remedy.diagnose(ctx)
        assert diagnosis is None

    @patch("sys.platform", "linux")
    def test_no_diagnosis_for_unix_paths(self, mock_config):
        """Test no diagnosis when paths are already Unix-style."""
        ctx = ErrorContext(
            error_code="E601",
            error_message="File not found: /home/test/output.txt",
            error_category="preflight",
            config=mock_config,
        )
        remedy = FixPathSeparatorsRemedy()
        diagnosis = remedy.diagnose(ctx)
        assert diagnosis is None

    @patch("sys.platform", "linux")
    def test_apply_returns_informational_result(self, mock_config):
        """Test that apply provides information (no file changes)."""
        ctx = ErrorContext(
            error_code="E601",
            error_message=r"File not found: C:\workspace\output.txt",
            error_category="preflight",
            config=mock_config,
        )
        remedy = FixPathSeparatorsRemedy()
        result = remedy.apply(ctx)
        assert result.success
        assert "diagnosis provided" in result.action_taken

    @patch("sys.platform", "linux")
    def test_generate_diagnostic(self, mock_config):
        """Test diagnostic message for path separator issues."""
        ctx = ErrorContext(
            error_code="E601",
            error_message=r"File not found: C:\workspace\output.txt",
            error_category="preflight",
            config=mock_config,
        )
        remedy = FixPathSeparatorsRemedy()
        diagnostic = remedy.generate_diagnostic(ctx)
        assert "Windows-style" in diagnostic or "backslash" in diagnostic.lower()


# =============================================================================
# DiagnoseMissingCLIRemedy Tests (NEW - previously untested)
# =============================================================================


class TestDiagnoseMissingCLIRemedy:
    """Tests for DiagnoseMissingCLIRemedy."""

    def test_properties(self):
        """Test remedy properties."""
        remedy = DiagnoseMissingCLIRemedy()
        assert remedy.name == "diagnose_missing_cli"
        assert remedy.category == RemedyCategory.DIAGNOSTIC

    def test_diagnoses_cli_not_found(self, mock_config):
        """Test diagnosis when Claude CLI not found."""
        ctx = ErrorContext(
            error_code="E601",
            error_message="claude: command not found",
            error_category="execution",
            config=mock_config,
        )
        remedy = DiagnoseMissingCLIRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert "claude" in diagnosis.issue.lower() or "cli" in diagnosis.issue.lower()

    def test_no_diagnosis_for_unrelated_error(self, mock_config):
        """Test no diagnosis for unrelated errors."""
        ctx = ErrorContext(
            error_code="E101",
            error_message="Rate limit exceeded",
            error_category="rate_limit",
            config=mock_config,
        )
        remedy = DiagnoseMissingCLIRemedy()
        assert remedy.diagnose(ctx) is None

    @patch("shutil.which", return_value=None)
    def test_cli_not_in_path(self, mock_which, mock_config):
        """Test diagnosis when claude not in PATH."""
        ctx = ErrorContext(
            error_code="E901",
            error_message="claude cli not installed",
            error_category="execution",
            config=mock_config,
        )
        remedy = DiagnoseMissingCLIRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert diagnosis.confidence == 0.95
        assert not diagnosis.context.get("cli_found")

    @patch("shutil.which", return_value="/usr/local/bin/claude")
    def test_cli_found_but_broken(self, mock_which, mock_config):
        """Test diagnosis when claude found but not working."""
        ctx = ErrorContext(
            error_code="E901",
            error_message="claude cli error",
            error_category="execution",
            config=mock_config,
        )
        remedy = DiagnoseMissingCLIRemedy()
        diagnosis = remedy.diagnose(ctx)

        assert diagnosis is not None
        assert diagnosis.confidence == 0.70
        assert diagnosis.context.get("cli_found")

    @patch("shutil.which", return_value=None)
    def test_generate_diagnostic_includes_install_instructions(self, mock_which, mock_config):
        """Test that diagnostic includes installation instructions."""
        ctx = ErrorContext(
            error_code="E901",
            error_message="claude command not found",
            error_category="execution",
            config=mock_config,
        )
        remedy = DiagnoseMissingCLIRemedy()
        diagnostic = remedy.generate_diagnostic(ctx)

        assert "npm" in diagnostic.lower()
        assert "install" in diagnostic.lower()


# =============================================================================
# ErrorContext.from_execution_result Tests (NEW - previously untested)
# =============================================================================


class TestErrorContextFromExecutionResult:
    """Tests for ErrorContext.from_execution_result factory method."""

    def test_creates_context_from_result(self, mock_config):
        """Test creating context from an ExecutionResult."""
        result = MagicMock()
        result.exit_code = 1
        result.exit_signal = None
        result.stdout = "output text"
        result.stderr = "error text"

        ctx = ErrorContext.from_execution_result(
            result=result,
            config=mock_config,
            config_path=Path("/tmp/config.yaml"),
            sheet_number=3,
            error_code="E302",
            error_message="Validation failed",
            error_category="validation",
            retry_count=2,
            max_retries=5,
            previous_errors=["E301"],
        )

        assert ctx.error_code == "E302"
        assert ctx.exit_code == 1
        assert ctx.sheet_number == 3
        assert ctx.retry_count == 2
        assert ctx.max_retries == 5
        assert "E301" in ctx.previous_errors
        assert ctx.stdout_tail == "output text"
        assert ctx.stderr_tail == "error text"

    def test_truncates_long_output(self, mock_config):
        """Test that long stdout/stderr are truncated."""
        result = MagicMock()
        result.exit_code = 1
        result.exit_signal = None
        result.stdout = "x" * 20000
        result.stderr = "y" * 20000

        ctx = ErrorContext.from_execution_result(
            result=result,
            config=mock_config,
            config_path=None,
            sheet_number=1,
            error_code="E302",
            error_message="Test",
            error_category="test",
        )

        # HEALING_CONTEXT_TAIL_CHARS is 10000
        assert len(ctx.stdout_tail) <= 10000
        assert len(ctx.stderr_tail) <= 10000

    def test_handles_empty_output(self, mock_config):
        """Test handling of empty stdout/stderr."""
        result = MagicMock()
        result.exit_code = 0
        result.exit_signal = None
        result.stdout = None
        result.stderr = None

        ctx = ErrorContext.from_execution_result(
            result=result,
            config=mock_config,
            config_path=None,
            sheet_number=1,
            error_code="E000",
            error_message="Test",
            error_category="test",
        )

        assert ctx.stdout_tail == ""
        assert ctx.stderr_tail == ""

    def test_captures_api_key_masked(self, mock_config):
        """Test that API key is masked in environment."""
        result = MagicMock()
        result.exit_code = 0
        result.exit_signal = None
        result.stdout = ""
        result.stderr = ""

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-secret"}):
            ctx = ErrorContext.from_execution_result(
                result=result,
                config=mock_config,
                config_path=None,
                sheet_number=1,
                error_code="E000",
                error_message="Test",
                error_category="test",
            )

        assert ctx.environment.get("ANTHROPIC_API_KEY") == "***"


# =============================================================================
# Diagnosis Formatting Tests (NEW - previously untested)
# =============================================================================


class TestDiagnosisFormatting:
    """Tests for Diagnosis format_short and format_full methods."""

    def _make_diagnosis(self) -> Diagnosis:
        return Diagnosis(
            error_code="E601",
            issue="Workspace missing",
            explanation="The workspace directory does not exist.",
            suggestion="Create it with mkdir.",
            confidence=0.95,
            remedy_name="create_missing_workspace",
            requires_confirmation=False,
            context={},
        )

    def test_format_short(self):
        """Test short format contains key info."""
        d = self._make_diagnosis()
        short = d.format_short()
        assert "E601" in short
        assert "Workspace missing" in short

    def test_format_full(self):
        """Test full format contains all details."""
        d = self._make_diagnosis()
        full = d.format_full()
        assert "E601" in full
        assert "Workspace missing" in full
        assert "explanation" in full.lower() or "workspace directory" in full
        assert "mkdir" in full


# =============================================================================
# Coordinator Edge Case Tests (NEW - previously untested)
# =============================================================================


class TestCoordinatorEdgeCases:
    """Additional edge case tests for SelfHealingCoordinator."""

    @pytest.mark.asyncio
    async def test_reset_clears_attempt_counter(self, default_registry, mock_config):
        """Test that reset() allows healing again after max attempts."""
        coordinator = SelfHealingCoordinator(
            default_registry,
            max_healing_attempts=1,
        )
        ctx = ErrorContext(
            error_code="E999",
            error_message="Unknown error",
            error_category="unknown",
            config=mock_config,
        )

        # First attempt works
        await coordinator.heal(ctx)

        # Second attempt blocked
        report = await coordinator.heal(ctx)
        assert any("Max healing" in reason for _, reason in report.actions_skipped)

        # Reset and try again
        coordinator.reset()
        report = await coordinator.heal(ctx)
        # Should not be blocked anymore
        assert not any("Max healing" in reason for _, reason in report.actions_skipped)

    @pytest.mark.asyncio
    async def test_heal_with_no_applicable_remedies(self, default_registry, mock_config):
        """Test healing when no remedies apply to the error."""
        ctx = ErrorContext(
            error_code="E999",
            error_message="A completely novel error nobody has seen",
            error_category="unknown",
            config=mock_config,
        )
        coordinator = SelfHealingCoordinator(default_registry)
        report = await coordinator.heal(ctx)

        assert not report.any_remedies_applied
        assert not report.should_retry
