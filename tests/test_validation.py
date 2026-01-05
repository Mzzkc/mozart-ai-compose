"""Tests for mozart.execution.validation module."""

from pathlib import Path

import pytest

from mozart.core.config import ValidationRule
from mozart.execution.validation import (
    SheetValidationResult,
    ValidationEngine,
    ValidationResult,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result(self):
        """Test creating a validation result."""
        rule = ValidationRule(
            type="file_exists",
            path="/test/path.txt",
            description="Test file",
        )
        result = ValidationResult(
            rule=rule,
            passed=True,
            actual_value="/test/path.txt",
            expected_value="/test/path.txt",
        )
        assert result.passed is True
        assert result.confidence == 1.0  # default

    def test_result_with_confidence(self):
        """Test validation result with custom confidence."""
        rule = ValidationRule(type="file_exists", path="/test.txt")
        result = ValidationResult(
            rule=rule,
            passed=True,
            actual_value="exists",
            expected_value="exists",
            confidence=0.8,
            confidence_factors={"file_age": 0.8},
        )
        assert result.confidence == 0.8
        assert result.confidence_factors["file_age"] == 0.8

    def test_to_dict(self):
        """Test converting result to dictionary."""
        rule = ValidationRule(
            type="content_contains",
            path="/test.txt",
            pattern="SUCCESS",
            description="Success marker",
        )
        result = ValidationResult(
            rule=rule,
            passed=True,
            actual_value="contains=True",
            expected_value="SUCCESS",
        )
        data = result.to_dict()
        assert data["rule_type"] == "content_contains"
        assert data["passed"] is True
        assert data["pattern"] == "SUCCESS"


class TestSheetValidationResult:
    """Tests for SheetValidationResult dataclass."""

    def _make_result(
        self, passed: bool, confidence: float = 1.0
    ) -> ValidationResult:
        """Helper to create a ValidationResult."""
        rule = ValidationRule(type="file_exists", path="/test.txt")
        return ValidationResult(
            rule=rule,
            passed=passed,
            actual_value="x" if passed else None,
            expected_value="x",
            confidence=confidence,
        )

    def test_all_passed(self):
        """Test all_passed property."""
        result1 = self._make_result(passed=True)
        result2 = self._make_result(passed=True)
        sheet_result = SheetValidationResult(sheet_num=1, results=[result1, result2])
        assert sheet_result.all_passed is True

    def test_all_passed_false(self):
        """Test all_passed is False when any fails."""
        result1 = self._make_result(passed=True)
        result2 = self._make_result(passed=False)
        sheet_result = SheetValidationResult(sheet_num=1, results=[result1, result2])
        assert sheet_result.all_passed is False

    def test_pass_percentage(self):
        """Test pass_percentage calculation."""
        passed = self._make_result(passed=True)
        failed = self._make_result(passed=False)
        sheet_result = SheetValidationResult(
            sheet_num=1, results=[passed, passed, failed]
        )
        assert sheet_result.pass_percentage == pytest.approx(66.67, rel=0.01)

    def test_aggregate_confidence(self):
        """Test aggregate confidence calculation."""
        # High confidence pass
        result1 = self._make_result(passed=True, confidence=0.9)
        # Lower confidence pass
        result2 = self._make_result(passed=True, confidence=0.7)
        sheet_result = SheetValidationResult(sheet_num=1, results=[result1, result2])
        # Both passed, so aggregate is weighted average
        assert sheet_result.aggregate_confidence == pytest.approx(0.8, rel=0.01)

    def test_get_passed_results(self):
        """Test getting only passed results."""
        passed = self._make_result(passed=True)
        failed = self._make_result(passed=False)
        sheet_result = SheetValidationResult(
            sheet_num=1, results=[passed, failed, passed]
        )
        assert len(sheet_result.get_passed_results()) == 2

    def test_get_failed_results(self):
        """Test getting only failed results."""
        passed = self._make_result(passed=True)
        failed = self._make_result(passed=False)
        sheet_result = SheetValidationResult(sheet_num=1, results=[passed, failed])
        assert len(sheet_result.get_failed_results()) == 1

    def test_passed_count(self):
        """Test passed_count property."""
        sheet_result = SheetValidationResult(
            sheet_num=1,
            results=[
                self._make_result(passed=True),
                self._make_result(passed=True),
                self._make_result(passed=False),
            ],
        )
        assert sheet_result.passed_count == 2

    def test_failed_count(self):
        """Test failed_count property."""
        sheet_result = SheetValidationResult(
            sheet_num=1,
            results=[
                self._make_result(passed=True),
                self._make_result(passed=False),
                self._make_result(passed=False),
            ],
        )
        assert sheet_result.failed_count == 2


class TestValidationEngine:
    """Tests for ValidationEngine."""

    def test_file_exists_pass(self, temp_workspace: Path):
        """Test file_exists validation passes when file exists."""
        # Create test file
        test_file = temp_workspace / "output.txt"
        test_file.write_text("test content")

        rule = ValidationRule(
            type="file_exists",
            path=str(test_file),
            description="Output file",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is True

    def test_file_exists_fail(self, temp_workspace: Path):
        """Test file_exists validation fails when file missing."""
        rule = ValidationRule(
            type="file_exists",
            path=str(temp_workspace / "nonexistent.txt"),
            description="Missing file",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is False

    def test_content_contains_pass(self, temp_workspace: Path):
        """Test content_contains validation passes when pattern found."""
        test_file = temp_workspace / "log.txt"
        test_file.write_text("Operation completed: SUCCESS")

        rule = ValidationRule(
            type="content_contains",
            path=str(test_file),
            pattern="SUCCESS",
            description="Success marker",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is True

    def test_content_contains_fail(self, temp_workspace: Path):
        """Test content_contains validation fails when pattern not found."""
        test_file = temp_workspace / "log.txt"
        test_file.write_text("Operation failed: ERROR")

        rule = ValidationRule(
            type="content_contains",
            path=str(test_file),
            pattern="SUCCESS",
            description="Success marker",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is False

    def test_path_template_expansion(self, temp_workspace: Path):
        """Test path templates are expanded correctly."""
        test_file = temp_workspace / "sheet-1-output.txt"
        test_file.write_text("content")

        rule = ValidationRule(
            type="file_exists",
            path="{workspace}/sheet-{sheet_num}-output.txt",
            description="Sheet output",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is True

    def test_run_validations(self, temp_workspace: Path):
        """Test validating multiple rules."""
        # Create test files
        (temp_workspace / "file1.txt").write_text("SUCCESS")
        (temp_workspace / "file2.txt").write_text("DONE")

        rules = [
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "file1.txt"),
            ),
            ValidationRule(
                type="content_contains",
                path=str(temp_workspace / "file1.txt"),
                pattern="SUCCESS",
            ),
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "nonexistent.txt"),
            ),
        ]
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations(rules)

        assert len(sheet_result.results) == 3
        assert len(sheet_result.get_passed_results()) == 2
        assert len(sheet_result.get_failed_results()) == 1
        assert sheet_result.pass_percentage == pytest.approx(66.67, rel=0.01)

    def test_command_succeeds_pass(self, temp_workspace: Path):
        """Test command_succeeds validation passes when command succeeds."""
        rule = ValidationRule(
            type="command_succeeds",
            command="echo 'hello'",
            description="Echo command",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is True
        assert sheet_result.results[0].actual_value == "exit_code=0"

    def test_command_succeeds_fail(self, temp_workspace: Path):
        """Test command_succeeds validation fails when command fails."""
        rule = ValidationRule(
            type="command_succeeds",
            command="exit 1",
            description="Failing command",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is False
        assert sheet_result.results[0].actual_value == "exit_code=1"

    def test_command_succeeds_with_working_directory(self, temp_workspace: Path):
        """Test command_succeeds uses specified working directory."""
        # Create a subdirectory
        subdir = temp_workspace / "subdir"
        subdir.mkdir()
        (subdir / "marker.txt").write_text("found")

        rule = ValidationRule(
            type="command_succeeds",
            command="cat marker.txt",
            working_directory=str(subdir),
            description="Cat in subdir",
        )
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        sheet_result = engine.run_validations([rule])
        assert sheet_result.results[0].passed is True


class TestStagedValidation:
    """Tests for staged validation with fail-fast behavior."""

    def test_staged_validation_all_pass(self, temp_workspace: Path):
        """Test staged validation when all stages pass."""
        # Create test files
        (temp_workspace / "file1.txt").write_text("content1")
        (temp_workspace / "file2.txt").write_text("content2")

        rules = [
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "file1.txt"),
                stage=1,
                description="Stage 1 check",
            ),
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "file2.txt"),
                stage=2,
                description="Stage 2 check",
            ),
        ]
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        result, failed_stage = engine.run_staged_validations(rules)

        assert failed_stage is None
        assert result.all_passed is True
        assert len(result.results) == 2

    def test_staged_validation_fail_fast(self, temp_workspace: Path):
        """Test that failure in stage 1 skips stage 2."""
        # Create only file2, not file1
        (temp_workspace / "file2.txt").write_text("content2")

        rules = [
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "missing.txt"),
                stage=1,
                description="Stage 1 - missing",
            ),
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "file2.txt"),
                stage=2,
                description="Stage 2 - exists",
            ),
        ]
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        result, failed_stage = engine.run_staged_validations(rules)

        assert failed_stage == 1
        assert result.all_passed is False
        assert len(result.results) == 2
        # Stage 1 failed
        assert result.results[0].passed is False
        # Stage 2 was skipped
        assert result.results[1].passed is False
        assert result.results[1].failure_category == "skipped"

    def test_staged_validation_multiple_in_same_stage(self, temp_workspace: Path):
        """Test multiple validations in the same stage."""
        (temp_workspace / "file1.txt").write_text("content1")
        # file2 doesn't exist

        rules = [
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "file1.txt"),
                stage=1,
                description="Stage 1 - exists",
            ),
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "missing.txt"),
                stage=1,
                description="Stage 1 - missing",
            ),
            ValidationRule(
                type="command_succeeds",
                command="echo 'stage 2'",
                stage=2,
                description="Stage 2 - command",
            ),
        ]
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        result, failed_stage = engine.run_staged_validations(rules)

        assert failed_stage == 1
        assert len(result.results) == 3
        # First in stage 1 passed
        assert result.results[0].passed is True
        # Second in stage 1 failed
        assert result.results[1].passed is False
        # Stage 2 was skipped
        assert result.results[2].failure_category == "skipped"

    def test_staged_validation_empty_rules(self, temp_workspace: Path):
        """Test staged validation with empty rules."""
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        result, failed_stage = engine.run_staged_validations([])

        assert failed_stage is None
        assert len(result.results) == 0

    def test_staged_validation_default_stage(self, temp_workspace: Path):
        """Test that rules without explicit stage default to stage 1."""
        (temp_workspace / "file.txt").write_text("content")

        rule = ValidationRule(
            type="file_exists",
            path=str(temp_workspace / "file.txt"),
            description="Default stage check",
            # Note: no stage specified, should default to 1
        )
        assert rule.stage == 1

    def test_staged_validation_non_sequential_stages(self, temp_workspace: Path):
        """Test staged validation with non-sequential stage numbers."""
        (temp_workspace / "file.txt").write_text("content")

        rules = [
            ValidationRule(
                type="file_exists",
                path=str(temp_workspace / "file.txt"),
                stage=5,  # Jump to stage 5
                description="Stage 5 check",
            ),
            ValidationRule(
                type="command_succeeds",
                command="echo hello",
                stage=10,  # Jump to stage 10
                description="Stage 10 check",
            ),
        ]
        engine = ValidationEngine(
            workspace=temp_workspace,
            sheet_context={"sheet_num": 1, "workspace": str(temp_workspace)},
        )
        result, failed_stage = engine.run_staged_validations(rules)

        assert failed_stage is None
        assert result.all_passed is True
