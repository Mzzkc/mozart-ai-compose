"""F-523 regression tests: Schema validation error message clarity.

Root cause: Users trying onboarding hit confusing Pydantic errors that don't
explain the actual structure. "sheets" (plural) is rejected but the error
doesn't explain that the field is "sheet" (singular) with size/total_items.

Tests verify that validate command provides clear, actionable error messages
for common structural mistakes.
"""

import pytest
from pathlib import Path
from marianne.cli.commands.validate import _schema_error_hints


class TestSheetPluralError:
    """Test error messages for sheets (plural) vs sheet (singular) confusion."""

    def test_sheets_plural_gets_specific_hint(self) -> None:
        """sheets (plural) error should explain sheet (singular) structure."""
        # Pydantic v2 format for extra_forbidden:
        error_msg = "sheets\n  Extra inputs are not permitted"

        hints = _schema_error_hints(error_msg)

        # Should provide specific guidance, not generic unknown field message
        assert any("sheet" in h.lower() and "singular" in h.lower() for h in hints), \
            f"Expected hint about singular 'sheet', got: {hints}"
        assert any("size" in h.lower() and "total_items" in h.lower() for h in hints), \
            f"Expected hint about sheet structure (size, total_items), got: {hints}"

    def test_prompts_plural_gets_specific_hint(self) -> None:
        """prompts (plural) error should explain prompt (singular) structure."""
        error_msg = "prompts\n  Extra inputs are not permitted"

        hints = _schema_error_hints(error_msg)

        assert any("prompt" in h.lower() and "singular" in h.lower() for h in hints), \
            f"Expected hint about singular 'prompt', got: {hints}"
        assert any("template" in h.lower() for h in hints), \
            f"Expected hint about prompt.template, got: {hints}"


class TestMovementsStructureError:
    """Test error messages for movements structure confusion."""

    def test_movements_with_list_gets_structure_hint(self) -> None:
        """movements expecting dict, not list."""
        # User tried: movements: [1, 2, 3] or movements: - name: foo
        error_msg = "movements\n  Input should be a valid dictionary"

        hints = _schema_error_hints(error_msg)

        assert any("movement" in h.lower() and "dictionary" in h.lower() for h in hints), \
            f"Expected hint about movements as dict, got: {hints}"
        assert any("1:" in h or "2:" in h for h in hints), \
            f"Expected example with movement numbers as keys, got: {hints}"


class TestMissingRequiredFields:
    """Test error messages for missing required fields."""

    def test_missing_sheet_gets_structure_example(self) -> None:
        """Missing 'sheet' field should provide minimal working example."""
        error_msg = "sheet\n  Field required"

        hints = _schema_error_hints(error_msg)

        # Should show what a minimal sheet looks like
        assert any("sheet:" in h for h in hints), \
            f"Expected YAML example with 'sheet:', got: {hints}"
        assert any("size" in h.lower() or "total_items" in h.lower() for h in hints), \
            f"Expected hint about required sheet fields, got: {hints}"

    def test_missing_prompt_gets_structure_example(self) -> None:
        """Missing 'prompt' field should provide minimal working example."""
        error_msg = "prompt\n  Field required"

        hints = _schema_error_hints(error_msg)

        assert any("prompt:" in h for h in hints), \
            f"Expected YAML example with 'prompt:', got: {hints}"
        assert any("template" in h.lower() for h in hints), \
            f"Expected hint about prompt.template, got: {hints}"


class TestMultipleErrorsInOneMessage:
    """Test error messages when multiple validation errors occur."""

    def test_multiple_extra_forbidden_errors(self) -> None:
        """Multiple unknown fields should each get suggestions."""
        error_msg = """2 validation errors for JobConfig
sheets
  Extra inputs are not permitted
prompts
  Extra inputs are not permitted"""

        hints = _schema_error_hints(error_msg)

        # Should address both errors
        assert any("sheets" in h.lower() for h in hints), \
            f"Expected hint about 'sheets', got: {hints}"
        assert any("prompts" in h.lower() for h in hints), \
            f"Expected hint about 'prompts', got: {hints}"

    def test_extra_forbidden_plus_field_required(self) -> None:
        """Extra field + missing field should provide both fixes."""
        error_msg = """2 validation errors for JobConfig
sheets
  Extra inputs are not permitted
prompt
  Field required"""

        hints = _schema_error_hints(error_msg)

        # Should explain both the wrong field and the missing field
        assert len(hints) >= 2, f"Expected multiple hints, got: {hints}"
        assert any("sheet" in h.lower() for h in hints), \
            f"Expected hint about correct field name, got: {hints}"
        assert any("prompt" in h.lower() and "required" in h.lower() for h in hints), \
            f"Expected hint about missing prompt field, got: {hints}"


class TestRealWorldOnboardingScenarios:
    """Test error messages for actual scenarios from F-523."""

    def test_list_of_sheets_attempt(self) -> None:
        """User tried: sheets: - prompt: ..."""
        # This produces two errors: sheets (extra) and sheet (missing)
        error_msg = """2 validation errors for JobConfig
sheets
  Extra inputs are not permitted
sheet
  Field required"""

        hints = _schema_error_hints(error_msg)

        # Should be clear that sheets (plural) is wrong and sheet (singular) is needed
        assert any("sheet" in h.lower() and "singular" in h.lower() for h in hints) or \
               any("sheets" in h.lower() and "sheet:" in h.lower() for h in hints), \
            f"Expected clear guidance from sheets→sheet, got: {hints}"

        # Should show what a correct sheet: block looks like
        assert any("size" in h.lower() for h in hints), \
            f"Expected example showing sheet structure, got: {hints}"
