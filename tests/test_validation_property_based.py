"""Property-based tests for validation and healing models.

Uses Hypothesis to generate randomized inputs and verify invariants
across the validation/healing subsystems. Each test is marked with
``@pytest.mark.property_based`` for selective execution.

Targets:
1. ValidationRule: round-trip through model_dump / model_validate
2. Validation conditions: any valid expression evaluates without crash
3. Healing remedy configs: all remedy types serialize/deserialize correctly
4. Reporter output: any issue set produces valid report output
5. Check results: validation check pass/fail is deterministic for same inputs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from tests.conftest_adversarial import (
    _short_text,
    _unit_float,
    validation_rule_strategy,
)

# ---------------------------------------------------------------------------
# Local strategies
# ---------------------------------------------------------------------------

_severity_strategy = st.sampled_from(["error", "warning", "info"])

_condition_strategy = st.one_of(
    st.none(),
    st.builds(
        lambda var, op, val: f"{var} {op} {val}",
        var=st.sampled_from(["sheet_num", "start_item", "end_item", "stage"]),
        op=st.sampled_from([">=", "<=", "==", "!=", ">", "<"]),
        val=st.integers(min_value=0, max_value=100),
    ),
)

_compound_condition_strategy = st.one_of(
    _condition_strategy,
    st.builds(
        lambda a, b: f"{a} and {b}",
        a=st.builds(
            lambda var, op, val: f"{var} {op} {val}",
            var=st.sampled_from(["sheet_num", "start_item"]),
            op=st.sampled_from([">=", "<=", "==", "!="]),
            val=st.integers(min_value=0, max_value=50),
        ),
        b=st.builds(
            lambda var, op, val: f"{var} {op} {val}",
            var=st.sampled_from(["end_item", "stage"]),
            op=st.sampled_from([">=", "<=", "==", "!="]),
            val=st.integers(min_value=0, max_value=50),
        ),
    ),
)

_condition_context_strategy = st.fixed_dictionaries({
    "sheet_num": st.integers(min_value=1, max_value=100),
    "start_item": st.integers(min_value=1, max_value=1000),
    "end_item": st.integers(min_value=1, max_value=1000),
    "stage": st.integers(min_value=1, max_value=10),
    "instance": st.integers(min_value=1, max_value=10),
    "fan_count": st.integers(min_value=1, max_value=10),
    "total_sheets": st.integers(min_value=1, max_value=100),
    "total_stages": st.integers(min_value=1, max_value=10),
})


def _full_validation_rule_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Strategy that generates all five validation rule types with valid fields."""
    return st.one_of(
        # file_exists
        st.fixed_dictionaries({
            "type": st.just("file_exists"),
            "path": st.just("{workspace}/output.txt"),
            "stage": st.integers(min_value=1, max_value=10),
            "retry_count": st.integers(min_value=0, max_value=10),
            "retry_delay_ms": st.integers(min_value=0, max_value=5000),
            "condition": _condition_strategy,
            "description": st.one_of(st.none(), _short_text),
        }),
        # file_modified
        st.fixed_dictionaries({
            "type": st.just("file_modified"),
            "path": st.just("{workspace}/output.txt"),
            "stage": st.integers(min_value=1, max_value=10),
            "retry_count": st.integers(min_value=0, max_value=10),
            "retry_delay_ms": st.integers(min_value=0, max_value=5000),
            "condition": _condition_strategy,
            "description": st.one_of(st.none(), _short_text),
        }),
        # content_contains
        st.fixed_dictionaries({
            "type": st.just("content_contains"),
            "path": st.just("{workspace}/output.txt"),
            "pattern": _short_text,
            "stage": st.integers(min_value=1, max_value=10),
            "retry_count": st.integers(min_value=0, max_value=10),
            "retry_delay_ms": st.integers(min_value=0, max_value=5000),
            "condition": _condition_strategy,
            "description": st.one_of(st.none(), _short_text),
        }),
        # content_regex (only valid regex patterns)
        st.fixed_dictionaries({
            "type": st.just("content_regex"),
            "path": st.just("{workspace}/output.txt"),
            "pattern": st.sampled_from([r"\d+", r"[a-z]+", r"^DONE$", r"pass(ed)?", r"\w+"]),
            "stage": st.integers(min_value=1, max_value=10),
            "retry_count": st.integers(min_value=0, max_value=10),
            "retry_delay_ms": st.integers(min_value=0, max_value=5000),
            "condition": _condition_strategy,
            "description": st.one_of(st.none(), _short_text),
        }),
        # command_succeeds
        st.fixed_dictionaries({
            "type": st.just("command_succeeds"),
            "command": st.sampled_from(["echo ok", "true", "test -f /dev/null"]),
            "stage": st.integers(min_value=1, max_value=10),
            "retry_count": st.integers(min_value=0, max_value=10),
            "retry_delay_ms": st.integers(min_value=0, max_value=5000),
            "condition": _condition_strategy,
            "description": st.one_of(st.none(), _short_text),
            "working_directory": st.one_of(st.none(), st.just("/tmp")),
        }),
    )


def _validation_issue_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Strategy for constructing ValidationIssue kwargs."""
    return st.fixed_dictionaries({
        "check_id": st.from_regex(r"V[0-9]{3}", fullmatch=True),
        "severity": _severity_strategy,
        "message": _short_text,
        "line": st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
        "column": st.one_of(st.none(), st.integers(min_value=1, max_value=200)),
        "context": st.one_of(st.none(), _short_text),
        "suggestion": st.one_of(st.none(), _short_text),
        "auto_fixable": st.booleans(),
    })


def _diagnosis_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Strategy for constructing Diagnosis kwargs."""
    return st.fixed_dictionaries({
        "error_code": st.from_regex(r"E[0-9]{3}", fullmatch=True),
        "issue": _short_text,
        "explanation": _short_text,
        "suggestion": _short_text,
        "confidence": _unit_float,
        "remedy_name": st.one_of(st.none(), _short_text),
        "requires_confirmation": st.booleans(),
    })


def _healing_report_context_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Strategy for constructing minimal ErrorContext kwargs for HealingReport."""
    return st.fixed_dictionaries({
        "error_code": st.from_regex(r"E[0-9]{3}", fullmatch=True),
        "error_message": _short_text,
        "error_category": st.sampled_from(["preflight", "configuration", "execution", "process"]),
        "sheet_number": st.integers(min_value=0, max_value=100),
    })


# ---------------------------------------------------------------------------
# Test: ValidationRule round-trip
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(data=_full_validation_rule_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_validation_rule_roundtrip(data: dict[str, Any]) -> None:
    """ValidationRule survives model_dump → model_validate round-trip."""
    from mozart.core.config.execution import ValidationRule

    rule = ValidationRule.model_validate(data)
    dumped = rule.model_dump()
    restored = ValidationRule.model_validate(dumped)

    assert restored.type == rule.type
    assert restored.path == rule.path
    assert restored.pattern == rule.pattern
    assert restored.stage == rule.stage
    assert restored.condition == rule.condition
    assert restored.retry_count == rule.retry_count
    assert restored.retry_delay_ms == rule.retry_delay_ms
    assert restored.command == rule.command


# ---------------------------------------------------------------------------
# Test: ValidationRule from conftest_adversarial strategy also round-trips
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(data=validation_rule_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_validation_rule_conftest_strategy_roundtrip(data: dict[str, Any]) -> None:
    """Shared validation_rule_strategy produces valid, round-trippable rules."""
    from mozart.core.config.execution import ValidationRule

    rule = ValidationRule.model_validate(data)
    dumped = rule.model_dump()
    restored = ValidationRule.model_validate(dumped)

    assert restored.type == rule.type
    assert restored.model_dump() == dumped


# ---------------------------------------------------------------------------
# Test: Condition evaluation never crashes
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    condition=_compound_condition_strategy,
    context=_condition_context_strategy,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_condition_evaluation_never_crashes(
    condition: str | None,
    context: dict[str, int],
) -> None:
    """Any well-formed condition expression evaluates without raising."""
    from mozart.validation.rendering import _check_condition

    result = _check_condition(condition, context)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Test: None condition always returns True
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(context=_condition_context_strategy)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_none_condition_always_true(context: dict[str, int]) -> None:
    """A None condition is unconditional — always returns True."""
    from mozart.validation.rendering import _check_condition

    assert _check_condition(None, context) is True


# ---------------------------------------------------------------------------
# Test: Condition evaluation is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    condition=_compound_condition_strategy,
    context=_condition_context_strategy,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_condition_evaluation_deterministic(
    condition: str | None,
    context: dict[str, int],
) -> None:
    """Same condition + context always produces the same result."""
    from mozart.validation.rendering import _check_condition

    result1 = _check_condition(condition, context)
    result2 = _check_condition(condition, context)
    assert result1 == result2


# ---------------------------------------------------------------------------
# Test: ValidationIssue format_short / format_full never crash
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(data=_validation_issue_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_validation_issue_format_roundtrip(data: dict[str, Any]) -> None:
    """Any ValidationIssue can produce format_short and format_full without error."""
    from mozart.validation.base import ValidationIssue, ValidationSeverity

    severity = ValidationSeverity(data.pop("severity"))
    issue = ValidationIssue(severity=severity, **data)

    short = issue.format_short()
    full = issue.format_full()

    assert isinstance(short, str)
    assert isinstance(full, str)
    assert issue.check_id in short
    assert issue.check_id in full
    assert issue.message in short


# ---------------------------------------------------------------------------
# Test: Reporter JSON output is always valid JSON
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(issues=st.lists(_validation_issue_strategy(), min_size=0, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_reporter_json_always_valid(issues: list[dict[str, Any]]) -> None:
    """Any set of ValidationIssues produces parseable JSON from report_json."""
    from mozart.validation.base import ValidationIssue, ValidationSeverity
    from mozart.validation.reporter import ValidationReporter

    issue_objects = []
    for data in issues:
        severity = ValidationSeverity(data.pop("severity"))
        issue_objects.append(ValidationIssue(severity=severity, **data))

    reporter = ValidationReporter()
    json_str = reporter.report_json(issue_objects)

    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "valid" in parsed
    assert "error_count" in parsed
    assert "warning_count" in parsed
    assert "info_count" in parsed
    assert "issues" in parsed
    assert isinstance(parsed["issues"], list)
    assert len(parsed["issues"]) == len(issue_objects)


# ---------------------------------------------------------------------------
# Test: Reporter JSON valid flag is consistent with error presence
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(issues=st.lists(_validation_issue_strategy(), min_size=1, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_reporter_json_valid_flag_consistency(issues: list[dict[str, Any]]) -> None:
    """JSON 'valid' flag is True iff no ERROR-severity issues exist."""
    from mozart.validation.base import ValidationIssue, ValidationSeverity
    from mozart.validation.reporter import ValidationReporter

    issue_objects = []
    for data in issues:
        severity = ValidationSeverity(data.pop("severity"))
        issue_objects.append(ValidationIssue(severity=severity, **data))

    reporter = ValidationReporter()
    json_str = reporter.report_json(issue_objects)
    parsed = json.loads(json_str)

    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issue_objects)
    assert parsed["valid"] == (not has_errors)
    assert parsed["error_count"] == sum(
        1 for i in issue_objects if i.severity == ValidationSeverity.ERROR
    )


# ---------------------------------------------------------------------------
# Test: Reporter format_plain always produces a string
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(issues=st.lists(_validation_issue_strategy(), min_size=0, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_reporter_format_plain_always_string(issues: list[dict[str, Any]]) -> None:
    """format_plain always returns a string with expected summary structure."""
    from mozart.validation.base import ValidationIssue, ValidationSeverity
    from mozart.validation.reporter import ValidationReporter

    issue_objects = []
    for data in issues:
        severity = ValidationSeverity(data.pop("severity"))
        issue_objects.append(ValidationIssue(severity=severity, **data))

    reporter = ValidationReporter()
    plain = reporter.format_plain(issue_objects)

    assert isinstance(plain, str)
    if not issue_objects:
        assert "no issues found" in plain.lower()
    else:
        assert "FAILED" in plain or "PASSED" in plain


# ---------------------------------------------------------------------------
# Test: ValidationRunner sorts errors before warnings before info
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(issues=st.lists(_validation_issue_strategy(), min_size=2, max_size=15))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_runner_severity_sorting(issues: list[dict[str, Any]]) -> None:
    """ValidationRunner.validate sorts issues: ERROR < WARNING < INFO."""
    from unittest.mock import MagicMock

    from mozart.validation.base import ValidationIssue, ValidationSeverity
    from mozart.validation.runner import ValidationRunner

    issue_objects = []
    for data in issues:
        severity = ValidationSeverity(data.pop("severity"))
        issue_objects.append(ValidationIssue(severity=severity, **data))

    # Create a mock check that returns our issues
    mock_check = MagicMock()
    mock_check.check_id = "VTEST"
    mock_check.check.return_value = issue_objects

    runner = ValidationRunner(checks=[mock_check])
    config = MagicMock()
    sorted_issues = runner.validate(config, Path("/tmp"), "")

    severity_order = {
        ValidationSeverity.ERROR: 0,
        ValidationSeverity.WARNING: 1,
        ValidationSeverity.INFO: 2,
    }
    for i in range(len(sorted_issues) - 1):
        assert severity_order[sorted_issues[i].severity] <= severity_order[sorted_issues[i + 1].severity]


# ---------------------------------------------------------------------------
# Test: ValidationRunner exit_code is 1 iff errors exist
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(issues=st.lists(_validation_issue_strategy(), min_size=0, max_size=10))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_runner_exit_code_consistency(issues: list[dict[str, Any]]) -> None:
    """Exit code is 1 if any ERROR, 0 otherwise."""
    from mozart.validation.base import ValidationIssue, ValidationSeverity
    from mozart.validation.runner import ValidationRunner

    issue_objects = []
    for data in issues:
        severity = ValidationSeverity(data.pop("severity"))
        issue_objects.append(ValidationIssue(severity=severity, **data))

    runner = ValidationRunner()
    exit_code = runner.get_exit_code(issue_objects)
    has_errors = any(i.severity == ValidationSeverity.ERROR for i in issue_objects)

    assert exit_code == (1 if has_errors else 0)


# ---------------------------------------------------------------------------
# Test: Diagnosis round-trip formatting
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(data=_diagnosis_strategy())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_diagnosis_format_roundtrip(data: dict[str, Any]) -> None:
    """Any Diagnosis produces valid format_short and format_full strings."""
    from mozart.healing.diagnosis import Diagnosis

    diag = Diagnosis(**data)

    short = diag.format_short()
    full = diag.format_full()

    assert isinstance(short, str)
    assert isinstance(full, str)
    assert diag.error_code in short
    assert diag.error_code in full
    assert diag.issue in short


# ---------------------------------------------------------------------------
# Test: RemedyResult __str__ never crashes
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    success=st.booleans(),
    message=_short_text,
    action_taken=_short_text,
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_remedy_result_str(success: bool, message: str, action_taken: str) -> None:
    """RemedyResult.__str__ produces a consistent status marker."""
    from mozart.healing.remedies.base import RemedyResult

    result = RemedyResult(success=success, message=message, action_taken=action_taken)
    text = str(result)

    assert isinstance(text, str)
    if success:
        assert "✓" in text
    else:
        assert "✗" in text
    assert action_taken in text


# ---------------------------------------------------------------------------
# Test: HealingReport properties are consistent
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    ctx_data=_healing_report_context_strategy(),
    action_successes=st.lists(st.booleans(), min_size=0, max_size=5),
    n_diagnostics=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_healing_report_properties(
    ctx_data: dict[str, Any],
    action_successes: list[bool],
    n_diagnostics: int,
) -> None:
    """HealingReport computed properties are consistent with input data."""
    from mozart.healing.context import ErrorContext
    from mozart.healing.coordinator import HealingReport
    from mozart.healing.remedies.base import RemedyResult

    context = ErrorContext(**ctx_data)
    actions_taken = [
        (f"remedy_{i}", RemedyResult(success=s, message="msg", action_taken="act"))
        for i, s in enumerate(action_successes)
    ]
    diagnostic_outputs = [
        (f"diag_{i}", f"Guidance for issue {i}") for i in range(n_diagnostics)
    ]

    report = HealingReport(
        error_context=context,
        actions_taken=actions_taken,
        diagnostic_outputs=diagnostic_outputs,
    )

    # any_remedies_applied iff at least one action succeeded
    assert report.any_remedies_applied == any(action_successes)
    # should_retry mirrors any_remedies_applied
    assert report.should_retry == report.any_remedies_applied
    # issues_remaining = diagnostics + failed actions
    failed_actions = sum(1 for s in action_successes if not s)
    assert report.issues_remaining == n_diagnostics + failed_actions


# ---------------------------------------------------------------------------
# Test: HealingReport.format always produces a string
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    ctx_data=_healing_report_context_strategy(),
    verbose=st.booleans(),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_healing_report_format_never_crashes(
    ctx_data: dict[str, Any],
    verbose: bool,
) -> None:
    """HealingReport.format(verbose) always returns a non-empty string."""
    from mozart.healing.context import ErrorContext
    from mozart.healing.coordinator import HealingReport

    context = ErrorContext(**ctx_data)
    report = HealingReport(error_context=context)

    text = report.format(verbose=verbose)
    assert isinstance(text, str)
    assert len(text) > 0
    assert ctx_data["error_code"] in text


# ---------------------------------------------------------------------------
# Test: _expand_path is idempotent when no placeholders remain
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    template=st.sampled_from([
        "{workspace}/output.txt",
        "{workspace}/{sheet_num}/result.json",
        "{workspace}/stage-{stage}/instance-{instance}.txt",
        "no-placeholders-here.txt",
    ]),
    workspace=st.just("/tmp/test-workspace"),
    sheet_num=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_expand_path_idempotent(
    template: str, workspace: str, sheet_num: int
) -> None:
    """After full expansion, applying _expand_path again has no effect."""
    from mozart.validation.rendering import _expand_path

    ctx = {
        "workspace": workspace,
        "sheet_num": str(sheet_num),
        "stage": str(sheet_num),
        "instance": "1",
    }
    once = _expand_path(template, ctx)
    twice = _expand_path(once, ctx)
    assert once == twice


# ---------------------------------------------------------------------------
# Test: _build_snippet truncation invariant
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(
    n_lines=st.integers(min_value=0, max_value=50),
    max_lines=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_build_snippet_truncation(n_lines: int, max_lines: int) -> None:
    """Snippet is truncated with '...' iff input exceeds max_lines."""
    from mozart.validation.rendering import _build_snippet

    text = "\n".join(f"line {i}" for i in range(n_lines))
    snippet = _build_snippet(text, max_lines=max_lines)

    if n_lines <= max_lines:
        assert snippet == text
    else:
        assert snippet.endswith("...")
        # Should have exactly max_lines lines before the "..."
        output_lines = snippet.split("\n")
        # Last element is "...", preceding lines = max_lines
        assert len(output_lines) == max_lines + 1


# ---------------------------------------------------------------------------
# Test: count_by_severity totals match input length
# ---------------------------------------------------------------------------


@pytest.mark.property_based
@given(issues=st.lists(_validation_issue_strategy(), min_size=0, max_size=15))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_count_by_severity_totals(issues: list[dict[str, Any]]) -> None:
    """Sum of severity counts equals total issue count."""
    from mozart.validation.base import ValidationIssue, ValidationSeverity
    from mozart.validation.runner import ValidationRunner

    issue_objects = []
    for data in issues:
        severity = ValidationSeverity(data.pop("severity"))
        issue_objects.append(ValidationIssue(severity=severity, **data))

    runner = ValidationRunner()
    counts = runner.count_by_severity(issue_objects)

    total = sum(counts.values())
    assert total == len(issue_objects)
