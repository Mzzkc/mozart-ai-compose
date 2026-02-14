"""Cross-sheet semantic validation.

Compares key-value pairs extracted from sheet outputs to detect
semantic inconsistencies between sequential sheets.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mozart.utils.time import utc_now


@dataclass
class KeyVariable:
    """A key-value pair extracted from sheet output."""

    key: str
    value: str
    source_line: str = ""
    line_number: int = 0


@dataclass
class SemanticInconsistency:
    """Represents a semantic inconsistency between sheets."""

    key: str
    sheet_a: int
    value_a: str
    sheet_b: int
    value_b: str
    severity: str = "warning"

    def format_message(self) -> str:
        """Format as human-readable message."""
        return (
            f"Key '{self.key}' has inconsistent values: "
            f"sheet {self.sheet_a}='{self.value_a}' vs "
            f"sheet {self.sheet_b}='{self.value_b}'"
        )


@dataclass
class SemanticConsistencyResult:
    """Result of cross-sheet semantic consistency check."""

    sheets_compared: list[int] = field(default_factory=list)
    inconsistencies: list[SemanticInconsistency] = field(default_factory=list)
    keys_checked: int = 0
    checked_at: datetime = field(default_factory=utc_now)

    @property
    def is_consistent(self) -> bool:
        """True if no inconsistencies were found."""
        return not self.inconsistencies

    @property
    def error_count(self) -> int:
        """Count of error-severity inconsistencies."""
        return sum(1 for i in self.inconsistencies if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-severity inconsistencies."""
        return sum(1 for i in self.inconsistencies if i.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "sheets_compared": self.sheets_compared,
            "inconsistencies": [
                {
                    "key": i.key,
                    "sheet_a": i.sheet_a,
                    "value_a": i.value_a,
                    "sheet_b": i.sheet_b,
                    "value_b": i.value_b,
                    "severity": i.severity,
                }
                for i in self.inconsistencies
            ],
            "keys_checked": self.keys_checked,
            "checked_at": self.checked_at.isoformat(),
            "is_consistent": self.is_consistent,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


class KeyVariableExtractor:
    """Extracts key-value pairs from sheet output content."""

    _KEY_VALUE_PATTERN = re.compile(
        r"^([A-Z][A-Z0-9_]*)\s*(?::|=)\s*(.+)$",
        re.MULTILINE
    )

    def __init__(
        self,
        key_filter: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize extractor."""
        self.key_filter = key_filter
        self.case_sensitive = case_sensitive

    def extract(self, content: str) -> list[KeyVariable]:
        """Extract key-value pairs from content."""
        if not content:
            return []

        variables: list[KeyVariable] = []
        seen_keys: set[str] = set()

        lines = content.split("\n")
        line_map: dict[str, int] = {}
        for i, line in enumerate(lines, 1):
            line_map[line] = i

        for match in self._KEY_VALUE_PATTERN.finditer(content):
            key = match.group(1)
            value = match.group(2).strip()
            source_line = match.group(0)

            if self._should_include(key) and key not in seen_keys:
                variables.append(KeyVariable(
                    key=key,
                    value=value,
                    source_line=source_line,
                    line_number=line_map.get(source_line, 0),
                ))
                seen_keys.add(key)

        return variables

    def _should_include(self, key: str) -> bool:
        """Check if key should be included based on filter."""
        if self.key_filter is None:
            return True

        if self.case_sensitive:
            return key in self.key_filter

        key_lower = key.lower()
        return any(k.lower() == key_lower for k in self.key_filter)


class SemanticConsistencyChecker:
    """Checks semantic consistency between sequential sheet outputs."""

    def __init__(
        self,
        extractor: KeyVariableExtractor | None = None,
        strict_mode: bool = False,
    ) -> None:
        """Initialize checker."""
        self.extractor = extractor or KeyVariableExtractor()
        self.strict_mode = strict_mode

    def check_consistency(
        self,
        sheet_outputs: dict[int, str],
        sequential_only: bool = True,
    ) -> SemanticConsistencyResult:
        """Check semantic consistency across sheet outputs."""
        result = SemanticConsistencyResult(
            sheets_compared=sorted(sheet_outputs.keys()),
        )

        if len(sheet_outputs) < 2:
            return result

        sheet_variables: dict[int, dict[str, KeyVariable]] = {}
        for sheet_num, content in sheet_outputs.items():
            variables = self.extractor.extract(content)
            sheet_variables[sheet_num] = {v.key: v for v in variables}

        all_keys: set[str] = set()
        for vars_dict in sheet_variables.values():
            all_keys.update(vars_dict.keys())
        result.keys_checked = len(all_keys)

        sheets_sorted = sorted(sheet_outputs.keys())

        if sequential_only:
            for i in range(len(sheets_sorted) - 1):
                sheet_a = sheets_sorted[i]
                sheet_b = sheets_sorted[i + 1]
                self._compare_sheets(
                    sheet_a, sheet_variables[sheet_a],
                    sheet_b, sheet_variables[sheet_b],
                    result,
                )
        else:
            for key in all_keys:
                value_groups: dict[str, list[int]] = defaultdict(list)
                for sheet_num in sheets_sorted:
                    var = sheet_variables[sheet_num].get(key)
                    if var is not None:
                        value_groups[var.value.lower()].append(sheet_num)

                if len(value_groups) <= 1:
                    continue

                group_list = list(value_groups.values())
                for gi in range(len(group_list)):
                    for gj in range(gi + 1, len(group_list)):
                        for sheet_a in group_list[gi]:
                            for sheet_b in group_list[gj]:
                                var_a = sheet_variables[sheet_a][key]
                                var_b = sheet_variables[sheet_b][key]
                                result.inconsistencies.append(SemanticInconsistency(
                                    key=key,
                                    sheet_a=sheet_a,
                                    value_a=var_a.value,
                                    sheet_b=sheet_b,
                                    value_b=var_b.value,
                                    severity="error" if self.strict_mode else "warning",
                                ))

        return result

    def _compare_sheets(
        self,
        sheet_a: int,
        vars_a: dict[str, KeyVariable],
        sheet_b: int,
        vars_b: dict[str, KeyVariable],
        result: SemanticConsistencyResult,
    ) -> None:
        """Compare variables between two sheets."""
        common_keys = vars_a.keys() & vars_b.keys()

        for key in common_keys:
            var_a = vars_a[key]
            var_b = vars_b[key]

            if var_a.value.lower() != var_b.value.lower():
                result.inconsistencies.append(SemanticInconsistency(
                    key=key,
                    sheet_a=sheet_a,
                    value_a=var_a.value,
                    sheet_b=sheet_b,
                    value_b=var_b.value,
                    severity="error" if self.strict_mode else "warning",
                ))
