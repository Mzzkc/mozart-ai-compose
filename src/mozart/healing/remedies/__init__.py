"""Remedy implementations for self-healing.

This package contains all built-in remedies:

Automatic remedies (safe, apply without confirmation):
- CreateMissingWorkspaceRemedy: Creates missing workspace directories
- CreateMissingParentDirsRemedy: Creates missing parent directories
- FixPathSeparatorsRemedy: Fixes Windows path separators on Unix

Suggested remedies (require user confirmation):
- SuggestJinjaFixRemedy: Suggests fixes for Jinja template errors

Diagnostic remedies (provide guidance only):
- DiagnoseAuthErrorRemedy: Diagnoses authentication failures
- DiagnoseMissingCLIRemedy: Diagnoses missing Claude CLI
"""

from mozart.healing.remedies.base import Remedy, RemedyCategory, RemedyResult, RiskLevel
from mozart.healing.remedies.diagnostics import DiagnoseAuthErrorRemedy, DiagnoseMissingCLIRemedy
from mozart.healing.remedies.jinja import SuggestJinjaFixRemedy
from mozart.healing.remedies.paths import (
    CreateMissingParentDirsRemedy,
    CreateMissingWorkspaceRemedy,
    FixPathSeparatorsRemedy,
)

__all__ = [
    # Base
    "Remedy",
    "RemedyCategory",
    "RemedyResult",
    "RiskLevel",
    # Automatic
    "CreateMissingWorkspaceRemedy",
    "CreateMissingParentDirsRemedy",
    "FixPathSeparatorsRemedy",
    # Suggested
    "SuggestJinjaFixRemedy",
    # Diagnostic
    "DiagnoseAuthErrorRemedy",
    "DiagnoseMissingCLIRemedy",
]
