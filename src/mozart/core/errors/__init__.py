"""Error classification and handling.

Re-exports all public symbols for backward compatibility.
"""

from mozart.core.errors.codes import (
    ErrorCategory,
    ErrorCode,
    ExitReason,
    RetryBehavior,
    RetryDelays,
    Severity,
)
from mozart.core.errors.models import (
    ClassificationResult,
    ClassifiedError,
    ErrorChain,
    ErrorInfo,
    ParsedCliError,
)
from mozart.core.errors.signals import (
    FATAL_SIGNALS,
    RETRIABLE_SIGNALS,
    get_signal_name,
)
from mozart.core.errors.parsers import (
    ROOT_CAUSE_PRIORITY,
    classify_single_json_error,
    select_root_cause,
    try_parse_json_errors,
)
from mozart.core.errors.classifier import ErrorClassifier

__all__ = [
    "ErrorCategory",
    "ErrorCode",
    "ExitReason",
    "RetryBehavior",
    "RetryDelays",
    "Severity",
    "ClassificationResult",
    "ClassifiedError",
    "ErrorChain",
    "ErrorInfo",
    "ParsedCliError",
    "FATAL_SIGNALS",
    "RETRIABLE_SIGNALS",
    "get_signal_name",
    "ROOT_CAUSE_PRIORITY",
    "classify_single_json_error",
    "select_root_cause",
    "try_parse_json_errors",
    "ErrorClassifier",
]
