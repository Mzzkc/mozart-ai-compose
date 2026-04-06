"""Error classification and handling.

Re-exports all public symbols for backward compatibility.
"""

from marianne.core.errors.classifier import ErrorClassifier
from marianne.core.errors.codes import (
    ErrorCategory,
    ErrorCode,
    ExitReason,
    RetryBehavior,
    RetryDelays,
    Severity,
)
from marianne.core.errors.models import (
    ClassificationInput,
    ClassificationResult,
    ClassifiedError,
    ErrorChain,
    ErrorInfo,
    ParsedCliError,
)
from marianne.core.errors.parsers import (
    ROOT_CAUSE_PRIORITY,
    classify_single_json_error,
    select_root_cause,
    try_parse_json_errors,
)
from marianne.core.errors.signals import (
    FATAL_SIGNALS,
    RETRIABLE_SIGNALS,
    get_signal_name,
)

__all__ = [
    "ErrorCategory",
    "ErrorCode",
    "ExitReason",
    "RetryBehavior",
    "RetryDelays",
    "Severity",
    "ClassificationInput",
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
