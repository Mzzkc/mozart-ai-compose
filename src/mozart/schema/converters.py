"""Type converters for serialising/deserialising Python values to/from SQLite.

Provides bidirectional conversion between Python types (datetime, bool, Enum,
list, dict, Pydantic models) and SQLite column values (TEXT, INTEGER, REAL).
"""

from __future__ import annotations

import dataclasses
import json
import types
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from mozart.core.logging import get_logger

_logger = get_logger("schema.converters")

# Type alias matching sqlite3.execute() signature.
SQLParam = str | int | float | bytes | None


def _json_default(obj: Any) -> Any:
    """JSON serialisation fallback for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap Optional[X] or X | None to (inner_type, is_optional)."""
    origin = get_origin(annotation)
    is_union = origin is Union
    if not is_union and hasattr(types, "UnionType"):
        is_union = isinstance(annotation, types.UnionType)
    if is_union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], True
    return annotation, False


def _is_json_type(annotation: Any) -> bool:
    """Check if a type should be JSON-serialised to TEXT."""
    origin = get_origin(annotation)
    if origin in (list, dict, frozenset, set, tuple):
        return True
    if hasattr(annotation, "__required_keys__") or hasattr(annotation, "__optional_keys__"):
        return True  # TypedDict
    if isinstance(annotation, type):
        if dataclasses.is_dataclass(annotation):
            return True
        try:
            from pydantic import BaseModel

            if issubclass(annotation, BaseModel):
                return True
        except ImportError:  # pragma: no cover
            pass
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize_field(value: Any, annotation: Any) -> SQLParam:
    """Convert a Python value to a SQLite-compatible parameter.

    Args:
        value: The Python value to serialize.
        annotation: The type annotation for the field.

    Returns:
        A value suitable for sqlite3 execute() binding.
    """
    if value is None:
        return None

    # Unwrap Optional so we operate on the inner type
    inner, _ = _unwrap_optional(annotation)

    # datetime → ISO string (must check before generic str check)
    if isinstance(value, datetime):
        return value.isoformat()

    # bool → int (must check before int, since bool is subclass of int)
    if isinstance(value, bool):
        return 1 if value else 0

    # Enum → .value string
    if isinstance(value, Enum):
        return value.value

    # Complex types → JSON
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=_json_default)

    # Pydantic model instance → JSON
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), default=_json_default)

    # Dataclass instance → JSON
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json.dumps(dataclasses.asdict(value), default=_json_default)

    # Direct types (str, int, float, bytes)
    return value


def deserialize_field(value: Any, annotation: Any) -> Any:
    """Convert a SQLite value back to the expected Python type.

    Args:
        value: The raw value from SQLite.
        annotation: The type annotation for the target field.

    Returns:
        The reconstructed Python value.
    """
    if value is None:
        return None

    inner, _ = _unwrap_optional(annotation)

    # datetime
    if inner is datetime:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                _logger.warning("corrupt_timestamp", value=value)
                return None
        return value

    # bool (stored as INTEGER 0/1)
    if inner is bool:
        return bool(value)

    # Enum
    if isinstance(inner, type) and issubclass(inner, Enum):
        try:
            return inner(value)
        except ValueError:
            _logger.warning("unknown_enum_value", type=inner.__name__, value=value)
            return value

    # Literal — return as-is
    if get_origin(inner) is Literal:
        return value

    # JSON types (list, dict, TypedDict, Pydantic sub-models, dataclasses)
    if _is_json_type(inner):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                _logger.warning(
                    "json_parse_failed",
                    raw_length=len(value),
                    raw_preview=value[:100],
                    error=str(exc),
                )
                return None
        return value

    # Direct types (str, int, float)
    return value
