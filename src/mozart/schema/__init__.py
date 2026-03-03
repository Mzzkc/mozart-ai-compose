"""Unified schema management for Mozart's SQLite databases.

Provides a model-driven schema registry that maps Pydantic and dataclass
models to SQLite tables, enabling automatic DDL generation, type-safe
save/load, migration management, and CI drift detection.
"""

from mozart.schema.converters import (
    SQLParam,
    deserialize_field,
    serialize_field,
)
from mozart.schema.registry import (
    ColumnSource,
    FieldSpec,
    TableMapping,
    generate_create_table,
    generate_upsert,
    get_column_sources,
    get_expected_columns,
    get_field_specs,
    get_sqlite_type,
)

__all__ = [
    "ColumnSource",
    "FieldSpec",
    "SQLParam",
    "TableMapping",
    "deserialize_field",
    "generate_create_table",
    "generate_upsert",
    "get_column_sources",
    "get_expected_columns",
    "get_field_specs",
    "get_sqlite_type",
    "serialize_field",
]
