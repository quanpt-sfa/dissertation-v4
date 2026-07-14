"""Compiled schema API; future steps validate frames without parsing YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .errors import UnknownReferenceError


class DataFrameLike(Protocol):
    """Small structural protocol used without requiring pandas at P0."""

    @property
    def columns(self) -> object: ...


@dataclass(frozen=True)
class SchemaRegistry:
    """Resolved schema declarations keyed by schema ID."""

    schemas: Mapping[str, object]

    def get(self, schema_id: str) -> object:
        """Return one compiled declaration."""
        try:
            return self.schemas[schema_id]
        except KeyError as exc:
            raise UnknownReferenceError(f"schema={schema_id}: unknown schema ID") from exc

    def validate_dataframe(self, schema_id: str, dataframe: DataFrameLike) -> None:
        """Perform the P0 structural column-contract check without artifact I/O."""
        schema = self.get(schema_id)
        if not isinstance(schema, dict):
            raise UnknownReferenceError(f"schema={schema_id}: malformed compiled schema")
        columns = schema.get("columns", [])
        expected = [item["physical_name"] for item in columns if isinstance(item, dict)]
        actual = list(dataframe.columns)  # type: ignore[arg-type]
        missing = set(expected) - set(actual)
        if missing:
            raise ValueError(f"schema={schema_id}: dataframe misses physical columns {sorted(missing)}")


def compile_schemas(raw: object, columns: object) -> SchemaRegistry:
    """Resolve logical column references to physical names in schema declarations."""
    if not isinstance(raw, dict) or not isinstance(columns, dict):
        raise UnknownReferenceError("schemas: schemas and columns must be mappings")
    compiled: dict[str, object] = {}
    for schema_id, specification in raw.items():
        if not isinstance(specification, dict):
            raise UnknownReferenceError(f"schema={schema_id}: schema must be a mapping")
        resolved = dict(specification)
        entries = specification.get("columns")
        if not isinstance(entries, list) or not entries:
            raise UnknownReferenceError(f"schema={schema_id}: ordered columns are required")
        rendered: list[dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("column"), str):
                raise UnknownReferenceError(f"schema={schema_id}: each column needs a logical column reference")
            column_id = entry["column"]
            col = columns.get(column_id)
            if not isinstance(col, dict) or not isinstance(col.get("physical_name"), str):
                raise UnknownReferenceError(f"schema={schema_id}, column={column_id}: unknown column reference")
            item = dict(entry)
            item["physical_name"] = col["physical_name"]
            rendered.append(item)
        resolved["columns"] = rendered
        compiled[str(schema_id)] = resolved
    return SchemaRegistry(compiled)


def validate_dataframe(schema_registry: SchemaRegistry, schema_id: str, dataframe: DataFrameLike) -> None:
    """Reusable function form of schema validation for later core services."""
    schema_registry.validate_dataframe(schema_id, dataframe)
