# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
"""Executable contracts compiled from YAML; Pandera is isolated to this boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pandera.pandas as pa

from .errors import UnknownReferenceError


@dataclass(frozen=True)
class SchemaRegistry:
    schemas: Mapping[str, object]

    def get(self, schema_id: str) -> object:
        try:
            return self.schemas[schema_id]
        except KeyError as exc:
            raise UnknownReferenceError(f"schema={schema_id}: unknown schema") from exc


@dataclass(frozen=True)
class ContractRegistry:
    schemas: SchemaRegistry
    artifacts: Mapping[str, object]

    def validate(self, artifact_id: str, value: object) -> None:
        artifact = self.artifacts.get(artifact_id)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("schema_id"), str):
            raise UnknownReferenceError(f"artifact={artifact_id}: unknown artifact")
        specification = self.schemas.get(str(artifact["schema_id"]))
        if not isinstance(specification, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: malformed schema")
        spec = cast(dict[str, Any], specification)
        kind = spec.get("contract_type")
        if kind == "dataframe":
            self._dataframe(spec, value)
        elif kind in {"json_object", "receipt"}:
            if not isinstance(value, dict):
                raise ValueError(f"artifact={artifact_id}: JSON object required")
            self._required_keys(spec, value)
        elif kind == "json_array":
            if not isinstance(value, list):
                raise ValueError(f"artifact={artifact_id}: JSON array required")
            if spec.get("item_type") == "object" and not all(
                isinstance(item, dict) for item in value
            ):
                raise ValueError(f"artifact={artifact_id}: every array item must be an object")
        elif kind == "text":
            if not isinstance(value, str) or len(value) < int(spec.get("min_length", 0)):
                raise ValueError(f"artifact={artifact_id}: text constraint violated")
        elif kind == "markdown":
            prefix = spec.get("required_prefix")
            if not isinstance(value, str) or (
                isinstance(prefix, str) and not value.startswith(prefix)
            ):
                raise ValueError(f"artifact={artifact_id}: Markdown prefix required")
        else:
            raise ValueError(f"artifact={artifact_id}: unsupported contract type {kind}")

    def schema_version(self, schema_id: str) -> int:
        specification = self.schemas.get(schema_id)
        if not isinstance(specification, dict) or not isinstance(specification.get("version"), int):
            raise UnknownReferenceError(f"schema={schema_id}: version required")
        return int(specification["version"])

    @staticmethod
    def _required_keys(spec: Mapping[str, Any], value: Mapping[object, object]) -> None:
        missing = [str(key) for key in spec.get("required_keys", []) if key not in value]
        if missing:
            raise ValueError(f"contract: missing required keys {missing}")

    @staticmethod
    def _dataframe(spec: dict[str, Any], value: object) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("contract: DataFrame required")
        entries = spec.get("columns")
        if not isinstance(entries, list) or not entries:
            raise ValueError("contract: columns required")
        names = [str(entry["physical_name"]) for entry in entries]
        strict = bool(spec.get("strict_columns", True))
        if strict and list(value.columns) != names:
            raise ValueError(f"contract: exact ordered columns required {names}")
        if not strict and list(value.columns)[: len(names)] != names:
            raise ValueError("contract: ordered required columns differ")

        columns: dict[str, pa.Column] = {}
        for raw in entries:
            if not isinstance(raw, dict):
                raise ValueError("contract: column mapping required")
            entry = cast(dict[str, Any], raw)
            name = str(entry["physical_name"])
            checks = _column_checks(spec, entry)
            columns[name] = pa.Column(
                _pandera_dtype(str(entry["dtype"])),
                nullable=bool(entry.get("nullable", True)),
                checks=checks,
                coerce=bool(spec.get("coerce", False)),
            )
        for key in spec.get("uniqueness_constraints", []):
            physical = [_physical(spec, logical) for logical in key]
            if value.duplicated(physical).any():
                raise ValueError(f"contract: uniqueness violated for {physical}")
        schema = pa.DataFrameSchema(
            columns,
            strict=strict,
            coerce=bool(spec.get("coerce", False)),
        )
        schema.validate(value, lazy=True)


def _column_checks(spec: Mapping[str, Any], entry: Mapping[str, Any]) -> list[pa.Check]:
    checks: list[pa.Check] = []
    logical = entry.get("column")
    for raw in spec.get("row_checks", []):
        if not isinstance(raw, dict) or raw.get("column") != logical:
            continue
        check = raw.get("check")
        if check == "between":
            checks.append(pa.Check.between(raw["min"], raw["max"]))
        elif check == "greater_than":
            checks.append(pa.Check.greater_than(raw["min"]))
        elif check == "greater_than_or_equal":
            checks.append(pa.Check.greater_than_or_equal_to(raw["min"]))
    return checks


def _physical(spec: Mapping[str, Any], logical: object) -> str:
    for entry in spec.get("columns", []):
        if isinstance(entry, dict) and entry.get("column") == logical:
            return str(entry["physical_name"])
    raise ValueError(f"contract: unknown logical column {logical}")


def _pandera_dtype(dtype: str) -> object:
    mapping: dict[str, object] = {
        "string": pa.String,
        "int16": pa.Int16,
        "int64": pa.Int64,
        "float64": pa.Float64,
        "bool": pa.Bool,
        "datetime64[ns]": pa.DateTime,
    }
    return mapping.get(dtype, dtype)


def compile_schemas(raw: object, columns: object) -> SchemaRegistry:
    if not isinstance(raw, dict) or not isinstance(columns, dict):
        raise UnknownReferenceError("schemas and columns must be mappings")
    compiled: dict[str, object] = {}
    for schema_id, raw_spec in raw.items():
        if not isinstance(raw_spec, dict) or not isinstance(raw_spec.get("contract_type"), str):
            raise UnknownReferenceError(f"schema={schema_id}: contract_type required")
        spec = cast(dict[str, Any], raw_spec)
        resolved = dict(spec)
        if spec["contract_type"] == "dataframe":
            rendered: list[dict[str, object]] = []
            for raw_entry in spec.get("columns", []):
                if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("column"), str):
                    raise UnknownReferenceError(f"schema={schema_id}: logical column required")
                entry = cast(dict[str, Any], raw_entry)
                column = columns.get(entry["column"])
                if not isinstance(column, dict) or not isinstance(column.get("physical_name"), str):
                    raise UnknownReferenceError(
                        f"schema={schema_id}: unknown logical column {entry['column']}"
                    )
                item = dict(entry)
                item["physical_name"] = column["physical_name"]
                rendered.append(item)
            resolved["columns"] = rendered
        compiled[str(schema_id)] = resolved
    return SchemaRegistry(compiled)


def contract_registry(registry: Mapping[str, object]) -> ContractRegistry:
    schemas = registry.get("schemas")
    artifacts = registry.get("artifacts")
    if not isinstance(schemas, dict) or not isinstance(artifacts, dict):
        raise UnknownReferenceError("compiled registry lacks schemas or artifacts")
    return ContractRegistry(SchemaRegistry(schemas), artifacts)
