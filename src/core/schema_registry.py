"""Executable contracts compiled from YAML; Pandera is isolated to this boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import pandera.pandas as pa

from .errors import UnknownReferenceError

PA: Any = pa


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
        if not isinstance(artifact, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: unknown artifact")
        artifact_spec = cast(dict[str, Any], artifact)
        if not isinstance(artifact_spec.get("schema_id"), str):
            raise UnknownReferenceError(f"artifact={artifact_id}: unknown artifact")
        specification = self.schemas.get(str(artifact_spec["schema_id"]))
        if not isinstance(specification, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: malformed schema")
        spec = cast(dict[str, Any], specification)
        kind = spec.get("contract_type")
        if kind == "dataframe":
            self._dataframe(spec, value)
        elif kind in {"json_object", "receipt"}:
            if not isinstance(value, dict):
                raise ValueError(f"artifact={artifact_id}: JSON object required")
            self._required_keys(spec, cast(dict[object, object], value))
        elif kind == "json_array":
            if not isinstance(value, list):
                raise ValueError(f"artifact={artifact_id}: JSON array required")
            if spec.get("item_type") == "object" and not all(
                isinstance(item, dict) for item in cast(list[object], value)
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
        if not isinstance(specification, dict):
            raise UnknownReferenceError(f"schema={schema_id}: version required")
        spec = cast(dict[str, Any], specification)
        if not isinstance(spec.get("version"), int):
            raise UnknownReferenceError(f"schema={schema_id}: version required")
        return int(spec["version"])

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
        entry_objects = cast(list[object], entries)
        names = [str(_column_entry(entry)["physical_name"]) for entry in entry_objects]
        strict = bool(spec.get("strict_columns", True))
        if strict and list(value.columns) != names:
            raise ValueError(f"contract: exact ordered columns required {names}")
        if not strict and list(value.columns)[: len(names)] != names:
            raise ValueError("contract: ordered required columns differ")

        columns: dict[str, Any] = {}
        for raw in entry_objects:
            entry = _column_entry(raw)
            name = str(entry["physical_name"])
            checks = _column_checks(spec, entry)
            columns[name] = PA.Column(
                _pandera_dtype(str(entry["dtype"])),
                nullable=bool(entry.get("nullable", True)),
                checks=checks,
                coerce=bool(spec.get("coerce", False)),
            )
        constraints = spec.get("uniqueness_constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("contract: uniqueness_constraints must be a list")
        for key in cast(list[object], constraints):
            if not isinstance(key, list):
                raise ValueError("contract: uniqueness constraint must be a list")
            physical = [_physical(spec, logical) for logical in cast(list[object], key)]
            if value.duplicated(physical).any():
                raise ValueError(f"contract: uniqueness violated for {physical}")
        schema = PA.DataFrameSchema(
            columns,
            strict=strict,
            coerce=bool(spec.get("coerce", False)),
        )
        schema.validate(value, lazy=True)


def _column_entry(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("contract: column mapping required")
    return cast(dict[str, Any], value)


def _column_checks(spec: Mapping[str, Any], entry: Mapping[str, Any]) -> list[Any]:
    checks: list[Any] = []
    logical = entry.get("column")
    raw_checks = spec.get("row_checks", [])
    if not isinstance(raw_checks, list):
        raise ValueError("contract: row_checks must be a list")
    for raw in cast(list[object], raw_checks):
        if not isinstance(raw, dict):
            continue
        raw_check = cast(dict[str, Any], raw)
        if raw_check.get("column") != logical:
            continue
        check = raw_check.get("check")
        if check == "between":
            checks.append(PA.Check.between(raw_check["min"], raw_check["max"]))
        elif check == "greater_than":
            checks.append(PA.Check.greater_than(raw_check["min"]))
        elif check == "greater_than_or_equal":
            checks.append(PA.Check.greater_than_or_equal_to(raw_check["min"]))
    return checks


def _physical(spec: Mapping[str, Any], logical: object) -> str:
    columns = spec.get("columns", [])
    if not isinstance(columns, list):
        raise ValueError("contract: columns required")
    for entry in cast(list[object], columns):
        if not isinstance(entry, dict):
            continue
        column = cast(dict[str, Any], entry)
        if column.get("column") == logical:
            return str(column["physical_name"])
    raise ValueError(f"contract: unknown logical column {logical}")


def _pandera_dtype(dtype: str) -> object:
    mapping: dict[str, object] = {
        "string": PA.String,
        "int16": PA.Int16,
        "int64": PA.Int64,
        "float64": PA.Float64,
        "bool": PA.Bool,
        "datetime64[ns]": PA.DateTime,
    }
    return mapping.get(dtype, dtype)


def compile_schemas(raw: object, columns: object) -> SchemaRegistry:
    if not isinstance(raw, dict) or not isinstance(columns, dict):
        raise UnknownReferenceError("schemas and columns must be mappings")
    compiled: dict[str, object] = {}
    raw_schemas = cast(dict[str, Any], raw)
    column_registry = cast(dict[str, Any], columns)
    for schema_id, raw_spec in raw_schemas.items():
        if not isinstance(raw_spec, dict):
            raise UnknownReferenceError(f"schema={schema_id}: contract_type required")
        spec = cast(dict[str, Any], raw_spec)
        if not isinstance(spec.get("contract_type"), str):
            raise UnknownReferenceError(f"schema={schema_id}: contract_type required")
        resolved = dict(spec)
        if spec["contract_type"] == "dataframe":
            rendered: list[dict[str, object]] = []
            raw_entries = spec.get("columns", [])
            if not isinstance(raw_entries, list):
                raise UnknownReferenceError(f"schema={schema_id}: columns required")
            for raw_entry in cast(list[object], raw_entries):
                if not isinstance(raw_entry, dict):
                    raise UnknownReferenceError(f"schema={schema_id}: logical column required")
                entry = cast(dict[str, Any], raw_entry)
                if not isinstance(entry.get("column"), str):
                    raise UnknownReferenceError(f"schema={schema_id}: logical column required")
                column_name = str(entry["column"])
                column = column_registry.get(column_name)
                if not isinstance(column, dict):
                    raise UnknownReferenceError(
                        f"schema={schema_id}: unknown logical column {column_name}"
                    )
                column_spec = cast(dict[str, Any], column)
                if not isinstance(column_spec.get("physical_name"), str):
                    raise UnknownReferenceError(
                        f"schema={schema_id}: unknown logical column {column_name}"
                    )
                item = dict(entry)
                item["physical_name"] = column_spec["physical_name"]
                rendered.append(item)
            resolved["columns"] = rendered
        compiled[str(schema_id)] = resolved
    return SchemaRegistry(compiled)


def contract_registry(registry: Mapping[str, object]) -> ContractRegistry:
    schemas = registry.get("schemas")
    artifacts = registry.get("artifacts")
    if not isinstance(schemas, dict) or not isinstance(artifacts, dict):
        raise UnknownReferenceError("compiled registry lacks schemas or artifacts")
    return ContractRegistry(
        SchemaRegistry(cast(dict[str, object], schemas)),
        cast(dict[str, object], artifacts),
    )
