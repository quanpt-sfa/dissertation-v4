"""Executable artifact contracts compiled once during P00."""

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
            raise UnknownReferenceError(f"schema={schema_id}: unknown schema ID") from exc


@dataclass(frozen=True)
class ContractRegistry:
    schemas: SchemaRegistry
    artifacts: Mapping[str, object]

    def validate(self, artifact_id: str, value: object) -> None:
        artifact = self.artifacts.get(artifact_id)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("schema_id"), str):
            raise UnknownReferenceError(f"artifact={artifact_id}: unknown artifact ID")
        artifact = cast(dict[str, Any], artifact)
        specification = self.schemas.get(str(artifact["schema_id"]))
        if not isinstance(specification, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: malformed schema")
        specification = cast(dict[str, Any], specification)
        kind = specification.get("contract_type")
        if kind == "dataframe":
            self._dataframe(specification, value)
        elif kind == "json_object":
            if not isinstance(value, dict):
                raise ValueError(f"artifact={artifact_id}: JSON object required")
            self._required_keys(specification, value)
        elif kind == "json_array":
            if not isinstance(value, list):
                raise ValueError(f"artifact={artifact_id}: JSON array required")
        elif kind == "text":
            if not isinstance(value, str) or len(value) < int(specification.get("min_length", 0)):
                raise ValueError(f"artifact={artifact_id}: non-empty text required")
        elif kind == "markdown":
            prefix = specification.get("required_prefix")
            if not isinstance(value, str) or (
                isinstance(prefix, str) and not value.startswith(prefix)
            ):
                raise ValueError(f"artifact={artifact_id}: Markdown heading required")
        elif kind == "receipt":
            if not isinstance(value, dict):
                raise ValueError(f"artifact={artifact_id}: receipt object required")
            self._required_keys(specification, value)
        else:
            raise ValueError(f"artifact={artifact_id}: unsupported contract type")

    def schema_version(self, schema_id: str) -> int:
        specification = self.schemas.get(schema_id)
        if not isinstance(specification, dict) or not isinstance(specification.get("version"), int):
            raise UnknownReferenceError(f"schema={schema_id}: version required")
        return int(specification["version"])

    @staticmethod
    def _required_keys(specification: dict[str, object], value: dict[object, object]) -> None:
        keys = [str(key) for key in _as_list(specification.get("required_keys", []))]
        missing = [key for key in keys if key not in value]
        if missing:
            raise ValueError(f"contract: missing required keys {missing}")

    @staticmethod
    def _dataframe(specification: dict[str, object], value: object) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("contract: dataframe value required")
        columns = specification.get("columns")
        if not isinstance(columns, list):
            raise ValueError("contract: dataframe columns required")
        entries = [_as_dict(item) for item in columns]
        names = [str(item["physical_name"]) for item in entries]
        if list(value.columns)[: len(names)] != names:
            raise ValueError("contract: ordered logical columns do not match")
        pandera_columns: dict[str, pa.Column] = {}
        for entry in entries:
            name = entry["physical_name"]
            if not isinstance(name, str):
                raise ValueError("contract: physical column name must be a string")
            checks = _column_checks(specification, entry)
            pandera_columns[name] = pa.Column(
                _pandera_dtype(str(entry.get("dtype", "object"))),
                nullable=bool(entry.get("nullable", True)),
                checks=checks,
                coerce=True,
            )
            if not entry.get("nullable", True) and value[name].isna().any():
                raise ValueError(f"contract: non-null column {name} contains null")
            expected = entry.get("dtype")
            if isinstance(expected, str):
                _validate_dtype(name, expected, value[name])
        unique: list[list[str]] = []
        for key in _as_list(specification.get("uniqueness_constraints", [])):
            physical = [_physical(specification, item) for item in _as_list(key)]
            unique.append(physical)
            if value.duplicated(physical).any():
                raise ValueError("contract: uniqueness constraint violated")
        for logical, permitted in _as_dict(specification.get("allowed_values", {})).items():
            name = _physical(specification, logical)
            permitted_values = _as_list(permitted)
            if not bool(value[name].dropna().isin(permitted_values).all()):
                raise ValueError(f"contract: allowed values violated for {name}")
        for check_raw in _as_list(specification.get("cross_column_checks", [])):
            if isinstance(check_raw, dict):
                check = cast(dict[str, Any], check_raw)
            else:
                continue
            if check.get("check") == "not_both_null":
                left = _physical(specification, check["left"])
                right = _physical(specification, check["right"])
                if bool(value[left].isna().all()) and bool(value[right].isna().all()):
                    raise ValueError("contract: cross-column not_both_null violated")
        pandera_unique = unique[0] if len(unique) == 1 else None
        pa.DataFrameSchema(pandera_columns, unique=pandera_unique, strict=True).validate(
            value, lazy=True
        )


def _physical(specification: dict[str, object], logical: object) -> str:
    for entry in _as_list(specification.get("columns", [])):
        if isinstance(entry, dict) and entry.get("column") == logical:
            return str(entry["physical_name"])
    raise ValueError(f"contract: unknown logical column {logical}")


def _column_checks(specification: dict[str, object], entry: dict[str, object]) -> list[pa.Check]:
    checks: list[pa.Check] = []
    for raw in _as_list(specification.get("row_checks", [])):
        if not isinstance(raw, dict) or raw.get("column") != entry.get("column"):
            continue
        raw = cast(dict[str, Any], raw)
        if raw.get("check") == "between":
            checks.append(pa.Check.between(raw["min"], raw["max"]))
    return checks


def _pandera_dtype(dtype: str) -> object:
    if dtype == "string":
        return pa.String
    if dtype == "int16":
        return pa.Int16
    if dtype in {"float64", "float"}:
        return pa.Float64
    return dtype


def _validate_dtype(name: str, expected: str, series: pd.Series) -> None:
    observed = str(series.dtype)
    if expected == "string":
        if observed not in {"string", "object"}:
            raise ValueError(f"contract: column {name} dtype differs from {expected}")
        return
    if observed != expected:
        raise ValueError(f"contract: column {name} dtype differs from {expected}")


def _as_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("contract: mapping expected")
    return cast(dict[str, Any], value)


def _as_list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError("contract: list expected")
    return cast(list[Any], value)


def compile_schemas(raw: object, columns: object) -> SchemaRegistry:
    if not isinstance(raw, dict) or not isinstance(columns, dict):
        raise UnknownReferenceError("schemas and columns must be mappings")
    compiled: dict[str, object] = {}
    for schema_id, specification in raw.items():
        if not isinstance(specification, dict) or not isinstance(
            specification.get("contract_type"), str
        ):
            raise UnknownReferenceError(f"schema={schema_id}: contract_type required")
        specification = cast(dict[str, Any], specification)
        resolved: dict[str, object] = dict(specification)
        if specification["contract_type"] == "dataframe":
            entries = specification.get("columns")
            if not isinstance(entries, list) or not entries:
                raise UnknownReferenceError(f"schema={schema_id}: ordered columns required")
            rendered: list[dict[str, object]] = []
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("column"), str):
                    raise UnknownReferenceError(f"schema={schema_id}: logical column required")
                entry = cast(dict[str, Any], entry)
                column = columns.get(entry["column"])
                if not isinstance(column, dict) or not isinstance(column.get("physical_name"), str):
                    raise UnknownReferenceError(f"schema={schema_id}: unknown logical column")
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
