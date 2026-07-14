"""Executable artifact contracts compiled once during P00."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

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
        specification = self.schemas.get(artifact["schema_id"])
        if not isinstance(specification, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: malformed schema")
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

    @staticmethod
    def _required_keys(specification: dict[str, object], value: dict[object, object]) -> None:
        missing = [key for key in specification.get("required_keys", []) if key not in value]
        if missing:
            raise ValueError(f"contract: missing required keys {missing}")

    @staticmethod
    def _dataframe(specification: dict[str, object], value: object) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("contract: dataframe value required")
        columns = specification.get("columns")
        if not isinstance(columns, list):
            raise ValueError("contract: dataframe columns required")
        names = [item["physical_name"] for item in columns if isinstance(item, dict)]
        if list(value.columns)[: len(names)] != names:
            raise ValueError("contract: ordered logical columns do not match")
        for entry in columns:
            if not isinstance(entry, dict):
                continue
            name = entry["physical_name"]
            if not entry.get("nullable", True) and value[name].isna().any():
                raise ValueError(f"contract: non-null column {name} contains null")
            expected = entry.get("dtype")
            if isinstance(expected, str) and str(value[name].dtype) != expected:
                raise ValueError(f"contract: column {name} dtype differs from {expected}")
        for key in specification.get("uniqueness_constraints", []):
            physical = [_physical(specification, item) for item in key]
            if value.duplicated(physical).any():
                raise ValueError("contract: uniqueness constraint violated")
        for logical, permitted in dict(specification.get("allowed_values", {})).items():
            name = _physical(specification, logical)
            if not value[name].dropna().isin(permitted).all():
                raise ValueError(f"contract: allowed values violated for {name}")


def _physical(specification: dict[str, object], logical: object) -> str:
    for entry in specification.get("columns", []):
        if isinstance(entry, dict) and entry.get("column") == logical:
            return str(entry["physical_name"])
    raise ValueError(f"contract: unknown logical column {logical}")


def compile_schemas(raw: object, columns: object) -> SchemaRegistry:
    if not isinstance(raw, dict) or not isinstance(columns, dict):
        raise UnknownReferenceError("schemas and columns must be mappings")
    compiled: dict[str, object] = {}
    for schema_id, specification in raw.items():
        if not isinstance(specification, dict) or not isinstance(
            specification.get("contract_type"), str
        ):
            raise UnknownReferenceError(f"schema={schema_id}: contract_type required")
        resolved = dict(specification)
        if specification["contract_type"] == "dataframe":
            entries = specification.get("columns")
            if not isinstance(entries, list) or not entries:
                raise UnknownReferenceError(f"schema={schema_id}: ordered columns required")
            rendered: list[dict[str, object]] = []
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("column"), str):
                    raise UnknownReferenceError(f"schema={schema_id}: logical column required")
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
