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
            minimum_items = int(spec.get("min_items", 0))
            if len(value) < minimum_items:
                raise ValueError(f"artifact={artifact_id}: at least {minimum_items} items required")
            required_item_keys = [str(key) for key in spec.get("required_item_keys", [])]
            for index, item in enumerate(value):
                if spec.get("item_type") == "object" and not isinstance(item, dict):
                    raise ValueError(f"artifact={artifact_id}: item={index} must be an object")
                if isinstance(item, dict):
                    missing = [key for key in required_item_keys if key not in item]
                    if missing:
                        raise ValueError(
                            f"artifact={artifact_id}: item={index} missing required keys={missing}"
                        )
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

        # Collect which columns the schema expects to be string dtype so we
        # can normalise them to object before Pandera sees the DataFrame.
        # pandas/PyArrow expose multiple StringDtype variants with identical
        # repr but unequal __eq__ (StringDtype(na_value=nan) vs
        # StringDtype(na_value=<NA>)), causing Pandera's strict dtype check to
        # fail spuriously.  Casting to object is lossless for pure-string data
        # and PA.String maps to object in pandera, so the check becomes exact.
        _STRING_PROTOCOL_DTYPES = frozenset({"string", "string[python]", "string[pyarrow]"})
        string_cols: set[str] = set()
        for raw in entry_objects:
            entry = _column_entry(raw)
            if str(entry.get("dtype", "")) in _STRING_PROTOCOL_DTYPES:
                string_cols.add(str(entry["physical_name"]))

        # Normalise: cast any pandas/arrow string-backed variant to object so
        # Pandera sees a single, unambiguous dtype.  Three distinct objects all
        # represent "string" but require different detection strategies:
        #   - pd.StringDtype(na_value=nan)  → str() gives "<StringDtype(...)>",
        #     NOT "string"; must use isinstance(pd.StringDtype).
        #   - pd.StringDtype(na_value=<NA>) → str() gives "string[pyarrow]";
        #     also caught by isinstance(pd.StringDtype).
        #   - pd.ArrowDtype(pa.string())    → str() gives "string[pyarrow]";
        #     different class, NOT caught by isinstance(pd.StringDtype).
        # Combining both checks handles all variants robustly.
        df = cast(pd.DataFrame, value)
        if string_cols:
            cast_map = {
                col: "object"
                for col in string_cols
                if col in df.columns
                and (
                    isinstance(df[col].dtype, pd.StringDtype)
                    or str(df[col].dtype).startswith("string")
                )
            }
            if cast_map:
                df = df.astype(cast_map)

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
            if df.duplicated(physical).any():
                raise ValueError(f"contract: uniqueness violated for {physical}")
        schema = PA.DataFrameSchema(
            columns,
            strict=strict,
            coerce=bool(spec.get("coerce", False)),
        )
        schema.validate(df, lazy=True)


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
    """Map protocol dtypes to stable semantic Pandera dtypes.

    Pandas/PyArrow can expose multiple internal StringDtype variants that
    render identically as ``string[pyarrow]`` but compare as unequal when
    one is pandas-native (StringDtype with na_value=nan, created by
    pd.DataFrame(list_of_dicts)) and the other is the pyarrow-backed variant
    (StringDtype with na_value=<NA>, created by pandas read_csv method with
    dtype='string' or PA.String in pandera>=0.20).
    """
    mapping: dict[str, object] = {
        # String columns are pre-cast to object dtype in _dataframe() before
        # Pandera sees the DataFrame (to sidestep the string[pyarrow] storage-
        # backend ambiguity).  PA.String in pandera>=0.20 resolves to
        # string[pyarrow], which no longer matches object.  PA.Object
        # (numpy dtype object) is the correct counterpart after the cast.
        "string": PA.Object,
        "string[python]": PA.Object,
        "string[pyarrow]": PA.Object,
        "int16": PA.Int16,
        "int64": PA.Int64,
        "float64": PA.Float64,
        "bool": PA.Bool,
        "boolean": pd.BooleanDtype(),
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
