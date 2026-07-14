"""Typed P01 source specifications and audit results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

IssueSeverity = Literal["FAIL", "ISOLATE", "WARN", "INFO"]
AuditStatus = Literal["PASS", "PASS_WITH_ISSUES", "FAILED"]
SourceFormat = Literal["csv", "tsv", "parquet", "json", "jsonl"]


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: non-empty string required")
    return value.strip()


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: list required")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise ValueError(f"{context}: non-empty strings required")
    return [cast(str, item).strip() for item in items]


@dataclass(frozen=True)
class NumericFieldSpec:
    unit: str
    allow_negative: bool = True
    decimal_mark: str = "."
    thousands_separator: str | None = None

    @classmethod
    def from_mapping(cls, value: object, context: str) -> NumericFieldSpec:
        raw = _mapping(value, context)
        unit = _string(raw.get("unit"), f"{context}.unit")
        allow_negative = raw.get("allow_negative", True)
        if not isinstance(allow_negative, bool):
            raise ValueError(f"{context}.allow_negative: bool required")
        decimal_mark = raw.get("decimal_mark", ".")
        if not isinstance(decimal_mark, str) or decimal_mark not in {".", ","}:
            raise ValueError(f"{context}.decimal_mark: '.' or ',' required")
        thousands = raw.get("thousands_separator")
        if thousands is not None and not isinstance(thousands, str):
            raise ValueError(f"{context}.thousands_separator: string or null required")
        return cls(
            unit=unit,
            allow_negative=allow_negative,
            decimal_mark=decimal_mark,
            thousands_separator=thousands,
        )


@dataclass(frozen=True)
class RawSchemaSpec:
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    date_columns: tuple[str, ...]
    required_date_columns: tuple[str, ...]
    numeric_columns: dict[str, NumericFieldSpec]
    allow_extra_columns: bool
    key_unique: bool
    row_count_min: int

    @classmethod
    def from_mapping(cls, value: object, context: str) -> RawSchemaSpec:
        raw = _mapping(value, context)
        numeric_raw = _mapping(raw.get("numeric_columns", {}), f"{context}.numeric_columns")
        numeric = {
            field_name: NumericFieldSpec.from_mapping(
                spec, f"{context}.numeric_columns.{field_name}"
            )
            for field_name, spec in numeric_raw.items()
        }
        row_count_min = raw.get("row_count_min", 1)
        if not isinstance(row_count_min, int) or row_count_min < 0:
            raise ValueError(f"{context}.row_count_min: non-negative integer required")
        allow_extra = raw.get("allow_extra_columns", False)
        key_unique = raw.get("key_unique", True)
        if not isinstance(allow_extra, bool) or not isinstance(key_unique, bool):
            raise ValueError(f"{context}: boolean schema flags required")
        return cls(
            required_columns=tuple(
                _string_list(raw.get("required_columns", []), f"{context}.required_columns")
            ),
            optional_columns=tuple(
                _string_list(raw.get("optional_columns", []), f"{context}.optional_columns")
            ),
            key_columns=tuple(_string_list(raw.get("key_columns", []), f"{context}.key_columns")),
            date_columns=tuple(
                _string_list(raw.get("date_columns", []), f"{context}.date_columns")
            ),
            required_date_columns=tuple(
                _string_list(
                    raw.get("required_date_columns", []),
                    f"{context}.required_date_columns",
                )
            ),
            numeric_columns=numeric,
            allow_extra_columns=allow_extra,
            key_unique=key_unique,
            row_count_min=row_count_min,
        )

    @property
    def registered_columns(self) -> set[str]:
        return {
            *self.required_columns,
            *self.optional_columns,
            *self.key_columns,
            *self.date_columns,
            *self.numeric_columns,
        }


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    enabled: bool
    channel_id: str
    source_type: str
    source_agency: str
    original_unit: str
    related_period_field: str | None
    availability_date_field: str
    availability_date_source: str
    coverage_dimensions: tuple[str, ...]
    role: str
    verification_status: str
    data_risks: tuple[str, ...]
    relative_path: str
    format: SourceFormat
    encoding: str
    delimiter: str | None
    locked_sha256: str
    schema: RawSchemaSpec

    @classmethod
    def from_mapping(cls, source_id: str, value: object) -> SourceSpec:
        raw = _mapping(value, f"source={source_id}")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"source={source_id}.enabled: bool required")
        format_value = _string(raw.get("format"), f"source={source_id}.format").lower()
        if format_value not in {"csv", "tsv", "parquet", "json", "jsonl"}:
            raise ValueError(f"source={source_id}.format: unsupported format {format_value}")
        delimiter = raw.get("delimiter")
        if delimiter is not None and (not isinstance(delimiter, str) or len(delimiter) != 1):
            raise ValueError(f"source={source_id}.delimiter: one character or null required")
        related_period = raw.get("related_period_field")
        if related_period is not None and not isinstance(related_period, str):
            raise ValueError(f"source={source_id}.related_period_field: string or null required")
        locked_hash = _string(raw.get("locked_sha256"), f"source={source_id}.locked_sha256").lower()
        if len(locked_hash) != 64 or any(char not in "0123456789abcdef" for char in locked_hash):
            raise ValueError(f"source={source_id}.locked_sha256: lowercase SHA-256 required")
        return cls(
            source_id=source_id,
            enabled=enabled,
            channel_id=_string(raw.get("channel_id"), f"source={source_id}.channel_id"),
            source_type=_string(raw.get("source_type"), f"source={source_id}.source_type"),
            source_agency=_string(raw.get("source_agency"), f"source={source_id}.source_agency"),
            original_unit=_string(raw.get("original_unit"), f"source={source_id}.original_unit"),
            related_period_field=related_period,
            availability_date_field=_string(
                raw.get("availability_date_field"),
                f"source={source_id}.availability_date_field",
            ),
            availability_date_source=_string(
                raw.get("availability_date_source"),
                f"source={source_id}.availability_date_source",
            ),
            coverage_dimensions=tuple(
                _string_list(
                    raw.get("coverage_dimensions", []),
                    f"source={source_id}.coverage_dimensions",
                )
            ),
            role=_string(raw.get("role"), f"source={source_id}.role"),
            verification_status=_string(
                raw.get("verification_status"),
                f"source={source_id}.verification_status",
            ),
            data_risks=tuple(
                _string_list(raw.get("data_risks", []), f"source={source_id}.data_risks")
            ),
            relative_path=_string(raw.get("relative_path"), f"source={source_id}.relative_path"),
            format=cast(SourceFormat, format_value),
            encoding=_string(raw.get("encoding", "utf-8"), f"source={source_id}.encoding"),
            delimiter=delimiter,
            locked_sha256=locked_hash,
            schema=RawSchemaSpec.from_mapping(raw.get("schema"), f"source={source_id}.schema"),
        )


@dataclass(frozen=True)
class AuditIssue:
    code: str
    severity: IssueSeverity
    message: str
    field: str | None = None
    count: int | None = None
    sample_rows: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RawAuditReport:
    run_id: str
    source_id: str
    protocol_hash: str
    audited_at_utc: str
    status: AuditStatus
    source_metadata: dict[str, object]
    file_signature: dict[str, object]
    schema_audit: dict[str, object]
    duplicate_key_audit: dict[str, object]
    date_audit: dict[str, object]
    unit_audit: dict[str, object]
    coverage: dict[str, object]
    issues: list[AuditIssue]
    decision: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "source_id": self.source_id,
            "protocol_hash": self.protocol_hash,
            "audited_at_utc": self.audited_at_utc,
            "status": self.status,
            "source_metadata": self.source_metadata,
            "file_signature": self.file_signature,
            "schema_audit": self.schema_audit,
            "duplicate_key_audit": self.duplicate_key_audit,
            "date_audit": self.date_audit,
            "unit_audit": self.unit_audit,
            "coverage": self.coverage,
            "issues": [issue.to_dict() for issue in self.issues],
            "decision": self.decision,
        }
