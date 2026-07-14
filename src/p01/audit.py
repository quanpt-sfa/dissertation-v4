"""Pure P01 audit functions and orchestration."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import AuditIssue, AuditStatus, RawAuditReport, SourceSpec
from .readers import inspect_file_signature, iter_rows, read_header


@dataclass
class DateFieldAudit:
    nonmissing: int = 0
    parsed: int = 0
    invalid: int = 0
    minimum: str | None = None
    maximum: str | None = None
    sample_invalid_rows: list[int] = field(default_factory=lambda: list[int]())

    def to_dict(self) -> dict[str, object]:
        return {
            "nonmissing": self.nonmissing,
            "parsed": self.parsed,
            "invalid": self.invalid,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "sample_invalid_rows": list(self.sample_invalid_rows),
        }


@dataclass
class NumericFieldAudit:
    unit: str
    nonmissing: int = 0
    parsed: int = 0
    invalid: int = 0
    negative: int = 0
    sample_invalid_rows: list[int] = field(default_factory=lambda: list[int]())

    def to_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "nonmissing": self.nonmissing,
            "parsed": self.parsed,
            "invalid": self.invalid,
            "negative": self.negative,
            "sample_invalid_rows": list(self.sample_invalid_rows),
        }


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def _parse_date(value: object) -> datetime | None:
    if _is_missing(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)

    text = str(value).strip()
    normalized = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for date_format in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            continue
    return None


def _parse_decimal(
    value: object,
    decimal_mark: str,
    thousands_separator: str | None,
) -> Decimal | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if thousands_separator:
        text = text.replace(thousands_separator, "")
    if decimal_mark == ",":
        text = text.replace(".", "").replace(",", ".")
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def classify_data_issue(
    code: str,
    message: str,
    *,
    fatal: bool = False,
    isolate: bool = False,
    field_name: str | None = None,
    count: int | None = None,
    sample_rows: Iterable[int] = (),
) -> AuditIssue:
    severity = "FAIL" if fatal else "ISOLATE" if isolate else "WARN"
    return AuditIssue(
        code=code,
        severity=severity,
        message=message,
        field=field_name,
        count=count,
        sample_rows=tuple(sample_rows),
    )


def audit_source(
    *,
    run_id: str,
    protocol_hash: str,
    source_path: Path,
    spec: SourceSpec,
) -> RawAuditReport:
    issues: list[AuditIssue] = []
    signature = inspect_file_signature(source_path, spec)

    if signature["hash_matches_lock"] is not True:
        issues.append(
            classify_data_issue(
                "SOURCE_HASH_CHANGED_AFTER_LOCK",
                "Raw file SHA-256 differs from the locked source hash; "
                "a new protocol run is required.",
                fatal=True,
            )
        )

    columns = read_header(source_path, spec)
    actual = set(columns)
    required = set(spec.schema.required_columns)
    missing_required = sorted(required - actual)
    extra_columns = sorted(actual - spec.schema.registered_columns)

    if missing_required:
        issues.append(
            classify_data_issue(
                "REQUIRED_COLUMNS_MISSING",
                f"Required columns are absent: {missing_required}",
                fatal=True,
                count=len(missing_required),
            )
        )

    if extra_columns and not spec.schema.allow_extra_columns:
        issues.append(
            classify_data_issue(
                "UNREGISTERED_SOURCE_FIELDS",
                f"Unregistered fields are present: {extra_columns}",
                fatal=True,
                count=len(extra_columns),
            )
        )

    row_count = 0
    nonmissing: Counter[str] = Counter({column: 0 for column in columns})
    missing_required_values: Counter[str] = Counter()
    missing_key_values: Counter[str] = Counter()
    row_extra_fields: Counter[str] = Counter()
    key_counts: Counter[tuple[str, ...]] = Counter()
    duplicate_examples: list[dict[str, object]] = []
    date_stats = {column: DateFieldAudit() for column in spec.schema.date_columns}
    numeric_stats = {
        column: NumericFieldAudit(unit=numeric_spec.unit)
        for column, numeric_spec in spec.schema.numeric_columns.items()
    }

    for row_number, row in enumerate(iter_rows(source_path, spec), start=1):
        row_count += 1

        for column in columns:
            if not _is_missing(row.get(column)):
                nonmissing[column] += 1

        for column in set(row) - spec.schema.registered_columns:
            row_extra_fields[column] += 1
        for column in spec.schema.required_columns:
            if _is_missing(row.get(column)):
                missing_required_values[column] += 1
        for column in spec.schema.key_columns:
            if _is_missing(row.get(column)):
                missing_key_values[column] += 1

        if spec.schema.key_columns and all(
            not _is_missing(row.get(column)) for column in spec.schema.key_columns
        ):
            key = tuple(
                "" if _is_missing(row.get(column)) else str(row.get(column))
                for column in spec.schema.key_columns
            )
            key_counts[key] += 1

        for column in spec.schema.date_columns:
            value = row.get(column)
            if _is_missing(value):
                continue

            stats = date_stats[column]
            stats.nonmissing += 1
            parsed = _parse_date(value)

            if parsed is None:
                stats.invalid += 1
                if len(stats.sample_invalid_rows) < 20:
                    stats.sample_invalid_rows.append(row_number)
                continue

            stats.parsed += 1
            iso = parsed.isoformat()
            if stats.minimum is None or iso < stats.minimum:
                stats.minimum = iso
            if stats.maximum is None or iso > stats.maximum:
                stats.maximum = iso

        for column, numeric_spec in spec.schema.numeric_columns.items():
            value = row.get(column)
            if _is_missing(value):
                continue

            stats = numeric_stats[column]
            stats.nonmissing += 1
            parsed = _parse_decimal(
                value,
                numeric_spec.decimal_mark,
                numeric_spec.thousands_separator,
            )

            if parsed is None:
                stats.invalid += 1
                if len(stats.sample_invalid_rows) < 20:
                    stats.sample_invalid_rows.append(row_number)
                continue

            stats.parsed += 1
            if parsed < 0:
                stats.negative += 1

    if row_extra_fields and not spec.schema.allow_extra_columns:
        issues.append(
            classify_data_issue(
                "UNREGISTERED_SOURCE_FIELDS_IN_ROWS",
                f"Unregistered row fields are present: {sorted(row_extra_fields)}",
                fatal=True,
                count=sum(row_extra_fields.values()),
            )
        )

    for column, count in sorted(missing_key_values.items()):
        if count:
            issues.append(
                classify_data_issue(
                    "MISSING_KEY_VALUES",
                    f"{count} rows have a missing registered key value.",
                    fatal=True,
                    field_name=column,
                    count=count,
                )
            )

    for column, count in sorted(missing_required_values.items()):
        if not count or column in spec.schema.key_columns:
            continue
        issues.append(
            classify_data_issue(
                "MISSING_REQUIRED_DATE_VALUES"
                if column in spec.schema.required_date_columns
                else "MISSING_REQUIRED_VALUES",
                f"{count} rows have a missing required value.",
                isolate=True,
                field_name=column,
                count=count,
            )
        )

    if row_count < spec.schema.row_count_min:
        issues.append(
            classify_data_issue(
                "ROW_COUNT_BELOW_MINIMUM",
                f"Observed {row_count} rows; minimum is {spec.schema.row_count_min}.",
                fatal=True,
                count=row_count,
            )
        )

    duplicate_rows = sum(count for count in key_counts.values() if count > 1)
    duplicate_groups = sum(1 for count in key_counts.values() if count > 1)

    for key, count in key_counts.items():
        if count > 1 and len(duplicate_examples) < 20:
            duplicate_examples.append(
                {
                    "key": dict(
                        zip(
                            spec.schema.key_columns,
                            key,
                            strict=True,
                        )
                    ),
                    "count": count,
                }
            )

    if duplicate_groups and spec.schema.key_unique:
        issues.append(
            classify_data_issue(
                "DUPLICATE_KEYS",
                f"{duplicate_groups} duplicate key groups were detected.",
                fatal=True,
                count=duplicate_groups,
            )
        )

    for column, stats in date_stats.items():
        if stats.invalid:
            issues.append(
                classify_data_issue(
                    "INVALID_DATE_VALUES",
                    f"{stats.invalid} values could not be parsed as dates.",
                    fatal=column in spec.schema.required_date_columns,
                    isolate=column not in spec.schema.required_date_columns,
                    field_name=column,
                    count=stats.invalid,
                    sample_rows=stats.sample_invalid_rows,
                )
            )

    for column, stats in numeric_stats.items():
        if stats.invalid:
            issues.append(
                classify_data_issue(
                    "INVALID_NUMERIC_VALUES",
                    f"{stats.invalid} values could not be parsed as numeric.",
                    isolate=True,
                    field_name=column,
                    count=stats.invalid,
                    sample_rows=stats.sample_invalid_rows,
                )
            )

        numeric_spec = spec.schema.numeric_columns[column]
        if stats.negative and not numeric_spec.allow_negative:
            issues.append(
                classify_data_issue(
                    "NEGATIVE_VALUES_NOT_ALLOWED",
                    f"{stats.negative} negative values violate the registered unit rule.",
                    isolate=True,
                    field_name=column,
                    count=stats.negative,
                )
            )

    if any(issue.severity == "FAIL" for issue in issues):
        status: AuditStatus = "FAILED"
    elif issues:
        status = "PASS_WITH_ISSUES"
    else:
        status = "PASS"

    coverage_columns: dict[str, object] = {
        column: {
            "nonmissing_count": nonmissing[column],
            "row_count": row_count,
            "coverage_rate": (nonmissing[column] / row_count if row_count else None),
        }
        for column in columns
    }
    date_audit: dict[str, object] = {
        column: stats.to_dict() for column, stats in date_stats.items()
    }
    unit_audit: dict[str, object] = {
        column: stats.to_dict() for column, stats in numeric_stats.items()
    }

    return RawAuditReport(
        run_id=run_id,
        source_id=spec.source_id,
        protocol_hash=protocol_hash,
        audited_at_utc=datetime.now(UTC).isoformat(),
        status=status,
        source_metadata={
            "channel_id": spec.channel_id,
            "source_type": spec.source_type,
            "source_agency": spec.source_agency,
            "original_unit": spec.original_unit,
            "related_period_field": spec.related_period_field,
            "availability_date_field": spec.availability_date_field,
            "availability_date_source": spec.availability_date_source,
            "coverage_dimensions": list(spec.coverage_dimensions),
            "role": spec.role,
            "verification_status": spec.verification_status,
            "data_risks": list(spec.data_risks),
        },
        file_signature=signature,
        schema_audit={
            "actual_columns": columns,
            "required_columns": list(spec.schema.required_columns),
            "optional_columns": list(spec.schema.optional_columns),
            "missing_required_columns": missing_required,
            "extra_columns": extra_columns,
            "allow_extra_columns": spec.schema.allow_extra_columns,
            "missing_required_value_counts": dict(sorted(missing_required_values.items())),
            "missing_key_value_counts": dict(sorted(missing_key_values.items())),
            "unregistered_row_field_counts": dict(sorted(row_extra_fields.items())),
        },
        duplicate_key_audit={
            "key_columns": list(spec.schema.key_columns),
            "key_unique": spec.schema.key_unique,
            "duplicate_group_count": duplicate_groups,
            "duplicate_row_count": duplicate_rows,
            "examples": duplicate_examples,
        },
        date_audit=date_audit,
        unit_audit=unit_audit,
        coverage={
            "row_count": row_count,
            "columns": coverage_columns,
            "coverage_dimensions": list(spec.coverage_dimensions),
        },
        issues=issues,
        decision={
            "status": status,
            "pipeline_may_advance": status != "FAILED",
            "new_run_required": any(
                issue.code == "SOURCE_HASH_CHANGED_AFTER_LOCK" for issue in issues
            ),
            "isolated_issue_count": sum(issue.severity == "ISOLATE" for issue in issues),
            "fatal_issue_count": sum(issue.severity == "FAIL" for issue in issues),
        },
    )
