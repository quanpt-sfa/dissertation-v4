"""Validate raw S3 and known-case source contracts before snapshot locking."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_KNOWN_CASES = {
    ("K1", "TAR", 2020),
    ("K1", "TAR", 2022),
    ("K2", "TTF", 2016),
    ("K3", "ROS", 2018),
    ("K3", "ROS", 2019),
    ("K4", "FHH", 2019),
}
EXPECTED_CASE_ID_BY_FIRM_YEAR = {
    (firm_id, fiscal_year): case_id for case_id, firm_id, fiscal_year in EXPECTED_KNOWN_CASES
}
CANONICAL_KNOWN_CASE_COLUMNS = [
    "case_id",
    "firm_id",
    "fiscal_year",
    "case_construct",
    "role",
    "training_include_flag",
    "calibration_include_flag",
    "model_selection_include_flag",
    "external_validation_include_flag",
]
KNOWN_CASE_ALIASES = {
    "case_id": ("case_id", "known_case_id"),
    "firm_id": ("firm_id", "firm_master_id", "issuer_ticker", "ticker", "code"),
    "fiscal_year": ("fiscal_year", "year", "target_fiscal_year"),
    "case_construct": ("case_construct", "construct", "case_type", "label_type"),
    "role": ("role", "validation_role", "case_role"),
    "training_include_flag": ("training_include_flag", "train_include_flag"),
    "calibration_include_flag": ("calibration_include_flag",),
    "model_selection_include_flag": ("model_selection_include_flag",),
    "external_validation_include_flag": ("external_validation_include_flag",),
}
EXPECTED_KNOWN_CASE_VALUES = {
    "case_construct": "CONFIRMED_FINANCIAL_REPORTING_CASE",
    "role": "SIMULATION_EXTERNAL_VALIDATION",
    "training_include_flag": "false",
    "calibration_include_flag": "false",
    "model_selection_include_flag": "false",
    "external_validation_include_flag": "true",
}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header required: {path}")
        return [{str(key): str(value or "") for key, value in row.items()} for row in reader]


def _bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"row={row_number} field={field}: boolean value required")


def _require_columns(rows: list[dict[str, str]], required: set[str], source: str) -> None:
    if not rows:
        raise ValueError(f"{source}: at least one row is required")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _first_value(row: dict[str, str], aliases: tuple[str, ...]) -> str:
    for alias in aliases:
        value = row.get(alias, "").strip()
        if value:
            return value
    return ""


def _validate_sanctions(rows: list[dict[str, str]]) -> dict[str, Any]:
    required = {
        "issuer_ticker",
        "document_id",
        "target_fiscal_year",
        "train_include_flag",
        "is_direct_label",
        "normalized_violation_code",
    }
    _require_columns(rows, required, "sanction panel")
    included = 0
    excluded = 0
    excluded_unresolved = 0
    eligible_unresolved = 0
    duplicate_keys: set[tuple[str, str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows, start=2):
        firm = row["issuer_ticker"].strip()
        document = row["document_id"].strip()
        target_year = row["target_fiscal_year"].strip()
        if not firm or not document:
            raise ValueError(f"sanction row={index}: issuer_ticker and document_id are required")
        include = _bool(row["train_include_flag"], "train_include_flag", index)
        direct = _bool(row["is_direct_label"], "is_direct_label", index)
        eligible = include and direct
        if eligible:
            included += 1
            if not target_year:
                eligible_unresolved += 1
        else:
            excluded += 1
            if not target_year:
                excluded_unresolved += 1
        key = (document, firm, row["normalized_violation_code"].strip())
        if key in seen:
            duplicate_keys.add(key)
        seen.add(key)
    if eligible_unresolved:
        raise ValueError(
            f"sanction panel contains {eligible_unresolved} eligible rows without target_fiscal_year"
        )
    if excluded == 0:
        raise ValueError("sanction panel must retain excluded rows for provenance audit")
    return {
        "row_count": len(rows),
        "eligible_row_count": included,
        "excluded_row_count": excluded,
        "excluded_unresolved_row_count": excluded_unresolved,
        "eligible_unresolved_row_count": eligible_unresolved,
        "duplicate_document_firm_code_count": len(duplicate_keys),
    }


def _canonicalize_known_cases(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if not rows:
        raise ValueError("known-case registry: at least one row is required")
    canonical: list[dict[str, str]] = []
    used_input_columns = {alias for aliases in KNOWN_CASE_ALIASES.values() for alias in aliases}
    extra_columns = [column for column in rows[0] if column not in used_input_columns]
    filled_fields: set[str] = set()

    for index, row in enumerate(rows, start=2):
        firm_id = _first_value(row, KNOWN_CASE_ALIASES["firm_id"]).upper()
        raw_year = _first_value(row, KNOWN_CASE_ALIASES["fiscal_year"])
        if not firm_id or not raw_year:
            raise ValueError(
                f"known-case row={index}: a firm identifier and fiscal year are required"
            )
        try:
            fiscal_year = int(raw_year)
        except ValueError as exc:
            raise ValueError(f"known-case row={index}: fiscal_year must be an integer") from exc
        expected_case_id = EXPECTED_CASE_ID_BY_FIRM_YEAR.get((firm_id, fiscal_year))
        if expected_case_id is None:
            raise ValueError(
                f"known-case row={index}: unexpected firm-year {(firm_id, fiscal_year)}"
            )
        raw_case_id = _first_value(row, KNOWN_CASE_ALIASES["case_id"])
        if raw_case_id and raw_case_id != expected_case_id:
            raise ValueError(
                f"known-case row={index}: case_id={raw_case_id} conflicts with "
                f"locked case_id={expected_case_id}"
            )
        if not raw_case_id:
            filled_fields.add("case_id")

        output: dict[str, str] = {
            "case_id": expected_case_id,
            "firm_id": firm_id,
            "fiscal_year": str(fiscal_year),
        }
        for field, expected in EXPECTED_KNOWN_CASE_VALUES.items():
            raw_value = _first_value(row, KNOWN_CASE_ALIASES[field])
            if not raw_value:
                output[field] = expected
                filled_fields.add(field)
                continue
            if field.endswith("_flag"):
                parsed = _bool(raw_value, field, index)
                expected_bool = expected == "true"
                if parsed is not expected_bool:
                    raise ValueError(
                        f"known-case row={index}: {field} conflicts with sealed external-validation role"
                    )
                output[field] = "true" if parsed else "false"
            else:
                if raw_value != expected:
                    raise ValueError(
                        f"known-case row={index}: {field}={raw_value!r} conflicts with {expected!r}"
                    )
                output[field] = raw_value
        for column in extra_columns:
            output[column] = row.get(column, "")
        canonical.append(output)

    observed = {(row["case_id"], row["firm_id"], int(row["fiscal_year"])) for row in canonical}
    if observed != EXPECTED_KNOWN_CASES:
        missing = sorted(EXPECTED_KNOWN_CASES - observed)
        unexpected = sorted(observed - EXPECTED_KNOWN_CASES)
        raise ValueError(f"known-case registry mismatch missing={missing} unexpected={unexpected}")
    if len(canonical) != len(observed):
        raise ValueError("known-case registry contains duplicate case-firm-year rows")
    return canonical, {
        "row_count": len(canonical),
        "filled_fields": sorted(filled_fields),
        "preserved_extra_columns": extra_columns,
    }


def _write_known_cases(path: Path, rows: list[dict[str, str]]) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.before_canonicalization.{timestamp}{path.suffix}")
    shutil.copy2(path, backup)
    extra_columns = [column for column in rows[0] if column not in CANONICAL_KNOWN_CASE_COLUMNS]
    fieldnames = [*CANONICAL_KNOWN_CASE_COLUMNS, *extra_columns]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return backup


def _validate_known_cases(rows: list[dict[str, str]]) -> dict[str, Any]:
    required = set(CANONICAL_KNOWN_CASE_COLUMNS)
    _require_columns(rows, required, "known-case registry")
    observed: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows, start=2):
        record = (
            row["case_id"].strip(),
            row["firm_id"].strip().upper(),
            int(row["fiscal_year"].strip()),
        )
        if record in observed:
            raise ValueError(f"known-case row={index}: duplicate case-firm-year {record}")
        observed.add(record)
        if row["case_construct"].strip() != "CONFIRMED_FINANCIAL_REPORTING_CASE":
            raise ValueError(f"known-case row={index}: invalid case_construct")
        if row["role"].strip() != "SIMULATION_EXTERNAL_VALIDATION":
            raise ValueError(f"known-case row={index}: invalid role")
        flags = {
            "training_include_flag": _bool(
                row["training_include_flag"], "training_include_flag", index
            ),
            "calibration_include_flag": _bool(
                row["calibration_include_flag"], "calibration_include_flag", index
            ),
            "model_selection_include_flag": _bool(
                row["model_selection_include_flag"], "model_selection_include_flag", index
            ),
            "external_validation_include_flag": _bool(
                row["external_validation_include_flag"],
                "external_validation_include_flag",
                index,
            ),
        }
        if flags != {
            "training_include_flag": False,
            "calibration_include_flag": False,
            "model_selection_include_flag": False,
            "external_validation_include_flag": True,
        }:
            raise ValueError(f"known-case row={index}: invalid inclusion flags")
    if observed != EXPECTED_KNOWN_CASES:
        missing = sorted(EXPECTED_KNOWN_CASES - observed)
        unexpected = sorted(observed - EXPECTED_KNOWN_CASES)
        raise ValueError(f"known-case registry mismatch missing={missing} unexpected={unexpected}")
    return {
        "row_count": len(rows),
        "unique_case_count": len({case_id for case_id, _, _ in observed}),
        "firm_year_count": len(observed),
        "case_ids": sorted({case_id for case_id, _, _ in observed}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--normalize-known-cases",
        action="store_true",
        help="Canonicalize a legacy known_case_registry.csv before strict validation.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the canonical registry and create a timestamped backup.",
    )
    args = parser.parse_args()

    if args.write and not args.normalize_known_cases:
        parser.error("--write requires --normalize-known-cases")

    root = args.raw_root.expanduser().resolve()
    sanction_path = root / "data" / "source" / "firm_event_sanction_panel.csv"
    known_case_path = root / "data" / "source" / "known_case_registry.csv"
    sanction_result = _validate_sanctions(_rows(sanction_path))
    known_case_rows = _rows(known_case_path)
    normalization: dict[str, Any] | None = None
    if args.normalize_known_cases:
        known_case_rows, normalization = _canonicalize_known_cases(known_case_rows)
        normalization["write_applied"] = bool(args.write)
        if args.write:
            normalization["backup_path"] = str(_write_known_cases(known_case_path, known_case_rows))

    result = {
        "status": "PASS",
        "raw_root": str(root),
        "sanction_panel": sanction_result,
        "known_case_registry": _validate_known_cases(known_case_rows),
    }
    if normalization is not None:
        result["known_case_normalization"] = normalization
    print(
        "PRODUCTION_SOURCE_CONTRACTS_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
