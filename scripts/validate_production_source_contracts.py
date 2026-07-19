"""Validate raw S3 and known-case source contracts before snapshot locking."""

from __future__ import annotations

import argparse
import csv
import json
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


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
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


def _validate_known_cases(rows: list[dict[str, str]]) -> dict[str, Any]:
    required = {
        "case_id",
        "firm_id",
        "fiscal_year",
        "case_construct",
        "role",
        "training_include_flag",
        "calibration_include_flag",
        "model_selection_include_flag",
        "external_validation_include_flag",
    }
    _require_columns(rows, required, "known-case registry")
    observed: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows, start=2):
        record = (
            row["case_id"].strip(),
            row["firm_id"].strip(),
            int(row["fiscal_year"].strip()),
        )
        observed.add(record)
        if row["case_construct"].strip() != "CONFIRMED_FINANCIAL_REPORTING_CASE":
            raise ValueError(f"known-case row={index}: invalid case_construct")
        if row["role"].strip() != "SIMULATION_EXTERNAL_VALIDATION":
            raise ValueError(f"known-case row={index}: invalid role")
        flags = {
            "training_include_flag": _bool(row["training_include_flag"], "training_include_flag", index),
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
    args = parser.parse_args()

    root = args.raw_root.expanduser().resolve()
    sanction_path = root / "data" / "source" / "firm_event_sanction_panel.csv"
    known_case_path = root / "data" / "source" / "known_case_registry.csv"
    result = {
        "status": "PASS",
        "raw_root": str(root),
        "sanction_panel": _validate_sanctions(_rows(sanction_path)),
        "known_case_registry": _validate_known_cases(_rows(known_case_path)),
    }
    print("PRODUCTION_SOURCE_CONTRACTS_JSON=" + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
