# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from p01.audit import audit_source
from p01.models import SourceSpec
from p01.readers import hash_file


def _profile(path: Path, *, format_name: str, locked_hash: str) -> SourceSpec:
    delimiter = "," if format_name == "csv" else None
    return SourceSpec.from_mapping(
        "source_a",
        {
            "enabled": True,
            "channel_id": "S1",
            "source_type": "official",
            "source_agency": "Test agency",
            "original_unit": "firm-year",
            "related_period_field": "fiscal_year",
            "availability_date_field": "availability_date",
            "availability_date_source": "Test publication record",
            "coverage_dimensions": ["year", "listing_board"],
            "role": "predictor",
            "verification_status": "observed",
            "data_risks": ["duplicates", "missing_dates"],
            "relative_path": path.name,
            "format": format_name,
            "encoding": "utf-8",
            "delimiter": delimiter,
            "locked_sha256": locked_hash,
            "schema": {
                "required_columns": [
                    "firm_id",
                    "fiscal_year",
                    "availability_date",
                ],
                "optional_columns": ["amount"],
                "key_columns": ["firm_id", "fiscal_year"],
                "date_columns": ["availability_date"],
                "required_date_columns": ["availability_date"],
                "numeric_columns": {
                    "amount": {
                        "unit": "VND",
                        "allow_negative": False,
                        "decimal_mark": ".",
                        "thousands_separator": ",",
                    }
                },
                "allow_extra_columns": False,
                "key_unique": True,
                "row_count_min": 1,
            },
        },
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "firm_id",
                "fiscal_year",
                "availability_date",
                "amount",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_csv_source_passes(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    _write_csv(
        path,
        [
            {
                "firm_id": "A",
                "fiscal_year": "2023",
                "availability_date": "2024-03-31",
                "amount": "1,000",
            },
            {
                "firm_id": "B",
                "fiscal_year": "2023",
                "availability_date": "2024-04-01",
                "amount": "2500",
            },
        ],
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "PASS"
    assert report.coverage["row_count"] == 2
    assert report.duplicate_key_audit["duplicate_group_count"] == 0


def test_hash_change_requires_new_run(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    _write_csv(
        path,
        [
            {
                "firm_id": "A",
                "fiscal_year": "2023",
                "availability_date": "2024-03-31",
                "amount": "1000",
            }
        ],
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash="0" * 64),
    )
    assert report.status == "FAILED"
    assert report.decision["new_run_required"] is True


def test_missing_required_column_fails(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_text(
        "firm_id,fiscal_year,amount\nA,2023,1000\n",
        encoding="utf-8",
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "FAILED"
    missing = report.schema_audit["missing_required_columns"]
    assert isinstance(missing, list)
    assert "availability_date" in missing


def test_duplicate_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    rows = [
        {
            "firm_id": "A",
            "fiscal_year": "2023",
            "availability_date": "2024-03-31",
            "amount": "1000",
        },
        {
            "firm_id": "A",
            "fiscal_year": "2023",
            "availability_date": "2024-04-01",
            "amount": "1200",
        },
    ]
    _write_csv(path, rows)
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "FAILED"
    assert report.duplicate_key_audit["duplicate_group_count"] == 1


def test_unregistered_field_fails(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_text(
        "firm_id,fiscal_year,availability_date,amount,unexpected\nA,2023,2024-03-31,1000,x\n",
        encoding="utf-8",
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "FAILED"
    assert report.schema_audit["extra_columns"] == ["unexpected"]


def test_invalid_optional_numeric_is_isolated(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    _write_csv(
        path,
        [
            {
                "firm_id": "A",
                "fiscal_year": "2023",
                "availability_date": "2024-03-31",
                "amount": "not-a-number",
            }
        ],
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "PASS_WITH_ISSUES"
    amount_audit = report.unit_audit["amount"]
    assert isinstance(amount_audit, dict)
    assert amount_audit["invalid"] == 1


def test_parquet_source_passes(tmp_path: Path) -> None:
    path = tmp_path / "source.parquet"
    table = pa.table(
        {
            "firm_id": ["A"],
            "fiscal_year": [2023],
            "availability_date": ["2024-03-31"],
            "amount": [1000],
        }
    )
    pq.write_table(table, path)
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="parquet", locked_hash=hash_file(path)),
    )
    assert report.status == "PASS"
    assert report.file_signature["format"] == "parquet"


def test_missing_required_date_value_is_isolated(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    _write_csv(
        path,
        [
            {
                "firm_id": "A",
                "fiscal_year": "2023",
                "availability_date": "",
                "amount": "1000",
            }
        ],
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "PASS_WITH_ISSUES"
    assert any(issue.code == "MISSING_REQUIRED_DATE_VALUES" for issue in report.issues)


def test_missing_key_value_fails(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    _write_csv(
        path,
        [
            {
                "firm_id": "",
                "fiscal_year": "2023",
                "availability_date": "2024-03-31",
                "amount": "1000",
            }
        ],
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="csv", locked_hash=hash_file(path)),
    )
    assert report.status == "FAILED"
    assert any(issue.code == "MISSING_KEY_VALUES" for issue in report.issues)


def test_jsonl_later_row_unregistered_field_fails(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text(
        '{"firm_id":"A","fiscal_year":2023,"availability_date":"2024-03-31","amount":1}\n'
        '{"firm_id":"B","fiscal_year":2023,"availability_date":"2024-04-01","amount":2,"later_extra":"x"}\n',
        encoding="utf-8",
    )
    report = audit_source(
        run_id="run-a",
        protocol_hash="a" * 64,
        source_path=path,
        spec=_profile(path, format_name="jsonl", locked_hash=hash_file(path)),
    )
    assert report.status == "FAILED"
    assert any(issue.code == "UNREGISTERED_SOURCE_FIELDS_IN_ROWS" for issue in report.issues)
