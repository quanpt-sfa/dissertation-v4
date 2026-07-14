from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from p01.models import SourceSpec
from p01.readers import hash_file
from p02.builder import SourceInput, build_firm_panel
from p02.models import EntityResolutionSpec, PanelSourceSpec

OUTPUT_COLUMNS = {
    "firm_id": "firm_master_id",
    "fiscal_year": "fiscal_year",
    "prediction_time": "prediction_time",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "entity_code",
        "ticker",
        "fiscal_period",
        "available_at",
        "fiscal_end",
        "registered_predictor",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _source_input(
    tmp_path: Path,
    source_id: str,
    rows: list[dict[str, object]],
    *,
    core: bool = True,
    audit_advance: bool = True,
    firm_id_field: str = "entity_code",
) -> SourceInput:
    path = tmp_path / f"{source_id}.csv"
    _write_csv(path, rows)
    source_mapping: dict[str, object] = {
        "enabled": True,
        "channel_id": "S1",
        "source_type": "official",
        "source_agency": "Agency",
        "original_unit": "firm-year",
        "related_period_field": "fiscal_period",
        "availability_date_field": "available_at",
        "availability_date_source": "publication record",
        "coverage_dimensions": ["year"],
        "role": "predictor",
        "verification_status": "observed",
        "data_risks": ["entity_link"],
        "relative_path": path.name,
        "format": "csv",
        "encoding": "utf-8",
        "delimiter": ",",
        "locked_sha256": hash_file(path),
        "schema": {
            "required_columns": [
                "entity_code",
                "fiscal_period",
                "available_at",
            ],
            "optional_columns": ["ticker", "fiscal_end", "registered_predictor"],
            "key_columns": ["entity_code", "fiscal_period"],
            "date_columns": ["available_at", "fiscal_end"],
            "required_date_columns": ["available_at"],
            "numeric_columns": {},
            "allow_extra_columns": False,
            "key_unique": True,
            "row_count_min": 1,
        },
        "panel_mapping": {
            "firm_id_field": firm_id_field,
            "fiscal_year_field": "fiscal_period",
            "availability_date_field": "available_at",
            "fiscal_year_end_field": "fiscal_end",
            "ticker_field": "ticker",
            "contributes_to_firm_master": True,
            "core_predictor": core,
        },
    }
    source_spec = SourceSpec.from_mapping(source_id, source_mapping)
    panel_spec = PanelSourceSpec.from_source_mapping(source_id, source_mapping)
    return SourceInput(
        source_spec=source_spec,
        panel_spec=panel_spec,
        path=path,
        audit={
            "source_id": source_id,
            "status": "PASS" if audit_advance else "FAILED",
            "decision": {"pipeline_may_advance": audit_advance},
        },
    )


def _entity_spec(
    *,
    aliases: dict[str, dict[str, str]] | None = None,
    allow_identity: bool = True,
    early_exceptions: list[dict[str, object]] | None = None,
) -> EntityResolutionSpec:
    return EntityResolutionSpec.from_mapping(
        {
            "policy": "registered_only",
            "allow_identity_mapping": allow_identity,
            "collision_policy": "fail",
            "normalization": {
                "unicode_nfkc": True,
                "trim": True,
                "uppercase": True,
                "collapse_internal_whitespace": True,
            },
            "aliases": aliases or {},
            "reporting_calendar": {
                "default_fiscal_year_end_month_day": "12-31",
                "firm_exceptions": {},
                "early_report_exceptions": early_exceptions or [],
            },
        }
    )


def _row(
    firm: str,
    year: int,
    available: str,
    *,
    ticker: str = "AAA",
    fiscal_end: str | None = None,
    predictor: float | None = None,
) -> dict[str, object]:
    return {
        "entity_code": firm,
        "ticker": ticker,
        "fiscal_period": year,
        "available_at": available,
        "fiscal_end": fiscal_end or f"{year}-12-31",
        "registered_predictor": predictor,
    }


def test_builds_prediction_time_as_latest_core_availability(tmp_path: Path) -> None:
    source_a = _source_input(
        tmp_path,
        "source_a",
        [_row(" firm-a ", 2023, "2024-03-15")],
    )
    source_b = _source_input(
        tmp_path,
        "source_b",
        [_row("FIRM-A", 2023, "2024-04-20")],
    )
    result = build_firm_panel(
        run_id="run-1",
        protocol_hash="a" * 64,
        source_inputs=[source_a, source_b],
        entity_spec=_entity_spec(),
        sample_start=2015,
        sample_end=2025,
        output_columns=OUTPUT_COLUMNS,
    )
    assert result.firm_master.to_dict("records") == [{"firm_master_id": "FIRM-A"}]
    assert result.firm_year_panel.loc[0, "prediction_time"] == pd.Timestamp("2024-04-20")


def test_registered_predictor_is_carried_into_the_as_of_panel(tmp_path: Path) -> None:
    source = _source_input(
        tmp_path,
        "source_a",
        [_row("A", 2023, "2024-03-15", predictor=1.25)],
    )
    result = build_firm_panel(
        run_id="run-features",
        protocol_hash="a" * 64,
        source_inputs=[source],
        entity_spec=_entity_spec(),
        sample_start=2015,
        sample_end=2025,
        output_columns=OUTPUT_COLUMNS,
        registered_predictor_columns=["registered_predictor"],
    )
    assert result.firm_year_panel.loc[0, "registered_predictor"] == pytest.approx(1.25)


def test_incomplete_core_availability_is_excluded(tmp_path: Path) -> None:
    source_a = _source_input(
        tmp_path,
        "source_a",
        [
            _row("A", 2023, "2024-03-15"),
            _row("B", 2023, "2024-03-20"),
        ],
    )
    source_b = _source_input(
        tmp_path,
        "source_b",
        [_row("A", 2023, "2024-04-20")],
    )
    result = build_firm_panel(
        run_id="run-1",
        protocol_hash="a" * 64,
        source_inputs=[source_a, source_b],
        entity_spec=_entity_spec(),
        sample_start=2015,
        sample_end=2025,
        output_columns=OUTPUT_COLUMNS,
    )
    assert list(result.firm_year_panel["firm_master_id"]) == ["A"]
    excluded = result.resolution_report["excluded_firm_years"]
    assert isinstance(excluded, list)
    assert excluded[0]["canonical_firm_id"] == "B"
    assert excluded[0]["missing_source_ids"] == ["source_b"]


def test_registered_alias_resolves_cross_source_identity(tmp_path: Path) -> None:
    source_a = _source_input(
        tmp_path,
        "source_a",
        [_row("OLD CODE", 2023, "2024-03-15")],
    )
    source_b = _source_input(
        tmp_path,
        "source_b",
        [_row("CANON", 2023, "2024-04-20")],
    )
    result = build_firm_panel(
        run_id="run-1",
        protocol_hash="a" * 64,
        source_inputs=[source_a, source_b],
        entity_spec=_entity_spec(aliases={"source_a": {"OLD CODE": "CANON"}}),
        sample_start=2015,
        sample_end=2025,
        output_columns=OUTPUT_COLUMNS,
    )
    assert list(result.firm_master["firm_master_id"]) == ["CANON"]
    assert len(result.firm_year_panel) == 1


def test_ticker_cannot_be_immutable_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ticker cannot be used"):
        _source_input(
            tmp_path,
            "source_a",
            [_row("A", 2023, "2024-03-15")],
            firm_id_field="ticker",
        )


def test_early_availability_without_exception_fails(tmp_path: Path) -> None:
    source = _source_input(
        tmp_path,
        "source_a",
        [_row("A", 2023, "2023-10-01")],
    )
    with pytest.raises(ValueError, match="precedes fiscal-year end"):
        build_firm_panel(
            run_id="run-1",
            protocol_hash="a" * 64,
            source_inputs=[source],
            entity_spec=_entity_spec(),
            sample_start=2015,
            sample_end=2025,
            output_columns=OUTPUT_COLUMNS,
        )


def test_registered_early_availability_exception_is_allowed(tmp_path: Path) -> None:
    source = _source_input(
        tmp_path,
        "source_a",
        [_row("A", 2023, "2023-10-01")],
    )
    result = build_firm_panel(
        run_id="run-1",
        protocol_hash="a" * 64,
        source_inputs=[source],
        entity_spec=_entity_spec(
            early_exceptions=[{"firm_id": "A", "fiscal_year": 2023, "source_id": "source_a"}]
        ),
        sample_start=2015,
        sample_end=2025,
        output_columns=OUTPUT_COLUMNS,
    )
    assert len(result.firm_year_panel) == 1


def test_failed_p01_audit_blocks_p02(tmp_path: Path) -> None:
    source = _source_input(
        tmp_path,
        "source_a",
        [_row("A", 2023, "2024-03-15")],
        audit_advance=False,
    )
    with pytest.raises(ValueError, match="P01 audit blocks P02"):
        build_firm_panel(
            run_id="run-1",
            protocol_hash="a" * 64,
            source_inputs=[source],
            entity_spec=_entity_spec(),
            sample_start=2015,
            sample_end=2025,
            output_columns=OUTPUT_COLUMNS,
        )


def test_duplicate_source_firm_year_after_resolution_fails(tmp_path: Path) -> None:
    source = _source_input(
        tmp_path,
        "source_a",
        [
            _row("A", 2023, "2024-03-15"),
            _row(" a ", 2023, "2024-03-20"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate source firm-year"):
        build_firm_panel(
            run_id="run-1",
            protocol_hash="a" * 64,
            source_inputs=[source],
            entity_spec=_entity_spec(),
            sample_start=2015,
            sample_end=2025,
            output_columns=OUTPUT_COLUMNS,
        )


def test_output_dtypes_match_locked_contract(tmp_path: Path) -> None:
    source = _source_input(
        tmp_path,
        "source_a",
        [_row("A", 2023, "2024-03-15")],
    )
    result = build_firm_panel(
        run_id="run-1",
        protocol_hash="a" * 64,
        source_inputs=[source],
        entity_spec=_entity_spec(),
        sample_start=2015,
        sample_end=2025,
        output_columns=OUTPUT_COLUMNS,
    )
    assert str(result.firm_master.dtypes["firm_master_id"]) == "string"
    assert str(result.firm_year_panel.dtypes["firm_master_id"]) == "string"
    assert str(result.firm_year_panel.dtypes["fiscal_year"]) == "int16"
    assert str(result.firm_year_panel.dtypes["prediction_time"]) == "datetime64[ns]"
