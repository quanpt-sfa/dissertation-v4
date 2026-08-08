# ruff: noqa: I001
"""Regression coverage for P07 domain metadata propagation and P13 domain refits."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, OUTCOME, TARGET_ID
from features.panel_source import assemble_from_final
from sensitivity.service import domain_transfer


FIRM = "firm_master_id"
YEAR = "fiscal_year"
TIME = "prediction_time"
TARGET = "target_id"
Y = "outcome"


def _write_final_fixture(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            FIRM: ["A", "B"],
            YEAR: [2021, 2021],
            TIME: pd.to_datetime(["2021-04-01", "2021-04-01"]),
            "exchange_or_board": ["HOSE", "HNX"],
            "industry_code": ["I1", "I2"],
            "x": [1.0, 2.0],
        }
    )
    frame.to_parquet(path, index=False)
    return frame


def test_p07_preserves_domain_metadata_without_registering_it_as_feature(tmp_path: Path) -> None:
    final_path = tmp_path / "final.parquet"
    final_frame = _write_final_fixture(final_path)
    base_panel = final_frame[[FIRM, YEAR, TIME]].copy()
    definitions: list[dict[str, object]] = [
        {
            "feature_id": "x",
            "physical_column": "x",
            "research_decision_status": "LOCKED",
        }
    ]

    result = assemble_from_final(
        base_panel=base_panel,
        feature_definitions=definitions,
        intended_definitions=[],
        raw_source_path=final_path,
        firm_column=FIRM,
        year_column=YEAR,
        prediction_time_column=TIME,
    )

    assert result.panel["exchange_or_board"].tolist() == ["HOSE", "HNX"]
    assert result.panel["industry_code"].tolist() == ["I1", "I2"]
    assert result.validation_report["preserved_domain_metadata_columns"] == [
        "exchange_or_board",
        "industry_code",
    ]
    assert all(
        item["feature_id"] not in {"exchange_or_board", "industry_code"} for item in definitions
    )


def test_p07_rejects_domain_metadata_registered_as_feature(tmp_path: Path) -> None:
    final_path = tmp_path / "final.parquet"
    final_frame = _write_final_fixture(final_path)
    base_panel = final_frame[[FIRM, YEAR, TIME]].copy()
    definitions: list[dict[str, object]] = [
        {
            "feature_id": "board_as_feature",
            "physical_column": "exchange_or_board",
            "research_decision_status": "LOCKED",
        }
    ]

    with pytest.raises(ValueError, match="domain metadata must remain outside"):
        assemble_from_final(
            base_panel=base_panel,
            feature_definitions=definitions,
            intended_definitions=[],
            raw_source_path=final_path,
            firm_column=FIRM,
            year_column=YEAR,
            prediction_time_column=TIME,
        )


def test_p13_uses_configured_domain_column_not_feature_registry_domain_id() -> None:
    feature_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    pattern = [(0.0, False), (1.0, True), (2.0, False), (3.0, True)]
    for year in (2020, 2021):
        for board in ("HOSE", "HNX"):
            for index, (value, outcome) in enumerate(pattern):
                firm_id = f"{board}-{year}-{index}"
                feature_rows.append(
                    {
                        FIRM: firm_id,
                        YEAR: year,
                        "exchange_or_board": board,
                        "obs_feature": value,
                        "content_feature": value,
                    }
                )
                outcome_rows.append(
                    {
                        FIRM: firm_id,
                        YEAR: year,
                        TARGET: "L1_ANNUAL",
                        Y: outcome,
                    }
                )

    feature_panel = pd.DataFrame(feature_rows)
    outcomes = pd.DataFrame(outcome_rows)
    predictions = pd.DataFrame({YEAR: [2021]})
    feature_registry: list[dict[str, object]] = [
        {"feature_id": "obs_feature", "role": "observability"},
        {"feature_id": "content_feature", "role": "content"},
    ]
    columns = {
        FIRM_ID: FIRM,
        FISCAL_YEAR: YEAR,
        TARGET_ID: TARGET,
        OUTCOME: Y,
    }

    result = domain_transfer(
        predictions=predictions,
        outcomes=outcomes,
        feature_panel=feature_panel,
        feature_registry=feature_registry,
        domain_bindings=[
            {
                "domain_id": "exchange_or_board",
                "column": "exchange_or_board",
            }
        ],
        noninferiority_margin=0.05,
        support_fraction_minimum=0.8,
        evaluation_target_id="L1_ANNUAL",
        columns=columns,
    )

    assert result["status"] == "PASS"
    assert result["reason_code"] is None
    assert result["leave_one_domain_out_refit_executed"] is True
    domains_raw = result["domains"]
    assert isinstance(domains_raw, list)
    domains = cast(list[dict[str, object]], domains_raw)
    assert {row["level"] for row in domains} == {"HOSE", "HNX"}
    assert {row["domain_column"] for row in domains} == {"exchange_or_board"}
    assert all(row["refit_executed"] is True for row in domains)


def test_p13_reports_missing_configured_domain_column() -> None:
    columns = {
        FIRM_ID: FIRM,
        FISCAL_YEAR: YEAR,
        TARGET_ID: TARGET,
        OUTCOME: Y,
    }
    result = domain_transfer(
        predictions=pd.DataFrame({YEAR: [2021]}),
        outcomes=pd.DataFrame(
            {
                FIRM: ["A"],
                YEAR: [2021],
                TARGET: ["L1_ANNUAL"],
                Y: [True],
            }
        ),
        feature_panel=pd.DataFrame(
            {
                FIRM: ["A"],
                YEAR: [2021],
                "obs_feature": [1.0],
                "content_feature": [1.0],
            }
        ),
        feature_registry=[
            {"feature_id": "obs_feature", "role": "observability"},
            {"feature_id": "content_feature", "role": "content"},
        ],
        domain_bindings=[
            {
                "domain_id": "exchange_or_board",
                "column": "exchange_or_board",
            }
        ],
        noninferiority_margin=0.05,
        support_fraction_minimum=0.8,
        evaluation_target_id="L1_ANNUAL",
        columns=columns,
    )

    assert result["status"] == "SKIPPED"
    assert result["reason_code"] == "DOMAIN_COLUMN_UNAVAILABLE"
    assert result["missing_domain_columns"] == ["exchange_or_board"]
