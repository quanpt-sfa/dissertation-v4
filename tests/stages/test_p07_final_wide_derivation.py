"""Regression coverage for deriving locked P07 features from the final wide input."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from features.panel_source import _assemble_from_final

FIRM = "firm_master_id"
YEAR = "fiscal_year"
TIME = "prediction_time"


def _definition(
    feature_id: str,
    *,
    formula: str,
    dependencies: list[str],
    step: str,
) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "physical_column": feature_id,
        "formula": formula,
        "dependencies": dependencies,
        "research_decision_status": "LOCKED",
        "lineage": [{"transformation_step": step}],
    }


def test_final_wide_input_derives_prepost_and_lagged_ratio_features(tmp_path: Path) -> None:
    prediction_time = pd.to_datetime(["2021-03-31", "2022-03-31"])
    final_frame = pd.DataFrame(
        {
            FIRM: ["F001", "F001"],
            YEAR: [2020, 2021],
            TIME: prediction_time,
            "fs_aud_net_revenue": [100.0, 120.0],
            "fs_unaud_net_revenue": [90.0, 100.0],
        }
    )
    path = tmp_path / "final.parquet"
    final_frame.to_parquet(path, index=False)
    base_panel = final_frame.loc[:, [FIRM, YEAR, TIME]].copy()
    definitions = [
        _definition(
            "fs_aud_net_revenue",
            formula="source",
            dependencies=[],
            step="source_selection",
        ),
        _definition(
            "fs_unaud_net_revenue",
            formula="source",
            dependencies=[],
            step="source_selection",
        ),
        _definition(
            "audit_adj_net_revenue_signed",
            formula="fs_aud_net_revenue - fs_unaud_net_revenue",
            dependencies=["fs_aud_net_revenue", "fs_unaud_net_revenue"],
            step="prepost_difference",
        ),
        _definition(
            "audit_adj_net_revenue_relative",
            formula=(
                "(fs_aud_net_revenue - fs_unaud_net_revenue) / "
                "abs(fs_aud_net_revenue)"
            ),
            dependencies=["fs_aud_net_revenue", "fs_unaud_net_revenue"],
            step="prepost_difference",
        ),
        _definition(
            "ratio_sales_growth",
            formula=(
                "(net_revenue_t - net_revenue_t_minus_1) / "
                "abs(net_revenue_t_minus_1)"
            ),
            dependencies=["fs_aud_net_revenue[t]", "fs_aud_net_revenue[t-1]"],
            step="registered_ratio",
        ),
    ]

    result = _assemble_from_final(
        base_panel=base_panel,
        feature_definitions=definitions,
        intended_definitions=[],
        raw_source_path=path,
        firm_column=FIRM,
        year_column=YEAR,
        prediction_time_column=TIME,
    )

    assert result.panel["audit_adj_net_revenue_signed"].tolist() == [10.0, 20.0]
    assert result.panel["audit_adj_net_revenue_relative"].tolist() == [0.1, 1.0 / 6.0]
    assert pd.isna(result.panel.loc[0, "ratio_sales_growth"])
    assert result.panel.loc[1, "ratio_sales_growth"] == pytest.approx(0.2)
    assert set(result.validation_report["derived_features"]) == {
        "audit_adj_net_revenue_signed",
        "audit_adj_net_revenue_relative",
        "ratio_sales_growth",
    }
    derived_audit = result.research_decision_audit.set_index("feature_id")
    assert (
        derived_audit.loc["audit_adj_net_revenue_signed", "reason_code"]
        == "DERIVED_FROM_FINAL_INPUT"
    )


def test_final_wide_input_fails_closed_for_missing_locked_atomic_feature(tmp_path: Path) -> None:
    final_frame = pd.DataFrame(
        {
            FIRM: ["F001"],
            YEAR: [2020],
            TIME: pd.to_datetime(["2021-03-31"]),
        }
    )
    path = tmp_path / "final.parquet"
    final_frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="fs_aud_total_assets"):
        _assemble_from_final(
            base_panel=final_frame.copy(),
            feature_definitions=[
                _definition(
                    "fs_aud_total_assets",
                    formula="source",
                    dependencies=[],
                    step="source_selection",
                )
            ],
            intended_definitions=[],
            raw_source_path=path,
            firm_column=FIRM,
            year_column=YEAR,
            prediction_time_column=TIME,
        )
