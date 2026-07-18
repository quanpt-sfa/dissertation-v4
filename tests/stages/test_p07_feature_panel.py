"""P07 feature governance tests: raw as-of panel and target-specific controls."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from features.service import build_feature_panel


def _columns() -> dict[str, str]:
    return {
        "firm_id": "firm_id",
        "fiscal_year": "fiscal_year",
        "prediction_time": "prediction_time",
        "eligible": "eligible",
        "annual_measurement_mature": "annual_measurement_mature",
        "s3_next_year_mature": "s3_next_year_mature",
    }


def _locked_feature(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "feature_id": "prior_signal",
        "canonical_name": "prior_signal",
        "display_name": "Prior signal",
        "description": "A valid registered prior-history predictor.",
        "theoretical_block": "history",
        "source_dataset": "registered_source",
        "source_table_or_file": "registered.csv",
        "source_column_or_columns": "signal",
        "physical_column": "signal",
        "formula": "identity",
        "unit": "ratio",
        "sign_convention": "higher_is_higher",
        "fiscal_period_reference": "fiscal_year_t_minus_1",
        "lag_structure": "one_fiscal_year_lag",
        "availability_rule": "available_from_prior_fiscal_year",
        "availability_date_field": "prediction_time",
        "temporal_role": "annual_measurement_at_anchor",
        "maximum_information_date": "prediction_time",
        "expected_dtype": "float64",
        "allowed_missingness": "allowed",
        "missingness_interpretation": "missing_is_unknown",
        "content_observability_role": "content",
        "target_component_flag": False,
        "source_specific_flag": False,
        "source_channel": "S1",
        "model_eligibility": "eligible",
        "allowed_in_label_model": False,
        "confirmatory_status": "confirmatory",
        "research_decision_status": "LOCKED",
        "version": 1,
        "lineage": [{"source_column": "signal", "temporal_shift": "t_minus_1"}],
    }
    return {**value, **overrides}


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.DataFrame(
        {
            "firm_id": ["A", "B"],
            "fiscal_year": [2023, 2023],
            "prediction_time": pd.to_datetime(["2024-03-31", "2024-03-31"]),
            "signal": [1.0, None],
        }
    )
    risk = pd.DataFrame(
        {
            "firm_id": ["A", "B"],
            "fiscal_year": [2023, 2023],
            "eligible": [True, True],
            "annual_measurement_mature": [True, True],
            "s3_next_year_mature": [False, False],
        }
    )
    return panel, risk


def test_p07_raw_panel_is_unique_and_excludes_outcomes() -> None:
    panel, risk = _inputs()
    result = build_feature_panel(
        firm_year_panel=panel,
        risk_sets=risk,
        feature_definitions=[_locked_feature()],
        columns=_columns(),
        target_ids=["S3_BROAD"],
        temporal_estimands={"S3_BROAD": "next_calendar_year_regulatory_event"},
    )
    assert not result.panel.duplicated(["firm_id", "fiscal_year"]).any()
    assert result.panel["prior_signal"].isna().sum() == 1
    assert "outcome" not in result.panel.columns
    assert result.summary["panel_features"] == 1


def test_p07_source_blind_and_component_views_exclude_explicit_dependencies() -> None:
    panel, risk = _inputs()
    result = build_feature_panel(
        firm_year_panel=panel,
        risk_sets=risk,
        feature_definitions=[
            _locked_feature(source_specific_flag=True, target_component_flag=True)
        ],
        columns=_columns(),
        target_ids=["S3_BROAD"],
    )
    view_rows = cast(list[dict[str, object]], result.feature_views["views"])
    views = {str(item["view_id"]): item for item in view_rows}
    assert views["source_blind"]["included_feature_ids"] == []
    assert views["target_component_ablation"]["included_feature_ids"] == []
    assert set(result.leakage_rows["decision"]) == {"ALLOW_WITH_ABLATION"}


def test_p07_rejects_outer_outcome_or_known_case_columns() -> None:
    panel, risk = _inputs()
    panel["outer_outcome"] = True
    with pytest.raises(ValueError, match="forbidden outcome/known-case"):
        build_feature_panel(
            firm_year_panel=panel,
            risk_sets=risk,
            feature_definitions=[_locked_feature()],
            columns=_columns(),
        )


def test_p07_unresolved_features_are_not_confirmatory() -> None:
    panel, risk = _inputs()
    unresolved = _locked_feature(
        feature_id="unresolved_ratio",
        research_decision_status="RESEARCH_DECISION_REQUIRED",
        confirmatory_status="blocked",
        physical_column=None,
        lineage=None,
    )
    result = build_feature_panel(
        firm_year_panel=panel,
        risk_sets=risk,
        feature_definitions=[],
        intended_definitions=[unresolved],
        columns=_columns(),
        target_ids=["L1_ANNUAL", "S3_BROAD"],
    )
    assert result.summary["panel_features"] == 0
    assert set(result.leakage_rows["decision"]) == {"RESEARCH_DECISION_REQUIRED"}
    assert all(
        not item["included_feature_ids"]
        for item in cast(list[dict[str, object]], result.feature_views["views"])
    )


def test_target_component_is_blocked_only_for_affected_target() -> None:
    panel, risk = _inputs()
    feature = _locked_feature(
        feature_id="fs_aud_profit_after_tax",
        target_component_flag=True,
        source_semantics=["post_audit_profit_after_tax"],
    )
    result = build_feature_panel(
        firm_year_panel=panel,
        risk_sets=risk,
        feature_definitions=[feature],
        columns=_columns(),
        target_ids=["L1_ANNUAL", "S3_BROAD"],
    )
    decisions = {
        str(row.target_id): str(row.decision) for row in result.leakage_rows.itertuples(index=False)
    }
    assert decisions == {"L1_ANNUAL": "BLOCK", "S3_BROAD": "ALLOW"}
    views = cast(list[dict[str, object]], result.feature_views["views"])
    confirmatory = {
        str(item["target_id"]): item for item in views if item["view_id"] == "core_confirmatory"
    }
    assert confirmatory["L1_ANNUAL"]["included_feature_ids"] == []
    assert confirmatory["S3_BROAD"]["included_feature_ids"] == ["fs_aud_profit_after_tax"]
