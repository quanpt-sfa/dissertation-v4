"""Regression coverage for batched P07 feature materialization."""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.errors import PerformanceWarning

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


def _feature(feature_id: str, physical_column: str) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "canonical_name": feature_id,
        "display_name": feature_id,
        "description": "Synthetic registered predictor used to test column materialization.",
        "theoretical_block": "history",
        "source_dataset": "registered_source",
        "source_table_or_file": "registered.parquet",
        "source_column_or_columns": physical_column,
        "physical_column": physical_column,
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
        "lineage": [{"source_column": physical_column, "temporal_shift": "t_minus_1"}],
    }


def test_many_features_are_materialized_without_dataframe_fragmentation_warning() -> None:
    feature_count = 128
    feature_ids = [f"feature_{index:03d}" for index in range(feature_count)]
    physical_columns = [f"signal_{index:03d}" for index in range(feature_count)]

    panel_data: dict[str, object] = {
        "firm_id": ["A", "B"],
        "fiscal_year": [2023, 2023],
        "prediction_time": pd.to_datetime(["2024-03-31", "2024-03-31"]),
    }
    for index, column in enumerate(physical_columns):
        panel_data[column] = [float(index), float(index + 1)]
    panel = pd.DataFrame(panel_data)
    risk = pd.DataFrame(
        {
            "firm_id": ["A", "B"],
            "fiscal_year": [2023, 2023],
            "eligible": [True, True],
            "annual_measurement_mature": [True, True],
            "s3_next_year_mature": [False, False],
        }
    )
    definitions = [
        _feature(feature_id, physical_column)
        for feature_id, physical_column in zip(feature_ids, physical_columns, strict=True)
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", PerformanceWarning)
        result = build_feature_panel(
            firm_year_panel=panel,
            risk_sets=risk,
            feature_definitions=definitions,
            columns=_columns(),
        )

    fragmentation = [
        warning for warning in caught if issubclass(warning.category, PerformanceWarning)
    ]
    assert fragmentation == []
    assert result.summary["panel_features"] == feature_count
    assert result.panel.loc[:, feature_ids].dtypes.eq("float64").all()
