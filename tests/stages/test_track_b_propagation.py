from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from core.pipeline import physical_columns
from core.registry_compiler import compile_registry
from measurement.service import (
    MeasurementResult,
    build_measurement_inputs,
    measurement_target_frame,
)
from modeling.service import fit_anchor_pu_fold, fit_fold_models

ROOT = Path(__file__).resolve().parents[2]


def _columns() -> dict[str, str]:
    return physical_columns(compile_registry(ROOT / "config" / "pipeline.yaml").registry)


def test_l2_target_preserves_coverage_and_drives_track_b_fit() -> None:
    columns = _columns()
    firm = columns["firm_id"]
    year = columns["fiscal_year"]
    target = columns["target_id"]
    outcome = columns["outcome"]
    weight = columns["weight"]
    matrices = {
        "rows": [
            {
                "firm_id": f"F{index}",
                "fiscal_year": 2015 + index,
                "observed_channel_count": 2 if index != 1 else 1,
                "l2_score": value,
            }
            for index, value in enumerate([0.05, 0.2, 0.35, 0.65, 0.8, 0.95, 0.55])
        ]
    }
    targets = measurement_target_frame(
        matrices=matrices,
        measurement_id="L2",
        minimum_observed_channels=2,
        columns=columns,
    )
    assert pd.isna(targets.loc[1, columns["target_value"]])
    panel = pd.DataFrame(
        {
            firm: pd.Series([f"F{index}" for index in range(7)], dtype="string"),
            year: pd.Series(range(2015, 2022), dtype="int16"),
            "observability": pd.Series([0.0, 0.2, 0.3, 0.6, 0.7, 1.0, 0.5], dtype="float64"),
            "content": pd.Series([0.1, 0.1, 0.4, 0.5, 0.9, 0.8, 0.6], dtype="float64"),
        }
    )
    labels = pd.DataFrame(
        {
            firm: pd.Series([f"F{index}" for index in range(7)], dtype="string"),
            year: pd.Series(range(2015, 2022), dtype="int16"),
            target: pd.Series(["L1"] * 7, dtype="string"),
            outcome: pd.Series([False, False, False, True, True, True, True], dtype="boolean"),
        }
    )
    weights = pd.DataFrame(
        {
            firm: pd.Series([f"F{index}" for index in range(6)], dtype="string"),
            year: pd.Series(range(2015, 2021), dtype="int16"),
            columns["outer_fold"]: pd.Series(["2021"] * 6, dtype="string"),
            weight: pd.Series([1.0] * 6, dtype="float64"),
        }
    )
    result = fit_fold_models(
        feature_panel=panel,
        feature_registry=[
            {"feature_id": "observability", "role": "observability"},
            {"feature_id": "content", "role": "content"},
        ],
        label_inputs=labels,
        weights=weights,
        outer_year=2021,
        learner_ids=["elastic_net_logistic"],
        learner_settings={
            "elastic_net_logistic": {
                "inverse_regularization": 1.0,
                "l1_ratio": 0.5,
                "maximum_iterations": 500,
            }
        },
        target_id="L2",
        measurement_id="L2",
        columns=columns,
        random_state=42,
        target_values=targets,
        soft_target=True,
        target_transform="training_ecdf",
        track_id="track_b",
    )
    assert result.models["status"] == "PASS"
    assert set(result.outer_predictions[target]) == {"L2"}
    assert set(result.outer_predictions[columns["measurement_id"]]) == {"L2"}
    model_rows = cast(list[dict[str, Any]], result.models["models"])
    assert all(
        item["training_objective"] == "soft_cross_entropy"
        for item in model_rows
        if item["status"] == "PASS"
    )


def test_l2_quality_delay_formula_is_operational_and_unlocked_formula_is_pending() -> None:
    columns = _columns()
    firm = columns["firm_id"]
    year = columns["fiscal_year"]
    prediction = columns["prediction_time"]
    risk = pd.DataFrame(
        {
            firm: pd.Series(["F1"], dtype="string"),
            year: pd.Series([2020], dtype="int16"),
            prediction: pd.to_datetime(["2021-03-31"]),
            columns["mature"]: pd.Series([True], dtype=bool),
            columns["eligible"]: pd.Series([True], dtype=bool),
        }
    )
    evidence = pd.DataFrame(
        {
            firm: pd.Series(["F1", "F1"], dtype="string"),
            year: pd.Series([2020, 2020], dtype="int16"),
            columns["source_id"]: pd.Series(["a", "b"], dtype="string"),
            columns["channel_id"]: pd.Series(["c1", "c2"], dtype="string"),
            columns["availability_date"]: pd.to_datetime(["2021-04-15", "2021-04-30"]),
            columns["outcome"]: pd.Series([True, False], dtype="boolean"),
            columns["source_opportunity"]: pd.Series([True, True], dtype="boolean"),
        }
    )

    def build(l2_scoring: dict[str, Any]) -> MeasurementResult:
        return build_measurement_inputs(
            risk_sets=risk,
            evidence=evidence,
            expected_sources={"a": "c1", "b": "c2"},
            horizon_months=12,
            columns=columns,
            pending_status="EMPIRICALLY_PENDING",
            unavailable_status="UNAVAILABLE_BY_DESIGN",
            insufficient_channels_reason="INSUFFICIENT_CHANNELS",
            anchor_source_ids=[],
            source_profiles={"a": "strong", "b": "weak"},
            l2_scoring=l2_scoring,
        )

    pending = build({})
    assert pending.matrices["l2_scoring"]["status"] == "EMPIRICALLY_PENDING"
    assert pending.matrices["rows"][0]["l2_score"] is None
    available = build(
        {
            "formula": "quality_delay_weighted_observed_source_mean",
            "supported_formula": "quality_delay_weighted_observed_source_mean",
            "source_quality_by_profile": {"strong": 1.0, "weak": 0.5},
            "delay_half_life_days": 30,
            "channel_weights": "equal",
        }
    )
    row = cast(dict[str, Any], available.matrices["rows"][0])
    assert row["observed_channel_count"] == 2
    assert row["l2_score_status"] == "AVAILABLE"
    assert float(row["l2_score"]) == pytest.approx(0.5 * 2 ** (-15 / 30))


def test_anchor_pu_uses_only_anchor_positive_rows() -> None:
    columns = _columns()
    firm = columns["firm_id"]
    year = columns["fiscal_year"]
    target = columns["target_id"]
    outcome = columns["outcome"]
    weight = columns["weight"]
    panel = pd.DataFrame(
        {
            firm: pd.Series([f"P{index}" for index in range(8)], dtype="string"),
            year: pd.Series(range(2014, 2022), dtype="int16"),
            "observability": pd.Series(range(8), dtype="float64"),
            "content": pd.Series([0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.6], dtype="float64"),
        }
    )
    labels = pd.DataFrame(
        {
            firm: pd.Series([f"P{index}" for index in range(8)], dtype="string"),
            year: pd.Series(range(2014, 2022), dtype="int16"),
            target: pd.Series(["L0:anchor"] * 8, dtype="string"),
            outcome: pd.Series(
                [False, True, False, False, True, False, None, None], dtype="boolean"
            ),
        }
    )
    weights = pd.DataFrame(
        {
            firm: pd.Series([f"P{index}" for index in range(7)], dtype="string"),
            year: pd.Series(range(2014, 2021), dtype="int16"),
            columns["outer_fold"]: pd.Series(["2021"] * 7, dtype="string"),
            weight: pd.Series([1.0] * 7, dtype="float64"),
        }
    )
    result = fit_anchor_pu_fold(
        feature_panel=panel,
        feature_registry=[
            {"feature_id": "observability", "role": "observability"},
            {"feature_id": "content", "role": "content"},
        ],
        label_inputs=labels,
        weights=weights,
        anchor_source_id="anchor",
        outer_year=2021,
        settings={
            "anchor_pu": {
                "bags": 3,
                "unlabeled_to_positive_ratio": 1.0,
                "inverse_regularization": 1.0,
                "maximum_iterations": 200,
            }
        },
        columns=columns,
        random_state=7,
    )
    rows = cast(list[dict[str, Any]], result.models["models"])
    assert rows[0]["training_mechanism"] == "bagging_pu"
    assert rows[0]["positive_rows"] == 2
    assert rows[0]["l1_union_used_as_clean_positive"] is False
