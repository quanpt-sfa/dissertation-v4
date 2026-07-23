from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from core.semantic_keys import (
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    MEASUREMENT_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    TARGET_ID,
    TARGET_VALUE,
    WEIGHT,
)
from selection.nested_refit import run_nested_channel_refit, validate_nested_refit_receipt
from selection.service import select_measurement


def _columns() -> dict[str, str]:
    return {
        FIRM_ID: "firm_key",
        FISCAL_YEAR: "year_key",
        LEARNER_ID: "model_key",
        MEASUREMENT_ID: "measurement_key",
        OUTCOME: "binary_result",
        OUTER_FOLD: "fold_key",
        PREDICTION: "score",
        TARGET_ID: "target_key",
        TARGET_VALUE: "soft_target",
        WEIGHT: "analysis_weight",
    }


def _fixture() -> tuple[
    dict[str, object],
    pd.DataFrame,
    list[dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
]:
    panel_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    for year in (2018, 2019, 2020):
        for index in range(16):
            firm_id = f"F{index:02d}"
            first = index % 2 == 0
            second = (index // 2) % 2 == 0
            panel_rows.append(
                {
                    "firm_key": firm_id,
                    "year_key": year,
                    "feature_s1": float(first) + 0.01 * (year - 2018),
                    "feature_s2": float(second) + 0.02 * (year - 2018),
                    "feature_neutral": float(index) / 16.0,
                }
            )
            matrix_rows.append(
                {
                    FIRM_ID: firm_id,
                    FISCAL_YEAR: year,
                    ELIGIBLE: True,
                    MATURE: True,
                    "channel_outcomes": {"S1": first, "S2": second},
                    "channel_evidence_scores": {
                        "S1": 0.8 if first else 0.2,
                        "S2": 0.75 if second else 0.25,
                    },
                }
            )
            weight_rows.append(
                {
                    "firm_key": firm_id,
                    "year_key": year,
                    "analysis_weight": 1.0,
                }
            )
    matrices: dict[str, object] = {
        "expected_channels": ["S1", "S2"],
        "rows": matrix_rows,
        "l2_scoring": {"status": "AVAILABLE"},
    }
    registry: list[dict[str, object]] = [
        {"feature_id": "feature_s1", "role": "content", "source_channel": "S1"},
        {"feature_id": "feature_s2", "role": "content", "source_channel": "S2"},
        {"feature_id": "feature_neutral", "role": "ambiguous", "source_channel": ""},
    ]
    label_inputs = pd.DataFrame(columns=["firm_key", "year_key", "target_key", "binary_result"])
    return (
        matrices,
        pd.DataFrame(panel_rows),
        registry,
        label_inputs,
        pd.DataFrame(weight_rows),
    )


def test_nested_refit_reruns_complete_development_procedure_per_channel() -> None:
    matrices, panel, registry, labels, weights = _fixture()
    result = run_nested_channel_refit(
        matrices=matrices,
        feature_panel=panel,
        feature_registry=registry,
        label_inputs=labels,
        weights=weights,
        outer_year=2021,
        candidates=["L2", "none"],
        l3_fold_result=None,
        minimum_observed_channels=1,
        gate1_learner_id="elastic_net_logistic",
        gate1_feature_group="full",
        learner_settings={
            "elastic_net_logistic": {
                "inverse_regularization": 1.0,
                "l1_ratio": 0.0,
                "maximum_iterations": 500,
            }
        },
        learner_search_spaces={
            "elastic_net_logistic": {
                "inverse_regularization": [1.0],
                "l1_ratio": [0.0],
            }
        },
        maximum_valid_configurations=5,
        columns=_columns(),
        random_state=17,
    )

    candidate = result.candidate_results[0]
    assert candidate["candidate"] == "L2"
    assert candidate[ELIGIBLE] is True
    assert candidate["completed_heldout_channels"] == 2
    assert result.receipt["status"] == "PASS"
    assert result.receipt["outer_rows_used_in_selection"] == 0
    assert result.receipt["outer_outcomes_accessed"] is False

    cells = cast(list[dict[str, object]], result.receipt["cell_results"])
    assert {str(item["heldout_channel"]) for item in cells} == {"S1", "S2"}
    assert all(item["status"] == "PASS" for item in cells)
    assert all(item["eligible_feature_count"] == 2 for item in cells)
    assert all(item["heldout_removed_from_tuning"] is True for item in cells)
    assert all(item["heldout_removed_from_calibration"] is True for item in cells)


def test_nested_receipt_fails_closed_before_freeze_or_outer_open() -> None:
    receipt = {
        "status": "PASS",
        OUTER_FOLD: "2021",
        "fit_scope": "development_history_only",
        "outer_outcomes_accessed": False,
        "outer_rows_used_in_selection": 0,
        "heldout_channel_removed_from": [
            "target",
            "label_model",
            "features",
            "tuning",
            "calibration",
        ],
        "candidate_results": [{"candidate": "L2", ELIGIBLE: True, "objective": 0.5}],
    }
    validated = validate_nested_refit_receipt(
        {"nested_refit_receipt": receipt},
        outer_fold="2021",
        required_optional_measurements=["L2"],
    )
    assert validated["status"] == "PASS"

    tampered = {**receipt, "outer_rows_used_in_selection": 1}
    with pytest.raises(RuntimeError, match="OUTER_ROWS_USED"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": tampered},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )

    incomplete = {
        **receipt,
        "candidate_results": [{"candidate": "L2", ELIGIBLE: False, "objective": None}],
    }
    with pytest.raises(RuntimeError, match="REQUIRED_CANDIDATE_INCOMPLETE"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": incomplete},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )


def test_nested_results_override_proxy_only_measurement_scores() -> None:
    result = select_measurement(
        matrices={
            "expected_channels": ["S1", "S2"],
            "rows": [{FISCAL_YEAR: 2020, "channel_outcomes": {"S1": True, "S2": False}}],
            "l2_scoring": {"status": "AVAILABLE"},
        },
        outer_year=2021,
        candidates=["L2"],
        l3_capability={"status": "UNAVAILABLE_BY_DESIGN", "pilot_executed": False},
        minimum_observed_channels=1,
        nested_candidate_results=[
            {
                "candidate": "L2",
                ELIGIBLE: True,
                "objective": 0.321,
                "reason_code": None,
                "selection_procedure": "full_refit_channel_within_time",
            }
        ],
    )
    assert result.candidates[0][ELIGIBLE] is True
    assert result.candidates[0]["objective"] == pytest.approx(0.321)
    assert result.selection["selected_measurement"] == "L2"
