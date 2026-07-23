from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import selection.service as selection_service
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
from labels.latent_class import LatentClassResult
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


def _valid_receipt() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for heldout, objective in (("S1", 0.41), ("S2", 0.39)):
        cells.append(
            {
                "candidate": "L2",
                "heldout_channel": heldout,
                "status": "PASS",
                "reason_code": None,
                "rows": 12,
                "soft_cross_entropy": objective,
                "fit_scope": "development_history_only",
                "outer_rows_used": 0,
                "outer_outcomes_accessed": False,
                "heldout_removed_from_target": True,
                "heldout_removed_from_label_model": True,
                "heldout_removed_from_features": True,
                "heldout_removed_from_tuning": True,
                "heldout_removed_from_calibration": True,
            }
        )
    return {
        "status": "PASS",
        "reason_code": None,
        OUTER_FOLD: "2021",
        "selection_procedure": "full_refit_channel_within_time",
        "fit_scope": "development_history_only",
        "outer_outcomes_accessed": False,
        "outer_rows_used_in_selection": 0,
        "complete_channel_grid_required": True,
        "heldout_channel_removed_from": [
            "target",
            "label_model",
            "features",
            "tuning",
            "calibration",
        ],
        "expected_channels": ["S1", "S2"],
        "gate1_learner_id": "elastic_net_logistic",
        "gate1_feature_group": "full",
        "candidate_results": [
            {
                "candidate": "L2",
                ELIGIBLE: True,
                "objective": 0.4,
                "required_heldout_channels": 2,
                "completed_heldout_channels": 2,
            }
        ],
        "cell_results": cells,
    }


def test_nested_receipt_fails_closed_before_freeze_or_outer_open() -> None:
    receipt = _valid_receipt()
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

    missing_cell = {
        **receipt,
        "cell_results": cast(list[dict[str, object]], receipt["cell_results"])[:-1],
    }
    with pytest.raises(RuntimeError, match="CHANNEL_GRID_INCOMPLETE"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": missing_cell},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )

    bad_cells = [dict(item) for item in cast(list[dict[str, object]], receipt["cell_results"])]
    bad_cells[0]["heldout_removed_from_calibration"] = False
    bad_removal = {**receipt, "cell_results": bad_cells}
    with pytest.raises(RuntimeError, match="CELL_REMOVAL_INVALID"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": bad_removal},
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


def test_l3_target_inclusion_does_not_depend_on_heldout_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, Any]] = [
        {
            FIRM_ID: "F0",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 1,
            "source_outcomes": {"a": None, "b": True},
            "channel_outcomes": {"S1": None, "S2": True},
        },
        {
            FIRM_ID: "F1",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 2,
            "source_outcomes": {"a": True, "b": False},
            "channel_outcomes": {"S1": True, "S2": False},
        },
        {
            FIRM_ID: "F2",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 1,
            "source_outcomes": {"a": False, "b": None},
            "channel_outcomes": {"S1": False, "S2": None},
        },
        {
            FIRM_ID: "F3",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 2,
            "source_outcomes": {"a": True, "b": True},
            "channel_outcomes": {"S1": True, "S2": True},
        },
    ]

    def fake_fit_l3(
        *,
        rows: list[dict[str, Any]],
        source_channels: dict[str, str],
        accuracy_priors: dict[str, dict[str, float]],
        fixed_pi: float,
        mcmc: dict[str, Any],
        rng: np.random.Generator,
    ) -> LatentClassResult:
        _ = (accuracy_priors, fixed_pi, mcmc, rng)
        probabilities = [0.2 + 0.15 * index for index in range(len(rows))]
        return LatentClassResult(
            posterior_mean=probabilities,
            posterior_draws=[[0 for _ in rows]],
            parameter_draws=[],
            source_accuracy={source: {} for source in source_channels},
            channel_random_effect_sd={channel: 0.0 for channel in set(source_channels.values())},
            diagnostics={"eligible_for_gate1": True},
        )

    monkeypatch.setattr(selection_service, "_fit_l3", fake_fit_l3)
    result = selection_service.fit_l3_fold_candidate(
        matrices={"rows": rows},
        outer_year=2021,
        source_channels={"a": "S1", "b": "S2"},
        accuracy_priors={"a": {}, "b": {}},
        fixed_pi_grid=[0.1],
        mcmc={},
        minimum_observed_channels=1,
        robust_fraction=1.0,
        rng=np.random.default_rng(9),
    )
    assert result["status"] == "PASS"
    targets = cast(list[dict[str, object]], result["heldout_target_rows"])
    included = {(str(item["heldout_channel"]), str(item[FIRM_ID])) for item in targets}
    assert ("S1", "F0") in included
    assert ("S2", "F2") in included
