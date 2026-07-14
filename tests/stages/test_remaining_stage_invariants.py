"""Oracle-style checks for the methodological invariants implemented in P03-P17."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    PREDICTION_TIME,
    SOURCE_ID,
    TARGET_VALUE,
    WEIGHT,
)
from evaluation.service import evaluate_outer_fold
from evidence.service import EvidenceRecord, build_evidence_ledger
from features.service import build_feature_panel
from gates.service import breakpoint_stability_pass, gate2_verdict, gate3_verdict
from labels.service import aggregate_l1, evidence_score_l2, posterior_l3_fixed_pi
from risksets.service import build_risk_set
from selection.service import select_measurement
from simulation.service import (
    attach_development_covariate_pools,
    run_batch,
    summarize_mcse,
)
from splits.service import build_splits_and_weights


def _columns() -> dict[str, str]:
    return {
        FIRM_ID: "firm_key",
        FISCAL_YEAR: "year_key",
        PREDICTION_TIME: "as_of",
        AVAILABILITY_DATE: "available_at",
        SOURCE_ID: "source_key",
        CHANNEL_ID: "channel_key",
        OUTCOME: "binary_result",
        ELIGIBLE: "is_eligible",
        MATURE: "is_mature",
        OUTER_FOLD: "fold_key",
        LEARNER_ID: "model_key",
        PREDICTION: "score",
        TARGET_VALUE: "soft_target",
        WEIGHT: "analysis_weight",
    }


def test_label_aggregation_preserves_missingness_and_observed_channel_denominator() -> None:
    assert aggregate_l1({"a": None, "b": False}) is None
    assert aggregate_l1({"a": False, "b": False}) is False
    assert aggregate_l1({"a": None, "b": True}) is True
    assert evidence_score_l2({"a": None, "b": False, "c": True}) == pytest.approx(0.5)
    assert evidence_score_l2({"a": None}) is None


def test_p03_deduplicates_upstream_events_and_enforces_lag_identity() -> None:
    columns = _columns()
    panel = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}])
    records = [
        EvidenceRecord(
            source_id="anchor",
            channel_id="audit",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2021, 4, 15),
            outcome=True,
            event_id="event-a",
            event_cluster_id="cluster-1",
        ),
        EvidenceRecord(
            source_id="secondary",
            channel_id="news",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2021, 4, 16),
            outcome=True,
            event_id="event-b",
            event_cluster_id="cluster-1",
        ),
    ]
    result = build_evidence_ledger(
        panel=panel,
        records=records,
        columns=columns,
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    assert len(result.ledger) == 1
    assert result.lag_decomposition["accepted_event_count"] == 1
    assert result.lag_decomposition["deduplicated_event_count"] == 1
    records_raw = result.lag_decomposition["records"]
    assert isinstance(records_raw, list)
    lag = cast(dict[str, Any], cast(list[Any], records_raw)[0])
    assert lag["total_lag_days"] == lag["report_lag_days"] + lag["detection_lag_days"]


def test_p04_marks_prospective_rows_immature_without_assigning_outcome() -> None:
    result = build_risk_set(
        panel=pd.DataFrame(
            [
                {"firm_key": "F1", "year_key": 2020, "as_of": "2021-01-01"},
                {"firm_key": "F2", "year_key": 2021, "as_of": "2022-01-01"},
            ]
        ),
        data_cutoff=datetime(2022, 6, 30),
        horizon_months=12,
        columns=_columns(),
    )
    assert result.risk_sets["is_mature"].tolist() == [True, False]
    assert OUTCOME not in result.risk_sets.columns
    assert result.maturity_audit["immature_assigned_negative_count"] == 0


def test_p07_rejects_content_predictor_from_label_model() -> None:
    panel = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31", "text": 1.0}])
    risk = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "is_eligible": True}])
    with pytest.raises(ValueError, match="content predictors cannot enter label models"):
        build_feature_panel(
            firm_year_panel=panel,
            risk_sets=risk,
            feature_definitions=[
                {
                    "feature_id": "content_signal",
                    "physical_column": "text",
                    "role": "content",
                    "allowed_in_label_model": True,
                    "availability_rule": "as_of_prediction_time",
                    "theoretical_block": "K1",
                }
            ],
            columns=_columns(),
        )


def test_p08_batch_is_deterministic_for_same_registered_rng_seed() -> None:
    scenario = {
        "scenario_id": "s1",
        "sample_size": 40,
        "prevalence": 0.2,
        "anchor_sensitivity": 0.8,
        "anchor_false_positive": 0.05,
        "weak_sensitivity": 0.5,
        "weak_false_positive": 0.15,
        "content_signal": 0.7,
    }
    first = run_batch(
        scenario,
        method_id="full",
        replications=range(3),
        rng=np.random.default_rng(42),
    )
    second = run_batch(
        scenario,
        method_id="full",
        replications=range(3),
        rng=np.random.default_rng(42),
    )
    pd.testing.assert_frame_equal(first, second)
    assert not any("pi_bias" in value for value in first["metric_id"].astype(str))


def test_p08_semi_synthetic_pool_uses_development_covariates_only() -> None:
    scenario = {
        "scenario_id": "semi",
        "tier": "semi_synthetic_development_covariates",
        "semi_synthetic_feature_ids": ["content"],
    }
    attached = attach_development_covariate_pools(
        scenarios=[scenario],
        feature_panel=pd.DataFrame(
            {
                "year_key": [2018, 2019, 2021],
                "content": [1.0, 3.0, 1000.0],
            }
        ),
        feature_registry=[{"feature_id": "content", "role": "content"}],
        year_column="year_key",
        development_year_maximum=2019,
    )
    assert attached[0]["semi_synthetic_pool_rows"] == 2
    assert attached[0]["semi_synthetic_development_year_maximum"] == 2019
    assert attached[0]["outer_rows_used_in_pool"] == 0
    assert attached[0]["semi_synthetic_content_pool"] == pytest.approx([-1.0, 1.0])


def test_fixed_pi_l3_ignores_missing_sources_and_adaptive_mcse_reports_actual_status() -> None:
    accuracy = {"anchor": (0.8, 0.99), "weak": (0.6, 0.8)}
    missing_weak = posterior_l3_fixed_pi(
        {"anchor": True, "weak": None}, accuracy, fixed_prevalence=0.05
    )
    anchor_only = posterior_l3_fixed_pi(
        {"anchor": True}, {"anchor": accuracy["anchor"]}, fixed_prevalence=0.05
    )
    assert missing_weak == pytest.approx(anchor_only)
    batch = pd.DataFrame(
        {
            "scenario_id": ["s1", "s1", "s1"],
            "method_id": ["full", "full", "full"],
            "metric_id": ["pass_probability"] * 3,
            "estimate": [0.0, 1.0, 1.0],
        }
    )
    report = summarize_mcse(
        [batch],
        minimum_replications=3,
        maximum_replications=9,
        pass_fail_mcse_maximum=0.01,
    )
    assert report["status"] == "CONTINUE"
    assert report["precision_target_met"] is False


def test_p09_weight_fit_excludes_outer_year_rows() -> None:
    columns = _columns()
    feature_panel = pd.DataFrame(
        [
            {"firm_key": "F1", "year_key": 2020},
            {"firm_key": "F1", "year_key": 2021},
        ]
    )
    risk = pd.DataFrame(
        [
            {"firm_key": "F1", "year_key": 2020, "is_eligible": True, "is_mature": True},
            {"firm_key": "F1", "year_key": 2021, "is_eligible": True, "is_mature": True},
        ]
    )
    result = build_splits_and_weights(
        feature_panel=feature_panel,
        risk_sets=risk,
        matrices={
            "rows": [
                {
                    FIRM_ID: "F1",
                    FISCAL_YEAR: 2020,
                    "channel_outcomes": {"audit": True},
                }
            ]
        },
        observability={
            "channels": {"audit": {"verification_classification": "observed_verification"}}
        },
        outer_years=[2021],
        fold_eligibility=[{OUTER_FOLD: "2021", "assigned_role": "confirmatory"}],
        support_bounds=(0.05, 0.95),
        support_fraction_minimum=0.0,
        ess_fraction_minimum=0.0,
        ess_absolute_minimum=0.0,
        columns=columns,
    )
    assert result.splits[0]["weight_fit_max_year"] == 2020
    assert result.splits[0]["outer_rows_used_in_weight_fit"] == 0
    assert result.weights["2021"]["year_key"].tolist() == [2020]
    assert result.weight_diagnostics["2021"]["fit_scope"] == "development_history"
    assert result.weight_diagnostics["2021"]["outer_rows_used_in_fit"] == 0


def test_p10_selection_uses_development_history_and_removes_held_channel() -> None:
    result = select_measurement(
        matrices={
            "expected_channels": ["audit", "news"],
            "rows": [
                {
                    FISCAL_YEAR: 2020,
                    "channel_outcomes": {"audit": True, "news": False},
                },
                {
                    FISCAL_YEAR: 2021,
                    "channel_outcomes": {"audit": True, "news": True},
                },
            ],
        },
        outer_year=2021,
        candidates=["L2"],
        l3_capability={"status": "UNAVAILABLE_BY_DESIGN", "pilot_executed": False},
    )
    assert result.selection["fit_max_year"] == 2020
    assert result.selection["outer_outcomes_accessed"] is False
    assert result.channel_selection["heldout_channel_removed_from_all_selection_inputs"] is True


def test_p12_calibration_parameters_depend_only_on_development_oof() -> None:
    columns = _columns()
    oof = pd.DataFrame(
        [
            {"firm_key": f"D{i}", "year_key": 2020, "model_key": "m", "score": score}
            for i, score in enumerate([0.1, 0.3, 0.7, 0.9])
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"firm_key": "D0", "year_key": 2020, "binary_result": False},
            {"firm_key": "D1", "year_key": 2020, "binary_result": False},
            {"firm_key": "D2", "year_key": 2020, "binary_result": True},
            {"firm_key": "D3", "year_key": 2020, "binary_result": True},
            {"firm_key": "O1", "year_key": 2021, "binary_result": False},
            {"firm_key": "O2", "year_key": 2021, "binary_result": True},
        ]
    )

    def evaluate(scores: tuple[float, float]) -> dict[str, object]:
        outer = pd.DataFrame(
            [
                {"firm_key": "O1", "year_key": 2021, "model_key": "m", "score": scores[0]},
                {"firm_key": "O2", "year_key": 2021, "model_key": "m", "score": scores[1]},
            ]
        )
        result = evaluate_outer_fold(
            oof_predictions=oof,
            outer_predictions=outer,
            outcomes=outcomes,
            outer_year=2021,
            review_fraction=0.5,
            utility_scenarios=[],
            bootstrap_replications=2,
            confidence_level=0.95,
            columns=columns,
            rng=np.random.default_rng(7),
        )
        models = result.calibration.get("models")
        assert isinstance(models, list)
        return cast(dict[str, object], cast(list[object], models)[0])

    first = evaluate((0.2, 0.8))
    second = evaluate((0.99, 0.01))
    assert first["intercept"] == pytest.approx(second["intercept"])
    assert first["slope"] == pytest.approx(second["slope"])
    assert first["fit_scope"] == "pooled_cross_fitted_development_predictions"


def test_p12_operational_utility_computes_costs_increment_and_uncertainty() -> None:
    columns = _columns()
    oof_rows: list[dict[str, object]] = []
    for model_id in ("m:full", "m:observability_only"):
        for index, (score, truth) in enumerate(
            [(0.1, False), (0.2, False), (0.8, True), (0.9, True)]
        ):
            oof_rows.append(
                {
                    "firm_key": f"D{index}",
                    "year_key": 2020,
                    "model_key": model_id,
                    "score": score,
                    "binary_result": truth,
                }
            )
    oof = pd.DataFrame(oof_rows).drop(columns=["binary_result"])
    outer = pd.DataFrame(
        [
            {"firm_key": "O0", "year_key": 2021, "model_key": "m:full", "score": 0.9},
            {"firm_key": "O1", "year_key": 2021, "model_key": "m:full", "score": 0.1},
            {
                "firm_key": "O0",
                "year_key": 2021,
                "model_key": "m:observability_only",
                "score": 0.1,
            },
            {
                "firm_key": "O1",
                "year_key": 2021,
                "model_key": "m:observability_only",
                "score": 0.9,
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            *[
                {
                    "firm_key": f"D{index}",
                    "year_key": 2020,
                    "binary_result": truth,
                }
                for index, truth in enumerate([False, False, True, True])
            ],
            {"firm_key": "O0", "year_key": 2021, "binary_result": True},
            {"firm_key": "O1", "year_key": 2021, "binary_result": False},
        ]
    )
    result = evaluate_outer_fold(
        oof_predictions=oof,
        outer_predictions=outer,
        outcomes=outcomes,
        outer_year=2021,
        review_fraction=0.5,
        utility_scenarios=[
            {
                "scenario_id": "operational-1",
                "true_positive_benefit": 10.0,
                "review_cost": 1.0,
                "additional_false_positive_cost": 2.0,
                "false_negative_cost": 3.0,
                "measurement_fixed_pi": 0.2,
            }
        ],
        bootstrap_replications=20,
        confidence_level=0.95,
        columns=columns,
        rng=np.random.default_rng(17),
        latent_risk_scenarios={
            "operational-1": [
                {"O0": 0.75, "O1": 0.25},
                {"O0": 0.85, "O1": 0.15},
            ]
        },
    )
    full = next(item for item in result.utility if item[LEARNER_ID] == "m:full")
    assert full["status"] == "PASS"
    assert full["reviewed_cases"] == 1
    assert full["expected_true_positives"] == pytest.approx(0.8)
    assert full["expected_false_positives"] == pytest.approx(0.2)
    assert full["expected_false_negatives"] == pytest.approx(0.2)
    assert full["net_utility"] == pytest.approx(6.0)
    assert full["incremental_utility"] == pytest.approx(9.0)
    uncertainty = cast(dict[str, object], full["utility_uncertainty"])
    assert uncertainty["posterior_parameter_draw_count"] == 2
    assert uncertainty["net_utility_interval"] is not None
    assert uncertainty["incremental_utility_interval"] is not None
    unavailable = evaluate_outer_fold(
        oof_predictions=oof,
        outer_predictions=outer,
        outcomes=outcomes,
        outer_year=2021,
        review_fraction=0.5,
        utility_scenarios=[
            {
                "scenario_id": "operational-1",
                "measurement_fixed_pi": 0.2,
                "true_positive_benefit": 10.0,
                "review_cost": 1.0,
                "additional_false_positive_cost": 2.0,
                "false_negative_cost": 3.0,
            }
        ],
        bootstrap_replications=2,
        confidence_level=0.95,
        columns=columns,
        rng=np.random.default_rng(18),
    )
    assert unavailable.utility[0]["status"] == "INSUFFICIENT_EVIDENCE"
    assert unavailable.utility[0]["reason_code"] == "LATENT_RISK_SCENARIO_UNAVAILABLE"
    assert "net_utility" not in unavailable.utility[0]


def test_gates_fail_closed_when_required_evidence_or_bindings_are_absent() -> None:
    gate2 = gate2_verdict(
        evaluations=[],
        bootstraps=[],
        domain_transfer={"robust_scenario_fraction": None},
        gate={
            "simultaneous_interval_level": 0.95,
            "minimum_meaningful_improvement": {
                "absolute": 0.01,
                "relative_to_reference_ap": 0.05,
            },
            "fold_count": 4,
            "same_direction_min_folds": 3,
            "yield_at_primary_budget_relative_decline_max": 0.1,
            "confirmatory_positive_count_min": 20,
        },
        common={"robust_scenario_fraction_min": 0.8},
        confirmatory_folds=["2021", "2022", "2023", "2024"],
    )
    assert gate2["verdict"] == "INSUFFICIENT_EVIDENCE"
    threshold, gate3 = gate3_verdict(
        gate2={"verdict": "PASS"},
        known_case_results=[],
        feature_panel=pd.DataFrame(),
        predictions=pd.DataFrame(),
        outcomes=pd.DataFrame(),
        bindings={
            "pressure_feature_id": None,
            "monitoring_feature_id": None,
            "parent_model_id": None,
            "domain_feature_id": None,
        },
        gate={},
        confirmatory_folds=[],
        columns=_columns(),
        measurement_selections=[{"selected_measurement": "L2"}] * 4,
    )
    assert threshold["reason_code"] == "INTERACTION_BINDINGS_UNAVAILABLE"
    assert gate3["verdict"] == "INSUFFICIENT_EVIDENCE"
    threshold, gate3 = gate3_verdict(
        gate2={"verdict": "PASS"},
        known_case_results=[],
        feature_panel=pd.DataFrame(),
        predictions=pd.DataFrame(),
        outcomes=pd.DataFrame(),
        bindings={
            "threshold_feature_ids": [],
            "pressure_feature_id": "pressure",
            "monitoring_feature_id": "monitoring",
            "parent_model_id": "model",
            "domain_feature_id": "domain",
        },
        gate={},
        confirmatory_folds=[],
        columns=_columns(),
        measurement_selections=[{"selected_measurement": "L2"}] * 4,
    )
    assert threshold["reason_code"] == "TWO_THRESHOLD_BINDINGS_REQUIRED"
    assert gate3["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_gate3_breakpoint_stability_uses_dispersion_not_distance_from_zero() -> None:
    assert breakpoint_stability_pass([1.49, 1.50, 1.51], 0.02)
    assert not breakpoint_stability_pass([-0.4, 0.0, 0.4], 0.2)
