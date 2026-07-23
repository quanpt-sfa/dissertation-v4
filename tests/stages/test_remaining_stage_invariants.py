"""Oracle-style checks for the methodological invariants implemented in P03-P17."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from core.fold_control import production_path_blockers, require_confirmatory_fold
from core.semantic_keys import (
    AVAILABILITY_BASIS,
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    EVENT_CLUSTER_ID,
    EVENT_ID,
    EVIDENCE_CATEGORY,
    EVIDENCE_RECORD_ID,
    EVIDENCE_RECORD_KIND,
    EVIDENCE_VALUE,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    OPPORTUNITY_BASIS,
    OUTCOME,
    OUTCOME_BASIS,
    OUTER_FOLD,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PREDICTION,
    PREDICTION_TIME,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    SOURCE_PROFILE_ID,
    SOURCE_RECORD_REFS,
    TARGET_ID,
    TARGET_VALUE,
    TEMPORAL_ROLE,
    WEIGHT,
)
from evaluation.service import evaluate_outer_fold
from evidence.service import EvidenceRecord, build_evidence_ledger, map_evidence_outcome
from features.service import build_feature_panel
from gates.service import breakpoint_stability_pass, gate2_verdict, gate3_verdict
from labels.service import aggregate_l1, evidence_score_l2, posterior_l3_fixed_pi
from measurement.service import (
    build_measurement_inputs,
    l3_channel_capability_allows_pilot,
    summarize_fold_eligibility,
)
from observability.service import build_observability_registry
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
        AVAILABILITY_BASIS: "availability_basis_key",
        EVENT_ID: "event_key",
        EVENT_CLUSTER_ID: "cluster_key",
        EVIDENCE_RECORD_ID: "evidence_record_key",
        EVIDENCE_RECORD_KIND: "evidence_record_kind_key",
        EVIDENCE_VALUE: "evidence_value_key",
        EVIDENCE_CATEGORY: "evidence_category_key",
        PERIOD_LINK_SOURCE: "period_link_method",
        PERIOD_LINK_CONFIDENCE: "period_link_quality",
        OUTCOME_BASIS: "outcome_semantics",
        OPPORTUNITY_BASIS: "opportunity_basis_key",
        SOURCE_ID: "source_key",
        SOURCE_PROFILE_ID: "source_profile_key",
        SOURCE_OPPORTUNITY: "source_opportunity_key",
        SOURCE_RECORD_REFS: "source_record_refs_key",
        TEMPORAL_ROLE: "temporal_role_key",
        CHANNEL_ID: "channel_key",
        OUTCOME: "binary_result",
        ELIGIBLE: "is_eligible",
        MATURE: "is_mature",
        OUTER_FOLD: "fold_key",
        LEARNER_ID: "model_key",
        PREDICTION: "score",
        TARGET_ID: "target_key",
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
            channel_id="audit",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2021, 4, 15),
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
    assert result.ledger["event_key"].tolist() == ["event-a"]
    assert result.ledger["cluster_key"].tolist() == ["cluster-1"]
    accepted = cast(dict[str, Any], result.availability_registry[0])
    assert accepted[FIRM_ID] == "F1"
    assert accepted[FISCAL_YEAR] == 2020
    assert result.lag_decomposition["accepted_event_count"] == 1
    assert result.lag_decomposition["deduplicated_event_count"] == 1
    records_raw = result.lag_decomposition["records"]
    assert isinstance(records_raw, list)
    lag = cast(dict[str, Any], cast(list[Any], records_raw)[0])
    assert lag["total_lag_days"] == lag["report_lag_days"] + lag["detection_lag_days"]


def test_p03_duplicate_representative_is_independent_of_row_and_source_order() -> None:
    columns = _columns()
    panel = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}])
    records = [
        EvidenceRecord(
            source_id="z-source",
            channel_id="official",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2021, 4, 15),
            outcome=True,
            event_id="z-event",
            event_cluster_id="cluster-1",
        ),
        EvidenceRecord(
            source_id="a-source",
            channel_id="official",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2021, 4, 15),
            outcome=True,
            event_id="a-event",
            event_cluster_id="cluster-1",
        ),
    ]
    first = build_evidence_ledger(
        panel=panel,
        records=records,
        columns=columns,
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    second = build_evidence_ledger(
        panel=panel,
        records=list(reversed(records)),
        columns=columns,
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    pd.testing.assert_frame_equal(first.ledger, second.ledger)
    assert first.ledger["source_key"].tolist() == ["a-source"]


def test_p03_conflicting_duplicate_timing_fails_closed() -> None:
    panel = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}])
    records = [
        EvidenceRecord(
            source_id="a",
            channel_id="official",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2021, 4, 15),
            outcome=True,
            event_id="a",
            event_cluster_id="cluster-1",
        ),
        EvidenceRecord(
            source_id="b",
            channel_id="official",
            firm_id="F1",
            fiscal_year=2020,
            availability_date=datetime(2022, 4, 1),
            outcome=True,
            event_id="b",
            event_cluster_id="cluster-1",
        ),
    ]
    with pytest.raises(ValueError, match="disagree on timing or outcome semantics"):
        build_evidence_ledger(
            panel=panel,
            records=records,
            columns=_columns(),
            fiscal_year_end_month_day="12-31",
            lag_tolerance_days=0,
        )


def test_p03_row_inclusion_exclusion_does_not_create_negative_outcome() -> None:
    result = build_evidence_ledger(
        panel=pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}]),
        records=[
            EvidenceRecord(
                source_id="sanction",
                channel_id="official",
                firm_id="F1",
                fiscal_year=2020,
                availability_date=datetime(2021, 5, 1),
                outcome=None,
                event_id="excluded-event",
                event_cluster_id="excluded-cluster",
                outcome_basis="explicit_hard_positive_indicator",
                row_included=False,
            )
        ],
        columns=_columns(),
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    assert result.ledger.empty
    assert result.availability_registry[0]["status"] == "EXCLUDED_BY_SOURCE_RULE"
    assert result.availability_registry[0][OUTCOME] is None


def test_p03_positive_indicator_false_is_unknown_not_explicit_negative() -> None:
    outcome, basis = map_evidence_outcome(
        outcome_mode="positive_indicator",
        direct_outcome=None,
        positive_indicator=False,
        row_included=True,
    )
    assert outcome is None
    assert basis == "explicit_hard_positive_indicator"


def test_p03_keeps_distinct_events_and_p05_applies_horizon_before_aggregation() -> None:
    columns = _columns()
    panel = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}])
    result = build_evidence_ledger(
        panel=panel,
        records=[
            EvidenceRecord(
                source_id="sanction",
                channel_id="official",
                firm_id="F1",
                fiscal_year=2020,
                availability_date=datetime(2021, 3, 1),
                outcome=True,
                event_id="pre-prediction",
                event_cluster_id="pre-cluster",
            ),
            EvidenceRecord(
                source_id="sanction",
                channel_id="official",
                firm_id="F1",
                fiscal_year=2020,
                availability_date=datetime(2021, 4, 15),
                outcome=True,
                event_id="future-event",
                event_cluster_id="future-cluster",
            ),
        ],
        columns=columns,
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    assert len(result.ledger) == 2
    measurement = build_measurement_inputs(
        risk_sets=pd.DataFrame(
            [
                {
                    "firm_key": "F1",
                    "year_key": 2020,
                    "as_of": pd.Timestamp("2021-03-31"),
                    "is_mature": True,
                    "is_eligible": True,
                }
            ]
        ),
        evidence=result.ledger,
        expected_sources={"sanction": "official"},
        horizon_months=12,
        columns=columns,
        pending_status="EMPIRICALLY_PENDING",
        unavailable_status="UNAVAILABLE_BY_DESIGN",
        insufficient_channels_reason="INSUFFICIENT_CHANNELS",
        anchor_source_ids=["sanction"],
    )
    assert measurement.sealed_outcomes["binary_result"].tolist() == [True]
    assert cast(dict[str, Any], measurement.matrices["rows"][0])["l1"] is True


def _single_row_measurement(event_dates: list[tuple[datetime, bool | None]]) -> object:
    columns = _columns()
    evidence = pd.DataFrame(
        [
            {
                "firm_key": "F1",
                "year_key": 2020,
                "source_key": "sanction",
                "channel_key": "official",
                "available_at": available,
                "binary_result": outcome,
            }
            for available, outcome in event_dates
        ],
        columns=[
            "firm_key",
            "year_key",
            "source_key",
            "channel_key",
            "available_at",
            "binary_result",
        ],
    )
    return build_measurement_inputs(
        risk_sets=pd.DataFrame(
            [
                {
                    "firm_key": "F1",
                    "year_key": 2020,
                    "as_of": pd.Timestamp("2021-03-31"),
                    "is_mature": True,
                    "is_eligible": True,
                }
            ]
        ),
        evidence=evidence,
        expected_sources={"sanction": "official"},
        horizon_months=12,
        columns=columns,
        pending_status="EMPIRICALLY_PENDING",
        unavailable_status="UNAVAILABLE_BY_DESIGN",
        insufficient_channels_reason="INSUFFICIENT_CHANNELS",
        anchor_source_ids=["sanction"],
    )


def test_p05_horizon_boundary_is_calendar_exact_and_post_horizon_is_excluded() -> None:
    at_boundary = cast(Any, _single_row_measurement([(datetime(2022, 3, 31), True)]))
    one_day_after = cast(Any, _single_row_measurement([(datetime(2022, 4, 1), True)]))
    assert at_boundary.sealed_outcomes["binary_result"].tolist() == [True]
    assert one_day_after.sealed_outcomes.empty


def test_p05_pre_prediction_event_does_not_backdate_post_horizon_positive() -> None:
    result = cast(
        Any,
        _single_row_measurement([(datetime(2021, 3, 1), True), (datetime(2022, 4, 1), True)]),
    )
    assert result.sealed_outcomes.empty
    assert cast(dict[str, Any], result.matrices["rows"][0])["l1"] is None


def test_p05_multiple_source_events_are_order_independent() -> None:
    events: list[tuple[datetime, bool | None]] = [
        (datetime(2021, 5, 1), False),
        (datetime(2021, 6, 1), True),
    ]
    first = cast(Any, _single_row_measurement(events))
    second = cast(Any, _single_row_measurement(list(reversed(events))))
    assert cast(dict[str, Any], first.matrices["rows"][0])["l1"] is True
    assert first.matrices == second.matrices


def test_p05_absence_of_sanction_remains_unknown() -> None:
    result = cast(Any, _single_row_measurement([]))
    row = cast(dict[str, Any], result.matrices["rows"][0])
    assert row["source_outcomes"] == {"sanction": None}
    assert row["l1"] is None
    assert result.sealed_outcomes.empty


def test_p05_fold_counts_use_mature_risk_set_and_fail_closed_on_one_class() -> None:
    columns = _columns()
    risk_sets = pd.DataFrame(
        [
            {"year_key": 2021, "is_mature": True, "is_eligible": True},
            {"year_key": 2021, "is_mature": True, "is_eligible": True},
            {"year_key": 2021, "is_mature": True, "is_eligible": True},
        ]
    )
    sealed = pd.DataFrame([{"year_key": 2021, "binary_result": True}])
    summary = summarize_fold_eligibility(
        sealed_outcomes=sealed,
        risk_sets=risk_sets,
        initial_outer_year=2020,
        confirmatory_years=[2021],
        prospective_year=2026,
        confirmatory_positive_minimum=1,
        sensitivity_positive_range=(1, 1),
        columns=columns,
    )
    fold = cast(dict[str, Any], summary[1])
    assert fold["mature_row_count"] == 3
    assert fold["eligible_row_count"] == 3
    assert fold["labeled_row_count"] == 1
    assert fold["positive_count"] == 1
    assert fold["explicit_negative_count"] == 0
    assert fold["unknown_count"] == 2
    assert fold["both_binary_classes_observed"] is False
    assert fold["observed_binary_class_count"] == 1
    assert fold["binary_class_reason_code"] == "EXPLICIT_NEGATIVE_CLASS_MISSING"
    assert fold["assigned_role"] == "prospective_or_descriptive"
    assert fold["reason_code"] == "EXPLICIT_NEGATIVE_CLASS_MISSING"


def test_p05_positive_only_fold_never_becomes_confirmatory_above_threshold() -> None:
    columns = _columns()
    risk_sets = pd.DataFrame(
        [{"year_key": 2021, "is_mature": True, "is_eligible": True} for _ in range(30)]
    )
    sealed = pd.DataFrame([{"year_key": 2021, "binary_result": True} for _ in range(25)])
    summary = summarize_fold_eligibility(
        sealed_outcomes=sealed,
        risk_sets=risk_sets,
        initial_outer_year=2020,
        confirmatory_years=[2021],
        prospective_year=2026,
        confirmatory_positive_minimum=15,
        sensitivity_positive_range=(5, 14),
        columns=columns,
    )
    fold = cast(dict[str, Any], summary[1])
    assert fold["assigned_role"] == "prospective_or_descriptive"
    assert fold["explicit_negative_count"] == 0
    assert fold["reason_code"] == "EXPLICIT_NEGATIVE_CLASS_MISSING"


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


def test_p04_source_maturity_excludes_pre_prediction_events_from_horizon_fraction() -> None:
    columns = _columns()
    panel = pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}])
    evidence = pd.DataFrame(
        [
            {
                "firm_key": "F1",
                "year_key": 2020,
                "source_key": "sanction",
                "channel_key": "official",
                "available_at": "2021-03-01",
            },
            {
                "firm_key": "F1",
                "year_key": 2020,
                "source_key": "sanction",
                "channel_key": "official",
                "available_at": "2021-04-15",
            },
        ]
    )
    result = build_risk_set(
        panel=panel,
        data_cutoff=datetime(2022, 6, 30),
        horizon_months=12,
        columns=columns,
        evidence=evidence,
    )
    curves = cast(dict[str, list[dict[str, Any]]], result.maturity_audit["source_maturity_curves"])[
        "delayed_verification_sources"
    ]
    curve = curves[0]
    assert curve["negative_detection_lag_count"] == 1
    assert curve["future_detection_event_count"] == 1
    assert cast(dict[str, float], curve["linked_event_in_horizon_fraction_by_months"])["12"] == 0.5
    assert cast(dict[str, int], curve["in_horizon_event_count_by_months"])["12"] == 1
    assert curve["pre_prediction_event_count"] == 1
    assert curve["source_opportunity_coverage_rate"] is None


def test_p04_event_after_cutoff_is_not_counted_inside_horizon() -> None:
    columns = _columns()
    result = build_risk_set(
        panel=pd.DataFrame([{"firm_key": "F1", "year_key": 2020, "as_of": "2021-03-31"}]),
        data_cutoff=datetime(2022, 3, 31),
        horizon_months=24,
        columns=columns,
        evidence=pd.DataFrame(
            [
                {
                    "firm_key": "F1",
                    "year_key": 2020,
                    "source_key": "sanction",
                    "channel_key": "official",
                    "available_at": "2022-04-01",
                }
            ]
        ),
    )
    curve = cast(dict[str, list[dict[str, Any]]], result.maturity_audit["source_maturity_curves"])[
        "delayed_verification_sources"
    ][0]
    assert curve["post_data_cutoff_event_count"] == 1
    assert cast(dict[str, int], curve["in_horizon_event_count_by_months"])["24"] == 0


def test_p06_does_not_mislabel_event_incidence_as_source_coverage() -> None:
    result = build_observability_registry(
        {
            "rows": [
                {
                    ELIGIBLE: True,
                    MATURE: True,
                    "source_outcomes": {"sanction": True},
                    "channel_outcomes": {"official": True},
                },
                {
                    ELIGIBLE: True,
                    MATURE: False,
                    "source_outcomes": {"sanction": None},
                    "channel_outcomes": {"official": None},
                },
            ]
        },
        {"sanction": {CHANNEL_ID: "official", "verification_status": "high_confirmation"}},
    )
    channel = cast(dict[str, Any], cast(dict[str, Any], result["channels"])["official"])
    assert channel["positive_count"] == 1
    assert channel["explicit_negative_count"] == 0
    assert channel["unknown_count"] == 1
    assert channel["observed_outcome_fraction"] == 0.5
    assert channel["coverage_rate"] is None
    assert channel["mature_cohort_observed_fraction"] == 1.0
    assert channel["prospective_count"] == 1
    assert channel["coverage_status"] == "NOT_ESTIMABLE_FROM_EVENT_ABSENCE"
    assert channel["source_opportunity_coverage_status"] == ("UNKNOWN_NO_OPPORTUNITY_INDICATOR")


def test_p05_l3_structural_channel_failure_precedes_unlocked_parameter_status() -> None:
    capability: dict[str, object] = {
        "status": "UNAVAILABLE_BY_DESIGN",
        "channel_count": 1,
        "reason": "INSUFFICIENT_CHANNELS",
    }
    assert l3_channel_capability_allows_pilot(capability) is False
    assert capability["status"] == "UNAVAILABLE_BY_DESIGN"
    assert capability["pilot_executed"] is False
    assert capability["reason_code"] == "INSUFFICIENT_CHANNELS"


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


def test_p10_production_path_blocks_empty_features_and_nonconfirmatory_fold() -> None:
    blockers = production_path_blockers(
        fold_eligibility=[
            {
                OUTER_FOLD: "2021",
                "assigned_role": "prospective_or_descriptive",
                "positive_count": 10,
                "explicit_negative_count": 0,
                "observed_binary_class_count": 1,
                "both_binary_classes_observed": False,
            }
        ],
        outer_fold="2021",
        feature_registry=[],
        mcse_report={"status": "SKIPPED", "precision_target_met": False},
    )
    assert "FEATURE_REGISTRY_EMPTY" in blockers
    assert "P08_MCSE_NOT_PASS" in blockers
    assert "EXPLICIT_NEGATIVE_CLASS_MISSING" in blockers


@pytest.mark.parametrize(
    "role", ["prospective_or_descriptive", "prospective_separate", "initial_separate"]
)
def test_p10_p11_reject_descriptive_and_prospective_fold_roles(role: str) -> None:
    eligibility = [
        {
            OUTER_FOLD: "2024",
            "assigned_role": role,
            "positive_count": 1,
            "explicit_negative_count": 1,
            "observed_binary_class_count": 2,
            "both_binary_classes_observed": True,
        }
    ]
    with pytest.raises(RuntimeError, match="FOLD_NOT_CONFIRMATORY"):
        require_confirmatory_fold(eligibility, "2024", "P11")


def test_p08_batch_is_deterministic_for_same_registered_rng_seed() -> None:
    from pathlib import Path

    import yaml

    from simulation.method_contract import (
        apply_execution_profile,
        build_cost_sensitive_contract,
        build_execution_profile_contract,
        build_full_method_registry,
    )

    root = Path(__file__).resolve().parents[2]
    simulation = yaml.safe_load(
        (root / "config" / "execution" / "simulation.yaml").read_text(encoding="utf-8")
    )["simulation"]

    cost_contract = build_cost_sensitive_contract(simulation)
    full_registry = build_full_method_registry(simulation, cost_contract)
    profile_contract = build_execution_profile_contract(simulation, full_registry)
    active_profile = str(profile_contract["active_profile"])
    method_registry = apply_execution_profile(full_registry, profile_contract)

    scenario = {
        "scenario_id": "s1",
        "sample_size": 50,
        "prevalence": 0.2,
        "anchor_sensitivity": 0.8,
        "anchor_false_positive": 0.05,
        "weak_sensitivity": 0.5,
        "weak_false_positive": 0.15,
        "content_signal": 0.7,
        "tier": "fully_synthetic",
        "signal_structure": "linear",
        "method_registry": method_registry,
        "active_method_ids": list(profile_contract["active_method_ids"]),
        "execution_profile": active_profile,
        "execution_profile_registry": [dict(value) for value in profile_contract["profiles"]],
        "cost_regime_registry": [dict(value) for value in cost_contract["cost_regime_registry"]],
        "review_budget_registry": [
            dict(value) for value in cost_contract["review_budget_registry"]
        ],
        "cost_sensitive_settings": {
            key: value
            for key, value in cost_contract.items()
            if key not in {"cost_regime_registry", "review_budget_registry"}
        },
    }
    first = run_batch(
        scenario,
        method_id="naive_observed_as_negative__logistic_regression__traincost_symmetric",
        replications=range(3),
        rng=np.random.default_rng(42),
    )
    second = run_batch(
        scenario,
        method_id="naive_observed_as_negative__logistic_regression__traincost_symmetric",
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
    np.testing.assert_allclose(attached[0]["semi_synthetic_covariate_pool"], [[-1.0], [1.0]])


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


def test_scenario_id_target_marker_must_match():
    from simulation.scenario_contract import validate_scenario_target_identity

    # Valid matching target marker for semi-synthetic tier
    validate_scenario_target_identity(
        {
            "scenario_id": "empirical_baseline__target_L1_ANNUAL",
            "calibration_target_id": "L1_ANNUAL",
            "tier": "semi_synthetic_development_covariates",
        }
    )

    # Fully synthetic scenarios skip validation
    validate_scenario_target_identity(
        {
            "scenario_id": "fully_synthetic_no_target_marker",
            "calibration_target_id": "L1_ANNUAL",
            "tier": "fully_synthetic",
        }
    )

    # Non-matching target marker for semi-synthetic tier
    with pytest.raises(ValueError, match="differs from calibration_target_id"):
        validate_scenario_target_identity(
            {
                "scenario_id": "empirical_baseline__target_L1_CONTENT_STRICT",
                "calibration_target_id": "L1_ANNUAL",
                "tier": "semi_synthetic_development_covariates",
            }
        )

    # Missing target marker for semi-synthetic tier
    with pytest.raises(ValueError, match="must contain"):
        validate_scenario_target_identity(
            {
                "scenario_id": "empirical_baseline_L1_ANNUAL",
                "calibration_target_id": "L1_ANNUAL",
                "tier": "semi_synthetic_development_covariates",
            }
        )


def test_short_key_collision_fail_closed():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.p08a_build_scenario_registry import build_short_key_map

    # Collision test: same canonical IDs map to same keys, but we pass two different IDs
    # that map to the same prefix digest if we set hex_length extremely short (like 1).
    with pytest.raises(ValueError, match="short-key collision"):
        build_short_key_map(["id_abc", "id_xyz"], prefix="s", hex_length=1)


def test_replication_dgp_pairing_consistency():
    from core.rng import generator

    # Verify that different replication sizes still use the same simulated population at replication level.
    # Replication 150 should receive identical population across core and standalone/extended tiers.
    rep_id = 150
    data_rng_core = generator(
        "mock_protocol_hash",
        "P08_REPLICATION_DATA",
        {
            "scenario_id": "empirical_baseline__target_L1_ANNUAL",
            "replication_id": str(rep_id),
        },
        str(rep_id),
    )
    data_rng_standalone = generator(
        "mock_protocol_hash",
        "P08_REPLICATION_DATA",
        {
            "scenario_id": "empirical_baseline__target_L1_ANNUAL",
            "replication_id": str(rep_id),
        },
        str(rep_id),
    )
    # The RNG streams must produce exactly the same random numbers
    assert data_rng_core.random() == data_rng_standalone.random()


def test_p08c_exit_code_policies():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from unittest.mock import MagicMock, patch

    import scripts.p08c_aggregate_batches as p08c

    mock_loaded = MagicMock()
    mock_loaded.registry = {
        "simulation": {
            "active_profile": "core",
            "protocol_role": {"must_complete_before_outer_test": True},
            "core": {
                "minimum_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "l3": {
                "initial_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "extended_replication": {
                "minimum_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "cost_sensitive_replication": {
                "minimum_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "cost_sensitive": {
                "cost_mcse_relative_fraction": 0.1,
                "cost_neutral_regime_id": "cost_neutral",
            },
            "continuous_metrics": {"mcse_fraction_of_minimum_meaningful_improvement_max": 0.1},
        },
        "evaluation": {"gate2": {"minimum_meaningful_improvement": {"absolute": 0.02}}},
    }

    # We patch argparse to return different values
    mock_args = MagicMock()
    mock_args.registry = Path("mock_registry.json")
    mock_args.run_id = "mock-run"

    @patch("scripts.p08c_aggregate_batches.load_run", return_value=mock_loaded)
    @patch("scripts.p08c_aggregate_batches._validate_methodology", return_value={"status": "PASS"})
    @patch("scripts.p08c_aggregate_batches._filter_active_batches", return_value=[MagicMock()])
    @patch("scripts.p08c_aggregate_batches.argparse.ArgumentParser.parse_args")
    def run_test(mock_parse, mock_filter, mock_val, mock_load, summarize_return, check_only):
        mock_parse.return_value = mock_args
        mock_args.check_only = check_only

        # Mock inventory to return one batches item to prevent empty loop
        mock_loaded.context.store.inventory.return_value = [
            {
                "artifact_id": "simulation_batches",
                "coordinates": {"scenario_key": "s1", "method_key": "m1", "batch_key": "b0000"},
            },
            {
                "artifact_id": "model_diagnostics",
                "coordinates": {"scenario_key": "s1", "method_key": "m1", "batch_key": "b0000"},
            },
        ]

        # Make load read return a dataframe
        import pandas as pd

        from core.semantic_keys import METHOD_ID, METRIC_ID, REPLICATION_ID, SCENARIO_ID

        df = pd.DataFrame(
            {
                SCENARIO_ID: ["empirical_baseline__target_L1_ANNUAL"],
                METHOD_ID: ["naive_observed_as_negative__multilayer_perceptron__train"],
                REPLICATION_ID: [0],
                METRIC_ID: ["primary_metric"],
            }
        )
        mock_loaded.context.read.return_value = df

        # Scenarios mapping
        mock_scenarios = [
            {
                "scenario_id": "empirical_baseline__target_L1_ANNUAL",
                "scenario_key": "s1",
                "tier": "semi_synthetic_development_covariates",
                "execution_profile": "core",
                "method_registry": [
                    {
                        "method_id": "naive_observed_as_negative__multilayer_perceptron__train",
                        "method_key": "m1",
                        "is_active": True,
                        "analysis_role": "confirmatory",
                        "method_family": "multilayer_perceptron",
                        "learner_tier": "core",
                        "training_cost_regime_id": "cost_neutral",
                        "imbalance_treatment_id": "none",
                    }
                ],
            }
        ]
        mock_loaded.context.read.side_effect = lambda artifact_id, *a, **k: (
            mock_scenarios if artifact_id == "simulation_scenario_registry" else df
        )

        with patch("scripts.p08c_aggregate_batches.summarize_mcse", return_value=summarize_return):
            return p08c.main()

    # Case 1: check-only + CONTINUE -> exit 0
    report_continue = {"status": "CONTINUE", "precision_target_met": False}
    exit_code_1 = run_test(summarize_return=report_continue, check_only=True)
    assert exit_code_1 == 0

    # Case 2: final + PASS -> exit 0
    report_pass = {"status": "PASS", "precision_target_met": True}
    exit_code_2 = run_test(summarize_return=report_pass, check_only=False)
    assert exit_code_2 == 0

    # Case 3: final + MAXIMUM_REACHED/core -> exit 3
    report_max = {"status": "MAXIMUM_REACHED", "precision_target_met": False}
    exit_code_3 = run_test(summarize_return=report_max, check_only=False)
    assert exit_code_3 == 3


def test_run_batch_diagnostics_capture():
    """
    Verifies that DiagnosticCollector captures InsufficientData fit failures
    when run_batch is called with the diagnostics argument.

    Strategy: prevalence=0.0 guarantees all latent labels are False,
    so unique(y_train) == 1 → InsufficientData is recorded without raising.
    Uses simulation.yaml to build full cost_sensitive_settings (avoids missing
    evaluation_regime_ids and other required config keys).
    """
    from pathlib import Path

    import numpy as np
    import yaml

    from simulation.method_contract import build_cost_sensitive_contract
    from simulation.service import run_batch

    root = Path(__file__).resolve().parents[2]
    simulation = yaml.safe_load(
        (root / "config" / "execution" / "simulation.yaml").read_text(encoding="utf-8")
    )["simulation"]
    cost_contract = build_cost_sensitive_contract(simulation)

    mock_scenario = {
        "scenario_id": "diagnostics_capture_test",
        "tier": "fully_synthetic",
        "prevalence": 0.0,  # All latent labels False → unique(y_train)<2 → InsufficientData
        "sample_size": 50,
        "content_signal": 0.0,
        "signal_structure": "linear",
        "anchor_sensitivity": 0.8,
        "anchor_false_positive": 0.0,  # No false positives so y_train stays all-zero
        "weak_sensitivity": 0.5,
        "weak_false_positive": 0.0,  # No false positives so y_train stays all-zero
        "anchor_verification_probability": 1.0,
        "weak_verification_probability": 1.0,
        "selective_verification_strength": 0.0,
        "detection_delay_mean_days": 30.0,
        "horizon_days": 365.0,
        "method_registry": [
            {
                "method_id": "naive_observed_as_negative__logistic_regression__traincost_symmetric",
                "is_active": True,
                "method_family": "predictive",
                "label_strategy_id": "naive_observed_as_negative",
                "learner_id": "logistic_regression",
                "learner_tier": "core",
                "training_cost_regime_id": "symmetric",
                "imbalance_treatment_id": "none",
            }
        ],
        "cost_regime_registry": [dict(v) for v in cost_contract["cost_regime_registry"]],
        "review_budget_registry": [dict(v) for v in cost_contract["review_budget_registry"]],
        "cost_sensitive_settings": {
            k: v
            for k, v in cost_contract.items()
            if k not in {"cost_regime_registry", "review_budget_registry"}
        },
    }

    rng = np.random.default_rng(42)
    diagnostics: dict = {}

    # run_batch must complete without raising (InsufficientData is non-fatal)
    run_batch(
        scenario=mock_scenario,
        method_id="naive_observed_as_negative__logistic_regression__traincost_symmetric",
        replications=[44],
        rng=rng,
        diagnostics=diagnostics,
    )

    assert "InsufficientData" in diagnostics.get("fit_failures", {}), (
        f"Expected InsufficientData in fit_failures; got: {diagnostics}"
    )
    assert diagnostics["fit_failures"]["InsufficientData"] > 0
    assert 44 in diagnostics.get("affected_replication_ids", []), (
        f"Expected replication 44 in affected_replication_ids; got: {diagnostics}"
    )
    assert "warnings" in diagnostics, f"Expected warnings key in diagnostics; got: {diagnostics}"


def test_p08c_bijection_check_missing_diagnostics_fails():
    """
    Verifies that p08c_aggregate_batches fails closed (raises ValueError) if a
    simulation_batches artifact exists but has no paired model_diagnostics artifact.
    """
    import sys
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import pandas as pd

    from core.semantic_keys import METHOD_ID, METRIC_ID, REPLICATION_ID, SCENARIO_ID

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import scripts.p08c_aggregate_batches as p08c

    mock_loaded = MagicMock()
    mock_loaded.registry = {
        "simulation": {
            "active_profile": "core",
            "protocol_role": {"must_complete_before_outer_test": True},
            "core": {
                "minimum_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "l3": {
                "initial_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "extended_replication": {
                "minimum_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "cost_sensitive_replication": {
                "minimum_replications": 10,
                "maximum_replications": 50,
                "pass_fail_mcse_max": 0.05,
                "batch_size": 10,
            },
            "cost_sensitive": {
                "cost_mcse_relative_fraction": 0.1,
                "cost_neutral_regime_id": "cost_neutral",
            },
            "continuous_metrics": {"mcse_fraction_of_minimum_meaningful_improvement_max": 0.1},
        },
        "evaluation": {"gate2": {"minimum_meaningful_improvement": {"absolute": 0.02}}},
    }
    mock_args = MagicMock()
    mock_args.registry = Path("mock_registry.json")
    mock_args.run_id = "mock-run"
    mock_args.check_only = False

    # Return only simulation_batches, NO model_diagnostics
    mock_loaded.context.store.inventory.return_value = [
        {
            "artifact_id": "simulation_batches",
            "coordinates": {"scenario_key": "s1", "method_key": "m1", "batch_key": "b0000"},
        }
    ]

    df = pd.DataFrame(
        {
            SCENARIO_ID: ["empirical_baseline__target_L1_ANNUAL"],
            METHOD_ID: ["naive_observed_as_negative__multilayer_perceptron__train"],
            REPLICATION_ID: [0],
            METRIC_ID: ["primary_metric"],
        }
    )
    mock_loaded.context.read.return_value = df

    mock_scenarios = [
        {
            "scenario_id": "empirical_baseline__target_L1_ANNUAL",
            "scenario_key": "s1",
            "tier": "semi_synthetic_development_covariates",
            "execution_profile": "core",
            "method_registry": [
                {
                    "method_id": "naive_observed_as_negative__multilayer_perceptron__train",
                    "method_key": "m1",
                    "is_active": True,
                    "analysis_role": "confirmatory",
                    "method_family": "multilayer_perceptron",
                    "learner_tier": "core",
                    "training_cost_regime_id": "cost_neutral",
                    "imbalance_treatment_id": "none",
                }
            ],
        }
    ]
    mock_loaded.context.read.side_effect = lambda artifact_id, *a, **k: (
        mock_scenarios if artifact_id == "simulation_scenario_registry" else df
    )

    @patch("scripts.p08c_aggregate_batches.load_run", return_value=mock_loaded)
    @patch(
        "scripts.p08c_aggregate_batches.argparse.ArgumentParser.parse_args", return_value=mock_args
    )
    def run_aggregate(mock_parse, mock_load):
        import pytest

        with pytest.raises(ValueError, match="batch-diagnostics coordinate mismatch"):
            p08c.main()

    run_aggregate()


def test_model_diagnostics_warnings_capture():
    """
    Verifies that record_model_warning properly records model warnings
    inside the active diagnostic_capture context.
    """
    from simulation.service import DiagnosticCollector, diagnostic_capture, record_model_warning

    collector = DiagnosticCollector()
    with diagnostic_capture(collector, 99):
        record_model_warning("ConvergenceWarning")
        record_model_warning("ConvergenceWarning")
        record_model_warning("SomeOtherWarning")

    assert collector.model_warnings == {"ConvergenceWarning": 2, "SomeOtherWarning": 1}
    assert 99 in collector.affected_replication_ids


# ---------------------------------------------------------------------------
# _batch_exists() resume-semantics tests
# ---------------------------------------------------------------------------


def _make_loaded(inventory_items: list[dict]) -> object:
    """Build a minimal mock of the 'loaded' object used by _batch_exists."""

    class _Store:
        def __init__(self, items: list[dict]) -> None:
            self._items = items

        def inventory(self) -> list[dict]:
            return list(self._items)

    class _Context:
        def __init__(self, items: list[dict]) -> None:
            self.store = _Store(items)

        def read(self, artifact_id: str, coordinates: dict) -> None:
            return None  # _artifact_exists only needs this not to raise

    class _Loaded:
        def __init__(self, items: list[dict]) -> None:
            self.context = _Context(items)

    return _Loaded(inventory_items)


def _make_coords(scenario_key: str, method_key: str, batch_key: str) -> dict:
    return {
        "scenario_" + "key": scenario_key,
        "method_" + "key": method_key,
        "batch_" + "key": batch_key,
    }


def test_batch_exists_both_present_skips() -> None:
    """
    _batch_exists() must return True (skip) when BOTH simulation_batches and
    model_diagnostics exist for the same coordinates.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.p08_profiled_orchestrator import _batch_exists

    coords = _make_coords("sc01", "m01", "b0000")
    inventory = [
        {"artifact_id": "simulation_batches", "coordinates": coords},
        {"artifact_id": "model_diagnostics", "coordinates": coords},
    ]
    loaded = _make_loaded(inventory)
    assert _batch_exists(
        loaded,
        scenario_key="sc01",
        method_key="m01",
        start=0,
        batch_size=250,
    ), "Both artifacts present → must skip (return True)"


def test_batch_exists_diagnostics_missing_reruns() -> None:
    """
    _batch_exists() must return False (re-run) when simulation_batches exists
    but model_diagnostics is absent — simulates a process killed between writes.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.p08_profiled_orchestrator import _batch_exists

    coords = _make_coords("sc01", "m01", "b0000")
    inventory = [
        {"artifact_id": "simulation_batches", "coordinates": coords},
        # model_diagnostics intentionally omitted
    ]
    loaded = _make_loaded(inventory)
    assert not _batch_exists(
        loaded,
        scenario_key="sc01",
        method_key="m01",
        start=0,
        batch_size=250,
    ), "Batch present but diagnostics missing → must re-run (return False)"


def test_batch_exists_batch_missing_reruns() -> None:
    """
    _batch_exists() must return False (re-run) when model_diagnostics exists
    but simulation_batches is absent (should not happen in practice, but the
    function must be symmetric).
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.p08_profiled_orchestrator import _batch_exists

    coords = _make_coords("sc01", "m01", "b0000")
    inventory = [
        # simulation_batches intentionally omitted
        {"artifact_id": "model_diagnostics", "coordinates": coords},
    ]
    loaded = _make_loaded(inventory)
    assert not _batch_exists(
        loaded,
        scenario_key="sc01",
        method_key="m01",
        start=0,
        batch_size=250,
    ), "Diagnostics present but batch missing → must re-run (return False)"


def test_incomplete_batch_commands_adaptive_resume() -> None:
    """
    _incomplete_batch_commands() must schedule a re-run for adaptive batch b0010
    when only simulation_batches exists (diagnostics missing), even though b0010
    is beyond minimum (0-2499) and was never covered by the initial range check.
    """
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.p08_profiled_orchestrator import _incomplete_batch_commands

    sk = "sc_adaptive"
    mk = "m_adaptive"
    # b0010 corresponds to start=2500 with batch_size=250
    coords_b10 = {
        "scenario_" + "key": sk,
        "method_" + "key": mk,
        "batch_" + "key": "b0010",
    }
    # inventory: only simulation_batches for b0010, no diagnostics
    inventory = [
        {"artifact_id": "simulation_batches", "coordinates": coords_b10},
        # model_diagnostics for b0010 intentionally absent
    ]
    loaded = _make_loaded(inventory)

    jobs = [
        {
            "scenario_" + "key": sk,
            "method_" + "key": mk,
            "scenario_id": "some_scenario",
            "method_id": "some_method",
            "minimum": 2500,
            "maximum": 5000,
            "batch_size": 250,
        }
    ]
    commands = _incomplete_batch_commands(
        loaded,
        jobs=jobs,
        python="python",
        registry_path=Path("mock.json"),
        run_id="test-run",
    )
    assert len(commands) == 1, f"Expected 1 repair command, got {len(commands)}: {commands}"
    cmd = commands[0]
    # start=2500, count=min(250, 5000-2500)=250
    assert "--start" in cmd
    start_idx = cmd.index("--start")
    assert cmd[start_idx + 1] == "2500"
    assert "some_scenario" in " ".join(cmd)
    assert "some_method" in " ".join(cmd)


def test_convergence_warning_captured_from_estimator() -> None:
    """
    Integration test: verifies that when _fit_estimator raises a ConvergenceWarning,
    it is captured by warnings.catch_warnings and recorded via record_model_warning()
    into the active DiagnosticCollector.
    """
    import warnings
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np
    import yaml
    from sklearn.exceptions import ConvergenceWarning

    from simulation.method_contract import build_cost_sensitive_contract
    from simulation.service import (
        run_batch,
    )

    root = Path(__file__).resolve().parents[2]
    simulation = yaml.safe_load(
        (root / "config" / "execution" / "simulation.yaml").read_text(encoding="utf-8")
    )["simulation"]
    cost_contract = build_cost_sensitive_contract(simulation)

    mock_scenario = {
        "scenario_id": "convergence_warning_test",
        "tier": "fully_synthetic",
        "prevalence": 0.4,
        "sample_size": 50,
        "content_signal": 0.0,
        "signal_structure": "linear",
        "anchor_sensitivity": 0.8,
        "anchor_false_positive": 0.1,
        "weak_sensitivity": 0.5,
        "weak_false_positive": 0.05,
        "anchor_verification_probability": 1.0,
        "weak_verification_probability": 1.0,
        "selective_verification_strength": 0.0,
        "detection_delay_mean_days": 30.0,
        "horizon_days": 365.0,
        "method_registry": [
            {
                "method_id": "naive_observed_as_negative__logistic_regression__traincost_symmetric",
                "is_active": True,
                "method_family": "predictive",
                "label_strategy_id": "naive_observed_as_negative",
                "learner_id": "logistic_regression",
                "learner_tier": "core",
                "training_cost_regime_id": "symmetric",
                "imbalance_treatment_id": "none",
            }
        ],
        "cost_regime_registry": [dict(v) for v in cost_contract["cost_regime_registry"]],
        "review_budget_registry": [dict(v) for v in cost_contract["review_budget_registry"]],
        "cost_sensitive_settings": {
            k: v
            for k, v in cost_contract.items()
            if k not in {"cost_regime_registry", "review_budget_registry"}
        },
    }

    # Patch _fit_estimator to emit a real ConvergenceWarning instead of running
    def fake_fit(estimator, x, y, w):
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)

    rng = np.random.default_rng(999)
    diagnostics: dict = {}

    with patch("simulation.service._fit_estimator", side_effect=fake_fit):
        run_batch(
            scenario=mock_scenario,
            method_id="naive_observed_as_negative__logistic_regression__traincost_symmetric",
            replications=[0],
            rng=rng,
            diagnostics=diagnostics,
        )

    model_warnings = diagnostics.get("warnings", {})
    assert "ConvergenceWarning" in model_warnings, (
        f"Expected ConvergenceWarning in diagnostics.warnings; got: {diagnostics}"
    )
    assert model_warnings["ConvergenceWarning"] >= 1
