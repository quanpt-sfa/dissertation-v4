from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core.registry_compiler import compile_registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


def get(registry: dict[str, Any], path: str) -> Any:
    value: Any = registry
    for part in path.split("."):
        value = value[part]
    return value


def canonical(registry: dict[str, Any], decision_id: str, title: str) -> None:
    row = registry["appendix_b"][decision_id]
    assert row["canonical_title"] == title
    assert row["chapter_reference"] == f"Appendix B, Table B.1, {decision_id}"


def test_T001_prediction_date(registry: dict[str, Any]) -> None:
    canonical(registry, "D01", "Ngày dự báo τit")
    assert get(registry, "study.prediction_time.availability_date_audit_required") is True


def test_T002_horizon(registry: dict[str, Any]) -> None:
    canonical(registry, "D02", "Chân trời chính")
    assert get(registry, "study.horizons_months.primary") == 12
    assert get(registry, "study.horizons_months.sensitivity") == [24]


def test_T003_source_set(registry: dict[str, Any]) -> None:
    canonical(registry, "D03", "Tập nguồn J")
    assert get(registry, "data_sources.leave_one_channel_out_required") is True
    assert get(registry, "data_sources.source_set_changes_create_new_estimand") is True


def test_T004_anchor_source(registry: dict[str, Any]) -> None:
    canonical(registry, "D04", "Nguồn neo")
    assert get(registry, "data_sources.anchor.false_positive_rate_grid") == [0.0, 0.01, 0.03, 0.05]


def test_T005_primary_outcome(registry: dict[str, Any]) -> None:
    canonical(registry, "D05", "Outcome chính")
    assert get(registry, "measurement.track_a_primary_endpoint") == "L1"
    assert get(registry, "measurement.roles.L3") == "measurement_sensitivity"


def test_T006_prior_accuracy(registry: dict[str, Any]) -> None:
    canonical(registry, "D06", "Miền prior/accuracy")
    assert get(registry, "measurement.prior_accuracy_domain.primary_prior") == "fixed_pi_grid"
    assert (
        get(registry, "measurement.prior_accuracy_domain.hierarchical_prior_role")
        == "sensitivity_only"
    )


def test_T007_review_budget(registry: dict[str, Any]) -> None:
    canonical(registry, "D07", "Ngân sách rà soát")
    assert get(registry, "evaluation.review_budget.primary_fraction") == 0.05
    assert get(registry, "evaluation.review_budget.sensitivity_fractions") == [0.01, 0.10]


def test_T008_learner_roster(registry: dict[str, Any]) -> None:
    canonical(registry, "D08", "Learner roster")
    assert get(registry, "learners.confirmatory") == [
        "elastic_net_logistic",
        "random_forest",
        "main_boosting",
    ]
    assert get(registry, "learners.pu_branch") == ["anchor_pu"]


def test_T009_tuning_budget(registry: dict[str, Any]) -> None:
    canonical(registry, "D09", "Tuning budget")
    assert get(registry, "learners.tuning.max_valid_configurations_per_learner_inner_fold") == 50


def test_T010_outer_folds(registry: dict[str, Any]) -> None:
    canonical(registry, "D10", "Outer folds")
    assert get(registry, "folds.initial_outer_year") == 2020
    assert get(registry, "folds.fully_nested_outer_years") == [2021, 2022, 2023, 2024]
    assert get(registry, "folds.prospective_year") == 2026


def test_T011_gate2(registry: dict[str, Any]) -> None:
    canonical(registry, "D11", "Cổng 2")
    assert get(registry, "evaluation.gate2.same_direction_min_folds") == 3
    assert get(registry, "evaluation.gate2.confirmatory_positive_count_min") == 25


def test_T012_gate3(registry: dict[str, Any]) -> None:
    canonical(registry, "D12", "Cổng 3")
    assert get(registry, "evaluation.gate3.breakpoint_tolerance_training_sd") == 0.10
    assert get(registry, "evaluation.gate3.minimum_domains") == 2


def test_T013_transfer_domains(registry: dict[str, Any]) -> None:
    canonical(registry, "D13", "Miền chuyển giao")
    assert get(registry, "domains.primary.listing_boards") == ["HOSE", "HNX", "UPCoM"]


def test_T014_measurement_roles(registry: dict[str, Any]) -> None:
    canonical(registry, "D14", "Vai trò L0–L3")
    assert get(registry, "measurement.roles") == {
        "L0": "single_source_benchmark",
        "L1": "primary",
        "L2": "alternative_aggregation",
        "L3": "measurement_sensitivity",
    }


def test_T015_measurement_learner(registry: dict[str, Any]) -> None:
    canonical(registry, "D15", "Measurement × learner")
    assert get(registry, "measurement.selection.development_only") is True


def test_T016_soft_objectives(registry: dict[str, Any]) -> None:
    canonical(registry, "D16", "Objective L2/L3")
    assert get(registry, "measurement.objectives.L2") == ["soft_cross_entropy"]
    assert "posterior_draws" in get(registry, "measurement.objectives.L3")


def test_T017_l3_model(registry: dict[str, Any]) -> None:
    canonical(registry, "D17", "Mô hình L3")
    assert get(registry, "measurement.l3_model.primary") == "fixed_pi_latent_class"
    assert get(registry, "measurement.l3_model.hierarchical_pi.role") == "sensitivity_only"


def test_T018_selective_verification(registry: dict[str, Any]) -> None:
    canonical(registry, "D18", "Xác minh chọn lọc")
    assert get(registry, "weighting.verification_diagnostics_always") is True
    assert get(registry, "weighting.trimming_changes_estimand") is True


def test_T019_known_case_firewall(registry: dict[str, Any]) -> None:
    canonical(registry, "D19", "Firewall K1–K4")
    assert get(registry, "known_cases.sealed_before_development") is True
    assert "tuning" in get(registry, "known_cases.forbidden_uses_before_opening")


def test_T020_label_model_firewall(registry: dict[str, Any]) -> None:
    canonical(registry, "D20", "Biến trong label model")
    assert get(registry, "features.label_model_allows_content") is False
    assert get(registry, "evidence.label_model_allowed_blocks") == ["S", "M", "T", "Q", "Z_M"]


def test_T021_maturity(registry: dict[str, Any]) -> None:
    canonical(registry, "D21", "Maturity")
    assert get(registry, "risksets.main_requires_tau_plus_h_le_tcut") is True
    assert get(registry, "risksets.complete_followup_required") is True


def test_T022_source_horizon(registry: dict[str, Any]) -> None:
    canonical(registry, "D22", "Horizon theo nguồn")
    assert get(registry, "study.source_maturity_curves_required") is True


def test_T023_gate_thresholds(registry: dict[str, Any]) -> None:
    canonical(registry, "D23", "Tiêu chí cổng")
    assert get(registry, "evaluation.gate_common.noninferiority_relative_ap_margin") == 0.05
    assert get(registry, "evaluation.gate_common.familywise_alpha") == 0.05


def test_T024_interaction_library(registry: dict[str, Any]) -> None:
    canonical(registry, "D24", "Interaction library")
    assert get(registry, "inference.interaction_library.confirmatory_threshold_claims") == 2
    assert (
        get(registry, "inference.interaction_library.confirmatory_block") == "pressure_x_monitoring"
    )


def test_T025_exit_events(registry: dict[str, Any]) -> None:
    canonical(registry, "D25", "Exit/competing events")
    assert get(registry, "evidence.exit_events.delisting_or_merger_is_negative") is False


def test_T026_track_propagation(registry: dict[str, Any]) -> None:
    canonical(registry, "D26", "Truyền CHNC1")
    assert get(registry, "measurement.selection.no_candidate_value") == "none"


def test_T027_source_holdout(registry: dict[str, Any]) -> None:
    canonical(registry, "D27", "Giữ lại nguồn")
    assert get(registry, "measurement.strict_selection.design") == "nested_channel_within_time"
    assert "target" in get(registry, "measurement.strict_selection.remove_heldout_channel_from")


def test_T028_pu_positive_set(registry: dict[str, Any]) -> None:
    canonical(registry, "D28", "Tập dương PU")
    assert get(registry, "learners.pu_branch") == ["anchor_pu"]
    assert get(registry, "learners.sensitivity") == ["noisy_positive_pu"]


def test_T029_uncertainty_unit(registry: dict[str, Any]) -> None:
    canonical(registry, "D29", "Đơn vị bất định")
    assert get(registry, "inference.bootstrap.primary_unit") == "firm"
    assert get(registry, "inference.bootstrap.stratify_by") == "outer_year"


def test_T030_known_case_veto(registry: dict[str, Any]) -> None:
    canonical(registry, "D30", "Soft veto known case")
    assert get(registry, "known_cases.soft_veto.minimum_cases") == 4
    assert get(registry, "known_cases.soft_veto.scenario_fraction_min") == 0.75


def test_T031_track_b_metrics(registry: dict[str, Any]) -> None:
    canonical(registry, "D31", "Metric Tuyến B")
    assert get(registry, "evaluation.track_b_metrics.ap_directly_on_soft_targets") is False


def test_T032_measurement_selection(registry: dict[str, Any]) -> None:
    canonical(registry, "D32", "Chọn measurement specification")
    assert get(registry, "measurement.selection.time_only") == "M_f_star"
    assert get(registry, "measurement.selection.strict_channel") == "M_f_c_star"


def test_T033_measurement_stability(registry: dict[str, Any]) -> None:
    canonical(registry, "D33", "Ổn định đo lường")
    assert get(registry, "measurement.selection.stability_minimum_selected_folds") == 3
    assert get(registry, "measurement.selection.stability_denominator_folds") == 4


def test_T034_l2_missingness(registry: dict[str, Any]) -> None:
    canonical(registry, "D34", "Missingness L2")
    assert get(registry, "measurement.l2_missingness.normalize_over_observed_channels") is True
    assert get(registry, "measurement.l2_missingness.minimum_coverage_required") is True


def test_T035_calibration(registry: dict[str, Any]) -> None:
    canonical(registry, "D35", "Hiệu chuẩn cuối")
    assert get(registry, "calibration.source") == "pooled_cross_fitted_development_predictions"
    assert get(registry, "calibration.in_sample_predictions_after_refit_forbidden") is True


def test_T036_multiple_testing(registry: dict[str, Any]) -> None:
    canonical(registry, "D36", "Nhiều phép thử")
    assert get(registry, "inference.families") == ["CHNC2", "CHNC3"]
    assert "max_T" in get(registry, "inference.family_adjustment_methods")


def test_T037_censoring(registry: dict[str, Any]) -> None:
    canonical(registry, "D37", "Censoring")
    assert get(registry, "risksets.ipcw_role") == "sensitivity_only_when_diagnostics_pass"


def test_T038_simulation_objective(registry: dict[str, Any]) -> None:
    canonical(registry, "D38", "Mục tiêu mô phỏng")
    assert get(registry, "simulation.objective.ground_truth_known") is True
    assert get(registry, "simulation.objective.estimate_vietnam_prevalence") is False


def test_T039_simulation_tiers(registry: dict[str, Any]) -> None:
    canonical(registry, "D39", "Hai tầng mô phỏng")
    assert get(registry, "simulation.tiers") == [
        "fully_synthetic",
        "semi_synthetic_development_covariates",
    ]
    assert get(registry, "simulation.outer_outcomes_allowed") is False


def test_T040_simulation_dgp(registry: dict[str, Any]) -> None:
    canonical(registry, "D40", "Cơ chế sinh dữ liệu")
    assert "Y_star" in get(registry, "simulation.dgp")
    assert get(registry, "simulation.l3_correct_and_misspecified_checked") is True


def test_T041_scenario_space(registry: dict[str, Any]) -> None:
    canonical(registry, "D41", "Không gian kịch bản")
    assert get(registry, "simulation.full_cartesian_product") is False
    assert "prevalence" in get(registry, "simulation.scenario_space")


def test_T042_simulation_methods(registry: dict[str, Any]) -> None:
    canonical(registry, "D42", "Phương pháp mô phỏng")
    assert get(registry, "simulation.oracle_role") == "upper_benchmark_only"
    assert get(registry, "simulation.methods") == [
        "observability_only",
        "content_only",
        "full",
        "anchor_pu",
    ]


def test_T043_simulation_metrics(registry: dict[str, Any]) -> None:
    canonical(registry, "D43", "Chỉ số mô phỏng")
    assert "ranking_recovery" in get(registry, "simulation.metrics.fixed_pi")
    assert "pi_coverage" in get(registry, "simulation.metrics.hierarchical_pi")
    assert "actual_mcse" in get(registry, "simulation.metrics.other")


def test_T044_replications(registry: dict[str, Any]) -> None:
    canonical(registry, "D44", "Số lần lặp")
    assert get(registry, "simulation.core.minimum_replications") == 2500
    assert get(registry, "simulation.l3.initial_replications") == 1000
    assert (
        get(
            registry,
            "simulation.continuous_metrics.mcse_fraction_of_minimum_meaningful_improvement_max",
        )
        == 0.10
    )


def test_T045_protocol_role(registry: dict[str, Any]) -> None:
    canonical(registry, "D45", "Vai trò protocol")
    assert get(registry, "simulation.protocol_role.must_complete_before_outer_test") is True
    assert (
        get(registry, "simulation.protocol_role.gate_optimization_on_outer_performance_forbidden")
        is True
    )
