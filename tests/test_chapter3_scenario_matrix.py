from __future__ import annotations

from pathlib import Path

import yaml

from simulation.method_contract import (
    apply_execution_profile,
    build_cost_sensitive_contract,
    build_execution_profile_contract,
    build_full_method_registry,
    build_imbalance_treatment_contract,
)
from simulation.service import validate_scenarios


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "execution" / "simulation.yaml"


def _simulation() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["simulation"]


def test_chapter3_scenario_matrix_is_locked_and_complete_for_supported_dgp() -> None:
    simulation = _simulation()
    scenarios = simulation["operational_scenarios"]
    contract = simulation["chapter3_scenario_contract"]

    assert simulation["active_profile"] == "chapter3_measurement"
    assert len(scenarios) == 17
    assert sum(item["tier"] == "semi_synthetic_development_covariates" for item in scenarios) == 1
    assert sum(item["tier"] == "fully_synthetic" for item in scenarios) == 16

    scenario_ids = [item["scenario_id"] for item in scenarios]
    scenario_blocks = [item["scenario_block"] for item in scenarios]
    assert len(scenario_ids) == len(set(scenario_ids))
    assert len(scenario_blocks) == len(set(scenario_blocks))
    assert scenario_blocks == contract["required_scenario_blocks"]
    assert set(contract["baseline_scenario_ids"]) <= set(scenario_ids)

    # The current engine validates these dimensions but does not yet apply shift.
    # Keep operational scenarios honest until the DGP implementation is extended.
    assert all(float(item["shift_strength"]) == 0.0 for item in scenarios)

    deferred = {item["block"] for item in contract["deferred_engine_blocks"]}
    assert deferred == {
        "S3_complete_next_calendar_year_maturity",
        "covariate_concept_and_label_policy_shift",
        "gate_1_to_3_operating_characteristics",
        "L3_correct_vs_misspecified_variants",
    }


def test_fully_synthetic_scenarios_pass_runtime_validation() -> None:
    simulation = _simulation()
    fully_synthetic = [
        dict(item)
        for item in simulation["operational_scenarios"]
        if item["tier"] == "fully_synthetic"
    ]
    validated = validate_scenarios(fully_synthetic)
    assert len(validated) == 16


def test_chapter3_measurement_profile_holds_learner_fixed() -> None:
    simulation = _simulation()
    cost = build_cost_sensitive_contract(simulation)
    imbalance = build_imbalance_treatment_contract(simulation)
    full = build_full_method_registry(simulation, cost, imbalance)
    profile = build_execution_profile_contract(simulation, full)
    registry = apply_execution_profile(full, profile)

    active = [item for item in registry if item["is_active"] is True]
    predictive = [item for item in active if item["method_family"] == "predictive"]
    standalone = [item for item in active if item["method_family"] == "standalone_estimator"]

    assert profile["active_profile"] == "chapter3_measurement"
    assert len(active) == 6
    assert len(predictive) == 4
    assert len(standalone) == 2
    assert {item["learner_id"] for item in predictive} == {"logistic_regression"}
    assert {item["label_strategy_id"] for item in predictive} == {
        "naive_observed_as_negative",
        "verified_only",
        "ipw",
        "pu",
    }
