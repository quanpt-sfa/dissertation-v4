from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "simulation" / "method_contract.py"
CONFIG_PATH = ROOT / "config" / "execution" / "simulation.yaml"


def _load_module():
    spec = importlib.util.spec_from_file_location("p08_method_contract", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _simulation():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["simulation"]


def _registry_for(profile: str):
    module = _load_module()
    simulation = _simulation()
    simulation["active_profile"] = profile
    cost = module.build_cost_sensitive_contract(simulation)
    imbalance = module.build_imbalance_treatment_contract(simulation)
    full = module.build_full_method_registry(simulation, cost, imbalance)
    profile_contract = module.build_execution_profile_contract(simulation, full)
    registry = module.apply_execution_profile(full, profile_contract)
    return module, simulation, cost, imbalance, full, profile_contract, registry


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("core", 22),
        ("cost_sensitive_robustness", 18),
        ("methodological_robustness", 6),
        ("algorithmic_robustness", 14),
        ("imbalance_treatment_robustness", 12),
        ("prespecified_all", 72),
        ("full", 114),
    ],
)
def test_profile_counts(profile: str, expected: int) -> None:
    module, _, cost, imbalance, full, contract, registry = _registry_for(profile)
    assert len(full) == 114
    assert module.expected_counts(cost, imbalance)["method_total"] == 114
    assert contract["active_method_count"] == expected
    assert module.profile_expected_counts(registry)["method_total"] == expected
    assert sum(item[module.IS_ACTIVE] is True for item in registry) == expected


def test_core_matches_vietnamese_learner_roster() -> None:
    module, _, _, _, full, contract, registry = _registry_for("core")
    assert len(full) == 114
    active = [item for item in registry if item[module.IS_ACTIVE]]
    predictive = [item for item in active if item[module.METHOD_FAMILY] == "predictive"]
    assert len(predictive) == 20
    assert len(active) == 22
    assert {
        item[module.LEARNER_ID] for item in predictive
    } == {
        "logistic_regression",
        "linear_svm",
        "random_forest",
        "xgboost",
        "multilayer_perceptron",
    }
    assert {item[module.LABEL_STRATEGY_ID] for item in predictive} == set(
        module.LABEL_STRATEGIES
    )
    assert {item[module.TRAINING_COST_REGIME_ID] for item in predictive} == {
        "symmetric"
    }
    assert {item[module.IMBALANCE_TREATMENT_ID] for item in predictive} == {"none"}
    assert contract["active_profile"] == "core"


def test_cost_sensitive_robustness_is_restricted_to_operational_anchor_learners() -> None:
    module, *_rest, registry = _registry_for("cost_sensitive_robustness")
    active = [item for item in registry if item[module.IS_ACTIVE]]
    assert {item[module.LEARNER_ID] for item in active} == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert {item[module.LABEL_STRATEGY_ID] for item in active} == {
        "naive_observed_as_negative",
        "ipw",
        "pu",
    }
    assert {item[module.TRAINING_COST_REGIME_ID] for item in active} == {
        "moderate_fn",
        "high_fn",
    }
    assert {item[module.IMBALANCE_TREATMENT_ID] for item in active} == {"none"}


def test_smote_adasyn_are_separate_from_ipw_and_pu() -> None:
    module, *_rest, registry = _registry_for("imbalance_treatment_robustness")
    active = [item for item in registry if item[module.IS_ACTIVE]]
    assert len(active) == 12
    assert {item[module.LEARNER_ID] for item in active} == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert {item[module.LABEL_STRATEGY_ID] for item in active} == {
        "naive_observed_as_negative",
        "verified_only",
    }
    assert {item[module.IMBALANCE_TREATMENT_ID] for item in active} == {
        "smote",
        "adasyn",
    }
    assert {item[module.TRAINING_COST_REGIME_ID] for item in active} == {
        "symmetric"
    }


def test_inactive_method_is_not_executable() -> None:
    module, _, _, _, _, _, registry = _registry_for("core")
    scenario = {
        "method_registry": registry,
        "active_method_ids": [
            item["method_id"] for item in registry if item[module.IS_ACTIVE]
        ],
    }
    inactive = next(item for item in registry if not item[module.IS_ACTIVE])
    with pytest.raises(ValueError, match="inactive"):
        module.method_by_id(scenario, inactive["method_id"])
    assert module.method_by_id(
        scenario, inactive["method_id"], require_active=False
    )["method_id"] == inactive["method_id"]


def test_all_methods_have_profile_membership_role_and_treatment() -> None:
    module, _, _, _, _, _, registry = _registry_for("core")
    for item in registry:
        assert item[module.AVAILABLE_PROFILES]
        assert item[module.ANALYSIS_ROLE]
        assert item[module.EXECUTION_PROFILE] == "core"
        assert item[module.PROTOCOL_STATUS]
        assert item[module.IMBALANCE_TREATMENT_ID]
