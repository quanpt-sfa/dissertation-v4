from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("core.metrics")
pytest.importorskip("labels.service")

from simulation.method_contract import (
    apply_execution_profile,
    build_cost_sensitive_contract,
    build_execution_profile_contract,
    build_full_method_registry,
    build_imbalance_treatment_contract,
)
from simulation.service import run_batch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "execution" / "simulation.yaml"


def _scenario():
    simulation = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["simulation"]
    simulation["active_profile"] = "imbalance_treatment_robustness"
    cost = build_cost_sensitive_contract(simulation)
    imbalance = build_imbalance_treatment_contract(simulation)
    full = build_full_method_registry(simulation, cost, imbalance)
    profile = build_execution_profile_contract(simulation, full)
    registry = apply_execution_profile(full, profile)
    return {
        "scenario_id": "imbalance_smoke",
        "sample_size": 500,
        "prevalence": 0.20,
        "anchor_sensitivity": 0.90,
        "anchor_false_positive": 0.02,
        "weak_sensitivity": 0.75,
        "weak_false_positive": 0.05,
        "content_signal": 1.20,
        "tier": "fully_synthetic",
        "fixed_pi": 0.20,
        "anchor_verification_probability": 0.90,
        "weak_verification_probability": 0.85,
        "selective_verification_strength": 0.05,
        "channel_dependence": 0.10,
        "horizon_days": 365,
        "detection_delay_mean_days": 30,
        "shift_strength": 0.10,
        "signal_structure": "linear",
        "method_registry": registry,
        "active_method_ids": profile["active_method_ids"],
        "execution_profile": profile["active_profile"],
        "cost_regime_registry": cost["cost_regime_registry"],
        "review_budget_registry": cost["review_budget_registry"],
        "cost_sensitive_settings": {
            key: value
            for key, value in cost.items()
            if key not in {"cost_regime_registry", "review_budget_registry"}
        },
        "imbalance_treatment_registry": imbalance["treatment_registry"],
        "imbalance_treatment_settings": {
            key: value
            for key, value in imbalance.items()
            if key != "treatment_registry"
        },
    }


@pytest.mark.parametrize(
    ("method_id", "optional_dependency"),
    [
        (
            "naive_observed_as_negative__logistic_regression__imbalance_smote__traincost_symmetric",
            None,
        ),
        (
            "verified_only__xgboost__imbalance_adasyn__traincost_symmetric",
            "xgboost",
        ),
    ],
)
def test_synthetic_imbalance_treatment_is_applied(
    method_id: str,
    optional_dependency: str | None,
) -> None:
    if optional_dependency and importlib.util.find_spec(optional_dependency) is None:
        pytest.skip(
            f"optional learner dependency {optional_dependency!r} is not installed"
        )

    batch = run_batch(
        _scenario(),
        method_id=method_id,
        replications=range(1),
        data_rng=np.random.default_rng(123),
        model_rng=np.random.default_rng(456),
    )
    values = batch.set_index("metric_id")["estimate"]
    assert values["fit_success"] == 1.0
    assert values["imbalance_treatment_applied"] == 1.0
    assert values["synthetic_rows_added"] > 0
    assert values["post_resampling_positive_rate"] >= values["training_positive_rate"]
