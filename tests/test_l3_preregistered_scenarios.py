"""Regression tests for the P0-locked L3 scenario registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from core.l3_scenarios import locked_l3_scenario_registry
from core.registry_compiler import compile_registry
from core.semantic_keys import ELIGIBLE, FIRM_ID, FISCAL_YEAR, MATURE
from labels.latent_class import LatentClassResult
from selection import preregistered_l3

ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, object]:
    return cast(
        dict[str, object],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


def test_l3_scenarios_are_locked_in_the_p00_registry() -> None:
    registry = _registry()
    scenarios = locked_l3_scenario_registry(registry)

    assert scenarios.primary_scenario_id == "neutral_pi_03"
    assert [item.scenario_id for item in scenarios.scenarios] == [
        "high_pi_05",
        "low_pi_01",
        "neutral_pi_03",
    ]
    assert [item.scenario_id for item in scenarios.scenarios if item.role == "primary"] == [
        "neutral_pi_03"
    ]
    assert scenarios.execution_policy["run_all_registered_scenarios"] is True
    assert scenarios.execution_policy["performance_based_scenario_selection_forbidden"] is True
    assert scenarios.execution_policy["outer_outcomes_accessed"] is False
    assert scenarios.execution_policy["known_cases_accessed"] is False


def test_measurement_delegates_l3_parameters_to_scenario_registry() -> None:
    registry = _registry()
    measurement = cast(dict[str, Any], registry["measurement"])
    model = cast(dict[str, Any], measurement["l3_model"])
    operational = cast(dict[str, Any], model["operational"])

    assert model["scenario_registry_module"] == "l3_scenarios"
    assert model["scenario_execution"] == "run_all_registered_scenarios"
    assert model["scenario_selection_rule"] == "preregistered_primary_only"
    assert model["performance_based_scenario_selection_forbidden"] is True
    assert "fixed_pi_grid" not in operational
    assert "accuracy_priors_by_profile" not in operational


def test_workflow_has_no_post_preparation_lock_mode() -> None:
    workflow = (ROOT / "scripts" / "s3_l3_production_workflow.ps1").read_text(
        encoding="utf-8"
    )

    assert 'ValidateSet("Migrate", "Prepare", "Final")' in workflow
    assert '"Lock" {' not in workflow
    assert "LockFile" not in workflow
    assert "PreparationReceipt" not in workflow
    assert "lock_l3_parameters.py" not in workflow
    assert not (ROOT / "scripts" / "lock_l3_parameters.py").exists()
    assert not (ROOT / "templates" / "l3_parameter_lock.template.yaml").exists()


def test_p10_uses_preregistered_executor_not_fixed_pi_optimization() -> None:
    p10 = (ROOT / "scripts" / "p10_select_measurement.py").read_text(encoding="utf-8")
    executor = (ROOT / "src" / "selection" / "preregistered_l3.py").read_text(
        encoding="utf-8"
    )

    assert "fit_l3_preregistered_scenarios" in p10
    assert "fit_l3_fold_candidate(" not in p10
    assert '"scenario_selection_rule": "PRE_REGISTERED_PRIMARY"' in executor
    assert '"performance_based_scenario_selection": False' in executor
    assert "min(eligible_scenarios" not in executor


def test_worse_diagnostic_loss_cannot_replace_preregistered_primary(monkeypatch: Any) -> None:
    scenario_registry = locked_l3_scenario_registry(_registry())
    diagnostic_loss = {
        "low_pi_01": 0.10,
        "neutral_pi_03": 0.90,
        "high_pi_05": 0.20,
    }

    def fake_strict_channel_diagnostics(**kwargs: Any) -> list[dict[str, Any]]:
        scenario = kwargs["scenario"]
        return [
            {
                "scenario_id": scenario.scenario_id,
                "fixed_pi": scenario.fixed_pi,
                "heldout_channel": "S1",
                "rows": 1,
                "soft_cross_entropy": diagnostic_loss[scenario.scenario_id],
                "diagnostics_eligible": True,
                "reason_code": None,
                "heldout_removed_from_target_and_measurement": True,
            }
        ]

    def fake_fit_l3(**kwargs: Any) -> LatentClassResult:
        fixed_pi = float(kwargs["fixed_pi"])
        return LatentClassResult(
            posterior_mean=[fixed_pi],
            posterior_draws=[[1]],
            parameter_draws=[],
            source_accuracy={
                "S1_profit_adjustment": {
                    "sensitivity_mean": 0.8,
                    "specificity_mean": 0.9,
                }
            },
            channel_random_effect_sd={"S1": 0.0},
            diagnostics={"eligible_for_gate1": True},
        )

    monkeypatch.setattr(
        preregistered_l3,
        "_strict_channel_diagnostics",
        fake_strict_channel_diagnostics,
    )
    monkeypatch.setattr(preregistered_l3, "_fit_l3", fake_fit_l3)

    result = preregistered_l3.fit_l3_preregistered_scenarios(
        matrices={
            "rows": [
                {
                    FIRM_ID: "F1",
                    FISCAL_YEAR: 2019,
                    MATURE: True,
                    "observed_channel_count": 1,
                    "source_outcomes": {"S1_profit_adjustment": True},
                    "channel_outcomes": {"S1": True},
                }
            ]
        },
        outer_year=2020,
        source_channels={"S1_profit_adjustment": "S1"},
        source_profiles={"S1_profit_adjustment": "financial_statement_core_long"},
        scenario_registry=scenario_registry,
        mcmc={},
        minimum_observed_channels=1,
        robust_fraction=1.0,
        rng=np.random.default_rng(42),
    )

    assert result["status"] == "PASS"
    assert result["primary_scenario_id"] == "neutral_pi_03"
    assert result["primary_fixed_pi"] == 0.03
    assert result["selection_objective"] == 0.90
    assert result["performance_based_scenario_selection"] is False
    assert result["target_rows"][0]["l3_scenario_id"] == "neutral_pi_03"
    assert result["target_rows"][0]["target_value"] == 0.03
    assert all(item[ELIGIBLE] is True for item in result["scenario_results"])
