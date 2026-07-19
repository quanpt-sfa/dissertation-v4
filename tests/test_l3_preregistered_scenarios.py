"""Regression tests for the P0-locked L3 scenario registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from core.l3_scenarios import locked_l3_scenario_registry
from core.registry_compiler import compile_registry

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
