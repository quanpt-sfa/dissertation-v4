"""Regression tests for the locked P08 MCSE metric-policy registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core.registry_compiler import compile_registry
from simulation.l3_variants import extend_method_registry
from simulation.mcse_contract import (
    MetricPolicy,
    compile_metric_policies,
    resolve_metric_policies,
    resolve_metric_policy,
)
from simulation.method_contract import (
    build_cost_sensitive_contract,
    build_full_method_registry,
    build_imbalance_treatment_contract,
    required_metric_ids,
)

ROOT = Path(__file__).resolve().parents[2]


def _production_registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


def test_production_policy_registry_covers_every_registered_p08_metric() -> None:
    registry = _production_registry()
    simulation = cast(dict[str, object], registry["simulation"])
    mcse_registry = cast(dict[str, object], registry["simulation_mcse"])
    cost_contract = build_cost_sensitive_contract(simulation)
    imbalance_contract = build_imbalance_treatment_contract(simulation)
    methods = build_full_method_registry(
        simulation,
        cost_contract=cost_contract,
        imbalance_contract=imbalance_contract,
    )
    methods = extend_method_registry(simulation, methods)
    metric_ids: set[str] = set()
    for method in methods:
        metric_ids.update(required_metric_ids(method))

    policies = compile_metric_policies(mcse_registry, simulation)
    coverage = resolve_metric_policies(metric_ids, policies)

    assert coverage["status"] == "PASS"
    assert coverage["metric_count"] == len(metric_ids)
    assert coverage["gated_metric_count"] >= 1
    assert coverage["role_counts"]["confirmatory"] >= 1


def test_primary_cost_is_diagnostic_in_measurement_focused_p08() -> None:
    registry = _production_registry()
    simulation = cast(dict[str, object], registry["simulation"])
    mcse_registry = cast(dict[str, object], registry["simulation_mcse"])

    policies = compile_metric_policies(mcse_registry, simulation)
    policy = resolve_metric_policy("threshold_cost::high_fn::cost_per_firm", policies)

    assert policy["policy_id"] == "diagnostic_primary_threshold_cost"
    assert policy["role"] == "diagnostic"
    assert policy["mcse_gate_required"] is False


def test_fixed_pi_measurement_metrics_are_confirmatory() -> None:
    registry = _production_registry()
    simulation = cast(dict[str, object], registry["simulation"])
    mcse_registry = cast(dict[str, object], registry["simulation_mcse"])

    policies = compile_metric_policies(mcse_registry, simulation)
    for metric_id in (
        "fixed_pi_fit_success",
        "row_brier",
        "row_log_loss",
        "row_calibration_bias",
        "paired_brier_regret",
        "review_sample_size_ratio_vs_l1",
    ):
        policy = resolve_metric_policy(metric_id, policies)
        assert policy["role"] == "confirmatory"
        assert policy["mcse_gate_required"] is True


def test_hierarchical_pi_recovery_is_nonblocking_sensitivity() -> None:
    registry = _production_registry()
    simulation = cast(dict[str, object], registry["simulation"])
    mcse_registry = cast(dict[str, object], registry["simulation_mcse"])

    policies = compile_metric_policies(mcse_registry, simulation)
    policy = resolve_metric_policy("hierarchical_pi_coverage", policies)

    assert policy["policy_id"] == "diagnostic_hierarchical_pi"
    assert policy["role"] == "diagnostic"
    assert policy["mcse_gate_required"] is False


def test_policy_resolution_rejects_unknown_metric() -> None:
    policy: MetricPolicy = {
        "policy_id": "known",
        "priority": 1,
        "metric_pattern": "known_metric",
        "resolved_pattern": "known_metric",
        "role": "diagnostic",
        "mcse_gate_required": False,
        "undefined_policy": "allow_nonblocking",
        "target_rule": "none",
        "minimum_finite_fraction": 0.0,
    }

    with pytest.raises(ValueError, match="policy missing"):
        resolve_metric_policy("new_unregistered_metric", [policy])


def test_policy_resolution_rejects_equal_rank_ambiguity() -> None:
    base: MetricPolicy = {
        "policy_id": "first",
        "priority": 10,
        "metric_pattern": "metric_*",
        "resolved_pattern": "metric_*",
        "role": "diagnostic",
        "mcse_gate_required": False,
        "undefined_policy": "allow_nonblocking",
        "target_rule": "none",
        "minimum_finite_fraction": 0.0,
    }
    second = dict(base)
    second["policy_id"] = "second"

    with pytest.raises(ValueError, match="policy ambiguous"):
        resolve_metric_policy("metric_value", [base, cast(MetricPolicy, second)])
