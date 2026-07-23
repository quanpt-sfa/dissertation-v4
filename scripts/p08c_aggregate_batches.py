"""P08C collector: validate active artifacts and compute policy-aware MCSE."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, sequence
from core.semantic_keys import (
    METHOD_ID,
    METHOD_KEY,
    SCENARIO_ID,
    SCENARIO_KEY,
)
from simulation.mcse_contract import compile_metric_policies, resolve_metric_policies
from simulation.mcse_service import (
    sanitize_normalized_cost_regret,
    summarize_mcse,
    validate_worker_batch,
)
from simulation.method_contract import ANALYSIS_ROLE, method_by_id
from simulation.p08_aggregation import (
    filter_active_batches,
    required_active_metric_ids,
    validate_methodology,
)
from simulation.replication_contract import replication_plan

CoordinateKey = tuple[tuple[str, str], ...]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    registry_path = cast(Path, args.registry)
    run_id = cast(str, args.run_id)
    check_only = cast(bool, args.check_only)

    loaded = load_run(
        registry_path=registry_path,
        run_id=run_id,
        step_id="P08",
        state="FEATURED",
    )
    scenarios = [
        mapping(item, "simulation scenario")
        for item in sequence(
            loaded.context.read("simulation_scenario_registry", {}),
            "simulation scenario registry",
        )
    ]
    scenario_by_id = {str(item[SCENARIO_ID]): item for item in scenarios}
    if len(scenario_by_id) != len(scenarios):
        raise ValueError("P08C scenario registry contains duplicate scenario_id")
    profiles = {str(item.get("execution_profile", "")) for item in scenarios}
    if len(profiles) != 1 or "" in profiles:
        raise ValueError("P08C requires one config-locked execution profile per run")
    active_profile = next(iter(profiles))

    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    batches: list[pd.DataFrame] = []
    coordinates_seen: list[dict[str, str]] = []
    range_audits: list[dict[str, object]] = []
    inventory = [
        cast(dict[str, object], item)
        for item in loaded.context.store.inventory()
        if isinstance(item, dict)
    ]
    for item in inventory:
        if item.get("artifact_id") != "simulation_batches":
            continue
        raw_coordinates = item.get("coordinates")
        if not isinstance(raw_coordinates, dict):
            raise ValueError("simulation batch manifest coordinates required")
        coordinates = {
            str(key): str(value)
            for key, value in cast(dict[object, object], raw_coordinates).items()
        }
        value = loaded.context.read("simulation_batches", coordinates)
        if not isinstance(value, pd.DataFrame):
            raise ValueError("simulation_batches artifact must be a DataFrame")
        if value.empty:
            continue

        records = cast(list[dict[str, object]], value.to_dict(orient="records"))
        scenario_ids = {str(row.get(SCENARIO_ID, "")) for row in records}
        method_ids = {str(row.get(METHOD_ID, "")) for row in records}
        if len(scenario_ids) != 1 or len(method_ids) != 1:
            raise ValueError("Each simulation batch must contain exactly one scenario and method")
        scenario_id = next(iter(scenario_ids))
        method_id = next(iter(method_ids))
        scenario = scenario_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"batch scenario_id={scenario_id} is not registered")
        method_spec = method_by_id(scenario, method_id, require_active=False)
        if coordinates.get(SCENARIO_KEY) != str(scenario[SCENARIO_KEY]):
            raise ValueError("simulation batch scenario_key does not match scenario registry")
        if coordinates.get(METHOD_KEY) != str(method_spec[METHOD_KEY]):
            raise ValueError("simulation batch method_key does not match method registry")

        plan = replication_plan(simulation, method_spec)
        audit = validate_worker_batch(
            value,
            coordinates=coordinates,
            locked_worker_batch_size=int(plan["batch_size"]),
        )
        clean = sanitize_normalized_cost_regret(value)
        batches.append(clean)
        coordinates_seen.append(coordinates)
        range_audits.append(dict(audit))

    batch_coordinates = {_coordinate_key(value) for value in coordinates_seen}
    diagnostics_coordinates = {
        _coordinate_key(_coordinates(item))
        for item in inventory
        if item.get("artifact_id") == "model_diagnostics"
    }
    missing_diagnostics = batch_coordinates - diagnostics_coordinates
    orphan_diagnostics = diagnostics_coordinates - batch_coordinates
    if missing_diagnostics or orphan_diagnostics:
        raise ValueError(
            "P08 batch-diagnostics coordinate mismatch; re-run affected immutable batches. "
            f"missing_diagnostics={_display_coordinates(missing_diagnostics)}, "
            f"orphan_diagnostics={_display_coordinates(orphan_diagnostics)}"
        )

    methodology = validate_methodology(
        batches=batches,
        scenario_by_id=scenario_by_id,
    )
    active_batches = filter_active_batches(batches, scenario_by_id)
    if methodology.get("status") != "PASS":
        return _finish(
            loaded=loaded,
            check_only=check_only,
            report={
                "status": "FAIL",
                "mcse_status": "NOT_EVALUATED",
                "precision_target_met": False,
                "reason_code": "P08_ACTIVE_PROFILE_OR_COST_METRIC_CONTRACT_INCOMPLETE",
                "execution_profile": active_profile,
                "methodology_validation": methodology,
                "batch_artifact_count": len(batches),
                "active_batch_count": len(active_batches),
                "batch_coordinates": coordinates_seen,
                "batch_range_audit": range_audits,
                "metrics": [],
            },
            failure_code=2,
        )

    try:
        mcse_registry = mapping(loaded.registry.get("simulation_mcse"), "simulation_mcse")
        compiled_policies = compile_metric_policies(mcse_registry, simulation)
        policy_coverage = resolve_metric_policies(
            required_active_metric_ids(scenario_by_id),
            compiled_policies,
        )
    except ValueError as exc:
        return _finish(
            loaded=loaded,
            check_only=check_only,
            report={
                "status": "FAIL",
                "mcse_status": "NOT_EVALUATED",
                "precision_target_met": False,
                "reason_code": "P08_MCSE_POLICY_CONTRACT_INCOMPLETE",
                "policy_error": str(exc),
                "execution_profile": active_profile,
                "methodology_validation": methodology,
                "batch_artifact_count": len(batches),
                "active_batch_count": len(active_batches),
                "batch_coordinates": coordinates_seen,
                "batch_range_audit": range_audits,
                "metrics": [],
            },
            failure_code=2,
        )

    core = mapping(simulation.get("core"), "simulation.core")
    l3 = mapping(simulation.get("l3"), "simulation.l3")
    extended = mapping(simulation.get("extended_replication"), "simulation.extended_replication")
    cost_replication = mapping(
        simulation.get("cost_sensitive_replication"),
        "simulation.cost_sensitive_replication",
    )
    cost_settings = mapping(simulation.get("cost_sensitive"), "simulation.cost_sensitive")
    continuous = mapping(simulation.get("continuous_metrics"), "simulation.continuous_metrics")
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")
    improvement = mapping(
        gate2.get("minimum_meaningful_improvement"),
        "evaluation.gate2.minimum_meaningful_improvement",
    )
    report = summarize_mcse(
        active_batches,
        metric_policies=policy_coverage["policies"],
        minimum_replications=int(core["minimum_replications"]),
        maximum_replications=int(core["maximum_replications"]),
        pass_fail_mcse_maximum=float(core["pass_fail_mcse_max"]),
        l3_minimum_replications=int(l3["initial_replications"]),
        l3_maximum_replications=int(l3["maximum_replications"]),
        l3_pass_fail_mcse_maximum=float(l3["pass_fail_mcse_max"]),
        extended_minimum_replications=int(extended["minimum_replications"]),
        extended_maximum_replications=int(extended["maximum_replications"]),
        extended_pass_fail_mcse_maximum=float(extended["pass_fail_mcse_max"]),
        cost_sensitive_minimum_replications=int(cost_replication["minimum_replications"]),
        cost_sensitive_maximum_replications=int(cost_replication["maximum_replications"]),
        cost_sensitive_pass_fail_mcse_maximum=float(cost_replication["pass_fail_mcse_max"]),
        cost_mcse_relative_fraction=float(cost_settings["cost_mcse_relative_fraction"]),
        continuous_mcse_fraction=float(
            continuous["mcse_fraction_of_minimum_meaningful_improvement_max"]
        ),
        minimum_meaningful_improvement=float(improvement["absolute"]),
    )
    mcse_status = str(report.get("status", "FAIL"))
    report.update(
        {
            "mcse_status": mcse_status,
            "execution_profile": active_profile,
            "methodology_validation": methodology,
            "metric_policy_coverage": {
                key: value for key, value in policy_coverage.items() if key != "policies"
            },
            "batch_artifact_count": len(batches),
            "active_batch_count": len(active_batches),
            "batch_coordinates": coordinates_seen,
            "batch_range_audit": range_audits,
        }
    )
    if check_only:
        print(f"P08_CONTROL_JSON={json.dumps(report, sort_keys=True)}")
        return 0

    protocol_role = mapping(simulation.get("protocol_role"), "simulation.protocol_role")
    must_complete = bool(protocol_role.get("must_complete_before_outer_test"))
    confirmatory = any(
        str(method.get(ANALYSIS_ROLE)) == "confirmatory"
        for scenario in scenario_by_id.values()
        for method in cast(list[dict[str, object]], scenario.get("method_registry", []))
        if isinstance(method, dict) and method.get("is_active") is True
    )
    if must_complete and confirmatory and report.get("precision_target_met") is not True:
        report["status"] = "FAIL"
        report["reason_code"] = (
            "CONFIRMATORY_MCSE_INCOMPLETE"
            if mcse_status == "CONTINUE"
            else "CONFIRMATORY_MCSE_TARGET_NOT_MET"
        )
        loaded.context.write("mcse_report", report, {})
        return 3

    loaded.context.write("mcse_report", report, {})
    print(
        f"P08C aggregate status={report['status']} profile={active_profile} "
        f"batches={len(active_batches)} methods_complete={methodology['status'] == 'PASS'}"
    )
    return 0


def _coordinates(item: dict[str, object]) -> dict[str, str]:
    raw = item.get("coordinates")
    if not isinstance(raw, dict):
        raise ValueError("artifact manifest coordinates required")
    return {str(key): str(value) for key, value in cast(dict[object, object], raw).items()}


def _coordinate_key(coordinates: dict[str, str]) -> CoordinateKey:
    return tuple(sorted(coordinates.items()))


def _display_coordinates(values: set[CoordinateKey]) -> list[str]:
    return [str(dict(value)) for value in sorted(values)]


def _finish(
    *,
    loaded: object,
    check_only: bool,
    report: dict[str, object],
    failure_code: int,
) -> int:
    if check_only:
        print(f"P08_CONTROL_JSON={json.dumps(report, sort_keys=True)}")
    else:
        context = getattr(loaded, "context")
        context.write("mcse_report", report, {})
    return failure_code


if __name__ == "__main__":
    raise SystemExit(main())
