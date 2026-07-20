"""P08C collector: validate the active profile, then compute MCSE."""

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
    BATCH_KEY,
    LEARNER_ID,
    METHOD_ID,
    METHOD_KEY,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
    SCENARIO_KEY,
)
from simulation.method_contract import (
    ANALYSIS_ROLE,
    EXECUTION_PROFILE,
    IMBALANCE_TREATMENT_ID,
    PROTOCOL_STATUS,
    active_method_ids,
    method_by_id,
    profile_expected_counts,
    required_metric_ids,
)
from simulation.replication_contract import replication_plan as _replication_plan
from simulation.service import summarize_mcse


def _coordinate_tuple(coordinates: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(coordinates.items()))


def _artifact_batch_size(
    *,
    plan: dict[str, int],
    diagnostics: dict[str, object],
) -> int:
    """Resolve artifact partition size from the worker's execution receipt.

    Legacy, non-compacted artifacts do not contain ``execution_batching`` and
    therefore retain the configured replication batch size. Compacted workers
    record both sizes, allowing P08C to validate coordinates without changing
    replication budgets or MCSE rules.
    """
    configured = int(plan["batch_size"])
    batching = diagnostics.get("execution_batching")
    if batching is None:
        return configured
    if not isinstance(batching, dict):
        raise ValueError("model diagnostics execution_batching must be an object")

    recorded_configured = int(str(batching.get("configured_batch_size")))
    artifact_size = int(str(batching.get("artifact_batch_size")))
    minimum = int(plan["minimum"])
    maximum = int(plan["maximum"])

    if recorded_configured != configured:
        raise ValueError(
            "model diagnostics configured batch size differs from the locked "
            f"replication plan: recorded={recorded_configured}, locked={configured}"
        )
    if artifact_size < configured or artifact_size % configured != 0:
        raise ValueError(
            "artifact batch size must be a positive integer multiple of the "
            "configured batch size"
        )
    if artifact_size > maximum:
        raise ValueError("artifact batch size exceeds maximum replications")
    if minimum % artifact_size != 0:
        raise ValueError(
            "minimum replications must end on an artifact batch boundary"
        )
    return artifact_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
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

    batches: list[pd.DataFrame] = []
    coordinates_seen: list[dict[str, str]] = []
    batch_partition_receipts: list[dict[str, object]] = []
    simulation = mapping(loaded.registry.get("simulation"), "simulation")

    # Call inventory exactly once. Inventory may decode and validate each
    # artifact, so repeated calls substantially increase every MCSE polling wave.
    inventory = list(loaded.context.store.inventory())
    diagnostics_coordinates: dict[
        tuple[tuple[str, str], ...], dict[str, str]
    ] = {}
    for item in inventory:
        if item.get("artifact_id") != "model_diagnostics":
            continue
        raw = item.get("coordinates")
        if not isinstance(raw, dict):
            raise ValueError("model diagnostics manifest coordinates required")
        coordinates = {
            str(key): str(value)
            for key, value in cast(dict[object, object], raw).items()
        }
        key = _coordinate_tuple(coordinates)
        if key in diagnostics_coordinates:
            raise ValueError(
                f"duplicate model diagnostics coordinates={coordinates}"
            )
        diagnostics_coordinates[key] = coordinates

    for item in inventory:
        if item.get("artifact_id") != "simulation_batches":
            continue
        raw = item.get("coordinates")
        if not isinstance(raw, dict):
            raise ValueError("simulation batch manifest coordinates required")
        coordinates = {
            str(key): str(value)
            for key, value in cast(dict[object, object], raw).items()
        }
        coordinate_key = _coordinate_tuple(coordinates)
        diagnostic_coordinates = diagnostics_coordinates.get(coordinate_key)
        if diagnostic_coordinates is None:
            raise ValueError(
                "P08 batch-diagnostics coordinate mismatch — batch exists but "
                f"diagnostics are missing at coordinates={coordinates}"
            )
        diagnostics = loaded.context.read(
            "model_diagnostics",
            diagnostic_coordinates,
        )
        if not isinstance(diagnostics, dict):
            raise ValueError("model_diagnostics artifact must be a JSON object")

        value = loaded.context.read("simulation_batches", coordinates)
        if not isinstance(value, pd.DataFrame):
            raise ValueError("simulation_batches artifact must be a DataFrame")
        if value.empty:
            continue

        scenario_ids = set(value[SCENARIO_ID].dropna().astype(str))
        method_ids = set(value[METHOD_ID].dropna().astype(str))
        if len(scenario_ids) != 1 or len(method_ids) != 1:
            raise ValueError(
                "Each simulation batch must contain exactly one scenario and method"
            )

        scenario_id = next(iter(scenario_ids))
        method_id = next(iter(method_ids))
        scenario = scenario_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"batch scenario_id={scenario_id} is not registered")

        method_spec = next(
            (
                method
                for method in scenario.get("method_registry", [])
                if method.get(METHOD_ID) == method_id
            ),
            None,
        )
        if method_spec is None:
            raise ValueError(
                f"batch method_id={method_id} is not registered under "
                f"scenario={scenario_id}"
            )

        expected_scenario_key = str(scenario[SCENARIO_KEY])
        expected_method_key = str(method_spec[METHOD_KEY])
        if coordinates.get(SCENARIO_KEY) != expected_scenario_key:
            raise ValueError(
                f"batch scenario_key={coordinates.get(SCENARIO_KEY)} does not "
                f"match scenario registry scenario_id={scenario_id}"
            )
        if coordinates.get(METHOD_KEY) != expected_method_key:
            raise ValueError(
                f"batch method_key={coordinates.get(METHOD_KEY)} does not match "
                f"method registry method_id={method_id}"
            )

        plan = _replication_plan(simulation, method_spec)
        artifact_batch_size = _artifact_batch_size(
            plan=plan,
            diagnostics=cast(dict[str, object], diagnostics),
        )
        replication_ids = value[REPLICATION_ID].dropna().astype(int)
        for replication_id in replication_ids:
            expected_batch_key = (
                f"b{replication_id // artifact_batch_size:04d}"
            )
            if coordinates.get(BATCH_KEY) != expected_batch_key:
                raise ValueError(
                    f"replication_id={replication_id} in batch maps to "
                    f"batch_key={expected_batch_key} using "
                    f"artifact_batch_size={artifact_batch_size}, but coordinate "
                    f"is batch_key={coordinates.get(BATCH_KEY)}"
                )

        batching = diagnostics.get("execution_batching")
        batch_partition_receipts.append(
            {
                SCENARIO_ID: scenario_id,
                METHOD_ID: method_id,
                BATCH_KEY: coordinates.get(BATCH_KEY),
                "configured_batch_size": int(plan["batch_size"]),
                "artifact_batch_size": artifact_batch_size,
                "batch_multiplier": (
                    int(str(batching.get("batch_multiplier")))
                    if isinstance(batching, dict)
                    else 1
                ),
            }
        )
        batches.append(value)
        coordinates_seen.append(coordinates)

    diagnostics_coord_set = set(diagnostics_coordinates)
    batch_coord_set = {
        _coordinate_tuple(coordinates) for coordinates in coordinates_seen
    }
    missing_diagnostics = batch_coord_set - diagnostics_coord_set
    orphan_diagnostics = diagnostics_coord_set - batch_coord_set
    if missing_diagnostics or orphan_diagnostics:
        raise ValueError(
            "P08 batch-diagnostics coordinate mismatch — batch write was likely "
            "interrupted; re-run affected batches before aggregating. "
            f"missing_diagnostics={sorted(str(dict(c)) for c in missing_diagnostics)}, "
            f"orphan_diagnostics={sorted(str(dict(c)) for c in orphan_diagnostics)}"
        )

    methodology = _validate_methodology(
        batches=batches,
        scenario_by_id=scenario_by_id,
    )
    active_batches = _filter_active_batches(
        batches=batches,
        scenario_by_id=scenario_by_id,
    )

    core = mapping(simulation.get("core"), "simulation.core")
    l3 = mapping(simulation.get("l3"), "simulation.l3")
    extended = mapping(
        simulation.get("extended_replication"),
        "simulation.extended_replication",
    )
    cost_replication = mapping(
        simulation.get("cost_sensitive_replication"),
        "simulation.cost_sensitive_replication",
    )
    cost_settings = mapping(
        simulation.get("cost_sensitive"),
        "simulation.cost_sensitive",
    )
    continuous = mapping(
        simulation.get("continuous_metrics"),
        "simulation.continuous_metrics",
    )
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")
    improvement = mapping(
        gate2.get("minimum_meaningful_improvement"),
        "evaluation.gate2.minimum_meaningful_improvement",
    )

    report = summarize_mcse(
        active_batches,
        minimum_replications=int(core["minimum_replications"]),
        maximum_replications=int(core["maximum_replications"]),
        pass_fail_mcse_maximum=float(core["pass_fail_mcse_max"]),
        l3_minimum_replications=int(l3["initial_replications"]),
        l3_maximum_replications=int(l3["maximum_replications"]),
        l3_pass_fail_mcse_maximum=float(l3["pass_fail_mcse_max"]),
        extended_minimum_replications=int(
            extended["minimum_replications"]
        ),
        extended_maximum_replications=int(
            extended["maximum_replications"]
        ),
        extended_pass_fail_mcse_maximum=float(
            extended["pass_fail_mcse_max"]
        ),
        cost_sensitive_minimum_replications=int(
            cost_replication["minimum_replications"]
        ),
        cost_sensitive_maximum_replications=int(
            cost_replication["maximum_replications"]
        ),
        cost_sensitive_pass_fail_mcse_maximum=float(
            cost_replication["pass_fail_mcse_max"]
        ),
        cost_mcse_relative_fraction=float(
            cost_settings["cost_mcse_relative_fraction"]
        ),
        continuous_mcse_fraction=float(
            continuous[
                "mcse_fraction_of_minimum_meaningful_improvement_max"
            ]
        ),
        minimum_meaningful_improvement=float(improvement["absolute"]),
    )

    mcse_status = str(report.get("status"))
    report["mcse_status"] = mcse_status
    report["execution_profile"] = active_profile
    report["methodology_validation"] = methodology
    report["batch_artifact_count"] = len(batches)
    report["active_batch_count"] = len(active_batches)
    report["batch_coordinates"] = coordinates_seen
    report["batch_partition_receipts"] = batch_partition_receipts
    report["batch_compaction_changes_replications_or_mcse"] = False
    if methodology["status"] != "PASS":
        report["status"] = "FAIL"
        report["reason_code"] = (
            "P08_ACTIVE_PROFILE_OR_COST_METRIC_CONTRACT_INCOMPLETE"
        )
        if args.check_only:
            print(f"P08_CONTROL_JSON={json.dumps(report, sort_keys=True)}")
        else:
            loaded.context.write("mcse_report", report, {})
        return 2

    # Control polling must never fail merely because more replications are needed.
    if args.check_only:
        print(f"P08_CONTROL_JSON={json.dumps(report, sort_keys=True)}")
        return 0

    # Only final aggregation applies the confirmatory terminal policy.
    protocol_role = mapping(
        simulation.get("protocol_role"),
        "simulation.protocol_role",
    )
    must_complete = bool(
        protocol_role.get("must_complete_before_outer_test")
    )
    confirmatory = False
    for scenario in scenario_by_id.values():
        if any(
            str(item.get("analysis_role")) == "confirmatory"
            for item in scenario.get("method_registry", [])
            if item.get("is_active") is True
        ):
            confirmatory = True
            break

    if must_complete and confirmatory:
        if report.get("mcse_status") == "CONTINUE":
            raise RuntimeError(
                "Final P08 aggregation reached while additional replications remain"
            )
        if report.get("precision_target_met") is not True:
            report["status"] = "FAIL"
            report["reason_code"] = "CONFIRMATORY_MCSE_TARGET_NOT_MET"
            loaded.context.write("mcse_report", report, {})
            return 3

    loaded.context.write("mcse_report", report, {})
    print(
        f"P08C aggregate status={report['status']} "
        f"profile={active_profile} batches={len(active_batches)} "
        f"methods_complete={methodology['status'] == 'PASS'}"
    )
    return 0


def _filter_active_batches(
    *,
    batches: list[pd.DataFrame],
    scenario_by_id: dict[str, dict[str, object]],
) -> list[pd.DataFrame]:
    output: list[pd.DataFrame] = []
    for batch in batches:
        if batch.empty:
            continue
        scenario_values = set(batch[SCENARIO_ID].astype(str))
        method_values = set(batch[METHOD_ID].astype(str))
        if len(scenario_values) != 1 or len(method_values) != 1:
            raise ValueError(
                "Each simulation batch must contain exactly one scenario and method"
            )
        scenario_id = next(iter(scenario_values))
        method_id = next(iter(method_values))
        scenario = scenario_by_id.get(scenario_id)
        if scenario is None:
            continue
        if method_id in active_method_ids(scenario):
            output.append(batch)
    return output


def _validate_methodology(
    *,
    batches: list[pd.DataFrame],
    scenario_by_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not scenario_by_id:
        return {"status": "FAIL", "reason_codes": ["NO_REGISTERED_SCENARIOS"]}

    first_scenario = next(iter(scenario_by_id.values()))
    first_registry = first_scenario.get("method_registry")
    if not isinstance(first_registry, list):
        raise ValueError("P08 scenario method_registry required")
    counts = profile_expected_counts(
        [
            cast(dict[str, object], item)
            for item in first_registry
            if isinstance(item, dict)
        ],
        active_only=True,
    )
    active_profile = str(first_scenario.get("execution_profile", ""))

    if not batches:
        return {
            "status": "FAIL",
            "reason_codes": ["NO_SIMULATION_BATCHES"],
            "execution_profile": active_profile,
            "expected_counts": counts,
            "missing_methods": [],
            "unexpected_methods": [],
            "missing_metrics": [],
            "duplicate_rows": 0,
        }

    combined = pd.concat(batches, ignore_index=True)
    required_columns = {
        SCENARIO_ID,
        METHOD_ID,
        REPLICATION_ID,
        METRIC_ID,
        "label_strategy_id",
        LEARNER_ID,
        "learner_tier",
        "training_cost_regime_id",
        IMBALANCE_TREATMENT_ID,
        "method_family",
        ANALYSIS_ROLE,
        EXECUTION_PROFILE,
        PROTOCOL_STATUS,
    }
    missing_columns = sorted(required_columns - set(combined.columns))
    if missing_columns:
        return {
            "status": "FAIL",
            "reason_codes": ["BATCH_METADATA_COLUMNS_MISSING"],
            "missing_columns": missing_columns,
            "execution_profile": active_profile,
            "expected_counts": counts,
        }

    duplicate_rows = int(
        combined.duplicated(
            [SCENARIO_ID, METHOD_ID, REPLICATION_ID, METRIC_ID],
            keep=False,
        ).sum()
    )
    missing_methods: list[dict[str, str]] = []
    unexpected_methods: list[dict[str, str]] = []
    missing_metrics: list[dict[str, object]] = []
    metadata_conflicts: list[dict[str, str]] = []
    profile_conflicts: list[dict[str, str]] = []

    for scenario_id, scenario in sorted(scenario_by_id.items()):
        raw_registry = scenario.get("method_registry")
        if not isinstance(raw_registry, list):
            raise ValueError(f"scenario={scenario_id}: method_registry required")
        registry = [
            cast(dict[str, object], item)
            for item in raw_registry
            if isinstance(item, dict)
        ]
        expected_ids = active_method_ids(scenario)
        scenario_counts = profile_expected_counts(
            registry,
            active_only=True,
        )
        if len(expected_ids) != int(scenario_counts["method_total"]):
            raise ValueError(
                f"scenario={scenario_id}: active method count mismatch"
            )

        scenario_frame = combined.loc[
            combined[SCENARIO_ID].astype(str) == scenario_id
        ]
        actual_ids = set(scenario_frame[METHOD_ID].astype(str))

        for method_id in sorted(expected_ids - actual_ids):
            missing_methods.append(
                {SCENARIO_ID: scenario_id, METHOD_ID: method_id}
            )
        for method_id in sorted(actual_ids - expected_ids):
            unexpected_methods.append(
                {SCENARIO_ID: scenario_id, METHOD_ID: method_id}
            )

        for method_id in sorted(expected_ids & actual_ids):
            spec = method_by_id(
                scenario,
                method_id,
                require_active=True,
            )
            frame = scenario_frame.loc[
                scenario_frame[METHOD_ID].astype(str) == method_id
            ]
            absent = sorted(
                required_metric_ids(spec)
                - set(frame[METRIC_ID].astype(str))
            )
            if absent:
                missing_metrics.append(
                    {
                        SCENARIO_ID: scenario_id,
                        METHOD_ID: method_id,
                        "metric_ids": absent,
                    }
                )
            for field in (
                "label_strategy_id",
                LEARNER_ID,
                "learner_tier",
                "training_cost_regime_id",
                IMBALANCE_TREATMENT_ID,
                "method_family",
                ANALYSIS_ROLE,
                EXECUTION_PROFILE,
                PROTOCOL_STATUS,
            ):
                if set(frame[field].astype(str)) != {str(spec[field])}:
                    metadata_conflicts.append(
                        {
                            SCENARIO_ID: scenario_id,
                            METHOD_ID: method_id,
                            "field": field,
                        }
                    )
            if set(frame[EXECUTION_PROFILE].astype(str)) != {
                str(scenario["execution_profile"])
            }:
                profile_conflicts.append(
                    {
                        SCENARIO_ID: scenario_id,
                        METHOD_ID: method_id,
                    }
                )

    reason_codes: list[str] = []
    if duplicate_rows:
        reason_codes.append("DUPLICATE_REPLICATION_METRIC_ROWS")
    if missing_methods:
        reason_codes.append("ACTIVE_PROFILE_METHODS_INCOMPLETE")
    if unexpected_methods:
        reason_codes.append("INACTIVE_OR_UNREGISTERED_METHOD_EXECUTED")
    if missing_metrics:
        reason_codes.append("REQUIRED_COST_METRICS_MISSING")
    if metadata_conflicts:
        reason_codes.append("METHOD_METADATA_CONFLICT")
    if profile_conflicts:
        reason_codes.append("EXECUTION_PROFILE_CONFLICT")

    return {
        "status": "PASS" if not reason_codes else "FAIL",
        "reason_codes": reason_codes,
        "execution_profile": active_profile,
        "expected_counts": counts,
        "missing_methods": missing_methods,
        "unexpected_methods": unexpected_methods,
        "missing_metrics": missing_metrics,
        "metadata_conflicts": metadata_conflicts,
        "profile_conflicts": profile_conflicts,
        "duplicate_rows": duplicate_rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())
