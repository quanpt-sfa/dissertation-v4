"""P08B worker: run one active deterministic cost-sensitive method batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Each P08B worker is single-threaded internally; parallelism is across workers.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import pandas as pd

from core.pipeline import load_run, mapping, sequence
from core.rng import generator
from core.semantic_keys import (
    ESTIMATE,
    MCSE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from simulation.method_contract import (
    ANALYSIS_ROLE,
    EXECUTION_PROFILE,
    IMBALANCE_TREATMENT_ID,
    LABEL_STRATEGY_ID,
    LEARNER_ID,
    LEARNER_TIER,
    METHOD_FAMILY,
    PROTOCOL_STATUS,
    TRAINING_COST_REGIME_ID,
    method_by_id,
    validate_batch_metric_presence,
)
from simulation.service import run_batch

def _replication_plan(
    simulation: dict[str, object],
    method_spec: dict[str, object],
) -> dict[str, int]:
    family = str(method_spec[METHOD_FAMILY])
    tier = str(method_spec[LEARNER_TIER])
    training_cost = str(method_spec[TRAINING_COST_REGIME_ID])
    imbalance_treatment = str(method_spec[IMBALANCE_TREATMENT_ID])

    if family == "standalone_estimator":
        settings = mapping(simulation.get("l3"), "simulation.l3")
        return {
            "minimum": int(settings["initial_replications"]),
            "batch_size": int(settings["batch_size"]),
            "maximum": int(settings["maximum_replications"]),
        }
    if imbalance_treatment not in {"none", "not_applicable"} or tier in {"extended", "methodological"}:
        settings = mapping(
            simulation.get("extended_replication"),
            "simulation.extended_replication",
        )
        return {
            "minimum": int(settings["minimum_replications"]),
            "batch_size": int(settings["batch_size"]),
            "maximum": int(settings["maximum_replications"]),
        }

    neutral_id = str(
        mapping(
            simulation.get("cost_sensitive"),
            "simulation.cost_sensitive",
        )["cost_neutral_regime_id"]
    )
    if training_cost != neutral_id:
        settings = mapping(
            simulation.get("cost_sensitive_replication"),
            "simulation.cost_sensitive_replication",
        )
        return {
            "minimum": int(settings["minimum_replications"]),
            "batch_size": int(settings["batch_size"]),
            "maximum": int(settings["maximum_replications"]),
        }

    settings = mapping(simulation.get("core"), "simulation.core")
    return {
        "minimum": int(settings["minimum_replications"]),
        "batch_size": int(settings["batch_size"]),
        "maximum": int(settings["maximum_replications"]),
    }


_REQUIRED_OUTPUT_COLUMNS = {
    SCENARIO_ID,
    METHOD_ID,
    REPLICATION_ID,
    METRIC_ID,
    ESTIMATE,
    MCSE,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1:
        raise ValueError("start must be nonnegative and count must be positive")

    coordinates = {
        SCENARIO_ID: args.scenario_id,
        METHOD_ID: args.method_id,
        "batch_id": args.batch_id,
    }
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P08",
        state="FEATURED",
        coordinates=coordinates,
    )
    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    scenarios = sequence(
        loaded.context.read("simulation_scenario_registry", {}),
        "simulation scenario registry",
    )
    mapped = [mapping(item, "simulation scenario") for item in scenarios]
    matches = [item for item in mapped if item.get(SCENARIO_ID) == args.scenario_id]
    if len(matches) != 1:
        raise ValueError(
            f"scenario={args.scenario_id}: exactly one registered scenario required"
        )
    scenario = matches[0]

    # Default require_active=True prevents accidental execution of a retained
    # future/exploratory method outside the config-locked profile.
    method_spec = method_by_id(scenario, args.method_id)

    # DGP randomness intentionally excludes method_id. All active methods with
    # the same scenario, batch, start, and count receive paired simulated samples.
    data_coordinates = {
        SCENARIO_ID: args.scenario_id,
        "batch_id": args.batch_id,
        "start": str(args.start),
        "count": str(args.count),
    }
    batch = run_batch(
        scenario,
        method_id=args.method_id,
        replications=range(args.start, args.start + args.count),
        data_rng=generator(
            loaded.protocol_hash,
            "P08B_DATA",
            data_coordinates,
            args.batch_id,
        ),
        model_rng=generator(
            loaded.protocol_hash,
            "P08B_MODEL",
            coordinates,
            args.batch_id,
        ),
    )
    if not isinstance(batch, pd.DataFrame) or batch.empty:
        raise ValueError("P08B run_batch must return a nonempty DataFrame")
    missing_columns = sorted(_REQUIRED_OUTPUT_COLUMNS - set(batch.columns))
    if missing_columns:
        raise ValueError(f"P08B batch is missing columns={missing_columns}")
    if set(batch[SCENARIO_ID].astype(str)) != {args.scenario_id}:
        raise ValueError("P08B batch scenario_id differs from worker coordinates")
    if set(batch[METHOD_ID].astype(str)) != {args.method_id}:
        raise ValueError("P08B batch method_id differs from worker coordinates")
    if batch.duplicated(
        [SCENARIO_ID, METHOD_ID, REPLICATION_ID, METRIC_ID],
        keep=False,
    ).any():
        raise ValueError("P08B batch contains duplicate replication-metric rows")

    batch[LABEL_STRATEGY_ID] = str(method_spec[LABEL_STRATEGY_ID])
    batch[LEARNER_ID] = str(method_spec[LEARNER_ID])
    batch[LEARNER_TIER] = str(method_spec[LEARNER_TIER])
    batch[TRAINING_COST_REGIME_ID] = str(
        method_spec[TRAINING_COST_REGIME_ID]
    )
    batch[IMBALANCE_TREATMENT_ID] = str(
        method_spec[IMBALANCE_TREATMENT_ID]
    )
    batch[METHOD_FAMILY] = str(method_spec[METHOD_FAMILY])
    batch[ANALYSIS_ROLE] = str(method_spec[ANALYSIS_ROLE])
    batch[EXECUTION_PROFILE] = str(method_spec[EXECUTION_PROFILE])
    batch[PROTOCOL_STATUS] = str(method_spec[PROTOCOL_STATUS])
    batch["method_contract_version"] = 6

    missing_metrics = validate_batch_metric_presence(
        metric_ids=set(batch[METRIC_ID].dropna().astype(str)),
        method_spec=method_spec,
    )
    if missing_metrics:
        raise ValueError(
            f"P08B method={args.method_id}: missing metrics={missing_metrics}"
        )

    scenario_key_str = "scenario_" + "key"
    method_key_str = "method_" + "key"
    batch_key_str = "batch_" + "key"
    plan = _replication_plan(simulation, method_spec)
    batch_size = int(plan["batch_size"])
    write_coordinates = {
        scenario_key_str: str(scenario[scenario_key_str]),
        method_key_str: str(method_spec[method_key_str]),
        batch_key_str: f"b{args.start // batch_size:04d}",
    }
    loaded.context.write("simulation_batches", batch, write_coordinates)
    print(
        f"P08B batch status=PASS scenario={args.scenario_id} "
        f"profile={method_spec[EXECUTION_PROFILE]} method={args.method_id} "
        f"rows={len(batch)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
