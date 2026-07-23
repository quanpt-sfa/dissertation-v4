"""P08B worker: run one active deterministic cost-sensitive method batch."""

# ruff: noqa: E402

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
from simulation.l3_variants import (
    L3_VARIANT_IDS,
    run_l3_variant_batch,
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
    required_metric_ids,
)
from simulation.replication_contract import (
    replication_plan as _replication_plan,
)
from simulation.scenario_contract import (
    validate_scenario_target_identity,
)
from simulation.service import run_batch

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
    simulation = mapping(
        loaded.registry.get("simulation"),
        "simulation",
    )
    scenarios = sequence(
        loaded.context.read("simulation_scenario_registry", {}),
        "simulation scenario registry",
    )
    mapped = [mapping(item, "simulation scenario") for item in scenarios]
    matches = [item for item in mapped if item.get(SCENARIO_ID) == args.scenario_id]
    if len(matches) != 1:
        raise ValueError(f"scenario={args.scenario_id}: exactly one registered scenario required")
    scenario = matches[0]
    validate_scenario_target_identity(scenario)

    # Default require_active=True prevents accidental execution of a retained
    # future/exploratory method outside the config-locked profile.
    method_spec = method_by_id(scenario, args.method_id)

    plan = _replication_plan(simulation, method_spec)
    batch_size = int(plan["batch_size"])

    expected_batch_id = f"{args.start:06d}-{args.start + args.count - 1:06d}"

    if args.batch_id != expected_batch_id:
        raise ValueError(
            f"batch_id={args.batch_id} differs from start/count-derived value={expected_batch_id}"
        )

    if args.start % batch_size != 0:
        raise ValueError("batch start must align with the replication batch size")

    if args.count > batch_size:
        raise ValueError("batch count exceeds the locked batch size")

    import numpy as np

    def data_rng_factory(
        rep_id: int,
    ) -> np.random.Generator:
        return generator(
            loaded.protocol_hash,
            "P08_REPLICATION_DATA",
            {
                SCENARIO_ID: args.scenario_id,
                REPLICATION_ID: str(rep_id),
            },
            str(rep_id),
        )

    def model_rng_factory(
        rep_id: int,
    ) -> np.random.Generator:
        return generator(
            loaded.protocol_hash,
            "P08_REPLICATION_MODEL",
            {
                SCENARIO_ID: args.scenario_id,
                METHOD_ID: args.method_id,
                REPLICATION_ID: str(rep_id),
            },
            str(rep_id),
        )

    diagnostics: dict[str, object] = {}
    replication_range = range(
        args.start,
        args.start + args.count,
    )
    if args.method_id in L3_VARIANT_IDS:
        batch = run_l3_variant_batch(
            scenario,
            method_id=args.method_id,
            replications=replication_range,
            data_rng_factory=data_rng_factory,
            diagnostics=diagnostics,
        )
    else:
        batch = run_batch(
            scenario,
            method_id=args.method_id,
            replications=replication_range,
            data_rng_factory=data_rng_factory,
            model_rng_factory=model_rng_factory,
            diagnostics=diagnostics,
        )

    if not isinstance(batch, pd.DataFrame) or batch.empty:
        raise ValueError("P08B batch runner must return a nonempty DataFrame")
    missing_columns = sorted(_REQUIRED_OUTPUT_COLUMNS - set(batch.columns))
    if missing_columns:
        raise ValueError(f"P08B batch is missing columns={missing_columns}")
    if set(batch[SCENARIO_ID].astype(str)) != {args.scenario_id}:
        raise ValueError("P08B batch scenario_id differs from worker coordinates")
    if set(batch[METHOD_ID].astype(str)) != {args.method_id}:
        raise ValueError("P08B batch method_id differs from worker coordinates")
    if batch.duplicated(
        [
            SCENARIO_ID,
            METHOD_ID,
            REPLICATION_ID,
            METRIC_ID,
        ],
        keep=False,
    ).any():
        raise ValueError("P08B batch contains duplicate replication-metric rows")

    batch[LABEL_STRATEGY_ID] = str(method_spec[LABEL_STRATEGY_ID])
    batch[LEARNER_ID] = str(method_spec[LEARNER_ID])
    batch[LEARNER_TIER] = str(method_spec[LEARNER_TIER])
    batch[TRAINING_COST_REGIME_ID] = str(method_spec[TRAINING_COST_REGIME_ID])
    batch[IMBALANCE_TREATMENT_ID] = str(method_spec[IMBALANCE_TREATMENT_ID])
    batch[METHOD_FAMILY] = str(method_spec[METHOD_FAMILY])
    batch[ANALYSIS_ROLE] = str(method_spec[ANALYSIS_ROLE])
    batch[EXECUTION_PROFILE] = str(method_spec[EXECUTION_PROFILE])
    batch[PROTOCOL_STATUS] = str(method_spec[PROTOCOL_STATUS])
    batch["method_contract_version"] = 7

    expected_replications = set(replication_range)
    actual_replications = set(batch[REPLICATION_ID].dropna().astype(int))

    if actual_replications != expected_replications:
        raise ValueError(
            "P08B replication IDs differ from locked start/count; "
            f"missing={sorted(expected_replications - actual_replications)}, "
            f"unexpected={sorted(actual_replications - expected_replications)}"
        )

    required_metrics = required_metric_ids(method_spec)

    for replication_id, frame in batch.groupby(
        REPLICATION_ID,
        sort=False,
    ):
        actual_metrics = set(frame[METRIC_ID].astype(str))
        missing = sorted(required_metrics - actual_metrics)
        unexpected = sorted(actual_metrics - required_metrics)

        if missing or unexpected:
            raise ValueError(
                f"P08B replication={replication_id}: "
                f"missing_metrics={missing}, "
                f"unexpected_metrics={unexpected}"
            )

    scenario_key_str = "scenario_" + "key"
    method_key_str = "method_" + "key"
    batch_key_str = "batch_" + "key"
    write_coordinates = {
        scenario_key_str: str(scenario[scenario_key_str]),
        method_key_str: str(method_spec[method_key_str]),
        batch_key_str: (f"b{args.start // batch_size:04d}"),
    }
    loaded.context.write(
        "simulation_batches",
        batch,
        write_coordinates,
    )
    loaded.context.write(
        "model_diagnostics",
        diagnostics,
        write_coordinates,
    )
    print(
        f"P08B batch status=PASS "
        f"scenario={args.scenario_id} "
        f"profile={method_spec[EXECUTION_PROFILE]} "
        f"method={args.method_id} "
        f"rows={len(batch)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
