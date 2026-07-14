"""P08 worker: execute one deterministic scenario batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from core.rng import generator
from core.semantic_keys import METHOD_ID, SCENARIO_ID
from simulation.service import run_batch


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
    scenarios = sequence(
        loaded.context.read("simulation_scenario_registry", {}),
        "simulation scenario registry",
    )
    mapped_scenarios = [mapping(item, "simulation scenario") for item in scenarios]
    matches = [item for item in mapped_scenarios if item.get(SCENARIO_ID) == args.scenario_id]
    if len(matches) != 1:
        raise ValueError(f"scenario={args.scenario_id}: exactly one registered scenario required")
    if args.start < 0 or args.count < 1:
        raise ValueError("start must be nonnegative and count must be positive")
    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    methods = simulation.get("methods")
    if not isinstance(methods, list) or args.method_id not in methods:
        raise ValueError(f"method={args.method_id}: method is not registered")
    batch = run_batch(
        matches[0],
        method_id=args.method_id,
        replications=range(args.start, args.start + args.count),
        rng=generator(loaded.protocol_hash, "P08", coordinates, args.batch_id),
    )
    loaded.context.write("simulation_batches", batch, coordinates)
    print(
        f"P08 batch status=PASS scenario={args.scenario_id} "
        f"method={args.method_id} rows={len(batch)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
