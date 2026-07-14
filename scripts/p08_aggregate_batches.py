"""P08 collector: aggregate completed immutable batches into an MCSE report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping
from simulation.service import summarize_mcse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P08", state="FEATURED"
    )
    batches: list[pd.DataFrame] = []
    for item in loaded.context.store.inventory():
        if item.get("artifact_id") != "simulation_batches":
            continue
        coordinates_raw = item.get("coordinates")
        if not isinstance(coordinates_raw, dict):
            raise ValueError("simulation batch manifest coordinates required")
        coordinates = {
            str(key): str(value)
            for key, value in cast(dict[object, object], coordinates_raw).items()
        }
        value = loaded.context.read("simulation_batches", coordinates)
        if isinstance(value, pd.DataFrame):
            batches.append(value)
    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    core = mapping(simulation.get("core"), "simulation.core")
    report = summarize_mcse(
        batches,
        minimum_replications=int(core["minimum_replications"]),
        maximum_replications=int(core["maximum_replications"]),
        pass_fail_mcse_maximum=float(core["pass_fail_mcse_max"]),
    )
    if args.check_only:
        print(f"P08_CONTROL_JSON={json.dumps(report, sort_keys=True)}")
        return 0
    loaded.context.write("mcse_report", report, {})
    print(f"P08 aggregate status={report['status']} batches={len(batches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
