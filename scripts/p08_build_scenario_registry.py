"""P08 coordinator: publish the explicitly configured scenario registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from simulation.service import validate_scenarios


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P08", state="FEATURED"
    )
    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    scenarios = validate_scenarios(
        sequence(simulation.get("operational_scenarios"), "simulation.operational_scenarios")
    )
    if args.dry_run:
        print(f"P08 registry dry-run: scenarios={len(scenarios)}")
        return 0
    loaded.context.write("simulation_scenario_registry", scenarios, {})
    status = "PASS" if scenarios else "SKIPPED reason=NO_OPERATIONAL_SCENARIOS"
    print(f"P08 registry status={status} scenarios={len(scenarios)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
