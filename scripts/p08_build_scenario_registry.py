"""P08 coordinator: publish the explicitly configured scenario registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import FISCAL_YEAR
from simulation.service import attach_development_covariate_pools, validate_scenarios


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
    feature_panel = loaded.context.read("feature_panel", {})
    feature_registry = [
        mapping(item, "feature registry item")
        for item in sequence(loaded.context.read("feature_registry", {}), "feature registry")
    ]
    if not isinstance(feature_panel, pd.DataFrame):
        raise ValueError("P08 feature panel must be a DataFrame")
    folds = mapping(loaded.registry.get("folds"), "folds")
    scenarios = attach_development_covariate_pools(
        scenarios=scenarios,
        feature_panel=feature_panel,
        feature_registry=feature_registry,
        year_column=physical_columns(loaded.registry)[FISCAL_YEAR],
        development_year_maximum=int(folds["initial_outer_year"]) - 1,
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
