"""P07 CLI: validate the feature registry and publish the as-of feature panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from features.service import build_feature_panel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P07",
        state="OBSERVABLE",
    )
    features = mapping(loaded.registry.get("features"), "features")
    raw_registry = sequence(features.get("registry"), "features.registry")
    definitions = [mapping(item, "features.registry item") for item in raw_registry]
    if args.dry_run:
        print(f"P07 dry-run: registered_features={len(definitions)}")
        return 0
    panel = loaded.context.read("firm_year_panel", {})
    risk_sets = loaded.context.read("risk_sets", {})
    loaded.context.read("observability_registry", {})
    if not isinstance(panel, pd.DataFrame) or not isinstance(risk_sets, pd.DataFrame):
        raise ValueError("P07 panel inputs must be DataFrames")
    result = build_feature_panel(
        firm_year_panel=panel,
        risk_sets=risk_sets,
        feature_definitions=cast(list[dict[str, object]], definitions),
        columns=physical_columns(loaded.registry),
        blocked_label_semantics=[
            str(value)
            for value in sequence(
                mapping(
                    features.get("label_derived_feature_firewall"),
                    "features.label_derived_feature_firewall",
                ).get("same_firm_year_blocked_semantics"),
                "features.label_derived_feature_firewall.same_firm_year_blocked_semantics",
            )
        ],
    )
    if args.validate_only:
        return 0
    loaded.context.write("feature_panel", result.panel, {})
    loaded.context.write("feature_registry", result.registry, {})
    loaded.context.write("leakage_registry", result.leakage_registry, {})
    status = "PASS" if result.registry else "SKIPPED"
    reason = "" if result.registry else " reason=FEATURE_REGISTRY_EMPTY"
    print(f"P07 status={status} features={len(result.registry)}{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
