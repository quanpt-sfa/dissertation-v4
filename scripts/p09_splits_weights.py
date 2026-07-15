"""P09 CLI: rolling-origin split registries and fold-aware weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, outer_fold_ids, physical_columns, sequence, stable_hash
from splits.service import build_splits_and_weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P09", state="SIMULATED"
    )
    fold_ids = outer_fold_ids(loaded.registry)
    if args.dry_run:
        print(f"P09 dry-run: folds={','.join(fold_ids)}")
        return 0
    panel = loaded.context.read("feature_panel", {})
    feature_registry = [
        mapping(item, "feature registry item")
        for item in sequence(loaded.context.read("feature_registry", {}), "feature registry")
    ]
    risk_sets = loaded.context.read("risk_sets", {})
    matrices = mapping(loaded.context.read("source_channel_matrices", {}), "source matrices")
    observability = mapping(
        loaded.context.read("observability_registry", {}), "observability registry"
    )
    fold_eligibility = [
        mapping(item, "fold eligibility item")
        for item in sequence(loaded.context.read("fold_eligibility", {}), "fold eligibility")
    ]
    if not isinstance(panel, pd.DataFrame) or not isinstance(risk_sets, pd.DataFrame):
        raise ValueError("P09 panel inputs must be DataFrames")
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    common = mapping(evaluation.get("gate_common"), "evaluation.gate_common")
    support = sequence(common.get("propensity_support"), "gate_common.propensity_support")
    if len(support) != 2:
        raise ValueError("propensity support requires lower and upper bounds")
    result = build_splits_and_weights(
        feature_panel=panel,
        risk_sets=risk_sets,
        matrices=matrices,
        observability=observability,
        outer_years=[int(value) for value in fold_ids],
        fold_eligibility=fold_eligibility,
        support_bounds=(float(support[0]), float(support[1])),
        support_fraction_minimum=float(common["support_fraction_min"]),
        ess_fraction_minimum=float(common["ess_fraction_min"]),
        ess_absolute_minimum=float(common["ess_absolute_min"]),
        columns=physical_columns(loaded.registry),
        verification_feature_ids=[
            str(item["feature_id"])
            for item in feature_registry
            if item.get("role") == "observability"
            and item.get("available_before_verification", True) is True
        ],
        primary_target_id=(
            str(value)
            if (
                value := mapping(loaded.registry.get("measurement"), "measurement").get(
                    "primary_target_id"
                )
            )
            is not None
            else None
        ),
    )
    loaded.context.write("temporal_split_registry", result.splits, {})
    loaded.context.write("channel_time_split_registry", result.channel_splits, {})
    for fold_id, weights in result.weights.items():
        loaded.context.write("fold_aware_weights", weights, {"fold_id": fold_id})
        diagnostics = result.weight_diagnostics[fold_id]
        diagnostics["split_registry_hash"] = stable_hash(result.splits)
        loaded.context.write("weight_diagnostics", diagnostics, {"fold_id": fold_id})
    print(f"P09 status=PASS folds={len(result.splits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
