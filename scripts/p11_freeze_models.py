"""P11 CLI: fit fold-local models, emit predictions, and freeze all hashes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence, stable_hash
from core.rng import derive_seed
from core.semantic_keys import OUTER_FOLD
from modeling.service import fit_fold_models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outer-fold", required=True)
    args = parser.parse_args()
    coordinates = {OUTER_FOLD: args.outer_fold}
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P11",
        state="SELECTED",
        coordinates=coordinates,
    )
    selection = mapping(
        loaded.context.read("measurement_selection_registry", coordinates),
        "measurement selection",
    )
    splits = loaded.context.read("temporal_split_registry", {})
    panel = loaded.context.read("feature_panel", {})
    feature_registry = sequence(loaded.context.read("feature_registry", {}), "feature registry")
    inputs = loaded.context.read("l0_l1_inputs", {})
    weights = loaded.context.read("fold_aware_weights", {"fold_id": args.outer_fold})
    weight_diagnostics = mapping(
        loaded.context.read("weight_diagnostics", {"fold_id": args.outer_fold}),
        "weight diagnostics",
    )
    if (
        weight_diagnostics.get("fit_scope") != "development_history"
        or weight_diagnostics.get("outer_rows_used_in_fit") != 0
        or weight_diagnostics.get("analytical_use_allowed") is not True
    ):
        raise ValueError("P11 refuses weights without passing development-only diagnostics")
    source_manifest = mapping(loaded.context.read("source_config_manifest", {}), "source manifest")
    environment = mapping(
        loaded.context.read("environment_observation", {}), "environment observation"
    )
    if not all(isinstance(value, pd.DataFrame) for value in (panel, inputs, weights)):
        raise ValueError("P11 tabular inputs must be DataFrames")
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    primary = measurement.get("track_a_primary_endpoint")
    if not isinstance(primary, str):
        raise ValueError("measurement.track_a_primary_endpoint required")
    selected = selection.get("selected_measurement")
    measurement_id = str(selected) if selected not in {None, "none"} else primary
    learners = mapping(loaded.registry.get("learners"), "learners")
    learner_ids = [
        str(value) for value in sequence(learners.get("confirmatory"), "learners.confirmatory")
    ]
    settings = mapping(learners.get("settings"), "learners.settings")
    result = fit_fold_models(
        feature_panel=cast(pd.DataFrame, panel),
        feature_registry=[mapping(item, "feature registry item") for item in feature_registry],
        label_inputs=cast(pd.DataFrame, inputs),
        weights=cast(pd.DataFrame, weights),
        outer_year=int(args.outer_fold),
        learner_ids=learner_ids,
        learner_settings=settings,
        target_id=primary,
        measurement_id=measurement_id,
        columns=physical_columns(loaded.registry),
        random_state=derive_seed(loaded.protocol_hash, "P11", coordinates, "model_fit")
        % (2**32 - 1),
    )
    model_manifest = loaded.context.write("model_artifacts", result.models, coordinates)
    oof_manifest = loaded.context.write(
        "development_oof_predictions", result.oof_predictions, coordinates
    )
    outer_manifest = loaded.context.write(
        "raw_outer_predictions", result.outer_predictions, coordinates
    )
    feature_config = mapping(loaded.registry.get("features"), "features")
    calibration = mapping(loaded.registry.get("calibration"), "calibration")
    passed = (
        result.models["status"] == "PASS"
        and not result.oof_predictions.empty
        and not result.outer_predictions.empty
    )
    receipt: dict[str, object] = {
        "status": "PASS" if passed else "SKIPPED",
        "reason_code": None if passed else "FEATURE_REGISTRY_EMPTY",
        "protocol_hash": loaded.protocol_hash,
        "split_registry_hash": stable_hash(splits),
        "measurement_selection_hash": stable_hash(selection),
        "feature_registry_hash": stable_hash(feature_registry),
        "preprocessing_hash": stable_hash(feature_config.get("preprocessing")),
        "model_settings_hash": stable_hash(settings),
        "weight_diagnostics_hash": stable_hash(weight_diagnostics),
        "model_artifact_hash": model_manifest["content_hash"],
        "development_oof_hash": oof_manifest["content_hash"],
        "raw_outer_predictions_hash": outer_manifest["content_hash"],
        "calibration_plan_hash": stable_hash(calibration),
        "git_commit": source_manifest.get("git_commit"),
        "environment_hash": stable_hash(environment),
    }
    loaded.context.write("model_freeze_receipt", receipt, coordinates)
    print(
        f"P11 status={receipt['status']} fold={args.outer_fold} "
        f"models={len(sequence(result.models['models'], 'fitted models'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
