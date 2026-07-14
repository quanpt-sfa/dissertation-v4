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
from measurement.service import fold_local_l3_target_frame, measurement_target_frame
from modeling.service import ModelFitResult, fit_anchor_pu_fold, fit_fold_models


def _combine_fits(fits: list[ModelFitResult], *, selected_track_b: bool) -> ModelFitResult:
    model_rows: list[object] = []
    target_rows: list[object] = []
    for fit in fits:
        model_rows.extend(sequence(fit.models.get("models"), "fitted models"))
        target_rows.extend(sequence(fit.models.get("oof_training_targets"), "OOF training targets"))
    required_count = 2 if selected_track_b else 1
    required_pass = all(fit.models.get("status") == "PASS" for fit in fits[:required_count])
    return ModelFitResult(
        models={
            "status": "PASS" if required_pass else "SKIPPED",
            "track_a_executed": True,
            "track_b_required": selected_track_b,
            "track_b_executed": selected_track_b and len(fits) > 1,
            "models": model_rows,
            "oof_training_targets": target_rows,
        },
        oof_predictions=pd.concat([fit.oof_predictions for fit in fits], ignore_index=True),
        outer_predictions=pd.concat([fit.outer_predictions for fit in fits], ignore_index=True),
    )


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
    channel_selection = mapping(
        loaded.context.read("channel_measurement_selection", coordinates),
        "channel measurement selection",
    )
    matrices = mapping(
        loaded.context.read("source_channel_matrices", {}), "source-channel matrices"
    )
    anchor_capability = mapping(loaded.context.read("anchor_capability", {}), "anchor capability")
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
    selected_measurement = str(selected) if selected not in {None, "none"} else None
    learners = mapping(loaded.registry.get("learners"), "learners")
    learner_ids = [
        str(value) for value in sequence(learners.get("confirmatory"), "learners.confirmatory")
    ]
    settings = mapping(learners.get("settings"), "learners.settings")
    tuning = mapping(learners.get("tuning"), "learners.tuning")
    search_spaces = mapping(tuning.get("search_spaces"), "learners.tuning.search_spaces")
    missing_search_spaces = sorted(set(learner_ids) - set(search_spaces))
    if missing_search_spaces:
        raise RuntimeError(
            "TUNING_SEARCH_SPACES_NOT_LOCKED: add config for learners "
            f"{missing_search_spaces} under learners.tuning.search_spaces"
        )
    maximum_configurations = int(tuning["max_valid_configurations_per_learner_inner_fold"])
    track_a = fit_fold_models(
        feature_panel=cast(pd.DataFrame, panel),
        feature_registry=[mapping(item, "feature registry item") for item in feature_registry],
        label_inputs=cast(pd.DataFrame, inputs),
        weights=cast(pd.DataFrame, weights),
        outer_year=int(args.outer_fold),
        learner_ids=learner_ids,
        learner_settings=settings,
        target_id=primary,
        measurement_id=primary,
        columns=physical_columns(loaded.registry),
        random_state=derive_seed(loaded.protocol_hash, "P11", coordinates, "model_fit")
        % (2**32 - 1),
        track_id="track_a",
        learner_search_spaces=search_spaces,
        maximum_valid_configurations=maximum_configurations,
    )
    fits = [track_a]
    if selected_measurement is not None:
        missingness = mapping(measurement.get("l2_missingness"), "measurement.l2_missingness")
        minimum_channels = missingness.get("minimum_observed_channels")
        if not isinstance(minimum_channels, int) or minimum_channels < 1:
            raise RuntimeError(
                "selected Track B requires measurement.l2_missingness."
                "minimum_observed_channels to be empirically locked"
            )
        if selected_measurement == "L3_fixed_pi":
            target_values = fold_local_l3_target_frame(
                channel_selection=channel_selection,
                outer_year=int(args.outer_fold),
                columns=physical_columns(loaded.registry),
            )
        else:
            target_values = measurement_target_frame(
                matrices=matrices,
                measurement_id=selected_measurement,
                minimum_observed_channels=minimum_channels,
                columns=physical_columns(loaded.registry),
            )
        track_b_learners = [
            value for value in learner_ids if value in {"elastic_net_logistic", "main_boosting"}
        ]
        if not track_b_learners:
            raise RuntimeError("Track B requires registered logistic and/or boosting learners")
        fits.append(
            fit_fold_models(
                feature_panel=cast(pd.DataFrame, panel),
                feature_registry=[
                    mapping(item, "feature registry item") for item in feature_registry
                ],
                label_inputs=cast(pd.DataFrame, inputs),
                weights=cast(pd.DataFrame, weights),
                outer_year=int(args.outer_fold),
                learner_ids=track_b_learners,
                learner_settings=settings,
                target_id=selected_measurement,
                measurement_id=selected_measurement,
                columns=physical_columns(loaded.registry),
                random_state=derive_seed(
                    loaded.protocol_hash, "P11", coordinates, "track_b_model_fit"
                )
                % (2**32 - 1),
                target_values=target_values,
                soft_target=True,
                target_transform="training_ecdf" if selected_measurement == "L2" else "identity",
                track_id="track_b",
                learner_search_spaces=search_spaces,
                maximum_valid_configurations=maximum_configurations,
            )
        )
    pu_results: list[ModelFitResult] = []
    if anchor_capability.get("status") == "AVAILABLE":
        for anchor_source_id in sequence(
            anchor_capability.get("anchor_source_ids"), "anchor source IDs"
        ):
            pu_results.append(
                fit_anchor_pu_fold(
                    feature_panel=cast(pd.DataFrame, panel),
                    feature_registry=[
                        mapping(item, "feature registry item") for item in feature_registry
                    ],
                    label_inputs=cast(pd.DataFrame, inputs),
                    weights=cast(pd.DataFrame, weights),
                    anchor_source_id=str(anchor_source_id),
                    outer_year=int(args.outer_fold),
                    settings=settings,
                    columns=physical_columns(loaded.registry),
                    random_state=derive_seed(
                        loaded.protocol_hash,
                        "P11",
                        coordinates,
                        f"anchor_pu:{anchor_source_id}",
                    )
                    % (2**32 - 1),
                )
            )
    fits.extend(pu_results)
    result = _combine_fits(fits, selected_track_b=selected_measurement is not None)
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
    fitted_model_rows = [
        mapping(item, "fitted model")
        for item in sequence(result.models.get("models"), "fitted models")
    ]
    receipt: dict[str, object] = {
        "status": "PASS" if passed else "SKIPPED",
        "reason_code": None if passed else "REQUIRED_MODEL_TRACK_INCOMPLETE",
        "track_a_status": track_a.models["status"],
        "track_b_status": fits[1].models["status"]
        if selected_measurement is not None
        else "NOT_SELECTED",
        "anchor_pu_statuses": [fit.models["status"] for fit in pu_results]
        if pu_results
        else ["UNAVAILABLE_BY_DESIGN"],
        "selected_measurement": selected_measurement or "none",
        "tuning_valid_configuration_counts": {
            str(item["model_id"]): item.get("valid_configuration_count")
            for item in fitted_model_rows
            if item.get("training_mechanism") != "bagging_pu"
        },
        "tuning_runtime_seconds": sum(
            float(item.get("tuning_runtime_seconds", 0.0)) for item in fitted_model_rows
        ),
        "tuning_budget_maximum": maximum_configurations,
        "protocol_hash": loaded.protocol_hash,
        "split_registry_hash": stable_hash(splits),
        "measurement_selection_hash": stable_hash(selection),
        "channel_measurement_selection_hash": stable_hash(channel_selection),
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
