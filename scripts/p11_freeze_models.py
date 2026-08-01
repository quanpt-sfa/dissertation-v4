"""P11 CLI: fit sequential fold-local measurement tracks and freeze hashes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.fold_control import require_confirmatory_fold, require_primary_target
from core.pipeline import load_run, mapping, physical_columns, sequence, stable_hash
from core.rng import derive_seed
from core.semantic_keys import FISCAL_YEAR, MEASUREMENT_ID, OUTER_FOLD
from measurement.service import fold_local_l3_target_frame, measurement_target_frame
from modeling.pilot_contracts import prepare_model_feature_contract
from modeling.service import ModelFitResult, fit_anchor_pu_fold, fit_fold_models
from selection.nested_refit import validate_nested_refit_receipt
from selection.track_plan import executable_tracks


def _combine_fits(
    fits: list[ModelFitResult],
    *,
    required_fit: ModelFitResult,
    track_statuses: dict[str, str],
) -> ModelFitResult:
    model_rows: list[object] = []
    target_rows: list[object] = []
    oof_frames: list[pd.DataFrame] = []
    outer_frames: list[pd.DataFrame] = []
    for fit in fits:
        model_rows.extend(sequence(fit.models.get("models"), "fitted models"))
        target_rows.extend(sequence(fit.models.get("oof_training_targets"), "OOF training targets"))
        if not fit.oof_predictions.empty:
            oof_frames.append(fit.oof_predictions)
        if not fit.outer_predictions.empty:
            outer_frames.append(fit.outer_predictions)
    required_pass = (
        required_fit.models.get("status") == "PASS"
        and not required_fit.oof_predictions.empty
        and not required_fit.outer_predictions.empty
    )
    return ModelFitResult(
        models={
            "status": "PASS" if required_pass else "SKIPPED",
            "sequential_tracks": True,
            "required_primary_track_passed": required_pass,
            "track_statuses": track_statuses,
            "models": model_rows,
            "oof_training_targets": target_rows,
        },
        oof_predictions=pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame(),
        outer_predictions=pd.concat(outer_frames, ignore_index=True)
        if outer_frames
        else pd.DataFrame(),
    )


def _progress(message: str) -> None:
    print(message, flush=True)


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
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    primary = require_primary_target(measurement, "P11")
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
    if not feature_registry:
        raise RuntimeError("P11_PRODUCTION_PATH_BLOCKED: FEATURE_REGISTRY_EMPTY")
    fold_eligibility = loaded.context.read("fold_eligibility", {})
    require_confirmatory_fold(fold_eligibility, args.outer_fold, "P11", primary)
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

    columns = physical_columns(loaded.registry)
    feature_items = [mapping(item, "feature registry item") for item in feature_registry]
    feature_contract = prepare_model_feature_contract(
        panel=cast(pd.DataFrame, panel),
        registry=feature_items,
        year_column=columns[FISCAL_YEAR],
        outer_year=int(args.outer_fold),
    )
    effective_feature_registry = feature_contract.registry
    _progress(
        "P11 preflight "
        f"fold={args.outer_fold} effective_features={feature_contract.summary['effective_feature_count']} "
        f"excluded_features={feature_contract.summary['runtime_exclusion_count']}"
    )
    if feature_contract.exclusions:
        _progress(
            "P11 fail-closed exclusions="
            + ",".join(
                f"{item['feature_id']}:{item['reason_code']}"
                for item in feature_contract.exclusions
            )
        )

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
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")
    reference_by_candidate = mapping(
        gate2.get("reference_by_candidate"), "evaluation.gate2.reference_by_candidate"
    )
    required_feature_groups = sorted(
        {
            str(model_id).rsplit(":", 1)[-1]
            for model_id in [*reference_by_candidate.keys(), *reference_by_candidate.values()]
        }
    )
    reproducibility = mapping(loaded.registry.get("reproducibility"), "reproducibility")
    seed_offsets = mapping(reproducibility.get("seed_offsets"), "reproducibility.seed_offsets")
    candidate_seed_offset = int(seed_offsets["tuning_candidate"])
    plan_tracks = executable_tracks(selection)
    optional_measurements = [
        str(item[MEASUREMENT_ID]) for item in plan_tracks if item.get("role") != "required_primary"
    ]
    nested_receipt = validate_nested_refit_receipt(
        channel_selection,
        outer_fold=args.outer_fold,
        required_optional_measurements=optional_measurements,
    )
    primary_track = next(
        (item for item in plan_tracks if item.get("role") == "required_primary"), None
    )
    if primary_track is None or primary_track.get(MEASUREMENT_ID) != primary:
        raise RuntimeError("P11 sequential plan is missing the required primary track")

    primary_started = time.perf_counter()
    _progress(
        f"P11 track_start fold={args.outer_fold} track={primary_track['track_id']} "
        f"learners={learner_ids}"
    )
    track_l1 = fit_fold_models(
        feature_panel=cast(pd.DataFrame, panel),
        feature_registry=effective_feature_registry,
        label_inputs=cast(pd.DataFrame, inputs),
        weights=cast(pd.DataFrame, weights),
        outer_year=int(args.outer_fold),
        learner_ids=learner_ids,
        learner_settings=settings,
        target_id=primary,
        measurement_id=primary,
        columns=columns,
        random_state=derive_seed(loaded.protocol_hash, "P11", coordinates, "track_l1_model_fit")
        % (2**32 - 1),
        track_id=str(primary_track["track_id"]),
        learner_search_spaces=search_spaces,
        maximum_valid_configurations=maximum_configurations,
        required_feature_groups=required_feature_groups,
        candidate_seed_offset=candidate_seed_offset,
    )
    _progress(
        f"P11 track_end fold={args.outer_fold} track={primary_track['track_id']} "
        f"status={track_l1.models['status']} elapsed_seconds={time.perf_counter() - primary_started:.3f}"
    )
    fits = [track_l1]
    track_statuses: dict[str, str] = {
        str(primary_track["track_id"]): str(track_l1.models["status"])
    }

    optional_learners = [
        value for value in learner_ids if value in {"elastic_net_logistic", "main_boosting"}
    ]
    if not optional_learners:
        raise RuntimeError("Optional measurement tracks require logistic and/or boosting learners")
    missingness = mapping(measurement.get("l2_missingness"), "measurement.l2_missingness")
    minimum_channels = missingness.get("minimum_observed_channels")
    for track in plan_tracks:
        if track.get("role") == "required_primary":
            continue
        measurement_id = str(track[MEASUREMENT_ID])
        track_id = str(track["track_id"])
        if not isinstance(minimum_channels, int) or minimum_channels < 1:
            track_statuses[track_id] = "SKIPPED_CONFIGURATION_UNAVAILABLE"
            _progress(
                f"P11 track_skip fold={args.outer_fold} track={track_id} "
                "reason=MINIMUM_OBSERVED_CHANNELS_UNAVAILABLE"
            )
            continue
        if measurement_id == "L3_fixed_pi":
            target_values = fold_local_l3_target_frame(
                channel_selection=channel_selection,
                outer_year=int(args.outer_fold),
                columns=columns,
            )
        else:
            target_values = measurement_target_frame(
                matrices=matrices,
                measurement_id=measurement_id,
                minimum_observed_channels=minimum_channels,
                columns=columns,
            )
        optional_started = time.perf_counter()
        _progress(
            f"P11 track_start fold={args.outer_fold} track={track_id} "
            f"learners={optional_learners}"
        )
        optional_fit = fit_fold_models(
            feature_panel=cast(pd.DataFrame, panel),
            feature_registry=effective_feature_registry,
            label_inputs=cast(pd.DataFrame, inputs),
            weights=cast(pd.DataFrame, weights),
            outer_year=int(args.outer_fold),
            learner_ids=optional_learners,
            learner_settings=settings,
            target_id=primary,
            measurement_id=measurement_id,
            columns=columns,
            random_state=derive_seed(
                loaded.protocol_hash, "P11", coordinates, f"{track_id}_model_fit"
            )
            % (2**32 - 1),
            target_values=target_values,
            soft_target=True,
            target_transform="training_ecdf" if measurement_id == "L2" else "identity",
            track_id=track_id,
            learner_search_spaces=search_spaces,
            maximum_valid_configurations=maximum_configurations,
            candidate_seed_offset=candidate_seed_offset,
        )
        _progress(
            f"P11 track_end fold={args.outer_fold} track={track_id} "
            f"status={optional_fit.models['status']} "
            f"elapsed_seconds={time.perf_counter() - optional_started:.3f}"
        )
        fits.append(optional_fit)
        track_statuses[track_id] = str(optional_fit.models["status"])

    pu_results: list[ModelFitResult] = []
    if anchor_capability.get("status") == "AVAILABLE":
        for anchor_source_id in sequence(
            anchor_capability.get("anchor_source_ids"), "anchor source IDs"
        ):
            pu_started = time.perf_counter()
            _progress(
                f"P11 anchor_pu_start fold={args.outer_fold} source={anchor_source_id}"
            )
            pu_fit = fit_anchor_pu_fold(
                feature_panel=cast(pd.DataFrame, panel),
                feature_registry=effective_feature_registry,
                label_inputs=cast(pd.DataFrame, inputs),
                weights=cast(pd.DataFrame, weights),
                anchor_source_id=str(anchor_source_id),
                outer_year=int(args.outer_fold),
                settings=settings,
                columns=columns,
                random_state=derive_seed(
                    loaded.protocol_hash,
                    "P11",
                    coordinates,
                    f"anchor_pu:{anchor_source_id}",
                )
                % (2**32 - 1),
                evaluation_target_id=primary,
            )
            pu_results.append(pu_fit)
            _progress(
                f"P11 anchor_pu_end fold={args.outer_fold} source={anchor_source_id} "
                f"status={pu_fit.models['status']} "
                f"elapsed_seconds={time.perf_counter() - pu_started:.3f}"
            )
    fits.extend(pu_results)
    result = _combine_fits(fits, required_fit=track_l1, track_statuses=track_statuses)
    if (
        result.models["status"] != "PASS"
        or result.oof_predictions.empty
        or result.outer_predictions.empty
    ):
        raise RuntimeError(
            "P11_REQUIRED_PRIMARY_TRACK_INCOMPLETE: optional tracks may skip, primary L1 may not"
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
    fitted_model_rows = [
        mapping(item, "fitted model")
        for item in sequence(result.models.get("models"), "fitted models")
    ]
    receipt: dict[str, object] = {
        "status": "PASS",
        "reason_code": None,
        "required_primary_track": str(primary_track["track_id"]),
        "required_primary_status": track_l1.models["status"],
        "track_statuses": track_statuses,
        "optional_track_failure_policy": "skip_track_not_pipeline",
        "anchor_pu_statuses": [fit.models["status"] for fit in pu_results]
        if pu_results
        else ["UNAVAILABLE_BY_DESIGN"],
        "selected_measurement": primary,
        "tuning_valid_configuration_counts": {
            str(item["model_id"]): item.get("valid_configuration_count")
            for item in fitted_model_rows
            if item.get("training_mechanism") != "bagging_pu"
        },
        "tuning_runtime_seconds": sum(
            float(item.get("tuning_runtime_seconds", 0.0)) for item in fitted_model_rows
        ),
        "tuning_budget_maximum": maximum_configurations,
        "model_feature_contract_summary": feature_contract.summary,
        "runtime_feature_exclusions": feature_contract.exclusions,
        "model_matrix_manifest": feature_contract.rows,
        "protocol_hash": loaded.protocol_hash,
        "split_registry_hash": stable_hash(splits),
        "measurement_selection_hash": stable_hash(selection),
        "channel_measurement_selection_hash": stable_hash(channel_selection),
        "nested_selection_receipt_hash": stable_hash(nested_receipt),
        "nested_selection_status": nested_receipt.get("status"),
        "optional_measurement_tracks": optional_measurements,
        "feature_registry_hash": stable_hash(feature_registry),
        "effective_feature_registry_hash": stable_hash(effective_feature_registry),
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
        f"tracks={track_statuses} models={len(fitted_model_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
