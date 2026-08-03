"""P12 CLI: open one frozen outer fold, calibrate, and evaluate it once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.fold_control import require_primary_target
from core.pipeline import load_run, mapping, physical_columns, sequence, stable_hash
from core.resume import verify_p12_implementation
from core.rng import generator
from core.semantic_keys import OUTER_FOLD
from evaluation.pairing import validate_paired_prediction_keys
from evaluation.service import build_latent_risk_scenarios, evaluate_outer_fold
from selection.nested_refit import validate_nested_refit_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outer-fold", required=True)
    args = parser.parse_args()
    coordinates = {OUTER_FOLD: args.outer_fold}
    project_root = Path(__file__).resolve().parents[1]
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P12",
        state="FROZEN",
        coordinates=coordinates,
    )
    compatibility = verify_p12_implementation(loaded.run_root, project_root)
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    primary_target_id = require_primary_target(measurement, "P12")
    mcse = mapping(loaded.context.read("mcse_report", {}), "MCSE report")
    if mcse.get("status") != "PASS" or mcse.get("precision_target_met") is not True:
        raise RuntimeError(
            "P12 outer firewall closed: P08 simulation must PASS every locked replication and MCSE rule"
        )
    freeze = mapping(loaded.context.read("model_freeze_receipt", coordinates), "freeze receipt")
    models = mapping(loaded.context.read("model_artifacts", coordinates), "model artifacts")
    if freeze.get("status") != "PASS":
        raise RuntimeError(f"fold={args.outer_fold}: a PASS model-freeze receipt is required")
    if freeze.get("protocol_hash") != loaded.protocol_hash:
        raise RuntimeError("model-freeze receipt protocol hash mismatch")
    measurement_selection_hash = freeze.get("measurement_selection_hash")
    if not isinstance(measurement_selection_hash, str) or not measurement_selection_hash:
        raise RuntimeError("model-freeze receipt measurement selection hash is missing")
    channel_selection = mapping(
        loaded.context.read("channel_measurement_selection", coordinates),
        "channel measurement selection",
    )
    optional_tracks_raw = freeze.get("optional_measurement_tracks", [])
    optional_tracks = [
        str(value) for value in sequence(optional_tracks_raw, "freeze optional measurement tracks")
    ]
    nested_receipt = validate_nested_refit_receipt(
        channel_selection,
        outer_fold=args.outer_fold,
        required_optional_measurements=optional_tracks,
    )
    if freeze.get("channel_measurement_selection_hash") != stable_hash(channel_selection):
        raise RuntimeError("channel measurement selection hash mismatch")
    if freeze.get("nested_selection_receipt_hash") != stable_hash(nested_receipt):
        raise RuntimeError("nested selection receipt hash mismatch")

    oof = loaded.context.read("development_oof_predictions", coordinates)
    predictions = loaded.context.read("raw_outer_predictions", coordinates)
    if not isinstance(oof, pd.DataFrame) or not isinstance(predictions, pd.DataFrame):
        raise ValueError("P12 prediction inputs must be DataFrames")
    columns = physical_columns(loaded.registry)
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    review_budget = mapping(evaluation.get("review_budget"), "evaluation.review_budget")
    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")
    reference_by_candidate = {
        str(key): str(value)
        for key, value in mapping(
            gate2.get("reference_by_candidate"), "evaluation.gate2.reference_by_candidate"
        ).items()
    }
    pairing_audit = validate_paired_prediction_keys(
        predictions,
        columns,
        reference_by_candidate,
    )

    open_receipt: dict[str, object] = {
        "status": "PASS",
        "protocol_hash": loaded.protocol_hash,
        "model_freeze_receipt_hash": loaded.context.store.receipt_hash(
            "model_freeze_receipt", coordinates
        ),
        "mcse_report_hash": loaded.context.store.receipt_hash("mcse_report", {}),
        "measurement_selection_hash": measurement_selection_hash,
        "channel_measurement_selection_hash": stable_hash(channel_selection),
        "nested_selection_receipt_hash": stable_hash(nested_receipt),
        "pairing_scope": pairing_audit["pair_scope"],
        "pair_count": pairing_audit["pair_count"],
        "opened_at_state": "FROZEN",
        OUTER_FOLD: args.outer_fold,
    }
    if compatibility is not None:
        open_receipt.update(
            {
                "compatibility_patch_id": compatibility["patch_id"],
                "compatibility_receipt_hash": compatibility["receipt_hash"],
                "evaluation_git_commit": compatibility["current_git_commit"],
                "locked_git_commit": compatibility["locked_git_commit"],
            }
        )
    loaded.context.write("outer_open_receipt", open_receipt, coordinates)

    outcomes = loaded.context.read("sealed_outcome_store", {})
    if not isinstance(outcomes, pd.DataFrame):
        raise ValueError("P12 outcome input must be a DataFrame")
    calibration = mapping(loaded.registry.get("calibration"), "calibration")
    platt = mapping(calibration.get("platt"), "calibration.platt")
    utility = mapping(loaded.registry.get("utility"), "utility")
    scenarios = [
        mapping(item, "utility scenario")
        for item in sequence(utility.get("operational_scenarios"), "utility.operational_scenarios")
    ]
    source_matrices = mapping(
        loaded.context.read("source_channel_matrices", {}), "source-channel matrices"
    )
    latent_risk_scenarios = build_latent_risk_scenarios(
        matrices=source_matrices,
        channel_selection=channel_selection,
        outer_year=int(args.outer_fold),
        scenarios=scenarios,
    )
    inference = mapping(loaded.registry.get("inference"), "inference")
    s3_taxonomy = mapping(loaded.registry.get("s3_taxonomy"), "s3_taxonomy")
    bootstrap = mapping(inference.get("bootstrap"), "inference.bootstrap")
    result = evaluate_outer_fold(
        oof_predictions=oof,
        outer_predictions=predictions,
        outcomes=outcomes,
        outer_year=int(args.outer_fold),
        review_fraction=float(review_budget["primary_fraction"]),
        utility_scenarios=scenarios,
        bootstrap_replications=int(bootstrap["replications"]),
        confidence_level=float(bootstrap["confidence_level"]),
        columns=columns,
        rng=generator(loaded.protocol_hash, "P12", coordinates, "firm_bootstrap"),
        oof_training_targets=[
            mapping(item, "OOF training target")
            for item in sequence(models.get("oof_training_targets"), "OOF training targets")
        ],
        latent_risk_scenarios=latent_risk_scenarios,
        required_reference_by_candidate=reference_by_candidate,
        platt_inverse_regularization=float(platt["inverse_regularization"]),
        platt_maximum_iterations=int(platt["maximum_iterations"]),
        target_id=primary_target_id,
        temporal_estimand_metadata=(
            mapping(s3_taxonomy.get("temporal_estimand"), "s3 temporal estimand")
            if primary_target_id.startswith("S3_")
            else {
                "temporal_estimand_id": "target_specific_composite_or_annual",
                "horizon_applicable": primary_target_id not in {"L1_ANNUAL"},
            }
        ),
    )
    result.metrics["pairing_audit"] = pairing_audit
    loaded.context.write("calibration_outputs", result.calibration, coordinates)
    loaded.context.write("evaluation_metrics", result.metrics, coordinates)
    loaded.context.write("bootstrap_batches", result.bootstrap, coordinates)
    loaded.context.write("utility_scenarios", result.utility, coordinates)
    print(
        f"P12 status=PASS fold={args.outer_fold} "
        f"models={len(sequence(result.metrics['models'], 'evaluation models'))} "
        f"pairs={pairing_audit['pair_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
