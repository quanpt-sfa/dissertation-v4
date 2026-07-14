"""P12 CLI: open one frozen outer fold, calibrate, and evaluate it once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.rng import generator
from core.semantic_keys import OUTER_FOLD
from evaluation.service import evaluate_outer_fold


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
        step_id="P12",
        state="FROZEN",
        coordinates=coordinates,
    )
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
    open_receipt = {
        "status": "PASS",
        "protocol_hash": loaded.protocol_hash,
        "model_freeze_receipt_hash": loaded.context.store.receipt_hash(
            "model_freeze_receipt", coordinates
        ),
        "mcse_report_hash": loaded.context.store.receipt_hash("mcse_report", {}),
        "opened_at_state": "FROZEN",
        OUTER_FOLD: args.outer_fold,
    }
    loaded.context.write("outer_open_receipt", open_receipt, coordinates)
    oof = loaded.context.read("development_oof_predictions", coordinates)
    predictions = loaded.context.read("raw_outer_predictions", coordinates)
    outcomes = loaded.context.read("sealed_outcome_store", {})
    if not all(isinstance(value, pd.DataFrame) for value in (oof, predictions, outcomes)):
        raise ValueError("P12 prediction and outcome inputs must be DataFrames")
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    review_budget = mapping(evaluation.get("review_budget"), "evaluation.review_budget")
    utility = mapping(loaded.registry.get("utility"), "utility")
    scenarios = [
        mapping(item, "utility scenario")
        for item in sequence(utility.get("operational_scenarios"), "utility.operational_scenarios")
    ]
    inference = mapping(loaded.registry.get("inference"), "inference")
    bootstrap = mapping(inference.get("bootstrap"), "inference.bootstrap")
    result = evaluate_outer_fold(
        oof_predictions=cast(pd.DataFrame, oof),
        outer_predictions=cast(pd.DataFrame, predictions),
        outcomes=cast(pd.DataFrame, outcomes),
        outer_year=int(args.outer_fold),
        review_fraction=float(review_budget["primary_fraction"]),
        utility_scenarios=scenarios,
        bootstrap_replications=int(bootstrap["replications"]),
        confidence_level=float(bootstrap["confidence_level"]),
        columns=physical_columns(loaded.registry),
        rng=generator(loaded.protocol_hash, "P12", coordinates, "firm_bootstrap"),
        oof_training_targets=[
            mapping(item, "OOF training target")
            for item in sequence(models.get("oof_training_targets"), "OOF training targets")
        ],
    )
    loaded.context.write("calibration_outputs", result.calibration, coordinates)
    loaded.context.write("evaluation_metrics", result.metrics, coordinates)
    loaded.context.write("bootstrap_batches", result.bootstrap, coordinates)
    loaded.context.write("utility_scenarios", result.utility, coordinates)
    print(
        f"P12 status=PASS fold={args.outer_fold} "
        f"models={len(sequence(result.metrics['models'], 'evaluation models'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
