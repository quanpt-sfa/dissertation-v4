"""P13 CLI: transfer, identification status, source sensitivity, and ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.fold_control import require_primary_target
from core.pipeline import load_run, mapping, outer_fold_ids, physical_columns, sequence
from core.rng import derive_seed
from core.semantic_keys import OUTCOME, OUTER_FOLD, TARGET_ID
from identification.service import build_identification_summary
from sensitivity.service import (
    ablation_summary,
    censoring_sensitivity_summary,
    domain_transfer,
    hierarchical_pi_status,
    source_exclusion_refits,
    source_sensitivity_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P13", state="EVALUATED"
    )
    fold_ids = outer_fold_ids(loaded.registry, include_initial=False)
    columns = physical_columns(loaded.registry)
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    primary_target_id = require_primary_target(measurement, "P13")
    evaluations: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold_id in fold_ids:
        metric = mapping(
            loaded.context.read("evaluation_metrics", {OUTER_FOLD: fold_id}),
            "evaluation metrics",
        )
        prediction = loaded.context.read("raw_outer_predictions", {OUTER_FOLD: fold_id})
        evaluations.append(metric)
        if isinstance(prediction, pd.DataFrame):
            predictions.append(prediction)
    outcomes = loaded.context.read("sealed_outcome_store", {})
    panel = loaded.context.read("feature_panel", {})
    registry = sequence(loaded.context.read("feature_registry", {}), "feature registry")
    capability = mapping(loaded.context.read("l3_pilot_capability", {}), "L3 capability")
    matrices = mapping(
        loaded.context.read("source_channel_matrices", {}), "source-channel matrices"
    )
    evidence_ledger = loaded.context.read("evidence_ledger", {})
    lag = mapping(loaded.context.read("lag_decomposition", {}), "lag decomposition")
    censoring = [
        mapping(item, "censoring registry item")
        for item in sequence(loaded.context.read("censoring_registry", {}), "censoring registry")
    ]
    weight_diagnostics = [
        mapping(
            loaded.context.read("weight_diagnostics", {"fold_id": fold_id}),
            "weight diagnostics",
        )
        for fold_id in fold_ids
    ]
    weights_by_fold = {
        fold_id: loaded.context.read("fold_aware_weights", {"fold_id": fold_id})
        for fold_id in fold_ids
    }
    if not all(isinstance(value, pd.DataFrame) for value in weights_by_fold.values()):
        raise ValueError("P13 fold weights must be DataFrames")
    if not all(isinstance(value, pd.DataFrame) for value in (outcomes, panel, evidence_ledger)):
        raise ValueError("P13 panel inputs must be DataFrames")
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    common = mapping(evaluation.get("gate_common"), "evaluation.gate_common")
    domain = domain_transfer(
        predictions=pd.concat(predictions, ignore_index=True),
        outcomes=cast(pd.DataFrame, outcomes),
        feature_panel=cast(pd.DataFrame, panel),
        feature_registry=[mapping(item, "feature registry item") for item in registry],
        noninferiority_margin=float(common["noninferiority_relative_ap_margin"]),
        support_fraction_minimum=float(common["support_fraction_min"]),
        evaluation_target_id=primary_target_id,
        columns=columns,
    )
    loaded.context.write("domain_transfer_outputs", domain, {})
    source_summary = source_sensitivity_summary(
        cast(pd.DataFrame, evidence_ledger),
        lag,
        columns,
    )
    learners = mapping(loaded.registry.get("learners"), "learners")
    learner_ids = [
        str(value) for value in sequence(learners.get("confirmatory"), "learners.confirmatory")
    ]
    tuning = mapping(learners.get("tuning"), "learners.tuning")
    search_spaces = mapping(tuning.get("search_spaces"), "learners.tuning.search_spaces")
    missing_search_spaces = sorted(set(learner_ids) - set(search_spaces))
    if missing_search_spaces:
        raise RuntimeError(
            "P13 source sensitivity requires the locked P11 tuning search spaces for "
            f"{missing_search_spaces}"
        )
    source_ids = sequence(source_summary.get("source_ids"), "source sensitivity source_ids")
    channel_ids = sequence(source_summary.get("channel_ids"), "source sensitivity channel_ids")
    exclusion_ids = [
        *[f"source:{value}" for value in source_ids],
        *[f"channel:{value}" for value in channel_ids],
    ]
    source_refits = source_exclusion_refits(
        matrices=matrices,
        feature_panel=cast(pd.DataFrame, panel),
        feature_registry=[mapping(item, "feature registry item") for item in registry],
        weights_by_fold={key: cast(pd.DataFrame, value) for key, value in weights_by_fold.items()},
        outcomes=cast(pd.DataFrame, outcomes),
        outer_folds=fold_ids,
        learner_ids=learner_ids,
        learner_settings=mapping(learners.get("settings"), "learners.settings"),
        learner_search_spaces=search_spaces,
        maximum_valid_configurations=int(tuning["max_valid_configurations_per_learner_inner_fold"]),
        evaluation_target_id=primary_target_id,
        columns=columns,
        seed_by_fold_and_exclusion={
            (fold_id, exclusion_id): derive_seed(
                loaded.protocol_hash,
                "P13",
                {OUTER_FOLD: fold_id},
                f"source_exclusion:{exclusion_id}",
            )
            % (2**32 - 1)
            for fold_id in fold_ids
            for exclusion_id in exclusion_ids
        },
    )
    source_refits["registered_summary"] = source_summary

    target_column = columns[TARGET_ID]
    outcome_column = columns[OUTCOME]
    observed_positive_rate: float | None = None
    outcome_frame = cast(pd.DataFrame, outcomes)
    if target_column in outcome_frame.columns and outcome_column in outcome_frame.columns:
        primary_rows = outcome_frame.loc[
            outcome_frame[target_column].astype(str) == primary_target_id,
            outcome_column,
        ].dropna()
        if len(primary_rows):
            observed_positive_rate = float(primary_rows.astype(bool).mean())

    identification_summary = build_identification_summary(
        measurement,
        observed_positive_rate=observed_positive_rate,
    )
    source_refits["identification_robustness"] = identification_summary
    loaded.context.write("source_sensitivity_outputs", source_refits, {})
    loaded.context.write(
        "censoring_sensitivity_outputs",
        censoring_sensitivity_summary(weight_diagnostics, censoring),
        {},
    )
    loaded.context.write("hierarchical_pi_sensitivity", hierarchical_pi_status(capability), {})
    loaded.context.write("ablation_results", ablation_summary(evaluations), {})
    print(
        "P13 status=PASS "
        f"domain_status={domain['status']} "
        f"identification_status={identification_summary['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
