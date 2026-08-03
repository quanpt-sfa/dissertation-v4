"""P16 CLI: locked threshold/interaction tests and Gate 3 verdict."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.fold_control import require_primary_target
from core.pipeline import load_run, mapping, outer_fold_ids, physical_columns, sequence
from core.rng import generator
from core.semantic_keys import OUTER_FOLD
from gates.service import gate3_verdict
from sensitivity.service import select_observed_target_outcomes


def _default_workers() -> int:
    raw = os.environ.get("P16_WORKERS")
    if raw is None:
        return min(10, max(1, os.cpu_count() or 1))
    try:
        workers = int(raw)
    except ValueError as exc:
        raise ValueError("P16_WORKERS must be an integer") from exc
    if workers < 1:
        raise ValueError("P16_WORKERS must be positive")
    return workers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=_default_workers())
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P16", state="KNOWN_CASES_OPEN"
    )
    columns = physical_columns(loaded.registry)
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    primary_target_id = require_primary_target(measurement, "P16")
    known = sequence(loaded.context.read("known_case_results", {}), "known case results")
    gate2 = mapping(loaded.context.read("gate2_verdict", {}), "Gate 2 receipt")
    panel = loaded.context.read("feature_panel", {})
    feature_registry = sequence(loaded.context.read("feature_registry", {}), "feature registry")
    outcomes = loaded.context.read("sealed_outcome_store", {})
    predictions: list[pd.DataFrame] = []
    for fold_id in outer_fold_ids(loaded.registry, include_initial=False):
        loaded.context.read("evaluation_metrics", {OUTER_FOLD: fold_id})
        value = loaded.context.read("raw_outer_predictions", {OUTER_FOLD: fold_id})
        if isinstance(value, pd.DataFrame):
            predictions.append(value)
    if not isinstance(panel, pd.DataFrame) or not isinstance(outcomes, pd.DataFrame):
        raise ValueError("P16 panel inputs must be DataFrames")
    observed_outcomes = select_observed_target_outcomes(
        outcomes,
        target_id=primary_target_id,
        columns=columns,
        context="P16 Gate 3",
    )
    registered_ids = {
        str(mapping(item, "feature registry item")["feature_id"]) for item in feature_registry
    }
    inference = mapping(loaded.registry.get("inference"), "inference")
    library = mapping(inference.get("interaction_library"), "inference.interaction_library")
    bindings = mapping(library.get("operational_bindings"), "interaction operational_bindings")
    for name in ("pressure_feature_id", "monitoring_feature_id", "domain_feature_id"):
        if bindings.get(name) is not None and str(bindings[name]) not in registered_ids:
            raise ValueError(f"Gate 3 binding {name} is not a registered feature")
    raw_threshold_features = bindings.get("threshold_feature_ids")
    if isinstance(raw_threshold_features, list):
        missing_thresholds = sorted(
            str(value)
            for value in cast(list[object], raw_threshold_features)
            if str(value) not in registered_ids
        )
        if missing_thresholds:
            raise ValueError(f"Gate 3 threshold bindings are not registered: {missing_thresholds}")
    folds = mapping(loaded.registry.get("folds"), "folds")
    confirmatory = [
        int(value)
        for value in sequence(folds.get("fully_nested_outer_years"), "fully_nested_outer_years")
    ]
    measurement_selections = [
        mapping(
            loaded.context.read("measurement_selection_registry", {OUTER_FOLD: str(fold_id)}),
            "measurement selection",
        )
        for fold_id in confirmatory
    ]
    selection_config = mapping(measurement.get("selection"), "measurement.selection")
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    bootstrap = mapping(inference.get("bootstrap"), "inference.bootstrap")
    common = mapping(evaluation.get("gate_common"), "evaluation.gate_common")
    print(
        f"P16 Gate 3 bootstrap replications={int(bootstrap['replications'])} "
        f"workers={args.workers}",
        flush=True,
    )
    threshold, receipt = gate3_verdict(
        gate2=gate2,
        known_case_results=[mapping(item, "known case result") for item in known],
        feature_panel=panel,
        predictions=pd.concat(predictions, ignore_index=True),
        outcomes=observed_outcomes,
        bindings=bindings,
        gate=mapping(evaluation.get("gate3"), "evaluation.gate3"),
        confirmatory_folds=confirmatory,
        columns=columns,
        bootstrap_replications=int(bootstrap["replications"]),
        familywise_alpha=float(common["familywise_alpha"]),
        confirmatory_threshold_claims=int(library["confirmatory_threshold_claims"]),
        measurement_selections=measurement_selections,
        measurement_stability_minimum=int(selection_config["stability_minimum_selected_folds"]),
        measurement_stability_denominator=int(selection_config["stability_denominator_folds"]),
        rng=generator(loaded.protocol_hash, "P16", {}, "gate3_family_bootstrap"),
        workers=args.workers,
    )
    receipt["protocol_hash"] = loaded.protocol_hash
    receipt["workers"] = args.workers
    threshold["workers"] = args.workers
    loaded.context.write("threshold_interaction_results", threshold, {})
    loaded.context.write("gate3_verdict", receipt, {})
    print(f"P16 status={receipt['status']} verdict={receipt['verdict']} workers={args.workers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
