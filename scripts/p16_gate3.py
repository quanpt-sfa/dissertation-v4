"""P16 CLI: locked threshold/interaction tests and Gate 3 verdict."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, outer_fold_ids, physical_columns, sequence
from core.semantic_keys import OUTER_FOLD
from gates.service import gate3_verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P16", state="KNOWN_CASES_OPEN"
    )
    known = sequence(loaded.context.read("known_case_results", {}), "known case results")
    gate2 = mapping(loaded.context.read("gate2_verdict", {}), "Gate 2 receipt")
    panel = loaded.context.read("feature_panel", {})
    feature_registry = sequence(loaded.context.read("feature_registry", {}), "feature registry")
    outcomes = loaded.context.read("sealed_outcome_store", {})
    predictions: list[pd.DataFrame] = []
    for fold_id in outer_fold_ids(loaded.registry):
        loaded.context.read("evaluation_metrics", {OUTER_FOLD: fold_id})
        value = loaded.context.read("raw_outer_predictions", {OUTER_FOLD: fold_id})
        if isinstance(value, pd.DataFrame):
            predictions.append(value)
    if not isinstance(panel, pd.DataFrame) or not isinstance(outcomes, pd.DataFrame):
        raise ValueError("P16 panel inputs must be DataFrames")
    registered_ids = {
        str(mapping(item, "feature registry item")["feature_id"]) for item in feature_registry
    }
    inference = mapping(loaded.registry.get("inference"), "inference")
    library = mapping(inference.get("interaction_library"), "inference.interaction_library")
    bindings = mapping(library.get("operational_bindings"), "interaction operational_bindings")
    for name in ("pressure_feature_id", "monitoring_feature_id", "domain_feature_id"):
        if bindings.get(name) is not None and str(bindings[name]) not in registered_ids:
            raise ValueError(f"Gate 3 binding {name} is not a registered feature")
    folds = mapping(loaded.registry.get("folds"), "folds")
    confirmatory = [
        int(value)
        for value in sequence(folds.get("fully_nested_outer_years"), "fully_nested_outer_years")
    ]
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    threshold, receipt = gate3_verdict(
        gate2=gate2,
        known_case_results=[mapping(item, "known case result") for item in known],
        feature_panel=panel,
        predictions=pd.concat(predictions, ignore_index=True),
        outcomes=outcomes,
        bindings=bindings,
        gate=mapping(evaluation.get("gate3"), "evaluation.gate3"),
        confirmatory_folds=confirmatory,
        columns=physical_columns(loaded.registry),
    )
    receipt["protocol_hash"] = loaded.protocol_hash
    loaded.context.write("threshold_interaction_results", threshold, {})
    loaded.context.write("gate3_verdict", receipt, {})
    print(f"P16 status=PASS verdict={receipt['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
