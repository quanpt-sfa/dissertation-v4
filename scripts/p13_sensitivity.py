"""P13 CLI: domain transfer, hierarchical-pi status, and block ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, outer_fold_ids, physical_columns, sequence
from core.semantic_keys import OUTER_FOLD
from sensitivity.service import (
    ablation_summary,
    censoring_sensitivity_summary,
    domain_transfer,
    hierarchical_pi_status,
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
    evaluations: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold_id in outer_fold_ids(loaded.registry):
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
        for fold_id in outer_fold_ids(loaded.registry)
    ]
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
        columns=physical_columns(loaded.registry),
    )
    loaded.context.write("domain_transfer_outputs", domain, {})
    loaded.context.write(
        "source_sensitivity_outputs",
        source_sensitivity_summary(
            cast(pd.DataFrame, evidence_ledger),
            lag,
            physical_columns(loaded.registry),
        ),
        {},
    )
    loaded.context.write(
        "censoring_sensitivity_outputs",
        censoring_sensitivity_summary(weight_diagnostics, censoring),
        {},
    )
    loaded.context.write("hierarchical_pi_sensitivity", hierarchical_pi_status(capability), {})
    loaded.context.write("ablation_results", ablation_summary(evaluations), {})
    print(f"P13 status=PASS domain_status={domain['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
