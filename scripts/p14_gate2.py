"""P14 CLI: evaluate the locked Gate 2 criteria over confirmatory outer folds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from core.semantic_keys import OUTER_FOLD
from gates.evidence_contract import insufficient_gate2_verdict, prepare_gate2_inputs
from gates.service import gate2_verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry, run_id=args.run_id, step_id="P14", state="SENSITIVITY"
    )
    folds = mapping(loaded.registry.get("folds"), "folds")
    confirmatory = [
        str(value)
        for value in sequence(
            folds.get("fully_nested_outer_years"), "folds.fully_nested_outer_years"
        )
    ]
    evaluations: list[dict[str, Any]] = []
    bootstraps: list[list[dict[str, Any]]] = []
    for fold_id in confirmatory:
        evaluation = mapping(
            loaded.context.read("evaluation_metrics", {OUTER_FOLD: fold_id}),
            "evaluation metrics",
        )
        bootstrap = sequence(
            loaded.context.read("bootstrap_batches", {OUTER_FOLD: fold_id}),
            "bootstrap batches",
        )
        evaluations.append(evaluation)
        bootstraps.append(
            [{OUTER_FOLD: fold_id, **mapping(item, "bootstrap item")} for item in bootstrap]
        )
    domain = mapping(loaded.context.read("domain_transfer_outputs", {}), "domain transfer")
    loaded.context.read("source_sensitivity_outputs", {})
    loaded.context.read("censoring_sensitivity_outputs", {})
    loaded.context.read("ablation_results", {})
    evaluation_config = mapping(loaded.registry.get("evaluation"), "evaluation")
    gate_config = mapping(evaluation_config.get("gate2"), "evaluation.gate2")
    prepared = prepare_gate2_inputs(
        evaluations=evaluations,
        bootstraps=bootstraps,
        gate=gate_config,
        confirmatory_folds=confirmatory,
    )
    blockers = [str(value) for value in cast(list[object], prepared["blockers"])]
    candidate_ids = [str(value) for value in cast(list[object], prepared["candidate_ids"])]
    if blockers:
        result = insufficient_gate2_verdict(blockers, candidate_ids)
    else:
        result = gate2_verdict(
            evaluations=cast(list[dict[str, Any]], prepared["evaluations"]),
            bootstraps=cast(list[list[dict[str, Any]]], prepared["bootstraps"]),
            domain_transfer=domain,
            gate=gate_config,
            common=mapping(evaluation_config.get("gate_common"), "evaluation.gate_common"),
            confirmatory_folds=confirmatory,
        )
        result["locked_candidate_ids"] = candidate_ids
        result["evidence_blockers"] = []
    result["protocol_hash"] = loaded.protocol_hash
    loaded.context.write("gate2_verdict", result, {})
    print(f"P14 status={result['status']} verdict={result['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
