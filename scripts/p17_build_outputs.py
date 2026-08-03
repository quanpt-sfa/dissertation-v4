"""P17 CLI: render the final ledger and report without refitting anything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, outer_fold_ids, read_optional, sequence
from core.semantic_keys import OUTER_FOLD
from reporting.service import (
    build_chapter4_tables,
    build_decision_log,
    build_final_ledger,
    build_verdict_matrix,
    render_audit_report,
    render_final_report,
    render_gate_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    loaded = load_run(registry_path=args.registry, run_id=args.run_id, step_id="P17", state="GATE3")
    gate3 = mapping(loaded.context.read("gate3_verdict", {}), "Gate 3 receipt")
    gate2_raw = read_optional(loaded.context, "gate2_verdict", {})
    gate2 = mapping(gate2_raw, "Gate 2 receipt") if gate2_raw is not None else None
    evaluations: list[dict[str, Any]] = []
    for fold_id in outer_fold_ids(loaded.registry, include_initial=False):
        value = read_optional(loaded.context, "evaluation_metrics", {OUTER_FOLD: fold_id})
        if value is not None:
            evaluations.append(mapping(value, "evaluation metrics"))
        for artifact_id in ("calibration_outputs", "bootstrap_batches", "utility_scenarios"):
            read_optional(loaded.context, artifact_id, {OUTER_FOLD: fold_id})
    domain_raw = read_optional(loaded.context, "domain_transfer_outputs", {})
    read_optional(loaded.context, "source_sensitivity_outputs", {})
    read_optional(loaded.context, "censoring_sensitivity_outputs", {})
    hierarchy_raw = read_optional(loaded.context, "hierarchical_pi_sensitivity", {})
    read_optional(loaded.context, "ablation_results", {})
    known_raw = read_optional(loaded.context, "known_case_results", {})
    threshold_raw = read_optional(loaded.context, "threshold_interaction_results", {})
    domain = mapping(domain_raw, "domain transfer") if domain_raw is not None else None
    hierarchy = mapping(hierarchy_raw, "hierarchical pi") if hierarchy_raw is not None else None
    known = (
        [mapping(item, "known case result") for item in sequence(known_raw, "known cases")]
        if known_raw is not None
        else None
    )
    threshold = mapping(threshold_raw, "threshold results") if threshold_raw is not None else None
    ledger = build_final_ledger(
        gate2=gate2,
        gate3=gate3,
        evaluations=evaluations,
        domain=domain,
        hierarchical_pi=hierarchy,
        known_cases=known,
        threshold=threshold,
    )
    verdict_matrix = build_verdict_matrix(gate2, gate3, known)
    traceability = [
        mapping(item, "decision traceability item")
        for item in sequence(loaded.registry.get("decision_traceability"), "decision traceability")
    ]
    decision_log = build_decision_log(traceability)
    chapter4 = build_chapter4_tables(ledger, verdict_matrix, evaluations)
    figure = render_gate_figure(verdict_matrix)
    upstream_inventory = loaded.context.store.inventory()
    audit = render_audit_report(
        protocol_hash=loaded.protocol_hash,
        ledger=ledger,
        verdict_matrix=verdict_matrix,
        upstream_inventory=upstream_inventory,
    )
    report = render_final_report(ledger, gate2, gate3)
    loaded.context.write("final_result_ledger", ledger, {})
    loaded.context.write("final_verdict_matrix", verdict_matrix, {})
    loaded.context.write("final_decision_log", decision_log, {})
    loaded.context.write("chapter4_input_tables", chapter4, {})
    loaded.context.write("final_gate_figure", figure, {})
    loaded.context.write("final_audit_report", audit, {})
    loaded.context.write("final_report", report, {})
    final_inventory = loaded.context.store.inventory()
    loaded.context.write(
        "final_artifact_manifest",
        {
            "protocol_hash": loaded.protocol_hash,
            "artifacts": final_inventory,
            "self_excluded_to_avoid_recursive_hash": True,
        },
        {},
    )
    print(f"P17 status=PASS ledger_rows={len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
