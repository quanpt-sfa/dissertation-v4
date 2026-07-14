"""P05 CLI: construct L0/L1 inputs, source matrices, and L3 capability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import CHANNEL_ID
from measurement.service import build_measurement_inputs, summarize_fold_eligibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P05",
        state="RISK_SET",
    )
    if args.dry_run:
        print("P05 dry-run: L0/L1 and missingness-preserving L2 inputs")
        return 0
    risk_sets = loaded.context.read("risk_sets", {})
    evidence = loaded.context.read("evidence_ledger", {})
    if not isinstance(risk_sets, pd.DataFrame) or not isinstance(evidence, pd.DataFrame):
        raise ValueError("P05 input artifacts must be DataFrames")
    expected_sources, anchor_sources = _evidence_sources(loaded.registry)
    study = mapping(loaded.registry.get("study"), "study")
    horizons = mapping(study.get("horizons_months"), "study.horizons_months")
    horizon = horizons.get("primary")
    if not isinstance(horizon, int):
        raise ValueError("study.horizons_months.primary must be an integer")
    vocabulary = mapping(loaded.registry.get("vocabulary"), "vocabulary")
    statuses = sequence(vocabulary.get("capability_statuses"), "capability_statuses")
    if "EMPIRICALLY_PENDING" not in statuses or "UNAVAILABLE_BY_DESIGN" not in statuses:
        raise ValueError("required capability statuses are not registered")
    reasons = sequence(vocabulary.get("reason_codes"), "reason_codes")
    if "INSUFFICIENT_CHANNELS" not in reasons:
        raise ValueError("INSUFFICIENT_CHANNELS reason is not registered")
    result = build_measurement_inputs(
        risk_sets=risk_sets,
        evidence=evidence,
        expected_sources=expected_sources,
        horizon_months=horizon,
        columns=physical_columns(loaded.registry),
        pending_status="EMPIRICALLY_PENDING",
        unavailable_status="UNAVAILABLE_BY_DESIGN",
        insufficient_channels_reason="INSUFFICIENT_CHANNELS",
        anchor_source_ids=anchor_sources,
    )
    if args.validate_only:
        return 0
    loaded.context.write("source_channel_matrices", result.matrices, {})
    loaded.context.write("l0_l1_inputs", result.inputs, {})
    loaded.context.write("l3_pilot_capability", result.l3_capability, {})
    loaded.context.write("measurement_variable_registry", result.measurement_variables, {})
    loaded.context.write("channel_capability", result.channel_capability, {})
    loaded.context.write("anchor_capability", result.anchor_capability, {})
    folds = mapping(loaded.registry.get("folds"), "folds")
    nested = [
        int(value)
        for value in sequence(
            folds.get("fully_nested_outer_years"), "folds.fully_nested_outer_years"
        )
    ]
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")
    sensitivity_range = sequence(
        gate2.get("sensitivity_positive_count_range"),
        "evaluation.gate2.sensitivity_positive_count_range",
    )
    if len(sensitivity_range) != 2:
        raise ValueError("sensitivity positive-count range requires two values")
    fold_eligibility = summarize_fold_eligibility(
        sealed_outcomes=result.sealed_outcomes,
        initial_outer_year=int(folds["initial_outer_year"]),
        confirmatory_years=nested,
        prospective_year=int(folds["prospective_year"]),
        confirmatory_positive_minimum=int(gate2["confirmatory_positive_count_min"]),
        sensitivity_positive_range=(
            int(sensitivity_range[0]),
            int(sensitivity_range[1]),
        ),
        columns=physical_columns(loaded.registry),
    )
    loaded.context.write("fold_eligibility", fold_eligibility, {})
    loaded.context.write("sealed_outcome_store", result.sealed_outcomes, {})
    print(
        f"P05 status=PASS targets={len(result.inputs)} "
        f"sealed_outcomes={len(result.sealed_outcomes)}"
    )
    return 0


def _evidence_sources(registry: dict[str, object]) -> tuple[dict[str, str], list[str]]:
    data_sources = mapping(registry.get("data_sources"), "data_sources")
    source_registry = mapping(data_sources.get("source_registry"), "source_registry")
    sources = mapping(source_registry.get("sources"), "source_registry.sources")
    result: dict[str, str] = {}
    anchors: list[str] = []
    for source_id, raw in sources.items():
        source = mapping(raw, f"source={source_id}")
        if source.get("role") == "evidence" and source.get("enabled") is True:
            channel = source.get(CHANNEL_ID)
            if not isinstance(channel, str):
                raise ValueError(f"source={source_id}: channel_id required")
            result[source_id] = channel
            if source.get("verification_status") == "high_confirmation":
                anchors.append(source_id)
    if not result:
        raise ValueError("P05 requires registered evidence sources")
    return result, anchors


if __name__ == "__main__":
    raise SystemExit(main())
