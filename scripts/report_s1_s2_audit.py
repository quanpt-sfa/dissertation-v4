"""Create reproducible S1/S2 capability tables from verified run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import (
    ELIGIBLE,
    EVIDENCE_CATEGORY,
    EVIDENCE_VALUE,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    OUTCOME_BASIS,
    OUTER_FOLD,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    TEMPORAL_ROLE,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    p05 = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P05",
        state="RISK_SET",
    )
    evidence = p05.context.read("evidence_ledger", {})
    risk_sets = p05.context.read("risk_sets", {})
    if not isinstance(evidence, pd.DataFrame) or not isinstance(risk_sets, pd.DataFrame):
        raise TypeError("verified P03/P04 dataframe artifacts are required")

    p06 = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P06",
        state="MEASURED",
    )
    matrices = mapping(p06.context.read("source_channel_matrices", {}), "source_channel_matrices")
    l3_capability = mapping(p06.context.read("l3_pilot_capability", {}), "l3_pilot_capability")

    p09 = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P09",
        state="SIMULATED",
    )
    folds = [
        mapping(item, "fold eligibility")
        for item in sequence(p09.context.read("fold_eligibility", {}), "fold_eligibility")
    ]
    observability = mapping(
        p09.context.read("observability_registry", {}), "observability_registry"
    )

    # Audit-only reads still pass through ArtifactStore manifest and schema verification.
    annual_audit = mapping(p05.context.store.read("annual_evidence_audit", {}), "annual audit")
    maturity_audit = mapping(p05.context.store.read("maturity_audit", {}), "maturity audit")
    mcse_report = mapping(p05.context.store.read("mcse_report", {}), "MCSE report")
    source_observability = mapping(observability.get("sources"), "observability sources")
    s3_observability = mapping(source_observability.get("S3_sanction_evidence"), "S3 observability")
    annual_availability = [
        mapping(item, "annual measurement availability")
        for item in sequence(
            maturity_audit.get("annual_measurement_availability"),
            "annual_measurement_availability",
        )
    ]

    columns = physical_columns(p05.registry)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    s1_counts = _s1_counts(evidence, columns)
    s2_counts = _s2_counts(evidence, columns)
    l1_counts = _l1_counts(matrices, folds)
    overlap = _channel_overlap(matrices)
    s1_failures = _failure_counts(
        sequence(annual_audit.get("s1_pair_failures"), "s1_pair_failures"),
        fields=[
            SOURCE_ID,
            FISCAL_YEAR,
            "endpoint_id",
            "canonical_item",
            "reason_code",
            SOURCE_OPPORTUNITY,
            "pre_record_count",
            "post_record_count",
        ],
    )
    s2_exceptions = _failure_counts(
        sequence(
            annual_audit.get("s2_normalization_exceptions"),
            "s2_normalization_exceptions",
        ),
        fields=[
            SOURCE_ID,
            FISCAL_YEAR,
            "raw_values",
            "normalized_category",
            "reason_code",
            SOURCE_OPPORTUNITY,
            "record_count",
        ],
    )
    outputs = {
        "s1_counts_by_year_endpoint.csv": s1_counts,
        "s2_opinion_counts_by_year.csv": s2_counts,
        "l1_counts_after_s1_s2.csv": l1_counts,
        "channel_overlap_after_s1_s2.csv": overlap,
        "s1_pair_failures.csv": s1_failures,
        "s2_normalization_exceptions.csv": s2_exceptions,
    }
    for name, frame in outputs.items():
        _write_csv(output_dir / name, frame)

    summary = {
        "run_id": args.run_id,
        "protocol_hash": p05.protocol_hash,
        "evidence_rows": len(evidence),
        "annual_measurement_rows": int(
            evidence[columns[TEMPORAL_ROLE]].eq("annual_measurement_at_anchor").sum()
        ),
        "delayed_event_rows": int(
            evidence[columns[TEMPORAL_ROLE]].eq("delayed_verification").sum()
        ),
        "firm_year_rows": len(risk_sets),
        "mature_rows": int(risk_sets[columns[MATURE]].sum()),
        "prospective_rows": int((~risk_sets[columns[MATURE]]).sum()),
        "s1_opportunity_rows": int(s1_counts["opportunity_observed"].sum()),
        "s1_positive_rows": int(s1_counts["positive"].sum()),
        "s1_explicit_negative_rows": int(s1_counts["explicit_negative"].sum()),
        "s2_positive_rows": int(s2_counts["positive"].sum()),
        "s2_explicit_negative_rows": int(s2_counts["explicit_negative"].sum()),
        "s3_positive_rows": int(str(s3_observability["positive_count"])),
        "l1_positive_rows": int(l1_counts["positive"].sum()),
        "l1_explicit_negative_rows": int(l1_counts["explicit_negative"].sum()),
        "l1_unknown_rows": int(l1_counts["unknown"].sum()),
        "valid_channels": l3_capability.get("valid_channels"),
        "l3_status": l3_capability.get("status"),
        "l3_reason_code": l3_capability.get("reason_code"),
        "simulation_status": mcse_report.get("status"),
        "simulation_reason_code": mcse_report.get("reason_code"),
        "annual_anchor_mismatch_count": sum(
            int(item.get("annual_anchor_mismatch_count", 0)) for item in annual_availability
        ),
    }
    print("S1_S2_AUDIT_JSON=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _s1_counts(evidence: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    source = columns[SOURCE_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    opportunity = columns[SOURCE_OPPORTUNITY]
    basis = columns[OUTCOME_BASIS]
    value = columns[EVIDENCE_VALUE]
    frame = evidence.loc[evidence[source].astype(str).str.startswith("S1_")].copy()
    rows: list[dict[str, object]] = []
    for (source_id, fiscal_year), group in frame.groupby([source, year], sort=True):
        observed = group[opportunity].eq(True)
        observed_outcomes = group[outcome].notna()
        positive = group[outcome].eq(True)
        negative = group[outcome].eq(False)
        outcome_count = int(observed_outcomes.sum())
        rows.append(
            {
                SOURCE_ID: str(source_id),
                "endpoint_id": str(source_id).removeprefix("S1_"),
                FISCAL_YEAR: int(str(fiscal_year)),
                "firm_year_records": len(group),
                "opportunity_observed": int(observed.sum()),
                "opportunity_unmatched_or_missing": int(group[opportunity].eq(False).sum()),
                "opportunity_unknown": int(group[opportunity].isna().sum()),
                "positive": int(positive.sum()),
                "explicit_negative": int(negative.sum()),
                "unknown": int(group[outcome].isna().sum()),
                "invalid_denominator": int(
                    group[basis].astype(str).str.startswith("S1_INVALID_DENOMINATOR").sum()
                ),
                "duplicate_or_conflict": int(
                    group[basis].astype(str).str.startswith("S1_DUPLICATE").sum()
                ),
                "ratio_computed": int(group[value].notna().sum()),
                "observed_outcome_count": outcome_count,
                "prevalence_in_observed_outcomes": (
                    float(positive.sum()) / outcome_count if outcome_count else None
                ),
                "empirical_rule_locked": not bool(
                    group[basis].eq("S1_EMPIRICAL_RULE_NOT_LOCKED").any()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values([FISCAL_YEAR, SOURCE_ID], kind="stable")


def _s2_counts(evidence: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    source = columns[SOURCE_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    category = columns[EVIDENCE_CATEGORY]
    opportunity = columns[SOURCE_OPPORTUNITY]
    basis = columns[OUTCOME_BASIS]
    frame = evidence.loc[evidence[source].eq("S2_audit_opinion")].copy()
    rows: list[dict[str, object]] = []
    for fiscal_year, group in frame.groupby(year, sort=True):
        rows.append(
            {
                FISCAL_YEAR: int(str(fiscal_year)),
                "firm_year_records": len(group),
                "opinion_observed": int(group[opportunity].eq(True).sum()),
                "clean": int(group[category].eq("clean").sum()),
                "qualified": int(group[category].eq("qualified").sum()),
                "disclaimer": int(group[category].eq("disclaimer").sum()),
                "adverse": int(group[category].eq("adverse").sum()),
                "missing": int(
                    group[basis].isin(["S2_OPINION_RECORD_MISSING", "S2_OPINION_MISSING"]).sum()
                ),
                "positive": int(group[outcome].eq(True).sum()),
                "explicit_negative": int(group[outcome].eq(False).sum()),
                "unknown": int(group[outcome].isna().sum()),
                "duplicate_or_conflicting": int(group[basis].eq("S2_CONFLICTING_OPINIONS").sum()),
                "normalization_unmapped_or_pending": int(
                    group[basis]
                    .isin(
                        [
                            "S2_OPINION_UNMAPPED",
                            "S2_CATEGORY_EMPIRICALLY_PENDING",
                            "S2_NORMALIZED_CATEGORY_UNMAPPED",
                        ]
                    )
                    .sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(FISCAL_YEAR, kind="stable")


def _l1_counts(matrices: dict[str, Any], folds: list[dict[str, Any]]) -> pd.DataFrame:
    matrix_rows = [
        mapping(item, "source matrix row")
        for item in sequence(matrices.get("rows"), "source matrix rows")
    ]
    fold_by_year = {int(str(item[OUTER_FOLD])): item for item in folds}
    rows: list[dict[str, object]] = []
    years = sorted({int(item[FISCAL_YEAR]) for item in matrix_rows})
    for fiscal_year in years:
        group = [item for item in matrix_rows if int(item[FISCAL_YEAR]) == fiscal_year]
        values = [item.get("l1") for item in group]
        mature = sum(item.get(MATURE) is True for item in group)
        fold = fold_by_year.get(fiscal_year)
        positive = sum(value is True for value in values)
        negative = sum(value is False for value in values)
        rows.append(
            {
                FISCAL_YEAR: fiscal_year,
                OUTER_FOLD: str(fiscal_year) if fold is not None else "",
                "eligible_rows": sum(item.get(ELIGIBLE) is True for item in group),
                "mature_rows": mature,
                "positive": positive,
                "explicit_negative": negative,
                "unknown": sum(value is None for value in values),
                "observed_class_count": int(positive > 0) + int(negative > 0),
                "fold_role": (
                    str(fold.get("assigned_role")) if fold is not None else "development_history"
                ),
                "reason_code": (
                    str(fold.get("reason_code")) if fold is not None else "NO_OUTER_FOLD"
                ),
            }
        )
    return pd.DataFrame(rows)


def _channel_overlap(matrices: dict[str, Any]) -> pd.DataFrame:
    matrix_rows = [
        mapping(item, "source matrix row")
        for item in sequence(matrices.get("rows"), "source matrix rows")
    ]
    channel_ids = [str(value) for value in sequence(matrices.get("expected_channels"), "channels")]
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in matrix_rows:
        if row.get(ELIGIBLE) is not True:
            continue
        outcome_raw = mapping(row.get("channel_outcomes"), "channel outcomes")
        opportunity_raw = mapping(row.get("channel_opportunities"), "channel opportunities")
        for basis, observed in (
            (
                "observed_outcome",
                [item for item in channel_ids if outcome_raw.get(item) is not None],
            ),
            (
                "observed_opportunity",
                [item for item in channel_ids if opportunity_raw.get(item) is True],
            ),
        ):
            pattern = "∩".join(observed) if observed else "NONE"
            cell = counts.setdefault(
                (basis, pattern),
                {"eligible_count": 0, "mature_count": 0, "prospective_count": 0},
            )
            cell["eligible_count"] += 1
            cell["mature_count" if row.get(MATURE) is True else "prospective_count"] += 1
        observed_count = sum(outcome_raw.get(item) is not None for item in channel_ids)
        distribution = counts.setdefault(
            ("observed_channel_count_distribution", str(observed_count)),
            {"eligible_count": 0, "mature_count": 0, "prospective_count": 0},
        )
        distribution["eligible_count"] += 1
        distribution["mature_count" if row.get(MATURE) is True else "prospective_count"] += 1
    return pd.DataFrame(
        [
            {"basis": basis, "channel_pattern": pattern, **values}
            for (basis, pattern), values in sorted(counts.items())
        ]
    )


def _failure_counts(values: list[Any], *, fields: list[str]) -> pd.DataFrame:
    rows = [mapping(item, "annual evidence failure") for item in values]
    if not rows:
        return pd.DataFrame(columns=[*fields, "failure_count"])
    frame = pd.DataFrame(rows)
    for field in fields:
        if field not in frame.columns:
            frame[field] = None
    return (
        frame.groupby(fields, dropna=False, sort=True).size().rename("failure_count").reset_index()
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write a documentation table without bypassing production artifact I/O."""
    fieldnames = [str(value) for value in frame.columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for raw in frame.to_dict(orient="records"):
            row = {str(key): _csv_value(value) for key, value in raw.items()}
            writer.writerow(row)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


if __name__ == "__main__":
    raise SystemExit(main())
