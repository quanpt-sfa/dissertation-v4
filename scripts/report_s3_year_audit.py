"""Audit S3 decision mapping and endpoint propagation by fiscal year."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import (
    DECISION_COUNT,
    DOCUMENT_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    HARD_POSITIVE,
    OUTCOME,
    OUTCOME_BASIS,
    ROW_INCLUSION,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    TARGET_FISCAL_YEAR,
    TAXONOMY_REASON_CODE,
)

S3_PREFIX = "S3_"


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
    if not isinstance(evidence, pd.DataFrame):
        raise TypeError("verified P03 evidence ledger must be a DataFrame")

    p06 = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P06",
        state="MEASURED",
    )
    matrices = mapping(
        p06.context.read("source_channel_matrices", {}),
        "source_channel_matrices",
    )
    if p05.protocol_hash != p06.protocol_hash:
        raise ValueError("S3 audit contexts must share one protocol hash")

    # Audit-only reads still pass through ArtifactStore manifest and schema verification.
    sanction_ledger = p05.context.store.read("sanction_decision_ledger", {})
    annual_audit = mapping(p05.context.store.read("annual_evidence_audit", {}), "annual audit")
    if not isinstance(sanction_ledger, pd.DataFrame):
        raise TypeError("verified sanction decision ledger must be a DataFrame")

    columns = physical_columns(p05.registry)
    initial_outer_year = int(mapping(p05.registry.get("folds"), "folds")["initial_outer_year"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    decision_counts = _decision_counts_by_year(sanction_ledger, columns)
    p03_counts = _p03_endpoint_counts_by_year(evidence, columns, initial_outer_year)
    p05_counts = _p05_endpoint_counts_by_year(matrices, initial_outer_year)
    unknown_reasons = _unknown_reasons_by_year(evidence, columns, initial_outer_year)
    reconciliation, mismatch_detail = _reconcile_p03_p05(evidence, matrices, columns)

    outputs = {
        "s3_decisions_by_target_year.csv": decision_counts,
        "s3_p03_endpoint_counts_by_year.csv": p03_counts,
        "s3_p05_endpoint_counts_by_year.csv": p05_counts,
        "s3_unknown_reasons_by_year_endpoint.csv": unknown_reasons,
        "s3_p03_p05_reconciliation_by_year.csv": reconciliation,
        "s3_p03_p05_mismatch_detail.csv": mismatch_detail,
    }
    for name, frame in outputs.items():
        _write_csv(output_dir / name, frame)

    development = p03_counts.loc[p03_counts["development_history"].eq(True)].copy()
    development_positive = int(development["positive"].sum()) if not development.empty else 0
    development_positive_by_endpoint = (
        {
            str(endpoint): int(group["positive"].sum())
            for endpoint, group in development.groupby(SOURCE_ID, sort=True)
        }
        if not development.empty
        else {}
    )
    content_positive = int(development_positive_by_endpoint.get("S3_CONTENT", 0))
    sanction_year_unresolved_firm_year_count = int(
        unknown_reasons.loc[
            unknown_reasons[OUTCOME_BASIS].astype(str).eq("SANCTION_YEAR_UNRESOLVED"),
            "unknown_count",
        ].sum()
    ) if not unknown_reasons.empty else 0
    mismatch_count = int(reconciliation["outcome_mismatch"].sum()) if not reconciliation.empty else 0
    missing_key_count = (
        int(reconciliation["p03_missing_key"].sum() + reconciliation["p05_missing_key"].sum())
        if not reconciliation.empty
        else 0
    )
    earliest_positive = _earliest_positive_by_endpoint(p03_counts)
    sanction_audit = mapping(
        annual_audit.get("sanction_calendar_year"),
        "annual_audit.sanction_calendar_year",
    )
    status = (
        "BLOCK_L3_RECONCILIATION"
        if mismatch_count or missing_key_count
        else "BLOCK_L3_NO_S3_CONTENT_DEVELOPMENT_POSITIVES"
        if content_positive == 0
        else "REVIEW_FOR_L3_LOCK"
    )
    summary = {
        "run_id": args.run_id,
        "protocol_hash": p05.protocol_hash,
        "initial_outer_year": initial_outer_year,
        "development_s3_positive_count": development_positive,
        "development_positive_count_by_endpoint": development_positive_by_endpoint,
        "development_s3_content_positive_count": content_positive,
        "sanction_year_unresolved_firm_year_count": sanction_year_unresolved_firm_year_count,
        "earliest_positive_fiscal_year_by_endpoint": earliest_positive,
        "p03_p05_outcome_mismatch_count": mismatch_count,
        "p03_p05_missing_key_count": missing_key_count,
        "sanction_input_row_count": sanction_audit.get("input_row_count"),
        "sanction_unique_decision_count": sanction_audit.get("unique_decision_count"),
        "sanction_firm_mapping_count": sanction_audit.get("firm_mapping_count"),
        "sanction_unresolved_year_mapping_count": sanction_audit.get(
            "unresolved_sanction_year_mapping_count"
        ),
        "excluded_source_rule_mapping_count": sanction_audit.get(
            "excluded_source_rule_mapping_count"
        ),
        "excluded_source_rule_row_count": sanction_audit.get(
            "excluded_source_rule_row_count"
        ),
        "excluded_unresolved_mapping_count": sanction_audit.get(
            "excluded_unresolved_mapping_count"
        ),
        "eligible_unresolved_sanction_year_mapping_count": sanction_audit.get(
            "eligible_unresolved_sanction_year_mapping_count"
        ),
        "status": status,
        "outer_outcomes_accessed": False,
        "scope_note": (
            "The audit reconciles verified P03 endpoint records with P05 source matrices. "
            "Decision rows excluded before P03 are summarized from the verified sanction decision "
            "ledger and annual audit; the raw CSV is not re-read."
        ),
    }
    (output_dir / "s3_year_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("S3_YEAR_AUDIT_JSON=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _decision_counts_by_year(frame: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    target_year = columns[TARGET_FISCAL_YEAR]
    document = columns[DOCUMENT_ID]
    firm = columns[FIRM_ID]
    included = columns[ROW_INCLUSION]
    hard_positive = columns[HARD_POSITIVE]
    reason = columns[TAXONOMY_REASON_CODE]

    rows: list[dict[str, object]] = []
    working = frame.copy()
    working["__year_group__"] = working[target_year].astype("Int64").astype("string").fillna("UNRESOLVED")
    for year_group, group in working.groupby("__year_group__", sort=True, dropna=False):
        included_mask = group[included].eq(True)
        hard_mask = group[hard_positive].eq(True)
        reason_text = group[reason].astype("string")
        year_value: int | None
        try:
            year_value = int(str(year_group))
        except ValueError:
            year_value = None
        rows.append(
            {
                TARGET_FISCAL_YEAR: year_value,
                "year_status": "resolved" if year_value is not None else "unresolved",
                "decision_firm_rows": len(group),
                "unique_documents": int(group[document].nunique(dropna=True)),
                "unique_firms": int(group[firm].nunique(dropna=True)),
                "row_included": int(included_mask.sum()),
                "hard_positive": int(hard_mask.sum()),
                "included_hard_positive": int((included_mask & hard_mask).sum()),
                "excluded_by_source_rule": int(reason_text.eq("EXCLUDED_BY_SOURCE_RULE").sum()),
                "taxonomy_unmapped": int(reason_text.eq("UNMAPPED_S3_TAXONOMY").sum()),
                "sanction_year_unresolved": int(reason_text.eq("SANCTION_YEAR_UNRESOLVED").sum()),
                "taxonomy_reason_missing": int(group[reason].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def _p03_endpoint_counts_by_year(
    evidence: pd.DataFrame,
    columns: dict[str, str],
    initial_outer_year: int,
) -> pd.DataFrame:
    source = columns[SOURCE_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    opportunity = columns[SOURCE_OPPORTUNITY]
    basis = columns[OUTCOME_BASIS]
    decision_count = columns[DECISION_COUNT]

    frame = evidence.loc[evidence[source].astype(str).str.startswith(S3_PREFIX)].copy()
    rows: list[dict[str, object]] = []
    for (fiscal_year, endpoint), group in frame.groupby([year, source], sort=True):
        observed = group[outcome].notna()
        positive = group[outcome].eq(True)
        negative = group[outcome].eq(False)
        basis_text = group[basis].astype("string")
        rows.append(
            {
                FISCAL_YEAR: int(str(fiscal_year)),
                SOURCE_ID: str(endpoint),
                "development_history": int(str(fiscal_year)) < initial_outer_year,
                "firm_year_records": len(group),
                "opportunity_observed": int(group[opportunity].eq(True).sum()),
                "opportunity_unknown": int(group[opportunity].isna().sum()),
                "positive": int(positive.sum()),
                "explicit_negative": int(negative.sum()),
                "unknown": int(group[outcome].isna().sum()),
                "observed_outcomes": int(observed.sum()),
                "observed_positive_rate": (
                    float(positive.sum()) / float(observed.sum()) if observed.any() else None
                ),
                "decision_count_sum": int(
                    pd.to_numeric(group[decision_count], errors="coerce").fillna(0).sum()
                ),
                "source_year_incomplete": int(basis_text.eq("SOURCE_YEAR_INCOMPLETE").sum()),
                "taxonomy_unmapped": int(basis_text.eq("UNMAPPED_S3_TAXONOMY").sum()),
                "sanction_year_unresolved": int(basis_text.eq("SANCTION_YEAR_UNRESOLVED").sum()),
                "complete_year_no_endpoint_decision": int(
                    basis_text.eq("no_endpoint_decision_in_complete_source_year").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _p05_endpoint_counts_by_year(
    matrices: dict[str, Any],
    initial_outer_year: int,
) -> pd.DataFrame:
    matrix_rows = [
        mapping(item, "source matrix row")
        for item in sequence(matrices.get("rows"), "source matrix rows")
    ]
    endpoints = sorted(
        {
            source_id
            for row in matrix_rows
            for source_id in mapping(row.get("source_outcomes"), "source outcomes")
            if str(source_id).startswith(S3_PREFIX)
        }
    )
    years = sorted({int(row[FISCAL_YEAR]) for row in matrix_rows})
    rows: list[dict[str, object]] = []
    for fiscal_year in years:
        year_rows = [row for row in matrix_rows if int(row[FISCAL_YEAR]) == fiscal_year]
        for endpoint in endpoints:
            outcomes = [
                mapping(row.get("source_outcomes"), "source outcomes").get(endpoint)
                for row in year_rows
            ]
            opportunities = [
                mapping(row.get("source_opportunities"), "source opportunities").get(endpoint)
                for row in year_rows
            ]
            observed = [value for value in outcomes if value is not None]
            positive = sum(value is True for value in outcomes)
            rows.append(
                {
                    FISCAL_YEAR: fiscal_year,
                    SOURCE_ID: endpoint,
                    "development_history": fiscal_year < initial_outer_year,
                    "matrix_rows": len(year_rows),
                    "eligible_rows": sum(row.get(ELIGIBLE) is True for row in year_rows),
                    "opportunity_observed": sum(value is True for value in opportunities),
                    "opportunity_unknown": sum(value is None for value in opportunities),
                    "positive": positive,
                    "explicit_negative": sum(value is False for value in outcomes),
                    "unknown": sum(value is None for value in outcomes),
                    "observed_outcomes": len(observed),
                    "observed_positive_rate": positive / len(observed) if observed else None,
                }
            )
    return pd.DataFrame(rows)


def _unknown_reasons_by_year(
    evidence: pd.DataFrame,
    columns: dict[str, str],
    initial_outer_year: int,
) -> pd.DataFrame:
    source = columns[SOURCE_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    basis = columns[OUTCOME_BASIS]
    frame = evidence.loc[
        evidence[source].astype(str).str.startswith(S3_PREFIX) & evidence[outcome].isna()
    ].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[FISCAL_YEAR, SOURCE_ID, OUTCOME_BASIS, "development_history", "unknown_count"]
        )
    result = (
        frame.groupby([year, source, basis], dropna=False, sort=True)
        .size()
        .rename("unknown_count")
        .reset_index()
    )
    result["development_history"] = result[year].astype(int) < initial_outer_year
    return result.rename(
        columns={year: FISCAL_YEAR, source: SOURCE_ID, basis: OUTCOME_BASIS}
    )[[FISCAL_YEAR, SOURCE_ID, OUTCOME_BASIS, "development_history", "unknown_count"]]


def _reconcile_p03_p05(
    evidence: pd.DataFrame,
    matrices: dict[str, Any],
    columns: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    source = columns[SOURCE_ID]
    outcome = columns[OUTCOME]
    opportunity = columns[SOURCE_OPPORTUNITY]

    p03: dict[tuple[str, int, str], tuple[bool | None, bool | None]] = {}
    frame = evidence.loc[evidence[source].astype(str).str.startswith(S3_PREFIX)]
    for raw in frame.to_dict(orient="records"):
        key = (str(raw[firm]), int(raw[year]), str(raw[source]))
        if key in p03:
            raise ValueError(f"duplicate P03 S3 key: {key}")
        p03[key] = (_tri_state(raw[outcome]), _tri_state(raw[opportunity]))

    p05: dict[tuple[str, int, str], tuple[bool | None, bool | None]] = {}
    for row in [
        mapping(item, "source matrix row")
        for item in sequence(matrices.get("rows"), "source matrix rows")
    ]:
        outcomes = mapping(row.get("source_outcomes"), "source outcomes")
        opportunities = mapping(row.get("source_opportunities"), "source opportunities")
        for endpoint, value in outcomes.items():
            endpoint_id = str(endpoint)
            if not endpoint_id.startswith(S3_PREFIX):
                continue
            key = (str(row[FIRM_ID]), int(row[FISCAL_YEAR]), endpoint_id)
            if key in p05:
                raise ValueError(f"duplicate P05 S3 key: {key}")
            p05[key] = (_tri_state(value), _tri_state(opportunities.get(endpoint_id)))

    detail_rows: list[dict[str, object]] = []
    for key in sorted(set(p03) | set(p05)):
        p03_value = p03.get(key)
        p05_value = p05.get(key)
        p03_outcome, p03_opportunity = p03_value if p03_value is not None else (None, None)
        p05_outcome, p05_opportunity = p05_value if p05_value is not None else (None, None)
        missing_p03 = key not in p03
        missing_p05 = key not in p05
        outcome_mismatch = not missing_p03 and not missing_p05 and p03_outcome != p05_outcome
        opportunity_mismatch = (
            not missing_p03 and not missing_p05 and p03_opportunity != p05_opportunity
        )
        if missing_p03 or missing_p05 or outcome_mismatch or opportunity_mismatch:
            detail_rows.append(
                {
                    FIRM_ID: key[0],
                    FISCAL_YEAR: key[1],
                    SOURCE_ID: key[2],
                    "p03_present": not missing_p03,
                    "p05_present": not missing_p05,
                    "p03_outcome": p03_outcome,
                    "p05_outcome": p05_outcome,
                    "p03_opportunity": p03_opportunity,
                    "p05_opportunity": p05_opportunity,
                    "outcome_mismatch": outcome_mismatch,
                    "opportunity_mismatch": opportunity_mismatch,
                }
            )

    detail = pd.DataFrame(detail_rows)
    all_rows: list[dict[str, object]] = []
    keys = sorted(set(p03) | set(p05))
    for fiscal_year, endpoint in sorted({(key[1], key[2]) for key in keys}):
        group_keys = [key for key in keys if key[1] == fiscal_year and key[2] == endpoint]
        all_rows.append(
            {
                FISCAL_YEAR: fiscal_year,
                SOURCE_ID: endpoint,
                "compared_keys": len(group_keys),
                "p03_missing_key": sum(key not in p03 for key in group_keys),
                "p05_missing_key": sum(key not in p05 for key in group_keys),
                "outcome_mismatch": sum(
                    key in p03 and key in p05 and p03[key][0] != p05[key][0]
                    for key in group_keys
                ),
                "opportunity_mismatch": sum(
                    key in p03 and key in p05 and p03[key][1] != p05[key][1]
                    for key in group_keys
                ),
            }
        )
    summary = pd.DataFrame(all_rows)
    if detail.empty:
        detail = pd.DataFrame(
            columns=[
                FIRM_ID,
                FISCAL_YEAR,
                SOURCE_ID,
                "p03_present",
                "p05_present",
                "p03_outcome",
                "p05_outcome",
                "p03_opportunity",
                "p05_opportunity",
                "outcome_mismatch",
                "opportunity_mismatch",
            ]
        )
    return summary, detail


def _earliest_positive_by_endpoint(frame: pd.DataFrame) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    if frame.empty:
        return result
    for endpoint, group in frame.groupby(SOURCE_ID, sort=True):
        positive_years = group.loc[group["positive"].gt(0), FISCAL_YEAR]
        result[str(endpoint)] = int(positive_years.min()) if not positive_years.empty else None
    return result


def _tri_state(value: object) -> bool | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    return bool(value)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    fieldnames = [str(value) for value in frame.columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for raw in frame.to_dict(orient="records"):
            writer.writerow({str(key): _csv_value(value) for key, value in raw.items()})


def _csv_value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


if __name__ == "__main__":
    raise SystemExit(main())
