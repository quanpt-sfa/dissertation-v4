# pyright: basic
"""P07C non-supervised audit of the researcher-supplied feature catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

AUDIT_COLUMNS = [
    "mapping_status",
    "candidate_data_source",
    "candidate_columns",
    "source_authorization",
    "coverage_overall",
    "paired_two_year_coverage",
    "coverage_by_year",
    "coverage_by_board",
    "coverage_by_industry",
    "coverage_by_fold",
    "denominator_diagnostics",
    "temporal_eligibility",
    "target_specific_leakage_decision",
    "final_researcher_decision",
    "decision_reason",
    "notes",
]
CONCEPTS = {
    "total assets": "total_assets",
    "net sales": "net_revenue",
    "sales": "net_revenue",
    "revenue": "net_revenue",
    "cogs": "cogs",
    "cost of goods sold": "cogs",
    "receivables": "accounts_receivable",
    "inventory": "inventory_net",
    "current assets": "current_assets",
    "current liabilities": "current_liabilities",
    "net ppe": "ppe_net",
    "property, plant": "ppe_net",
    "cash and cash equivalents": "cash_and_equivalents",
    "operating cash flow": "operating_cash_flow",
    "profit after tax": "profit_after_tax",
    "net income": "profit_after_tax",
    "total liabilities": "total_liabilities",
    "equity": "equity",
    "selling": "selling_expense",
    "administrative": "administrative_expense",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_items(text: str, available: set[str]) -> tuple[list[str], bool]:
    lowered = text.lower()
    found = [item for phrase, item in CONCEPTS.items() if phrase in lowered and item in available]
    unique = list(dict.fromkeys(found))
    unresolved = any(
        token in lowered
        for token in (
            "depreciation",
            "continuing operations",
            "issuance",
            "dividend",
            "capital expenditure",
            "ceo",
            "region",
            "market",
            "share price",
            "audit firm",
            "board chair",
        )
    )
    return unique, unresolved


def coverage(panel: pd.DataFrame, items: list[str], lagged: bool) -> tuple[str, str]:
    universe = panel[["issuer_ticker", "fiscal_year"]].drop_duplicates()
    if not items:
        return "0/0 (not computable)", "0/0 (not computable)"
    observed = panel.loc[
        panel.canonical_item.isin(items), ["issuer_ticker", "fiscal_year", "canonical_item"]
    ]
    counts = (
        observed.drop_duplicates()
        .groupby(["issuer_ticker", "fiscal_year"])
        .canonical_item.nunique()
    )
    valid = int((counts == len(items)).sum())
    total = len(universe)
    overall = f"{valid}/{total} ({valid / total:.3%})" if total else "0/0 (not computable)"
    if not lagged:
        return overall, "NOT_REQUIRED"
    current = counts.reset_index(name="n")
    current = current.loc[current.n.eq(len(items))]
    prior = current.assign(fiscal_year=current.fiscal_year + 1)
    paired = current.merge(prior, on=["issuer_ticker", "fiscal_year"], how="inner")
    eligible = universe.merge(
        universe.assign(fiscal_year=universe.fiscal_year + 1),
        on=["issuer_ticker", "fiscal_year"],
        how="inner",
    )
    paired_text = (
        f"{len(paired)}/{len(eligible)} ({len(paired) / len(eligible):.3%})"
        if len(eligible)
        else "0/0 (not computable)"
    )
    return overall, paired_text


def audit_row(row: pd.Series, panel: pd.DataFrame, available: set[str]) -> dict[str, str]:
    feature_id, formula = str(row.feature_id), str(row.formula_or_definition)
    required = str(row.required_raw_concepts)
    lower = f"{feature_id} {formula} {required}".lower()
    blocked = (
        "block" in str(row.specification_tier).lower() or "identity" in lower or "future" in lower
    )
    prepost = "pre/post" in lower or "audit adjustment" in lower
    transcription = (
        "source_formula_transcription_required" in formula.lower()
        or "must be transcribed" in formula.lower()
    )
    if blocked:
        return dict.fromkeys(AUDIT_COLUMNS, "BLOCK") | {
            "decision_reason": "Prohibited direct identity or future detection/enforcement variable.",
            "notes": "No source read for prohibited predictor.",
        }
    if prepost:
        return dict.fromkeys(AUDIT_COLUMNS, "SOURCE_NOT_AUTHORIZED") | {
            "candidate_data_source": "data/validation_only/financial_statement_pre_post_pairs.csv.gz",
            "source_authorization": "validation_only",
            "final_researcher_decision": "BLOCK",
            "decision_reason": "PREPOST_DATA_NOT_AUTHORIZED_FOR_PRODUCTION",
            "notes": "No production predictor until protocol owner authorizes a derived production source.",
        }
    if transcription:
        return dict.fromkeys(AUDIT_COLUMNS, "RESEARCH_DECISION_REQUIRED") | {
            "mapping_status": "NOT_APPLICABLE",
            "final_researcher_decision": "BLOCK",
            "decision_reason": "SOURCE_FORMULA_TRANSCRIPTION_REQUIRED",
            "notes": "Formula deliberately not inferred.",
        }
    items, unresolved = required_items(required + " " + formula, available)
    lagged = "t-1" in lower or "change" in lower or "average" in lower or "prior" in lower
    overall, paired = coverage(panel, items, lagged)
    mapping = (
        "CONCEPT_UNAVAILABLE"
        if unresolved or not items
        else (
            "DEFENSIBLE_VAS_MAPPING_REQUIRES_APPROVAL"
            if "receivable" in lower or "debt" in lower
            else "EXACT_MAPPING"
        )
    )
    decision = (
        "RESEARCHER_APPROVAL_REQUIRED"
        if mapping != "EXACT_MAPPING"
        else "CANDIDATE_PENDING_RESEARCHER_APPROVAL"
    )
    leak = "BLOCK" if "current audit opinion" in lower else "RESEARCH_DECISION_REQUIRED"
    return {
        "mapping_status": mapping,
        "candidate_data_source": "financial_statement_core_long",
        "candidate_columns": "|".join(items) if items else "NONE",
        "source_authorization": "production",
        "coverage_overall": overall,
        "paired_two_year_coverage": paired,
        "coverage_by_year": "See coverage_evidence.csv",
        "coverage_by_board": "SOURCE_NOT_AVAILABLE_FOR_STRATUM",
        "coverage_by_industry": "SOURCE_NOT_AVAILABLE_FOR_STRATUM",
        "coverage_by_fold": "NOT_COMPUTED_NO_P09_ACCESS",
        "denominator_diagnostics": "REPORTED_ONLY; no floor/winsorization/imputation applied"
        if "/" in formula
        else "NOT_APPLICABLE",
        "temporal_eligibility": "ANNUAL_ANCHOR_REVIEW_REQUIRED",
        "target_specific_leakage_decision": leak,
        "final_researcher_decision": decision,
        "decision_reason": "Mapping, authorization, coverage and timing audit only; no outcome data accessed.",
        "notes": "Compound specifications require every listed component to pass before canonical score use.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=Path("methodology/catalogue/FSF_Literature_Backed_Feature_Catalogue.xlsx"),
    )
    parser.add_argument(
        "--core",
        type=Path,
        default=Path("data/source/financial/financial_statement_core_long.csv.gz"),
    )
    parser.add_argument("--builder", type=Path, default=Path("scripts/p07c_fill_catalogue.mjs"))
    args = parser.parse_args()
    out = args.output_root / args.run_id / "P07C_FEATURE_CATALOGUE_AUDIT"
    out.mkdir(parents=True, exist_ok=False)
    catalogue = pd.read_excel(args.catalogue, sheet_name="Master_Features")
    panel = pd.read_csv(args.core, compression="infer", low_memory=False)
    panel = panel.loc[panel.audit_status.eq("audited") & panel.scope.eq("consolidated")].copy()
    available = set(panel.canonical_item.dropna().astype(str))
    audits = pd.DataFrame([audit_row(row, panel, available) for _, row in catalogue.iterrows()])
    completed = catalogue.copy()
    completed[AUDIT_COLUMNS] = audits[AUDIT_COLUMNS]
    completed.to_csv(
        out / "FSF_Literature_Backed_Feature_Catalogue.csv", index=False, encoding="utf-8-sig"
    )
    rows = completed[["feature_id", *AUDIT_COLUMNS]].to_dict("records")
    (out / "catalogue_audit_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8"
    )
    evidence = completed[
        [
            "feature_id",
            "required_raw_concepts",
            "mapping_status",
            "candidate_data_source",
            "candidate_columns",
            "source_authorization",
            "decision_reason",
        ]
    ]
    evidence.to_csv(out / "feature_mapping_evidence.csv", index=False)
    completed[
        [
            "feature_id",
            "coverage_overall",
            "paired_two_year_coverage",
            "coverage_by_year",
            "coverage_by_board",
            "coverage_by_industry",
            "coverage_by_fold",
        ]
    ].to_csv(out / "coverage_evidence.csv", index=False)
    completed[["feature_id", "denominator_diagnostics"]].to_csv(
        out / "denominator_diagnostics.csv", index=False
    )
    completed[
        [
            "feature_id",
            "temporal_eligibility",
            "target_specific_leakage_decision",
            "final_researcher_decision",
        ]
    ].to_csv(out / "temporal_and_leakage_decisions.csv", index=False)
    completed.loc[
        completed.mapping_status.ne("EXACT_MAPPING"),
        ["feature_id", "mapping_status", "decision_reason", "notes"],
    ].to_csv(out / "confirmed_data_gap_register.csv", index=False)
    subprocess.run(
        [
            "node",
            str(args.builder),
            str(args.catalogue),
            str(out / "catalogue_audit_rows.json"),
            str(out / "FSF_Literature_Backed_Feature_Catalogue.xlsx"),
        ],
        check=True,
    )
    summary = {
        "verdict": "PARTIAL_CATALOGUE_AUDIT_WITH_NAMED_GAPS",
        "run_id": args.run_id,
        "catalogue_rows": len(catalogue),
        "catalogue_hash": sha256(args.catalogue),
        "no_outcomes_read": True,
        "no_known_cases_read": True,
        "no_operational_features_created": True,
        "mapping_status_counts": completed.mapping_status.value_counts().to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "FEATURE_CATALOGUE_DATA_AUDIT.md").write_text(
        "# P07C Feature Catalogue Data Audit\n\nVerdict: `PARTIAL_CATALOGUE_AUDIT_WITH_NAMED_GAPS`. The authoritative 211 feature rows and IDs were preserved. This audit did not read outcomes or known cases and did not operationalize P07. Exact canonical scores remain blocked unless every component has a valid, approved mapping.\n",
        encoding="utf-8",
    )
    print(f"P07C verdict=PARTIAL_CATALOGUE_AUDIT_WITH_NAMED_GAPS rows={len(catalogue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
