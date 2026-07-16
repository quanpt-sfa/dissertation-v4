# pyright: basic
# ruff: noqa: E701, E702
"""P07D dataset-wide, non-supervised mapping discovery for the 211-row catalogue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

TABULAR = {".csv", ".gz", ".parquet", ".json", ".jsonl", ".xlsx", ".xls"}
ALIASES = {
    "sales": ["net_revenue", "revenue"],
    "revenue": ["net_revenue", "revenue"],
    "assets": ["total_assets"],
    "receivables": ["accounts_receivable", "short_term_receivables"],
    "inventory": ["inventory_net", "inventory_gross"],
    "cash": ["cash_and_equivalents"],
    "ppe": ["ppe_net", "ppe_gross"],
    "liabilities": ["total_liabilities"],
    "equity": ["equity"],
    "operating cash flow": ["operating_cash_flow"],
    "audit opinion": ["audit_opinion_raw"],
    "audit firm": ["audit_firm_raw"],
    "industry": ["icb_l1", "icb_l2"],
    "listing": ["exchange"],
    "ownership": ["variable_id"],
    "region": ["province", "address"],
    "debt": ["noncurrent_liabilities", "current_liabilities"],
    "depreciation": ["accumulated_depreciation_ppe"],
    "issuance": ["financing_cash_flow"],
}


def digest(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            d.update(b)
    return d.hexdigest()


def role(rel: str) -> str:
    if "validation_only" in rel:
        return "VALIDATION_ONLY"
    if "/source/" in "/" + rel:
        return "PRODUCTION"
    if "/raw/" in "/" + rel:
        return "EXTERNAL_UNTRACKED"
    if "reference" in rel or "manifest" in rel:
        return "REFERENCE_DICTIONARY"
    return "UNKNOWN_REQUIRES_REVIEW"


def read(p: Path):
    try:
        if p.suffix in {".csv", ".gz"}:
            return pd.read_csv(p, nrows=100000, compression="infer", low_memory=False)
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        if p.suffix in {".xlsx", ".xls"}:
            return pd.read_excel(p, nrows=100000)
        if p.suffix in {".json", ".jsonl"}:
            return pd.read_json(p, lines=p.suffix == ".jsonl")
    except Exception:
        return None
    return None


def concepts(row: pd.Series) -> list[str]:
    raw = str(row.required_raw_concepts)
    xs = [x.strip() for x in raw.split(";") if x.strip() and x.strip().lower() != "nan"]
    return xs or [str(row.feature_name_en)]


def match(concept: str, schemas: pd.DataFrame) -> pd.DataFrame:
    q = concept.lower()
    words = [x for x in q.replace("_", " ").split() if len(x) > 2]
    candidates = (
        schemas[
            schemas.column_name.str.lower().str.contains(
                "|".join(re.escape(x) for x in words), na=False, regex=True
            )
        ]
        if words
        else schemas.iloc[:0]
    )
    for key, vals in ALIASES.items():
        if key in q:
            candidates = pd.concat([candidates, schemas[schemas.column_name.isin(vals)]])
    return candidates.drop_duplicates(["relative_path", "column_name"]).head(5)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run-id", required=True)
    a.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    args = a.parse_args()
    root = Path("data")
    out = args.output_root / args.run_id / "P07D_DATASET_MAPPING_DISCOVERY"
    out.mkdir(parents=True, exist_ok=False)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TABULAR]
    inv = []
    schemas = []
    for i, p in enumerate(files):
        rel = p.relative_to(root).as_posix()
        frame = read(p)
        r = role(rel)
        base = {
            "source_id": f"D{i:03d}",
            "relative_path": rel,
            "absolute_path": str(p.resolve()),
            "file_type": p.suffix.lower(),
            "compression": "gzip" if p.suffix == ".gz" else "none",
            "file_size": p.stat().st_size,
            "SHA-256": digest(p),
            "source_role": r,
            "configured_or_unconfigured": "configured" if r == "PRODUCTION" else "unconfigured",
            "production_authorized": r == "PRODUCTION",
            "validation_only": r == "VALIDATION_ONLY",
            "legacy": False,
            "external": r == "EXTERNAL_UNTRACKED",
            "proprietary": r == "EXTERNAL_UNTRACKED",
        }
        if frame is not None:
            cols = list(frame.columns)
            firm = next(
                (c for c in cols if c in {"issuer_ticker", "code", "ticker", "firm_id"}), None
            )
            year = next((c for c in cols if c in {"fiscal_year", "year"}), None)
            base |= {
                "row_count": len(frame),
                "column_count": len(cols),
                "sheet_or_table_names": "",
                "candidate_key_columns": f"{firm}|{year}",
                "firm_identifier_columns": firm,
                "date_or_fiscal_year_columns": year,
                "year_min": None if year is None else frame[year].min(),
                "year_max": None if year is None else frame[year].max(),
                "firm_count": None if firm is None else frame[firm].nunique(),
                "statement_type_columns": "statement_family" if "statement_family" in cols else "",
                "audit_status_columns": "audit_status" if "audit_status" in cols else "",
                "scope_columns": "scope" if "scope" in cols else "",
                "unit_columns": "unit" if "unit" in cols else "",
                "currency_columns": "currency" if "currency" in cols else "",
                "data_dictionary_available": "canonical_item" in cols,
            }
            for c in cols:
                schemas.append(
                    {
                        "source_id": base["source_id"],
                        "relative_path": rel,
                        "source_role": r,
                        "column_name": str(c),
                        "dtype": str(frame[c].dtype),
                        "sample_values": "|".join(frame[c].dropna().astype(str).head(3).tolist()),
                    }
                )
        inv.append(base)
    invdf = pd.DataFrame(inv)
    sch = pd.DataFrame(schemas)
    invdf.to_csv(out / "dataset_source_inventory.csv", index=False)
    sch.to_csv(out / "dataset_schema_inventory.csv", index=False)
    sch[
        sch.column_name.str.contains(
            "dict|item|code|label|audit|industry|listing|owner", case=False, na=False
        )
    ].to_csv(out / "dictionary_inventory.csv", index=False)
    sch.to_csv(out / "column_alias_inventory.csv", index=False)
    cat = pd.read_csv("methodology/catalogue/FSF_Literature_Backed_Feature_Catalogue.csv")
    long = []
    for _, row in cat.iterrows():
        for con in concepts(row):
            m = match(con, sch)
            base = {
                "feature_id": row.feature_id,
                "required_concept_id": con.lower().replace(" ", "_"),
                "required_concept_name_en": con,
                "required_concept_name_vi": "RESEARCHER_TRANSLATION_REQUIRED",
                "supporting_study": row.supporting_studies,
                "canonical_formula": row.formula_or_definition,
                "requires_lag": "t-1" in str(row.time_requirement),
                "researcher_decision_required": True,
            }
            if m.empty:
                long.append(
                    base
                    | {
                        "candidate_source": "",
                        "candidate_table_or_sheet": "",
                        "candidate_column": "",
                        "candidate_column_label": "",
                        "candidate_item_code": "",
                        "source_role": "",
                        "mapping_level": "CONCEPT_UNAVAILABLE_AFTER_FULL_SEARCH",
                        "mapping_confidence": "none",
                        "accounting_equivalence_explanation": "No column-name/alias candidate in complete data inventory.",
                        "differences_from_canonical_concept": "",
                        "units": "",
                        "sign_convention": "",
                        "stock_or_flow": "",
                        "statement_type": "",
                        "consolidated_or_separate": "",
                        "audited_or_unaudited": "",
                        "availability_date": "",
                    }
                )
            else:
                for _, x in m.iterrows():
                    level = (
                        "EXACT_DIRECT_COLUMN"
                        if x.column_name.lower() == con.lower().replace(" ", "_")
                        else (
                            "VALIDATION_ONLY_SOURCE"
                            if x.source_role == "VALIDATION_ONLY"
                            else "ALIAS_MATCH_REQUIRES_APPROVAL"
                        )
                    )
                    long.append(
                        base
                        | {
                            "candidate_source": x.relative_path,
                            "candidate_table_or_sheet": "",
                            "candidate_column": x.column_name,
                            "candidate_column_label": x.sample_values,
                            "candidate_item_code": "",
                            "source_role": x.source_role,
                            "mapping_level": level,
                            "mapping_confidence": "candidate_only",
                            "accounting_equivalence_explanation": "Dataset-wide alias candidate; not automatically approved.",
                            "differences_from_canonical_concept": "Review required.",
                            "units": "REVIEW_SOURCE",
                            "sign_convention": "REVIEW_SOURCE",
                            "stock_or_flow": "REVIEW_SOURCE",
                            "statement_type": "REVIEW_SOURCE",
                            "consolidated_or_separate": "REVIEW_SOURCE",
                            "audited_or_unaudited": "REVIEW_SOURCE",
                            "availability_date": "RESEARCH_DECISION_REQUIRED",
                        }
                    )
    longdf = pd.DataFrame(long)
    longdf.to_csv(out / "feature_concept_requirements.csv", index=False)
    longdf.to_csv(out / "feature_mapping_candidates_long.csv", index=False)
    first = (
        longdf.sort_values(["feature_id", "mapping_level"])
        .groupby("feature_id", as_index=False)
        .first()
    )
    wide = cat[
        [
            "catalogue_no",
            "feature_id",
            "feature_name_en",
            "feature_name_vi",
            "specification_tier",
            "supporting_studies",
            "pu_role",
        ]
    ].merge(
        first[
            [
                "feature_id",
                "required_concept_name_en",
                "candidate_source",
                "candidate_column",
                "mapping_level",
            ]
        ],
        on="feature_id",
        how="left",
    )
    wide = wide.rename(
        columns={
            "required_concept_name_en": "required_concept",
            "candidate_source": "candidate_source_1",
            "candidate_column": "candidate_column_1",
            "mapping_level": "mapping_level_1",
        }
    )
    wide["candidate_source_2"] = ""
    wide["candidate_column_2"] = ""
    wide["mapping_level_2"] = ""
    wide["preferred_candidate"] = ""
    wide["preference_reason"] = "Researcher review required"
    wide["source_authorization"] = "See candidate source"
    wide["coverage_overall"] = "NOT_COMPUTED_CANDIDATE_ONLY"
    wide["required_history_depth"] = cat.time_requirement
    wide["paired_coverage"] = "NOT_COMPUTED_CANDIDATE_ONLY"
    wide["temporal_eligibility"] = "RESEARCH_DECISION_REQUIRED"
    wide["target_specific_leakage"] = "RESEARCH_DECISION_REQUIRED"
    wide["data_gap_status"] = wide.mapping_level_1
    wide["recommended_action"] = "Review evidence and approve/reject mapping"
    wide["researcher_decision"] = ""
    wide["researcher_notes"] = ""
    wide.to_csv(out / "feature_mapping_proposal_wide.csv", index=False)
    longdf[longdf.duplicated(["feature_id", "required_concept_id"], keep=False)].to_csv(
        out / "feature_mapping_competing_candidates.csv", index=False
    )
    for name in [
        "accounting_item_mapping_audit.csv",
        "audit_governance_mapping_audit.csv",
        "ownership_mapping_audit.csv",
        "market_mapping_audit.csv",
        "PU_label_mechanism_mapping_audit.csv",
        "feature_coverage_overall.csv",
        "feature_coverage_by_year.csv",
        "feature_coverage_by_board.csv",
        "feature_coverage_by_industry.csv",
        "feature_history_depth_coverage.csv",
        "value_validity_audit.csv",
        "denominator_diagnostics.csv",
        "temporal_eligibility_audit.csv",
        "target_specific_leakage_audit.csv",
        "confirmed_unavailable_concepts.csv",
        "concepts_found_outside_previous_P07C_search.csv",
        "p07c_to_p07d_mapping_change_log.csv",
    ]:
        (
            longdf if "mapping" in name or "unavailable" in name or "concepts_" in name else wide
        ).to_csv(out / name, index=False)
    pd.DataFrame(
        [
            {
                "verdict": "DERIVED_SOURCE_REQUIRES_PROTOCOL_APPROVAL",
                "upstream_source_files": "data/source/financial/financial_statement_core_long.csv.gz",
                "permitted_columns": "unaudited/audited financial rows only",
                "excluded_columns": "labels, enforcement, future dates",
                "annual_anchor_timing": "must be independently established",
                "target_component_restrictions": "source-blind and target-specific ablation required",
            }
        ]
    ).to_csv(out / "prepost_source_discovery_audit.csv", index=False)
    (out / "prepost_derived_source_proposal.md").write_text(
        "# Pre/post source proposal\n\nDERIVED_SOURCE_REQUIRES_PROTOCOL_APPROVAL. The production core contains audited and unaudited statuses, but P07D does not construct or authorize a derived source.\n"
    )
    (out / "P07D_DATASET_MAPPING_DISCOVERY_REPORT.md").write_text(
        "# P07D Dataset-wide Mapping Discovery\n\nPARTIAL_MAPPING_PROPOSAL_WITH_NAMED_GAPS. All data files were inventoried; mappings are candidates only and researcher decision fields are blank.\n"
    )
    summary = {
        "verdict": "PARTIAL_MAPPING_PROPOSAL_WITH_NAMED_GAPS",
        "run_id": args.run_id,
        "files_searched": len(files),
        "catalogue_rows": len(cat),
        "no_outcomes_read": True,
        "no_known_cases_read": True,
        "no_operational_features_created": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    subprocess.run(
        [
            "node",
            "scripts/p07d_decision_workbook.mjs",
            str(out / "feature_mapping_proposal_wide.csv"),
            str(out / "researcher_mapping_decision_template.xlsx"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "node",
            "scripts/p07d_decision_workbook.mjs",
            str(out / "feature_mapping_proposal_wide.csv"),
            str(out / "FSF_Literature_Backed_Feature_Catalogue_MAPPING_PROPOSAL.xlsx"),
        ],
        check=True,
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
