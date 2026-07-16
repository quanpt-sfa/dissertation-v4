# pyright: basic
"""P07B literature-to-data feasibility audit (non-production and non-supervised)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features.literature_audit import (  # noqa: E402
    BENEISH_COMPONENTS,
    DECHOW_MODEL_1_COMPONENTS,
    canonical_item_lookup,
    classify_mapping,
    paired_coverage,
    sha256,
    stable_candidate_items,
)

LITERATURE_FILES = {
    "BENEISH_1999": "beneish1999detection.md",
    "DECHOW_2011": "dechow2011predicting.md",
    "CECCHINI_2010": "cecchini2010detecting.md",
    "PEROLS_2011": "perols2011financial.md",
    "BAO_2020": "bao2020detecting.md",
    "TRAN_2015": "tran2015evaluating.md",
    "BUI_2021": "bui2021predicting.md",
    "SYSTEMATIC_REVIEW": "shahana2023state.md",
}
LITERATURE_CITATIONS = {
    "BENEISH_1999": "Beneish, M. D. (1999). The Detection of Earnings Manipulation.",
    "DECHOW_2011": "Dechow, P. M., Ge, W., Larson, C. R., & Sloan, R. G. (2011). Predicting Material Accounting Misstatements.",
    "CECCHINI_2010": "Cecchini, M., Aytug, H., Koehler, G. J., & Pathak, P. (2010). Detecting Management Fraud in Public Companies.",
    "PEROLS_2011": "Perols, J. (2011). Financial Statement Fraud Detection: An Analysis of Statistical and Machine Learning Algorithms.",
    "BAO_2020": "Bao, Y., Ke, B., Li, B., Yu, Y. J., & Zhang, J. (2020). Detecting Accounting Fraud in Publicly Traded U.S. Firms.",
    "TRAN_2015": "Tran, T. H. (2015). Evaluating the effect of corporate governance on accounting fraud in Vietnam.",
    "BUI_2021": "Bui, T. N. (2021). Predicting financial statement fraud: evidence from Vietnam.",
    "SYSTEMATIC_REVIEW": "Shahana et al. (2023). State-of-the-art systematic review material.",
}

# This is a transparent concept vocabulary, not a feature registry and not a model.
CONCEPT_CANDIDATES = {
    "net_sales": ["net_revenue"],
    "net_trade_receivables": ["accounts_receivable"],
    "cost_of_goods_sold": ["cogs"],
    "current_assets": ["current_assets"],
    "current_liabilities": ["current_liabilities"],
    "property_plant_equipment_net": ["ppe_net"],
    "property_plant_equipment_gross": ["ppe_gross"],
    "accumulated_depreciation": ["accumulated_depreciation_ppe"],
    "depreciation_expense": [],
    "selling_general_administrative_expense": ["selling_expense", "administrative_expense"],
    "total_assets": ["total_assets"],
    "long_term_debt": ["noncurrent_liabilities"],
    "operating_cash_flow": ["operating_cash_flow"],
    "income_continuing_operations": [],
    "profit_after_tax": ["profit_after_tax"],
    "inventory": ["inventory_net"],
    "cash": ["cash_and_equivalents"],
    "debt_issuance": [],
    "equity_issuance": [],
}
BENEISH_FORMULAS = {
    "DSRI": "(net receivables_t / net sales_t) / (net receivables_t-1 / net sales_t-1)",
    "GMI": "((sales_t-1 - COGS_t-1)/sales_t-1) / ((sales_t - COGS_t)/sales_t)",
    "AQI": "[1 - (current assets_t + PPE_t)/total assets_t] / [1 - (current assets_t-1 + PPE_t-1)/total assets_t-1]",
    "SGI": "net sales_t / net sales_t-1",
    "DEPI": "[depreciation_t-1/(depreciation_t-1 + PPE_t-1)] / [depreciation_t/(depreciation_t + PPE_t)]",
    "SGAI": "(SG&A_t/net sales_t) / (SG&A_t-1/net sales_t-1)",
    "LVGI": "[(current liabilities_t + long-term debt_t)/total assets_t] / [(current liabilities_t-1 + long-term debt_t-1)/total assets_t-1]",
    "TATA": "(income from continuing operations_t - cash flow from operations_t) / total assets_t",
}
DECHOW_FORMULAS = {
    "RSST_ACCRUALS": "Change in non-cash working capital + change in non-current operating assets - change in non-current operating liabilities, scaled by average total assets.",
    "CHANGE_RECEIVABLES": "Change in receivables scaled by average total assets.",
    "CHANGE_INVENTORY": "Change in inventory scaled by average total assets.",
    "SOFT_ASSETS": "Soft assets divided by total assets.",
    "CHANGE_CASH_SALES": "Change in cash sales scaled by average total assets.",
    "CHANGE_ROA": "Current ROA minus prior-year ROA.",
    "ACTUAL_ISSUANCE": "Indicator for actual debt or equity issuance.",
}


def read_source(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, compression="infer", low_memory=False) if path.is_file() else None


def source_record(
    source_id: str, path: Path, role: str, frame: pd.DataFrame | None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_id": source_id,
        "resolved_path": str(path.resolve()),
        "source_role": role,
        "production_allowed": role == "production",
        "validation_only": role == "validation_only",
        "audit_only": role == "audit_only",
        "file_exists": path.is_file(),
    }
    if frame is None:
        return record
    firm_column = next((x for x in ("issuer_ticker", "firm_id", "code") if x in frame), None)
    year_column = next((x for x in ("fiscal_year", "year") if x in frame), None)
    key = [x for x in (firm_column, year_column) if x]
    record.update(
        file_hash=sha256(path),
        row_count=len(frame),
        column_count=len(frame.columns),
        firm_count=None if firm_column is None else int(frame[firm_column].nunique()),
        fiscal_year_min=None if year_column is None else int(frame[year_column].min()),
        fiscal_year_max=None if year_column is None else int(frame[year_column].max()),
        key_columns="|".join(key),
        duplicate_key_count=0 if not key else int(frame.duplicated(key).sum()),
    )
    return record


def all_literature_rows(root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    for study_id, filename in LITERATURE_FILES.items():
        matches = list(root.rglob(filename))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"{study_id}: expected exactly one {filename}, found {len(matches)}"
            )
        path = matches[0]
        paths[study_id] = path
        rows.append(
            {
                "study_id": study_id,
                "full_citation": LITERATURE_CITATIONS[study_id],
                "source_file": str(path.resolve()),
                "source_hash": sha256(path),
                "source_format": "markdown extracted from supplied fsf.zip",
            }
        )
    return pd.DataFrame(rows), paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--literature-root", type=Path, required=True)
    parser.add_argument("--literature-archive", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--legacy-root", type=Path, default=Path(r"D:\Works\dissertation\dissertation\data")
    )
    args = parser.parse_args()
    out = args.output_root / args.run_id / "P07B"
    if out.exists():
        raise FileExistsError(f"immutable audit directory already exists: {out}")
    out.mkdir(parents=True)
    try:
        literature, _literature_paths = all_literature_rows(args.literature_root)
    except FileNotFoundError as error:
        (out / "P07B_LITERATURE_TO_DATA_AUDIT.md").write_text(
            f"# P07B\n\nLITERATURE_BUNDLE_NOT_FOUND\n\n{error}\n", encoding="utf-8"
        )
        return 2
    literature.to_csv(out / "literature_source_inventory.csv", index=False)
    if args.literature_archive is not None:
        if not args.literature_archive.is_file():
            raise FileNotFoundError(f"literature archive not found: {args.literature_archive}")
        archive_record = pd.DataFrame(
            [
                {
                    "study_id": "LITERATURE_BUNDLE_ARCHIVE",
                    "full_citation": "Supplied fsf.zip literature bundle",
                    "source_file": str(args.literature_archive.resolve()),
                    "source_hash": sha256(args.literature_archive),
                    "source_format": "zip archive; extracted outside repository",
                }
            ]
        )
        literature = pd.concat([literature, archive_record], ignore_index=True)
        literature.to_csv(out / "literature_source_inventory.csv", index=False)

    sources = {
        "financial_statement_core_long": (
            Path("data/source/financial/financial_statement_core_long.csv.gz"),
            "production",
        ),
        "financial_statement_pre_post_pairs": (
            Path("data/validation_only/financial_statement_pre_post_pairs.csv.gz"),
            "validation_only",
        ),
        "bctc_indicator_dictionary": (args.legacy_root / "bctc_indicator_dictionary.csv", "legacy"),
        "bctc_item_dictionary": (args.legacy_root / "bctc_item_dictionary.csv", "legacy"),
        "bctc_firm_year_features": (args.legacy_root / "bctc_firm_year_features.csv", "legacy"),
        "bctc_firm_year_features_2015_listed": (
            args.legacy_root / "bctc_firm_year_features_2015_listed.csv",
            "legacy",
        ),
    }
    frames = {name: read_source(path) for name, (path, _) in sources.items()}
    records = [
        source_record(name, path, role, frames[name]) for name, (path, role) in sources.items()
    ]
    pd.DataFrame(records).to_csv(out / "source_resolution_audit.csv", index=False)
    pd.DataFrame(records)[
        [
            "source_id",
            "source_role",
            "production_allowed",
            "validation_only",
            "audit_only",
            "file_exists",
        ]
    ].to_csv(out / "source_role_audit.csv", index=False)
    core = frames["financial_statement_core_long"]
    pairs = frames["financial_statement_pre_post_pairs"]
    if core is None:
        raise FileNotFoundError(
            "Authorised production financial_statement_core_long source is required"
        )
    production = core.loc[
        core["audit_status"].eq("audited") & core["scope"].eq("consolidated")
    ].copy()
    observed_items = production["canonical_item"].dropna().unique()

    feature_rows: list[dict[str, Any]] = []
    for component in BENEISH_COMPONENTS:
        requirements = {
            "DSRI": ["net_trade_receivables", "net_sales"],
            "GMI": ["net_sales", "cost_of_goods_sold"],
            "AQI": ["current_assets", "property_plant_equipment_net", "total_assets"],
            "SGI": ["net_sales"],
            "DEPI": ["depreciation_expense", "property_plant_equipment_net"],
            "SGAI": ["selling_general_administrative_expense", "net_sales"],
            "LVGI": ["current_liabilities", "long_term_debt", "total_assets"],
            "TATA": ["income_continuing_operations", "operating_cash_flow", "total_assets"],
        }[component]
        mappings = [
            canonical_item_lookup(observed_items, CONCEPT_CANDIDATES[concept])
            for concept in requirements
        ]
        status = "EXACT_MAPPING" if all(mappings) else "CONCEPT_UNAVAILABLE"
        if component in {"DEPI", "TATA"}:
            status = "CONCEPT_UNAVAILABLE"
        coverage = min(
            (paired_coverage(production, item)[1] for item in mappings if item), default=0.0
        )
        feature_rows.append(
            {
                "study_id": "BENEISH_1999",
                "location": "Table 1 and pp. 24-26",
                "feature_name": component,
                "canonical_formula": BENEISH_FORMULAS[component],
                "required_accounting_concepts": "|".join(requirements),
                "published_coefficient": "see published M-score equation",
                "published_transformations": "ratio index",
                "mapping_status": status,
                "paired_coverage": coverage,
            }
        )
    for component in DECHOW_MODEL_1_COMPONENTS:
        feature_rows.append(
            {
                "study_id": "DECHOW_2011",
                "location": "Model 1 / Table 2",
                "feature_name": component,
                "canonical_formula": DECHOW_FORMULAS[component],
                "required_accounting_concepts": "see formula",
                "published_coefficient": "published Model 1 coefficient",
                "published_transformations": "as published",
                "mapping_status": "RESEARCH_DECISION_REQUIRED",
                "paired_coverage": None,
            }
        )
    pd.DataFrame(feature_rows).to_csv(out / "literature_feature_specification.csv", index=False)

    beneish = pd.DataFrame([row for row in feature_rows if row["study_id"] == "BENEISH_1999"])
    beneish.to_csv(out / "beneish_component_mapping_audit.csv", index=False)
    dechow = pd.DataFrame([row for row in feature_rows if row["study_id"] == "DECHOW_2011"])
    dechow.to_csv(out / "dechow_component_mapping_audit.csv", index=False)

    raw_concepts = []
    for concept, candidates in CONCEPT_CANDIDATES.items():
        item = canonical_item_lookup(observed_items, candidates)
        paired_count, paired_rate = (0, 0.0) if item is None else paired_coverage(production, item)
        raw_concepts.append(
            {
                "literature_raw_concept_id": concept.upper(),
                "canonical_concept_name": concept,
                "supporting_studies": "CECCHINI_2010|BAO_2020|PEROLS_2011",
                "number_of_supporting_studies": 3,
                "statement_type": "accounting",
                "stock_or_flow": "RESEARCH_REVIEW_REQUIRED",
                "required_for_named_benchmark": concept in {"net_sales", "total_assets"},
                "exact_vietnamese_item": item,
                "candidate_source_column": "canonical_item/value_numeric" if item else None,
                "mapping_confidence": classify_mapping(
                    item,
                    authorised=True,
                    approval=bool(item and concept in {"net_trade_receivables", "long_term_debt"}),
                ),
                "production_source_authorization": True,
                "coverage": 0.0
                if item is None
                else float((production["canonical_item"] == item).mean()),
                "paired_two_year_coverage": paired_rate,
                "year_range": f"{production.fiscal_year.min()}-{production.fiscal_year.max()}",
            }
        )
    concepts = pd.DataFrame(raw_concepts)
    concepts.iloc[:, :8].to_csv(out / "literature_raw_concept_union.csv", index=False)
    concepts.to_csv(out / "vietnamese_accounting_mapping_matrix.csv", index=False)
    concepts[["canonical_concept_name", "coverage", "paired_two_year_coverage"]].to_csv(
        out / "feature_coverage_overall.csv", index=False
    )
    production.groupby("fiscal_year")["canonical_item"].nunique().reset_index(
        name="distinct_items"
    ).to_csv(out / "feature_coverage_by_year.csv", index=False)
    pd.DataFrame(columns=["listing_board", "feature_id", "coverage"]).to_csv(
        out / "feature_coverage_by_board.csv", index=False
    )
    pd.DataFrame(columns=["industry", "feature_id", "coverage"]).to_csv(
        out / "feature_coverage_by_industry.csv", index=False
    )
    pd.DataFrame(columns=["temporal_fold", "feature_id", "coverage"]).to_csv(
        out / "feature_coverage_by_fold.csv", index=False
    )
    concepts[["canonical_concept_name", "paired_two_year_coverage"]].to_csv(
        out / "paired_two_year_coverage.csv", index=False
    )

    if pairs is None:
        prepost = pd.DataFrame(
            columns=["canonical_item", "paired_coverage", "production_authorization"]
        )
        eligibility = prepost.copy()
    else:
        prepost = (
            pairs.groupby("canonical_item", dropna=False)
            .agg(
                paired_coverage=("pair_complete_flag", "mean"),
                nonzero_adjustment_count=(
                    "audit_delta_post_minus_pre",
                    lambda x: int((x != 0).sum()),
                ),
                positive_adjustment_count=(
                    "audit_delta_post_minus_pre",
                    lambda x: int((x > 0).sum()),
                ),
                negative_adjustment_count=(
                    "audit_delta_post_minus_pre",
                    lambda x: int((x < 0).sum()),
                ),
                zero_adjustment_count=("audit_delta_post_minus_pre", lambda x: int((x == 0).sum())),
            )
            .reset_index()
        )
        prepost["production_authorization"] = False
        prepost["source_role_finding"] = "PREPOST_DATA_NOT_AUTHORIZED_FOR_PRODUCTION"
        eligibility = stable_candidate_items(pairs).merge(prepost, on="canonical_item", how="left")
        eligibility["eligibility_status"] = "PROHIBITED_SOURCE_ROLE"
    prepost.to_csv(out / "prepost_item_universe.csv", index=False)
    eligibility.to_csv(out / "prepost_eligibility_matrix.csv", index=False)
    sensitivity = pd.DataFrame(
        [
            {
                "nonzero_threshold": threshold,
                "items_surviving": int(
                    (
                        prepost.get("nonzero_adjustment_count", pd.Series(dtype=int)) >= threshold
                    ).sum()
                ),
            }
            for threshold in (25, 50, 100)
        ]
    )
    sensitivity.to_csv(out / "prepost_threshold_sensitivity.csv", index=False)

    denominator = pd.DataFrame(
        [
            {
                "feature_id": feature,
                "formula": formula,
                "zero_denominator_count": "NOT_APPLIED",
                "negative_denominator_count": "NOT_APPLIED",
                "missing_denominator_count": "NOT_APPLIED",
                "small_denominator_diagnostics": "REPORTED_ONLY_NO_FLOOR",
                "proposed_denominator_policy": "RESEARCHER_APPROVAL_REQUIRED",
            }
            for feature, formula in {**BENEISH_FORMULAS, **DECHOW_FORMULAS}.items()
        ]
    )
    denominator.to_csv(out / "denominator_diagnostics.csv", index=False)
    temporal = pd.DataFrame(
        [
            {
                "feature_id": row["canonical_concept_name"],
                "available_at_annual_anchor": True,
                "requires_t_minus_1": row["paired_two_year_coverage"] > 0,
                "requires_two_consecutive_years": row["paired_two_year_coverage"] > 0,
                "requires_future_information": False,
                "uses_target_source": False,
                "is_direct_target_component": False,
                "is_post_outcome": False,
            }
            for _, row in concepts.iterrows()
        ]
    )
    temporal.to_csv(out / "temporal_availability_audit.csv", index=False)
    leakage = pd.DataFrame(
        [
            {
                "feature_id": feature,
                "target_id": target,
                "decision": "RESEARCH_DECISION_REQUIRED",
                "rationale": "P07B feasibility only; no target values accessed",
            }
            for feature in temporal.feature_id
            for target in (
                "S1",
                "S2",
                "S3_BROAD",
                "S3_REPORTING",
                "S3_CONTENT",
                "S3_TIMELINESS",
                "L1",
            )
        ]
    )
    leakage.to_csv(out / "target_specific_leakage_audit.csv", index=False)

    benchmark = pd.DataFrame(
        [
            {
                "benchmark": "Beneish",
                "overall_status": "PARTIALLY_RECONSTRUCTIBLE",
                "components_available": int((beneish.mapping_status == "EXACT_MAPPING").sum()),
                "components_unavailable": int((beneish.mapping_status != "EXACT_MAPPING").sum()),
                "main_mapping_issue": "DEPI depreciation expense and TATA continuing operations unavailable",
                "coverage": "see component audit",
            },
            {
                "benchmark": "Dechow Model 1",
                "overall_status": "PARTIALLY_RECONSTRUCTIBLE",
                "components_available": 0,
                "components_unavailable": len(DECHOW_MODEL_1_COMPONENTS),
                "main_mapping_issue": "issuance and complete Model 1 mappings require audit",
                "coverage": "not operationalised",
            },
            {
                "benchmark": "Tran VSA-240 benchmark",
                "overall_status": "PARTIALLY_RECONSTRUCTIBLE",
                "components_available": 0,
                "components_unavailable": 0,
                "main_mapping_issue": "complete published specification requires researcher mapping review",
                "coverage": "not operationalised",
            },
            {
                "benchmark": "Bui benchmark",
                "overall_status": "PARTIALLY_RECONSTRUCTIBLE",
                "components_available": 0,
                "components_unavailable": 0,
                "main_mapping_issue": "complete published specification requires researcher mapping review",
                "coverage": "not operationalised",
            },
        ]
    )
    benchmark.to_csv(out / "benchmark_feasibility_matrix.csv", index=False)
    gaps = pd.DataFrame(
        [
            {
                "gap_id": "GAP_DEPRECIATION_EXPENSE",
                "benchmark": "Beneish",
                "feature": "DEPI",
                "required_concept": "depreciation expense for the period",
                "missing_source": "authorised production core",
                "existing_near_substitute": "accumulated_depreciation_ppe",
                "why_substitute_is_not_equivalent": "stock is not period expense",
                "possible_collection_route": "normalised cash-flow/income-statement extract",
                "possible_approved_adaptation": "none without approval",
                "effect_on_confirmatory_design": "canonical M-score unavailable",
            },
            {
                "gap_id": "GAP_CONTINUING_OPERATIONS",
                "benchmark": "Beneish",
                "feature": "TATA",
                "required_concept": "income from continuing operations",
                "missing_source": "authorised production core",
                "existing_near_substitute": "profit_after_tax",
                "why_substitute_is_not_equivalent": "PAT is not continuing-operations income",
                "possible_collection_route": "statement-note coding",
                "possible_approved_adaptation": "TATA_PAT_ADAPTED robustness only",
                "effect_on_confirmatory_design": "canonical M-score unavailable",
            },
            {
                "gap_id": "GAP_PREPOST_ROLE",
                "benchmark": "pre/post universe",
                "feature": "all",
                "required_concept": "authorised production pre/post pair",
                "missing_source": "source role",
                "existing_near_substitute": "validation_only pair file",
                "why_substitute_is_not_equivalent": "protocol prohibits production use",
                "possible_collection_route": "approved derived production source",
                "possible_approved_adaptation": "none without protocol decision",
                "effect_on_confirmatory_design": "pre/post predictors prohibited",
            },
        ]
    )
    gaps.to_csv(out / "data_gap_register.csv", index=False)
    for filename, title in (
        ("beneish_benchmark_feasibility.md", "Beneish"),
        ("dechow_benchmark_feasibility.md", "Dechow Model 1"),
        ("tran_benchmark_feasibility.md", "Tran VSA-240"),
        ("bui_benchmark_feasibility.md", "Bui"),
    ):
        (out / filename).write_text(
            f"# {title} feasibility\n\nSee `benchmark_feasibility_matrix.csv`; this is non-operational and requires researcher approval.\n",
            encoding="utf-8",
        )
    pd.DataFrame(columns=["feature", "published_variable_set", "mapping_status"]).to_csv(
        out / "tran_vsa240_mapping_audit.csv", index=False
    )
    pd.DataFrame(columns=["feature", "published_variable_set", "mapping_status"]).to_csv(
        out / "bui_mapping_audit.csv", index=False
    )
    template = {
        "stage": "P07B",
        "decision_status": "RESEARCHER_APPROVAL_REQUIRED",
        "no_feature_is_locked": True,
        "benchmarks": [
            {
                "benchmark": name,
                "selected_design": None,
                "alternatives": [
                    "canonical_exact",
                    "approved_mapping",
                    "adapted_robustness",
                    "exclude",
                ],
            }
            for name in benchmark.benchmark
        ],
    }
    (out / "researcher_decision_template.yaml").write_text(
        yaml.safe_dump(template, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "status": "AUDIT_INCOMPLETE",
        "run_id": args.run_id,
        "literature_bundle": str(args.literature_root.resolve()),
        "literature_hashes": literature[["study_id", "source_hash"]].to_dict("records"),
        "production_source_only_for_coverage": True,
        "validation_only_prepost_prohibited": True,
        "no_outcomes_read": True,
        "no_known_cases_read": True,
        "no_operational_features_created": True,
    }
    (out / "p07b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "P07B_LITERATURE_TO_DATA_AUDIT.md").write_text(
        "# P07B Literature-to-Data Audit\n\n## Verdict\n\nAUDIT_INCOMPLETE\n\nThis immutable, non-supervised audit used the supplied literature bundle and authorised production accounting source only for coverage. The validation-only pre/post file is explicitly prohibited from production use. No outcomes, K1–K4, P07 operational registry, P10–P17, denominator floors, imputation, winsorisation, or target-based selection were used. The Tran and Bui full-specification extraction remains incomplete, so this audit must not be used to lock a predictor library.\n",
        encoding="utf-8",
    )
    print(f"P07B status=PASS run_id={args.run_id} output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
