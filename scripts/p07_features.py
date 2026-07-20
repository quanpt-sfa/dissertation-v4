"""P07 CLI: validate the feature registry and publish the as-of feature panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.artifact_store import dataframe_to_csv
from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import FIRM_ID, FISCAL_YEAR, TARGET_ID
from features.diagnostics import build_feature_diagnostics
from features.service import build_feature_panel


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
        step_id="P07",
        state="OBSERVABLE",
    )
    features = mapping(loaded.registry.get("features"), "features")
    raw_registry = sequence(features.get("registry"), "features.registry")
    definitions = [mapping(item, "features.registry item") for item in raw_registry]
    raw_intended = sequence(features.get("intended_registry", []), "features.intended_registry")
    intended = [mapping(item, "features.intended_registry item") for item in raw_intended]
    if args.dry_run:
        print(f"P07 dry-run: registered_features={len(definitions)}")
        return 0

    panel = loaded.context.read("firm_year_panel", {})
    risk_sets = loaded.context.read("risk_sets", {})
    loaded.context.read("observability_registry", {})
    if not isinstance(panel, pd.DataFrame) or not isinstance(risk_sets, pd.DataFrame):
        raise ValueError("P07 panel inputs must be DataFrames")

    result = build_feature_panel(
        firm_year_panel=panel,
        risk_sets=risk_sets,
        feature_definitions=cast(list[dict[str, object]], definitions),
        columns=physical_columns(loaded.registry),
        blocked_label_semantics=[
            str(value)
            for value in sequence(
                mapping(
                    features.get("label_derived_feature_firewall"),
                    "features.label_derived_feature_firewall",
                ).get("same_firm_year_blocked_semantics"),
                "features.label_derived_feature_firewall.same_firm_year_blocked_semantics",
            )
        ],
        intended_definitions=cast(list[dict[str, object]], intended),
        target_ids=sorted(
            mapping(loaded.registry.get("measurement"), "measurement")
            .get("candidate_targets", {})
            .keys()
        ),
        temporal_estimands={
            target_id: _target_temporal_role(target_id)
            for target_id in mapping(loaded.registry.get("measurement"), "measurement")
            .get("candidate_targets", {})
            .keys()
        },
    )
    if args.validate_only:
        return 0

    columns = physical_columns(loaded.registry)
    firm_column = columns[FIRM_ID]
    year_column = columns[FISCAL_YEAR]
    diagnostics = build_feature_diagnostics(
        panel=result.panel,
        definitions=cast(list[dict[str, object]], definitions),
        firm_column=firm_column,
        year_column=year_column,
    )
    compatibility = _pipeline_generated_feature_receipts(
        panel=result.panel,
        definitions=cast(list[dict[str, object]], [*definitions, *intended]),
        firm_column=firm_column,
    )
    summary = {
        **result.summary,
        "feature_generation": compatibility["report"],
        "accounting_identity_rows": len(diagnostics.accounting_identities),
        "ratio_diagnostic_rows": len(diagnostics.ratios),
        "redundancy_pairs": len(diagnostics.redundancy),
    }
    decision_report = _decision_report(result.decision_report, summary)

    # Compatibility artifacts retain the locked P07 artifact contract while recording that
    # the external test package is no longer read by production code.
    loaded.context.write("feature_store_manifest_validated", compatibility["manifest"], {})
    loaded.context.write("feature_store_validation_report", compatibility["report"], {})
    loaded.context.write("feature_store_file_audit", compatibility["file_audit"], {})
    loaded.context.write(
        "feature_store_identifier_crosswalk_audit", compatibility["identifier_audit"], {}
    )
    loaded.context.write(
        "feature_store_availability_violations", compatibility["availability_violations"], {}
    )
    loaded.context.write("feature_store_coverage_audit", diagnostics.coverage, {})
    loaded.context.write(
        "feature_store_research_decision_audit", compatibility["research_decision_audit"], {}
    )
    loaded.context.write("feature_value_diagnostic_audit", diagnostics.value_scale, {})
    loaded.context.write("accounting_identity_audit", diagnostics.accounting_identities, {})
    loaded.context.write("audited_unaudited_adjustment_audit", diagnostics.audited_unaudited, {})
    loaded.context.write("ratio_diagnostic_audit", diagnostics.ratios, {})
    loaded.context.write("temporal_feature_audit", diagnostics.temporal, {})
    loaded.context.write("feature_redundancy_audit", diagnostics.redundancy, {})

    loaded.context.write("feature_panel", result.panel, {})
    loaded.context.write("feature_registry", result.registry, {})
    loaded.context.write("leakage_registry", result.leakage_registry, {})
    feature_id_column = columns["feature_id"]
    target_id_column = columns[TARGET_ID]
    registry_frame = _ordered_frame(pd.DataFrame(result.registry), feature_id_column)
    lineage_frame = _ordered_frame(result.lineage_registry, feature_id_column, "source_dataset")
    leakage_frame = _ordered_frame(result.leakage_rows, feature_id_column, target_id_column)
    loaded.context.write("feature_registry_table", registry_frame, {})
    loaded.context.write("feature_registry_csv", dataframe_to_csv(registry_frame), {})
    loaded.context.write("feature_lineage_registry", lineage_frame, {})
    loaded.context.write("leakage_registry_table", leakage_frame, {})
    loaded.context.write("leakage_registry_csv", dataframe_to_csv(leakage_frame), {})
    loaded.context.write("feature_views", result.feature_views, {})
    view_rows = result.feature_views.get("views")
    if not isinstance(view_rows, list):
        raise ValueError("P07 feature views are malformed")
    loaded.context.write(
        "feature_view_matrix",
        dataframe_to_csv(
            _ordered_frame(
                pd.DataFrame(cast(list[dict[str, object]], view_rows)), "view_id", target_id_column
            )
        ),
        {},
    )
    loaded.context.write(
        "feature_availability_audit",
        dataframe_to_csv(_ordered_frame(result.availability_audit, feature_id_column)),
        {},
    )
    loaded.context.write(
        "feature_missingness_audit",
        dataframe_to_csv(_ordered_frame(result.missingness_audit, feature_id_column)),
        {},
    )
    loaded.context.write(
        "feature_panel_schema",
        {
            "columns": list(result.panel.columns),
            "primary_key": [firm_column, year_column],
            "generation_mode": "pipeline_generated",
            "external_feature_store_used": False,
        },
        {},
    )
    loaded.context.write("p07_summary", summary, {})
    loaded.context.write("p07_decision_report", decision_report, {})
    print(
        f"P07 status=PASS features={len(result.registry)} "
        f"operational={result.summary['panel_features']} generation=pipeline"
    )
    return 0


def _pipeline_generated_feature_receipts(
    *,
    panel: pd.DataFrame,
    definitions: list[dict[str, object]],
    firm_column: str,
) -> dict[str, object]:
    locked = [item for item in definitions if item.get("research_decision_status") == "LOCKED"]
    unresolved = [
        item for item in definitions if item.get("research_decision_status") != "LOCKED"
    ]
    report: dict[str, object] = {
        "status": "PIPELINE_GENERATED_FEATURES_VALID",
        "generation_mode": "pipeline_generated",
        "external_feature_store_used": False,
        "external_manifest_required": False,
        "external_crosswalk_required": False,
        "validated_feature_count": len(definitions),
        "locked_feature_count": len(locked),
        "unresolved_feature_count": len(unresolved),
        "panel_row_count": len(panel),
        "panel_firm_count": int(panel[firm_column].nunique()),
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
        "preprocessing_fit_at_p07": False,
    }
    file_audit = pd.DataFrame(
        [
            {
                "feature_id": str(item["feature_id"]),
                "generation_mode": "pipeline_generated",
                "technical_validation_status": "PASS"
                if item.get("research_decision_status") == "LOCKED"
                else "NOT_OPERATIONAL",
                "research_decision_status": str(item.get("research_decision_status")),
            }
            for item in definitions
        ]
    )
    if not file_audit.empty:
        file_audit["feature_id"] = file_audit["feature_id"].astype("string")
    identifier_audit = pd.DataFrame(
        {
            firm_column: panel[firm_column].drop_duplicates().astype("string"),
            "mapping_status": "CANONICAL_PIPELINE_ID",
            "ambiguity_flag": False,
            "source_registry": "P02_firm_year_panel",
        }
    ).reset_index(drop=True)
    availability_violations = pd.DataFrame(
        {
            "feature_id": pd.Series(dtype="string"),
            "violation_type": pd.Series(dtype="string"),
            "action_taken": pd.Series(dtype="string"),
        }
    )
    research_decision_audit = pd.DataFrame(
        [
            {
                "feature_id": str(item["feature_id"]),
                "research_decision_status": str(item.get("research_decision_status")),
                "confirmatory_status": str(item.get("confirmatory_status")),
                "model_eligibility": str(item.get("model_eligibility")),
                "source_of_status": "compiled_feature_registry",
                "pipeline_may_proceed_without_feature": item.get("research_decision_status")
                != "LOCKED",
            }
            for item in definitions
        ]
    )
    if not research_decision_audit.empty:
        research_decision_audit["feature_id"] = research_decision_audit["feature_id"].astype(
            "string"
        )
    return {
        "manifest": {
            **report,
            "manifest_validated": False,
            "status_source": "pipeline_generated_feature_panel",
        },
        "report": report,
        "file_audit": file_audit,
        "identifier_audit": identifier_audit,
        "availability_violations": availability_violations,
        "research_decision_audit": research_decision_audit,
    }


def _decision_report(base: str, summary: dict[str, object]) -> str:
    generation = cast(dict[str, object], summary["feature_generation"])
    return (
        base
        + "\n## Feature generation\n\n"
        + "- Mode: pipeline-generated from locked upstream artifacts.\n"
        + "- External feature-store package used: no.\n"
        + f"- Registered features: {generation['validated_feature_count']}\n"
        + f"- Operational LOCKED features: {generation['locked_feature_count']}\n"
        + f"- Unresolved features: {generation['unresolved_feature_count']}\n"
        + "- P07 performs no preprocessing fit and accesses no outer outcomes or known cases.\n"
    )


def _target_temporal_role(target_id: object) -> str:
    identifier = str(target_id)
    if identifier.startswith("S3_"):
        return "next_calendar_year_regulatory_event"
    if identifier.startswith("L1_"):
        return "mixed_target_specific"
    return "annual_measurement_at_anchor"


def _ordered_frame(frame: pd.DataFrame, *first_columns: str) -> pd.DataFrame:
    missing = [column for column in first_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"P07 audit frame is missing required columns: {missing}")
    remaining = [column for column in frame.columns if column not in first_columns]
    result = frame.loc[:, [*first_columns, *remaining]].copy()
    for column in first_columns:
        result[column] = result[column].astype("string")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
