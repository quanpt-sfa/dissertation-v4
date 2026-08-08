"""P07 CLI: validate the feature registry and publish the as-of feature panel."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.artifact_store import dataframe_to_csv
from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import FIRM_ID, FISCAL_YEAR, PREDICTION_TIME, TARGET_ID
from features.diagnostics import build_feature_diagnostics
from features.domain_metadata import restore_domain_metadata
from features.panel_source import assemble_from_raw
from features.reporting import build_raw_computation_decision_report
from features.service import build_feature_panel
from p01.registry import resolve_source
from p02.models import EntityResolutionSpec


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
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    domain_config = mapping(evaluation.get("domain_transfer"), "evaluation.domain_transfer")
    domain_bindings = [
        mapping(item, "evaluation.domain_transfer.bindings item")
        for item in sequence(
            domain_config.get("bindings"),
            "evaluation.domain_transfer.bindings",
        )
    ]
    domain_columns = [str(binding["column"]) for binding in domain_bindings]
    if args.dry_run:
        print(
            f"P07 dry-run: registered_features={len(definitions)} "
            f"domain_columns={domain_columns}"
        )
        return 0
    panel = loaded.context.read("firm_year_panel", {})
    risk_sets = loaded.context.read("risk_sets", {})
    loaded.context.read("observability_registry", {})
    if not isinstance(panel, pd.DataFrame) or not isinstance(risk_sets, pd.DataFrame):
        raise ValueError("P07 panel inputs must be DataFrames")
    columns = physical_columns(loaded.registry)
    entity_spec = EntityResolutionSpec.from_mapping(loaded.registry.get("entity_resolution"))
    source_spec, source_path = resolve_source(
        cast(dict[str, object], loaded.registry), "financial_statement_core_long"
    )
    store_result = assemble_from_raw(
        base_panel=panel,
        feature_definitions=cast(list[dict[str, object]], definitions),
        intended_definitions=cast(list[dict[str, object]], intended),
        entity_spec=entity_spec,
        raw_source_path=source_path,
        reader={"encoding": source_spec.encoding},
        firm_column=columns[FIRM_ID],
        year_column=columns[FISCAL_YEAR],
        prediction_time_column=columns[PREDICTION_TIME],
    )
    result = build_feature_panel(
        firm_year_panel=store_result.panel,
        risk_sets=risk_sets,
        feature_definitions=cast(list[dict[str, object]], definitions),
        columns=columns,
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
    restored_panel = restore_domain_metadata(
        feature_panel=result.panel,
        source_panel=store_result.panel,
        domain_columns=domain_columns,
        firm_column=columns[FIRM_ID],
        year_column=columns[FISCAL_YEAR],
    )
    result = replace(result, panel=restored_panel)
    if args.validate_only:
        return 0
    diagnostics = build_feature_diagnostics(
        panel=result.panel,
        definitions=cast(list[dict[str, object]], definitions),
        firm_column=columns[FIRM_ID],
        year_column=columns[FISCAL_YEAR],
    )
    store_validation_report = {
        **store_result.validation_report,
        "stage_status": result.summary["status"],
        "restored_domain_metadata_columns": domain_columns,
    }
    summary = {
        **result.summary,
        "feature_store": store_validation_report,
        "identifier_mapping": {
            "store_identifiers": int(
                store_result.identifier_audit["feature_store_firm_id"].nunique()
            ),
            "matched_identifiers": int(
                store_result.identifier_audit.loc[
                    store_result.identifier_audit["mapping_status"] == "MATCHED",
                    "feature_store_firm_id",
                ].nunique()
            ),
            "unmatched_identifiers": int(
                store_result.identifier_audit.loc[
                    store_result.identifier_audit["mapping_status"] != "MATCHED",
                    "feature_store_firm_id",
                ].nunique()
            ),
            "ambiguous_identifiers": int(store_result.identifier_audit["ambiguity_flag"].sum()),
        },
        "accounting_identity_rows": len(diagnostics.accounting_identities),
        "ratio_diagnostic_rows": len(diagnostics.ratios),
        "redundancy_pairs": len(diagnostics.redundancy),
    }
    decision_report = build_raw_computation_decision_report(result.decision_report, summary)
    loaded.context.write(
        "feature_store_manifest_validated",
        {
            **store_validation_report,
            "manifest_validated": True,
            "status_source": "raw_registry_formula_and_locked_source_snapshot",
        },
        {},
    )
    loaded.context.write("feature_store_validation_report", store_validation_report, {})
    loaded.context.write("feature_store_file_audit", store_result.file_audit, {})
    loaded.context.write(
        "feature_store_identifier_crosswalk_audit", store_result.identifier_audit, {}
    )
    loaded.context.write(
        "feature_store_availability_violations",
        store_result.availability_violations,
        {},
    )
    loaded.context.write("feature_store_coverage_audit", diagnostics.coverage, {})
    loaded.context.write(
        "feature_store_research_decision_audit",
        store_result.research_decision_audit,
        {},
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
    feature_id_column = physical_columns(loaded.registry)["feature_id"]
    target_id_column = physical_columns(loaded.registry)[TARGET_ID]
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
            "primary_key": [
                physical_columns(loaded.registry)[FIRM_ID],
                physical_columns(loaded.registry)[FISCAL_YEAR],
            ],
        },
        {},
    )
    loaded.context.write("p07_summary", summary, {})
    loaded.context.write("p07_decision_report", decision_report, {})
    print(
        f"P07 status=PASS features={len(result.registry)} operational={result.summary['panel_features']}"
    )
    return 0


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
