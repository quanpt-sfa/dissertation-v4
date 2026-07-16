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
    loaded.context.write("p07_summary", result.summary, {})
    loaded.context.write("p07_decision_report", result.decision_report, {})
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
