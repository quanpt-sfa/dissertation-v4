"""P03 CLI for the single final firm-year Parquet input."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.evidence_registry import LogicalEvidenceSource, logical_evidence_sources
from core.pipeline import load_run, mapping, physical_columns
from core.semantic_keys import FIRM_ID, FISCAL_YEAR, PREDICTION_TIME, SOURCE_ID
from evidence.annual import build_audit_adjustment_records, build_audit_opinion_records
from evidence.final_firm_year import (
    build_wide_adjustment_rows,
    build_wide_opinion_rows,
    build_wide_s3_records,
)
from evidence.service import build_evidence_ledger
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
        step_id="P03",
        state="PANELLED",
    )
    if args.dry_run:
        print("P03 dry-run: one final firm-year Parquet will supply S1, S2, and S3")
        return 0

    panel = loaded.context.read("firm_year_panel", {})
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("firm_year_panel must be a DataFrame")
    columns = physical_columns(loaded.registry)
    panel_anchors = _panel_anchors(panel, columns)
    entity = EntityResolutionSpec.from_mapping(loaded.registry.get("entity_resolution"))
    logical = logical_evidence_sources(loaded.registry)
    by_physical: dict[str, list[LogicalEvidenceSource]] = {}
    for source in logical.values():
        by_physical.setdefault(source.physical_source_id, []).append(source)

    s1_id, s1_source = _physical_source(loaded.registry, "audit_adjustment")
    s2_id, s2_source = _physical_source(loaded.registry, "audit_opinion")
    s3_id, s3_source = _physical_source(loaded.registry, "sanction_calendar_year")
    for source_id in (s1_id, s2_id, s3_id):
        audit = mapping(
            loaded.context.read("raw_audit", {SOURCE_ID: source_id}),
            f"raw_audit source={source_id}",
        )
        decision = mapping(audit.get("decision"), f"raw_audit source={source_id}.decision")
        if decision.get("pipeline_may_advance") is not True:
            raise ValueError(f"source={source_id}: passing P01 audit required")

    s1_spec, s1_path = resolve_source(loaded.registry, s1_id)
    s2_spec, s2_path = resolve_source(loaded.registry, s2_id)
    s3_spec, s3_path = resolve_source(loaded.registry, s3_id)
    s1_semantics = mapping(s1_source.get("resolved_semantics"), "S1 resolved semantics")
    s2_semantics = mapping(s2_source.get("resolved_semantics"), "S2 resolved semantics")
    s3_semantics = mapping(s3_source.get("resolved_semantics"), "S3 resolved semantics")

    s1_logical = sorted(by_physical.get(s1_id, []), key=lambda item: item.source_id)
    s2_logical = sorted(by_physical.get(s2_id, []), key=lambda item: item.source_id)
    s3_logical = sorted(by_physical.get(s3_id, []), key=lambda item: item.source_id)
    if len(s1_logical) != 2:
        raise ValueError("single-input P03 requires two S1 logical endpoints")
    if len(s2_logical) != 1:
        raise ValueError("single-input P03 requires one S2 logical endpoint")
    if len(s3_logical) != 4:
        raise ValueError("single-input P03 requires four S3 logical endpoints")

    s1_build = build_audit_adjustment_records(
        panel_anchors=panel_anchors,
        rows=build_wide_adjustment_rows(
            source_id=s1_id,
            path=s1_path,
            spec=s1_spec,
            semantics=s1_semantics,
            entity=entity,
            logical_sources=s1_logical,
        ),
        sources=s1_logical,
    )
    s2_build = build_audit_opinion_records(
        panel_anchors=panel_anchors,
        rows=build_wide_opinion_rows(
            source_id=s2_id,
            path=s2_path,
            spec=s2_spec,
            semantics=s2_semantics,
            entity=entity,
            logical_source=s2_logical[0],
        ),
        source=s2_logical[0],
    )
    s3_build = build_wide_s3_records(
        source_id=s3_id,
        path=s3_path,
        spec=s3_spec,
        semantics=s3_semantics,
        entity=entity,
        logical_sources=s3_logical,
        panel_anchors=panel_anchors,
    )

    evidence = mapping(loaded.registry.get("evidence"), "evidence")
    tolerance = evidence.get("lag_identity_tolerance_days")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("evidence.lag_identity_tolerance_days must be nonnegative")
    result = build_evidence_ledger(
        panel=panel,
        records=[*s1_build.records, *s2_build.records, *s3_build.endpoint_records],
        columns=columns,
        fiscal_year_end_month_day=entity.reporting_calendar.default_fiscal_year_end_month_day,
        lag_tolerance_days=tolerance,
    )
    if args.validate_only:
        return 0

    loaded.context.write("evidence_ledger", result.ledger, {})
    loaded.context.write("sanction_decision_ledger", s3_build.decision_ledger, {})
    loaded.context.write("availability_registry", result.availability_registry, {})
    loaded.context.write("lag_decomposition", result.lag_decomposition, {})
    loaded.context.write(
        "annual_evidence_audit",
        {
            "input_mode": "single_final_firm_year_parquet",
            "physical_files_read": 1,
            "sources": [s1_build.audit, s2_build.audit],
            "s1_pair_failures": s1_build.failures,
            "s2_normalization_exceptions": s2_build.failures,
            "annual_measurement_records_are_events": False,
            "annual_anchor_equals_prediction_time": True,
            "missing_is_negative": False,
            "sanction_calendar_year": s3_build.audit,
        },
        {},
    )
    print(f"P03 status=PASS evidence_rows={len(result.ledger)} input_files=1")
    return 0


def _physical_source(
    registry: dict[str, Any], processor: str
) -> tuple[str, dict[str, Any]]:
    sources = mapping(
        mapping(
            mapping(registry.get("data_sources"), "data_sources").get("source_registry"),
            "source_registry",
        ).get("sources"),
        "source_registry.sources",
    )
    matches: list[tuple[str, dict[str, Any]]] = []
    for source_id, raw in sources.items():
        source = mapping(raw, f"source={source_id}")
        if source.get("enabled") is not True or source.get("role") != "evidence":
            continue
        evidence_mapping = mapping(
            source.get("evidence_mapping"), f"source={source_id}.evidence_mapping"
        )
        if evidence_mapping.get("processor") == processor:
            matches.append((str(source_id), cast(dict[str, Any], source)))
    if len(matches) != 1:
        raise ValueError(f"processor={processor}: exactly one physical source required")
    return matches[0]


def _panel_anchors(
    panel: pd.DataFrame, columns: dict[str, str]
) -> dict[tuple[str, int], datetime]:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    anchors: dict[tuple[str, int], datetime] = {}
    for row in cast(list[dict[str, object]], panel.loc[:, [firm, year, prediction]].to_dict("records")):
        key = (str(row[firm]), int(row[year]))
        if key in anchors:
            raise ValueError(f"P03 duplicate panel anchor={key}")
        anchors[key] = pd.Timestamp(row[prediction]).to_pydatetime()
    return anchors


if __name__ == "__main__":
    raise SystemExit(main())
