"""P03 CLI: build the deduplicated evidence ledger and lag audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.evidence_registry import LogicalEvidenceSource, logical_evidence_sources
from core.pipeline import load_run, mapping, physical_columns
from core.runtime import RunContext
from core.semantic_keys import (
    AFFECTED_FISCAL_YEAR,
    AUDIT_INDICATOR,
    AUDIT_OPINION,
    AUDIT_STATUS,
    AVAILABILITY_DATE,
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    DECISION_DATE,
    DECISION_NUMBER,
    DECISION_TOTAL_FINE,
    DOCUMENT_ID,
    EVENT_CLUSTER_ID,
    EVENT_ID,
    FIRM_ID,
    FISCAL_YEAR,
    HARD_POSITIVE,
    HAS_FINE,
    HAS_REMEDY,
    HAS_SUSPENSION,
    HAS_WARNING,
    ITEM_ID,
    LABEL_KNOWN_DATE,
    LEGACY_EVENT_ID,
    NORMALIZED_VIOLATION_CODE,
    OUTCOME,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PERIOD_TYPE,
    PREDICTION_TIME,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    PUBLISH_DATE,
    ROW_INCLUSION,
    SANCTION_YEAR,
    SOURCE_COLUMN,
    SOURCE_FILE,
    SOURCE_ID,
    SOURCE_ROW,
    STATEMENT_FAMILY,
    STATEMENT_SCOPE,
    UNIT,
    VALUE,
)
from evidence.annual import (
    AdjustmentRow,
    AnnualEvidenceBuild,
    OpinionRow,
    build_audit_adjustment_records,
    build_audit_opinion_records,
)
from evidence.sanctions import SanctionDecisionInput, build_s3_evidence
from evidence.service import EvidenceRecord, build_evidence_ledger, map_evidence_outcome
from p01.readers import iter_rows
from p01.registry import resolve_source
from p02.builder import normalize_entity_field, resolve_entity_link
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
        print("P03 dry-run: evidence sources will be resolved from the locked registry")
        return 0
    panel = loaded.context.read("firm_year_panel", {})
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("firm_year_panel must be a DataFrame")
    columns = physical_columns(loaded.registry)
    panel_anchors = _panel_anchors(panel, columns)
    records, annual_builds, sanction_ledger, sanction_audit = _records(
        loaded.registry,
        loaded.context,
        panel_anchors=panel_anchors,
        columns=columns,
    )
    entity = EntityResolutionSpec.from_mapping(loaded.registry.get("entity_resolution"))
    evidence = mapping(loaded.registry.get("evidence"), "evidence")
    tolerance = evidence.get("lag_identity_tolerance_days")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("evidence.lag_identity_tolerance_days must be nonnegative")
    result = build_evidence_ledger(
        panel=panel,
        records=records,
        columns=columns,
        fiscal_year_end_month_day=entity.reporting_calendar.default_fiscal_year_end_month_day,
        lag_tolerance_days=tolerance,
    )
    if args.validate_only:
        return 0
    loaded.context.write("evidence_ledger", result.ledger, {})
    loaded.context.write("sanction_decision_ledger", sanction_ledger, {})
    loaded.context.write("availability_registry", result.availability_registry, {})
    loaded.context.write("lag_decomposition", result.lag_decomposition, {})
    loaded.context.write(
        "annual_evidence_audit",
        {
            "sources": [build.audit for build in annual_builds],
            "s1_pair_failures": [
                failure
                for build in annual_builds
                if build.audit.get("processor") == "audit_adjustment"
                for failure in build.failures
            ],
            "s2_normalization_exceptions": [
                failure
                for build in annual_builds
                if build.audit.get("processor") == "audit_opinion"
                for failure in build.failures
            ],
            "annual_measurement_records_are_events": False,
            "annual_anchor_equals_prediction_time": True,
            "missing_is_negative": False,
            "sanction_calendar_year": sanction_audit,
        },
        {},
    )
    print(f"P03 status=PASS evidence_rows={len(result.ledger)}")
    return 0


def _records(
    registry: dict[str, Any],
    context: RunContext,
    *,
    panel_anchors: dict[tuple[str, int], datetime],
    columns: dict[str, str],
) -> tuple[list[EvidenceRecord], list[AnnualEvidenceBuild], pd.DataFrame, dict[str, object]]:
    data_sources = mapping(registry.get("data_sources"), "data_sources")
    source_registry = mapping(data_sources.get("source_registry"), "source_registry")
    sources = mapping(source_registry.get("sources"), "source_registry.sources")
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    logical_sources = logical_evidence_sources(registry)
    logical_by_physical: dict[str, list[LogicalEvidenceSource]] = {}
    for logical in logical_sources.values():
        logical_by_physical.setdefault(logical.physical_source_id, []).append(logical)
    result: list[EvidenceRecord] = []
    annual_builds: list[AnnualEvidenceBuild] = []
    sanction_ledgers: list[pd.DataFrame] = []
    sanction_audits: list[dict[str, object]] = []
    for source_id, raw in sorted(sources.items()):
        source = mapping(raw, f"source={source_id}")
        if source.get("role") != "evidence" or source.get("enabled") is not True:
            continue
        logical = sorted(logical_by_physical.get(source_id, []), key=lambda item: item.source_id)
        if not logical:
            raise ValueError(f"source={source_id}: logical evidence sources required")
        audit = mapping(context.read("raw_audit", {SOURCE_ID: source_id}), "raw_audit")
        if mapping(audit.get("decision"), "audit.decision").get("pipeline_may_advance") is not True:
            raise ValueError(f"source={source_id}: passing P01 audit required")
        spec, path = resolve_source(registry, source_id)
        semantics = mapping(source.get("resolved_semantics"), f"source={source_id}.semantics")
        evidence_mapping = mapping(
            source.get("evidence_mapping"), f"source={source_id}.evidence_mapping"
        )
        processor = evidence_mapping.get("processor")
        if processor == "audit_adjustment":
            build = build_audit_adjustment_records(
                panel_anchors=panel_anchors,
                rows=_adjustment_rows(
                    source_id=source_id,
                    path=path,
                    spec=spec,
                    semantics=semantics,
                    entity=entity,
                    logical_sources=logical,
                ),
                sources=logical,
            )
            result.extend(build.records)
            annual_builds.append(build)
            continue
        if processor == "audit_opinion":
            if len(logical) != 1:
                raise ValueError(f"source={source_id}: audit opinion requires one logical source")
            build = build_audit_opinion_records(
                panel_anchors=panel_anchors,
                rows=_opinion_rows(
                    source_id=source_id,
                    path=path,
                    spec=spec,
                    semantics=semantics,
                    entity=entity,
                ),
                source=logical[0],
            )
            result.extend(build.records)
            annual_builds.append(build)
            continue
        if processor == "sanction_calendar_year":
            taxonomy = mapping(registry.get("s3_taxonomy"), "s3_taxonomy")
            completeness = mapping(
                taxonomy.get("sanction_source_completeness"),
                "s3_taxonomy.sanction_source_completeness",
            )
            build = build_s3_evidence(
                panel_keys=set(panel_anchors),
                decisions=_sanction_rows(
                    source_id=source_id,
                    path=path,
                    spec=spec,
                    semantics=semantics,
                    entity=entity,
                ),
                taxonomy=taxonomy,
                complete_through_year=int(completeness["complete_through_year"]),
                incomplete_years={
                    int(str(value))
                    for value in cast(list[object], completeness.get("incomplete_years", []))
                },
                columns=columns,
                source_profile_id=source_id,
            )
            result.extend(build.endpoint_records)
            sanction_ledgers.append(build.decision_ledger)
            sanction_audits.append(build.audit)
            continue
        if processor != "delayed_event" or len(logical) != 1:
            raise ValueError(f"source={source_id}: unsupported evidence processor")
        logical_source = logical[0]
        outcome_mode = evidence_mapping.get("outcome_mode")
        row_inclusion_semantic = evidence_mapping.get("row_inclusion_semantic")
        positive_semantic = evidence_mapping.get("positive_semantic")
        duplicate_rule = evidence_mapping.get("duplicate_representative_rule")
        if evidence_mapping.get("absence_policy") != "unknown":
            raise ValueError(f"source={source_id}: evidence absence must remain unknown")
        if duplicate_rule != "identical_signature_then_source_event_id":
            raise ValueError(f"source={source_id}: unsupported duplicate representative rule")
        required = {FIRM_ID, FISCAL_YEAR, AVAILABILITY_DATE, EVENT_ID}
        if isinstance(row_inclusion_semantic, str):
            required.add(row_inclusion_semantic)
        if outcome_mode == "direct_outcome":
            required.add(OUTCOME)
        elif outcome_mode == "positive_indicator" and isinstance(positive_semantic, str):
            if positive_semantic != HARD_POSITIVE:
                raise ValueError(f"source={source_id}: hard-positive semantic required")
            required.add(positive_semantic)
        else:
            raise ValueError(f"source={source_id}: unsupported evidence outcome mode")
        if evidence_mapping.get("false_indicator_policy") != "unknown":
            raise ValueError(f"source={source_id}: false positive-indicator must remain unknown")
        if not required.issubset(semantics):
            raise ValueError(
                f"source={source_id}: unresolved evidence semantics {sorted(required - set(semantics))}"
            )
        for row in iter_rows(path, spec):
            firm_raw = _required(row.get(str(semantics[FIRM_ID])), FIRM_ID)
            normalized = normalize_entity_field(str(firm_raw), entity)
            canonical, _ = resolve_entity_link(source_id, str(firm_raw), normalized, entity)
            fiscal_year = int(str(_required(row.get(str(semantics[FISCAL_YEAR])), FISCAL_YEAR)))
            availability = _datetime(
                _required(row.get(str(semantics[AVAILABILITY_DATE])), AVAILABILITY_DATE)
            )
            row_included = True
            if isinstance(row_inclusion_semantic, str):
                parsed_inclusion = _boolean(row.get(str(semantics[row_inclusion_semantic])))
                if parsed_inclusion is None:
                    raise ValueError(f"source={source_id}: row inclusion cannot be missing")
                row_included = parsed_inclusion
            outcome, outcome_basis = map_evidence_outcome(
                outcome_mode=str(outcome_mode),
                direct_outcome=(
                    _boolean(row.get(str(semantics[OUTCOME])))
                    if outcome_mode == "direct_outcome"
                    else None
                ),
                positive_indicator=(
                    _boolean(row.get(str(semantics[str(positive_semantic)])))
                    if outcome_mode == "positive_indicator"
                    else None
                ),
                row_included=row_included,
            )
            event_id = _optional_text(row, semantics, EVENT_ID) or _derived_event_id(
                source_id, canonical, fiscal_year, availability, row
            )
            cluster = _optional_text(row, semantics, EVENT_CLUSTER_ID) or event_id
            result.append(
                EvidenceRecord(
                    source_id=logical_source.source_id,
                    source_profile_id=source_id,
                    channel_id=logical_source.channel_id,
                    firm_id=canonical,
                    fiscal_year=fiscal_year,
                    availability_date=availability,
                    outcome=outcome,
                    event_id=event_id,
                    event_cluster_id=cluster,
                    period_link_source=_optional_text(row, semantics, PERIOD_LINK_SOURCE),
                    period_link_confidence=_optional_text(row, semantics, PERIOD_LINK_CONFIDENCE),
                    outcome_basis=outcome_basis,
                    row_included=row_included,
                    evidence_record_kind="delayed_event",
                    temporal_role=logical_source.temporal_role,
                    availability_basis=logical_source.availability_rule,
                    source_opportunity=None,
                    opportunity_basis="unknown_no_opportunity_indicator",
                    source_record_refs=event_id,
                    duplicate_representative_rule=str(duplicate_rule),
                )
            )
    if not result:
        raise ValueError("P03 requires at least one registered evidence record")
    if len(sanction_ledgers) != 1 or len(sanction_audits) != 1:
        raise ValueError("P03 requires exactly one sanction calendar-year source")
    return result, annual_builds, sanction_ledgers[0], sanction_audits[0]


def _panel_anchors(panel: pd.DataFrame, columns: dict[str, str]) -> dict[tuple[str, int], datetime]:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    anchors: dict[tuple[str, int], datetime] = {}
    for row in panel.loc[:, [firm, year, prediction]].to_dict(orient="records"):
        key = (str(row[firm]), int(row[year]))
        if key in anchors:
            raise ValueError(f"P03 duplicate panel anchor={key}")
        anchors[key] = _datetime(row[prediction])
    return anchors


def _sanction_rows(
    *,
    source_id: str,
    path: Path,
    spec: object,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
) -> list[SanctionDecisionInput]:
    from p01.models import SourceSpec

    if not isinstance(spec, SourceSpec):
        raise TypeError("locked source specification required")
    required = {FIRM_ID, DOCUMENT_ID, ROW_INCLUSION, HARD_POSITIVE}
    if not required.issubset(semantics):
        raise ValueError(
            f"source={source_id}: unresolved S3 semantics {sorted(required - set(semantics))}"
        )
    result: list[SanctionDecisionInput] = []
    for row in iter_rows(path, spec):
        firm_raw = _required(row.get(str(semantics[FIRM_ID])), FIRM_ID)
        normalized = normalize_entity_field(str(firm_raw), entity)
        canonical, _ = resolve_entity_link(source_id, str(firm_raw), normalized, entity)
        row_included = _boolean(row.get(str(semantics[ROW_INCLUSION])))
        hard_positive = _boolean(row.get(str(semantics[HARD_POSITIVE])))
        if row_included is None or hard_positive is None:
            raise ValueError(
                f"source={source_id}: S3 inclusion and hard-positive flags are required"
            )
        result.append(
            SanctionDecisionInput(
                document_id=str(_required(row.get(str(semantics[DOCUMENT_ID])), DOCUMENT_ID)),
                firm_id=canonical,
                decision_number=_optional_text(row, semantics, DECISION_NUMBER),
                sanction_year=_optional_int(row, semantics, SANCTION_YEAR),
                decision_date=_optional_datetime(row, semantics, DECISION_DATE),
                publish_date=_optional_datetime(row, semantics, PUBLISH_DATE),
                label_known_date=_optional_datetime(row, semantics, LABEL_KNOWN_DATE),
                affected_fiscal_year=_optional_int(row, semantics, AFFECTED_FISCAL_YEAR),
                primary_violation_l1=_optional_text(row, semantics, PRIMARY_VIOLATION_L1),
                primary_violation_l2=_optional_text(row, semantics, PRIMARY_VIOLATION_L2),
                construct_family=_optional_text(row, semantics, CONSTRUCT_FAMILY),
                construct_target=_optional_text(row, semantics, CONSTRUCT_TARGET),
                normalized_violation_code=_optional_text(row, semantics, NORMALIZED_VIOLATION_CODE),
                row_included=row_included,
                hard_positive=hard_positive,
                legacy_event_id=_optional_text(row, semantics, LEGACY_EVENT_ID),
                period_link_source=_optional_text(row, semantics, PERIOD_LINK_SOURCE),
                period_link_confidence=_optional_text(row, semantics, PERIOD_LINK_CONFIDENCE),
                total_fine=_optional_float(row, semantics, DECISION_TOTAL_FINE),
                has_fine=_optional_boolean(row, semantics, HAS_FINE),
                has_suspension=_optional_boolean(row, semantics, HAS_SUSPENSION),
                has_warning=_optional_boolean(row, semantics, HAS_WARNING),
                has_remedy=_optional_boolean(row, semantics, HAS_REMEDY),
                source_ref=_source_ref(row, semantics, SOURCE_FILE, SOURCE_ROW),
            )
        )
    return result


def _adjustment_rows(
    *,
    source_id: str,
    path: Path,
    spec: object,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    logical_sources: list[LogicalEvidenceSource],
) -> list[AdjustmentRow]:
    from p01.models import SourceSpec

    if not isinstance(spec, SourceSpec):
        raise TypeError("locked source specification required")
    required = {
        FIRM_ID,
        FISCAL_YEAR,
        AUDIT_STATUS,
        ITEM_ID,
        VALUE,
        UNIT,
        STATEMENT_SCOPE,
        STATEMENT_FAMILY,
    }
    if not required.issubset(semantics):
        raise ValueError(
            f"source={source_id}: unresolved S1 semantics {sorted(required - set(semantics))}"
        )
    endpoint_items = {str(source.logical_config["canonical_item"]) for source in logical_sources}
    result: list[AdjustmentRow] = []
    for row in iter_rows(path, spec):
        item = str(row.get(str(semantics[ITEM_ID]), "")).strip()
        if item not in endpoint_items:
            continue
        firm_raw = _required(row.get(str(semantics[FIRM_ID])), FIRM_ID)
        normalized = normalize_entity_field(str(firm_raw), entity)
        canonical, _ = resolve_entity_link(source_id, str(firm_raw), normalized, entity)
        result.append(
            AdjustmentRow(
                firm_id=canonical,
                fiscal_year=int(str(_required(row.get(str(semantics[FISCAL_YEAR])), FISCAL_YEAR))),
                audit_status=str(
                    _required(row.get(str(semantics[AUDIT_STATUS])), AUDIT_STATUS)
                ).strip(),
                canonical_item=item,
                value=_float_or_none(row.get(str(semantics[VALUE]))),
                unit=_text_or_none(row.get(str(semantics[UNIT]))),
                statement_scope=_text_or_none(row.get(str(semantics[STATEMENT_SCOPE]))),
                statement_family=_text_or_none(row.get(str(semantics[STATEMENT_FAMILY]))),
                source_ref=_source_ref(row, semantics, SOURCE_FILE, SOURCE_ROW),
            )
        )
    return result


def _opinion_rows(
    *,
    source_id: str,
    path: Path,
    spec: object,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
) -> list[OpinionRow]:
    from p01.models import SourceSpec

    if not isinstance(spec, SourceSpec):
        raise TypeError("locked source specification required")
    required = {
        FIRM_ID,
        FISCAL_YEAR,
        AUDIT_INDICATOR,
        AUDIT_OPINION,
        PERIOD_TYPE,
        STATEMENT_SCOPE,
        AUDIT_STATUS,
    }
    if not required.issubset(semantics):
        raise ValueError(
            f"source={source_id}: unresolved S2 semantics {sorted(required - set(semantics))}"
        )
    result: list[OpinionRow] = []
    for row in iter_rows(path, spec):
        firm_raw = _required(row.get(str(semantics[FIRM_ID])), FIRM_ID)
        normalized = normalize_entity_field(str(firm_raw), entity)
        canonical, _ = resolve_entity_link(source_id, str(firm_raw), normalized, entity)
        result.append(
            OpinionRow(
                firm_id=canonical,
                fiscal_year=int(str(_required(row.get(str(semantics[FISCAL_YEAR])), FISCAL_YEAR))),
                opinion_raw=_text_or_none(row.get(str(semantics[AUDIT_OPINION]))),
                audit_indicator=_text_or_none(row.get(str(semantics[AUDIT_INDICATOR]))),
                period_type=_text_or_none(row.get(str(semantics[PERIOD_TYPE]))),
                statement_scope=_text_or_none(row.get(str(semantics[STATEMENT_SCOPE]))),
                audit_status=_text_or_none(row.get(str(semantics[AUDIT_STATUS]))),
                source_ref=_source_ref(row, semantics, SOURCE_FILE, SOURCE_COLUMN),
            )
        )
    return result


def _required(value: object, field: str) -> object:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"evidence row missing {field}")
    return value


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    parsed = pd.Timestamp(str(value))
    return parsed.to_pydatetime()


def _boolean(value: object) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"evidence outcome is not a registered binary representation: {value}")


def _float_or_none(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = float(str(value))
    if not pd.notna(parsed):
        return None
    return parsed


def _text_or_none(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _source_ref(
    row: dict[str, object],
    semantics: dict[str, Any],
    file_semantic: str,
    row_semantic: str,
) -> str:
    file_value = _optional_text(row, semantics, file_semantic) or "registered_source"
    row_value = _optional_text(row, semantics, row_semantic) or "unavailable"
    return f"{file_value}#{row_value}"


def _optional_text(row: dict[str, object], semantics: dict[str, Any], name: str) -> str | None:
    column = semantics.get(name)
    if not isinstance(column, str):
        return None
    value = row.get(column)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_int(row: dict[str, object], semantics: dict[str, Any], name: str) -> int | None:
    value = _optional_text(row, semantics, name)
    return None if value is None else int(float(value))


def _optional_datetime(
    row: dict[str, object], semantics: dict[str, Any], name: str
) -> datetime | None:
    column = semantics.get(name)
    if not isinstance(column, str):
        return None
    value = row.get(column)
    if value is None or not str(value).strip():
        return None
    return _datetime(value)


def _optional_float(row: dict[str, object], semantics: dict[str, Any], name: str) -> float | None:
    column = semantics.get(name)
    if not isinstance(column, str):
        return None
    return _float_or_none(row.get(column))


def _optional_boolean(row: dict[str, object], semantics: dict[str, Any], name: str) -> bool | None:
    column = semantics.get(name)
    if not isinstance(column, str):
        return None
    return _boolean(row.get(column))


def _derived_event_id(
    source_id: str,
    firm_id: str,
    fiscal_year: int,
    availability: datetime,
    row: dict[str, object],
) -> str:
    canonical_row = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    value = f"{source_id}|{firm_id}|{fiscal_year}|{availability.isoformat()}|{canonical_row}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
