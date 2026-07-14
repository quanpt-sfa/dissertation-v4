"""Pure P03 event normalization, upstream deduplication, and lag decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    AVAILABILITY_BASIS,
    AVAILABILITY_DATE,
    CHANNEL_ID,
    EVENT_CLUSTER_ID,
    EVENT_ID,
    EVIDENCE_CATEGORY,
    EVIDENCE_RECORD_ID,
    EVIDENCE_RECORD_KIND,
    EVIDENCE_VALUE,
    FIRM_ID,
    FISCAL_YEAR,
    OPPORTUNITY_BASIS,
    OUTCOME,
    OUTCOME_BASIS,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PREDICTION_TIME,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    SOURCE_PROFILE_ID,
    SOURCE_RECORD_REFS,
    TEMPORAL_ROLE,
)


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    channel_id: str
    firm_id: str
    fiscal_year: int
    availability_date: datetime
    outcome: bool | None
    event_id: str | None
    event_cluster_id: str | None
    evidence_record_id: str | None = None
    evidence_record_kind: str = "delayed_event"
    source_profile_id: str | None = None
    temporal_role: str = "delayed_verification"
    availability_basis: str = "actual_publish_date"
    source_opportunity: bool | None = None
    opportunity_basis: str = "unknown_no_opportunity_indicator"
    evidence_value: float | None = None
    evidence_category: str | None = None
    source_record_refs: str | None = None
    period_link_source: str | None = None
    period_link_confidence: str | None = None
    outcome_basis: str = "direct_source_outcome"
    row_included: bool = True
    duplicate_representative_rule: str = "identical_signature_then_source_event_id"


@dataclass(frozen=True)
class EvidenceBuildResult:
    ledger: pd.DataFrame
    availability_registry: list[dict[str, object]]
    lag_decomposition: dict[str, object]


def map_evidence_outcome(
    *,
    outcome_mode: str,
    direct_outcome: bool | None,
    positive_indicator: bool | None,
    row_included: bool,
) -> tuple[bool | None, str]:
    """Map explicit source semantics without converting false indicators to negatives."""
    if not row_included:
        return None, "excluded_by_row_inclusion"
    if outcome_mode == "direct_outcome":
        return direct_outcome, "direct_source_outcome"
    if outcome_mode == "positive_indicator":
        return (
            True if positive_indicator is True else None,
            "explicit_hard_positive_indicator",
        )
    raise ValueError("unsupported evidence outcome semantics")


def build_evidence_ledger(
    *,
    panel: pd.DataFrame,
    records: list[EvidenceRecord],
    columns: dict[str, str],
    fiscal_year_end_month_day: str,
    lag_tolerance_days: int,
) -> EvidenceBuildResult:
    """Build a source-level ledger without treating absent or immature data as zero."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    required = {firm, year, prediction}
    if not required.issubset(panel.columns):
        raise ValueError(f"P03 panel missing columns {sorted(required - set(panel.columns))}")
    panel_times: dict[tuple[str, int], datetime] = {}
    for raw in cast(list[dict[str, Any]], panel.to_dict(orient="records")):
        key = (str(raw[firm]), int(raw[year]))
        if key in panel_times:
            raise ValueError(f"P03 duplicate panel key: {key}")
        panel_times[key] = _as_datetime(raw[prediction])
    accepted: list[tuple[EvidenceRecord, datetime]] = []
    availability_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []

    ordered_records = sorted(
        records,
        key=lambda item: (
            item.firm_id,
            item.fiscal_year,
            item.event_cluster_id or "",
            item.source_id,
            item.evidence_record_id or item.event_id or "",
        ),
    )
    linked_groups: dict[tuple[str, int, str], list[EvidenceRecord]] = {}
    annual_records: dict[tuple[str, str], EvidenceRecord] = {}
    for record in ordered_records:
        if not record.row_included:
            availability_rows.append(_availability_row(record, "EXCLUDED_BY_SOURCE_RULE"))
            continue
        key = (record.firm_id, record.fiscal_year)
        if key not in panel_times:
            availability_rows.append(_availability_row(record, "UNLINKED_FIRM_YEAR"))
            continue
        if record.evidence_record_kind == "annual_source_result":
            if record.temporal_role != "annual_measurement_at_anchor":
                raise ValueError(
                    f"source={record.source_id}: annual record requires annual temporal role"
                )
            if record.availability_basis != "common_annual_anchor":
                raise ValueError(
                    f"source={record.source_id}: annual record requires common annual anchor"
                )
            if record.event_id is not None or record.event_cluster_id is not None:
                raise ValueError("annual source result must not manufacture event identifiers")
            if not record.evidence_record_id:
                raise ValueError("annual source result requires evidence_record_id")
            if record.availability_date != panel_times[key]:
                availability_rows.append(_availability_row(record, "ANNUAL_ANCHOR_MISMATCH"))
                raise ValueError(
                    f"source={record.source_id}: annual availability must equal prediction anchor"
                )
            annual_key = (record.source_id, record.evidence_record_id)
            previous = annual_records.get(annual_key)
            if previous is not None:
                availability_rows.append(
                    _availability_row(record, "DUPLICATE_ANNUAL_SOURCE_RESULT")
                )
                raise ValueError(f"duplicate annual source result={annual_key}")
            annual_records[annual_key] = record
            availability_rows.append(_availability_row(record, "ACCEPTED_ANNUAL_MEASUREMENT"))
            accepted.append((record, panel_times[key]))
            continue
        if record.evidence_record_kind != "delayed_event":
            raise ValueError(
                f"source={record.source_id}: unsupported evidence record kind "
                f"{record.evidence_record_kind}"
            )
        if record.temporal_role != "delayed_verification":
            raise ValueError(f"source={record.source_id}: delayed event temporal role required")
        if record.availability_basis != "actual_publish_date":
            raise ValueError(f"source={record.source_id}: actual publish date required")
        if not record.event_id or not record.event_cluster_id:
            raise ValueError("delayed event requires event_id and event_cluster_id")
        cluster_key = (record.firm_id, record.fiscal_year, record.event_cluster_id)
        linked_groups.setdefault(cluster_key, []).append(record)

    for cluster_key in sorted(linked_groups):
        group = linked_groups[cluster_key]
        rules = {record.duplicate_representative_rule for record in group}
        if rules != {"identical_signature_then_source_event_id"}:
            raise ValueError(
                f"cluster={cluster_key}: one supported locked duplicate rule is required"
            )
        signatures = {_record_signature(record) for record in group}
        if len(signatures) != 1:
            for record in group:
                availability_rows.append(_availability_row(record, "CONFLICTING_UPSTREAM_CLUSTER"))
            raise ValueError(
                f"cluster={cluster_key}: duplicate records disagree on timing or outcome semantics"
            )
        representative = min(group, key=lambda item: (item.source_id, item.event_id))
        for record in group:
            status = "ACCEPTED" if record is representative else "DUPLICATE_UPSTREAM_EVENT"
            availability_rows.append(_availability_row(record, status))
        key = (cluster_key[0], cluster_key[1])
        prediction_time = panel_times[key]
        accepted.append((representative, prediction_time))
        year_end = _fiscal_year_end(representative.fiscal_year, fiscal_year_end_month_day)
        report_lag = (prediction_time.date() - year_end).days
        detection_lag = (representative.availability_date.date() - prediction_time.date()).days
        total_lag = (representative.availability_date.date() - year_end).days
        identity_error = total_lag - report_lag - detection_lag
        if abs(identity_error) > lag_tolerance_days:
            raise ValueError(
                f"lag identity violated for event={representative.event_id}: error={identity_error}"
            )
        lag_rows.append(
            {
                EVENT_ID: representative.event_id,
                EVENT_CLUSTER_ID: representative.event_cluster_id,
                SOURCE_ID: representative.source_id,
                "report_lag_days": report_lag,
                "detection_lag_days": detection_lag,
                "total_lag_days": total_lag,
                "identity_error_days": identity_error,
            }
        )

    ledger_rows: list[dict[str, Any]] = []
    for record, _ in accepted:
        evidence_record_id = record.evidence_record_id or record.event_id
        if evidence_record_id is None:
            raise ValueError("evidence record ID is required")
        ledger_rows.append(
            {
                columns[FIRM_ID]: record.firm_id,
                columns[FISCAL_YEAR]: record.fiscal_year,
                columns[SOURCE_ID]: record.source_id,
                columns[SOURCE_PROFILE_ID]: record.source_profile_id or record.source_id,
                columns[CHANNEL_ID]: record.channel_id,
                columns[EVIDENCE_RECORD_ID]: evidence_record_id,
                columns[EVIDENCE_RECORD_KIND]: record.evidence_record_kind,
                columns[EVENT_ID]: record.event_id,
                columns[EVENT_CLUSTER_ID]: record.event_cluster_id,
                columns[TEMPORAL_ROLE]: record.temporal_role,
                columns[AVAILABILITY_BASIS]: record.availability_basis,
                columns[SOURCE_OPPORTUNITY]: record.source_opportunity,
                columns[OPPORTUNITY_BASIS]: record.opportunity_basis,
                columns[EVIDENCE_VALUE]: record.evidence_value,
                columns[EVIDENCE_CATEGORY]: record.evidence_category,
                columns[SOURCE_RECORD_REFS]: record.source_record_refs,
                columns[PERIOD_LINK_SOURCE]: record.period_link_source,
                columns[PERIOD_LINK_CONFIDENCE]: record.period_link_confidence,
                columns[OUTCOME_BASIS]: record.outcome_basis,
                columns[AVAILABILITY_DATE]: record.availability_date,
                columns[OUTCOME]: record.outcome,
            }
        )
    ledger = pd.DataFrame(
        ledger_rows,
        columns=[
            columns[FIRM_ID],
            columns[FISCAL_YEAR],
            columns[SOURCE_ID],
            columns[SOURCE_PROFILE_ID],
            columns[CHANNEL_ID],
            columns[EVIDENCE_RECORD_ID],
            columns[EVIDENCE_RECORD_KIND],
            columns[EVENT_ID],
            columns[EVENT_CLUSTER_ID],
            columns[TEMPORAL_ROLE],
            columns[AVAILABILITY_BASIS],
            columns[SOURCE_OPPORTUNITY],
            columns[OPPORTUNITY_BASIS],
            columns[EVIDENCE_VALUE],
            columns[EVIDENCE_CATEGORY],
            columns[SOURCE_RECORD_REFS],
            columns[PERIOD_LINK_SOURCE],
            columns[PERIOD_LINK_CONFIDENCE],
            columns[OUTCOME_BASIS],
            columns[AVAILABILITY_DATE],
            columns[OUTCOME],
        ],
    )
    if not ledger.empty:
        ledger[columns[FIRM_ID]] = ledger[columns[FIRM_ID]].astype("string")
        ledger[columns[FISCAL_YEAR]] = ledger[columns[FISCAL_YEAR]].astype("int16")
        ledger[columns[SOURCE_ID]] = ledger[columns[SOURCE_ID]].astype("string")
        ledger[columns[SOURCE_PROFILE_ID]] = ledger[columns[SOURCE_PROFILE_ID]].astype("string")
        ledger[columns[CHANNEL_ID]] = ledger[columns[CHANNEL_ID]].astype("string")
        ledger[columns[EVIDENCE_RECORD_ID]] = ledger[columns[EVIDENCE_RECORD_ID]].astype("string")
        ledger[columns[EVIDENCE_RECORD_KIND]] = ledger[columns[EVIDENCE_RECORD_KIND]].astype(
            "string"
        )
        ledger[columns[EVENT_ID]] = ledger[columns[EVENT_ID]].astype("string")
        ledger[columns[EVENT_CLUSTER_ID]] = ledger[columns[EVENT_CLUSTER_ID]].astype("string")
        ledger[columns[TEMPORAL_ROLE]] = ledger[columns[TEMPORAL_ROLE]].astype("string")
        ledger[columns[AVAILABILITY_BASIS]] = ledger[columns[AVAILABILITY_BASIS]].astype("string")
        ledger[columns[SOURCE_OPPORTUNITY]] = ledger[columns[SOURCE_OPPORTUNITY]].astype("boolean")
        ledger[columns[OPPORTUNITY_BASIS]] = ledger[columns[OPPORTUNITY_BASIS]].astype("string")
        ledger[columns[EVIDENCE_VALUE]] = ledger[columns[EVIDENCE_VALUE]].astype("float64")
        ledger[columns[EVIDENCE_CATEGORY]] = ledger[columns[EVIDENCE_CATEGORY]].astype("string")
        ledger[columns[SOURCE_RECORD_REFS]] = ledger[columns[SOURCE_RECORD_REFS]].astype("string")
        ledger[columns[PERIOD_LINK_SOURCE]] = ledger[columns[PERIOD_LINK_SOURCE]].astype("string")
        ledger[columns[PERIOD_LINK_CONFIDENCE]] = ledger[columns[PERIOD_LINK_CONFIDENCE]].astype(
            "string"
        )
        ledger[columns[OUTCOME_BASIS]] = ledger[columns[OUTCOME_BASIS]].astype("string")
        ledger[columns[AVAILABILITY_DATE]] = pd.to_datetime(
            ledger[columns[AVAILABILITY_DATE]]
        ).astype("datetime64[ns]")
        ledger[columns[OUTCOME]] = ledger[columns[OUTCOME]].astype("boolean")
    return EvidenceBuildResult(
        ledger=ledger,
        availability_registry=availability_rows,
        lag_decomposition={
            "records": lag_rows,
            "accepted_event_count": len(lag_rows),
            "accepted_delayed_event_count": len(lag_rows),
            "accepted_annual_measurement_count": len(annual_records),
            "deduplicated_event_count": sum(
                row["status"] == "DUPLICATE_UPSTREAM_EVENT" for row in availability_rows
            ),
            "excluded_by_source_rule_count": sum(
                row["status"] == "EXCLUDED_BY_SOURCE_RULE" for row in availability_rows
            ),
            "unlinked_event_count": sum(
                row["status"] == "UNLINKED_FIRM_YEAR" for row in availability_rows
            ),
            "duplicate_representative_rule": "identical_signature_then_source_event_id",
            "lag_identity_tolerance_days": lag_tolerance_days,
        },
    )


def _availability_row(record: EvidenceRecord, status: str) -> dict[str, object]:
    return {
        EVIDENCE_RECORD_ID: record.evidence_record_id or record.event_id,
        EVIDENCE_RECORD_KIND: record.evidence_record_kind,
        EVENT_ID: record.event_id,
        EVENT_CLUSTER_ID: record.event_cluster_id,
        FIRM_ID: record.firm_id,
        FISCAL_YEAR: record.fiscal_year,
        SOURCE_ID: record.source_id,
        SOURCE_PROFILE_ID: record.source_profile_id or record.source_id,
        CHANNEL_ID: record.channel_id,
        TEMPORAL_ROLE: record.temporal_role,
        AVAILABILITY_BASIS: record.availability_basis,
        SOURCE_OPPORTUNITY: record.source_opportunity,
        OPPORTUNITY_BASIS: record.opportunity_basis,
        EVIDENCE_VALUE: record.evidence_value,
        EVIDENCE_CATEGORY: record.evidence_category,
        SOURCE_RECORD_REFS: record.source_record_refs,
        AVAILABILITY_DATE: record.availability_date.isoformat(),
        OUTCOME: record.outcome,
        OUTCOME_BASIS: record.outcome_basis,
        PERIOD_LINK_SOURCE: record.period_link_source,
        PERIOD_LINK_CONFIDENCE: record.period_link_confidence,
        "row_included": record.row_included,
        "duplicate_representative_rule": record.duplicate_representative_rule,
        "status": status,
    }


def _record_signature(record: EvidenceRecord) -> tuple[object, ...]:
    return (
        record.channel_id,
        record.availability_date,
        record.outcome,
        record.outcome_basis,
        record.period_link_source,
        record.period_link_confidence,
        record.temporal_role,
        record.availability_basis,
    )


def _as_datetime(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    return parsed.to_pydatetime()


def _fiscal_year_end(fiscal_year: int, month_day: str) -> date:
    month, day = (int(value) for value in month_day.split("-"))
    return date(fiscal_year, month, day)
