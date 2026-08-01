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
    DECISION_COUNT,
    DECISION_NUMBERS,
    DECISION_TOTAL_FINE,
    DETERMINATION_DATE,
    DETERMINATION_STATUS,
    DOCUMENT_IDS,
    EVENT_CLUSTER_ID,
    EVENT_ID,
    EVIDENCE_CATEGORY,
    EVIDENCE_RECORD_ID,
    EVIDENCE_RECORD_KIND,
    EVIDENCE_VALUE,
    FIRM_ID,
    FIRST_LABEL_KNOWN_DATE,
    FISCAL_YEAR,
    HAS_FINE,
    HAS_REMEDY,
    HAS_SUSPENSION,
    HAS_WARNING,
    LAST_LABEL_KNOWN_DATE,
    LAYER_OBSERVED_MASK,
    OBSERVATION_OPPORTUNITY,
    OBSERVED_SOURCE_LABEL,
    OPPORTUNITY_BASIS,
    OUTCOME,
    OUTCOME_BASIS,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PREDICTION_TIME,
    REASON_UNKNOWN,
    RECORDING_DATE,
    RECORDING_STATUS,
    SANCTION_YEAR,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    SOURCE_PROFILE_ID,
    SOURCE_RECORD_REFS,
    TARGET_FISCAL_YEAR,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
    TEMPORAL_ROLE,
    VERIFICATION_DATE,
    VERIFICATION_STATUS,
)


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    channel_id: str
    firm_id: str
    fiscal_year: int
    availability_date: datetime | None
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
    verification_status: bool | None = None
    determination_status: bool | None = None
    recording_status: bool | None = None
    verification_date: datetime | None = None
    determination_date: datetime | None = None
    recording_date: datetime | None = None
    layer_observed_mask: str | None = None
    reason_unknown: str | None = None
    evidence_value: float | None = None
    evidence_category: str | None = None
    source_record_refs: str | None = None
    period_link_source: str | None = None
    period_link_confidence: str | None = None
    outcome_basis: str = "direct_source_outcome"
    row_included: bool = True
    duplicate_representative_rule: str = "identical_signature_then_source_event_id"
    sanction_year: int | None = None
    target_fiscal_year: int | None = None
    decision_count: int | None = None
    document_ids: str | None = None
    decision_numbers: str | None = None
    total_fine: float | None = None
    has_fine: bool | None = None
    has_suspension: bool | None = None
    has_warning: bool | None = None
    has_remedy: bool | None = None
    first_label_known_date: datetime | None = None
    last_label_known_date: datetime | None = None
    taxonomy_codes: str | None = None
    taxonomy_reason_code: str | None = None


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
    """Map observed proxy evidence without converting absence into a latent negative."""
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
    """Build source evidence while preserving unknown O-V-D-R-S process layers."""
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
    result_records: dict[tuple[str, str], EvidenceRecord] = {}
    for record in ordered_records:
        _validate_measurement_process(record)
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
            previous = result_records.get(annual_key)
            if previous is not None:
                availability_rows.append(
                    _availability_row(record, "DUPLICATE_ANNUAL_SOURCE_RESULT")
                )
                raise ValueError(f"duplicate annual source result={annual_key}")
            result_records[annual_key] = record
            availability_rows.append(_availability_row(record, "ACCEPTED_ANNUAL_MEASUREMENT"))
            accepted.append((record, panel_times[key]))
            continue
        if record.evidence_record_kind == "firm_year_endpoint_result":
            if record.temporal_role != "next_calendar_year_regulatory_event":
                raise ValueError(
                    f"source={record.source_id}: S3 endpoint result requires next-year temporal role"
                )
            if record.availability_basis != "sanction_calendar_year":
                raise ValueError(
                    f"source={record.source_id}: S3 endpoint result requires sanction-year basis"
                )
            if record.event_id is not None or record.event_cluster_id is not None:
                raise ValueError("S3 firm-year result must not manufacture event identifiers")
            if not record.evidence_record_id:
                raise ValueError("S3 firm-year result requires evidence_record_id")
            if record.target_fiscal_year != record.fiscal_year:
                raise ValueError("S3 endpoint target year must equal record fiscal year")
            if record.sanction_year != record.fiscal_year + 1:
                raise ValueError("S3 endpoint sanction year must equal target fiscal year plus one")
            endpoint_key = (record.source_id, record.evidence_record_id)
            if endpoint_key in result_records:
                raise ValueError(f"duplicate firm-year endpoint result={endpoint_key}")
            result_records[endpoint_key] = record
            availability_rows.append(_availability_row(record, "ACCEPTED_S3_ENDPOINT_RESULT"))
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
        if record.availability_date is None:
            raise ValueError(f"source={record.source_id}: delayed event requires availability date")
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
        representative_availability = representative.availability_date
        if representative_availability is None:
            raise ValueError("delayed representative requires availability date")
        year_end = _fiscal_year_end(representative.fiscal_year, fiscal_year_end_month_day)
        report_lag = (prediction_time.date() - year_end).days
        detection_lag = (representative_availability.date() - prediction_time.date()).days
        total_lag = (representative_availability.date() - year_end).days
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
        observation_opportunity = _observation_opportunity(record)
        layer_mask = record.layer_observed_mask or _layer_mask(record)
        reason_unknown = record.reason_unknown
        if reason_unknown is None and record.outcome is None:
            reason_unknown = _default_unknown_reason(record)
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
                # Legacy aliases remain physical during the v6 -> v7 transition.
                columns[SOURCE_OPPORTUNITY]: observation_opportunity,
                columns[OBSERVATION_OPPORTUNITY]: observation_opportunity,
                columns[OPPORTUNITY_BASIS]: record.opportunity_basis,
                columns[VERIFICATION_STATUS]: record.verification_status,
                columns[DETERMINATION_STATUS]: record.determination_status,
                columns[RECORDING_STATUS]: record.recording_status,
                columns[OBSERVED_SOURCE_LABEL]: record.outcome,
                columns[VERIFICATION_DATE]: record.verification_date,
                columns[DETERMINATION_DATE]: record.determination_date,
                columns[RECORDING_DATE]: record.recording_date,
                columns[LAYER_OBSERVED_MASK]: layer_mask,
                columns[REASON_UNKNOWN]: reason_unknown,
                columns[EVIDENCE_VALUE]: record.evidence_value,
                columns[EVIDENCE_CATEGORY]: record.evidence_category,
                columns[SOURCE_RECORD_REFS]: record.source_record_refs,
                columns.get(SANCTION_YEAR, SANCTION_YEAR): record.sanction_year,
                columns.get(TARGET_FISCAL_YEAR, TARGET_FISCAL_YEAR): record.target_fiscal_year,
                columns.get(DECISION_COUNT, DECISION_COUNT): record.decision_count,
                columns.get(DOCUMENT_IDS, DOCUMENT_IDS): record.document_ids,
                columns.get(DECISION_NUMBERS, DECISION_NUMBERS): record.decision_numbers,
                columns.get(DECISION_TOTAL_FINE, DECISION_TOTAL_FINE): record.total_fine,
                columns.get(HAS_FINE, HAS_FINE): record.has_fine,
                columns.get(HAS_SUSPENSION, HAS_SUSPENSION): record.has_suspension,
                columns.get(HAS_WARNING, HAS_WARNING): record.has_warning,
                columns.get(HAS_REMEDY, HAS_REMEDY): record.has_remedy,
                columns.get(FIRST_LABEL_KNOWN_DATE, FIRST_LABEL_KNOWN_DATE): (
                    record.first_label_known_date
                ),
                columns.get(LAST_LABEL_KNOWN_DATE, LAST_LABEL_KNOWN_DATE): (
                    record.last_label_known_date
                ),
                columns.get(TAXONOMY_CODES, TAXONOMY_CODES): record.taxonomy_codes,
                columns.get(TAXONOMY_REASON_CODE, TAXONOMY_REASON_CODE): (
                    record.taxonomy_reason_code
                ),
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
            columns[OBSERVATION_OPPORTUNITY],
            columns[OPPORTUNITY_BASIS],
            columns[VERIFICATION_STATUS],
            columns[DETERMINATION_STATUS],
            columns[RECORDING_STATUS],
            columns[OBSERVED_SOURCE_LABEL],
            columns[VERIFICATION_DATE],
            columns[DETERMINATION_DATE],
            columns[RECORDING_DATE],
            columns[LAYER_OBSERVED_MASK],
            columns[REASON_UNKNOWN],
            columns[EVIDENCE_VALUE],
            columns[EVIDENCE_CATEGORY],
            columns[SOURCE_RECORD_REFS],
            columns.get(SANCTION_YEAR, SANCTION_YEAR),
            columns.get(TARGET_FISCAL_YEAR, TARGET_FISCAL_YEAR),
            columns.get(DECISION_COUNT, DECISION_COUNT),
            columns.get(DOCUMENT_IDS, DOCUMENT_IDS),
            columns.get(DECISION_NUMBERS, DECISION_NUMBERS),
            columns.get(DECISION_TOTAL_FINE, DECISION_TOTAL_FINE),
            columns.get(HAS_FINE, HAS_FINE),
            columns.get(HAS_SUSPENSION, HAS_SUSPENSION),
            columns.get(HAS_WARNING, HAS_WARNING),
            columns.get(HAS_REMEDY, HAS_REMEDY),
            columns.get(FIRST_LABEL_KNOWN_DATE, FIRST_LABEL_KNOWN_DATE),
            columns.get(LAST_LABEL_KNOWN_DATE, LAST_LABEL_KNOWN_DATE),
            columns.get(TAXONOMY_CODES, TAXONOMY_CODES),
            columns.get(TAXONOMY_REASON_CODE, TAXONOMY_REASON_CODE),
            columns[PERIOD_LINK_SOURCE],
            columns[PERIOD_LINK_CONFIDENCE],
            columns[OUTCOME_BASIS],
            columns[AVAILABILITY_DATE],
            columns[OUTCOME],
        ],
    )
    if not ledger.empty:
        for name in (
            FIRM_ID,
            SOURCE_ID,
            SOURCE_PROFILE_ID,
            CHANNEL_ID,
            EVIDENCE_RECORD_ID,
            EVIDENCE_RECORD_KIND,
            EVENT_ID,
            EVENT_CLUSTER_ID,
            TEMPORAL_ROLE,
            AVAILABILITY_BASIS,
            OPPORTUNITY_BASIS,
            LAYER_OBSERVED_MASK,
            REASON_UNKNOWN,
            EVIDENCE_CATEGORY,
            SOURCE_RECORD_REFS,
            PERIOD_LINK_SOURCE,
            PERIOD_LINK_CONFIDENCE,
            OUTCOME_BASIS,
        ):
            ledger[columns[name]] = ledger[columns[name]].astype("string")
        ledger[columns[FISCAL_YEAR]] = ledger[columns[FISCAL_YEAR]].astype("int16")
        for name in (
            SOURCE_OPPORTUNITY,
            OBSERVATION_OPPORTUNITY,
            VERIFICATION_STATUS,
            DETERMINATION_STATUS,
            RECORDING_STATUS,
            OBSERVED_SOURCE_LABEL,
            OUTCOME,
        ):
            ledger[columns[name]] = ledger[columns[name]].astype("boolean")
        for name in (
            VERIFICATION_DATE,
            DETERMINATION_DATE,
            RECORDING_DATE,
            AVAILABILITY_DATE,
        ):
            ledger[columns[name]] = pd.to_datetime(ledger[columns[name]]).astype("datetime64[ns]")
        ledger[columns[EVIDENCE_VALUE]] = ledger[columns[EVIDENCE_VALUE]].astype("float64")
        sanction_year_column = columns.get(SANCTION_YEAR, SANCTION_YEAR)
        target_year_column = columns.get(TARGET_FISCAL_YEAR, TARGET_FISCAL_YEAR)
        decision_count_column = columns.get(DECISION_COUNT, DECISION_COUNT)
        document_ids_column = columns.get(DOCUMENT_IDS, DOCUMENT_IDS)
        decision_numbers_column = columns.get(DECISION_NUMBERS, DECISION_NUMBERS)
        total_fine_column = columns.get(DECISION_TOTAL_FINE, DECISION_TOTAL_FINE)
        ledger[sanction_year_column] = ledger[sanction_year_column].astype("Int16")
        ledger[target_year_column] = ledger[target_year_column].astype("Int16")
        ledger[decision_count_column] = ledger[decision_count_column].astype("Int64")
        ledger[document_ids_column] = ledger[document_ids_column].astype("string")
        ledger[decision_numbers_column] = ledger[decision_numbers_column].astype("string")
        ledger[total_fine_column] = ledger[total_fine_column].astype("float64")
        for name in (HAS_FINE, HAS_SUSPENSION, HAS_WARNING, HAS_REMEDY):
            physical = columns.get(name, name)
            ledger[physical] = ledger[physical].astype("boolean")
        for name in (FIRST_LABEL_KNOWN_DATE, LAST_LABEL_KNOWN_DATE):
            physical = columns.get(name, name)
            ledger[physical] = pd.to_datetime(ledger[physical]).astype("datetime64[ns]")
        taxonomy_codes_column = columns.get(TAXONOMY_CODES, TAXONOMY_CODES)
        taxonomy_reason_column = columns.get(TAXONOMY_REASON_CODE, TAXONOMY_REASON_CODE)
        ledger[taxonomy_codes_column] = ledger[taxonomy_codes_column].astype("string")
        ledger[taxonomy_reason_column] = ledger[taxonomy_reason_column].astype("string")
        if not ledger[columns[SOURCE_OPPORTUNITY]].equals(ledger[columns[OBSERVATION_OPPORTUNITY]]):
            raise ValueError("legacy source_opportunity must equal canonical O")
        if not ledger[columns[OUTCOME]].equals(ledger[columns[OBSERVED_SOURCE_LABEL]]):
            raise ValueError("legacy outcome must equal canonical S")
    return EvidenceBuildResult(
        ledger=ledger,
        availability_registry=availability_rows,
        lag_decomposition={
            "records": lag_rows,
            "accepted_event_count": len(lag_rows),
            "accepted_delayed_event_count": len(lag_rows),
            "accepted_annual_measurement_count": sum(
                record.evidence_record_kind == "annual_source_result"
                for record in result_records.values()
            ),
            "accepted_s3_endpoint_result_count": sum(
                record.evidence_record_kind == "firm_year_endpoint_result"
                for record in result_records.values()
            ),
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
            "measurement_process_contract": "O-V-D-R-S",
            "high_specificity_anchor_assumed": False,
        },
    )


def _validate_measurement_process(record: EvidenceRecord) -> None:
    if record.source_opportunity is False and record.outcome is not None:
        raise ValueError("O=0 requires the observed source label S to be missing")
    if record.reason_unknown is not None and record.outcome is not None:
        raise ValueError("reason_unknown is permitted only when S is missing")
    expected_mask = _layer_mask(record)
    if record.layer_observed_mask is not None and record.layer_observed_mask != expected_mask:
        raise ValueError("layer_observed_mask conflicts with the O-V-D-R-S fields")
    if record.determination_status is not None and record.verification_status is not True:
        raise ValueError("D may be observed only when V=1")
    if record.recording_status is True and record.determination_status is None:
        raise ValueError("R=1 requires an observed determination D")
    if record.verification_date is not None and record.verification_status is None:
        raise ValueError("verification_date requires observed V")
    if record.determination_date is not None and record.determination_status is None:
        raise ValueError("determination_date requires observed D")
    if record.recording_date is not None and record.recording_status is None:
        raise ValueError("recording_date requires observed R")
    if all(
        value is not None
        for value in (
            record.verification_status,
            record.determination_status,
            record.recording_status,
        )
    ):
        expected = bool(
            record.verification_status and record.determination_status and record.recording_status
        )
        if record.outcome is not None and bool(record.outcome) != expected:
            raise ValueError("observed S conflicts with the fully observed V-D-R product")


def _availability_row(record: EvidenceRecord, status: str) -> dict[str, object]:
    observation_opportunity = _observation_opportunity(record)
    layer_mask = record.layer_observed_mask or _layer_mask(record)
    reason_unknown = record.reason_unknown
    if reason_unknown is None and record.outcome is None:
        reason_unknown = _default_unknown_reason(record)
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
        SOURCE_OPPORTUNITY: observation_opportunity,
        OBSERVATION_OPPORTUNITY: observation_opportunity,
        OPPORTUNITY_BASIS: record.opportunity_basis,
        VERIFICATION_STATUS: record.verification_status,
        DETERMINATION_STATUS: record.determination_status,
        RECORDING_STATUS: record.recording_status,
        OBSERVED_SOURCE_LABEL: record.outcome,
        LAYER_OBSERVED_MASK: layer_mask,
        REASON_UNKNOWN: reason_unknown,
        VERIFICATION_DATE: _iso(record.verification_date),
        DETERMINATION_DATE: _iso(record.determination_date),
        RECORDING_DATE: _iso(record.recording_date),
        EVIDENCE_VALUE: record.evidence_value,
        EVIDENCE_CATEGORY: record.evidence_category,
        SOURCE_RECORD_REFS: record.source_record_refs,
        AVAILABILITY_DATE: _iso(record.availability_date),
        OUTCOME: record.outcome,
        OUTCOME_BASIS: record.outcome_basis,
        PERIOD_LINK_SOURCE: record.period_link_source,
        PERIOD_LINK_CONFIDENCE: record.period_link_confidence,
        "row_included": record.row_included,
        "duplicate_representative_rule": record.duplicate_representative_rule,
        "status": status,
        "observed_endpoint_zero": record.outcome is False,
        "observed_endpoint_zero_is_latent_negative": False,
    }


def _observation_opportunity(record: EvidenceRecord) -> bool | None:
    # An observed source label logically establishes that the source-specific
    # observation opportunity occurred, even when legacy records did not
    # materialise O separately. This is not an inference about latent fraud.
    if record.outcome is not None and record.source_opportunity is None:
        return True
    return record.source_opportunity


def _layer_mask(record: EvidenceRecord) -> str:
    return "".join(
        name if value is not None else "-"
        for name, value in (
            ("O", _observation_opportunity(record)),
            ("V", record.verification_status),
            ("D", record.determination_status),
            ("R", record.recording_status),
            ("S", record.outcome),
        )
    )


def _default_unknown_reason(record: EvidenceRecord) -> str:
    opportunity = _observation_opportunity(record)
    if opportunity is False:
        return "NO_OBSERVATION_OPPORTUNITY"
    if opportunity is None:
        return "OBSERVATION_OPPORTUNITY_UNKNOWN"
    if record.recording_status is False:
        return "DETERMINATION_NOT_RECORDED"
    return "SOURCE_LABEL_UNKNOWN"


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
        _observation_opportunity(record),
        record.verification_status,
        record.determination_status,
        record.recording_status,
    )


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _as_datetime(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    return parsed.to_pydatetime()


def _fiscal_year_end(fiscal_year: int, month_day: str) -> date:
    month, day = (int(value) for value in month_day.split("-"))
    return date(fiscal_year, month, day)
