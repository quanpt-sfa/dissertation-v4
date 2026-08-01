"""Deterministic S3 calendar-year mapping and normalized endpoint construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    CHANNEL_ID,
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    DECISION_COUNT,
    DECISION_DATE,
    DECISION_NUMBER,
    DECISION_NUMBERS,
    DECISION_TOTAL_FINE,
    DOCUMENT_ID,
    DOCUMENT_IDS,
    FIRM_ID,
    FIRST_LABEL_KNOWN_DATE,
    HARD_POSITIVE,
    HAS_FINE,
    HAS_REMEDY,
    HAS_SUSPENSION,
    HAS_WARNING,
    LABEL_KNOWN_DATE,
    LAST_LABEL_KNOWN_DATE,
    LEGACY_EVENT_ID,
    NORMALIZED_VIOLATION_CODE,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    PUBLISH_DATE,
    ROW_INCLUSION,
    SANCTION_YEAR,
    SOURCE_RECORD_REFS,
    TARGET_FISCAL_YEAR,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
    TEMPORAL_ROLE,
)
from evidence.service import EvidenceRecord

S3_ENDPOINTS = ("S3_BROAD", "S3_REPORTING", "S3_CONTENT", "S3_TIMELINESS")


@dataclass(frozen=True)
class SanctionDecisionInput:
    document_id: str
    firm_id: str
    decision_number: str | None = None
    sanction_year: int | None = None
    decision_date: datetime | None = None
    publish_date: datetime | None = None
    label_known_date: datetime | None = None
    affected_fiscal_year: int | None = None
    primary_violation_l1: str | None = None
    primary_violation_l2: str | None = None
    construct_family: str | None = None
    construct_target: str | None = None
    normalized_violation_code: str | None = None
    row_included: bool = True
    hard_positive: bool = True
    legacy_event_id: str | None = None
    period_link_source: str | None = None
    period_link_confidence: str | None = None
    total_fine: float | None = None
    has_fine: bool | None = None
    has_suspension: bool | None = None
    has_warning: bool | None = None
    has_remedy: bool | None = None
    source_ref: str | None = None
    target_fiscal_year: int | None = None


@dataclass(frozen=True)
class SanctionEvidenceBuild:
    decision_ledger: pd.DataFrame
    endpoint_records: list[EvidenceRecord]
    audit: dict[str, object]


def resolve_sanction_year(row: SanctionDecisionInput) -> tuple[int | None, str]:
    """Apply the locked field/date precedence without consulting label year."""
    if row.sanction_year is not None:
        return int(row.sanction_year), SANCTION_YEAR
    if row.decision_date is not None:
        return row.decision_date.year, DECISION_DATE
    if row.publish_date is not None:
        return row.publish_date.year, PUBLISH_DATE
    if row.target_fiscal_year is not None:
        return int(row.target_fiscal_year) + 1, TARGET_FISCAL_YEAR
    return None, "SANCTION_YEAR_UNRESOLVED"


def target_fiscal_year(row: SanctionDecisionInput) -> int | None:
    year, _ = resolve_sanction_year(row)
    return None if year is None else year - 1


def source_year_is_complete(
    source_year: int,
    *,
    complete_through_year: int,
    incomplete_years: set[int],
) -> bool:
    return source_year <= complete_through_year and source_year not in incomplete_years


def validate_s3_taxonomy(configuration: dict[str, Any]) -> None:
    """Reject endpoint drift and taxonomy rules that violate the locked nesting."""
    required_settings = {
        CHANNEL_ID: "S3",
        "processor": "sanction_calendar_year",
        TEMPORAL_ROLE: "next_calendar_year_regulatory_event",
        "availability_rule": "sanction_calendar_year",
        "target_year_rule": "sanction_year_minus_one",
        "decision_key": DOCUMENT_ID,
        "decision_provenance_field": DECISION_NUMBER,
        "affected_fiscal_year_role": "sensitivity_provenance_only",
    }
    for setting, expected in required_settings.items():
        if configuration.get(setting) != expected:
            raise ValueError(f"S3 taxonomy {setting} must equal {expected}")
    temporal_estimand = _mapping(
        configuration.get("temporal_estimand"), "s3_taxonomy.temporal_estimand"
    )
    if temporal_estimand != {
        "temporal_estimand_id": "next_calendar_year_regulatory_event",
        "target_calendar_rule": "fiscal_year_plus_one",
        "maturity_basis": "complete_next_calendar_year_source",
        "horizon_applicable": False,
        "horizon_months": None,
    }:
        raise ValueError("S3 temporal estimand metadata differs from the locked annual rule")
    if configuration.get("sanction_year_precedence") != [
        SANCTION_YEAR,
        DECISION_DATE,
        PUBLISH_DATE,
    ]:
        raise ValueError("S3 sanction-year precedence differs from the locked rule")
    if configuration.get("label_known_date_precedence") != [
        LABEL_KNOWN_DATE,
        PUBLISH_DATE,
        DECISION_DATE,
    ]:
        raise ValueError("S3 label-known-date precedence differs from the locked rule")
    endpoints = _mapping(configuration.get("endpoints"), "s3_taxonomy.endpoints")
    if set(endpoints) != set(S3_ENDPOINTS):
        raise ValueError("S3 taxonomy must register exactly four primary endpoints")
    content = _string_set(_mapping(endpoints["S3_CONTENT"], "S3_CONTENT").get("normalized_codes"))
    timeliness = _string_set(
        _mapping(endpoints["S3_TIMELINESS"], "S3_TIMELINESS").get("normalized_codes")
    )
    if content & timeliness:
        raise ValueError("S3_CONTENT and S3_TIMELINESS must be disjoint")
    invariants = _mapping(configuration.get("invariants"), "s3_taxonomy.invariants")
    required = {
        "content_subset_reporting",
        "timeliness_subset_reporting",
        "reporting_subset_broad",
        "content_timeliness_disjoint",
    }
    if any(invariants.get(name) is not True for name in required):
        raise ValueError("all S3 taxonomy nesting invariants must be true")
    unmapped = _mapping(configuration.get("unmapped_policy"), "s3_taxonomy.unmapped_policy")
    if unmapped != {
        "broad": "include",
        "reporting": "unknown",
        "content": "unknown",
        "timeliness": "unknown",
        "reason_code": "UNMAPPED_S3_TAXONOMY",
    }:
        raise ValueError("S3 unmapped-taxonomy policy differs from the locked rule")


def classify_s3_taxonomy(
    row: SanctionDecisionInput,
    configuration: dict[str, Any],
) -> tuple[dict[str, bool | None], str | None]:
    """Classify normalized taxonomy fields only; raw-text keyword matching is forbidden."""
    validate_s3_taxonomy(configuration)
    if not row.row_included or not row.hard_positive:
        return {endpoint: None for endpoint in S3_ENDPOINTS}, "EXCLUDED_BY_SOURCE_RULE"
    endpoints = _mapping(configuration["endpoints"], "s3_taxonomy.endpoints")
    reporting = _mapping(endpoints["S3_REPORTING"], "S3_REPORTING")
    content = _string_set(_mapping(endpoints["S3_CONTENT"], "S3_CONTENT")["normalized_codes"])
    timeliness = _string_set(
        _mapping(endpoints["S3_TIMELINESS"], "S3_TIMELINESS")["normalized_codes"]
    )
    reporting_l1 = _string_set(reporting.get("level_1_values"))
    reporting_codes = _string_set(reporting.get("normalized_codes"))
    broad_only = _string_set(configuration.get("broad_only_level_1_values"))
    level_1 = _normalized(row.primary_violation_l1)
    code = _normalized(row.normalized_violation_code)
    content_hit = code in content if code is not None else False
    timeliness_hit = code in timeliness if code is not None else False
    reporting_hit = (
        content_hit
        or timeliness_hit
        or (level_1 in reporting_l1 if level_1 is not None else False)
        or (code in reporting_codes if code is not None else False)
    )
    mapped_broad_only = level_1 in broad_only if level_1 is not None else False
    mapped = reporting_hit or mapped_broad_only
    if not mapped:
        return {
            "S3_BROAD": True,
            "S3_REPORTING": None,
            "S3_CONTENT": None,
            "S3_TIMELINESS": None,
        }, "UNMAPPED_S3_TAXONOMY"
    return {
        "S3_BROAD": True,
        "S3_REPORTING": reporting_hit,
        "S3_CONTENT": content_hit,
        "S3_TIMELINESS": timeliness_hit,
    }, None


def build_s3_evidence(
    *,
    panel_keys: set[tuple[str, int]],
    decisions: list[SanctionDecisionInput],
    taxonomy: dict[str, Any],
    complete_through_year: int,
    incomplete_years: set[int],
    columns: dict[str, str],
    source_profile_id: str = "sanction_evidence",
) -> SanctionEvidenceBuild:
    """Build a decision ledger and complete firm-year endpoint result records."""
    validate_s3_taxonomy(taxonomy)
    completeness = _mapping(
        taxonomy.get("sanction_source_completeness"),
        "s3_taxonomy.sanction_source_completeness",
    )
    configured_incomplete = {int(value) for value in completeness.get("incomplete_years", [])}
    if (
        completeness.get("complete_through_year") != complete_through_year
        or configured_incomplete != incomplete_years
    ):
        raise ValueError("S3 source-year completeness must be read from the locked taxonomy")
    grouped: dict[tuple[str, str], list[SanctionDecisionInput]] = {}
    for decision in decisions:
        document_id = decision.document_id.strip()
        firm_id = decision.firm_id.strip()
        if not document_id:
            raise ValueError("S3 decision requires document_id")
        if not firm_id:
            raise ValueError("S3 decision requires firm_id")
        grouped.setdefault((document_id, firm_id), []).append(decision)
    ledger_rows: list[dict[str, object]] = []
    decision_values: dict[
        tuple[str, int], list[tuple[str, dict[str, bool | None], str | None, dict[str, object]]]
    ] = {}
    excluded_source_rule_count = 0
    excluded_unresolved_count = 0
    eligible_unresolved_count = 0
    unresolved_firms: set[str] = set()
    duplicate_count = 0
    for key in sorted(grouped):
        raw_group = sorted(grouped[key], key=_decision_sort_key)
        by_signature: dict[tuple[object, ...], SanctionDecisionInput] = {}
        for item in raw_group:
            by_signature.setdefault(_decision_signature(item), item)
        group = sorted(by_signature.values(), key=_decision_sort_key)
        duplicate_count += len(raw_group) - len(group)

        included = [item for item in group if item.row_included and item.hard_positive]
        excluded = [item for item in group if not (item.row_included and item.hard_positive)]
        if excluded:
            excluded_source_rule_count += 1
            if any(resolve_sanction_year(item)[0] is None for item in excluded):
                excluded_unresolved_count += 1
            for item in excluded:
                ledger_rows.append(
                    _excluded_ledger_row(
                        [item],
                        columns,
                        reason="EXCLUDED_BY_SOURCE_RULE",
                    )
                )
        if not included:
            continue

        years = {resolve_sanction_year(item)[0] for item in included}
        years.discard(None)
        if len(years) != 1 or any(resolve_sanction_year(item)[0] is None for item in included):
            eligible_unresolved_count += 1
            unresolved_firms.add(key[1])
            ledger_rows.append(_unresolved_ledger_row(included, columns))
            continue

        sanction_year_value = next(iter(years))
        assert sanction_year_value is not None
        target_year = sanction_year_value - 1
        classifications = [classify_s3_taxonomy(item, taxonomy) for item in included]
        endpoint_values: dict[str, bool | None] = {}
        unmapped = any(reason == "UNMAPPED_S3_TAXONOMY" for _, reason in classifications)
        for endpoint in S3_ENDPOINTS:
            raw_values = [values[endpoint] for values, _ in classifications]
            if any(value is True for value in raw_values):
                endpoint_values[endpoint] = True
            elif not raw_values:
                endpoint_values[endpoint] = None
            elif any(value is None for value in raw_values):
                endpoint_values[endpoint] = None
            else:
                endpoint_values[endpoint] = False
        reason = "UNMAPPED_S3_TAXONOMY" if unmapped else None
        metadata = _decision_metadata(included)
        ledger_rows.append(
            _decision_ledger_row(
                group=included,
                sanction_year_value=sanction_year_value,
                target_year=target_year,
                reason=reason,
                columns=columns,
            )
        )
        for endpoint in S3_ENDPOINTS:
            outcome = endpoint_values[endpoint]
            if outcome is None:
                continue
            for k in ((key[1], target_year),):
                decision_values.setdefault(k, []).append(
                    (
                        key[0],
                        endpoint_values,
                        reason,
                        metadata,
                    )
                )

    unresolved_count = eligible_unresolved_count
    endpoint_records: list[EvidenceRecord] = []
    for firm_id, fiscal_year in sorted(panel_keys):
        required_year = fiscal_year + 1
        complete = source_year_is_complete(
            required_year,
            complete_through_year=complete_through_year,
            incomplete_years=incomplete_years,
        )
        firm_year_decisions = decision_values.get((firm_id, fiscal_year), [])
        for endpoint in S3_ENDPOINTS:
            positive = [item for item in firm_year_decisions if item[1][endpoint] is True]
            taxonomy_unknown = any(item[1][endpoint] is None for item in firm_year_decisions)
            taxonomy_reason = (
                "UNMAPPED_S3_TAXONOMY"
                if any(item[2] == "UNMAPPED_S3_TAXONOMY" for item in firm_year_decisions)
                else None
            )
            if positive:
                result: bool | None = True
                outcome_reason = None
            elif firm_id in unresolved_firms:
                result = None
                outcome_reason = "SANCTION_YEAR_UNRESOLVED"
            elif not complete:
                result = None
                outcome_reason = "SOURCE_YEAR_INCOMPLETE"
            elif taxonomy_unknown:
                result = None
                outcome_reason = "UNMAPPED_S3_TAXONOMY"
            else:
                result = False
                outcome_reason = None
            selected = (
                positive
                if positive
                else [item for item in firm_year_decisions if item[1][endpoint] is None]
                if taxonomy_unknown
                else []
            )
            metadata = _endpoint_metadata(selected)
            endpoint_records.append(
                EvidenceRecord(
                    source_id=endpoint,
                    source_profile_id=source_profile_id,
                    channel_id="S3",
                    firm_id=firm_id,
                    fiscal_year=fiscal_year,
                    availability_date=cast(datetime | None, metadata[FIRST_LABEL_KNOWN_DATE]),
                    outcome=result,
                    event_id=None,
                    event_cluster_id=None,
                    evidence_record_id=f"{endpoint}|{firm_id}|{fiscal_year}",
                    evidence_record_kind="firm_year_endpoint_result",
                    temporal_role="next_calendar_year_regulatory_event",
                    availability_basis="sanction_calendar_year",
                    source_opportunity=True if complete else None,
                    opportunity_basis=(
                        "complete_source_year_and_panel_universe"
                        if complete
                        else "incomplete_source_year"
                    ),
                    verification_status=True if result is True else None,
                    determination_status=True if result is True else None,
                    recording_status=True if result is True else None,
                    reason_unknown=outcome_reason if result is None else None,
                    evidence_category=endpoint,
                    source_record_refs=cast(str, metadata[DOCUMENT_IDS]),
                    outcome_basis=(
                        outcome_reason
                        or (
                            "eligible_decision_in_required_sanction_year"
                            if result is True
                            else "no_endpoint_decision_in_complete_source_year"
                        )
                    ),
                    duplicate_representative_rule="identical_signature_then_document_id_firm",
                    sanction_year=required_year,
                    target_fiscal_year=fiscal_year,
                    decision_count=cast(int, metadata[DECISION_COUNT]),
                    document_ids=cast(str, metadata[DOCUMENT_IDS]),
                    decision_numbers=cast(str, metadata[DECISION_NUMBERS]),
                    total_fine=cast(float | None, metadata[DECISION_TOTAL_FINE]),
                    has_fine=cast(bool | None, metadata[HAS_FINE]),
                    has_suspension=cast(bool | None, metadata[HAS_SUSPENSION]),
                    has_warning=cast(bool | None, metadata[HAS_WARNING]),
                    has_remedy=cast(bool | None, metadata[HAS_REMEDY]),
                    first_label_known_date=cast(datetime | None, metadata[FIRST_LABEL_KNOWN_DATE]),
                    last_label_known_date=cast(datetime | None, metadata[LAST_LABEL_KNOWN_DATE]),
                    taxonomy_codes=cast(str, metadata[TAXONOMY_CODES]),
                    taxonomy_reason_code=taxonomy_reason,
                )
            )

    ledger = pd.DataFrame(ledger_rows, columns=_decision_ledger_columns(columns))
    _coerce_decision_ledger(ledger, columns)
    unique_documents = {row.document_id for row in decisions if row.document_id.strip()}
    return SanctionEvidenceBuild(
        decision_ledger=ledger,
        endpoint_records=endpoint_records,
        audit={
            "processor": "sanction_calendar_year",
            TEMPORAL_ROLE: "next_calendar_year_regulatory_event",
            "target_year_rule": "sanction_year_minus_one",
            "input_row_count": len(decisions),
            "unique_decision_count": len(unique_documents),
            "firm_mapping_count": len(grouped),
            "duplicate_source_row_count": duplicate_count,
            "unresolved_sanction_year_mapping_count": unresolved_count,
            "excluded_source_rule_mapping_count": excluded_source_rule_count,
            "excluded_source_rule_row_count": sum(
                1 for row in decisions if not (row.row_included and row.hard_positive)
            ),
            "excluded_unresolved_mapping_count": excluded_unresolved_count,
            "eligible_unresolved_sanction_year_mapping_count": eligible_unresolved_count,
            "firms_with_unresolved_sanction_year": sorted(unresolved_firms),
            "endpoint_result_count": len(endpoint_records),
            "complete_through_year": complete_through_year,
            "incomplete_years": sorted(incomplete_years),
            "decision_number_used_as_key": False,
            "label_known_date_changes_target_year": False,
        },
    )


def _decision_metadata(group: list[SanctionDecisionInput]) -> dict[str, object]:
    known_dates = sorted(date for item in group if (date := _known_date(item)) is not None)
    codes = sorted(
        {value for item in group if (value := _normalized(item.normalized_violation_code))}
    )
    numbers = sorted({item.decision_number for item in group if item.decision_number})
    fines = {float(item.total_fine) for item in group if item.total_fine is not None}
    return {
        DECISION_NUMBERS: _json(numbers),
        DECISION_TOTAL_FINE: next(iter(fines)) if len(fines) == 1 else None,
        HAS_FINE: _any_optional(item.has_fine for item in group),
        HAS_SUSPENSION: _any_optional(item.has_suspension for item in group),
        HAS_WARNING: _any_optional(item.has_warning for item in group),
        HAS_REMEDY: _any_optional(item.has_remedy for item in group),
        FIRST_LABEL_KNOWN_DATE: known_dates[0] if known_dates else None,
        LAST_LABEL_KNOWN_DATE: known_dates[-1] if known_dates else None,
        TAXONOMY_CODES: _json(codes),
    }


def _endpoint_metadata(
    rows: list[tuple[str, dict[str, bool | None], str | None, dict[str, object]]],
) -> dict[str, object]:
    document_ids = sorted({item[0] for item in rows})
    numbers: set[str] = set()
    codes: set[str] = set()
    fine_values: list[float] = []
    known_dates: list[datetime] = []
    optional_flags: dict[str, list[bool | None]] = {
        HAS_FINE: [],
        HAS_SUSPENSION: [],
        HAS_WARNING: [],
        HAS_REMEDY: [],
    }
    for _, _, _, metadata in rows:
        numbers.update(cast(list[str], json.loads(cast(str, metadata[DECISION_NUMBERS]))))
        codes.update(cast(list[str], json.loads(cast(str, metadata[TAXONOMY_CODES]))))
        if isinstance(metadata[DECISION_TOTAL_FINE], (int, float)):
            fine_values.append(float(cast(float, metadata[DECISION_TOTAL_FINE])))
        for name in optional_flags:
            optional_flags[name].append(cast(bool | None, metadata[name]))
        for name in (FIRST_LABEL_KNOWN_DATE, LAST_LABEL_KNOWN_DATE):
            if isinstance(metadata[name], datetime):
                known_dates.append(cast(datetime, metadata[name]))
    known_dates.sort()
    return {
        DECISION_COUNT: len(document_ids),
        DOCUMENT_IDS: _json(document_ids),
        DECISION_NUMBERS: _json(sorted(numbers)),
        DECISION_TOTAL_FINE: float(sum(fine_values)) if fine_values else None,
        **{name: _any_optional(values) for name, values in optional_flags.items()},
        FIRST_LABEL_KNOWN_DATE: known_dates[0] if known_dates else None,
        LAST_LABEL_KNOWN_DATE: known_dates[-1] if known_dates else None,
        TAXONOMY_CODES: _json(sorted(codes)),
    }


def _decision_ledger_row(
    *,
    group: list[SanctionDecisionInput],
    sanction_year_value: int,
    target_year: int,
    reason: str | None,
    columns: dict[str, str],
) -> dict[str, object]:
    first = group[0]
    metadata = _decision_metadata(group)
    return {
        columns[DOCUMENT_ID]: first.document_id,
        columns[FIRM_ID]: first.firm_id,
        columns[TARGET_FISCAL_YEAR]: target_year,
        columns[PRIMARY_VIOLATION_L1]: first.primary_violation_l1,
        columns[PRIMARY_VIOLATION_L2]: first.primary_violation_l2,
        columns[CONSTRUCT_FAMILY]: first.construct_family,
        columns[CONSTRUCT_TARGET]: first.construct_target,
        columns[NORMALIZED_VIOLATION_CODE]: first.normalized_violation_code,
        columns[HARD_POSITIVE]: any(item.hard_positive for item in group),
        columns[ROW_INCLUSION]: any(item.row_included for item in group),
        columns[LEGACY_EVENT_ID]: first.legacy_event_id,
        columns[PERIOD_LINK_SOURCE]: first.period_link_source,
        columns[PERIOD_LINK_CONFIDENCE]: first.period_link_confidence,
        columns[SOURCE_RECORD_REFS]: _json(
            sorted({item.source_ref for item in group if item.source_ref})
        ),
        columns[TAXONOMY_CODES]: metadata[TAXONOMY_CODES],
        columns[TAXONOMY_REASON_CODE]: reason,
    }


def _unresolved_ledger_row(
    group: list[SanctionDecisionInput], columns: dict[str, str]
) -> dict[str, object]:
    row = _decision_ledger_row(
        group=group,
        sanction_year_value=0,
        target_year=0,
        reason="SANCTION_YEAR_UNRESOLVED",
        columns=columns,
    )
    row[columns[TARGET_FISCAL_YEAR]] = None
    return row


def _excluded_ledger_row(
    group: list[SanctionDecisionInput], columns: dict[str, str], reason: str
) -> dict[str, object]:
    row = _decision_ledger_row(
        group=group,
        sanction_year_value=0,
        target_year=0,
        reason=reason,
        columns=columns,
    )
    row[columns[TARGET_FISCAL_YEAR]] = None
    return row


def _decision_ledger_columns(columns: dict[str, str]) -> list[str]:
    return [
        columns[name]
        for name in (
            DOCUMENT_ID,
            FIRM_ID,
            TARGET_FISCAL_YEAR,
            PRIMARY_VIOLATION_L1,
            PRIMARY_VIOLATION_L2,
            CONSTRUCT_FAMILY,
            CONSTRUCT_TARGET,
            NORMALIZED_VIOLATION_CODE,
            HARD_POSITIVE,
            ROW_INCLUSION,
            LEGACY_EVENT_ID,
            PERIOD_LINK_SOURCE,
            PERIOD_LINK_CONFIDENCE,
            SOURCE_RECORD_REFS,
            TAXONOMY_CODES,
            TAXONOMY_REASON_CODE,
        )
    ]


def _coerce_decision_ledger(frame: pd.DataFrame, columns: dict[str, str]) -> None:
    if frame.empty:
        return
    string_fields = (
        DOCUMENT_ID,
        FIRM_ID,
        PRIMARY_VIOLATION_L1,
        PRIMARY_VIOLATION_L2,
        CONSTRUCT_FAMILY,
        CONSTRUCT_TARGET,
        NORMALIZED_VIOLATION_CODE,
        LEGACY_EVENT_ID,
        PERIOD_LINK_SOURCE,
        PERIOD_LINK_CONFIDENCE,
        SOURCE_RECORD_REFS,
        TAXONOMY_CODES,
        TAXONOMY_REASON_CODE,
    )
    for name in string_fields:
        frame[columns[name]] = frame[columns[name]].astype("string")
    for name in (TARGET_FISCAL_YEAR,):
        frame[columns[name]] = frame[columns[name]].astype("Int16")
    for name in (
        HARD_POSITIVE,
        ROW_INCLUSION,
    ):
        frame[columns[name]] = frame[columns[name]].astype("boolean")


def _known_date(row: SanctionDecisionInput) -> datetime | None:
    return row.label_known_date or row.publish_date or row.decision_date


def _decision_sort_key(row: SanctionDecisionInput) -> tuple[object, ...]:
    return (
        row.document_id,
        row.firm_id,
        resolve_sanction_year(row)[0] or 0,
        row.normalized_violation_code or "",
        row.primary_violation_l1 or "",
        row.source_ref or "",
    )


def _decision_signature(row: SanctionDecisionInput) -> tuple[object, ...]:
    return (
        row.document_id,
        row.firm_id,
        row.decision_number,
        row.sanction_year,
        row.decision_date,
        row.publish_date,
        row.label_known_date,
        row.affected_fiscal_year,
        row.primary_violation_l1,
        row.primary_violation_l2,
        row.construct_family,
        row.construct_target,
        row.normalized_violation_code,
        row.row_included,
        row.hard_positive,
        row.legacy_event_id,
        row.period_link_source,
        row.period_link_confidence,
        row.total_fine,
        row.has_fine,
        row.has_suspension,
        row.has_warning,
        row.has_remedy,
    )


def _any_optional(values: Any) -> bool | None:
    items = list(values)
    if any(value is True for value in items):
        return True
    if items and all(value is False for value in items):
        return False
    return None


def _normalized(value: str | None) -> str | None:
    return value.strip().upper() if value is not None and value.strip() else None


def _json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        raise ValueError("S3 taxonomy list required")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise ValueError("S3 taxonomy values must be non-empty strings")
    return {cast(str, item).strip().upper() for item in items}


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)
