"""Semantic views over the single final firm-year Parquet input.

The production pipeline receives one physical file.  Snapshot profiles expose
that file through separate S1, S2, S3, and known-case semantic views so the
existing evidence contracts remain source-specific without requiring multiple
raw files on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from core.evidence_registry import LogicalEvidenceSource
from evidence.annual import AdjustmentRow, OpinionRow
from evidence.service import EvidenceRecord
from p01.models import SourceSpec
from p01.readers import iter_rows
from p02.builder import normalize_entity_field, resolve_entity_link
from p02.models import EntityResolutionSpec


@dataclass(frozen=True)
class WideS3Build:
    """Pre-aggregated S3 endpoint records plus a provenance ledger."""

    endpoint_records: list[EvidenceRecord]
    decision_ledger: pd.DataFrame
    audit: dict[str, object]


def build_wide_adjustment_rows(
    *,
    source_id: str,
    path: Path,
    spec: SourceSpec,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    logical_sources: list[LogicalEvidenceSource],
) -> list[AdjustmentRow]:
    """Expand four wide S1 component columns into registered pre/post rows."""

    if not logical_sources:
        raise ValueError("wide S1 requires logical sources")
    config = logical_sources[0].processor_config
    wide = _mapping(config.get("wide_components"), "S1 wide_components")
    adjustment = _mapping(config.get("audit_adjustment"), "S1 audit_adjustment")
    unaudited_status = _required_text(adjustment.get("unaudited_status"), "unaudited_status")
    audited_status = _required_text(adjustment.get("audited_status"), "audited_status")
    expected_unit = _required_text(adjustment.get("expected_unit"), "expected_unit")
    expected_scope = _required_text(adjustment.get("expected_scope"), "expected_scope")
    firm_column = _semantic_column(semantics, "firm_id")
    year_column = _semantic_column(semantics, "fiscal_year")

    rows: list[AdjustmentRow] = []
    for raw in iter_rows(path, spec):
        firm_id = _canonical_firm(raw, firm_column, source_id, entity)
        fiscal_year = _required_year(raw.get(year_column))
        for source in sorted(logical_sources, key=lambda item: item.source_id):
            binding = _mapping(wide.get(source.source_id), f"S1 wide_components.{source.source_id}")
            pre_semantic = _required_text(
                binding.get("unaudited_semantic"),
                f"{source.source_id}.unaudited_semantic",
            )
            post_semantic = _required_text(
                binding.get("audited_semantic"),
                f"{source.source_id}.audited_semantic",
            )
            pre_column = _semantic_column(semantics, pre_semantic)
            post_column = _semantic_column(semantics, post_semantic)
            canonical_item = _required_text(
                source.logical_config.get("canonical_item"),
                f"{source.source_id}.canonical_item",
            )
            statement_family = _required_text(
                source.logical_config.get("statement_family"),
                f"{source.source_id}.statement_family",
            )
            prefix = f"{source_id}:{firm_id}:{fiscal_year}:{source.source_id}"
            rows.extend(
                [
                    AdjustmentRow(
                        firm_id=firm_id,
                        fiscal_year=fiscal_year,
                        audit_status=unaudited_status,
                        canonical_item=canonical_item,
                        value=_optional_float(raw.get(pre_column)),
                        unit=expected_unit,
                        statement_scope=expected_scope,
                        statement_family=statement_family,
                        source_ref=f"{prefix}:unaudited",
                    ),
                    AdjustmentRow(
                        firm_id=firm_id,
                        fiscal_year=fiscal_year,
                        audit_status=audited_status,
                        canonical_item=canonical_item,
                        value=_optional_float(raw.get(post_column)),
                        unit=expected_unit,
                        statement_scope=expected_scope,
                        statement_family=statement_family,
                        source_ref=f"{prefix}:audited",
                    ),
                ]
            )
    return rows


def build_wide_opinion_rows(
    *,
    source_id: str,
    path: Path,
    spec: SourceSpec,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    logical_source: LogicalEvidenceSource,
) -> list[OpinionRow]:
    """Create S2 opinion rows only when an annual audit report is observed."""

    config = logical_source.processor_config
    wide = _mapping(config.get("wide_opinion"), "S2 wide_opinion")
    opinion = _mapping(config.get("audit_opinion"), "S2 audit_opinion")
    observation_semantic = _required_text(
        wide.get("observation_semantic"), "wide_opinion.observation_semantic"
    )
    opinion_semantic = _required_text(
        wide.get("opinion_semantic"), "wide_opinion.opinion_semantic"
    )
    source_ref_semantic = _optional_text_value(wide.get("source_ref_semantic"))
    firm_column = _semantic_column(semantics, "firm_id")
    year_column = _semantic_column(semantics, "fiscal_year")
    observation_column = _semantic_column(semantics, observation_semantic)
    opinion_column = _semantic_column(semantics, opinion_semantic)
    source_ref_column = (
        _semantic_column(semantics, source_ref_semantic)
        if source_ref_semantic is not None
        else None
    )
    indicator_value = _required_text(opinion.get("indicator_value"), "indicator_value")
    period_type = _required_text(opinion.get("period_type"), "period_type")
    statement_scope = _required_text(opinion.get("statement_scope"), "statement_scope")
    audit_status = _required_text(opinion.get("audited_status"), "audited_status")

    rows: list[OpinionRow] = []
    for raw in iter_rows(path, spec):
        observed = _optional_bool(raw.get(observation_column))
        if observed is not True:
            continue
        firm_id = _canonical_firm(raw, firm_column, source_id, entity)
        fiscal_year = _required_year(raw.get(year_column))
        source_ref = _optional_text_value(raw.get(source_ref_column)) if source_ref_column else None
        rows.append(
            OpinionRow(
                firm_id=firm_id,
                fiscal_year=fiscal_year,
                opinion_raw=_optional_text_value(raw.get(opinion_column)),
                audit_indicator=indicator_value,
                period_type=period_type,
                statement_scope=statement_scope,
                audit_status=audit_status,
                source_ref=source_ref or f"{source_id}:{firm_id}:{fiscal_year}",
            )
        )
    return rows


def build_wide_s3_records(
    *,
    source_id: str,
    path: Path,
    spec: SourceSpec,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    logical_sources: list[LogicalEvidenceSource],
    panel_anchors: dict[tuple[str, int], datetime],
) -> WideS3Build:
    """Read pre-aggregated S3 endpoint incidence without recreating latent negatives."""

    if not logical_sources:
        raise ValueError("wide S3 requires logical sources")
    config = logical_sources[0].processor_config
    wide = _mapping(config.get("wide_endpoints"), "S3 wide_endpoints")
    endpoint_semantics = _mapping(wide.get("endpoint_semantics"), "S3 endpoint_semantics")
    opportunity_semantic = _required_text(
        wide.get("source_opportunity_semantic"),
        "wide_endpoints.source_opportunity_semantic",
    )
    document_semantic = _required_text(
        wide.get("document_ids_semantic"), "wide_endpoints.document_ids_semantic"
    )
    first_date_semantic = _required_text(
        wide.get("first_label_known_date_semantic"),
        "wide_endpoints.first_label_known_date_semantic",
    )
    last_date_semantic = _required_text(
        wide.get("last_label_known_date_semantic"),
        "wide_endpoints.last_label_known_date_semantic",
    )
    taxonomy_semantic = _required_text(
        wide.get("taxonomy_codes_semantic"),
        "wide_endpoints.taxonomy_codes_semantic",
    )
    firm_column = _semantic_column(semantics, "firm_id")
    year_column = _semantic_column(semantics, "fiscal_year")
    opportunity_column = _semantic_column(semantics, opportunity_semantic)
    document_column = _semantic_column(semantics, document_semantic)
    first_date_column = _semantic_column(semantics, first_date_semantic)
    last_date_column = _semantic_column(semantics, last_date_semantic)
    taxonomy_column = _semantic_column(semantics, taxonomy_semantic)
    endpoint_columns = {
        logical_id: _semantic_column(
            semantics,
            _required_text(endpoint_semantics.get(logical_id), f"endpoint_semantics.{logical_id}"),
        )
        for logical_id in sorted(endpoint_semantics)
    }

    raw_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for raw in iter_rows(path, spec):
        firm_id = _canonical_firm(raw, firm_column, source_id, entity)
        fiscal_year = _required_year(raw.get(year_column))
        key = (firm_id, fiscal_year)
        if key in raw_by_key:
            raise ValueError(f"wide S3 duplicate firm-year={key}")
        raw_by_key[key] = raw

    records: list[EvidenceRecord] = []
    decision_rows: list[dict[str, object]] = []
    positive_count = 0
    negative_count = 0
    unknown_count = 0
    seen_decision_keys: set[tuple[str, str]] = set()
    logical_by_id = {source.source_id: source for source in logical_sources}
    if set(logical_by_id) != set(endpoint_columns):
        raise ValueError("S3 endpoint semantic bindings differ from logical source registry")

    for (firm_id, fiscal_year), _anchor in sorted(panel_anchors.items()):
        raw = raw_by_key.get((firm_id, fiscal_year))
        if raw is None:
            raise ValueError(f"final firm-year input is missing panel row={(firm_id, fiscal_year)}")
        opportunity = _optional_bool(raw.get(opportunity_column))
        document_ids = _json_string_list(raw.get(document_column), document_semantic)
        taxonomy_codes = _json_string_list(raw.get(taxonomy_column), taxonomy_semantic)
        first_date = _optional_datetime(raw.get(first_date_column))
        last_date = _optional_datetime(raw.get(last_date_column))
        endpoint_values = {
            logical_id: _optional_bool(raw.get(column))
            for logical_id, column in endpoint_columns.items()
        }
        if opportunity is True and any(value is None for value in endpoint_values.values()):
            missing = sorted(key for key, value in endpoint_values.items() if value is None)
            raise ValueError(
                f"complete S3 source year has missing endpoint values for {(firm_id, fiscal_year)}: {missing}"
            )
        if opportunity is not True and any(value is not None for value in endpoint_values.values()):
            raise ValueError(
                f"S3 endpoint value observed without source opportunity for {(firm_id, fiscal_year)}"
            )
        positive_endpoints = sorted(key for key, value in endpoint_values.items() if value is True)
        if positive_endpoints and not document_ids:
            raise ValueError(
                f"positive S3 endpoint requires document provenance for {(firm_id, fiscal_year)}"
            )

        for logical_id in sorted(logical_by_id):
            source = logical_by_id[logical_id]
            outcome = endpoint_values[logical_id]
            if outcome is True:
                positive_count += 1
                reason_code = "S3_PUBLIC_ENDPOINT_POSITIVE"
            elif outcome is False:
                negative_count += 1
                reason_code = "S3_COMPLETE_SOURCE_YEAR_ENDPOINT_ZERO"
            else:
                unknown_count += 1
                reason_code = "S3_SOURCE_YEAR_INCOMPLETE_OR_UNKNOWN"
            fully_observed_positive = outcome is True
            records.append(
                EvidenceRecord(
                    source_id=source.source_id,
                    source_profile_id=source.physical_source_id,
                    channel_id=source.channel_id,
                    firm_id=firm_id,
                    fiscal_year=fiscal_year,
                    availability_date=first_date if fully_observed_positive else None,
                    outcome=outcome,
                    event_id=None,
                    event_cluster_id=None,
                    evidence_record_id=f"{source.source_id}:{firm_id}:{fiscal_year}",
                    evidence_record_kind="firm_year_endpoint_result",
                    temporal_role=source.temporal_role,
                    availability_basis=source.availability_rule,
                    source_opportunity=opportunity,
                    opportunity_basis=(
                        "S3_SOURCE_YEAR_COMPLETE"
                        if opportunity is True
                        else "S3_SOURCE_YEAR_INCOMPLETE_OR_UNKNOWN"
                    ),
                    verification_status=True if fully_observed_positive else None,
                    determination_status=True if fully_observed_positive else None,
                    recording_status=True if fully_observed_positive else None,
                    verification_date=first_date if fully_observed_positive else None,
                    determination_date=first_date if fully_observed_positive else None,
                    recording_date=first_date if fully_observed_positive else None,
                    reason_unknown=reason_code if outcome is None else None,
                    evidence_category=source.endpoint_id,
                    source_record_refs=(
                        json.dumps(document_ids, ensure_ascii=False) if document_ids else None
                    ),
                    period_link_source="precomputed_fiscal_year_endpoint",
                    period_link_confidence="deterministic_final_build",
                    outcome_basis=reason_code,
                    duplicate_representative_rule=str(config["duplicate_representative_rule"]),
                    sanction_year=fiscal_year + 1,
                    target_fiscal_year=fiscal_year,
                    decision_count=len(document_ids),
                    document_ids=json.dumps(document_ids, ensure_ascii=False),
                    first_label_known_date=first_date,
                    last_label_known_date=last_date,
                    taxonomy_codes=json.dumps(taxonomy_codes, ensure_ascii=False),
                    taxonomy_reason_code="FINAL_FIRM_YEAR_AGGREGATED_ENDPOINT",
                )
            )

        for document_id in document_ids:
            decision_key = (document_id, firm_id)
            if decision_key in seen_decision_keys:
                continue
            seen_decision_keys.add(decision_key)
            decision_rows.append(
                {
                    "document_id": document_id,
                    "firm_id": firm_id,
                    "target_fiscal_year": fiscal_year,
                    "primary_violation_l1": None,
                    "primary_violation_l2": None,
                    "construct_family": "AGGREGATED_PUBLIC_ENDPOINT",
                    "construct_target": json.dumps(positive_endpoints, ensure_ascii=False),
                    "normalized_violation_code": None,
                    "hard_positive": bool(positive_endpoints),
                    "row_inclusion": True,
                    "legacy_event_id": None,
                    "period_link_source": "precomputed_fiscal_year_endpoint",
                    "period_link_confidence": "deterministic_final_build",
                    "source_record_refs": json.dumps(document_ids, ensure_ascii=False),
                    "taxonomy_codes": json.dumps(taxonomy_codes, ensure_ascii=False),
                    "taxonomy_reason_code": "FINAL_FIRM_YEAR_AGGREGATED_ENDPOINT",
                }
            )

    ledger = _decision_ledger(decision_rows)
    return WideS3Build(
        endpoint_records=records,
        decision_ledger=ledger,
        audit={
            "processor": "sanction_calendar_year",
            "input_mode": "preaggregated_final_firm_year",
            "physical_source_id": source_id,
            "logical_source_ids": sorted(logical_by_id),
            "panel_firm_year_count": len(panel_anchors),
            "endpoint_record_count": len(records),
            "decision_firm_mapping_count": len(ledger),
            "source_opportunity_count": sum(
                _optional_bool(raw.get(opportunity_column)) is True for raw in raw_by_key.values()
            ),
            "positive_count": positive_count,
            "explicit_endpoint_zero_count": negative_count,
            "unknown_count": unknown_count,
            "missing_is_negative": False,
            "high_specificity_assumed": False,
        },
    )


def _decision_ledger(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "document_id",
        "firm_id",
        "target_fiscal_year",
        "primary_violation_l1",
        "primary_violation_l2",
        "construct_family",
        "construct_target",
        "normalized_violation_code",
        "hard_positive",
        "row_inclusion",
        "legacy_event_id",
        "period_link_source",
        "period_link_confidence",
        "source_record_refs",
        "taxonomy_codes",
        "taxonomy_reason_code",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for column in (
        "document_id",
        "firm_id",
        "primary_violation_l1",
        "primary_violation_l2",
        "construct_family",
        "construct_target",
        "normalized_violation_code",
        "legacy_event_id",
        "period_link_source",
        "period_link_confidence",
        "source_record_refs",
        "taxonomy_codes",
        "taxonomy_reason_code",
    ):
        frame[column] = frame[column].astype("string")
    frame["target_fiscal_year"] = frame["target_fiscal_year"].astype("Int16")
    frame["hard_positive"] = frame["hard_positive"].astype("boolean")
    frame["row_inclusion"] = frame["row_inclusion"].astype("boolean")
    return frame


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _required_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: non-empty string required")
    return value.strip()


def _semantic_column(semantics: dict[str, Any], name: str) -> str:
    value = semantics.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"final input semantic={name}: resolved physical column required")
    return value


def _canonical_firm(
    row: dict[str, object],
    firm_column: str,
    source_id: str,
    entity: EntityResolutionSpec,
) -> str:
    raw = row.get(firm_column)
    if raw is None or not str(raw).strip():
        raise ValueError("final input firm_id is missing")
    raw_text = str(raw)
    normalized = normalize_entity_field(raw_text, entity)
    canonical, _ = resolve_entity_link(source_id, raw_text, normalized, entity)
    return canonical


def _required_year(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("final input fiscal_year is missing")
    year = int(str(value))
    if not 1900 <= year <= 2200:
        raise ValueError(f"final input fiscal_year is invalid: {year}")
    return year


def _optional_text_value(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    text = _optional_text_value(value)
    if text is None:
        return None
    parsed = float(text)
    return parsed if pd.notna(parsed) else None


def _optional_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if not normalized:
        return None
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"final input boolean value is invalid: {value}")


def _optional_datetime(value: object) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip()
    if not text:
        return None
    return pd.Timestamp(text).to_pydatetime()


def _json_string_list(value: object, context: str) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        raw = cast(list[object], value)
    else:
        text = str(value).strip()
        if not text:
            return []
        parsed: object = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"{context}: JSON array required")
        raw = cast(list[object], parsed)
    values = [str(item).strip() for item in raw if str(item).strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"{context}: duplicate values are not allowed")
    return values
