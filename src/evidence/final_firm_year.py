"""Semantic views over the single final firm-year Parquet input.

The production file is wide at ``firm_master_id × fiscal_year``.  This module
binds the actual final-build column names to the existing S1, S2, and S3
measurement contracts without recreating latent negatives or reading any
additional physical source file.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class _ProvenanceMeta:
    canonical_json: str | None
    document_ids: tuple[str, ...]
    dates: tuple[datetime, ...]
    taxonomy_codes: tuple[str, ...]


def build_wide_adjustment_rows(
    *,
    source_id: str,
    path: Path,
    spec: SourceSpec,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    logical_sources: list[LogicalEvidenceSource],
) -> list[AdjustmentRow]:
    """Expand final-file PAT and revenue columns into registered pre/post rows."""

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
    """Create S2 rows from the final build's indicator/opinion fields.

    ``s2_audit_indicator`` may be a boolean observation flag or the retained
    indicator token ``audit_opinion``. Both representations are interpreted
    explicitly; arbitrary non-empty strings are rejected.
    """

    config = logical_source.processor_config
    wide = _mapping(config.get("wide_opinion"), "S2 wide_opinion")
    opinion = _mapping(config.get("audit_opinion"), "S2 audit_opinion")
    observation_semantic = _required_text(
        wide.get("observation_semantic"), "wide_opinion.observation_semantic"
    )
    opinion_semantic = _required_text(wide.get("opinion_semantic"), "wide_opinion.opinion_semantic")
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
        raw_opinion = _optional_text_value(raw.get(opinion_column))
        observed = _audit_indicator_observed(raw.get(observation_column), indicator_value)
        if not observed:
            if raw_opinion is not None:
                raise ValueError("S2 opinion is present while the audit indicator is not observed")
            continue
        firm_id = _canonical_firm(raw, firm_column, source_id, entity)
        fiscal_year = _required_year(raw.get(year_column))
        source_ref = _optional_text_value(raw.get(source_ref_column)) if source_ref_column else None
        rows.append(
            OpinionRow(
                firm_id=firm_id,
                fiscal_year=fiscal_year,
                opinion_raw=raw_opinion,
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
    """Read endpoint-specific S3 opportunity, outcome, and provenance columns."""

    if not logical_sources:
        raise ValueError("wide S3 requires logical sources")
    config = logical_sources[0].processor_config
    wide = _mapping(config.get("wide_endpoints"), "S3 wide_endpoints")
    raw_bindings = _mapping(wide.get("endpoint_bindings"), "S3 endpoint_bindings")
    firm_column = _semantic_column(semantics, "firm_id")
    year_column = _semantic_column(semantics, "fiscal_year")

    logical_by_id = {source.source_id: source for source in logical_sources}
    if set(logical_by_id) != set(raw_bindings):
        raise ValueError("S3 endpoint bindings differ from the logical source registry")

    bindings: dict[str, dict[str, str]] = {}
    for logical_id in sorted(logical_by_id):
        raw_binding = _mapping(raw_bindings.get(logical_id), f"S3 binding={logical_id}")
        binding: dict[str, str] = {}
        for field in (
            "outcome_semantic",
            "source_opportunity_semantic",
            "observation_opportunity_semantic",
            "provenance_semantic",
        ):
            semantic = _required_text(raw_binding.get(field), f"{logical_id}.{field}")
            binding[field] = _semantic_column(semantics, semantic)
        bindings[logical_id] = binding

    raw_by_key: dict[tuple[str, int], dict[str, object]] = {}
    for raw in iter_rows(path, spec):
        firm_id = _canonical_firm(raw, firm_column, source_id, entity)
        fiscal_year = _required_year(raw.get(year_column))
        key = (firm_id, fiscal_year)
        if key in raw_by_key:
            raise ValueError(f"wide S3 duplicate firm-year={key}")
        raw_by_key[key] = raw

    records: list[EvidenceRecord] = []
    positive_count = 0
    negative_count = 0
    unknown_count = 0
    opportunity_count = 0
    undetermined_observed_count = 0
    opportunity_by_endpoint = {logical_id: 0 for logical_id in logical_by_id}
    decision_map: dict[tuple[str, str, int], dict[str, object]] = {}

    for (firm_id, fiscal_year), _anchor in sorted(panel_anchors.items()):
        raw = raw_by_key.get((firm_id, fiscal_year))
        if raw is None:
            raise ValueError(f"final firm-year input is missing panel row={(firm_id, fiscal_year)}")

        for logical_id in sorted(logical_by_id):
            source = logical_by_id[logical_id]
            binding = bindings[logical_id]
            source_opportunity = _optional_bool(raw.get(binding["source_opportunity_semantic"]))
            observation_opportunity = _optional_bool(
                raw.get(binding["observation_opportunity_semantic"])
            )
            effective_opportunity = _effective_opportunity(
                source_opportunity,
                observation_opportunity,
            )
            raw_outcome = _optional_bool(raw.get(binding["outcome_semantic"]))
            provenance = _provenance_meta(
                raw.get(binding["provenance_semantic"]),
                f"{logical_id} provenance",
            )

            if effective_opportunity is True:
                opportunity_count += 1
                opportunity_by_endpoint[logical_id] += 1

            outcome, reason_code = _resolve_wide_s3_outcome(
                effective_opportunity=effective_opportunity,
                raw_outcome=raw_outcome,
                has_provenance=provenance.canonical_json is not None,
                context=(firm_id, fiscal_year, logical_id),
            )
            if reason_code == "S3_ENDPOINT_UNDETERMINED_WITH_OBSERVED_PROVENANCE":
                undetermined_observed_count += 1

            if outcome is True and provenance.canonical_json is None:
                raise ValueError(
                    f"positive S3 endpoint requires provenance for "
                    f"{(firm_id, fiscal_year, logical_id)}"
                )
            if outcome is not True and provenance.canonical_json is not None:
                # Provenance may document coverage or an observed decision whose
                # narrower endpoint remains undetermined. It never implies a zero.
                pass

            if outcome is True:
                positive_count += 1
            elif outcome is False:
                negative_count += 1
            else:
                unknown_count += 1

            retain_decision_metadata = outcome is True or (
                outcome is None
                and effective_opportunity is True
                and provenance.canonical_json is not None
            )
            document_ids = list(provenance.document_ids)
            if retain_decision_metadata and not document_ids:
                assert provenance.canonical_json is not None
                document_ids = [_derived_provenance_key(provenance.canonical_json)]
            first_date = min(provenance.dates) if provenance.dates else None
            last_date = max(provenance.dates) if provenance.dates else None
            fully_observed_positive = outcome is True
            records.append(
                EvidenceRecord(
                    source_id=source.source_id,
                    source_profile_id=source.physical_source_id,
                    channel_id=source.channel_id,
                    firm_id=firm_id,
                    fiscal_year=fiscal_year,
                    availability_date=first_date if retain_decision_metadata else None,
                    outcome=outcome,
                    event_id=None,
                    event_cluster_id=None,
                    evidence_record_id=f"{source.source_id}:{firm_id}:{fiscal_year}",
                    evidence_record_kind="firm_year_endpoint_result",
                    temporal_role=source.temporal_role,
                    availability_basis=source.availability_rule,
                    source_opportunity=effective_opportunity,
                    opportunity_basis=(
                        "S3_ENDPOINT_OPPORTUNITY_OBSERVED"
                        if effective_opportunity is True
                        else "S3_ENDPOINT_OPPORTUNITY_FALSE_OR_UNKNOWN"
                    ),
                    verification_status=True if fully_observed_positive else None,
                    determination_status=True if fully_observed_positive else None,
                    recording_status=True if fully_observed_positive else None,
                    verification_date=first_date if fully_observed_positive else None,
                    determination_date=first_date if fully_observed_positive else None,
                    recording_date=first_date if fully_observed_positive else None,
                    reason_unknown=reason_code if outcome is None else None,
                    evidence_category=source.endpoint_id,
                    source_record_refs=provenance.canonical_json,
                    period_link_source="precomputed_fiscal_year_endpoint",
                    period_link_confidence="deterministic_final_build",
                    outcome_basis=reason_code,
                    duplicate_representative_rule=str(config["duplicate_representative_rule"]),
                    sanction_year=fiscal_year + 1,
                    target_fiscal_year=fiscal_year,
                    decision_count=len(document_ids) if retain_decision_metadata else 0,
                    document_ids=json.dumps(document_ids, ensure_ascii=False),
                    first_label_known_date=first_date,
                    last_label_known_date=last_date,
                    taxonomy_codes=json.dumps(list(provenance.taxonomy_codes), ensure_ascii=False),
                    taxonomy_reason_code="FINAL_ENDPOINT_PROVENANCE_JSON",
                )
            )

            if outcome is True:
                for document_id in document_ids:
                    key = (document_id, firm_id, fiscal_year)
                    item = decision_map.setdefault(
                        key,
                        {
                            "document_id": document_id,
                            "firm_id": firm_id,
                            "target_fiscal_year": fiscal_year,
                            "endpoints": set(),
                            "source_refs": set(),
                            "taxonomy_codes": set(),
                        },
                    )
                    cast(set[str], item["endpoints"]).add(source.endpoint_id)
                    if provenance.canonical_json is not None:
                        cast(set[str], item["source_refs"]).add(provenance.canonical_json)
                    cast(set[str], item["taxonomy_codes"]).update(provenance.taxonomy_codes)

    decision_rows: list[dict[str, object]] = []
    for key in sorted(decision_map):
        item = decision_map[key]
        endpoints = sorted(cast(set[str], item["endpoints"]))
        refs = sorted(cast(set[str], item["source_refs"]))
        taxonomy = sorted(cast(set[str], item["taxonomy_codes"]))
        decision_rows.append(
            {
                "document_id": item["document_id"],
                "firm_id": item["firm_id"],
                "target_fiscal_year": item["target_fiscal_year"],
                "primary_violation_l1": None,
                "primary_violation_l2": None,
                "construct_family": "AGGREGATED_PUBLIC_ENDPOINT",
                "construct_target": json.dumps(endpoints, ensure_ascii=False),
                "normalized_violation_code": None,
                "hard_positive": True,
                "row_inclusion": True,
                "legacy_event_id": None,
                "period_link_source": "precomputed_fiscal_year_endpoint",
                "period_link_confidence": "deterministic_final_build",
                "source_record_refs": json.dumps(refs, ensure_ascii=False),
                "taxonomy_codes": json.dumps(taxonomy, ensure_ascii=False),
                "taxonomy_reason_code": "FINAL_ENDPOINT_PROVENANCE_JSON",
            }
        )

    ledger = _decision_ledger(decision_rows)
    return WideS3Build(
        endpoint_records=records,
        decision_ledger=ledger,
        audit={
            "processor": "sanction_calendar_year",
            "input_mode": "preaggregated_endpoint_specific_final_firm_year",
            "physical_source_id": source_id,
            "logical_source_ids": sorted(logical_by_id),
            "panel_firm_year_count": len(panel_anchors),
            "endpoint_record_count": len(records),
            "decision_firm_mapping_count": len(ledger),
            "source_opportunity_count": opportunity_count,
            "source_opportunity_count_by_endpoint": opportunity_by_endpoint,
            "positive_count": positive_count,
            "explicit_endpoint_zero_count": negative_count,
            "unknown_count": unknown_count,
            "undetermined_endpoint_with_observed_opportunity_count": undetermined_observed_count,
            "stored_zero_without_opportunity_is_unknown": True,
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
    string_columns = [
        column
        for column in columns
        if column not in {"target_fiscal_year", "hard_positive", "row_inclusion"}
    ]
    for column in string_columns:
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
    numeric = float(str(value))
    if not numeric.is_integer():
        raise ValueError(f"final input fiscal_year is invalid: {value}")
    year = int(numeric)
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


def _audit_indicator_observed(value: object, indicator_value: str) -> bool:
    try:
        parsed = _optional_bool(value)
    except ValueError:
        parsed = None
    if parsed is not None:
        return parsed
    text = _optional_text_value(value)
    if text is None:
        return False
    if _normalize_token(text) == _normalize_token(indicator_value):
        return True
    raise ValueError(
        f"S2 audit indicator={value!r} is neither boolean nor the registered token "
        f"{indicator_value!r}"
    )


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _effective_opportunity(
    source_opportunity: bool | None,
    observation_opportunity: bool | None,
) -> bool | None:
    """Return complete S3 opportunity only when both layers are observed.

    A complete public-source year is not, by itself, evidence that a particular
    firm-year endpoint had an observation opportunity. Likewise, a firm-level
    observation flag cannot establish source-year completeness. Therefore the
    observed endpoint zero is admissible only when both indicators are true.
    """

    if source_opportunity is False or observation_opportunity is False:
        return False
    if source_opportunity is True and observation_opportunity is True:
        return True
    return None


def _resolve_wide_s3_outcome(
    *,
    effective_opportunity: bool | None,
    raw_outcome: bool | None,
    has_provenance: bool,
    context: tuple[str, int, str],
) -> tuple[bool | None, str]:
    """Preserve determination uncertainty separately from opportunity.

    The canonical S3 taxonomy permits a complete source-year observation to
    remain unknown for a narrower endpoint when a public decision cannot be
    mapped to that endpoint. Such rows must retain provenance and must never be
    coerced to an observed zero. Missing outcomes without provenance remain a
    fail-closed data-contract violation.
    """

    if effective_opportunity is True:
        if raw_outcome is True:
            return True, "S3_PUBLIC_ENDPOINT_POSITIVE"
        if raw_outcome is False:
            return False, "S3_COMPLETE_SOURCE_YEAR_ENDPOINT_ZERO"
        if not has_provenance:
            raise ValueError(
                f"observed S3 opportunity has missing endpoint without provenance for {context}"
            )
        return None, "S3_ENDPOINT_UNDETERMINED_WITH_OBSERVED_PROVENANCE"

    if raw_outcome is True:
        raise ValueError(f"positive S3 endpoint lacks observation opportunity for {context}")
    # A stored zero outside an observed opportunity remains unknown.
    return None, "S3_SOURCE_OR_OBSERVATION_OPPORTUNITY_UNKNOWN"


def _provenance_meta(value: object, context: str) -> _ProvenanceMeta:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return _ProvenanceMeta(None, (), (), ())
    raw: object
    if isinstance(value, dict):
        raw = cast(dict[object, object], value)
    elif isinstance(value, list):
        raw = cast(list[object], value)
    else:
        text = str(value).strip()
        if not text:
            return _ProvenanceMeta(None, (), (), ())
        try:
            raw = cast(object, json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: valid JSON required") from exc
    if raw in ({}, []):
        return _ProvenanceMeta(None, (), (), ())
    canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document_ids: set[str] = set()
    dates: set[datetime] = set()
    taxonomy_codes: set[str] = set()
    _collect_provenance(raw, None, document_ids, dates, taxonomy_codes)
    return _ProvenanceMeta(
        canonical_json=canonical,
        document_ids=tuple(sorted(document_ids)),
        dates=tuple(sorted(dates)),
        taxonomy_codes=tuple(sorted(taxonomy_codes)),
    )


def _collect_provenance(
    value: object,
    parent_key: str | None,
    document_ids: set[str],
    dates: set[datetime],
    taxonomy_codes: set[str],
) -> None:
    if isinstance(value, dict):
        for key_raw, child in cast(dict[object, object], value).items():
            key = _normalize_token(str(key_raw))
            if key in {
                "documentid",
                "documentids",
                "decisionid",
                "decisionids",
                "decisionnumber",
                "decisionnumbers",
            }:
                document_ids.update(_scalar_strings(child))
            if key in {
                "labelknowndate",
                "firstlabelknowndate",
                "lastlabelknowndate",
                "decisiondate",
                "publishdate",
            }:
                dates.update(_date_values(child))
            if key in {
                "taxonomycode",
                "taxonomycodes",
                "normalizedviolationcode",
                "normalizedviolationcodes",
            }:
                taxonomy_codes.update(_scalar_strings(child))
            _collect_provenance(child, key, document_ids, dates, taxonomy_codes)
        return
    if isinstance(value, list):
        for child in cast(list[object], value):
            _collect_provenance(child, parent_key, document_ids, dates, taxonomy_codes)


def _scalar_strings(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, list):
        result: set[str] = set()
        for item in cast(list[object], value):
            result.update(_scalar_strings(item))
        return result
    text = str(value).strip()
    return {text} if text else set()


def _date_values(value: object) -> set[datetime]:
    result: set[datetime] = set()
    for text in _scalar_strings(value):
        parsed = _parse_datetime(text)
        if parsed is not None:
            result.add(parsed)
    return result


def _parse_datetime(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    try:
        timestamp = pd.Timestamp(text)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def _derived_provenance_key(canonical_json: str) -> str:
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]
    return f"embedded-provenance-{digest}"
