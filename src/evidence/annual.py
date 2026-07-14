"""Deterministic annual-anchor evidence construction for S1 and S2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from core.evidence_registry import LogicalEvidenceSource
from core.semantic_keys import (
    CHANNEL_ID,
    FIRM_ID,
    FISCAL_YEAR,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    SOURCE_RECORD_REFS,
)
from evidence.service import EvidenceRecord


@dataclass(frozen=True)
class AdjustmentRow:
    firm_id: str
    fiscal_year: int
    audit_status: str
    canonical_item: str
    value: float | None
    unit: str | None
    statement_scope: str | None
    statement_family: str | None
    source_ref: str


@dataclass(frozen=True)
class OpinionRow:
    firm_id: str
    fiscal_year: int
    opinion_raw: str | None
    audit_indicator: str | None
    period_type: str | None
    statement_scope: str | None
    audit_status: str | None
    source_ref: str


@dataclass(frozen=True)
class AnnualEvidenceBuild:
    records: list[EvidenceRecord]
    audit: dict[str, object]
    failures: list[dict[str, object]]


def build_audit_adjustment_records(
    *,
    panel_anchors: dict[tuple[str, int], datetime],
    rows: list[AdjustmentRow],
    sources: list[LogicalEvidenceSource],
) -> AnnualEvidenceBuild:
    """Construct one deterministic matched-pair result per panel row and endpoint."""
    if not sources or any(source.processor != "audit_adjustment" for source in sources):
        raise ValueError("S1 audit-adjustment logical sources are required")
    physical_sources = {source.physical_source_id for source in sources}
    if len(physical_sources) != 1:
        raise ValueError("S1 endpoints must share one physical audit-adjustment source")
    config = sources[0].processor_config
    adjustment = _mapping(config.get("audit_adjustment"), "audit_adjustment")
    unaudited = _text(adjustment.get("unaudited_status"), "unaudited_status")
    audited = _text(adjustment.get("audited_status"), "audited_status")
    expected_unit = _text(adjustment.get("expected_unit"), "expected_unit")
    expected_scope = _text(adjustment.get("expected_scope"), "expected_scope")
    floor_raw = adjustment.get("minimum_absolute_denominator")
    floor = float(floor_raw) if isinstance(floor_raw, (int, float)) else None
    endpoint_by_item = {
        _text(source.logical_config.get("canonical_item"), "canonical_item"): source
        for source in sources
    }
    grouped: dict[tuple[str, int, str, str], list[AdjustmentRow]] = {}
    for row in rows:
        if row.canonical_item not in endpoint_by_item:
            continue
        grouped.setdefault(
            (row.firm_id, row.fiscal_year, row.canonical_item, row.audit_status), []
        ).append(row)

    records: list[EvidenceRecord] = []
    failures: list[dict[str, object]] = []
    for (firm_id, fiscal_year), anchor in sorted(panel_anchors.items()):
        for source in sorted(sources, key=lambda item: item.source_id):
            item = _text(source.logical_config.get("canonical_item"), "canonical_item")
            expected_family = _text(
                source.logical_config.get("statement_family"), "statement_family"
            )
            pre_rows = grouped.get((firm_id, fiscal_year, item, unaudited), [])
            post_rows = grouped.get((firm_id, fiscal_year, item, audited), [])
            result = _evaluate_adjustment_pair(
                pre_rows=pre_rows,
                post_rows=post_rows,
                expected_unit=expected_unit,
                expected_scope=expected_scope,
                expected_family=expected_family,
                denominator_floor=floor,
                materiality_threshold=_optional_nonnegative(
                    source.logical_config.get("materiality_threshold"),
                    "materiality_threshold",
                ),
            )
            record_id = f"{source.source_id}:{firm_id}:{fiscal_year}"
            records.append(
                EvidenceRecord(
                    source_id=source.source_id,
                    source_profile_id=source.physical_source_id,
                    channel_id=source.channel_id,
                    firm_id=firm_id,
                    fiscal_year=fiscal_year,
                    availability_date=anchor,
                    outcome=result.outcome,
                    event_id=None,
                    event_cluster_id=None,
                    evidence_record_id=record_id,
                    evidence_record_kind="annual_source_result",
                    temporal_role=source.temporal_role,
                    availability_basis=source.availability_rule,
                    source_opportunity=result.opportunity,
                    opportunity_basis=result.reason_code,
                    evidence_value=result.ratio,
                    evidence_category=None,
                    source_record_refs=result.source_refs,
                    period_link_source="exact_fiscal_year",
                    period_link_confidence="deterministic",
                    outcome_basis=result.reason_code,
                    duplicate_representative_rule=str(config["duplicate_representative_rule"]),
                )
            )
            if result.outcome is None:
                failures.append(
                    {
                        SOURCE_ID: source.source_id,
                        CHANNEL_ID: source.channel_id,
                        FIRM_ID: firm_id,
                        FISCAL_YEAR: fiscal_year,
                        "endpoint_id": source.endpoint_id,
                        "canonical_item": item,
                        "reason_code": result.reason_code,
                        SOURCE_OPPORTUNITY: result.opportunity,
                        "pre_record_count": len(pre_rows),
                        "post_record_count": len(post_rows),
                        SOURCE_RECORD_REFS: result.source_refs,
                    }
                )
    return AnnualEvidenceBuild(
        records=records,
        failures=failures,
        audit={
            "processor": "audit_adjustment",
            "physical_source_id": next(iter(physical_sources)),
            "logical_source_ids": sorted(source.source_id for source in sources),
            "panel_firm_year_count": len(panel_anchors),
            "raw_endpoint_row_count": sum(row.canonical_item in endpoint_by_item for row in rows),
            "result_record_count": len(records),
            "source_opportunity_observed_count": sum(
                record.source_opportunity is True for record in records
            ),
            "positive_count": sum(record.outcome is True for record in records),
            "explicit_negative_count": sum(record.outcome is False for record in records),
            "unknown_count": sum(record.outcome is None for record in records),
            "materiality_thresholds_locked": all(
                source.logical_config.get("materiality_threshold") is not None for source in sources
            ),
            "denominator_floor_locked": floor is not None,
            "missing_is_negative": False,
        },
    )


@dataclass(frozen=True)
class _PairResult:
    outcome: bool | None
    opportunity: bool
    ratio: float | None
    reason_code: str
    source_refs: str | None


def _evaluate_adjustment_pair(
    *,
    pre_rows: list[AdjustmentRow],
    post_rows: list[AdjustmentRow],
    expected_unit: str,
    expected_scope: str,
    expected_family: str,
    denominator_floor: float | None,
    materiality_threshold: float | None,
) -> _PairResult:
    refs = _source_refs([*pre_rows, *post_rows])
    if not pre_rows and not post_rows:
        return _PairResult(None, False, None, "S1_PAIR_MISSING_BOTH", refs)
    if not pre_rows:
        return _PairResult(None, False, None, "S1_PAIR_MISSING_PRE_AUDIT", refs)
    if not post_rows:
        return _PairResult(None, False, None, "S1_PAIR_MISSING_POST_AUDIT", refs)
    if len(pre_rows) > 1 or len(post_rows) > 1:
        conflicting = _conflicting_adjustment_rows(pre_rows) or _conflicting_adjustment_rows(
            post_rows
        )
        return _PairResult(
            None,
            False,
            None,
            "S1_DUPLICATE_CONFLICT" if conflicting else "S1_DUPLICATE_STATUS_RECORD",
            refs,
        )
    pre = pre_rows[0]
    post = post_rows[0]
    if pre.value is None or post.value is None:
        return _PairResult(None, False, None, "S1_PAIR_VALUE_MISSING", refs)
    if pre.unit != expected_unit or post.unit != expected_unit or pre.unit != post.unit:
        return _PairResult(None, False, None, "S1_INCONSISTENT_UNIT", refs)
    if pre.statement_scope != expected_scope or post.statement_scope != expected_scope:
        return _PairResult(None, False, None, "S1_PERIOD_OR_SCOPE_MISMATCH", refs)
    if pre.statement_family != expected_family or post.statement_family != expected_family:
        return _PairResult(None, False, None, "S1_PERIOD_OR_FAMILY_MISMATCH", refs)
    denominator = abs(post.value)
    if denominator == 0:
        return _PairResult(None, False, None, "S1_INVALID_DENOMINATOR_ZERO", refs)
    ratio = abs(pre.value - post.value) / denominator
    if denominator_floor is not None and denominator <= denominator_floor:
        return _PairResult(None, False, ratio, "S1_INVALID_DENOMINATOR_BELOW_FLOOR", refs)
    if denominator_floor is None or materiality_threshold is None:
        return _PairResult(None, True, ratio, "S1_EMPIRICAL_RULE_NOT_LOCKED", refs)
    return _PairResult(
        ratio > materiality_threshold,
        True,
        ratio,
        "S1_MATERIAL_ADJUSTMENT"
        if ratio > materiality_threshold
        else "S1_BELOW_MATERIALITY_THRESHOLD",
        refs,
    )


def build_audit_opinion_records(
    *,
    panel_anchors: dict[tuple[str, int], datetime],
    rows: list[OpinionRow],
    source: LogicalEvidenceSource,
) -> AnnualEvidenceBuild:
    """Normalize one annual opinion result per panel row without using audit firm as outcome."""
    if source.processor != "audit_opinion":
        raise ValueError("S2 audit-opinion logical source required")
    config = source.processor_config
    opinion_config = _mapping(config.get("audit_opinion"), "audit_opinion")
    indicator_value = _text(opinion_config.get("indicator_value"), "indicator_value")
    expected_period = _text(opinion_config.get("period_type"), "period_type")
    expected_scope = _text(opinion_config.get("statement_scope"), "statement_scope")
    expected_status = _text(opinion_config.get("audited_status"), "audited_status")
    raw_taxonomy = _mapping(opinion_config.get("raw_to_normalized"), "raw_to_normalized")
    taxonomy = {
        _normalize_text(str(raw)): _text(category, "normalized category")
        for raw, category in raw_taxonomy.items()
    }
    positive = {
        str(item)
        for item in _list(opinion_config.get("positive_categories"), "positive_categories")
    }
    negative = {
        str(item)
        for item in _list(
            opinion_config.get("explicit_negative_categories"),
            "explicit_negative_categories",
        )
    }
    pending = {
        str(item)
        for item in _list(opinion_config.get("pending_categories", []), "pending_categories")
    }
    grouped: dict[tuple[str, int], list[OpinionRow]] = {}
    for row in rows:
        if row.audit_indicator == indicator_value:
            grouped.setdefault((row.firm_id, row.fiscal_year), []).append(row)
    records: list[EvidenceRecord] = []
    failures: list[dict[str, object]] = []
    for (firm_id, fiscal_year), anchor in sorted(panel_anchors.items()):
        group = grouped.get((firm_id, fiscal_year), [])
        evaluated = _evaluate_opinion(
            rows=group,
            taxonomy=taxonomy,
            positive_categories=positive,
            negative_categories=negative,
            pending_categories=pending,
            expected_period=expected_period,
            expected_scope=expected_scope,
            expected_status=expected_status,
        )
        record_id = f"{source.source_id}:{firm_id}:{fiscal_year}"
        records.append(
            EvidenceRecord(
                source_id=source.source_id,
                source_profile_id=source.physical_source_id,
                channel_id=source.channel_id,
                firm_id=firm_id,
                fiscal_year=fiscal_year,
                availability_date=anchor,
                outcome=evaluated.outcome,
                event_id=None,
                event_cluster_id=None,
                evidence_record_id=record_id,
                evidence_record_kind="annual_source_result",
                temporal_role=source.temporal_role,
                availability_basis=source.availability_rule,
                source_opportunity=evaluated.opportunity,
                opportunity_basis=evaluated.reason_code,
                evidence_value=None,
                evidence_category=evaluated.category,
                source_record_refs=evaluated.source_refs,
                period_link_source="exact_fiscal_year",
                period_link_confidence="deterministic",
                outcome_basis=evaluated.reason_code,
                duplicate_representative_rule=str(config["duplicate_representative_rule"]),
            )
        )
        if evaluated.outcome is None:
            failures.append(
                {
                    SOURCE_ID: source.source_id,
                    CHANNEL_ID: source.channel_id,
                    FIRM_ID: firm_id,
                    FISCAL_YEAR: fiscal_year,
                    "raw_values": " | ".join(
                        sorted({str(row.opinion_raw) for row in group if row.opinion_raw})
                    ),
                    "normalized_category": evaluated.category,
                    "reason_code": evaluated.reason_code,
                    SOURCE_OPPORTUNITY: evaluated.opportunity,
                    "record_count": len(group),
                    SOURCE_RECORD_REFS: evaluated.source_refs,
                }
            )
    return AnnualEvidenceBuild(
        records=records,
        failures=failures,
        audit={
            "processor": "audit_opinion",
            "physical_source_id": source.physical_source_id,
            "logical_source_ids": [source.source_id],
            "panel_firm_year_count": len(panel_anchors),
            "raw_opinion_row_count": sum(row.audit_indicator == indicator_value for row in rows),
            "result_record_count": len(records),
            "source_opportunity_observed_count": sum(
                record.source_opportunity is True for record in records
            ),
            "positive_count": sum(record.outcome is True for record in records),
            "explicit_negative_count": sum(record.outcome is False for record in records),
            "unknown_count": sum(record.outcome is None for record in records),
            "audit_firm_used_as_outcome": False,
            "missing_is_negative": False,
        },
    )


@dataclass(frozen=True)
class _OpinionResult:
    outcome: bool | None
    opportunity: bool
    category: str | None
    reason_code: str
    source_refs: str | None


def _evaluate_opinion(
    *,
    rows: list[OpinionRow],
    taxonomy: dict[str, str],
    positive_categories: set[str],
    negative_categories: set[str],
    pending_categories: set[str],
    expected_period: str,
    expected_scope: str,
    expected_status: str,
) -> _OpinionResult:
    refs = _opinion_source_refs(rows)
    if not rows:
        return _OpinionResult(None, False, None, "S2_OPINION_RECORD_MISSING", refs)
    if any(
        row.period_type != expected_period
        or row.statement_scope != expected_scope
        or row.audit_status != expected_status
        for row in rows
    ):
        return _OpinionResult(None, False, None, "S2_PERIOD_OR_SCOPE_MISMATCH", refs)
    normalized: list[str | None] = []
    for row in rows:
        if row.opinion_raw is None or not row.opinion_raw.strip():
            normalized.append(None)
        else:
            normalized.append(taxonomy.get(_normalize_text(row.opinion_raw)))
    categories = {category for category in normalized if category is not None}
    has_unmapped = any(
        row.opinion_raw is not None and row.opinion_raw.strip() and category is None
        for row, category in zip(rows, normalized, strict=True)
    )
    if has_unmapped:
        return _OpinionResult(None, False, None, "S2_OPINION_UNMAPPED", refs)
    if not categories:
        return _OpinionResult(None, False, None, "S2_OPINION_MISSING", refs)
    if len(categories) > 1 or (len(rows) > 1 and any(category is None for category in normalized)):
        return _OpinionResult(None, False, None, "S2_CONFLICTING_OPINIONS", refs)
    category = next(iter(categories))
    if category in positive_categories:
        return _OpinionResult(True, True, category, "S2_NON_CLEAN_OPINION", refs)
    if category in negative_categories:
        return _OpinionResult(False, True, category, "S2_CLEAN_OPINION", refs)
    if category in pending_categories:
        return _OpinionResult(None, True, category, "S2_CATEGORY_EMPIRICALLY_PENDING", refs)
    return _OpinionResult(None, False, category, "S2_NORMALIZED_CATEGORY_UNMAPPED", refs)


def _conflicting_adjustment_rows(rows: list[AdjustmentRow]) -> bool:
    signatures = {(row.value, row.unit, row.statement_scope, row.statement_family) for row in rows}
    return len(signatures) > 1


def _source_refs(rows: list[AdjustmentRow]) -> str | None:
    refs = sorted({row.source_ref for row in rows if row.source_ref})
    return " | ".join(refs) if refs else None


def _opinion_source_refs(rows: list[OpinionRow]) -> str | None:
    refs = sorted({row.source_ref for row in rows if row.source_ref})
    return " | ".join(refs) if refs else None


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return {str(key): item for key, item in cast(dict[object, object], value).items()}


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: list required")
    return cast(list[object], value)


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: non-empty string required")
    return value.strip()


def _optional_nonnegative(value: object, context: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
        raise ValueError(f"{context}: nonnegative number or null required")
    return float(value)
