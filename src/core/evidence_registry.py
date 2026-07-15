"""Locked logical evidence-source registry derived from physical snapshot sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from core.semantic_keys import CHANNEL_ID, DOCUMENT_ID, SOURCE_ID, TEMPORAL_ROLE


@dataclass(frozen=True)
class LogicalEvidenceSource:
    source_id: str
    endpoint_id: str
    channel_id: str
    physical_source_id: str
    profile_id: str
    processor: str
    temporal_role: str
    availability_rule: str
    explicit_negative_allowed: bool
    verification_status: str
    opportunity_semantic: str | None
    logical_config: dict[str, Any]
    processor_config: dict[str, Any]
    target_year_rule: str | None = None
    decision_key_semantic: str | None = None


def logical_evidence_sources(registry: dict[str, object]) -> dict[str, LogicalEvidenceSource]:
    """Expand one physical source into its registered source-specific endpoints."""
    data_sources = _mapping(registry.get("data_sources"), "data_sources")
    source_registry = _mapping(data_sources.get("source_registry"), "source_registry")
    physical_sources = _mapping(source_registry.get("sources"), "source_registry.sources")
    result: dict[str, LogicalEvidenceSource] = {}
    for physical_source_id, raw_source in sorted(physical_sources.items()):
        source = _mapping(raw_source, f"source={physical_source_id}")
        if source.get("enabled") is not True or source.get("role") != "evidence":
            continue
        evidence = _mapping(
            source.get("evidence_mapping"), f"source={physical_source_id}.evidence_mapping"
        )
        processor = _string(evidence.get("processor"), "evidence processor")
        temporal_role = _string(evidence.get(TEMPORAL_ROLE), "evidence temporal role")
        availability_rule = _string(evidence.get("availability_rule"), "evidence availability_rule")
        explicit_negative_allowed = evidence.get("explicit_negative_allowed")
        if not isinstance(explicit_negative_allowed, bool):
            raise ValueError(
                f"source={physical_source_id}: explicit_negative_allowed must be boolean"
            )
        raw_logical = evidence.get("logical_sources")
        if not isinstance(raw_logical, list) or not raw_logical:
            raise ValueError(f"source={physical_source_id}: logical_sources required")
        channel_id = _string(source.get(CHANNEL_ID), f"source={physical_source_id}.channel_id")
        profile_id = _string(source.get("profile_id"), f"source={physical_source_id}.profile_id")
        verification = _string(
            source.get("verification_status"), f"source={physical_source_id}.verification_status"
        )
        opportunity_raw = evidence.get("opportunity_semantic")
        opportunity = str(opportunity_raw) if isinstance(opportunity_raw, str) else None
        for raw_logical_source in cast(list[object], raw_logical):
            logical = _mapping(raw_logical_source, f"source={physical_source_id}.logical_source")
            logical_source_id = _string(logical.get(SOURCE_ID), "logical source ID")
            if logical_source_id in result:
                raise ValueError(f"duplicate logical evidence source={logical_source_id}")
            result[logical_source_id] = LogicalEvidenceSource(
                source_id=logical_source_id,
                endpoint_id=_string(logical.get("endpoint_id"), "logical endpoint_id"),
                channel_id=channel_id,
                physical_source_id=str(physical_source_id),
                profile_id=profile_id,
                processor=processor,
                temporal_role=temporal_role,
                availability_rule=availability_rule,
                explicit_negative_allowed=explicit_negative_allowed,
                verification_status=verification,
                opportunity_semantic=opportunity,
                logical_config=dict(logical),
                processor_config=dict(evidence),
                target_year_rule=(
                    str(evidence["target_year_rule"])
                    if isinstance(evidence.get("target_year_rule"), str)
                    else None
                ),
                decision_key_semantic=(
                    str(evidence["decision_key_semantic"])
                    if isinstance(evidence.get("decision_key_semantic"), str)
                    else None
                ),
            )
            if processor == "sanction_calendar_year":
                registered = result[logical_source_id]
                if registered.target_year_rule != "sanction_year_minus_one":
                    raise ValueError("S3 target-year rule is not locked")
                if registered.decision_key_semantic != DOCUMENT_ID:
                    raise ValueError("S3 decision key must be document_id")
    if not result:
        raise ValueError("locked registry contains no logical evidence sources")
    return result


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: non-empty string required")
    return value.strip()
