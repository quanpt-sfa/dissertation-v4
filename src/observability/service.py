"""Pure descriptive observability summaries; analytical weights are prohibited here."""

from __future__ import annotations

from typing import Any, cast

from core.semantic_keys import CHANNEL_ID, ELIGIBLE, MATURE


def build_observability_registry(
    matrices: dict[str, Any], source_metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows_raw = matrices.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("source matrices rows are required")
    rows_raw = cast(list[Any], rows_raw)
    rows = [cast(dict[str, Any], row) for row in rows_raw if isinstance(row, dict)]
    eligible_count = sum(row.get(ELIGIBLE, True) is True for row in rows)
    mature_count = sum(
        row.get(ELIGIBLE, True) is True and row.get(MATURE, True) is True for row in rows
    )
    prospective_count = sum(
        row.get(ELIGIBLE, True) is True and row.get(MATURE, True) is not True for row in rows
    )
    channels: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for source_id, metadata in sorted(source_metadata.items()):
        channel_id = metadata.get(CHANNEL_ID)
        if not isinstance(channel_id, str):
            raise ValueError(f"source={source_id}: channel_id required")
        status = str(metadata.get("verification_status", "unknown"))
        verification = _verification_classification(status)
        evidence_mapping = metadata.get("evidence_mapping")
        opportunity_semantic = (
            cast(dict[str, Any], evidence_mapping).get("opportunity_semantic")
            if isinstance(evidence_mapping, dict)
            else None
        )
        source_entry = _empty_observability_entry(
            eligible_count=eligible_count,
            mature_count=mature_count,
            prospective_count=prospective_count,
            verification=verification,
        )
        source_entry.update(
            {
                CHANNEL_ID: channel_id,
                "opportunity_semantic": opportunity_semantic,
                "source_opportunity_coverage_rate": None,
                "source_opportunity_coverage_status": (
                    "UNAVAILABLE_NOT_MATERIALIZED"
                    if isinstance(opportunity_semantic, str)
                    else "UNKNOWN_NO_OPPORTUNITY_INDICATOR"
                ),
                "source_opportunity_observed_count": 0,
                "source_opportunity_unknown_count": eligible_count,
            }
        )
        sources[source_id] = source_entry
        entry = channels.setdefault(
            channel_id,
            {
                "source_ids": [],
                **_empty_observability_entry(
                    eligible_count=eligible_count,
                    mature_count=mature_count,
                    prospective_count=prospective_count,
                    verification="unknown",
                ),
            },
        )
        source_ids = entry["source_ids"]
        if not isinstance(source_ids, list):
            raise ValueError("observability source_ids must be a list")
        cast(list[str], source_ids).append(source_id)
        if verification == "observed_verification":
            entry["verification_classification"] = "observed_verification"
        elif (
            verification == "observed_opportunity_only"
            and entry["verification_classification"] == "unknown"
        ):
            entry["verification_classification"] = "observed_opportunity_only"
    for row in rows:
        if row.get(ELIGIBLE, True) is not True:
            continue
        is_mature = row.get(MATURE, True) is True
        source_outcomes = row.get("source_outcomes")
        if isinstance(source_outcomes, dict):
            for source_id, value in cast(dict[str, Any], source_outcomes).items():
                if source_id in sources:
                    _count_outcome(sources[source_id], value, mature=is_mature)
        channel_outcomes = row.get("channel_outcomes")
        if isinstance(channel_outcomes, dict):
            for channel_id, value in cast(dict[str, Any], channel_outcomes).items():
                if channel_id in channels:
                    _count_outcome(channels[channel_id], value, mature=is_mature)
    for entry in [*sources.values(), *channels.values()]:
        _finalize_entry(entry)
        entry["source_opportunity_coverage_rate"] = None
        entry["coverage_rate"] = None
        entry["coverage_status"] = "NOT_ESTIMABLE_FROM_EVENT_ABSENCE"
    for entry in channels.values():
        source_ids = cast(list[str], entry["source_ids"])
        configured = any(
            isinstance(sources[source_id].get("opportunity_semantic"), str)
            for source_id in source_ids
        )
        entry["source_opportunity_coverage_status"] = (
            "UNAVAILABLE_NOT_MATERIALIZED" if configured else "UNKNOWN_NO_OPPORTUNITY_INDICATOR"
        )
        entry["source_opportunity_observed_count"] = 0
        entry["source_opportunity_unknown_count"] = eligible_count
    return {
        "sources": sources,
        "channels": channels,
        "eligible_observation_count": eligible_count,
        "mature_observation_count": mature_count,
        "prospective_observation_count": prospective_count,
        "fit_scope": "descriptive_full_sample",
        "analytical_use": "prohibited",
        "analytical_weights_created": False,
    }


def _empty_observability_entry(
    *, eligible_count: int, mature_count: int, prospective_count: int, verification: str
) -> dict[str, Any]:
    return {
        "verification_classification": verification,
        "observed_count": 0,
        "mature_observed_count": 0,
        "prospective_observed_count": 0,
        "positive_count": 0,
        "explicit_negative_count": 0,
        "eligible_count": eligible_count,
        "mature_count": mature_count,
        "prospective_count": prospective_count,
    }


def _count_outcome(entry: dict[str, Any], value: object, *, mature: bool) -> None:
    if value is None:
        return
    entry["observed_count"] = int(entry["observed_count"]) + 1
    if mature:
        entry["mature_observed_count"] = int(entry["mature_observed_count"]) + 1
    else:
        entry["prospective_observed_count"] = int(entry["prospective_observed_count"]) + 1
    count_key = "positive_count" if bool(value) else "explicit_negative_count"
    entry[count_key] = int(entry[count_key]) + 1


def _finalize_entry(entry: dict[str, Any]) -> None:
    eligible = int(entry["eligible_count"])
    mature = int(entry["mature_count"])
    observed = int(entry["observed_count"])
    mature_observed = int(entry["mature_observed_count"])
    entry["unknown_count"] = eligible - observed
    entry["mature_unknown_count"] = mature - mature_observed
    entry["event_incidence_fraction"] = (
        int(entry["positive_count"]) / eligible if eligible else None
    )
    entry["observed_outcome_fraction"] = observed / eligible if eligible else None
    entry["mature_cohort_observed_fraction"] = mature_observed / mature if mature else None


def _verification_classification(status: str) -> str:
    if status in {"observed_verification", "high_confirmation"}:
        return "observed_verification"
    if status in {"observed", "derived_from_audited_filings"}:
        return "observed_opportunity_only"
    return "unknown"
