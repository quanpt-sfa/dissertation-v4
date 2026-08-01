"""Pure descriptive observability summaries; analytical weights are prohibited here."""

from __future__ import annotations

from typing import Any, cast

from core.semantic_keys import CHANNEL_ID, ELIGIBLE, MATURE, VERIFICATION_STATUS


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
        raw_status = str(metadata.get(VERIFICATION_STATUS, "unknown"))
        verification = _verification_classification(raw_status)
        evidence_mapping = metadata.get("evidence_mapping")
        opportunity_semantic = (
            cast(dict[str, Any], evidence_mapping).get("opportunity_semantic")
            if isinstance(evidence_mapping, dict)
            else None
        )
        source_mature_count = sum(
            row.get(ELIGIBLE, True) is True
            and isinstance(row.get("source_maturity"), dict)
            and cast(dict[str, Any], row["source_maturity"]).get(source_id) is True
            for row in rows
        )
        source_entry = _empty_observability_entry(
            eligible_count=eligible_count,
            mature_count=source_mature_count,
            prospective_count=eligible_count - source_mature_count,
            verification=verification,
        )
        source_entry.update(
            {
                CHANNEL_ID: channel_id,
                "raw_verification_metadata": raw_status,
                "legacy_high_confirmation_is_not_verification": raw_status == "high_confirmation",
                "opportunity_semantic": opportunity_semantic,
                "source_opportunity_coverage_rate": None,
                "source_opportunity_coverage_status": (
                    "AVAILABLE_MATERIALIZED"
                    if isinstance(opportunity_semantic, str)
                    else "UNKNOWN_NO_OPPORTUNITY_INDICATOR"
                ),
                "source_opportunity_observed_count": 0,
                "source_opportunity_not_observed_count": 0,
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
        source_maturity_raw = row.get("source_maturity")
        source_maturity = (
            cast(dict[str, Any], source_maturity_raw)
            if isinstance(source_maturity_raw, dict)
            else {}
        )
        source_outcomes = row.get("source_outcomes")
        if isinstance(source_outcomes, dict):
            for source_id, value in cast(dict[str, Any], source_outcomes).items():
                if source_id in sources:
                    _count_outcome(
                        sources[source_id],
                        value,
                        mature=bool(source_maturity.get(source_id, is_mature)),
                    )
        source_opportunities = row.get("source_opportunities")
        if isinstance(source_opportunities, dict):
            for source_id, value in cast(dict[str, Any], source_opportunities).items():
                if source_id in sources:
                    _count_opportunity(sources[source_id], value)
        channel_outcomes = row.get("channel_outcomes")
        if isinstance(channel_outcomes, dict):
            for channel_id, value in cast(dict[str, Any], channel_outcomes).items():
                if channel_id in channels:
                    channel_sources = cast(list[str], channels[channel_id]["source_ids"])
                    channel_mature = all(
                        bool(source_maturity.get(source_id, is_mature))
                        for source_id in channel_sources
                    )
                    _count_outcome(channels[channel_id], value, mature=channel_mature)
        channel_opportunities = row.get("channel_opportunities")
        if isinstance(channel_opportunities, dict):
            for channel_id, value in cast(dict[str, Any], channel_opportunities).items():
                if channel_id in channels:
                    _count_opportunity(channels[channel_id], value)
    for entry in [*sources.values(), *channels.values()]:
        _finalize_entry(entry)
        configured = (
            isinstance(entry.get("opportunity_semantic"), str)
            or int(entry.get("source_opportunity_observed_count", 0))
            + int(entry.get("source_opportunity_not_observed_count", 0))
            > 0
        )
        eligible = int(entry["eligible_count"])
        if configured:
            observed = int(entry["source_opportunity_observed_count"])
            entry["source_opportunity_coverage_rate"] = observed / eligible if eligible else None
            entry["coverage_rate"] = entry["source_opportunity_coverage_rate"]
            entry["coverage_status"] = "AVAILABLE_MATERIALIZED"
        else:
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
            "AVAILABLE_MATERIALIZED" if configured else "UNKNOWN_NO_OPPORTUNITY_INDICATOR"
        )
    overlaps = _channel_overlap(rows, sorted(channels))
    return {
        "sources": sources,
        "channels": channels,
        "eligible_observation_count": eligible_count,
        "mature_observation_count": mature_count,
        "prospective_observation_count": prospective_count,
        "fit_scope": "descriptive_full_sample",
        "analytical_use": "prohibited",
        "analytical_weights_created": False,
        "terminology_contract": {
            "zero_label": "observed_endpoint_zero",
            "zero_label_is_latent_negative": False,
            "high_confirmation_is_observed_verification": False,
            "verification_requires_direct_observation": True,
        },
        "channel_overlap": overlaps,
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
        "observed_endpoint_zero_count": 0,
        "eligible_count": eligible_count,
        "mature_count": mature_count,
        "prospective_count": prospective_count,
        "source_opportunity_observed_count": 0,
        "source_opportunity_not_observed_count": 0,
        "source_opportunity_unknown_count": eligible_count,
    }


def _count_outcome(entry: dict[str, Any], value: object, *, mature: bool) -> None:
    if value is None:
        return
    entry["observed_count"] = int(entry["observed_count"]) + 1
    if mature:
        entry["mature_observed_count"] = int(entry["mature_observed_count"]) + 1
    else:
        entry["prospective_observed_count"] = int(entry["prospective_observed_count"]) + 1
    count_key = "positive_count" if bool(value) else "observed_endpoint_zero_count"
    entry[count_key] = int(entry[count_key]) + 1


def _count_opportunity(entry: dict[str, Any], value: object) -> None:
    if value is True:
        entry["source_opportunity_observed_count"] = (
            int(entry["source_opportunity_observed_count"]) + 1
        )
    elif value is False:
        entry["source_opportunity_not_observed_count"] = (
            int(entry["source_opportunity_not_observed_count"]) + 1
        )
    else:
        return
    eligible = int(entry["eligible_count"])
    observed = int(entry["source_opportunity_observed_count"])
    not_observed = int(entry["source_opportunity_not_observed_count"])
    entry["source_opportunity_unknown_count"] = eligible - observed - not_observed


def _finalize_entry(entry: dict[str, Any]) -> None:
    eligible = int(entry["eligible_count"])
    mature = int(entry["mature_count"])
    observed = int(entry["observed_count"])
    mature_observed = int(entry["mature_observed_count"])
    entry["unknown_count"] = eligible - observed
    entry["mature_unknown_count"] = mature - mature_observed
    entry["positive_incidence_fraction"] = (
        int(entry["positive_count"]) / eligible if eligible else None
    )
    entry["event_incidence_fraction"] = entry["positive_incidence_fraction"]
    entry["observed_outcome_fraction"] = observed / eligible if eligible else None
    entry["mature_cohort_observed_fraction"] = mature_observed / mature if mature else None
    # Backward-compatible output only.  The preferred term is observed endpoint zero.
    entry["explicit_negative_count"] = int(entry["observed_endpoint_zero_count"])
    entry["explicit_negative_count_deprecated"] = True


def _channel_overlap(rows: list[dict[str, Any]], channel_ids: list[str]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        if row.get(ELIGIBLE, True) is not True:
            continue
        outcomes_raw = row.get("channel_outcomes")
        opportunities_raw = row.get("channel_opportunities")
        outcomes = cast(dict[str, Any], outcomes_raw) if isinstance(outcomes_raw, dict) else {}
        opportunities = (
            cast(dict[str, Any], opportunities_raw) if isinstance(opportunities_raw, dict) else {}
        )
        for basis, observed_channels in (
            (
                "observed_outcome",
                tuple(channel for channel in channel_ids if outcomes.get(channel) is not None),
            ),
            (
                "observed_opportunity",
                tuple(channel for channel in channel_ids if opportunities.get(channel) is True),
            ),
        ):
            pattern = "∩".join(observed_channels) if observed_channels else "NONE"
            entry = counts.setdefault(
                (basis, pattern),
                {"eligible_count": 0, "mature_count": 0, "prospective_count": 0},
            )
            entry["eligible_count"] += 1
            if row.get(MATURE, True) is True:
                entry["mature_count"] += 1
            else:
                entry["prospective_count"] += 1
    return [
        {"basis": basis, "channel_pattern": pattern, **counts[(basis, pattern)]}
        for basis, pattern in sorted(counts)
    ]


def _verification_classification(status: str) -> str:
    if status == "observed_verification":
        return "observed_verification"
    if status in {"observed", "derived_from_audited_filings", "observed_opportunity_only"}:
        return "observed_opportunity_only"
    # Legacy labels such as high_confirmation describe evidentiary provenance, not V.
    return "unknown"
