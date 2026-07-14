"""Pure descriptive observability summaries; analytical weights are prohibited here."""

from __future__ import annotations

from typing import Any, cast

from core.semantic_keys import CHANNEL_ID


def build_observability_registry(
    matrices: dict[str, Any], source_metadata: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows_raw = matrices.get("rows")
    if not isinstance(rows_raw, list):
        raise ValueError("source matrices rows are required")
    rows_raw = cast(list[Any], rows_raw)
    rows = [cast(dict[str, Any], row) for row in rows_raw if isinstance(row, dict)]
    channels: dict[str, dict[str, Any]] = {}
    for source_id, metadata in sorted(source_metadata.items()):
        channel_id = metadata.get(CHANNEL_ID)
        if not isinstance(channel_id, str):
            raise ValueError(f"source={source_id}: channel_id required")
        status = str(metadata.get("verification_status", "unknown"))
        entry = channels.setdefault(
            channel_id,
            {
                "source_ids": [],
                "verification_classification": "unknown",
                "observed_count": 0,
                "eligible_count": len(rows),
            },
        )
        source_ids = entry["source_ids"]
        if not isinstance(source_ids, list):
            raise ValueError("observability source_ids must be a list")
        cast(list[str], source_ids).append(source_id)
        if status == "observed_verification":
            entry["verification_classification"] = "observed_verification"
        elif (
            status in {"observed", "derived_from_audited_filings"}
            and entry["verification_classification"] == "unknown"
        ):
            entry["verification_classification"] = "observed_opportunity_only"
    for row in rows:
        outcomes = row.get("channel_outcomes")
        if not isinstance(outcomes, dict):
            continue
        outcomes = cast(dict[str, Any], outcomes)
        for channel_id, value in outcomes.items():
            if channel_id in channels and value is not None:
                channels[channel_id]["observed_count"] = (
                    int(channels[channel_id]["observed_count"]) + 1
                )
    for entry in channels.values():
        eligible = int(entry["eligible_count"])
        observed = int(entry["observed_count"])
        entry["coverage_rate"] = observed / eligible if eligible else None
    return {
        "channels": channels,
        "fit_scope": "descriptive_full_sample",
        "analytical_use": "prohibited",
        "analytical_weights_created": False,
    }
