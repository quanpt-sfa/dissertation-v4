"""Sequential measurement-track planning without outer-outcome selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from core.semantic_keys import ELIGIBLE, OUTER_FOLD

SEQUENTIAL_MODE = "sequential_capability_gated"
PRIMARY_TRACK_ROLE = "required_primary"
OPTIONAL_TRACK_ROLE = "optional_capability_gated"

DEFAULT_TRACK_IDS = {
    "L1_ANNUAL": "track_l1",
    "L2": "track_l2",
    "L3_fixed_pi": "track_l3_fixed_pi",
}


def build_sequential_track_plan(
    *,
    primary_target_id: str,
    outer_fold: str,
    candidate_results: Sequence[Mapping[str, Any]],
    execution_order: Sequence[str],
    track_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build one required primary track plus capability-gated optional tracks."""
    if not primary_target_id:
        raise ValueError("primary_target_id is required for sequential tracks")
    ids = dict(DEFAULT_TRACK_IDS)
    if track_ids is not None:
        ids.update({str(key): str(value) for key, value in track_ids.items()})
    order = [str(value) for value in execution_order]
    if not order or order[0] != primary_target_id:
        raise ValueError("sequential track order must start with primary_target_id")
    if len(order) != len(set(order)):
        raise ValueError("sequential track order must be unique")

    candidates = {
        str(item.get("candidate")): cast(Mapping[str, Any], item)
        for item in candidate_results
        if isinstance(item, Mapping) and isinstance(item.get("candidate"), str)
    }
    tracks: list[dict[str, Any]] = []
    for position, measurement_id in enumerate(order):
        track_id = ids.get(measurement_id, f"track_{measurement_id.lower()}")
        if position == 0:
            tracks.append(
                {
                    "track_id": track_id,
                    "measurement_id": measurement_id,
                    "role": PRIMARY_TRACK_ROLE,
                    "status": "REQUIRED",
                    ELIGIBLE: True,
                    "reason_code": None,
                    "execution_position": position,
                }
            )
            continue
        result = candidates.get(measurement_id, {})
        eligible = result.get(ELIGIBLE) is True
        tracks.append(
            {
                "track_id": track_id,
                "measurement_id": measurement_id,
                "role": OPTIONAL_TRACK_ROLE,
                "status": "AVAILABLE" if eligible else "SKIPPED",
                ELIGIBLE: eligible,
                "reason_code": None if eligible else result.get("reason_code", "CAPABILITY_UNAVAILABLE"),
                "execution_position": position,
            }
        )
    return {
        OUTER_FOLD: str(outer_fold),
        "mode": SEQUENTIAL_MODE,
        "primary_target_id": primary_target_id,
        "tracks": tracks,
        "outer_outcomes_accessed": False,
        "selection_by_outer_performance": False,
    }


def executable_tracks(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return required and available tracks in locked execution order."""
    raw = plan.get("tracks")
    if not isinstance(raw, list):
        raise ValueError("sequential track plan requires tracks")
    tracks = [cast(dict[str, Any], item) for item in raw if isinstance(item, dict)]
    return sorted(
        [item for item in tracks if item.get(ELIGIBLE) is True],
        key=lambda item: int(item.get("execution_position", 0)),
    )
