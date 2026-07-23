from __future__ import annotations

import pytest

from core.fold_control import require_primary_target
from core.semantic_keys import ELIGIBLE
from selection.track_plan import build_sequential_track_plan, executable_tracks


def _measurement() -> dict[str, object]:
    return {
        "primary_target_id": "L1_ANNUAL",
        "candidate_targets": {"L1_ANNUAL": {"sources": ["a"]}},
        "execution_tracks": {
            "order": ["L1_ANNUAL", "L2", "L3_fixed_pi"],
        },
    }


def test_primary_target_resolves_only_from_canonical_binding() -> None:
    assert require_primary_target(_measurement(), "P10") == "L1_ANNUAL"
    legacy = _measurement()
    legacy["primary_target_id"] = None
    legacy["execution_tracks"] = {
        "primary_target_id": "L1_ANNUAL",
        "order": ["L1_ANNUAL", "L2", "L3_fixed_pi"],
    }
    with pytest.raises(RuntimeError, match="PRIMARY_TARGET_NOT_LOCKED"):
        require_primary_target(legacy, "P10")


def test_shadow_primary_target_alias_must_not_diverge() -> None:
    measurement = _measurement()
    measurement["execution_tracks"] = {
        "primary_target_id": "OTHER_TARGET",
        "order": ["L1_ANNUAL", "L2", "L3_fixed_pi"],
    }
    with pytest.raises(RuntimeError, match="PRIMARY_TARGET_CONTRACT_MISMATCH"):
        require_primary_target(measurement, "P10")


def test_sequential_plan_keeps_primary_and_skips_unavailable_optional_tracks() -> None:
    plan = build_sequential_track_plan(
        primary_target_id="L1_ANNUAL",
        outer_fold="2024",
        candidate_results=[
            {"candidate": "L2", ELIGIBLE: False, "reason_code": "L2_NOT_READY"},
            {"candidate": "L3_fixed_pi", ELIGIBLE: True, "objective": 0.4},
        ],
        execution_order=["L1_ANNUAL", "L2", "L3_fixed_pi"],
    )
    tracks = plan["tracks"]
    assert [item["status"] for item in tracks] == ["REQUIRED", "SKIPPED", "AVAILABLE"]
    assert plan["outer_outcomes_accessed"] is False
    assert plan["selection_by_outer_performance"] is False
    assert [item["measurement_id"] for item in executable_tracks(plan)] == [
        "L1_ANNUAL",
        "L3_fixed_pi",
    ]


def test_optional_track_unavailability_does_not_remove_primary_track() -> None:
    plan = build_sequential_track_plan(
        primary_target_id="L1_ANNUAL",
        outer_fold="2023",
        candidate_results=[],
        execution_order=["L1_ANNUAL", "L2", "L3_fixed_pi"],
    )
    executable = executable_tracks(plan)
    assert len(executable) == 1
    assert executable[0]["measurement_id"] == "L1_ANNUAL"
    assert executable[0]["role"] == "required_primary"
