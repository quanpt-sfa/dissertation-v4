"""Nested time/channel measurement selection without outer-outcome access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

from core.semantic_keys import ELIGIBLE, FISCAL_YEAR, OUTER_FOLD


@dataclass(frozen=True)
class SelectionResult:
    candidates: list[dict[str, Any]]
    selection: dict[str, Any]
    channel_selection: dict[str, Any]


def select_measurement(
    *,
    matrices: dict[str, Any],
    outer_year: int,
    candidates: list[str],
    l3_capability: dict[str, Any],
    minimum_observed_channels: int | None = None,
) -> SelectionResult:
    """Select using years before the outer fold and held-channel predictions only."""
    raw_rows = matrices.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("measurement matrices require rows")
    raw_rows = cast(list[Any], raw_rows)
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        row = cast(dict[str, Any], raw_row)
        if int(row[FISCAL_YEAR]) < outer_year:
            rows.append(row)
    expected = matrices.get("expected_channels")
    if not isinstance(expected, list):
        raise ValueError("measurement matrices require expected_channels")
    expected = cast(list[Any], expected)
    raw_l2_scoring = matrices.get("l2_scoring")
    l2_scoring = cast(dict[str, Any], raw_l2_scoring) if isinstance(raw_l2_scoring, dict) else {}
    l2_available = l2_scoring.get("status") == "AVAILABLE"
    channel_results: list[dict[str, Any]] = []
    for heldout in sorted(str(value) for value in expected):
        losses: list[float] = []
        for row in rows:
            outcomes = row.get("channel_outcomes")
            if not isinstance(outcomes, dict):
                continue
            outcomes = cast(dict[str, Any], outcomes)
            raw_scores = row.get("channel_evidence_scores")
            if not isinstance(raw_scores, dict) or not l2_available:
                continue
            scores = cast(dict[str, Any], raw_scores)
            if outcomes.get(heldout) is None:
                continue
            remaining = [
                float(value)
                for channel, value in scores.items()
                if channel != heldout and value is not None
            ]
            if minimum_observed_channels is None or len(remaining) < minimum_observed_channels:
                continue
            probability = min(1.0 - 1e-6, max(1e-6, sum(remaining) / len(remaining)))
            outcome = float(bool(outcomes[heldout]))
            losses.append(
                -(outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability))
            )
        channel_results.append(
            {
                "heldout_channel": heldout,
                "candidate": "L2",
                "rows": len(losses),
                "soft_cross_entropy": sum(losses) / len(losses) if losses else None,
                "heldout_removed_from_target_and_measurement": True,
            }
        )
    complete = bool(channel_results) and all(item["rows"] > 0 for item in channel_results)
    results: list[dict[str, Any]] = []
    if "L2" in candidates:
        losses = [
            float(item["soft_cross_entropy"])
            for item in channel_results
            if item["soft_cross_entropy"] is not None
        ]
        results.append(
            {
                "candidate": "L2",
                ELIGIBLE: complete and minimum_observed_channels is not None and l2_available,
                "objective": sum(losses) / len(losses) if complete and losses else None,
                "reason_code": None
                if complete and minimum_observed_channels is not None and l2_available
                else str(l2_scoring.get("reason_code"))
                if not l2_available
                else "L2_MINIMUM_COVERAGE_NOT_LOCKED"
                if minimum_observed_channels is None
                else "INSUFFICIENT_CHANNELS",
                "minimum_observed_channels": minimum_observed_channels,
            }
        )
    if "L3_fixed_pi" in candidates:
        available = l3_capability.get("status") == "AVAILABLE" and bool(
            l3_capability.get("pilot_executed")
        )
        results.append(
            {
                "candidate": "L3_fixed_pi",
                ELIGIBLE: available,
                "objective": l3_capability.get("selection_objective") if available else None,
                "reason_code": None if available else "CAPABILITY_UNAVAILABLE",
            }
        )
    eligible = [item for item in results if item[ELIGIBLE] and item["objective"] is not None]
    selected = (
        min(eligible, key=lambda item: float(item["objective"]))["candidate"]
        if eligible
        else "none"
    )
    reason = None if eligible else "NO_ELIGIBLE_CANDIDATE"
    return SelectionResult(
        candidates=results,
        selection={
            OUTER_FOLD: str(outer_year),
            "selected_measurement": selected,
            "reason_code": reason,
            "selection_scope": "development_history_only",
            "outer_outcomes_accessed": False,
            "fit_max_year": max((int(row[FISCAL_YEAR]) for row in rows), default=None),
        },
        channel_selection={
            OUTER_FOLD: str(outer_year),
            "strict_channel_results": channel_results,
            "heldout_channel_removed_from_all_selection_inputs": True,
        },
    )
