"""Fail-closed adaptive-control decisions for P08 replication orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict


class AdaptiveControlDecision(TypedDict):
    completed: int
    precision_met: bool
    terminal_at_cap: bool
    confirmatory_metric_count: int


def confirmatory_control_decision(
    rows: Sequence[Mapping[str, object]],
    *,
    scenario_id: str,
    method_id: str,
) -> AdaptiveControlDecision:
    """Derive adaptive replication state from confirmatory MCSE metrics only."""
    gate_rows = [row for row in rows if row.get("mcse_gate_required") is True]
    if not gate_rows:
        raise ValueError(
            "P08 control report has no confirmatory MCSE metrics for "
            f"scenario={scenario_id}, method={method_id}"
        )

    replication_counts = [
        _nonnegative_integer(
            row.get("replications"),
            context=(
                "P08 confirmatory control replications for "
                f"scenario={scenario_id}, method={method_id}"
            ),
        )
        for row in gate_rows
    ]
    completed = min(replication_counts)
    precision_met = all(
        row.get("minimum_replications_met") is True and row.get("mcse_target_met") is True
        for row in gate_rows
    )
    terminal_at_cap = all(
        row.get("mcse_target_met") is True or row.get("maximum_replications_reached") is True
        for row in gate_rows
    )
    return {
        "completed": completed,
        "precision_met": precision_met,
        "terminal_at_cap": terminal_at_cap,
        "confirmatory_metric_count": len(gate_rows),
    }


def _nonnegative_integer(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: nonnegative integer required")
    return value
