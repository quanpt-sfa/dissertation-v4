"""Regression tests for confirmatory-only P08 adaptive replication control."""

from __future__ import annotations

import pytest

from simulation.mcse_control import confirmatory_control_decision


def test_sparse_diagnostic_metric_does_not_control_completed_replications() -> None:
    decision = confirmatory_control_decision(
        [
            {
                "mcse_gate_required": True,
                "replications": 10,
                "minimum_replications_met": True,
                "mcse_target_met": True,
                "maximum_replications_reached": False,
            },
            {
                "mcse_gate_required": False,
                "replications": 2,
                "minimum_replications_met": True,
                "mcse_target_met": True,
                "maximum_replications_reached": False,
            },
        ],
        scenario_id="scenario",
        method_id="method",
    )

    assert decision == {
        "completed": 10,
        "precision_met": True,
        "terminal_at_cap": True,
        "confirmatory_metric_count": 1,
    }


def test_confirmatory_metrics_control_next_replication_start() -> None:
    decision = confirmatory_control_decision(
        [
            {
                "mcse_gate_required": True,
                "replications": 20,
                "minimum_replications_met": True,
                "mcse_target_met": True,
                "maximum_replications_reached": False,
            },
            {
                "mcse_gate_required": True,
                "replications": 15,
                "minimum_replications_met": True,
                "mcse_target_met": False,
                "maximum_replications_reached": False,
            },
        ],
        scenario_id="scenario",
        method_id="method",
    )

    assert decision["completed"] == 15
    assert decision["precision_met"] is False
    assert decision["terminal_at_cap"] is False


def test_missing_confirmatory_metrics_fails_closed() -> None:
    with pytest.raises(ValueError, match="no confirmatory MCSE metrics"):
        confirmatory_control_decision(
            [
                {
                    "mcse_gate_required": False,
                    "replications": 10,
                    "minimum_replications_met": True,
                    "mcse_target_met": True,
                    "maximum_replications_reached": True,
                }
            ],
            scenario_id="scenario",
            method_id="method",
        )
