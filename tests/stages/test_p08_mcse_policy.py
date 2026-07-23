"""Regression tests for policy-aware P08 MCSE aggregation."""

from __future__ import annotations

import math
from typing import cast

import pandas as pd
import pytest

from core.semantic_keys import (
    BATCH_KEY,
    ESTIMATE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from simulation.mcse_contract import MetricPolicy
from simulation.mcse_service import (
    normalized_cost_regret,
    sanitize_normalized_cost_regret,
    summarize_mcse,
    validate_worker_batch,
)


def _policy(
    metric_id: str,
    *,
    role: str,
    gate: bool,
    undefined_policy: str,
    target_rule: str,
    minimum_finite_fraction: float,
) -> MetricPolicy:
    return {
        "policy_id": f"policy_{metric_id}",
        "priority": 100,
        "metric_pattern": metric_id,
        "resolved_pattern": metric_id,
        "role": role,
        "mcse_gate_required": gate,
        "undefined_policy": undefined_policy,
        "target_rule": target_rule,
        "minimum_finite_fraction": minimum_finite_fraction,
    }


def _frame(metric_id: str, estimates: list[float]) -> pd.DataFrame:
    count = len(estimates)
    return pd.DataFrame(
        {
            SCENARIO_ID: ["scenario"] * count,
            METHOD_ID: ["method"] * count,
            REPLICATION_ID: list(range(count)),
            METRIC_ID: [metric_id] * count,
            ESTIMATE: estimates,
            "learner_tier": ["core"] * count,
            "training_cost_regime_id": ["symmetric"] * count,
            "imbalance_treatment_id": ["none"] * count,
            "method_family": ["predictive"] * count,
        }
    )


def _summary(
    batches: list[pd.DataFrame],
    policies: dict[str, MetricPolicy],
    *,
    minimum: int,
) -> dict[str, object]:
    return summarize_mcse(
        batches,
        metric_policies=policies,
        minimum_replications=minimum,
        maximum_replications=minimum,
        pass_fail_mcse_maximum=0.01,
    )


def test_normalized_cost_regret_is_undefined_when_scale_is_unidentified() -> None:
    value = normalized_cost_regret(
        total_cost=0.0,
        oracle_cost=0.0,
        all_negative_cost=0.0,
    )

    assert math.isnan(value)


def test_legacy_normalized_regret_is_sanitized_in_memory() -> None:
    prefix = "budget_cost::top_05pct::high_fn"
    frame = pd.DataFrame(
        {
            SCENARIO_ID: ["s"] * 6,
            METHOD_ID: ["m"] * 6,
            REPLICATION_ID: [0, 0, 0, 1, 1, 1],
            METRIC_ID: [
                f"{prefix}::normalized_cost_regret",
                f"{prefix}::cost_regret_vs_oracle",
                f"{prefix}::cost_savings_vs_all_negative",
                f"{prefix}::normalized_cost_regret",
                f"{prefix}::cost_regret_vs_oracle",
                f"{prefix}::cost_savings_vs_all_negative",
            ],
            ESTIMATE: [1.0e12, 2.0, -2.0, 999.0, 2.0, 3.0],
        }
    )

    clean = sanitize_normalized_cost_regret(frame)
    values = cast(
        list[float],
        clean.loc[
            clean[METRIC_ID].astype(str).str.endswith("::normalized_cost_regret"),
            ESTIMATE,
        ].tolist(),
    )

    assert math.isnan(values[0])
    assert values[1] == pytest.approx(0.4)


def test_undefined_diagnostic_metric_does_not_block_confirmatory_pass() -> None:
    policies = {
        "latent_average_precision": _policy(
            "latent_average_precision",
            role="confirmatory",
            gate=True,
            undefined_policy="forbid",
            target_rule="absolute",
            minimum_finite_fraction=1.0,
        ),
        "normalized_cost_regret": _policy(
            "normalized_cost_regret",
            role="diagnostic",
            gate=False,
            undefined_policy="allow_nonblocking",
            target_rule="none",
            minimum_finite_fraction=0.0,
        ),
    }
    report = _summary(
        [
            _frame("latent_average_precision", [0.5] * 10),
            _frame("normalized_cost_regret", [math.nan] * 10),
        ],
        policies,
        minimum=10,
    )
    rows = {str(item[METRIC_ID]): item for item in cast(list[dict[str, object]], report["metrics"])}

    assert report["status"] == "PASS"
    assert report["precision_target_met"] is True
    assert rows["normalized_cost_regret"]["undefined_replications"] == 10
    assert rows["normalized_cost_regret"]["mcse_gate_required"] is False


def test_undefined_confirmatory_metric_fails_at_replication_cap() -> None:
    policies = {
        "latent_average_precision": _policy(
            "latent_average_precision",
            role="confirmatory",
            gate=True,
            undefined_policy="forbid",
            target_rule="absolute",
            minimum_finite_fraction=1.0,
        )
    }

    report = _summary(
        [_frame("latent_average_precision", [math.nan] * 10)],
        policies,
        minimum=10,
    )

    assert report["status"] == "MAXIMUM_REACHED"
    assert report["precision_target_met"] is False
    assert report["evidence_blockers"] == ["scenario:method:latent_average_precision"]


def test_worker_batch_requires_contiguous_aligned_replication_ids() -> None:
    frame = pd.DataFrame(
        {
            SCENARIO_ID: ["s", "s"],
            METHOD_ID: ["m", "m"],
            REPLICATION_ID: [0, 2],
        }
    )

    with pytest.raises(ValueError, match="contiguous"):
        validate_worker_batch(
            frame,
            coordinates={BATCH_KEY: "b0000"},
            locked_worker_batch_size=10,
        )


def test_worker_batch_rejects_coordinate_mismatch() -> None:
    frame = pd.DataFrame(
        {
            SCENARIO_ID: ["s", "s"],
            METHOD_ID: ["m", "m"],
            REPLICATION_ID: [10, 11],
        }
    )

    with pytest.raises(ValueError, match="expected=b0001"):
        validate_worker_batch(
            frame,
            coordinates={BATCH_KEY: "b0000"},
            locked_worker_batch_size=10,
        )
