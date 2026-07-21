from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.semantic_keys import ESTIMATE, METHOD_ID, METRIC_ID, REPLICATION_ID, SCENARIO_ID
from scripts.p08c_aggregate_batches import (
    _sanitize_normalized_cost_regret,
    _validate_replication_artifact_ranges,
)
from simulation.service import (
    _classification_cost,
    _mcse_gate_required,
    summarize_mcse,
)


def _frame(metric_id: str, estimates: np.ndarray, *, family: str = "predictive", tier: str = "core") -> pd.DataFrame:
    count = len(estimates)
    return pd.DataFrame(
        {
            SCENARIO_ID: ["scenario"] * count,
            METHOD_ID: ["method"] * count,
            REPLICATION_ID: np.arange(count),
            METRIC_ID: [metric_id] * count,
            ESTIMATE: estimates,
            "learner_tier": [tier] * count,
            "training_cost_regime_id": ["not_applicable" if family == "standalone_estimator" else "symmetric"] * count,
            "imbalance_treatment_id": ["not_applicable" if family == "standalone_estimator" else "none"] * count,
            "method_family": [family] * count,
        }
    )


def test_normalized_cost_regret_is_undefined_when_scale_is_zero() -> None:
    result = _classification_cost(
        truth=np.zeros(20, dtype=bool),
        predicted_positive=np.zeros(20, dtype=bool),
        cost_regime={
            "false_positive_cost": 1.0,
            "false_negative_cost": 1.0,
            "review_cost": 0.0,
            "true_positive_benefit": 0.0,
        },
    )
    assert math.isnan(result["normalized_cost_regret"])


def test_standalone_tier_uses_method_metadata_for_all_metrics() -> None:
    report = summarize_mcse(
        [_frame("fit_success", np.ones(1000), family="standalone_estimator", tier="standalone")],
        minimum_replications=2500,
        maximum_replications=5000,
        pass_fail_mcse_maximum=0.01,
        l3_minimum_replications=1000,
        l3_maximum_replications=1000,
        l3_pass_fail_mcse_maximum=0.02,
        gated_metric_ids=["fit_success"],
    )
    row = report["metrics"][0]
    assert row["replication_tier"] == "standalone"
    assert row["minimum_replications_met"] is True
    assert report["status"] == "PASS"


def test_only_explicit_metrics_enter_mcse_gate() -> None:
    assert _mcse_gate_required("latent_average_precision", {"latent_average_precision"})
    assert not _mcse_gate_required(
        "budget_cost::top_05pct::high_fn::normalized_cost_regret",
        {"latent_average_precision"},
    )


def test_gate2_mmi_does_not_shrink_unlisted_metric_target() -> None:
    values = np.tile(np.array([0.3, 0.7]), 500)
    report = summarize_mcse(
        [_frame("latent_average_precision", values)],
        minimum_replications=1000,
        maximum_replications=1000,
        pass_fail_mcse_maximum=0.01,
        continuous_mcse_fraction=0.1,
        minimum_meaningful_improvement=0.01,
        gated_metric_ids=["latent_average_precision"],
        mmi_scaled_metric_ids=[],
    )
    row = report["metrics"][0]
    assert row["mcse_target"] == 0.01
    assert row["mcse_target_met"] is True


def test_undefined_non_gated_metric_does_not_block_completion() -> None:
    report = summarize_mcse(
        [
            _frame("latent_average_precision", np.full(1000, 0.5)),
            _frame(
                "budget_cost::top_05pct::high_fn::normalized_cost_regret",
                np.full(1000, np.nan),
            ),
        ],
        minimum_replications=1000,
        maximum_replications=1000,
        pass_fail_mcse_maximum=0.01,
        gated_metric_ids=["latent_average_precision"],
    )
    rows = {item[METRIC_ID]: item for item in report["metrics"]}
    diagnostic = rows[
        "budget_cost::top_05pct::high_fn::normalized_cost_regret"
    ]
    assert diagnostic["undefined_replications"] == 1000
    assert diagnostic["mcse_gate_required"] is False
    assert report["status"] == "PASS"


def test_legacy_normalized_regret_is_recomputed_during_aggregation() -> None:
    prefix = "budget_cost::top_05pct::high_fn"
    frame = pd.DataFrame(
        {
            SCENARIO_ID: ["s", "s", "s", "s", "s", "s"],
            METHOD_ID: ["m", "m", "m", "m", "m", "m"],
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
    clean = _sanitize_normalized_cost_regret(frame)
    normalized = clean.loc[
        clean[METRIC_ID].astype(str).str.endswith("::normalized_cost_regret"),
        ESTIMATE,
    ].tolist()
    assert math.isnan(normalized[0])
    assert normalized[1] == 0.4


def test_compact_artifact_ranges_use_artifact_ordinals() -> None:
    _validate_replication_artifact_ranges(
        [
            {
                SCENARIO_ID: "s",
                METHOD_ID: "m",
                "batch_key": "b0000",
                "start": 0,
                "end": 2499,
                "replications": 2500,
                "locked_worker_batch_size": 250,
            },
            {
                SCENARIO_ID: "s",
                METHOD_ID: "m",
                "batch_key": "b0001",
                "start": 2500,
                "end": 4999,
                "replications": 2500,
                "locked_worker_batch_size": 250,
            },
        ]
    )


def test_compact_artifact_ranges_reject_gap_or_wrong_ordinal() -> None:
    with pytest.raises(ValueError, match="expected compact-artifact ordinal=b0001"):
        _validate_replication_artifact_ranges(
            [
                {
                    SCENARIO_ID: "s",
                    METHOD_ID: "m",
                    "batch_key": "b0000",
                    "start": 0,
                    "end": 2499,
                    "replications": 2500,
                },
                {
                    SCENARIO_ID: "s",
                    METHOD_ID: "m",
                    "batch_key": "b0010",
                    "start": 2500,
                    "end": 4999,
                    "replications": 2500,
                },
            ]
        )
