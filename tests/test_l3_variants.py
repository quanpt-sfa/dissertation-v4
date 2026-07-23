from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from simulation.l3_variants import (
    L3_REQUIRED_METRICS,
    L3_VARIANT_IDS,
    estimate_l3_variant,
    run_l3_variant_batch,
)


def _settings() -> dict[str, object]:
    return {
        "posterior_grid_minimum": 0.001,
        "posterior_grid_maximum": 0.5,
        "posterior_grid_points": 500,
        "credible_interval_mass": 0.95,
        "wrong_fixed_pi_offset": 0.05,
        "misspecification_regret_definition": ("paired_excess_row_brier_relative_to_l3_correct"),
    }


def _scenario() -> dict[str, object]:
    return {
        "scenario_id": "l3_variant_test",
        "tier": "fully_synthetic",
        "sample_size": 800,
        "prevalence": 0.10,
        "fixed_pi": 0.10,
        "anchor_sensitivity": 0.85,
        "anchor_false_positive": 0.03,
        "weak_sensitivity": 0.60,
        "weak_false_positive": 0.08,
        "content_signal": 0.80,
        "anchor_verification_probability": 0.80,
        "weak_verification_probability": 0.75,
        "selective_verification_strength": 0.20,
        "channel_dependence": 0.70,
        "horizon_days": 365,
        "detection_delay_mean_days": 60,
        "shift_strength": 0.0,
        "signal_structure": "linear",
        "l3_variant_settings": _settings(),
    }


def _row(anchor: bool | None, weak: bool | None) -> dict[str, bool | None]:
    return {"anchor": anchor, "weak": weak}


def _rows() -> list[dict[str, bool | None]]:
    return (
        [_row(True, True)] * 24
        + [_row(True, False)] * 10
        + [_row(False, True)] * 8
        + [_row(False, False)] * 150
        + [_row(True, None)] * 5
        + [_row(None, False)] * 12
    )


def _accuracy() -> dict[str, tuple[float, float]]:
    return {
        "anchor": (0.85, 0.97),
        "weak": (0.60, 0.92),
    }


def test_l3_variants_use_distinct_locked_assumptions() -> None:
    scenario = _scenario()
    estimates = {
        variant: estimate_l3_variant(
            variant_id=variant,
            source_rows=_rows(),
            accuracy=_accuracy(),
            scenario=scenario,
        )
        for variant in L3_VARIANT_IDS
    }

    correct = cast(np.ndarray, estimates["l3_correct"]["row_probabilities"])
    assert bool(np.all((correct > 0.0) & (correct < 1.0)))
    assert bool(np.isclose(cast(float, estimates["l3_correct"]["fixed_pi_used"]), 0.10))
    assert bool(
        np.isclose(
            cast(float, estimates["l3_wrong_fixed_pi"]["fixed_pi_used"]),
            0.15,
        )
    )
    assert not bool(
        np.allclose(
            correct,
            cast(
                np.ndarray,
                estimates["l3_ignore_dependence"]["row_probabilities"],
            ),
        )
    )
    assert not bool(
        np.allclose(
            correct,
            cast(np.ndarray, estimates["l3_clean_anchor"]["row_probabilities"]),
        )
    )
    assert not bool(
        np.allclose(
            correct,
            cast(
                np.ndarray,
                estimates["l3_wrong_fixed_pi"]["row_probabilities"],
            ),
        )
    )


def test_l3_variant_batch_is_paired_and_metric_complete() -> None:
    scenario = _scenario()

    def data_rng_factory(replication_id: int) -> np.random.Generator:
        return np.random.default_rng(1000 + replication_id)

    outputs: dict[str, pd.DataFrame] = {}
    for variant in L3_VARIANT_IDS:
        diagnostics: dict[str, object] = {}
        batch = run_l3_variant_batch(
            scenario,
            method_id=variant,
            replications=range(2),
            data_rng_factory=data_rng_factory,
            diagnostics=diagnostics,
        )
        outputs[variant] = batch
        assert set(batch["metric_id"].astype(str)) == set(L3_REQUIRED_METRICS)
        assert set(batch["replication_id"].astype(int)) == {0, 1}
        assert diagnostics == {
            "fit_failures": {},
            "resampling_failures": {},
            "warnings": {},
            "affected_replication_ids": [],
        }

    correct_rows = cast(
        list[dict[str, object]],
        outputs["l3_correct"].to_dict(orient="records"),
    )

    def estimates_for(metric_id: str) -> np.ndarray:
        return np.asarray(
            [cast(float, row["estimate"]) for row in correct_rows if row["metric_id"] == metric_id],
            dtype=float,
        )

    assert bool(np.all(estimates_for("paired_brier_regret") == 0.0))
    assert bool(np.all(np.isfinite(estimates_for("review_sample_size_ratio_vs_l1"))))
    assert bool(np.all(np.isfinite(estimates_for("review_sample_size_ratio_vs_l2"))))


def test_fixed_pi_metrics_are_distinct_from_hierarchical_pi_sensitivity() -> None:
    scenario = _scenario()

    def data_rng_factory(replication_id: int) -> np.random.Generator:
        return np.random.default_rng(5000 + replication_id)

    batch = run_l3_variant_batch(
        scenario,
        method_id="l3_wrong_fixed_pi",
        replications=range(3),
        data_rng_factory=data_rng_factory,
    )
    metric_ids = set(batch["metric_id"].astype(str))

    assert "row_calibration_bias" in metric_ids
    assert "paired_brier_regret" in metric_ids
    assert "hierarchical_pi_error" in metric_ids
    assert "prevalence_error" not in metric_ids
