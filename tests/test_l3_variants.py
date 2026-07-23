from __future__ import annotations

import numpy as np

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


def _rows() -> list[dict[str, bool | None]]:
    return (
        [{"anchor": True, "weak": True}] * 24
        + [{"anchor": True, "weak": False}] * 10
        + [{"anchor": False, "weak": True}] * 8
        + [{"anchor": False, "weak": False}] * 150
        + [{"anchor": True, "weak": None}] * 5
        + [{"anchor": None, "weak": False}] * 12
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

    correct = np.asarray(estimates["l3_correct"]["row_probabilities"], dtype=float)
    assert np.all((correct > 0.0) & (correct < 1.0))
    assert np.isclose(float(estimates["l3_correct"]["fixed_pi_used"]), 0.10)
    assert np.isclose(float(estimates["l3_wrong_fixed_pi"]["fixed_pi_used"]), 0.15)
    assert not np.allclose(
        correct,
        np.asarray(estimates["l3_ignore_dependence"]["row_probabilities"], dtype=float),
    )
    assert not np.allclose(
        correct,
        np.asarray(estimates["l3_clean_anchor"]["row_probabilities"], dtype=float),
    )
    assert not np.allclose(
        correct,
        np.asarray(estimates["l3_wrong_fixed_pi"]["row_probabilities"], dtype=float),
    )


def test_l3_variant_batch_is_paired_and_metric_complete() -> None:
    scenario = _scenario()

    def data_rng_factory(replication_id: int) -> np.random.Generator:
        return np.random.default_rng(1000 + replication_id)

    outputs = {}
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

    correct = outputs["l3_correct"].set_index(["replication_id", "metric_id"])["estimate"]
    assert (
        correct.xs(
            "paired_brier_regret",
            level="metric_id",
        )
        == 0.0
    ).all()
    assert np.isfinite(
        correct.xs(
            "review_sample_size_ratio_vs_l1",
            level="metric_id",
        )
    ).all()
    assert np.isfinite(
        correct.xs(
            "review_sample_size_ratio_vs_l2",
            level="metric_id",
        )
    ).all()


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
