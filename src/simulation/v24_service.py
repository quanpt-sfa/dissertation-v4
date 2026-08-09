"""Chapter 3 v24 simulation adapter with an explicit O-V-D-R-S process.

The legacy simulation engine remains responsible for learner fitting, cost metrics,
and MCSE aggregation. This adapter replaces only the data-generating measurement
process and the partial-identification estimator used by P08B.
"""

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from core.semantic_keys import OBSERVED_SOURCE_LABEL
from identification.service import prevalence_bounds
from labels.service import aggregate_l1, evidence_score_l2
from simulation import service as legacy

_PATCH_LOCK = threading.Lock()
_MINIMUM_PU_CLASSIFIER_ROWS = 10

PUFitResult = tuple[np.ndarray, float, tuple[float, float]]
PUDelegate = Callable[..., PUFitResult]


def run_batch(
    scenario: Mapping[str, Any],
    *,
    method_id: str,
    replications: range,
    data_rng_factory: Callable[[int], np.random.Generator] | None = None,
    model_rng_factory: Callable[[int], np.random.Generator] | None = None,
    data_rng: np.random.Generator | None = None,
    model_rng: np.random.Generator | None = None,
    rng: np.random.Generator | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run the existing engine with v24 measurement-process functions installed."""

    with _PATCH_LOCK:
        original_generate = legacy._generate_replication
        original_partial_identification = legacy._partial_identification_bounds
        original_pu_ensemble = legacy._fit_pu_ensemble_cost_sensitive
        legacy._generate_replication = _generate_replication_v24
        legacy._partial_identification_bounds = _partial_identification_bounds_v24

        def guarded_pu_ensemble(
            *,
            learner_id: str,
            x_train: np.ndarray,
            predict_x: np.ndarray,
            positive_indices: np.ndarray,
            unlabeled_indices: np.ndarray,
            bags: int,
            unlabeled_to_positive_ratio: float,
            training_cost: Mapping[str, object],
            scenario: Mapping[str, Any],
            rng: np.random.Generator,
        ) -> PUFitResult:
            return _fit_pu_ensemble_cost_sensitive_v24(
                learner_id=learner_id,
                x_train=x_train,
                predict_x=predict_x,
                positive_indices=positive_indices,
                unlabeled_indices=unlabeled_indices,
                bags=bags,
                unlabeled_to_positive_ratio=unlabeled_to_positive_ratio,
                training_cost=training_cost,
                scenario=scenario,
                rng=rng,
                delegate=original_pu_ensemble,
            )

        legacy._fit_pu_ensemble_cost_sensitive = guarded_pu_ensemble
        try:
            return legacy.run_batch(
                scenario,
                method_id=method_id,
                replications=replications,
                data_rng_factory=data_rng_factory,
                model_rng_factory=model_rng_factory,
                data_rng=data_rng,
                model_rng=model_rng,
                rng=rng,
                diagnostics=diagnostics,
            )
        finally:
            legacy._generate_replication = original_generate
            legacy._partial_identification_bounds = original_partial_identification
            legacy._fit_pu_ensemble_cost_sensitive = original_pu_ensemble


def _fit_pu_ensemble_cost_sensitive_v24(
    *,
    learner_id: str,
    x_train: np.ndarray,
    predict_x: np.ndarray,
    positive_indices: np.ndarray,
    unlabeled_indices: np.ndarray,
    bags: int,
    unlabeled_to_positive_ratio: float,
    training_cost: Mapping[str, object],
    scenario: Mapping[str, Any],
    rng: np.random.Generator,
    delegate: PUDelegate,
) -> PUFitResult:
    """Short-circuit PU bags that cannot satisfy the legacy learner row guard.

    This is behavior-preserving for the affected replication: the legacy engine
    would attempt every bag, reject each one because the training view has fewer
    than ten rows, and finally return the same observed-positive prior with
    ``fit_success=0``. The adapter records one structured resampling diagnostic
    instead of repeating known-impossible learner fits.
    """

    if (
        len(positive_indices) < 2
        or len(unlabeled_indices) < 2
        or bags < 1
        or unlabeled_to_positive_ratio <= 0
    ):
        return delegate(
            learner_id=learner_id,
            x_train=x_train,
            predict_x=predict_x,
            positive_indices=positive_indices,
            unlabeled_indices=unlabeled_indices,
            bags=bags,
            unlabeled_to_positive_ratio=unlabeled_to_positive_ratio,
            training_cost=training_cost,
            scenario=scenario,
            rng=rng,
        )

    sampled_unlabeled_count = min(
        len(unlabeled_indices),
        max(2, int(math.ceil(len(positive_indices) * unlabeled_to_positive_ratio))),
    )
    if len(positive_indices) + sampled_unlabeled_count < _MINIMUM_PU_CLASSIFIER_ROWS:
        legacy.record_resampling_failure("InsufficientPUData")
        prior = len(positive_indices) / max(1, len(positive_indices) + len(unlabeled_indices))
        return np.full(len(predict_x), prior, dtype=float), 0.0, (1.0, 1.0)

    return delegate(
        learner_id=learner_id,
        x_train=x_train,
        predict_x=predict_x,
        positive_indices=positive_indices,
        unlabeled_indices=unlabeled_indices,
        bags=bags,
        unlabeled_to_positive_ratio=unlabeled_to_positive_ratio,
        training_cost=training_cost,
        scenario=scenario,
        rng=rng,
    )


def _generate_replication_v24(
    scenario: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, object]:
    n = int(scenario["sample_size"])
    empirical_base = legacy._replication_empirical_covariates(scenario, n, rng)
    latent, latent_probability_mean = legacy._draw_latent_truth(
        scenario=scenario,
        empirical_base=empirical_base,
        sample_size=n,
        rng=rng,
    )

    content = legacy._content_signal(
        scenario,
        latent,
        n,
        rng,
        base_covariates=empirical_base,
    )
    standardized_content = (content - content.mean()) / max(content.std(), 1e-12)
    observable_risk = 1.0 / (1.0 + np.exp(-standardized_content))
    selective = float(scenario.get("selective_verification_strength", 0.0))

    anchor_o = _opportunity_draw(
        n,
        float(scenario.get("anchor_opportunity_probability", 1.0)),
        rng,
    )
    weak_o = _opportunity_draw(
        n,
        float(scenario.get("weak_opportunity_probability", 1.0)),
        rng,
    )
    anchor_v = anchor_o & legacy._verification_draw(
        latent,
        observable_risk=observable_risk,
        base_probability=float(scenario.get("anchor_verification_probability", 1.0)),
        selective_strength=selective,
        rng=rng,
    )
    weak_v = weak_o & legacy._verification_draw(
        latent,
        observable_risk=observable_risk,
        base_probability=float(scenario.get("weak_verification_probability", 1.0)),
        selective_strength=selective,
        rng=rng,
    )

    shared_uniform = rng.random(n)
    dependence = float(scenario.get("channel_dependence", 0.0))
    anchor_d = legacy._source_draw(
        latent,
        sensitivity=float(scenario["anchor_sensitivity"]),
        false_positive=float(scenario["anchor_false_positive"]),
        shared_uniform=shared_uniform,
        dependence=dependence,
        rng=rng,
    )
    weak_d = legacy._source_draw(
        latent,
        sensitivity=float(scenario["weak_sensitivity"]),
        false_positive=float(scenario["weak_false_positive"]),
        shared_uniform=shared_uniform,
        dependence=dependence,
        rng=rng,
    )

    anchor_r = anchor_v & _recording_draw(
        anchor_d,
        base_probability=float(scenario.get("anchor_recording_probability", 1.0)),
        positive_multiplier=float(scenario.get("anchor_positive_recording_multiplier", 1.0)),
        rng=rng,
    )
    weak_r = weak_v & _recording_draw(
        weak_d,
        base_probability=float(scenario.get("weak_recording_probability", 1.0)),
        positive_multiplier=float(scenario.get("weak_positive_recording_multiplier", 1.0)),
        rng=rng,
    )

    delay = rng.exponential(float(scenario.get("detection_delay_mean_days", 30.0)), n)
    mature = delay <= float(scenario.get("horizon_days", 365.0))
    anchor_s_observed = anchor_o & anchor_v & anchor_r & mature
    weak_s_observed = weak_o & weak_v & weak_r & mature
    anchor_values = np.asarray(
        [
            bool(value) if observed else None
            for value, observed in zip(anchor_d, anchor_s_observed, strict=True)
        ],
        dtype=object,
    )
    weak_values = np.asarray(
        [
            bool(value) if observed else None
            for value, observed in zip(weak_d, weak_s_observed, strict=True)
        ],
        dtype=object,
    )

    source_rows = [
        {"anchor": anchor_value, "weak": weak_value}
        for anchor_value, weak_value in zip(anchor_values, weak_values, strict=True)
    ]
    l1_values = np.asarray([aggregate_l1(value) for value in source_rows], dtype=object)
    l2_values = [evidence_score_l2(value) for value in source_rows]
    fixed_pi = float(scenario.get("fixed_pi", scenario["prevalence"]))
    l2 = np.asarray(
        [fixed_pi if value is None else float(value) for value in l2_values],
        dtype=float,
    )

    features = legacy._feature_matrix(
        scenario=scenario,
        latent=latent,
        standardized_content=standardized_content,
        l2=l2,
        anchor_values=anchor_values,
        weak_values=weak_values,
        anchor_observed=anchor_v,
        weak_observed=weak_v,
        rng=rng,
        base_covariates=empirical_base,
    )
    observed_positive = np.asarray([value is True for value in l1_values], dtype=bool)
    label_observed = np.asarray([value is not None for value in l1_values], dtype=bool)
    verified = anchor_v | weak_v
    determination_positive = (anchor_v & anchor_d) | (weak_v & weak_d)
    feasible_target = determination_positive.astype(int)

    permutation = rng.permutation(n)
    test_fraction = float(scenario.get("test_fraction", 0.30))
    test_count = min(n - 20, max(20, int(round(n * test_fraction))))
    test_index = permutation[:test_count]
    train_index = permutation[test_count:]

    accuracy = {
        "anchor": (
            float(scenario["anchor_sensitivity"]),
            1.0 - float(scenario["anchor_false_positive"]),
        ),
        "weak": (
            float(scenario["weak_sensitivity"]),
            1.0 - float(scenario["weak_false_positive"]),
        ),
    }
    return {
        "latent": latent,
        "features": features,
        "observed_positive": observed_positive,
        "verified": verified,
        "feasible_target": feasible_target,
        "source_rows": source_rows,
        "accuracy": accuracy,
        "train_index": train_index,
        "test_index": test_index,
        "verification_rate": float(verified.mean()),
        "known_label_rate": float(label_observed.mean()),
        "opportunity_rate": float(np.mean(anchor_o | weak_o)),
        "determination_positive_rate": float(determination_positive.mean()),
        "recording_rate": float(np.mean(anchor_r | weak_r)),
        "maturity_rate": float(mature.mean()),
        "latent_probability_mean": float(latent_probability_mean),
        "measurement_process": {
            "anchor": {"O": anchor_o, "V": anchor_v, "D": anchor_d, "R": anchor_r},
            "weak": {"O": weak_o, "V": weak_v, "D": weak_d, "R": weak_r},
            OBSERVED_SOURCE_LABEL: {
                "anchor": anchor_values,
                "weak": weak_values,
                "label_observed": label_observed,
            },
        },
    }


def _partial_identification_bounds_v24(
    *,
    source_rows: list[dict[str, bool | None]],
    scenario: Mapping[str, Any],
) -> tuple[float, float, float]:
    anchor_values = [row["anchor"] for row in source_rows if row["anchor"] is not None]
    if not anchor_values:
        return 0.5, 0.0, 1.0
    observed_rate = float(np.mean(np.asarray(anchor_values, dtype=float)))

    false_positive_upper = scenario.get("identification_false_positive_upper")
    sensitivity_lower = scenario.get("identification_sensitivity_lower")
    independently_justified = bool(
        scenario.get("identification_restrictions_independently_justified", False)
    )
    bounds = prevalence_bounds(
        observed_rate,
        false_positive_upper=(
            float(false_positive_upper) if isinstance(false_positive_upper, (int, float)) else None
        ),
        sensitivity_lower=(
            float(sensitivity_lower) if isinstance(sensitivity_lower, (int, float)) else None
        ),
        restrictions_independently_justified=independently_justified,
    )
    lower = bounds.lower
    upper = bounds.upper
    estimate = (lower + upper) / 2.0
    return estimate, lower, upper


def _opportunity_draw(
    count: int,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if probability < 0.0 or probability > 1.0:
        raise ValueError("opportunity probability must be in [0, 1]")
    return rng.random(count) < probability


def _recording_draw(
    determination: np.ndarray,
    *,
    base_probability: float,
    positive_multiplier: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if base_probability < 0.0 or base_probability > 1.0:
        raise ValueError("recording probability must be in [0, 1]")
    if positive_multiplier < 0.0:
        raise ValueError("positive recording multiplier must be nonnegative")
    probability = np.where(
        determination,
        np.clip(base_probability * positive_multiplier, 0.0, 1.0),
        base_probability,
    )
    return rng.random(len(determination)) < probability
