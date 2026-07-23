"""Chapter 3 fixed-pi L3 measurement and misspecification estimators.

The registered variants are evaluated on the same replication-level DGP.  The
primary estimand is row-level latent risk under a locked prevalence scenario.
Prevalence recovery is retained only as a hierarchical-pi sensitivity
diagnostic; it is not used as evidence that the fixed-pi target recovered a
unique latent truth.

The four locked variants are:

- ``l3_correct``: locked source accuracy, channel dependence, and verification.
- ``l3_ignore_dependence``: conditional independence between sources.
- ``l3_wrong_fixed_pi``: a prespecified incorrect prevalence scenario.
- ``l3_clean_anchor``: an incorrectly perfect-specificity anchor.

Every misspecification regret is paired within replication against
``l3_correct``.  The row-level residual-variance metrics quantify how much a
proxy can reduce validation-review sample requirements in a downstream
prediction-powered inference design.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from core.semantic_keys import (
    ESTIMATE,
    MCSE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from labels.service import evidence_score_l2
from simulation.method_contract import (
    EVALUATION_TARGETS,
    IMBALANCE_TREATMENT_ID,
    LABEL_STRATEGY_ID,
    LEARNER_ID,
    LEARNER_TIER,
    METHOD_FAMILY,
    REQUIRED_METRICS,
    TRAINING_COST_REGIME_ID,
)

L3_VARIANT_IDS = (
    "l3_correct",
    "l3_ignore_dependence",
    "l3_wrong_fixed_pi",
    "l3_clean_anchor",
)

L3_REQUIRED_METRICS = (
    "fixed_pi_fit_success",
    "row_brier",
    "row_log_loss",
    "row_calibration_bias",
    "row_residual_variance",
    "row_probability_mean",
    "row_probability_sd",
    "paired_brier_regret",
    "variance_reduction_vs_l1",
    "variance_reduction_vs_l2",
    "review_sample_size_ratio_vs_l1",
    "review_sample_size_ratio_vs_l2",
    "hierarchical_pi_error",
    "hierarchical_pi_squared_error",
    "hierarchical_pi_coverage",
    "hierarchical_pi_interval_width",
    "empirical_panel_rows",
    "empirical_observed_positive_rate",
    "empirical_known_label_rate",
    "calibrated_latent_prevalence",
)

L3_VARIANT_SPECS: dict[str, dict[str, object]] = {
    "l3_correct": {
        "description": (
            "Fixed-pi L3 row-risk estimator with locked source accuracy, "
            "channel dependence, and verification mechanism."
        ),
        "chapter_2_role": "correctly_specified_l3_reference",
        "assumption_role": "correct",
    },
    "l3_ignore_dependence": {
        "description": (
            "Fixed-pi L3 misspecification that forces conditional independence "
            "across the anchor and weak source."
        ),
        "chapter_2_role": "l3_dependence_misspecification",
        "assumption_role": "misspecified",
    },
    "l3_wrong_fixed_pi": {
        "description": (
            "Fixed-pi L3 misspecification using a prespecified incorrect prevalence scenario."
        ),
        "chapter_2_role": "l3_fixed_pi_misspecification",
        "assumption_role": "misspecified",
    },
    "l3_clean_anchor": {
        "description": (
            "Fixed-pi L3 misspecification that incorrectly assumes the anchor "
            "has no false positives."
        ),
        "chapter_2_role": "l3_clean_anchor_misspecification",
        "assumption_role": "misspecified",
    },
}


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def extend_method_registry(
    simulation: Mapping[str, object],
    registry: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Append protocol-locked L3 variants to the complete method registry."""
    configured_raw = simulation.get("l3_variants")
    if not isinstance(configured_raw, list):
        raise ValueError("simulation.l3_variants must be a list")
    configured_values = cast(list[object], configured_raw)
    configured = [str(value) for value in configured_values]
    if len(configured) != len(set(configured)):
        raise ValueError("simulation.l3_variants must be unique")
    if set(configured) != set(L3_VARIANT_IDS):
        raise ValueError(f"simulation.l3_variants must register exactly {sorted(L3_VARIANT_IDS)}")

    settings = simulation.get("l3_variant_settings")
    if not isinstance(settings, dict):
        raise ValueError("simulation.l3_variant_settings mapping is required")
    _validate_settings(cast(Mapping[str, object], settings))

    output = [dict(item) for item in registry]
    seen = {str(item[METHOD_ID]) for item in output}
    collisions = sorted(set(configured) & seen)
    if collisions:
        raise ValueError(f"L3 method_id collision: {collisions}")

    for method_id in configured:
        spec = L3_VARIANT_SPECS[method_id]
        output.append(
            {
                METHOD_ID: method_id,
                LABEL_STRATEGY_ID: method_id,
                LEARNER_ID: "none",
                LEARNER_TIER: "standalone",
                TRAINING_COST_REGIME_ID: "not_applicable",
                IMBALANCE_TREATMENT_ID: "not_applicable",
                METHOD_FAMILY: "standalone_estimator",
                "estimator": dict(spec),
                REQUIRED_METRICS: list(L3_REQUIRED_METRICS),
                EVALUATION_TARGETS: [
                    "fixed_pi_row_risk",
                    "validation_review_efficiency",
                    "hierarchical_pi_sensitivity",
                    "l3_misspecification",
                ],
            }
        )
    return output


def estimate_l3_variant(
    *,
    variant_id: str,
    source_rows: Sequence[Mapping[str, bool | None]],
    accuracy: Mapping[str, tuple[float, float]],
    scenario: Mapping[str, object],
    observable_risk: np.ndarray | None = None,
) -> dict[str, object]:
    """Estimate fixed-pi row risks and hierarchical-pi sensitivity diagnostics."""
    if variant_id not in L3_VARIANT_IDS:
        raise ValueError(f"unsupported L3 variant={variant_id}")

    settings_raw = scenario.get("l3_variant_settings")
    if not isinstance(settings_raw, dict):
        raise ValueError("scenario is missing l3_variant_settings")
    settings = cast(Mapping[str, object], settings_raw)
    _validate_settings(settings)

    risks = _validated_observable_risk(source_rows, observable_risk)
    correct_accuracy = dict(accuracy)
    correct_dependence = _optional_float(scenario, "channel_dependence", 0.0)
    correct_fixed_pi = _optional_float(
        scenario,
        "fixed_pi",
        _required_float(scenario, "prevalence"),
    )

    variant_accuracy = dict(correct_accuracy)
    variant_dependence = correct_dependence
    variant_fixed_pi = correct_fixed_pi
    if variant_id == "l3_ignore_dependence":
        variant_dependence = 0.0
    elif variant_id == "l3_wrong_fixed_pi":
        variant_fixed_pi = min(
            max(
                correct_fixed_pi + _required_float(settings, "wrong_fixed_pi_offset"),
                _required_float(settings, "posterior_grid_minimum"),
            ),
            _required_float(settings, "posterior_grid_maximum"),
        )
    elif variant_id == "l3_clean_anchor":
        anchor_sensitivity, _ = variant_accuracy["anchor"]
        variant_accuracy["anchor"] = (anchor_sensitivity, 1.0)

    correct_probabilities = _fixed_pi_row_probabilities(
        source_rows=source_rows,
        accuracy=correct_accuracy,
        dependence=correct_dependence,
        fixed_pi=correct_fixed_pi,
        scenario=scenario,
        observable_risk=risks,
    )
    probabilities = _fixed_pi_row_probabilities(
        source_rows=source_rows,
        accuracy=variant_accuracy,
        dependence=variant_dependence,
        fixed_pi=variant_fixed_pi,
        scenario=scenario,
        observable_risk=risks,
    )

    hierarchical_estimate, hierarchical_lower, hierarchical_upper = _posterior_pi(
        source_rows=source_rows,
        accuracy=variant_accuracy,
        dependence=variant_dependence,
        settings=settings,
        scenario=scenario,
        observable_risk=risks,
    )
    return {
        ESTIMATE: float(np.mean(probabilities)),
        "row_probabilities": probabilities,
        "correct_reference_probabilities": correct_probabilities,
        "fixed_pi_used": variant_fixed_pi,
        "hierarchical_pi_estimate": hierarchical_estimate,
        "hierarchical_pi_lower": hierarchical_lower,
        "hierarchical_pi_upper": hierarchical_upper,
    }


def run_l3_variant_batch(
    scenario: Mapping[str, Any],
    *,
    method_id: str,
    replications: range,
    data_rng_factory: Callable[[int], np.random.Generator],
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run one fixed-pi L3 variant over a deterministic paired replication range."""
    if method_id not in L3_VARIANT_IDS:
        raise ValueError(f"unsupported L3 variant={method_id}")

    from simulation.service import _generate_replication  # pyright: ignore[reportPrivateUsage]

    rows: list[dict[str, object]] = []
    for replication_id in replications:
        data = _generate_replication(
            scenario,
            data_rng_factory(int(replication_id)),
        )
        source_rows = cast(list[dict[str, bool | None]], data["source_rows"])
        accuracy = cast(dict[str, tuple[float, float]], data["accuracy"])
        latent = np.asarray(data["latent"], dtype=float)
        standardized_content = np.asarray(data["features"], dtype=float)[:, 0]
        observable_risk = 1.0 / (1.0 + np.exp(-standardized_content))
        result = estimate_l3_variant(
            variant_id=method_id,
            source_rows=source_rows,
            accuracy=accuracy,
            scenario=scenario,
            observable_risk=observable_risk,
        )

        probabilities = np.asarray(result["row_probabilities"], dtype=float)
        correct_probabilities = np.asarray(
            result["correct_reference_probabilities"],
            dtype=float,
        )
        fixed_pi = _optional_float(
            scenario,
            "fixed_pi",
            _required_float(scenario, "prevalence"),
        )
        l1_proxy = _l1_proxy(source_rows, fixed_pi)
        l2_proxy = _l2_proxy(source_rows, fixed_pi)

        row_brier = float(np.mean((latent - probabilities) ** 2))
        correct_brier = float(np.mean((latent - correct_probabilities) ** 2))
        residual_variance = _sample_variance(latent - probabilities)
        l1_residual_variance = _sample_variance(latent - l1_proxy)
        l2_residual_variance = _sample_variance(latent - l2_proxy)
        ratio_l1 = _safe_variance_ratio(residual_variance, l1_residual_variance)
        ratio_l2 = _safe_variance_ratio(residual_variance, l2_residual_variance)

        true_prevalence = _required_float(scenario, "prevalence")
        hierarchical_estimate = cast(float, result["hierarchical_pi_estimate"])
        hierarchical_lower = cast(float, result["hierarchical_pi_lower"])
        hierarchical_upper = cast(float, result["hierarchical_pi_upper"])
        empirical_panel_rows = _optional_float(scenario, "empirical_panel_rows", 0.0)
        observed_positive_rate = _optional_float(
            scenario,
            "empirical_observed_positive_rate",
            float(np.mean([any(value is True for value in row.values()) for row in source_rows])),
        )
        known_label_rate = _optional_float(
            scenario,
            "empirical_known_label_rate",
            float(
                np.mean([any(value is not None for value in row.values()) for row in source_rows])
            ),
        )
        metrics: dict[str, float] = {
            "fixed_pi_fit_success": 1.0,
            "row_brier": row_brier,
            "row_log_loss": _binary_log_loss(latent, probabilities),
            "row_calibration_bias": float(np.mean(probabilities - latent)),
            "row_residual_variance": residual_variance,
            "row_probability_mean": float(np.mean(probabilities)),
            "row_probability_sd": float(np.std(probabilities, ddof=1)),
            "paired_brier_regret": row_brier - correct_brier,
            "variance_reduction_vs_l1": 1.0 - ratio_l1,
            "variance_reduction_vs_l2": 1.0 - ratio_l2,
            "review_sample_size_ratio_vs_l1": ratio_l1,
            "review_sample_size_ratio_vs_l2": ratio_l2,
            "hierarchical_pi_error": hierarchical_estimate - true_prevalence,
            "hierarchical_pi_squared_error": (hierarchical_estimate - true_prevalence) ** 2,
            "hierarchical_pi_coverage": float(
                hierarchical_lower <= true_prevalence <= hierarchical_upper
            ),
            "hierarchical_pi_interval_width": hierarchical_upper - hierarchical_lower,
            "empirical_panel_rows": empirical_panel_rows,
            "empirical_observed_positive_rate": observed_positive_rate,
            "empirical_known_label_rate": known_label_rate,
            "calibrated_latent_prevalence": true_prevalence,
        }
        for metric_id, value in metrics.items():
            rows.append(
                {
                    SCENARIO_ID: str(scenario[SCENARIO_ID]),
                    METHOD_ID: method_id,
                    REPLICATION_ID: int(replication_id),
                    METRIC_ID: metric_id,
                    ESTIMATE: float(value),
                    MCSE: None,
                }
            )

    if diagnostics is not None:
        diagnostics.update(
            {
                "fit_failures": {},
                "resampling_failures": {},
                "warnings": {},
                "affected_replication_ids": [],
            }
        )

    return pd.DataFrame(rows).astype(
        {
            SCENARIO_ID: "string",
            METHOD_ID: "string",
            REPLICATION_ID: "int64",
            METRIC_ID: "string",
            ESTIMATE: "float64",
            MCSE: "float64",
        }
    )


def _fixed_pi_row_probabilities(
    *,
    source_rows: Sequence[Mapping[str, bool | None]],
    accuracy: Mapping[str, tuple[float, float]],
    dependence: float,
    fixed_pi: float,
    scenario: Mapping[str, object],
    observable_risk: np.ndarray,
) -> np.ndarray:
    if not 0.0 < fixed_pi < 1.0:
        raise ValueError("fixed_pi must be in (0, 1)")
    output = np.empty(len(source_rows), dtype=float)
    for index, (row, risk) in enumerate(zip(source_rows, observable_risk, strict=True)):
        risk_value = cast(float, risk)
        likelihood_one = _pattern_probability(
            row=row,
            latent=True,
            accuracy=accuracy,
            dependence=dependence,
            scenario=scenario,
            observable_risk=risk_value,
        )
        likelihood_zero = _pattern_probability(
            row=row,
            latent=False,
            accuracy=accuracy,
            dependence=dependence,
            scenario=scenario,
            observable_risk=risk_value,
        )
        numerator = fixed_pi * likelihood_one
        denominator = numerator + (1.0 - fixed_pi) * likelihood_zero
        output[index] = numerator / max(denominator, 1e-300)
    return np.clip(output, 1e-9, 1.0 - 1e-9)


def _posterior_pi(
    *,
    source_rows: Sequence[Mapping[str, bool | None]],
    accuracy: Mapping[str, tuple[float, float]],
    dependence: float,
    settings: Mapping[str, object],
    scenario: Mapping[str, object],
    observable_risk: np.ndarray | None,
) -> tuple[float, float, float]:
    grid = np.linspace(
        _required_float(settings, "posterior_grid_minimum"),
        _required_float(settings, "posterior_grid_maximum"),
        _required_int(settings, "posterior_grid_points"),
    )
    log_likelihood = np.zeros(len(grid), dtype=float)
    risks = _validated_observable_risk(source_rows, observable_risk)

    for row, risk in zip(source_rows, risks, strict=True):
        risk_value = cast(float, risk)
        likelihood_one = _pattern_probability(
            row=row,
            latent=True,
            accuracy=accuracy,
            dependence=dependence,
            scenario=scenario,
            observable_risk=risk_value,
        )
        likelihood_zero = _pattern_probability(
            row=row,
            latent=False,
            accuracy=accuracy,
            dependence=dependence,
            scenario=scenario,
            observable_risk=risk_value,
        )
        mixture = grid * likelihood_one + (1.0 - grid) * likelihood_zero
        log_likelihood += np.log(np.clip(mixture, 1e-12, None))

    posterior = np.exp(log_likelihood - np.max(log_likelihood))
    total = float(posterior.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("invalid L3 posterior normalization")
    posterior /= total

    estimate = math.fsum(cast(list[float], (grid * posterior).tolist()))
    mass = _required_float(settings, "credible_interval_mass")
    tail = (1.0 - mass) / 2.0
    cumulative = np.cumsum(posterior)
    lower_index = min(int(np.searchsorted(cumulative, tail)), len(grid) - 1)
    upper_index = min(int(np.searchsorted(cumulative, 1.0 - tail)), len(grid) - 1)
    return estimate, float(grid[lower_index]), float(grid[upper_index])


def _pattern_probability(
    *,
    row: Mapping[str, bool | None],
    latent: bool,
    accuracy: Mapping[str, tuple[float, float]],
    dependence: float,
    scenario: Mapping[str, object],
    observable_risk: float,
) -> float:
    anchor_value = row.get("anchor")
    weak_value = row.get("weak")

    anchor_sensitivity, anchor_specificity = accuracy["anchor"]
    weak_sensitivity, weak_specificity = accuracy["weak"]
    p_anchor = float(anchor_sensitivity) if latent else 1.0 - float(anchor_specificity)
    p_weak = float(weak_sensitivity) if latent else 1.0 - float(weak_specificity)

    source_probability = 1.0
    if weak_value is None and anchor_value is not None:
        source_probability = p_anchor if bool(anchor_value) else 1.0 - p_anchor
    elif anchor_value is None and weak_value is not None:
        source_probability = p_weak if bool(weak_value) else 1.0 - p_weak
    elif anchor_value is not None and weak_value is not None:
        p11 = dependence * min(p_anchor, p_weak) + (1.0 - dependence) * p_anchor * p_weak
        probabilities = {
            (True, True): p11,
            (True, False): p_anchor - p11,
            (False, True): p_weak - p11,
            (False, False): 1.0 - p_anchor - p_weak + p11,
        }
        source_probability = probabilities[(bool(anchor_value), bool(weak_value))]

    selective = _optional_float(scenario, "selective_verification_strength", 0.0)
    latent_float = float(latent)
    q_anchor = float(
        np.clip(
            _optional_float(scenario, "anchor_verification_probability", 1.0)
            + 0.25 * selective * observable_risk
            + 0.75 * selective * latent_float,
            0.0,
            1.0,
        )
    )
    q_weak = float(
        np.clip(
            _optional_float(scenario, "weak_verification_probability", 1.0)
            + 0.25 * selective * observable_risk
            + 0.75 * selective * latent_float,
            0.0,
            1.0,
        )
    )
    horizon = _optional_float(scenario, "horizon_days", 365.0)
    delay_mean = _optional_float(scenario, "detection_delay_mean_days", 30.0)
    maturity_probability = 1.0 - math.exp(-horizon / delay_mean)

    anchor_present = anchor_value is not None
    weak_present = weak_value is not None
    if anchor_present and weak_present:
        observation_probability = maturity_probability * q_anchor * q_weak
    elif anchor_present:
        observation_probability = maturity_probability * q_anchor * (1.0 - q_weak)
    elif weak_present:
        observation_probability = maturity_probability * (1.0 - q_anchor) * q_weak
    else:
        observation_probability = (
            1.0 - maturity_probability + maturity_probability * (1.0 - q_anchor) * (1.0 - q_weak)
        )

    return float(
        np.clip(
            source_probability * observation_probability,
            1e-12,
            1.0,
        )
    )


def _validated_observable_risk(
    source_rows: Sequence[Mapping[str, bool | None]],
    observable_risk: np.ndarray | None,
) -> np.ndarray:
    risks = (
        np.full(len(source_rows), 0.5, dtype=float)
        if observable_risk is None
        else np.asarray(observable_risk, dtype=float)
    )
    if len(risks) != len(source_rows):
        raise ValueError("observable_risk length must match source_rows")
    if not np.all(np.isfinite(risks)):
        raise ValueError("observable_risk must be finite")
    return np.clip(risks, 0.0, 1.0)


def _l1_proxy(
    source_rows: Sequence[Mapping[str, bool | None]],
    fixed_pi: float,
) -> np.ndarray:
    values: list[float] = []
    for row in source_rows:
        observed = [value for value in row.values() if value is not None]
        if any(value is True for value in observed):
            values.append(1.0)
        elif observed:
            values.append(0.0)
        else:
            values.append(fixed_pi)
    return np.asarray(values, dtype=float)


def _l2_proxy(
    source_rows: Sequence[Mapping[str, bool | None]],
    fixed_pi: float,
) -> np.ndarray:
    return np.asarray(
        [
            fixed_pi if (value := evidence_score_l2(row)) is None else float(value)
            for row in source_rows
        ],
        dtype=float,
    )


def _binary_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(truth * np.log(clipped) + (1.0 - truth) * np.log(1.0 - clipped)))


def _sample_variance(values: np.ndarray) -> float:
    return float(np.var(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0


def _safe_variance_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-15 or not math.isfinite(denominator):
        return math.nan
    return float(numerator / denominator)


def _validate_settings(settings: Mapping[str, object]) -> None:
    required = {
        "posterior_grid_minimum",
        "posterior_grid_maximum",
        "posterior_grid_points",
        "credible_interval_mass",
        "wrong_fixed_pi_offset",
        "misspecification_regret_definition",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"l3_variant_settings missing={missing}")

    minimum = _required_float(settings, "posterior_grid_minimum")
    maximum = _required_float(settings, "posterior_grid_maximum")
    points = _required_int(settings, "posterior_grid_points")
    mass = _required_float(settings, "credible_interval_mass")
    offset = _required_float(settings, "wrong_fixed_pi_offset")

    if not 0.0 < minimum < maximum < 1.0:
        raise ValueError("L3 posterior grid must lie strictly inside (0, 1)")
    if points < 100:
        raise ValueError("L3 posterior_grid_points must be an integer >= 100")
    if not 0.5 < mass < 1.0:
        raise ValueError("L3 credible_interval_mass must be in (0.5, 1)")
    if math.isclose(offset, 0.0, abs_tol=1e-12):
        raise ValueError("L3 wrong_fixed_pi_offset must be nonzero")
