"""Chapter 3 L3 correct and misspecified standalone estimators.

The four variants are evaluated on the same replication-level DGP:

- ``l3_correct``: uses the locked source accuracies and channel-dependence parameter.
- ``l3_ignore_dependence``: forces conditional independence between sources.
- ``l3_wrong_fixed_pi``: fixes prevalence to a prespecified incorrect value.
- ``l3_clean_anchor``: incorrectly sets the anchor false-positive rate to zero.

Missing source values are treated as unavailable evidence. The variants isolate
measurement-model assumptions; selective-verification adjustment remains a
separate simulation dimension.
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
    "fit_success",
    "prevalence_error",
    "prevalence_squared_error",
    "prevalence_coverage",
    "interval_width",
    "misspecification_regret",
    "empirical_panel_rows",
    "empirical_observed_positive_rate",
    "empirical_known_label_rate",
    "calibrated_latent_prevalence",
)

L3_VARIANT_SPECS: dict[str, dict[str, object]] = {
    "l3_correct": {
        "description": (
            "L3 latent-class prevalence estimator with locked source accuracy "
            "and channel dependence."
        ),
        "chapter_2_role": "correctly_specified_l3_reference",
        "assumption_role": "correct",
    },
    "l3_ignore_dependence": {
        "description": (
            "L3 misspecification that forces conditional independence across "
            "the anchor and weak source."
        ),
        "chapter_2_role": "l3_dependence_misspecification",
        "assumption_role": "misspecified",
    },
    "l3_wrong_fixed_pi": {
        "description": (
            "L3 misspecification that fixes prevalence to a prespecified "
            "incorrect value."
        ),
        "chapter_2_role": "l3_fixed_pi_misspecification",
        "assumption_role": "misspecified",
    },
    "l3_clean_anchor": {
        "description": (
            "L3 misspecification that incorrectly assumes the anchor has no "
            "false positives."
        ),
        "chapter_2_role": "l3_clean_anchor_misspecification",
        "assumption_role": "misspecified",
    },
}


def extend_method_registry(
    simulation: Mapping[str, object],
    registry: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Append protocol-locked L3 variants to the complete method registry."""

    configured_raw = simulation.get("l3_variants")
    if not isinstance(configured_raw, list):
        raise ValueError("simulation.l3_variants must be a list")
    configured = [str(value) for value in configured_raw]
    if len(configured) != len(set(configured)):
        raise ValueError("simulation.l3_variants must be unique")
    if set(configured) != set(L3_VARIANT_IDS):
        raise ValueError(
            "simulation.l3_variants must register exactly "
            f"{sorted(L3_VARIANT_IDS)}"
        )

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
                    "latent_parameter",
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
) -> dict[str, float]:
    """Estimate prevalence under one locked L3 assumption set."""

    if variant_id not in L3_VARIANT_IDS:
        raise ValueError(f"unsupported L3 variant={variant_id}")

    settings_raw = scenario.get("l3_variant_settings")
    if not isinstance(settings_raw, dict):
        raise ValueError("scenario is missing l3_variant_settings")
    settings = cast(Mapping[str, object], settings_raw)
    _validate_settings(settings)

    correct_estimate, correct_lower, correct_upper = _posterior_pi(
        source_rows=source_rows,
        accuracy=accuracy,
        dependence=float(scenario.get("channel_dependence", 0.0)),
        settings=settings,
    )

    if variant_id == "l3_correct":
        estimate, lower, upper = (
            correct_estimate,
            correct_lower,
            correct_upper,
        )
    elif variant_id == "l3_ignore_dependence":
        estimate, lower, upper = _posterior_pi(
            source_rows=source_rows,
            accuracy=accuracy,
            dependence=0.0,
            settings=settings,
        )
    elif variant_id == "l3_clean_anchor":
        clean_accuracy = dict(accuracy)
        anchor_sensitivity, _ = clean_accuracy["anchor"]
        clean_accuracy["anchor"] = (anchor_sensitivity, 1.0)
        estimate, lower, upper = _posterior_pi(
            source_rows=source_rows,
            accuracy=clean_accuracy,
            dependence=float(scenario.get("channel_dependence", 0.0)),
            settings=settings,
        )
    else:
        offset = float(settings["wrong_fixed_pi_offset"])
        fixed_pi = float(scenario.get("fixed_pi", scenario["prevalence"]))
        estimate = float(
            np.clip(
                fixed_pi + offset,
                float(settings["posterior_grid_minimum"]),
                float(settings["posterior_grid_maximum"]),
            )
        )
        lower = estimate
        upper = estimate

    true_prevalence = float(scenario["prevalence"])
    correct_squared_error = (correct_estimate - true_prevalence) ** 2
    squared_error = (estimate - true_prevalence) ** 2
    return {
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
        "correct_reference_estimate": correct_estimate,
        "misspecification_regret": squared_error - correct_squared_error,
    }


def run_l3_variant_batch(
    scenario: Mapping[str, Any],
    *,
    method_id: str,
    replications: range,
    data_rng_factory: Callable[[int], np.random.Generator],
    diagnostics: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run one L3 variant over a deterministic replication range."""

    if method_id not in L3_VARIANT_IDS:
        raise ValueError(f"unsupported L3 variant={method_id}")

    # Import lazily to keep P08A registry construction independent of the DGP.
    from simulation.service import _generate_replication

    rows: list[dict[str, object]] = []
    for replication_id in replications:
        data = _generate_replication(
            scenario,
            data_rng_factory(int(replication_id)),
        )
        source_rows = cast(
            list[dict[str, bool | None]],
            data["source_rows"],
        )
        accuracy = cast(
            dict[str, tuple[float, float]],
            data["accuracy"],
        )
        result = estimate_l3_variant(
            variant_id=method_id,
            source_rows=source_rows,
            accuracy=accuracy,
            scenario=scenario,
        )
        true_prevalence = float(scenario["prevalence"])
        estimate = float(result["estimate"])
        lower = float(result["lower"])
        upper = float(result["upper"])

        metrics = {
            "fit_success": 1.0,
            "prevalence_error": estimate - true_prevalence,
            "prevalence_squared_error": (
                estimate - true_prevalence
            ) ** 2,
            "prevalence_coverage": float(
                lower <= true_prevalence <= upper
            ),
            "interval_width": upper - lower,
            "misspecification_regret": float(
                result["misspecification_regret"]
            ),
            "empirical_panel_rows": float(
                scenario.get("empirical_panel_rows", 0.0)
            ),
            "empirical_observed_positive_rate": float(
                scenario.get(
                    "empirical_observed_positive_rate",
                    np.mean(
                        [
                            any(value is True for value in row.values())
                            for row in source_rows
                        ]
                    ),
                )
            ),
            "empirical_known_label_rate": float(
                scenario.get(
                    "empirical_known_label_rate",
                    np.mean(
                        [
                            any(value is not None for value in row.values())
                            for row in source_rows
                        ]
                    ),
                )
            ),
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


def _posterior_pi(
    *,
    source_rows: Sequence[Mapping[str, bool | None]],
    accuracy: Mapping[str, tuple[float, float]],
    dependence: float,
    settings: Mapping[str, object],
) -> tuple[float, float, float]:
    grid = np.linspace(
        float(settings["posterior_grid_minimum"]),
        float(settings["posterior_grid_maximum"]),
        int(settings["posterior_grid_points"]),
    )
    log_likelihood = np.zeros(len(grid), dtype=float)

    for row in source_rows:
        likelihood_one = _pattern_probability(
            row=row,
            latent=True,
            accuracy=accuracy,
            dependence=dependence,
        )
        likelihood_zero = _pattern_probability(
            row=row,
            latent=False,
            accuracy=accuracy,
            dependence=dependence,
        )
        mixture = (
            grid * likelihood_one
            + (1.0 - grid) * likelihood_zero
        )
        log_likelihood += np.log(np.clip(mixture, 1e-12, None))

    posterior = np.exp(log_likelihood - np.max(log_likelihood))
    total = float(posterior.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError("invalid L3 posterior normalization")
    posterior /= total

    estimate = float(np.sum(grid * posterior))
    mass = float(settings["credible_interval_mass"])
    tail = (1.0 - mass) / 2.0
    cumulative = np.cumsum(posterior)
    lower_index = min(
        int(np.searchsorted(cumulative, tail)),
        len(grid) - 1,
    )
    upper_index = min(
        int(np.searchsorted(cumulative, 1.0 - tail)),
        len(grid) - 1,
    )
    return estimate, float(grid[lower_index]), float(grid[upper_index])


def _pattern_probability(
    *,
    row: Mapping[str, bool | None],
    latent: bool,
    accuracy: Mapping[str, tuple[float, float]],
    dependence: float,
) -> float:
    anchor_value = row.get("anchor")
    weak_value = row.get("weak")
    if anchor_value is None and weak_value is None:
        return 1.0

    anchor_sensitivity, anchor_specificity = accuracy["anchor"]
    weak_sensitivity, weak_specificity = accuracy["weak"]
    p_anchor = (
        float(anchor_sensitivity)
        if latent
        else 1.0 - float(anchor_specificity)
    )
    p_weak = (
        float(weak_sensitivity)
        if latent
        else 1.0 - float(weak_specificity)
    )

    if weak_value is None:
        return p_anchor if bool(anchor_value) else 1.0 - p_anchor
    if anchor_value is None:
        return p_weak if bool(weak_value) else 1.0 - p_weak

    p11 = (
        dependence**2 * min(p_anchor, p_weak)
        + (1.0 - dependence**2) * p_anchor * p_weak
    )
    probabilities = {
        (True, True): p11,
        (True, False): p_anchor - p11,
        (False, True): p_weak - p11,
        (False, False): 1.0 - p_anchor - p_weak + p11,
    }
    return float(
        np.clip(
            probabilities[(bool(anchor_value), bool(weak_value))],
            1e-12,
            1.0,
        )
    )


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

    minimum = float(settings["posterior_grid_minimum"])
    maximum = float(settings["posterior_grid_maximum"])
    points = settings["posterior_grid_points"]
    mass = float(settings["credible_interval_mass"])
    offset = float(settings["wrong_fixed_pi_offset"])

    if not 0.0 < minimum < maximum < 1.0:
        raise ValueError("L3 posterior grid must lie strictly inside (0, 1)")
    if not isinstance(points, int) or points < 100:
        raise ValueError("L3 posterior_grid_points must be an integer >= 100")
    if not 0.5 < mass < 1.0:
        raise ValueError("L3 credible_interval_mass must be in (0.5, 1)")
    if math.isclose(offset, 0.0, abs_tol=1e-12):
        raise ValueError("L3 wrong_fixed_pi_offset must be nonzero")
