"""Bayesian latent-class MCMC with source accuracy and channel random effects."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class LatentClassResult:
    posterior_mean: list[float]
    posterior_draws: list[list[int]]
    parameter_draws: list[dict[str, Any]]
    source_accuracy: dict[str, dict[str, float]]
    channel_random_effect_sd: dict[str, float]
    diagnostics: dict[str, object]


def attach_l3_pilot_posterior(
    *,
    rows: list[dict[str, Any]],
    fixed_pi: float,
    posterior_mean: list[float],
) -> None:
    """Attach an aligned fixed-pi posterior to P05 matrix rows."""
    if len(rows) != len(posterior_mean):
        raise ValueError("L3 row-level posterior length does not match matrix rows")
    pi_key = format(fixed_pi, ".17g")
    for row, value in zip(rows, posterior_mean, strict=True):
        raw_by_pi = row.setdefault("l3_posterior_by_fixed_pi", {})
        if not isinstance(raw_by_pi, dict):
            raise ValueError("L3 posterior-by-pi matrix field must be an object")
        by_pi = cast(dict[str, object], raw_by_pi)
        by_pi[pi_key] = float(value)


def finalize_l3_pilot_posteriors(
    *,
    rows: list[dict[str, Any]],
    eligible_fixed_pi: list[float],
) -> None:
    """Expose the pilot-only mean without presenting it as a fold-local target."""
    eligible_keys = {format(value, ".17g") for value in eligible_fixed_pi}
    for row in rows:
        raw_by_pi = row.get("l3_posterior_by_fixed_pi")
        by_pi = cast(dict[str, object], raw_by_pi) if isinstance(raw_by_pi, dict) else {}
        eligible_values: list[float] = []
        for key, value in by_pi.items():
            if key not in eligible_keys:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("L3 row-level posterior must be numeric")
            eligible_values.append(float(value))
        row["l3_posterior_mean"] = float(np.mean(eligible_values)) if eligible_values else None
        row["l3_posterior_scope"] = "p05_feasibility_pilot_only"
        row["l3_posterior_aggregation"] = "mean_across_diagnostically_eligible_fixed_pi_grid"


def predict_fixed_pi_latent_probability(
    *,
    rows: list[dict[str, Any]],
    source_channels: dict[str, str],
    source_accuracy: dict[str, dict[str, float]],
    channel_random_effect_sd: dict[str, float],
    fixed_pi: float,
    quadrature_points: int = 20,
) -> list[float]:
    """Apply a frozen L3 scenario, integrating shared channel effects."""
    if not 0 < fixed_pi < 1:
        raise ValueError("fixed_pi must be in (0, 1)")
    if quadrature_points < 3:
        raise ValueError("L3 predictive quadrature requires at least three points")
    if set(source_channels) != set(source_accuracy):
        raise ValueError("L3 predictive accuracy must bind every source")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    normalized_weights = weights / math.sqrt(math.pi)
    result: list[float] = []
    for row in rows:
        raw_outcomes = row.get("source_outcomes")
        if not isinstance(raw_outcomes, dict):
            raise ValueError("L3 predictive row requires source_outcomes")
        outcomes = cast(dict[str, object], raw_outcomes)
        log_one = math.log(fixed_pi)
        log_zero = math.log(1.0 - fixed_pi)
        for channel in sorted(set(source_channels.values())):
            sigma = channel_random_effect_sd.get(channel)
            if sigma is None or sigma < 0:
                raise ValueError(f"channel={channel}: frozen random-effect SD required")
            channel_sources = [
                source
                for source, source_channel in source_channels.items()
                if source_channel == channel and outcomes.get(source) is not None
            ]
            if not channel_sources:
                continue
            likelihood_one = 0.0
            likelihood_zero = 0.0
            for node, weight in zip(nodes, normalized_weights, strict=True):
                random_effect = math.sqrt(2.0) * float(sigma) * float(node)
                conditional_one = 1.0
                conditional_zero = 1.0
                for source in channel_sources:
                    value = outcomes[source]
                    if not isinstance(value, bool):
                        raise ValueError("L3 source outcomes must be bool or missing")
                    accuracy = source_accuracy[source]
                    sensitivity = float(accuracy["sensitivity_mean"])
                    false_positive = 1.0 - float(accuracy["specificity_mean"])
                    sensitivity_logit = float(_logit(np.asarray([sensitivity]))[0])
                    false_positive_logit = float(_logit(np.asarray([false_positive]))[0])
                    probability_one = float(
                        _expit(np.asarray([sensitivity_logit + random_effect]))[0]
                    )
                    probability_zero = float(
                        _expit(np.asarray([false_positive_logit + random_effect]))[0]
                    )
                    conditional_one *= probability_one if value else 1.0 - probability_one
                    conditional_zero *= probability_zero if value else 1.0 - probability_zero
                likelihood_one += float(weight) * conditional_one
                likelihood_zero += float(weight) * conditional_zero
            log_one += math.log(max(likelihood_one, 1e-300))
            log_zero += math.log(max(likelihood_zero, 1e-300))
        result.append(float(_expit(np.asarray([log_one - log_zero]))[0]))
    return result


def fit_fixed_pi_latent_class(
    *,
    rows: list[dict[str, Any]],
    source_channels: dict[str, str],
    accuracy_priors: dict[str, dict[str, float]],
    fixed_pi: float,
    chains: int,
    warmup: int,
    draws: int,
    alpha_step: float,
    random_effect_step: float,
    rhat_maximum: float,
    ess_minimum: float,
    ppc_rate_error_maximum: float,
    minimum_observations_per_source: int,
    rng: np.random.Generator,
) -> LatentClassResult:
    """Fit equation 3.15 using data-augmented Metropolis-within-Gibbs MCMC.

    Missingness remains distinct from a negative source outcome.  ``fixed_pi`` is
    never estimated in this primary model.  A row/channel random intercept is
    sampled for every observed channel opportunity and its variance is updated
    from an inverse-gamma prior.
    """
    _validate_controls(
        fixed_pi=fixed_pi,
        chains=chains,
        warmup=warmup,
        draws=draws,
        alpha_step=alpha_step,
        random_effect_step=random_effect_step,
    )
    sources = sorted(source_channels)
    if set(accuracy_priors) != set(sources):
        missing = sorted(set(sources) - set(accuracy_priors))
        extra = sorted(set(accuracy_priors) - set(sources))
        raise ValueError(
            f"L3 accuracy priors must bind every source; missing={missing}, extra={extra}"
        )
    channels = sorted(set(source_channels.values()))
    source_index = {value: index for index, value in enumerate(sources)}
    channel_index = {value: index for index, value in enumerate(channels)}
    source_channel = np.asarray(
        [channel_index[source_channels[source]] for source in sources], dtype=int
    )
    observed = np.full((len(rows), len(sources)), -1, dtype=np.int8)
    for row_index, row in enumerate(rows):
        raw = row.get("source_outcomes")
        if not isinstance(raw, dict):
            raise ValueError("L3 row requires source_outcomes")
        outcomes = cast(dict[str, object], raw)
        for source, value in outcomes.items():
            if source not in source_index or value is None:
                continue
            if not isinstance(value, bool):
                raise ValueError(f"source={source}: L3 outcomes must be bool or missing")
            observed[row_index, source_index[source]] = int(value)
    counts = {source: int(np.sum(observed[:, source_index[source]] >= 0)) for source in sources}
    if not rows or not np.any(observed >= 0):
        raise ValueError("L3 requires observed source outcomes")
    prior_arrays = _prior_arrays(sources, accuracy_priors)
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=chains, dtype=np.uint32)
    chain_y: list[np.ndarray] = []
    chain_parameters: list[np.ndarray] = []
    for seed in seeds:
        y_draws, chain_parameter_draws = _run_chain(
            observed=observed,
            source_channel=source_channel,
            fixed_pi=fixed_pi,
            warmup=warmup,
            draws=draws,
            alpha_step=alpha_step,
            random_effect_step=random_effect_step,
            priors=prior_arrays,
            channel_count=len(channels),
            rng=np.random.default_rng(int(seed)),
        )
        chain_y.append(y_draws)
        chain_parameters.append(chain_parameter_draws)
    y_array = np.stack(chain_y, axis=0)
    parameter_array = np.stack(chain_parameters, axis=0)
    flattened_y = y_array.reshape(chains * draws, len(rows))
    flattened_parameters = parameter_array.reshape(chains * draws, parameter_array.shape[2])
    rhat_values = _rhat(parameter_array)
    ess_values = _ess(parameter_array)
    source_accuracy: dict[str, dict[str, float]] = {}
    source_count = len(sources)
    for source, index in source_index.items():
        sensitivity = _expit(flattened_parameters[:, index])
        false_positive = _expit(flattened_parameters[:, source_count + index])
        source_accuracy[source] = {
            "sensitivity_mean": float(np.mean(sensitivity)),
            "sensitivity_sd": float(np.std(sensitivity, ddof=1)),
            "specificity_mean": float(np.mean(1.0 - false_positive)),
            "specificity_sd": float(np.std(false_positive, ddof=1)),
            "observed_count": float(counts[source]),
        }
    sigma_offset = 2 * source_count
    channel_sd = {
        channel: float(np.mean(np.exp(flattened_parameters[:, sigma_offset + index])))
        for channel, index in channel_index.items()
    }
    ppc = _posterior_predictive_rate_error(
        observed=observed,
        fixed_pi=fixed_pi,
        source_accuracy=source_accuracy,
        sources=sources,
    )
    max_rhat = float(np.nanmax(rhat_values))
    min_ess = float(np.nanmin(ess_values))
    sufficient_source_information = all(
        value >= minimum_observations_per_source for value in counts.values()
    )
    diagnostics: dict[str, object] = {
        "chains": chains,
        "warmup_per_chain": warmup,
        "draws_per_chain": draws,
        "total_posterior_draws": chains * draws,
        "fixed_pi": fixed_pi,
        "max_rhat": max_rhat,
        "minimum_ess": min_ess,
        "posterior_predictive_max_source_rate_error": ppc,
        "source_observation_counts": counts,
        "convergence_pass": max_rhat <= rhat_maximum and min_ess >= ess_minimum,
        "posterior_predictive_pass": ppc <= ppc_rate_error_maximum,
        "prior_dominance_screen_pass": sufficient_source_information,
        "eligible_for_gate1": max_rhat <= rhat_maximum
        and min_ess >= ess_minimum
        and ppc <= ppc_rate_error_maximum
        and sufficient_source_information,
        "channel_random_effect_sampled": True,
        "missingness_modeled_separately_from_zero": True,
    }
    retained_indices = np.linspace(0, len(flattened_y) - 1, min(100, len(flattened_y)), dtype=int)
    retained_draws = flattened_y[retained_indices]
    retained_parameters = flattened_parameters[retained_indices]
    posterior_parameter_draws: list[dict[str, Any]] = []
    for values in retained_parameters:
        posterior_parameter_draws.append(
            {
                "source_accuracy": {
                    source: {
                        "sensitivity_mean": float(_expit(np.asarray([values[index]]))[0]),
                        "specificity_mean": float(
                            1.0 - _expit(np.asarray([values[source_count + index]]))[0]
                        ),
                    }
                    for source, index in source_index.items()
                },
                "channel_random_effect_sd": {
                    channel: float(math.exp(values[sigma_offset + index]))
                    for channel, index in channel_index.items()
                },
            }
        )
    return LatentClassResult(
        posterior_mean=np.mean(flattened_y, axis=0).astype(float).tolist(),
        posterior_draws=retained_draws.astype(int).tolist(),
        parameter_draws=posterior_parameter_draws,
        source_accuracy=source_accuracy,
        channel_random_effect_sd=channel_sd,
        diagnostics=diagnostics,
    )


def _run_chain(
    *,
    observed: np.ndarray,
    source_channel: np.ndarray,
    fixed_pi: float,
    warmup: int,
    draws: int,
    alpha_step: float,
    random_effect_step: float,
    priors: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    channel_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n, source_count = observed.shape
    se_a, se_b, sp_a, sp_b = priors
    alpha_one = _logit(se_a / (se_a + se_b)) + rng.normal(0.0, 0.1, source_count)
    false_positive_mean = sp_b / (sp_a + sp_b)
    alpha_zero = _logit(false_positive_mean) + rng.normal(0.0, 0.1, source_count)
    channel_sd = np.full(channel_count, 0.35, dtype=float)
    random_effect = rng.normal(0.0, channel_sd, size=(n, channel_count))
    latent = cast(np.ndarray, rng.random(n) < fixed_pi)
    y_draws = np.empty((draws, n), dtype=np.int8)
    parameter_draws = np.empty((draws, 2 * source_count + channel_count), dtype=float)
    for iteration in range(warmup + draws):
        latent = _sample_latent(
            observed, alpha_one, alpha_zero, random_effect, source_channel, fixed_pi, rng
        )
        for source in range(source_count):
            alpha_one[source] = _update_alpha(
                current=alpha_one[source],
                outcomes=observed[:, source],
                include=latent,
                random_effect=random_effect[:, source_channel[source]],
                prior_a=se_a[source],
                prior_b=se_b[source],
                step=alpha_step,
                rng=rng,
            )
            alpha_zero[source] = _update_alpha(
                current=alpha_zero[source],
                outcomes=observed[:, source],
                include=~latent,
                random_effect=random_effect[:, source_channel[source]],
                prior_a=sp_b[source],
                prior_b=sp_a[source],
                step=alpha_step,
                rng=rng,
            )
        for channel in range(channel_count):
            source_ids = np.flatnonzero(source_channel == channel)
            random_effect[:, channel] = _update_random_effect(
                current=random_effect[:, channel],
                outcomes=observed[:, source_ids],
                latent=latent,
                alpha_one=alpha_one[source_ids],
                alpha_zero=alpha_zero[source_ids],
                sigma=channel_sd[channel],
                step=random_effect_step,
                rng=rng,
            )
            shape = 2.0 + n / 2.0
            scale = 0.25 + 0.5 * float(np.sum(random_effect[:, channel] ** 2))
            precision = rng.gamma(shape, 1.0 / scale)
            channel_sd[channel] = math.sqrt(1.0 / precision)
        if iteration >= warmup:
            index = iteration - warmup
            y_draws[index] = latent.astype(np.int8)
            parameter_draws[index] = np.concatenate([alpha_one, alpha_zero, np.log(channel_sd)])
    return y_draws, parameter_draws


def _sample_latent(
    observed: np.ndarray,
    alpha_one: np.ndarray,
    alpha_zero: np.ndarray,
    random_effect: np.ndarray,
    source_channel: np.ndarray,
    fixed_pi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    log_one = np.full(len(observed), math.log(fixed_pi), dtype=float)
    log_zero = np.full(len(observed), math.log(1.0 - fixed_pi), dtype=float)
    for source in range(observed.shape[1]):
        mask = observed[:, source] >= 0
        values = observed[mask, source].astype(float)
        u = random_effect[mask, source_channel[source]]
        log_one[mask] += _bernoulli_log_likelihood(values, alpha_one[source] + u)
        log_zero[mask] += _bernoulli_log_likelihood(values, alpha_zero[source] + u)
    probability = _expit(log_one - log_zero)
    return rng.random(len(observed)) < probability


def _update_alpha(
    *,
    current: float,
    outcomes: np.ndarray,
    include: np.ndarray,
    random_effect: np.ndarray,
    prior_a: float,
    prior_b: float,
    step: float,
    rng: np.random.Generator,
) -> float:
    mask = include & (outcomes >= 0)
    proposed = current + rng.normal(0.0, step)
    current_log = float(
        np.sum(
            _bernoulli_log_likelihood(outcomes[mask].astype(float), current + random_effect[mask])
        )
    ) + _beta_logit_prior(current, prior_a, prior_b)
    proposed_log = float(
        np.sum(
            _bernoulli_log_likelihood(outcomes[mask].astype(float), proposed + random_effect[mask])
        )
    ) + _beta_logit_prior(proposed, prior_a, prior_b)
    return proposed if math.log(rng.random()) < proposed_log - current_log else current


def _update_random_effect(
    *,
    current: np.ndarray,
    outcomes: np.ndarray,
    latent: np.ndarray,
    alpha_one: np.ndarray,
    alpha_zero: np.ndarray,
    sigma: float,
    step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    proposed = current + rng.normal(0.0, step, len(current))
    current_log = -(current**2) / (2.0 * sigma**2)
    proposed_log = -(proposed**2) / (2.0 * sigma**2)
    for local_source in range(outcomes.shape[1]):
        mask = outcomes[:, local_source] >= 0
        alpha = cast(
            np.ndarray,
            np.where(latent, alpha_one[local_source], alpha_zero[local_source]),
        )
        values = outcomes[mask, local_source].astype(float)
        current_log[mask] += _bernoulli_log_likelihood(values, alpha[mask] + current[mask])
        proposed_log[mask] += _bernoulli_log_likelihood(values, alpha[mask] + proposed[mask])
    accept = np.log(rng.random(len(current))) < proposed_log - current_log
    return np.where(accept, proposed, current)


def _prior_arrays(
    sources: list[str], priors: dict[str, dict[str, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = ("sensitivity_alpha", "sensitivity_beta", "specificity_alpha", "specificity_beta")
    arrays: list[np.ndarray] = []
    for key in keys:
        values = np.asarray([float(priors[source][key]) for source in sources], dtype=float)
        if np.any(values <= 0):
            raise ValueError("L3 beta-prior parameters must be positive")
        arrays.append(values)
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _rhat(chains: np.ndarray) -> np.ndarray:
    chain_count, draws, _ = chains.shape
    if chain_count < 2 or draws < 2:
        return np.full(chains.shape[2], np.inf)
    chain_means = np.mean(chains, axis=1)
    between = draws * np.var(chain_means, axis=0, ddof=1)
    within = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    variance = ((draws - 1) / draws) * within + between / draws
    return np.sqrt(
        np.divide(variance, within, out=np.full_like(variance, np.inf), where=within > 0)
    )


def _ess(chains: np.ndarray) -> np.ndarray:
    chain_count, draws, parameters = chains.shape
    flattened = chains.reshape(chain_count * draws, parameters)
    result = np.empty(parameters, dtype=float)
    for parameter in range(parameters):
        values = flattened[:, parameter] - np.mean(flattened[:, parameter])
        denominator = float(np.dot(values, values))
        if denominator <= 0:
            result[parameter] = 0.0
            continue
        correlation_sum = 0.0
        for lag in range(1, min(100, len(values) - 1)):
            correlation = float(np.dot(values[:-lag], values[lag:]) / denominator)
            if correlation <= 0:
                break
            correlation_sum += correlation
        result[parameter] = len(values) / (1.0 + 2.0 * correlation_sum)
    return result


def _posterior_predictive_rate_error(
    *,
    observed: np.ndarray,
    fixed_pi: float,
    source_accuracy: dict[str, dict[str, float]],
    sources: list[str],
) -> float:
    errors: list[float] = []
    for source_index, source in enumerate(sources):
        mask = observed[:, source_index] >= 0
        if not np.any(mask):
            continue
        actual = float(np.mean(observed[mask, source_index]))
        accuracy = source_accuracy[source]
        expected = fixed_pi * accuracy["sensitivity_mean"] + (1.0 - fixed_pi) * (
            1.0 - accuracy["specificity_mean"]
        )
        errors.append(abs(actual - expected))
    return max(errors, default=math.inf)


def _bernoulli_log_likelihood(outcomes: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return outcomes * (-np.logaddexp(0.0, -logits)) + (1.0 - outcomes) * (
        -np.logaddexp(0.0, logits)
    )


def _beta_logit_prior(logit: float, alpha: float, beta: float) -> float:
    probability = float(_expit(np.asarray(logit)))
    return alpha * math.log(max(probability, 1e-12)) + beta * math.log(
        max(1.0 - probability, 1e-12)
    )


def _expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -35.0, 35.0)))


def _logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _validate_controls(
    *,
    fixed_pi: float,
    chains: int,
    warmup: int,
    draws: int,
    alpha_step: float,
    random_effect_step: float,
) -> None:
    if not 0 < fixed_pi < 1:
        raise ValueError("fixed_pi must be in (0, 1)")
    if chains < 2 or warmup < 1 or draws < 2:
        raise ValueError("L3 requires at least two chains, warmup, and posterior draws")
    if alpha_step <= 0 or random_effect_step <= 0:
        raise ValueError("L3 MCMC proposal steps must be positive")
