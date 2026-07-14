"""Synthetic measurement simulation with deterministic, batch-local randomness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from core.metrics import average_precision, roc_auc
from core.semantic_keys import (
    ESTIMATE,
    MCSE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from labels.service import aggregate_l1, evidence_score_l2, posterior_l3_fixed_pi

_REQUIRED = {
    SCENARIO_ID,
    "sample_size",
    "prevalence",
    "anchor_sensitivity",
    "anchor_false_positive",
    "weak_sensitivity",
    "weak_false_positive",
    "content_signal",
    "tier",
    "fixed_pi",
    "anchor_verification_probability",
    "weak_verification_probability",
    "selective_verification_strength",
    "channel_dependence",
    "horizon_days",
    "detection_delay_mean_days",
    "shift_strength",
    "signal_structure",
}


def validate_scenarios(raw: Sequence[object]) -> list[dict[str, Any]]:
    """Validate only explicitly registered operational scenarios."""
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ValueError(f"simulation scenario {index}: mapping required")
        scenario = cast(dict[str, Any], value)
        missing = sorted(_REQUIRED - set(scenario))
        if missing:
            raise ValueError(f"simulation scenario {index}: missing {missing}")
        scenario_id = scenario[SCENARIO_ID]
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen:
            raise ValueError("simulation scenario_id must be unique and nonempty")
        seen.add(scenario_id)
        if not isinstance(scenario["sample_size"], int) or scenario["sample_size"] < 2:
            raise ValueError(f"scenario={scenario_id}: sample_size must be >= 2")
        for key in (
            "prevalence",
            "anchor_sensitivity",
            "anchor_false_positive",
            "weak_sensitivity",
            "weak_false_positive",
        ):
            number = scenario[key]
            if not isinstance(number, (int, float)) or not 0 <= float(number) <= 1:
                raise ValueError(f"scenario={scenario_id}: {key} must be in [0, 1]")
        if not 0 < float(scenario["prevalence"]) < 1:
            raise ValueError(f"scenario={scenario_id}: prevalence must be in (0, 1)")
        fixed_pi = scenario.get("fixed_pi", scenario["prevalence"])
        if not isinstance(fixed_pi, (int, float)) or not 0 < float(fixed_pi) < 1:
            raise ValueError(f"scenario={scenario_id}: fixed_pi must be in (0, 1)")
        if not isinstance(scenario["content_signal"], (int, float)):
            raise ValueError(f"scenario={scenario_id}: content_signal must be numeric")
        if scenario["tier"] not in {
            "fully_synthetic",
            "semi_synthetic_development_covariates",
        }:
            raise ValueError(f"scenario={scenario_id}: invalid simulation tier")
        if scenario["tier"] == "semi_synthetic_development_covariates":
            feature_ids = scenario.get("semi_synthetic_feature_ids")
            if (
                not isinstance(feature_ids, list)
                or not cast(list[object], feature_ids)
                or any(
                    not isinstance(item, str) or not item
                    for item in cast(list[object], feature_ids)
                )
            ):
                raise ValueError(
                    f"scenario={scenario_id}: semi_synthetic_feature_ids must be nonempty"
                )
        if scenario["signal_structure"] not in {"linear", "nonlinear", "interaction"}:
            raise ValueError(f"scenario={scenario_id}: invalid signal_structure")
        if not isinstance(scenario["shift_strength"], (int, float)):
            raise ValueError(f"scenario={scenario_id}: shift_strength must be numeric")
        for key in (
            "anchor_verification_probability",
            "weak_verification_probability",
            "selective_verification_strength",
            "channel_dependence",
        ):
            number = scenario.get(key, 0.0 if key != "channel_dependence" else 0.0)
            if not isinstance(number, (int, float)) or not 0 <= float(number) <= 1:
                raise ValueError(f"scenario={scenario_id}: {key} must be in [0, 1]")
        for key in ("horizon_days", "detection_delay_mean_days"):
            number = scenario.get(key, 365 if key == "horizon_days" else 30.0)
            if not isinstance(number, (int, float)) or float(number) <= 0:
                raise ValueError(f"scenario={scenario_id}: {key} must be positive")
        scenarios.append(dict(scenario))
    return scenarios


def attach_development_covariate_pools(
    *,
    scenarios: list[dict[str, Any]],
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, Any]],
    year_column: str,
    development_year_maximum: int,
) -> list[dict[str, Any]]:
    """Lock semi-synthetic covariate pools using development rows only."""
    if year_column not in feature_panel.columns:
        raise ValueError("semi-synthetic feature panel is missing the year binding")
    registered = {str(item.get("feature_id")) for item in feature_registry}
    development = feature_panel.loc[
        feature_panel[year_column].astype(int) <= development_year_maximum
    ]
    output: list[dict[str, Any]] = []
    for scenario in scenarios:
        bound = dict(scenario)
        if scenario["tier"] != "semi_synthetic_development_covariates":
            output.append(bound)
            continue
        raw_feature_ids = cast(list[object], scenario["semi_synthetic_feature_ids"])
        feature_ids = [str(item) for item in raw_feature_ids]
        missing_registry = sorted(set(feature_ids) - registered)
        missing_columns = sorted(set(feature_ids) - set(feature_panel.columns))
        if missing_registry or missing_columns:
            raise ValueError(
                f"scenario={scenario[SCENARIO_ID]}: semi-synthetic features unavailable; "
                f"unregistered={missing_registry}, absent={missing_columns}"
            )
        numeric = development[feature_ids].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(how="all")
        if len(numeric) < 2:
            raise ValueError(
                f"scenario={scenario[SCENARIO_ID]}: insufficient development covariate rows"
            )
        medians = numeric.median(axis=0)
        imputed = numeric.fillna(medians)
        standard_deviations = imputed.std(axis=0, ddof=0).replace(0.0, 1.0)
        standardized = (imputed - imputed.mean(axis=0)) / standard_deviations
        pool = standardized.mean(axis=1).to_numpy(dtype=float)
        bound.update(
            {
                "semi_synthetic_content_pool": pool.tolist(),
                "semi_synthetic_pool_rows": len(pool),
                "semi_synthetic_development_year_maximum": development_year_maximum,
                "semi_synthetic_preprocessing": "development_median_zscore_then_row_mean",
                "outer_rows_used_in_pool": 0,
            }
        )
        output.append(bound)
    return output


def run_batch(
    scenario: Mapping[str, Any],
    *,
    method_id: str,
    replications: range,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run a simulation batch; L1 and L2 are imported from production code."""
    rows: list[dict[str, object]] = []
    scenario_id = str(scenario[SCENARIO_ID])
    n = int(scenario["sample_size"])
    for replication_id in replications:
        latent = rng.random(n) < float(scenario["prevalence"])
        shared_uniform = rng.random(n)
        dependence = float(scenario.get("channel_dependence", 0.0))
        anchor = _source_draw(
            latent,
            sensitivity=float(scenario["anchor_sensitivity"]),
            false_positive=float(scenario["anchor_false_positive"]),
            shared_uniform=shared_uniform,
            dependence=dependence,
            rng=rng,
        )
        weak = _source_draw(
            latent,
            sensitivity=float(scenario["weak_sensitivity"]),
            false_positive=float(scenario["weak_false_positive"]),
            shared_uniform=shared_uniform,
            dependence=dependence,
            rng=rng,
        )
        selective = float(scenario.get("selective_verification_strength", 0.0))
        anchor_observed = _verification_draw(
            latent,
            base_probability=float(scenario.get("anchor_verification_probability", 1.0)),
            selective_strength=selective,
            rng=rng,
        )
        weak_observed = _verification_draw(
            latent,
            base_probability=float(scenario.get("weak_verification_probability", 1.0)),
            selective_strength=selective,
            rng=rng,
        )
        delay = rng.exponential(float(scenario.get("detection_delay_mean_days", 30.0)), n)
        mature = delay <= float(scenario.get("horizon_days", 365.0))
        anchor_values = [
            bool(value) if observed and is_mature else None
            for value, observed, is_mature in zip(anchor, anchor_observed, mature, strict=True)
        ]
        weak_values = [
            bool(value) if observed and is_mature else None
            for value, observed, is_mature in zip(weak, weak_observed, mature, strict=True)
        ]
        source_rows = [
            {"anchor": anchor_value, "weak": weak_value}
            for anchor_value, weak_value in zip(anchor_values, weak_values, strict=True)
        ]
        l1_values = [aggregate_l1(value) for value in source_rows]
        l2_values = [evidence_score_l2(value) for value in source_rows]
        fixed_pi = float(scenario.get("fixed_pi", scenario["prevalence"]))
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
        l3 = np.asarray(
            [posterior_l3_fixed_pi(value, accuracy, fixed_pi) for value in source_rows],
            dtype=float,
        )
        l2 = np.asarray(
            [fixed_pi if value is None else float(value) for value in l2_values], dtype=float
        )
        content = _content_signal(scenario, latent, n, rng)
        standardized_content = (content - content.mean()) / max(content.std(), 1e-12)
        method_scores = {
            "observability_only": l2,
            "content_only": 1.0 / (1.0 + np.exp(-standardized_content)),
            "full": 1.0 / (1.0 + np.exp(-(standardized_content + l2))),
            "anchor_pu": np.asarray(
                [fixed_pi if value is None else float(value) for value in anchor_values]
            ),
        }
        if method_id not in method_scores:
            raise ValueError(f"method={method_id}: unsupported simulation method")
        l1_observed = np.asarray([value is not None for value in l1_values], dtype=bool)
        l1_numeric = np.asarray(
            [False if value is None else bool(value) for value in l1_values], dtype=bool
        )
        l2_observed = np.asarray([value is not None for value in l2_values], dtype=bool)
        estimates = {
            "l1_coverage": float(l1_observed.mean()),
            "l1_latent_agreement": float(
                np.asarray(l1_numeric[l1_observed] == latent[l1_observed], dtype=float).mean()
            )
            if l1_observed.any()
            else 0.0,
            "l2_coverage": float(l2_observed.mean()),
            "l2_latent_mae": float(
                np.asarray(np.abs(l2[l2_observed] - latent[l2_observed].astype(float))).mean()
            )
            if l2_observed.any()
            else 0.0,
            "l3_fixed_latent_mae": float(np.abs(l3 - latent.astype(float)).mean()),
            "verification_rate": float(
                np.asarray(anchor_observed | weak_observed, dtype=float).mean()
            ),
            "maturity_rate": float(np.asarray(mature, dtype=float).mean()),
            "realized_prevalence": float(np.asarray(latent, dtype=float).mean()),
        }
        l2_mae = estimates["l2_latent_mae"]
        l3_mae = estimates["l3_fixed_latent_mae"]
        estimates["fixed_pi_ranking_recovery"] = _rank_correlation(l3, latent.astype(float))
        estimates["fixed_pi_misspecification_regret"] = max(0.0, l3_mae - min(l2_mae, 0.5))
        estimates["measurement_selection_stability"] = float(l3_mae <= l2_mae)
        pi_estimate, pi_lower, pi_upper = _hierarchical_pi_estimate(source_rows, accuracy)
        estimates["hierarchical_pi_error"] = pi_estimate - float(scenario["prevalence"])
        estimates["hierarchical_pi_squared_error"] = (
            pi_estimate - float(scenario["prevalence"])
        ) ** 2
        estimates["hierarchical_pi_coverage"] = float(
            pi_lower <= float(scenario["prevalence"]) <= pi_upper
        )
        l3_ap = average_precision(latent.tolist(), l3.tolist())
        l3_auc = roc_auc(latent.tolist(), l3.tolist())
        if l3_ap is not None:
            estimates["l3_fixed_latent_average_precision"] = l3_ap
        if l3_auc is not None:
            estimates["l3_fixed_latent_auc"] = l3_auc
        content_ap = average_precision(latent.tolist(), content.tolist())
        if content_ap is not None:
            estimates["content_average_precision"] = content_ap
        method_ap = average_precision(latent.tolist(), method_scores[method_id].tolist())
        if method_ap is not None:
            estimates["method_latent_average_precision"] = method_ap
        observability_ap = average_precision(latent.tolist(), l2.tolist())
        full_ap = average_precision(latent.tolist(), method_scores["full"].tolist())
        if observability_ap is not None and full_ap is not None:
            mmi = max(0.01, 0.10 * observability_ap)
            estimates["gate2_pass"] = float(full_ap - observability_ap >= mmi)
            estimates["gate3_pass"] = float(
                scenario.get("signal_structure") in {"nonlinear", "interaction"}
                and full_ap - observability_ap >= mmi
            )
        shifted_content = standardized_content + float(scenario.get("shift_strength", 0.0))
        shifted_score = 1.0 / (1.0 + np.exp(-(shifted_content + l2)))
        transfer_ap = average_precision(latent.tolist(), shifted_score.tolist())
        if transfer_ap is not None:
            estimates["transfer_average_precision"] = transfer_ap
        estimates["utility_regret"] = _utility_regret(
            truth=latent,
            scores=method_scores[method_id],
            review_fraction=float(scenario.get("review_fraction", 0.05)),
            review_cost=float(scenario.get("review_cost", 1.0)),
            false_positive_cost=float(scenario.get("false_positive_cost", 1.0)),
        )
        method_auc = roc_auc(latent.tolist(), method_scores[method_id].tolist())
        if method_auc is not None:
            estimates["method_latent_auc"] = method_auc
        for metric_id, estimate in estimates.items():
            rows.append(
                {
                    SCENARIO_ID: scenario_id,
                    METHOD_ID: method_id,
                    REPLICATION_ID: replication_id,
                    METRIC_ID: metric_id,
                    ESTIMATE: float(estimate),
                    MCSE: None,
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


def summarize_mcse(
    batches: Sequence[pd.DataFrame],
    *,
    minimum_replications: int,
    maximum_replications: int,
    pass_fail_mcse_maximum: float,
    l3_minimum_replications: int | None = None,
    l3_maximum_replications: int | None = None,
    l3_pass_fail_mcse_maximum: float | None = None,
    continuous_mcse_fraction: float | None = None,
    minimum_meaningful_improvement: float | None = None,
) -> dict[str, object]:
    """Compute actual Monte Carlo standard errors over completed replications."""
    if not batches:
        return {"status": "SKIPPED", "reason_code": "NO_SIMULATION_BATCHES", "metrics": []}
    if (
        minimum_replications < 1
        or maximum_replications < minimum_replications
        or pass_fail_mcse_maximum <= 0
    ):
        raise ValueError("simulation replication controls are invalid")
    combined = pd.concat(batches, ignore_index=True)
    metrics: list[dict[str, object]] = []
    grouped = combined.groupby([SCENARIO_ID, METHOD_ID, METRIC_ID], sort=True)
    for (scenario_id, method_id, metric_id), frame in grouped:
        count = int(len(frame))
        standard_deviation = float(frame[ESTIMATE].std(ddof=1)) if count > 1 else 0.0
        is_l3 = str(metric_id).startswith(("l3_", "fixed_pi_", "hierarchical_pi_"))
        required_replications = (
            l3_minimum_replications
            if is_l3 and l3_minimum_replications is not None
            else minimum_replications
        )
        replication_cap = (
            l3_maximum_replications
            if is_l3 and l3_maximum_replications is not None
            else maximum_replications
        )
        pass_fail_target = (
            l3_pass_fail_mcse_maximum
            if is_l3 and l3_pass_fail_mcse_maximum is not None
            else pass_fail_mcse_maximum
        )
        binary_metric = str(metric_id).endswith(("_pass", "_coverage", "_stability"))
        mcse_target = float(pass_fail_target)
        if (
            not binary_metric
            and continuous_mcse_fraction is not None
            and minimum_meaningful_improvement is not None
        ):
            mcse_target = min(
                mcse_target,
                continuous_mcse_fraction * minimum_meaningful_improvement,
            )
        actual_mcse = standard_deviation / math.sqrt(count)
        row: dict[str, object] = {
            SCENARIO_ID: str(scenario_id),
            METHOD_ID: str(method_id),
            METRIC_ID: str(metric_id),
            "replications": count,
            "mean": float(frame[ESTIMATE].mean()),
            MCSE: actual_mcse,
            "mcse_target": mcse_target,
            "replication_tier": "l3" if is_l3 else "core",
            "minimum_replications_met": count >= required_replications,
            "mcse_target_met": actual_mcse <= mcse_target,
            "maximum_replications_reached": count >= replication_cap,
        }
        if str(metric_id) == "hierarchical_pi_squared_error":
            row["rmse"] = math.sqrt(max(0.0, float(frame[ESTIMATE].mean())))
        metrics.append(row)
    precision_met = bool(metrics) and all(
        item["minimum_replications_met"] is True and item["mcse_target_met"] is True
        for item in metrics
    )
    maximum_reached = bool(metrics) and all(
        item["mcse_target_met"] is True or item["maximum_replications_reached"] is True
        for item in metrics
    )
    status = "PASS" if precision_met else "MAXIMUM_REACHED" if maximum_reached else "CONTINUE"
    return {
        "status": status,
        "reason_code": None
        if precision_met
        else "MCSE_TARGET_NOT_MET_AT_CAP"
        if maximum_reached
        else "ADDITIONAL_REPLICATIONS_REQUIRED",
        "minimum_replications": minimum_replications,
        "maximum_replications": maximum_replications,
        "pass_fail_mcse_maximum": pass_fail_mcse_maximum,
        "l3_minimum_replications": l3_minimum_replications,
        "l3_maximum_replications": l3_maximum_replications,
        "l3_pass_fail_mcse_maximum": l3_pass_fail_mcse_maximum,
        "continuous_mcse_fraction": continuous_mcse_fraction,
        "minimum_meaningful_improvement": minimum_meaningful_improvement,
        "precision_target_met": precision_met,
        "metrics": metrics,
    }


def _rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=float)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=float)
    if np.std(first_rank) <= 0 or np.std(second_rank) <= 0:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _hierarchical_pi_estimate(
    rows: list[dict[str, bool | None]],
    accuracy: dict[str, tuple[float, float]],
) -> tuple[float, float, float]:
    grid = np.linspace(0.001, 0.5, 500)
    log_likelihood = np.zeros(len(grid), dtype=float)
    for row in rows:
        likelihood_one = np.ones(len(grid), dtype=float)
        likelihood_zero = np.ones(len(grid), dtype=float)
        for source, value in row.items():
            if value is None:
                continue
            sensitivity, specificity = accuracy[source]
            likelihood_one *= sensitivity if value else 1.0 - sensitivity
            likelihood_zero *= 1.0 - specificity if value else specificity
        mixture = grid * likelihood_one + (1.0 - grid) * likelihood_zero
        log_likelihood += np.log(np.clip(mixture, 1e-12, None))
    posterior = cast(np.ndarray, np.exp(log_likelihood - np.max(log_likelihood)))
    posterior /= np.sum(posterior)
    estimate = sum(
        float(grid_value) * float(probability)
        for grid_value, probability in zip(grid, posterior, strict=True)
    )
    cumulative = cast(np.ndarray, np.cumsum(posterior))
    lower = float(grid[min(np.searchsorted(cumulative, 0.025), len(grid) - 1)])
    upper = float(grid[min(np.searchsorted(cumulative, 0.975), len(grid) - 1)])
    return estimate, lower, upper


def _utility_regret(
    *,
    truth: np.ndarray,
    scores: np.ndarray,
    review_fraction: float,
    review_cost: float,
    false_positive_cost: float,
) -> float:
    count = max(1, math.ceil(len(truth) * review_fraction))

    def utility(order: np.ndarray) -> float:
        reviewed = truth[order[:count]]
        return float(
            np.sum(reviewed) - review_cost * count - false_positive_cost * np.sum(~reviewed)
        )

    oracle_order = np.argsort(-truth.astype(float), kind="stable")
    method_order = np.argsort(-scores, kind="stable")
    return max(0.0, utility(oracle_order) - utility(method_order))


def _source_draw(
    latent: np.ndarray,
    *,
    sensitivity: float,
    false_positive: float,
    shared_uniform: np.ndarray,
    dependence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.where(latent, sensitivity, false_positive)
    independent = rng.random(len(latent))
    use_shared = rng.random(len(latent)) < dependence
    draws = np.where(use_shared, shared_uniform, independent)
    return draws < probability


def _verification_draw(
    latent: np.ndarray,
    *,
    base_probability: float,
    selective_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.clip(base_probability + selective_strength * latent.astype(float), 0.0, 1.0)
    return rng.random(len(latent)) < probability


def _content_signal(
    scenario: Mapping[str, Any],
    latent: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    strength = float(scenario["content_signal"])
    if scenario.get("tier") == "semi_synthetic_development_covariates":
        raw_pool = scenario.get("semi_synthetic_content_pool")
        if not isinstance(raw_pool, list) or len(cast(list[object], raw_pool)) < 2:
            raise ValueError("semi-synthetic scenario requires a locked development pool")
        pool = np.asarray(cast(list[object], raw_pool), dtype=float)
        base = pool[rng.integers(0, len(pool), size=sample_size)]
    else:
        base = rng.normal(0.0, 1.0, sample_size)
    structure = str(scenario.get("signal_structure", "linear"))
    latent_float = latent.astype(float)
    if structure == "linear":
        effect = strength * latent_float
    elif structure == "nonlinear":
        effect = strength * latent_float * np.tanh(base)
    elif structure == "interaction":
        effect = strength * latent_float * (base > np.median(base)).astype(float)
    else:
        raise ValueError(f"unsupported signal structure: {structure}")
    return base + effect
