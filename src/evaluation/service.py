"""Evaluate frozen predictions after the explicit outer-open checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.metrics import average_precision, brier_score, precision_at_fraction, roc_auc
from core.semantic_keys import (
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    SCENARIO_ID,
    TARGET_ID,
    TARGET_VALUE,
)
from labels.latent_class import predict_fixed_pi_latent_probability


@dataclass(frozen=True)
class EvaluationResult:
    calibration: dict[str, object]
    metrics: dict[str, object]
    bootstrap: list[dict[str, object]]
    utility: list[dict[str, object]]


def evaluate_outer_fold(
    *,
    oof_predictions: pd.DataFrame,
    outer_predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    outer_year: int,
    review_fraction: float,
    utility_scenarios: list[dict[str, object]],
    bootstrap_replications: int,
    confidence_level: float,
    columns: dict[str, str],
    rng: np.random.Generator,
    target_id: str = "L1",
    oof_training_targets: list[dict[str, object]] | None = None,
    latent_risk_scenarios: dict[str, list[dict[str, float]]] | None = None,
) -> EvaluationResult:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    prediction = columns[PREDICTION]
    outcome = columns[OUTCOME]
    target = columns[TARGET_ID]
    if target not in outcomes.columns:
        if target_id != "L1":
            raise ValueError("target-aware sealed outcomes are required")
        observed = outcomes.loc[:, [firm, year, outcome]]
    else:
        observed = outcomes.loc[outcomes[target].eq(target_id), [firm, year, outcome]]
    if observed.duplicated([firm, year]).any():
        raise ValueError(f"target={target_id}: sealed outcomes are not unique by firm-year")
    development = oof_predictions.merge(observed, on=[firm, year], how="inner", validate="m:1")
    outer = outer_predictions.merge(observed, on=[firm, year], how="inner", validate="m:1")
    outer = outer.loc[outer[year] == outer_year].copy()
    calibration_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    calibrated: dict[str, np.ndarray] = {}
    target_frame = pd.DataFrame(oof_training_targets or [])
    for learner_id, frame in outer.groupby(learner, sort=True):
        development_frame = development.loc[development[learner] == learner_id]
        intercept, slope, status = _fit_platt(
            development_frame[prediction], development_frame[outcome]
        )
        probabilities = _apply_platt(frame[prediction].to_numpy(dtype=float), intercept, slope)
        calibrated[str(learner_id)] = probabilities
        calibration_rows.append(
            {
                LEARNER_ID: str(learner_id),
                "status": status,
                "intercept": intercept,
                "slope": slope,
                "fit_scope": "pooled_cross_fitted_development_predictions",
                "in_sample_refit_predictions_used": False,
                "development_rows": len(development_frame),
            }
        )
        truth = frame[outcome].astype(bool).tolist()
        scores = probabilities.tolist()
        target_column = columns.get(TARGET_ID)
        target_ids = (
            sorted(str(value) for value in frame[target_column].unique())
            if isinstance(target_column, str) and target_column in frame.columns
            else ["L1"]
        )
        target_id = target_ids[0] if len(target_ids) == 1 else "mixed"
        fit_metrics = _track_fit_metrics(
            development_frame,
            target_frame,
            learner_id=str(learner_id),
            firm=firm,
            year=year,
            learner=learner,
            prediction=prediction,
        )
        metric_rows.append(
            {
                LEARNER_ID: str(learner_id),
                TARGET_ID: target_id,
                OUTER_FOLD: str(outer_year),
                "rows": len(frame),
                "positives": int(frame[outcome].sum()),
                "average_precision": average_precision(truth, scores),
                "roc_auc": roc_auc(truth, scores),
                "brier_score": brier_score(truth, scores) if truth else None,
                "precision_at_primary_budget": precision_at_fraction(truth, scores, review_fraction)
                if truth
                else None,
                **fit_metrics,
            }
        )
    comparisons = _comparisons(outer, calibrated, learner, outcome)
    bootstrap = _firm_bootstrap(
        outer,
        calibrated,
        learner=learner,
        firm=firm,
        outcome=outcome,
        replications=bootstrap_replications,
        confidence_level=confidence_level,
        rng=rng,
    )
    utility = _utility(
        outer=outer,
        calibrated=calibrated,
        scenarios=utility_scenarios,
        default_review_fraction=review_fraction,
        learner=learner,
        firm=firm,
        outcome=outcome,
        bootstrap_replications=bootstrap_replications,
        confidence_level=confidence_level,
        rng=rng,
        latent_risk_scenarios=latent_risk_scenarios or {},
    )
    return EvaluationResult(
        calibration={"status": "PASS", "models": calibration_rows},
        metrics={
            "status": "PASS",
            OUTER_FOLD: str(outer_year),
            "models": metric_rows,
            "comparisons": comparisons,
        },
        bootstrap=bootstrap,
        utility=utility,
    )


def _fit_platt(scores: pd.Series, outcomes: pd.Series) -> tuple[float, float, str]:
    valid = scores.notna() & outcomes.notna()
    scores = scores.loc[valid]
    outcomes = outcomes.loc[valid].astype(int)
    if scores.empty or outcomes.nunique() < 2:
        return 0.0, 1.0, "IDENTITY_INSUFFICIENT_CLASSES"
    logits = _logit(scores.to_numpy(dtype=float)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    cast(Any, model).fit(logits, outcomes.to_numpy())
    intercept = np.asarray(cast(Any, model).intercept_, dtype=float)
    coefficient = np.asarray(cast(Any, model).coef_, dtype=float)
    return float(intercept[0]), float(coefficient[0, 0]), "PLATT"


def _apply_platt(scores: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    linear = intercept + slope * _logit(scores)
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _comparisons(
    outer: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
    learner: str,
    outcome: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    by_id = {str(value): frame for value, frame in outer.groupby(learner, sort=True)}
    for full_id, full_frame in by_id.items():
        if not full_id.endswith(":full"):
            continue
        reference_id = full_id.removesuffix(":full") + ":observability_only"
        reference = by_id.get(reference_id)
        if reference is None or len(reference) != len(full_frame):
            continue
        truth = full_frame[outcome].astype(bool).tolist()
        full_ap = average_precision(truth, calibrated[full_id].tolist())
        reference_ap = average_precision(truth, calibrated[reference_id].tolist())
        results.append(
            {
                "comparison_id": f"{full_id}_vs_{reference_id}",
                "candidate": full_id,
                "reference": reference_id,
                "delta_average_precision": None
                if full_ap is None or reference_ap is None
                else full_ap - reference_ap,
            }
        )
    return results


def _firm_bootstrap(
    outer: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
    *,
    learner: str,
    firm: str,
    outcome: str,
    replications: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    if replications < 1 or not 0 < confidence_level < 1:
        raise ValueError("bootstrap configuration is invalid")
    working = outer.copy()
    working["_calibrated"] = np.nan
    for model_id, frame in outer.groupby(learner, sort=True):
        working.loc[frame.index, "_calibrated"] = calibrated[str(model_id)]
    results: list[dict[str, object]] = []
    models = sorted(str(value) for value in outer[learner].unique())
    for candidate in (value for value in models if value.endswith(":full")):
        reference = candidate.removesuffix(":full") + ":observability_only"
        candidate_frame = working.loc[working[learner] == candidate].sort_values(firm)
        reference_frame = working.loc[working[learner] == reference].sort_values(firm)
        if candidate_frame.empty or len(candidate_frame) != len(reference_frame):
            continue
        firms = candidate_frame[firm].astype(str).to_numpy()
        values: list[float] = []
        for _ in range(replications):
            sampled = rng.integers(0, len(firms), size=len(firms))
            truth = candidate_frame[outcome].astype(bool).to_numpy()[sampled].tolist()
            candidate_ap = average_precision(
                truth, candidate_frame["_calibrated"].to_numpy()[sampled].tolist()
            )
            reference_ap = average_precision(
                truth, reference_frame["_calibrated"].to_numpy()[sampled].tolist()
            )
            if candidate_ap is not None and reference_ap is not None:
                values.append(candidate_ap - reference_ap)
        alpha = (1.0 - confidence_level) / 2.0
        results.append(
            {
                "candidate": candidate,
                "reference": reference,
                "unit": "firm",
                "requested_replications": replications,
                "valid_replications": len(values),
                "delta_ap_mean": float(np.mean(values)) if values else None,
                "interval_lower": float(np.quantile(values, alpha)) if values else None,
                "interval_upper": float(np.quantile(values, 1.0 - alpha)) if values else None,
                "delta_ap_samples": values,
            }
        )
    return results


def _utility(
    *,
    outer: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
    scenarios: list[dict[str, object]],
    default_review_fraction: float,
    learner: str,
    firm: str,
    outcome: str,
    bootstrap_replications: int,
    confidence_level: float,
    rng: np.random.Generator,
    latent_risk_scenarios: dict[str, list[dict[str, float]]],
) -> list[dict[str, object]]:
    if not scenarios:
        return [
            {
                "status": "SKIPPED",
                "reason_code": "NO_OPERATIONAL_UTILITY_SCENARIOS",
                "review_fraction": default_review_fraction,
                "descriptive_yield": [
                    {
                        LEARNER_ID: str(model_id),
                        "precision": precision_at_fraction(
                            frame[outcome].astype(bool).tolist(),
                            calibrated[str(model_id)].tolist(),
                            default_review_fraction,
                        ),
                    }
                    for model_id, frame in outer.groupby(learner, sort=True)
                ],
            }
        ]
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        required = {
            SCENARIO_ID,
            "true_positive_benefit",
            "review_cost",
            "additional_false_positive_cost",
            "false_negative_cost",
            "measurement_fixed_pi",
        }
        if not required.issubset(scenario):
            raise ValueError("utility scenario is incomplete")
        scenario_id = scenario.get(SCENARIO_ID)
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("utility scenario_id must be a nonempty string")
        for component in required - {SCENARIO_ID}:
            if _scenario_number(scenario, component) < 0:
                raise ValueError(f"utility {component} must be nonnegative")
        review_fraction_raw = scenario.get("review_fraction", default_review_fraction)
        if not isinstance(review_fraction_raw, (int, float)) or isinstance(
            review_fraction_raw, bool
        ):
            raise ValueError("utility review_fraction must be numeric")
        review_fraction = float(review_fraction_raw)
        review_count_raw = scenario.get("review_count")
        if review_count_raw is not None and (
            not isinstance(review_count_raw, int) or isinstance(review_count_raw, bool)
        ):
            raise ValueError("utility review_count must be an integer")
        review_count = review_count_raw
        if not 0 < review_fraction <= 1 or (review_count is not None and review_count < 1):
            raise ValueError("utility review budget is invalid")
        model_frames: dict[str, pd.DataFrame] = {}
        point_results: dict[str, dict[str, float | int]] = {}
        risk_draws = latent_risk_scenarios.get(scenario_id)
        if not risk_draws:
            results.append(
                {
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason_code": "LATENT_RISK_SCENARIO_UNAVAILABLE",
                    SCENARIO_ID: scenario_id,
                    "estimand": "scenario_based_latent_utility",
                }
            )
            continue
        outer_firms = set(outer[firm].astype(str))
        missing_firms: set[str] = set()
        for risk_by_firm in risk_draws:
            missing_firms.update(outer_firms - set(risk_by_firm))
        if missing_firms:
            results.append(
                {
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason_code": "LATENT_RISK_MISSING_FOR_OUTER_FIRMS",
                    SCENARIO_ID: scenario_id,
                    "missing_firms": sorted(missing_firms),
                    "estimand": "scenario_based_latent_utility",
                }
            )
            continue
        risk_by_firm = {
            firm_id: float(np.mean([draw[firm_id] for draw in risk_draws]))
            for firm_id in sorted(outer_firms)
        }
        for model_id, raw_frame in outer.groupby(learner, sort=True):
            model_key = str(model_id)
            frame = raw_frame.sort_values(firm).copy()
            score_by_index = pd.Series(calibrated[model_key], index=raw_frame.index)
            frame["_utility_score"] = score_by_index.loc[frame.index].to_numpy(dtype=float)
            frame["_latent_risk"] = frame[firm].astype(str).map(risk_by_firm)
            model_frames[model_key] = frame
            point_results[model_key] = _utility_point(
                truth_probability=frame["_latent_risk"].to_numpy(dtype=float),
                scores=frame["_utility_score"].to_numpy(dtype=float),
                scenario=scenario,
                review_fraction=review_fraction,
                review_count=review_count,
            )
        alpha = (1.0 - confidence_level) / 2.0
        for model_id, frame in model_frames.items():
            point = point_results[model_id]
            net_samples: list[float] = []
            incremental_samples: list[float] = []
            reference_id = (
                model_id.removesuffix(":full") + ":observability_only"
                if model_id.endswith(":full")
                else None
            )
            reference_frame = model_frames.get(reference_id) if reference_id is not None else None
            paired_reference = bool(
                reference_frame is not None
                and frame[firm].astype(str).tolist() == reference_frame[firm].astype(str).tolist()
                and frame[outcome].astype(float).tolist()
                == reference_frame[outcome].astype(float).tolist()
            )
            for _ in range(bootstrap_replications):
                sampled = rng.integers(0, len(frame), size=len(frame))
                risk_draw = risk_draws[int(rng.integers(0, len(risk_draws)))]
                sampled_point = _utility_point(
                    truth_probability=frame[firm]
                    .astype(str)
                    .map(risk_draw)
                    .to_numpy(dtype=float)[sampled],
                    scores=frame["_utility_score"].to_numpy(dtype=float)[sampled],
                    scenario=scenario,
                    review_fraction=review_fraction,
                    review_count=review_count,
                )
                net_samples.append(float(sampled_point["net_utility"]))
                if paired_reference and reference_frame is not None:
                    reference_point = _utility_point(
                        truth_probability=reference_frame[firm]
                        .astype(str)
                        .map(risk_draw)
                        .to_numpy(dtype=float)[sampled],
                        scores=reference_frame["_utility_score"].to_numpy(dtype=float)[sampled],
                        scenario=scenario,
                        review_fraction=review_fraction,
                        review_count=review_count,
                    )
                    incremental_samples.append(
                        float(sampled_point["net_utility"]) - float(reference_point["net_utility"])
                    )
            reference_point = (
                point_results.get(reference_id)
                if reference_id is not None and paired_reference
                else None
            )
            incremental = (
                float(point["net_utility"]) - float(reference_point["net_utility"])
                if reference_point is not None
                else None
            )
            results.append(
                {
                    "status": "PASS",
                    SCENARIO_ID: scenario_id,
                    LEARNER_ID: model_id,
                    "estimand": "scenario_based_latent_utility",
                    "latent_risk_source": "frozen_fold_local_fixed_pi_l3",
                    "review_fraction": review_fraction,
                    "review_count_override": review_count,
                    **scenario,
                    **point,
                    "reference_model_id": reference_id,
                    "incremental_utility": incremental,
                    "utility_uncertainty": {
                        "unit": "firm",
                        "method": "firm_bootstrap_x_l3_posterior_parameter_draws",
                        "posterior_parameter_draw_count": len(risk_draws),
                        "replications": bootstrap_replications,
                        "confidence_level": confidence_level,
                        "net_utility_interval": _interval(net_samples, alpha),
                        "incremental_utility_interval": _interval(incremental_samples, alpha),
                    },
                }
            )
    return results


def _utility_point(
    *,
    truth_probability: np.ndarray,
    scores: np.ndarray,
    scenario: dict[str, object],
    review_fraction: float,
    review_count: int | None,
) -> dict[str, float | int]:
    count = (
        min(len(scores), review_count)
        if review_count is not None
        else min(len(scores), max(1, int(np.ceil(len(scores) * review_fraction))))
    )
    order = np.argsort(-scores, kind="stable")
    reviewed = order[:count]
    not_reviewed = order[count:]
    expected_true_positives = float(np.sum(truth_probability[reviewed]))
    expected_false_positives = float(np.sum(1.0 - truth_probability[reviewed]))
    expected_false_negatives = float(np.sum(truth_probability[not_reviewed]))
    review_cost_total = _scenario_number(scenario, "review_cost") * count
    false_positive_cost_total = (
        _scenario_number(scenario, "additional_false_positive_cost") * expected_false_positives
    )
    false_negative_cost_total = (
        _scenario_number(scenario, "false_negative_cost") * expected_false_negatives
    )
    true_positive_benefit_total = (
        _scenario_number(scenario, "true_positive_benefit") * expected_true_positives
    )
    return {
        "reviewed_cases": count,
        "expected_true_positives": expected_true_positives,
        "expected_false_positives": expected_false_positives,
        "expected_false_negatives": expected_false_negatives,
        "true_positives": expected_true_positives,
        "false_positives": expected_false_positives,
        "false_negatives": expected_false_negatives,
        "review_cost_total": review_cost_total,
        "additional_false_positive_cost_total": false_positive_cost_total,
        "false_negative_cost_total": false_negative_cost_total,
        "true_positive_benefit_total": true_positive_benefit_total,
        "net_utility": true_positive_benefit_total
        - false_positive_cost_total
        - false_negative_cost_total
        - review_cost_total,
    }


def _interval(values: list[float], alpha: float) -> list[float] | None:
    if not values:
        return None
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def _scenario_number(scenario: dict[str, object], key: str) -> float:
    value = scenario.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"utility {key} must be numeric")
    return float(value)


def build_latent_risk_scenarios(
    *,
    matrices: dict[str, Any],
    channel_selection: dict[str, Any],
    outer_year: int,
    scenarios: list[dict[str, object]],
) -> dict[str, list[dict[str, float]]]:
    """Apply frozen development-only L3 scenarios after the outer-open checkpoint."""
    raw_sources: object = matrices.get("expected_sources")
    raw_rows: object = matrices.get("rows")
    raw_fits: object = channel_selection.get("l3_scenario_results")
    if not isinstance(raw_sources, dict) or not isinstance(raw_rows, list):
        raise ValueError("latent utility requires source-channel matrices")
    if not isinstance(raw_fits, list):
        return {}
    source_channels: dict[str, str] = {}
    for source, channel in cast(dict[str, object], raw_sources).items():
        if not isinstance(channel, str):
            raise ValueError("latent utility source-channel binding must be a string")
        source_channels[source] = channel
    outer_rows: list[dict[str, Any]] = []
    for raw in cast(list[object], raw_rows):
        if not isinstance(raw, dict):
            continue
        row = cast(dict[str, Any], raw)
        if int(row[FISCAL_YEAR]) == outer_year and row.get(MATURE) is True:
            outer_rows.append(row)
    if not outer_rows:
        return {}
    fit_rows = [
        cast(dict[str, Any], raw) for raw in cast(list[object], raw_fits) if isinstance(raw, dict)
    ]
    result: dict[str, list[dict[str, float]]] = {}
    for scenario in scenarios:
        scenario_id = scenario.get(SCENARIO_ID)
        raw_fixed_pi = scenario.get("measurement_fixed_pi")
        if not isinstance(scenario_id, str) or not isinstance(raw_fixed_pi, (int, float)):
            continue
        fixed_pi = float(raw_fixed_pi)
        match = next(
            (
                item
                for item in fit_rows
                if isinstance(item.get("fixed_pi"), (int, float))
                and math.isclose(float(item["fixed_pi"]), fixed_pi, rel_tol=0.0, abs_tol=1e-12)
                and item.get(ELIGIBLE) is True
            ),
            None,
        )
        if match is None:
            continue
        raw_accuracy = match.get("source_accuracy")
        raw_channel_sd = match.get("channel_random_effect_sd")
        if not isinstance(raw_accuracy, dict) or not isinstance(raw_channel_sd, dict):
            continue
        raw_parameter_draws = match.get("posterior_parameter_draws")
        parameter_draws: list[object]
        if isinstance(raw_parameter_draws, list) and raw_parameter_draws:
            parameter_draws = cast(list[object], raw_parameter_draws)
        else:
            fallback_draw: dict[str, object] = {
                "source_accuracy": raw_accuracy,
                "channel_random_effect_sd": raw_channel_sd,
            }
            parameter_draws = [fallback_draw]
        scenario_risk_draws: list[dict[str, float]] = []
        for raw_draw in parameter_draws:
            if not isinstance(raw_draw, dict):
                raise ValueError("latent utility posterior parameter draw must be an object")
            source_accuracy, channel_sd = _latent_parameter_draw(cast(dict[str, object], raw_draw))
            probabilities = predict_fixed_pi_latent_probability(
                rows=outer_rows,
                source_channels=source_channels,
                source_accuracy=source_accuracy,
                channel_random_effect_sd=channel_sd,
                fixed_pi=fixed_pi,
            )
            risks: dict[str, float] = {}
            for row, probability in zip(outer_rows, probabilities, strict=True):
                firm_id = str(row[FIRM_ID])
                if firm_id in risks:
                    raise ValueError("latent utility requires unique outer firm-year rows")
                risks[firm_id] = probability
            scenario_risk_draws.append(risks)
        result[scenario_id] = scenario_risk_draws
    return result


def _latent_parameter_draw(
    draw: dict[str, object],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    raw_accuracy = draw.get("source_accuracy")
    raw_channel_sd = draw.get("channel_random_effect_sd")
    if not isinstance(raw_accuracy, dict) or not isinstance(raw_channel_sd, dict):
        raise ValueError("latent utility parameter draw is incomplete")
    source_accuracy: dict[str, dict[str, float]] = {}
    for source, raw_values in cast(dict[str, object], raw_accuracy).items():
        if not isinstance(raw_values, dict):
            raise ValueError("latent utility source accuracy draw must be an object")
        values = cast(dict[str, object], raw_values)
        parsed: dict[str, float] = {}
        for key in ("sensitivity_mean", "specificity_mean"):
            value = values.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"latent utility {key} must be numeric")
            parsed[key] = float(value)
        source_accuracy[source] = parsed
    channel_sd: dict[str, float] = {}
    for key, value in cast(dict[str, object], raw_channel_sd).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("latent utility channel SD must be numeric")
        channel_sd[key] = float(value)
    return source_accuracy, channel_sd


def _track_fit_metrics(
    development: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    learner_id: str,
    firm: str,
    year: str,
    learner: str,
    prediction: str,
) -> dict[str, object]:
    required = {FIRM_ID, FISCAL_YEAR, LEARNER_ID, TARGET_VALUE}
    if targets.empty or not required.issubset(targets.columns):
        return {
            "soft_log_loss": None,
            "spearman_target_concordance": None,
            "pairwise_concordance": None,
            "fit_metric_status": "UNAVAILABLE",
        }
    selected = targets.loc[targets[LEARNER_ID].astype(str) == learner_id].rename(
        columns={FIRM_ID: firm, FISCAL_YEAR: year, TARGET_VALUE: "_soft_target"}
    )
    merged = development.loc[:, [firm, year, learner, prediction]].merge(
        selected.loc[:, [firm, year, "_soft_target"]],
        on=[firm, year],
        how="inner",
        validate="1:1",
    )
    if merged.empty:
        return {
            "soft_log_loss": None,
            "spearman_target_concordance": None,
            "pairwise_concordance": None,
            "fit_metric_status": "UNAVAILABLE",
        }
    soft = merged["_soft_target"].to_numpy(dtype=float)
    score = np.clip(merged[prediction].to_numpy(dtype=float), 1e-9, 1.0 - 1e-9)
    loss = float(np.mean(-(soft * np.log(score) + (1.0 - soft) * np.log(1.0 - score))))
    soft_rank = merged["_soft_target"].rank(method="average").to_numpy(dtype=float)
    score_rank = merged[prediction].rank(method="average").to_numpy(dtype=float)
    spearman = (
        float(np.corrcoef(soft_rank, score_rank)[0, 1])
        if np.std(soft_rank) > 0 and np.std(score_rank) > 0
        else None
    )
    return {
        "soft_log_loss": loss,
        "spearman_target_concordance": spearman,
        "pairwise_concordance": _pairwise_concordance(soft, score),
        "fit_metric_status": "PASS",
    }


def _pairwise_concordance(target: np.ndarray, score: np.ndarray) -> float | None:
    order = np.argsort(target, kind="stable")
    target = target[order]
    score = score[order]
    comparable = 0
    concordant = 0.0
    for right in range(1, len(target)):
        mask = target[:right] < target[right]
        if not np.any(mask):
            continue
        differences = score[right] - score[:right][mask]
        comparable += len(differences)
        concordant += float(np.count_nonzero(differences > 0)) + 0.5 * float(
            np.count_nonzero(differences == 0)
        )
    return concordant / comparable if comparable else None
