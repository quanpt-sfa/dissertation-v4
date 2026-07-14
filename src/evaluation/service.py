"""Evaluate frozen predictions after the explicit outer-open checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.metrics import average_precision, brier_score, precision_at_fraction, roc_auc
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    SCENARIO_ID,
)


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
) -> EvaluationResult:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    prediction = columns[PREDICTION]
    outcome = columns[OUTCOME]
    observed = outcomes.loc[:, [firm, year, outcome]]
    development = oof_predictions.merge(observed, on=[firm, year], how="inner", validate="m:1")
    outer = outer_predictions.merge(observed, on=[firm, year], how="inner", validate="m:1")
    outer = outer.loc[outer[year] == outer_year].copy()
    calibration_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    calibrated: dict[str, np.ndarray] = {}
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
        metric_rows.append(
            {
                LEARNER_ID: str(learner_id),
                OUTER_FOLD: str(outer_year),
                "rows": len(frame),
                "positives": int(frame[outcome].sum()),
                "average_precision": average_precision(truth, scores),
                "roc_auc": roc_auc(truth, scores),
                "brier_score": brier_score(truth, scores) if truth else None,
                "precision_at_primary_budget": precision_at_fraction(truth, scores, review_fraction)
                if truth
                else None,
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
    utility = _utility(metric_rows, utility_scenarios, review_fraction)
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
    metrics: list[dict[str, object]],
    scenarios: list[dict[str, object]],
    review_fraction: float,
) -> list[dict[str, object]]:
    if not scenarios:
        return [
            {
                "status": "SKIPPED",
                "reason_code": "NO_OPERATIONAL_UTILITY_SCENARIOS",
                "review_fraction": review_fraction,
                "descriptive_yield": [
                    {
                        LEARNER_ID: item[LEARNER_ID],
                        "precision": item["precision_at_primary_budget"],
                    }
                    for item in metrics
                ],
            }
        ]
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        if not {SCENARIO_ID, "review_cost", "false_positive_cost"}.issubset(scenario):
            raise ValueError("utility scenario is incomplete")
        results.append({"status": "REGISTERED", **scenario})
    return results
