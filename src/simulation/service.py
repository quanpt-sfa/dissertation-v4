"""Synthetic measurement, learner, and cost-sensitive simulation.

Latent truth is used only to generate the DGP and evaluate methods. Predictive
training, verification adjustment, threshold selection, and cost-sensitive
weights use feasible information only.
"""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager, nullcontext
import math
from collections.abc import Mapping, Sequence
from typing import Any, Callable, cast

from simulation.scenario_contract import validate_scenario_target_identity

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from core.metrics import average_precision, roc_auc
from core.semantic_keys import (
    ESTIMATE,
    MCSE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from labels.service import aggregate_l1, evidence_score_l2
from simulation.method_contract import (
    IMBALANCE_TREATMENT_ID,
    LABEL_STRATEGY_ID,
    LEARNER_ID,
    METHOD_FAMILY,
    TRAINING_COST_REGIME_ID,
    budget_cost_metric_id,
    cost_regime_by_id,
    evaluation_cost_regimes,
    method_by_id,
    review_budgets,
    threshold_cost_metric_id,
)

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
        if not isinstance(scenario["sample_size"], int) or scenario["sample_size"] < 50:
            raise ValueError(f"scenario={scenario_id}: sample_size must be >= 50")
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
                or any(not isinstance(item, str) or not item for item in feature_ids)
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
            number = scenario.get(key, 0.0)
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
        feature_ids = [str(item) for item in scenario["semi_synthetic_feature_ids"]]
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
        bound.update(
            {
                "semi_synthetic_covariate_pool": standardized.to_numpy(dtype=float).tolist(),
                "semi_synthetic_feature_count": len(feature_ids),
                "semi_synthetic_pool_rows": len(standardized),
                "semi_synthetic_development_year_maximum": development_year_maximum,
                "semi_synthetic_preprocessing": "development_median_zscore",
                "outer_rows_used_in_pool": 0,
            }
        )
        output.append(bound)
    return output



class DiagnosticCollector:
    def __init__(self) -> None:
        self.fit_failures: dict[str, int] = {}
        self.resampling_failures: dict[str, int] = {}
        self.model_warnings: dict[str, int] = {}
        self.affected_replication_ids: set[int] = set()

    def record_fit_failure(self, error_name: str, replication_id: int) -> None:
        self.fit_failures[error_name] = self.fit_failures.get(error_name, 0) + 1
        self.affected_replication_ids.add(replication_id)

    def record_resampling_failure(self, error_name: str, replication_id: int) -> None:
        self.resampling_failures[error_name] = self.resampling_failures.get(error_name, 0) + 1
        self.affected_replication_ids.add(replication_id)

    def record_model_warning(self, warning_name: str, replication_id: int) -> None:
        self.model_warnings[warning_name] = self.model_warnings.get(warning_name, 0) + 1
        self.affected_replication_ids.add(replication_id)


_thread_local = threading.local()


@contextmanager
def diagnostic_capture(collector: DiagnosticCollector, rep_id: int):
    old_collector = getattr(_thread_local, "collector", None)
    old_rep_id = getattr(_thread_local, "_diag_rep_id", None)
    _thread_local.collector = collector
    _thread_local._diag_rep_id = rep_id
    try:
        yield
    finally:
        _thread_local.collector = old_collector
        _thread_local._diag_rep_id = old_rep_id


def record_fit_failure(error_name: str) -> None:
    collector = getattr(_thread_local, "collector", None)
    rep_id = getattr(_thread_local, "_diag_rep_id", None)
    if collector is not None and rep_id is not None:
        collector.record_fit_failure(error_name, rep_id)


def record_resampling_failure(error_name: str) -> None:
    collector = getattr(_thread_local, "collector", None)
    rep_id = getattr(_thread_local, "_diag_rep_id", None)
    if collector is not None and rep_id is not None:
        collector.record_resampling_failure(error_name, rep_id)


def record_model_warning(warning_name: str) -> None:
    collector = getattr(_thread_local, "collector", None)
    rep_id = getattr(_thread_local, "_diag_rep_id", None)
    if collector is not None and rep_id is not None:
        collector.record_model_warning(warning_name, rep_id)


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
    """Run one registered scenario-method batch.

    If data_rng_factory is provided, replication-level data RNG generators
    are generated per replication_id, guaranteeing pairing across different tier sizes.
    """
    if scenario.get("tier") == "semi_synthetic_development_covariates":
        validate_scenario_target_identity(scenario)

    method_spec = method_by_id(scenario, method_id)
    family = str(method_spec[METHOD_FAMILY])
    rows: list[dict[str, object]] = []

    collector = DiagnosticCollector() if diagnostics is not None else None

    for replication_id in replications:
        if data_rng_factory is not None:
            rep_data_rng = data_rng_factory(int(replication_id))
        else:
            rep_data_rng = data_rng if data_rng is not None else rng
            if rep_data_rng is None:
                raise ValueError("run_batch requires data_rng/data_rng_factory or rng")

        if model_rng_factory is not None:
            rep_model_rng = model_rng_factory(int(replication_id))
        else:
            rep_model_rng = model_rng if model_rng is not None else rng
            if rep_model_rng is None:
                rep_model_rng = rep_data_rng

        if collector is not None:
            ctx = diagnostic_capture(collector, int(replication_id))
        else:
            ctx = nullcontext()

        with ctx:
            data = _generate_replication(scenario, rep_data_rng)
            if family == "predictive":
                estimates = _run_predictive_method(
                    scenario=scenario,
                    method_spec=method_spec,
                    data=data,
                    rng=rep_model_rng,
                )
            elif family == "standalone_estimator":
                estimates = _run_standalone_estimator(
                    scenario=scenario,
                    method_id=method_id,
                    data=data,
                )
            else:
                raise ValueError(f"method={method_id}: unsupported method family {family}")

        for metric_id, estimate in estimates.items():
            rows.append(
                {
                    SCENARIO_ID: str(scenario[SCENARIO_ID]),
                    METHOD_ID: method_id,
                    REPLICATION_ID: int(replication_id),
                    METRIC_ID: metric_id,
                    ESTIMATE: float(estimate),
                    MCSE: None,
                }
            )

    if diagnostics is not None and collector is not None:
        diagnostics.update({
            "fit_failures": collector.fit_failures,
            "resampling_failures": collector.resampling_failures,
            "warnings": collector.model_warnings,
            "affected_replication_ids": sorted(list(collector.affected_replication_ids)),
        })

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


def _generate_replication(
    scenario: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, object]:
    n = int(scenario["sample_size"])
    empirical_base = _replication_empirical_covariates(scenario, n, rng)
    latent, latent_probability_mean = _draw_latent_truth(
        scenario=scenario,
        empirical_base=empirical_base,
        sample_size=n,
        rng=rng,
    )
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

    content = _content_signal(
        scenario, latent, n, rng, base_covariates=empirical_base
    )
    standardized_content = (content - content.mean()) / max(content.std(), 1e-12)
    selective = float(scenario.get("selective_verification_strength", 0.0))
    observable_risk = 1.0 / (1.0 + np.exp(-standardized_content))
    anchor_observed = _verification_draw(
        latent,
        observable_risk=observable_risk,
        base_probability=float(scenario.get("anchor_verification_probability", 1.0)),
        selective_strength=selective,
        rng=rng,
    )
    weak_observed = _verification_draw(
        latent,
        observable_risk=observable_risk,
        base_probability=float(scenario.get("weak_verification_probability", 1.0)),
        selective_strength=selective,
        rng=rng,
    )
    delay = rng.exponential(float(scenario.get("detection_delay_mean_days", 30.0)), n)
    mature = delay <= float(scenario.get("horizon_days", 365.0))

    anchor_values = np.asarray(
        [
            bool(value) if observed and is_mature else None
            for value, observed, is_mature in zip(
                anchor, anchor_observed, mature, strict=True
            )
        ],
        dtype=object,
    )
    weak_values = np.asarray(
        [
            bool(value) if observed and is_mature else None
            for value, observed, is_mature in zip(
                weak, weak_observed, mature, strict=True
            )
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

    features = _feature_matrix(
        scenario=scenario,
        latent=latent,
        standardized_content=standardized_content,
        l2=l2,
        anchor_values=anchor_values,
        weak_values=weak_values,
        anchor_observed=anchor_observed,
        weak_observed=weak_observed,
        rng=rng,
        base_covariates=empirical_base,
    )
    observed_positive = np.asarray([value is True for value in l1_values], dtype=bool)
    verified = np.asarray([value is not None for value in l1_values], dtype=bool)
    feasible_target = observed_positive.astype(int)

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
        "maturity_rate": float(mature.mean()),
        "latent_probability_mean": float(latent_probability_mean),
    }


def _run_predictive_method(
    *,
    scenario: Mapping[str, Any],
    method_spec: Mapping[str, object],
    data: Mapping[str, object],
    rng: np.random.Generator,
) -> dict[str, float]:
    strategy_id = str(method_spec[LABEL_STRATEGY_ID])
    learner_id = str(method_spec[LEARNER_ID])
    training_cost_regime_id = str(method_spec[TRAINING_COST_REGIME_ID])
    imbalance_treatment_id = str(method_spec[IMBALANCE_TREATMENT_ID])
    training_cost = cost_regime_by_id(scenario, training_cost_regime_id)

    features = cast(np.ndarray, data["features"])
    latent = cast(np.ndarray, data["latent"])
    observed_positive = cast(np.ndarray, data["observed_positive"])
    verified = cast(np.ndarray, data["verified"])
    feasible_target = cast(np.ndarray, data["feasible_target"])
    train_index = cast(np.ndarray, data["train_index"])
    test_index = cast(np.ndarray, data["test_index"])

    x_train = features[train_index]
    x_test = features[test_index]
    train_observed_positive = observed_positive[train_index]
    train_verified = verified[train_index]
    train_feasible = feasible_target[train_index]
    y_latent_test = latent[test_index].astype(int)
    y_observed_test = feasible_target[test_index].astype(int)

    fit_result = _fit_strategy_with_cost(
        strategy_id=strategy_id,
        learner_id=learner_id,
        x_train=x_train,
        x_test=x_test,
        observed_positive=train_observed_positive,
        verified=train_verified,
        feasible_target=train_feasible,
        training_cost=training_cost,
        imbalance_treatment_id=imbalance_treatment_id,
        scenario=scenario,
        rng=rng,
    )
    scores = fit_result["scores"]
    threshold = float(fit_result["selected_threshold"])
    predictions = scores >= threshold

    observed_ap = _safe_average_precision(y_observed_test, scores)
    latent_ap = _safe_average_precision(y_latent_test, scores)
    latent_auc = _safe_auc(y_latent_test, scores)
    latent_brier = float(brier_score_loss(y_latent_test, scores))
    primary_budget = _primary_review_budget(scenario)
    precision_at_k = _precision_at_k(
        y_latent_test.astype(bool), scores, float(primary_budget["fraction"])
    )

    estimates: dict[str, float] = {
        "fit_success": float(fit_result["fit_success"]),
        "source_training_rows": float(fit_result["source_training_rows"]),
        "training_rows": float(fit_result["training_rows"]),
        "training_positive_rate": float(fit_result["training_positive_rate"]),
        "post_resampling_positive_rate": float(
            fit_result["post_resampling_positive_rate"]
        ),
        "imbalance_treatment_applied": float(
            fit_result["imbalance_treatment_applied"]
        ),
        "synthetic_rows_added": float(fit_result["synthetic_rows_added"]),
        "selected_threshold": threshold,
        "threshold_selection_success": float(fit_result["threshold_selection_success"]),
        "training_cost_weight_mean": float(fit_result["training_cost_weight_mean"]),
        "training_cost_weight_max": float(fit_result["training_cost_weight_max"]),
        "observed_average_precision": observed_ap,
        "latent_average_precision": latent_ap,
        "latent_auc": latent_auc,
        "latent_brier": latent_brier,
        "latent_precision_at_k": precision_at_k,
        "observed_minus_latent_ap": observed_ap - latent_ap,
        "verification_rate": float(data["verification_rate"]),
        "maturity_rate": float(data["maturity_rate"]),
        "realized_prevalence": float(latent.mean()),
        "latent_probability_mean": float(data.get("latent_probability_mean", latent.mean())),
        "empirical_panel_rows": float(scenario.get("empirical_panel_rows", 0.0)),
        "empirical_observed_positive_rate": float(
            scenario.get("empirical_observed_positive_rate", observed_positive.mean())
        ),
        "empirical_known_label_rate": float(
            scenario.get("empirical_known_label_rate", verified.mean())
        ),
        "calibrated_latent_prevalence": float(scenario["prevalence"]),
    }

    for evaluation_cost in evaluation_cost_regimes(scenario):
        regime_id = str(evaluation_cost["cost_regime_id"])
        threshold_cost = _classification_cost(
            truth=y_latent_test.astype(bool),
            predicted_positive=predictions,
            cost_regime=evaluation_cost,
        )
        for metric in (
            "total_cost",
            "cost_per_firm",
            "normalized_cost_regret",
            "cost_savings_vs_all_negative",
            "cost_regret_vs_oracle",
            "false_negative_component",
            "false_positive_component",
            "review_component",
        ):
            estimates[threshold_cost_metric_id(regime_id, metric)] = float(
                threshold_cost[metric]
            )

        for budget in review_budgets(scenario):
            budget_id = str(budget["review_budget_id"])
            budget_result = _budget_cost(
                truth=y_latent_test.astype(bool),
                scores=scores,
                fraction=float(budget["fraction"]),
                cost_regime=evaluation_cost,
            )
            for metric in (
                "total_cost",
                "normalized_cost_regret",
                "cost_savings_vs_all_negative",
                "cost_regret_vs_oracle",
                "precision",
                "recall",
            ):
                estimates[budget_cost_metric_id(budget_id, regime_id, metric)] = float(
                    budget_result[metric]
                )
    return estimates


def _fit_strategy_with_cost(
    *,
    strategy_id: str,
    learner_id: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    observed_positive: np.ndarray,
    verified: np.ndarray,
    feasible_target: np.ndarray,
    training_cost: Mapping[str, object],
    imbalance_treatment_id: str,
    scenario: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, object]:
    validation_fraction = float(
        cast(dict[str, object], scenario["cost_sensitive_settings"])[
            "threshold_validation_fraction"
        ]
    )
    fit_positions, validation_positions = _validation_split(
        len(x_train), validation_fraction, rng
    )
    ipw_all = (
        _verification_weights(x_train, verified, np.arange(len(x_train)), rng)
        if strategy_id == "ipw"
        else np.ones(len(x_train), dtype=float)
    )

    (
        validation_scores,
        validation_success,
        _,
        _,
        _,
        validation_weight_stats,
        _,
    ) = _fit_for_subset(
        strategy_id=strategy_id,
        learner_id=learner_id,
        x_train=x_train,
        predict_x=x_train[validation_positions],
        subset_positions=fit_positions,
        observed_positive=observed_positive,
        verified=verified,
        feasible_target=feasible_target,
        ipw_all=ipw_all,
        training_cost=training_cost,
        imbalance_treatment_id=imbalance_treatment_id,
        scenario=scenario,
        rng=rng,
    )
    validation_target, validation_weights = _validation_target_and_weights(
        strategy_id=strategy_id,
        positions=validation_positions,
        observed_positive=observed_positive,
        verified=verified,
        feasible_target=feasible_target,
        ipw_all=ipw_all,
    )
    threshold, threshold_success = _select_cost_threshold(
        y_true=validation_target,
        scores=validation_scores,
        cost_regime=training_cost,
        sample_weight=validation_weights,
    )

    (
        scores,
        fit_success,
        source_training_rows,
        training_rows,
        positive_rate,
        full_weight_stats,
        resampling_stats,
    ) = _fit_for_subset(
        strategy_id=strategy_id,
        learner_id=learner_id,
        x_train=x_train,
        predict_x=x_test,
        subset_positions=np.arange(len(x_train)),
        observed_positive=observed_positive,
        verified=verified,
        feasible_target=feasible_target,
        ipw_all=ipw_all,
        training_cost=training_cost,
        imbalance_treatment_id=imbalance_treatment_id,
        scenario=scenario,
        rng=rng,
    )
    return {
        "scores": scores,
        "fit_success": min(float(fit_success), float(validation_success)),
        "source_training_rows": source_training_rows,
        "training_rows": training_rows,
        "training_positive_rate": positive_rate,
        "post_resampling_positive_rate": float(
            resampling_stats["post_resampling_positive_rate"]
        ),
        "imbalance_treatment_applied": float(resampling_stats["applied"]),
        "synthetic_rows_added": float(resampling_stats["synthetic_rows_added"]),
        "selected_threshold": threshold,
        "threshold_selection_success": threshold_success,
        "training_cost_weight_mean": full_weight_stats[0],
        "training_cost_weight_max": full_weight_stats[1],
        "validation_cost_weight_mean": validation_weight_stats[0],
    }


def _fit_for_subset(
    *,
    strategy_id: str,
    learner_id: str,
    x_train: np.ndarray,
    predict_x: np.ndarray,
    subset_positions: np.ndarray,
    observed_positive: np.ndarray,
    verified: np.ndarray,
    feasible_target: np.ndarray,
    ipw_all: np.ndarray,
    training_cost: Mapping[str, object],
    imbalance_treatment_id: str,
    scenario: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    float,
    int,
    int,
    float,
    tuple[float, float],
    dict[str, float],
]:
    if strategy_id == "pu":
        if imbalance_treatment_id != "none":
            raise ValueError("SMOTE/ADASYN are forbidden for PU learning")
        positive = subset_positions[observed_positive[subset_positions]]
        unlabeled = subset_positions[~observed_positive[subset_positions]]
        scores, success, weight_stats = _fit_pu_ensemble_cost_sensitive(
            learner_id=learner_id,
            x_train=x_train,
            predict_x=predict_x,
            positive_indices=positive,
            unlabeled_indices=unlabeled,
            bags=int(scenario.get("pu_bags", 5)),
            unlabeled_to_positive_ratio=float(
                scenario.get("pu_unlabeled_to_positive_ratio", 1.0)
            ),
            training_cost=training_cost,
            scenario=scenario,
            rng=rng,
        )
        training_rows = len(positive) + len(unlabeled)
        positive_rate = len(positive) / max(1, training_rows)
        stats = {
            "applied": 0.0,
            "synthetic_rows_added": 0.0,
            "post_resampling_positive_rate": positive_rate,
        }
        return (
            scores,
            success,
            training_rows,
            training_rows,
            positive_rate,
            weight_stats,
            stats,
        )

    positions, y, base_weights = _supervised_training_view(
        strategy_id=strategy_id,
        subset_positions=subset_positions,
        verified=verified,
        feasible_target=feasible_target,
        ipw_all=ipw_all,
    )
    source_rows = len(positions)
    source_positive_rate = float(y.mean()) if len(y) else 0.0
    x_fit = x_train[positions]
    x_fit, y_fit, selection_weights, resampling_stats, resampling_success = (
        _apply_imbalance_treatment(
            x=x_fit,
            y=y,
            selection_weights=base_weights,
            strategy_id=strategy_id,
            treatment_id=imbalance_treatment_id,
            scenario=scenario,
            rng=rng,
        )
    )
    cost_weights = _combined_training_weights(
        y=y_fit,
        selection_weights=selection_weights,
        cost_regime=training_cost,
        maximum_cost_weight=float(
            cast(dict[str, object], scenario["cost_sensitive_settings"])[
                "maximum_cost_weight"
            ]
        ),
    )
    scores, success, diag = _fit_and_predict(
        learner_id,
        x_fit,
        y_fit,
        predict_x,
        cost_weights,
        rng,
    )
    if success == 0.0:
        import sys
        print(f"Learner fit failure ({learner_id}): {diag}", file=sys.stderr, flush=True)
    success = min(float(success), float(resampling_success))
    stats = _weight_stats(cost_weights)
    return (
        scores,
        success,
        source_rows,
        len(y_fit),
        source_positive_rate,
        stats,
        resampling_stats,
    )


def _apply_imbalance_treatment(
    *,
    x: np.ndarray,
    y: np.ndarray,
    selection_weights: np.ndarray,
    strategy_id: str,
    treatment_id: str,
    scenario: Mapping[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], float]:
    source_rows = len(y)
    source_positive_rate = float(y.mean()) if source_rows else 0.0
    neutral_stats = {
        "applied": 0.0,
        "synthetic_rows_added": 0.0,
        "post_resampling_positive_rate": source_positive_rate,
    }
    if treatment_id == "none":
        return x, y, selection_weights, neutral_stats, 1.0
    if treatment_id not in {"smote", "adasyn"}:
        raise ValueError(f"unsupported imbalance treatment: {treatment_id}")
    if strategy_id not in {"naive_observed_as_negative", "verified_only"}:
        raise ValueError(
            f"imbalance treatment={treatment_id} is forbidden for strategy={strategy_id}"
        )
    if len(selection_weights) and not np.allclose(selection_weights, selection_weights[0]):
        raise ValueError("SMOTE/ADASYN cannot be combined with IPW selection weights")
    classes, counts = np.unique(y.astype(int), return_counts=True)
    if len(classes) != 2 or int(counts.min()) < 2:
        record_resampling_failure("InsufficientMinorityClass")
        return x, y, selection_weights, neutral_stats, 0.0

    settings = scenario.get("imbalance_treatment_settings")
    if not isinstance(settings, dict):
        raise ValueError("scenario is missing imbalance_treatment_settings")
    maximum_neighbors = int(cast(dict[str, object], settings)["maximum_neighbors"])
    neighbors = max(1, min(maximum_neighbors, int(counts.min()) - 1))
    random_state = int(rng.integers(0, 2**31 - 1))
    try:
        if treatment_id == "smote":
            from imblearn.over_sampling import SMOTE

            sampler = SMOTE(k_neighbors=neighbors, random_state=random_state)
        else:
            from imblearn.over_sampling import ADASYN

            sampler = ADASYN(n_neighbors=neighbors, random_state=random_state)
        resampled_x, resampled_y = sampler.fit_resample(x, y.astype(int))
    except Exception as exc:
        record_resampling_failure(type(exc).__name__)
        return x, y, selection_weights, neutral_stats, 0.0

    resampled_y = np.asarray(resampled_y, dtype=int)
    added = max(0, len(resampled_y) - source_rows)
    stats = {
        "applied": 1.0,
        "synthetic_rows_added": float(added),
        "post_resampling_positive_rate": float(resampled_y.mean())
        if len(resampled_y)
        else 0.0,
    }
    return (
        np.asarray(resampled_x, dtype=float),
        resampled_y,
        np.ones(len(resampled_y), dtype=float),
        stats,
        1.0,
    )

def _supervised_training_view(
    *,
    strategy_id: str,
    subset_positions: np.ndarray,
    verified: np.ndarray,
    feasible_target: np.ndarray,
    ipw_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if strategy_id == "naive_observed_as_negative":
        positions = subset_positions
        return positions, feasible_target[positions].astype(int), np.ones(len(positions))
    if strategy_id == "verified_only":
        positions = subset_positions[verified[subset_positions]]
        return positions, feasible_target[positions].astype(int), np.ones(len(positions))
    if strategy_id == "ipw":
        positions = subset_positions[verified[subset_positions]]
        return positions, feasible_target[positions].astype(int), ipw_all[positions]
    raise ValueError(f"unsupported supervised label strategy: {strategy_id}")


def _validation_target_and_weights(
    *,
    strategy_id: str,
    positions: np.ndarray,
    observed_positive: np.ndarray,
    verified: np.ndarray,
    feasible_target: np.ndarray,
    ipw_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy_id == "pu":
        return observed_positive[positions].astype(int), np.ones(len(positions))
    selected, y, weights = _supervised_training_view(
        strategy_id=strategy_id,
        subset_positions=positions,
        verified=verified,
        feasible_target=feasible_target,
        ipw_all=ipw_all,
    )
    position_lookup = {int(value): index for index, value in enumerate(positions)}
    score_positions = np.asarray([position_lookup[int(value)] for value in selected], dtype=int)
    # Caller scores all validation positions; return aligned target and attach
    # score positions as a compact third-party-free convention by expanding
    # nonselected rows to NaN weights. This branch is replaced immediately below.
    aligned_y = np.zeros(len(positions), dtype=int)
    aligned_w = np.zeros(len(positions), dtype=float)
    if len(score_positions):
        aligned_y[score_positions] = y
        aligned_w[score_positions] = weights
    return aligned_y, aligned_w


def _validation_split(
    count: int,
    fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 20:
        positions = np.arange(count)
        return positions, positions
    validation_count = max(10, min(count - 10, int(round(count * fraction))))
    permutation = rng.permutation(count)
    return permutation[validation_count:], permutation[:validation_count]


def _combined_training_weights(
    *,
    y: np.ndarray,
    selection_weights: np.ndarray,
    cost_regime: Mapping[str, object],
    maximum_cost_weight: float,
) -> np.ndarray:
    positive_weight = float(cost_regime["false_negative_cost"]) + float(
        cost_regime["true_positive_benefit"]
    )
    negative_weight = float(cost_regime["false_positive_cost"])
    class_weights = np.where(y.astype(bool), positive_weight, negative_weight)
    combined = np.asarray(selection_weights, dtype=float) * class_weights
    combined = np.clip(combined, 1e-12, maximum_cost_weight)
    mean = float(combined.mean()) if len(combined) else 1.0
    if mean <= 0 or not np.isfinite(mean):
        return np.ones(len(y), dtype=float)
    return combined / mean


def _fit_and_predict(
    learner_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    sample_weight: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict[str, str]]:
    fallback = float(np.average(y_train, weights=sample_weight)) if len(y_train) else 0.0
    fallback_scores = np.full(len(x_test), fallback, dtype=float)
    if len(y_train) < 10 or len(np.unique(y_train)) < 2:
        record_fit_failure("InsufficientData")
        return fallback_scores, 0.0, {
            "failure_type": "InsufficientData",
            "failure_message": "len(y_train) < 10 or unique < 2",
        }
    random_state = int(rng.integers(0, 2**31 - 1))
    try:
        estimator = _learner(learner_id, random_state)
        fit_x, fit_y, fit_weight = x_train, y_train, sample_weight
        if sample_weight is not None and not _supports_sample_weight(estimator):
            sampled = _weighted_bootstrap_indices(sample_weight, len(y_train), rng)
            fit_x = x_train[sampled]
            fit_y = y_train[sampled]
            fit_weight = None
        import warnings as _w
        with _w.catch_warnings(record=True) as captured_warnings:
            _w.simplefilter("always")
            _fit_estimator(estimator, fit_x, fit_y, fit_weight)
        for w in (captured_warnings or []):
            record_model_warning(w.category.__name__)
        scores = _predict_scores(estimator, x_test)
        if not np.all(np.isfinite(scores)):
            raise ValueError("non-finite predictions")
        return np.clip(scores, 0.0, 1.0), 1.0, {}
    except (ValueError, FloatingPointError) as exc:
        record_fit_failure(type(exc).__name__)
        return fallback_scores, 0.0, {
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:500],
        }


def _fit_pu_ensemble_cost_sensitive(
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
) -> tuple[np.ndarray, float, tuple[float, float]]:
    if (
        len(positive_indices) < 2
        or len(unlabeled_indices) < 2
        or bags < 1
        or unlabeled_to_positive_ratio <= 0
    ):
        record_resampling_failure("InsufficientPUData")
        prior = len(positive_indices) / max(1, len(positive_indices) + len(unlabeled_indices))
        return np.full(len(predict_x), prior, dtype=float), 0.0, (1.0, 1.0)

    bag_scores: list[np.ndarray] = []
    bag_weight_means: list[float] = []
    bag_weight_maxima: list[float] = []
    sample_count = min(
        len(unlabeled_indices),
        max(2, int(math.ceil(len(positive_indices) * unlabeled_to_positive_ratio))),
    )
    for _ in range(bags):
        sampled_unlabeled = rng.choice(
            unlabeled_indices,
            size=sample_count,
            replace=sample_count > len(unlabeled_indices),
        )
        indices = np.concatenate([positive_indices, sampled_unlabeled])
        y = np.concatenate(
            [
                np.ones(len(positive_indices), dtype=int),
                np.zeros(len(sampled_unlabeled), dtype=int),
            ]
        )
        weights = _combined_training_weights(
            y=y,
            selection_weights=np.ones(len(y)),
            cost_regime=training_cost,
            maximum_cost_weight=float(
                cast(dict[str, object], scenario["cost_sensitive_settings"])[
                    "maximum_cost_weight"
                ]
            ),
        )
        scores, success, diag = _fit_and_predict(
            learner_id,
            x_train[indices],
            y,
            predict_x,
            weights,
            rng,
        )
        if success == 0.0:
            import sys
            print(f"PU bag learner fit failure ({learner_id}): {diag}", file=sys.stderr, flush=True)
        if success == 1.0:
            bag_scores.append(scores)
            mean, maximum = _weight_stats(weights)
            bag_weight_means.append(mean)
            bag_weight_maxima.append(maximum)

    if not bag_scores:
        prior = len(positive_indices) / max(1, len(positive_indices) + len(unlabeled_indices))
        return np.full(len(predict_x), prior, dtype=float), 0.0, (1.0, 1.0)
    return (
        np.mean(np.vstack(bag_scores), axis=0),
        1.0,
        (
            float(np.mean(bag_weight_means)),
            float(np.max(bag_weight_maxima)),
        ),
    )


def _verification_weights(
    x_train: np.ndarray,
    verified: np.ndarray,
    requested_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(np.unique(verified.astype(int))) < 2:
        return np.ones(len(requested_indices), dtype=float)
    propensity = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    max_iter=1000,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                ),
            ),
        ]
    )
    propensity.fit(x_train, verified.astype(int))
    probabilities = propensity.predict_proba(x_train[requested_indices])[:, 1]
    probabilities = np.clip(probabilities, 0.05, 1.0)
    return np.minimum(1.0 / probabilities, 20.0)


def _select_cost_threshold(
    *,
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_regime: Mapping[str, object],
    sample_weight: np.ndarray,
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    valid = np.isfinite(s) & np.isfinite(w) & (w > 0)
    y, s, w = y[valid], s[valid], w[valid]
    fallback = _bayes_cost_threshold(cost_regime)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return fallback, 0.0

    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([0.0, fallback, 0.5, 1.0]),
                np.quantile(s, np.linspace(0.0, 1.0, 101)),
            ]
        )
    )
    best_threshold = fallback
    best_cost = math.inf
    for threshold in candidates:
        predicted = s >= float(threshold)
        cost = _weighted_feasible_cost(y.astype(bool), predicted, cost_regime, w)
        if cost < best_cost - 1e-12 or (
            math.isclose(cost, best_cost, rel_tol=0.0, abs_tol=1e-12)
            and abs(float(threshold) - fallback) < abs(best_threshold - fallback)
        ):
            best_cost = cost
            best_threshold = float(threshold)
    return float(np.clip(best_threshold, 0.0, 1.0)), 1.0


def _bayes_cost_threshold(cost_regime: Mapping[str, object]) -> float:
    numerator = float(cost_regime["false_positive_cost"]) + float(
        cost_regime["review_cost"]
    )
    denominator = (
        float(cost_regime["false_positive_cost"])
        + float(cost_regime["false_negative_cost"])
        + float(cost_regime["true_positive_benefit"])
    )
    return float(np.clip(numerator / max(denominator, 1e-12), 0.0, 1.0))


def _weighted_feasible_cost(
    truth: np.ndarray,
    predicted_positive: np.ndarray,
    cost_regime: Mapping[str, object],
    weights: np.ndarray,
) -> float:
    truth = truth.astype(bool)
    predicted_positive = predicted_positive.astype(bool)
    fp = (~truth) & predicted_positive
    fn = truth & (~predicted_positive)
    tp = truth & predicted_positive
    return float(
        np.sum(weights[predicted_positive]) * float(cost_regime["review_cost"])
        + np.sum(weights[fp]) * float(cost_regime["false_positive_cost"])
        + np.sum(weights[fn]) * float(cost_regime["false_negative_cost"])
        - np.sum(weights[tp]) * float(cost_regime["true_positive_benefit"])
    )


def _classification_cost(
    *,
    truth: np.ndarray,
    predicted_positive: np.ndarray,
    cost_regime: Mapping[str, object],
) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted_positive, dtype=bool)
    tp = truth & predicted
    fp = (~truth) & predicted
    fn = truth & (~predicted)

    review_component = float(predicted.sum()) * float(cost_regime["review_cost"])
    fp_component = float(fp.sum()) * float(cost_regime["false_positive_cost"])
    fn_component = float(fn.sum()) * float(cost_regime["false_negative_cost"])
    benefit_component = float(tp.sum()) * float(cost_regime["true_positive_benefit"])
    total = review_component + fp_component + fn_component - benefit_component
    all_negative = float(truth.sum()) * float(cost_regime["false_negative_cost"])
    oracle = _oracle_unconstrained_cost(truth, cost_regime)
    denominator = max(abs(all_negative - oracle), 1e-12)
    return {
        "total_cost": total,
        "cost_per_firm": total / max(1, len(truth)),
        "normalized_cost_regret": (total - oracle) / denominator,
        "cost_savings_vs_all_negative": all_negative - total,
        "cost_regret_vs_oracle": total - oracle,
        "false_negative_component": fn_component,
        "false_positive_component": fp_component,
        "review_component": review_component,
        "true_positive_benefit_component": benefit_component,
    }


def _oracle_unconstrained_cost(
    truth: np.ndarray,
    cost_regime: Mapping[str, object],
) -> float:
    positive_review_cost = float(cost_regime["review_cost"]) - float(
        cost_regime["true_positive_benefit"]
    )
    positive_skip_cost = float(cost_regime["false_negative_cost"])
    negative_review_cost = float(cost_regime["review_cost"]) + float(
        cost_regime["false_positive_cost"]
    )
    positive_count = int(np.sum(truth))
    negative_count = len(truth) - positive_count
    return (
        positive_count * min(positive_review_cost, positive_skip_cost)
        + negative_count * min(negative_review_cost, 0.0)
    )


def _budget_cost(
    *,
    truth: np.ndarray,
    scores: np.ndarray,
    fraction: float,
    cost_regime: Mapping[str, object],
) -> dict[str, float]:
    count = max(1, min(len(truth), int(math.ceil(len(truth) * fraction))))
    model_order = np.argsort(-scores, kind="stable")
    predicted = np.zeros(len(truth), dtype=bool)
    predicted[model_order[:count]] = True
    model = _classification_cost(
        truth=truth,
        predicted_positive=predicted,
        cost_regime=cost_regime,
    )

    review_savings = np.where(
        truth,
        float(cost_regime["false_negative_cost"])
        - float(cost_regime["review_cost"])
        + float(cost_regime["true_positive_benefit"]),
        -float(cost_regime["review_cost"])
        - float(cost_regime["false_positive_cost"]),
    )
    oracle_order = np.argsort(-review_savings, kind="stable")
    oracle_predicted = np.zeros(len(truth), dtype=bool)
    oracle_predicted[oracle_order[:count]] = True
    oracle = _classification_cost(
        truth=truth,
        predicted_positive=oracle_predicted,
        cost_regime=cost_regime,
    )["total_cost"]
    all_negative = float(np.sum(truth)) * float(cost_regime["false_negative_cost"])
    denominator = max(abs(all_negative - oracle), 1e-12)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum((~truth) & predicted))
    fn = int(np.sum(truth & (~predicted)))
    model.update(
        {
            "cost_regret_vs_oracle": model["total_cost"] - oracle,
            "normalized_cost_regret": (model["total_cost"] - oracle) / denominator,
            "cost_savings_vs_all_negative": all_negative - model["total_cost"],
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
        }
    )
    return model


def _weight_stats(weights: np.ndarray | None) -> tuple[float, float]:
    if weights is None or len(weights) == 0:
        return 1.0, 1.0
    values = np.asarray(weights, dtype=float)
    return float(values.mean()), float(values.max())


def _primary_review_budget(scenario: Mapping[str, object]) -> dict[str, object]:
    settings = cast(dict[str, object], scenario["cost_sensitive_settings"])
    primary_id = str(settings["primary_review_budget_id"])
    matches = [item for item in review_budgets(scenario) if item["review_budget_id"] == primary_id]
    if len(matches) != 1:
        raise ValueError("primary review budget must be registered exactly once")
    return matches[0]

def _learner(learner_id: str, random_state: int) -> object:
    if learner_id == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if learner_id == "elastic_net_logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        C=1.0,
                        l1_ratio=0.5,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if learner_id == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=5,
            random_state=random_state,
        )
    if learner_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=60,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=random_state,
        )
    if learner_id == "main_boosting":
        return HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=random_state,
        )
    if learner_id == "linear_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if learner_id == "rbf_svm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=False,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if learner_id == "k_nearest_neighbors":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        )
    if learner_id == "gaussian_naive_bayes":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GaussianNB()),
            ]
        )
    if learner_id == "adaboost":
        return AdaBoostClassifier(
            n_estimators=60,
            learning_rate=0.05,
            random_state=random_state,
        )
    if learner_id == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=60,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=1,
            random_state=random_state,
        )
    if learner_id == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        )
    if learner_id == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=60,
            learning_rate=0.05,
            num_leaves=15,
            verbosity=-1,
            n_jobs=1,
            random_state=random_state,
        )
    if learner_id == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=60,
            depth=5,
            learning_rate=0.05,
            verbose=False,
            thread_count=1,
            random_seed=random_state,
        )
    if learner_id == "multilayer_perceptron":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        alpha=0.001,
                        early_stopping=True,
                        max_iter=200,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"unsupported learner_id: {learner_id}")


def _supports_sample_weight(estimator: object) -> bool:
    if isinstance(estimator, Pipeline):
        final_name, final_estimator = estimator.steps[-1]
        return "sample_weight" in inspect.signature(final_estimator.fit).parameters
    return "sample_weight" in inspect.signature(estimator.fit).parameters


def _fit_estimator(
    estimator: object,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> None:
    if sample_weight is None:
        estimator.fit(x, y)
        return
    if isinstance(estimator, Pipeline):
        final_name = estimator.steps[-1][0]
        estimator.fit(x, y, **{f"{final_name}__sample_weight": sample_weight})
    else:
        estimator.fit(x, y, sample_weight=sample_weight)


def _predict_scores(estimator: object, x: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(x), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    return np.asarray(estimator.predict(x), dtype=float)


def _weighted_bootstrap_indices(
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    normalized = np.asarray(weights, dtype=float)
    normalized = np.clip(normalized, 0.0, None)
    if normalized.sum() <= 0:
        normalized = np.ones_like(normalized)
    normalized /= normalized.sum()
    return rng.choice(np.arange(len(weights)), size=count, replace=True, p=normalized)


def _run_standalone_estimator(
    *,
    scenario: Mapping[str, Any],
    method_id: str,
    data: Mapping[str, object],
) -> dict[str, float]:
    true_prevalence = float(scenario["prevalence"])
    source_rows = cast(list[dict[str, bool | None]], data["source_rows"])
    accuracy = cast(dict[str, tuple[float, float]], data["accuracy"])

    if method_id == "latent_bayesian":
        estimate, lower, upper = _hierarchical_pi_estimate(source_rows, accuracy)
    elif method_id == "partial_identification":
        estimate, lower, upper = _partial_identification_bounds(
            source_rows=source_rows,
            scenario=scenario,
        )
    else:
        raise ValueError(f"unsupported standalone estimator: {method_id}")

    return {
        "fit_success": 1.0,
        "prevalence_error": estimate - true_prevalence,
        "prevalence_squared_error": (estimate - true_prevalence) ** 2,
        "prevalence_coverage": float(lower <= true_prevalence <= upper),
        "interval_width": upper - lower,
        "empirical_panel_rows": float(scenario.get("empirical_panel_rows", 0.0)),
        "empirical_observed_positive_rate": float(
            scenario.get(
                "empirical_observed_positive_rate",
                np.mean([any(value is True for value in row.values()) for row in source_rows]),
            )
        ),
        "empirical_known_label_rate": float(
            scenario.get(
                "empirical_known_label_rate",
                np.mean([any(value is not None for value in row.values()) for row in source_rows]),
            )
        ),
        "calibrated_latent_prevalence": true_prevalence,
    }


def _partial_identification_bounds(
    *,
    source_rows: list[dict[str, bool | None]],
    scenario: Mapping[str, Any],
) -> tuple[float, float, float]:
    anchor_values = [row["anchor"] for row in source_rows if row["anchor"] is not None]
    if not anchor_values:
        return 0.5, 0.0, 1.0
    observed_rate = float(np.mean(np.asarray(anchor_values, dtype=float)))

    se_center = float(scenario["anchor_sensitivity"])
    fp_center = float(scenario["anchor_false_positive"])
    se_lower = float(scenario.get("sensitivity_lower", max(0.05, se_center - 0.15)))
    se_upper = float(scenario.get("sensitivity_upper", min(1.0, se_center + 0.15)))
    fp_lower = float(scenario.get("false_positive_lower", max(0.0, fp_center - 0.02)))
    fp_upper = float(scenario.get("false_positive_upper", min(0.25, fp_center + 0.03)))

    candidates: list[float] = []
    for sensitivity in (se_lower, se_upper):
        for false_positive in (fp_lower, fp_upper):
            denominator = sensitivity - false_positive
            if denominator <= 1e-6:
                continue
            candidates.append(
                float(np.clip((observed_rate - false_positive) / denominator, 0.0, 1.0))
            )
    if not candidates:
        return 0.5, 0.0, 1.0
    lower, upper = min(candidates), max(candidates)
    return (lower + upper) / 2.0, lower, upper



def summarize_mcse(
    batches: Sequence[pd.DataFrame],
    *,
    minimum_replications: int,
    maximum_replications: int,
    pass_fail_mcse_maximum: float,
    l3_minimum_replications: int | None = None,
    l3_maximum_replications: int | None = None,
    l3_pass_fail_mcse_maximum: float | None = None,
    extended_minimum_replications: int | None = None,
    extended_maximum_replications: int | None = None,
    extended_pass_fail_mcse_maximum: float | None = None,
    cost_sensitive_minimum_replications: int | None = None,
    cost_sensitive_maximum_replications: int | None = None,
    cost_sensitive_pass_fail_mcse_maximum: float | None = None,
    cost_mcse_relative_fraction: float = 0.02,
    continuous_mcse_fraction: float | None = None,
    minimum_meaningful_improvement: float | None = None,
) -> dict[str, object]:
    """Compute MCSE with method-tier and metric-scale-aware precision gates."""
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
        metric_name = str(metric_id)
        is_standalone = metric_name.startswith("prevalence_") or metric_name == "interval_width"
        learner_tiers = set(frame["learner_tier"].dropna().astype(str)) if "learner_tier" in frame.columns else set()
        training_costs = set(frame["training_cost_regime_id"].dropna().astype(str)) if "training_cost_regime_id" in frame.columns else set()
        treatments = set(frame["imbalance_treatment_id"].dropna().astype(str)) if "imbalance_treatment_id" in frame.columns else set()
        if len(learner_tiers) > 1 or len(training_costs) > 1 or len(treatments) > 1:
            raise ValueError(f"scenario={scenario_id}, method={method_id}: mixed method metadata")
        is_imbalance_robustness = treatments not in (set(), {"none"}, {"not_applicable"})
        is_extended = learner_tiers in ({"extended"}, {"methodological"}) or is_imbalance_robustness
        is_cost_sensitive_core = (
            learner_tiers == {"core"}
            and not is_imbalance_robustness
            and training_costs not in (set(), {"symmetric"})
        )

        required_replications = (
            l3_minimum_replications
            if is_standalone and l3_minimum_replications is not None
            else extended_minimum_replications
            if is_extended and extended_minimum_replications is not None
            else cost_sensitive_minimum_replications
            if is_cost_sensitive_core and cost_sensitive_minimum_replications is not None
            else minimum_replications
        )
        replication_cap = (
            l3_maximum_replications
            if is_standalone and l3_maximum_replications is not None
            else extended_maximum_replications
            if is_extended and extended_maximum_replications is not None
            else cost_sensitive_maximum_replications
            if is_cost_sensitive_core and cost_sensitive_maximum_replications is not None
            else maximum_replications
        )
        pass_fail_target = (
            l3_pass_fail_mcse_maximum
            if is_standalone and l3_pass_fail_mcse_maximum is not None
            else extended_pass_fail_mcse_maximum
            if is_extended and extended_pass_fail_mcse_maximum is not None
            else cost_sensitive_pass_fail_mcse_maximum
            if is_cost_sensitive_core and cost_sensitive_pass_fail_mcse_maximum is not None
            else pass_fail_mcse_maximum
        )

        mean = float(frame[ESTIMATE].mean())
        actual_mcse = standard_deviation / math.sqrt(count)
        gate_required = _mcse_gate_required(metric_name)
        binary_metric = metric_name.endswith(("_success", "_coverage", "_stability"))
        mcse_target = float(pass_fail_target)
        if gate_required and "cost_per_firm" in metric_name:
            mcse_target = max(mcse_target, cost_mcse_relative_fraction * max(abs(mean), 1e-6))
        elif (
            gate_required
            and not binary_metric
            and continuous_mcse_fraction is not None
            and minimum_meaningful_improvement is not None
        ):
            mcse_target = min(
                mcse_target,
                continuous_mcse_fraction * minimum_meaningful_improvement,
            )

        row: dict[str, object] = {
            SCENARIO_ID: str(scenario_id),
            METHOD_ID: str(method_id),
            METRIC_ID: metric_name,
            "replications": count,
            "mean": mean,
            MCSE: actual_mcse,
            "mcse_target": mcse_target,
            "mcse_gate_required": gate_required,
            "replication_tier": (
                "standalone"
                if is_standalone
                else "imbalance_robustness"
                if is_imbalance_robustness
                else "extended_predictive"
                if is_extended
                else "cost_sensitive_core"
                if is_cost_sensitive_core
                else "core_cost_neutral"
            ),
            "minimum_replications_met": count >= required_replications,
            "mcse_target_met": (not gate_required) or actual_mcse <= mcse_target,
            "maximum_replications_reached": count >= replication_cap,
        }
        if metric_name == "prevalence_squared_error":
            row["rmse"] = math.sqrt(max(0.0, mean))
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
        "reason_code": (
            None
            if precision_met
            else "MCSE_TARGET_NOT_MET_AT_CAP"
            if maximum_reached
            else "ADDITIONAL_REPLICATIONS_REQUIRED"
        ),
        "minimum_replications": minimum_replications,
        "maximum_replications": maximum_replications,
        "pass_fail_mcse_maximum": pass_fail_mcse_maximum,
        "l3_minimum_replications": l3_minimum_replications,
        "l3_maximum_replications": l3_maximum_replications,
        "l3_pass_fail_mcse_maximum": l3_pass_fail_mcse_maximum,
        "extended_minimum_replications": extended_minimum_replications,
        "extended_maximum_replications": extended_maximum_replications,
        "extended_pass_fail_mcse_maximum": extended_pass_fail_mcse_maximum,
        "cost_sensitive_minimum_replications": cost_sensitive_minimum_replications,
        "cost_sensitive_maximum_replications": cost_sensitive_maximum_replications,
        "cost_sensitive_pass_fail_mcse_maximum": cost_sensitive_pass_fail_mcse_maximum,
        "cost_mcse_relative_fraction": cost_mcse_relative_fraction,
        "continuous_mcse_fraction": continuous_mcse_fraction,
        "minimum_meaningful_improvement": minimum_meaningful_improvement,
        "precision_target_met": precision_met,
        "metrics": metrics,
    }


def _mcse_gate_required(metric_id: str) -> bool:
    descriptive_fragments = (
        "training_rows",
        "training_positive_rate",
        "training_cost_weight_",
        "selected_threshold",
        "total_cost",
        "cost_savings_vs_all_negative",
        "cost_regret_vs_oracle",
        "false_negative_component",
        "false_positive_component",
        "review_component",
        "verification_rate",
        "maturity_rate",
        "realized_prevalence",
        "latent_probability_mean",
        "empirical_panel_rows",
        "empirical_observed_positive_rate",
        "empirical_known_label_rate",
        "calibrated_latent_prevalence",
    )
    return not any(fragment in metric_id for fragment in descriptive_fragments)

def _feature_matrix(
    *,
    scenario: Mapping[str, Any],
    latent: np.ndarray,
    standardized_content: np.ndarray,
    l2: np.ndarray,
    anchor_values: np.ndarray,
    weak_values: np.ndarray,
    anchor_observed: np.ndarray,
    weak_observed: np.ndarray,
    rng: np.random.Generator,
    base_covariates: np.ndarray | None = None,
) -> np.ndarray:
    n = len(latent)
    requested = max(8, int(scenario.get("simulation_feature_count", 12)))
    if base_covariates is not None:
        base = np.asarray(base_covariates, dtype=float)
        if len(base) != n:
            raise ValueError("empirical covariate rows must equal P02-derived sample size")
        if base.ndim == 1:
            base = base.reshape(-1, 1)
    elif scenario.get("tier") == "semi_synthetic_development_covariates":
        raise ValueError("empirical semi-synthetic replication requires exact P02 covariates")
    else:
        base = rng.normal(0.0, 1.0, size=(n, max(1, requested - 7)))

    anchor_numeric = np.asarray(
        [0.0 if value is None else float(bool(value)) for value in anchor_values]
    )
    weak_numeric = np.asarray(
        [0.0 if value is None else float(bool(value)) for value in weak_values]
    )
    engineered = np.column_stack(
        [
            standardized_content,
            l2,
            anchor_observed.astype(float),
            weak_observed.astype(float),
            anchor_numeric,
            weak_numeric,
            standardized_content * l2,
            standardized_content**2,
        ]
    )
    matrix = np.column_stack([engineered, base])
    if matrix.shape[1] < requested:
        matrix = np.column_stack(
            [matrix, rng.normal(size=(n, requested - matrix.shape[1]))]
        )
    return matrix[:, :requested].astype(float)


def _safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(y_true) == 0 or int(np.sum(y_true)) == 0:
        return 0.0
    value = average_precision(y_true.astype(bool).tolist(), scores.tolist())
    return 0.0 if value is None else float(value)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    value = roc_auc(y_true.astype(bool).tolist(), scores.tolist())
    return 0.5 if value is None else float(value)


def _precision_at_k(
    truth: np.ndarray,
    scores: np.ndarray,
    fraction: float,
) -> float:
    count = max(1, min(len(truth), int(math.ceil(len(truth) * fraction))))
    order = np.argsort(-scores, kind="stable")[:count]
    return float(np.mean(truth[order]))


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
    estimate = float(np.sum(grid * posterior))
    cumulative = cast(np.ndarray, np.cumsum(posterior))
    lower = float(grid[min(np.searchsorted(cumulative, 0.025), len(grid) - 1)])
    upper = float(grid[min(np.searchsorted(cumulative, 0.975), len(grid) - 1)])
    return estimate, lower, upper


def _replication_empirical_covariates(
    scenario: Mapping[str, Any],
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if scenario.get("tier") != "semi_synthetic_development_covariates":
        return None
    if scenario.get("empirical_population_mode") != "exact_P02_rows_no_bootstrap":
        raise ValueError("semi-synthetic core must use exact P02 rows without bootstrap")
    raw_pool = scenario.get("semi_synthetic_covariate_pool")
    if not isinstance(raw_pool, list) or len(raw_pool) < 2:
        raise ValueError("semi-synthetic scenario requires a locked P02-aligned pool")
    pool = np.asarray(raw_pool, dtype=float)
    if len(pool) != sample_size:
        raise ValueError(
            "scenario sample_size must equal the exact P02 development population"
        )
    return pool[rng.permutation(sample_size)]


def _draw_latent_truth(
    *,
    scenario: Mapping[str, Any],
    empirical_base: np.ndarray | None,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    prevalence = float(scenario["prevalence"])
    if empirical_base is None:
        probabilities = np.full(sample_size, prevalence, dtype=float)
    else:
        base = empirical_base.mean(axis=1) if empirical_base.ndim == 2 else empirical_base
        centered = (base - base.mean()) / max(base.std(), 1e-12)
        strength = float(scenario.get("latent_risk_strength", 1.0))
        lower, upper = -35.0, 35.0
        for _ in range(80):
            midpoint = (lower + upper) / 2.0
            mean_probability = float(
                np.mean(1.0 / (1.0 + np.exp(-(midpoint + strength * centered))))
            )
            if mean_probability < prevalence:
                lower = midpoint
            else:
                upper = midpoint
        intercept = (lower + upper) / 2.0
        probabilities = 1.0 / (1.0 + np.exp(-(intercept + strength * centered)))
    latent = rng.random(sample_size) < probabilities
    return latent, float(probabilities.mean())


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
    observable_risk: np.ndarray,
    base_probability: float,
    selective_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.clip(
        base_probability
        + 0.25 * selective_strength * observable_risk
        + 0.75 * selective_strength * latent.astype(float),
        0.0,
        1.0,
    )
    return rng.random(len(latent)) < probability


def _content_signal(
    scenario: Mapping[str, Any],
    latent: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    *,
    base_covariates: np.ndarray | None = None,
) -> np.ndarray:
    strength = float(scenario["content_signal"])
    if base_covariates is not None:
        empirical = np.asarray(base_covariates, dtype=float)
        if len(empirical) != sample_size:
            raise ValueError("empirical content rows must equal P02-derived sample size")
        base = empirical.mean(axis=1) if empirical.ndim == 2 else empirical
    elif scenario.get("tier") == "semi_synthetic_development_covariates":
        raise ValueError("empirical semi-synthetic replication requires exact P02 covariates")
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
