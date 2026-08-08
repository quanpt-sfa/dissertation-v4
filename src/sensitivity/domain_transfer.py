"""Candidate-specific leave-one-domain-out transport robustness for P13."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.metrics import average_precision
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    WEIGHT,
)
from modeling.service import feature_groups, fit_fold_models
from sensitivity.service import select_observed_target_outcomes


def candidate_domain_transfer(
    *,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, Any]],
    domain_bindings: list[dict[str, object]],
    learner_ids: list[str],
    learner_settings: dict[str, Any],
    learner_search_spaces: dict[str, Any],
    maximum_valid_configurations: int,
    noninferiority_margin: float,
    support_fraction_minimum: float,
    support_bounds: tuple[float, float],
    minimum_domains: int,
    evaluation_target_id: str,
    columns: dict[str, str],
    seed_by_scenario: dict[tuple[str, str, str], int],
) -> dict[str, object]:
    """Refit each confirmatory learner outside a held board and test on that board.

    The held domain is excluded from model fitting and temporal tuning. Domain
    support is evaluated without outcomes by a balanced logistic domain-membership
    model. The support fraction is the share of held-domain test rows whose
    estimated held-domain propensity lies inside the locked support bounds.
    Domain refits intentionally use unit weights so no weighting model fitted on
    held-domain development rows can leak information into the transport refit.
    """

    bindings = _normalize_bindings(domain_bindings)
    if not bindings:
        return _skipped("DOMAIN_BINDINGS_UNAVAILABLE")

    lower, upper = support_bounds
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("P13 domain support bounds must satisfy 0 <= lower < upper <= 1")
    if not 0.0 <= support_fraction_minimum <= 1.0:
        raise ValueError("P13 domain support minimum must be in [0, 1]")
    if minimum_domains < 2:
        raise ValueError("P13 domain transfer requires at least two domains")

    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    weight_column = columns[WEIGHT]

    groups = feature_groups(feature_registry)
    reference_features = groups.get("observability_only", [])
    full_features = groups.get("full", [])
    if not reference_features or not full_features:
        return _skipped("DOMAIN_REFIT_FEATURE_BLOCKS_UNAVAILABLE")

    predictor_ids = set(full_features)
    domain_columns = {binding["column"] for binding in bindings}
    overlap = sorted(domain_columns & predictor_ids)
    if overlap:
        raise ValueError(f"P13 domain metadata must not enter predictor blocks: {overlap}")
    missing_domain_columns = sorted(domain_columns - set(feature_panel.columns))
    if missing_domain_columns:
        result = _skipped("DOMAIN_COLUMN_UNAVAILABLE")
        result["configured_bindings"] = bindings
        result["missing_domain_columns"] = missing_domain_columns
        return result

    observed_outcomes = select_observed_target_outcomes(
        outcomes,
        target_id=evaluation_target_id,
        columns=columns,
        context="P13 domain transfer",
    )
    outer_years = sorted(int(value) for value in predictions[year].dropna().unique())
    if not outer_years:
        return _skipped("DOMAIN_OUTER_FOLDS_UNAVAILABLE")

    all_rows: list[dict[str, object]] = []
    expected_scenarios: list[tuple[str, str, int]] = []
    domain_levels: set[tuple[str, str]] = set()

    for binding in bindings:
        domain_id = binding["domain_id"]
        domain_column = binding["column"]
        confirmatory_rows = feature_panel.loc[
            feature_panel[year].astype(int).isin(outer_years) & feature_panel[domain_column].notna()
        ]
        levels = sorted({str(value) for value in confirmatory_rows[domain_column].tolist()})
        for level in levels:
            domain_levels.add((domain_id, level))
            for outer_year in outer_years:
                held_test = feature_panel.loc[
                    (feature_panel[year].astype(int) == outer_year)
                    & feature_panel[domain_column].astype("string").eq(level)
                ].copy()
                if held_test.empty:
                    continue
                expected_scenarios.append((domain_id, level, outer_year))
                development = feature_panel.loc[
                    (feature_panel[year].astype(int) < outer_year)
                    & feature_panel[domain_column].notna()
                    & ~feature_panel[domain_column].astype("string").eq(level)
                ].copy()
                support = _held_test_propensity_support(
                    development=development,
                    held_test=held_test,
                    feature_ids=full_features,
                    lower=lower,
                    upper=upper,
                )
                test_outcomes = held_test.loc[:, [firm, year]].merge(
                    observed_outcomes,
                    on=[firm, year],
                    how="left",
                    validate="one_to_one",
                )
                observed_test = test_outcomes.dropna(subset=[outcome]).copy()
                positives = (
                    int(observed_test[outcome].astype(bool).sum()) if len(observed_test) else 0
                )
                negatives = int(len(observed_test) - positives)

                fit = None
                fit_reason: str | None = None
                if development.empty:
                    fit_reason = "NO_OTHER_DOMAIN_DEVELOPMENT_ROWS"
                else:
                    restricted_panel = pd.concat([development, held_test], ignore_index=True)
                    unit_weights = development.loc[:, [firm, year]].copy()
                    unit_weights[weight_column] = 1.0
                    seed = seed_by_scenario.get((str(outer_year), domain_id, level))
                    if seed is None:
                        raise ValueError(
                            f"P13 domain seed missing for fold={outer_year}, domain={domain_id}, level={level}"
                        )
                    fit = fit_fold_models(
                        feature_panel=restricted_panel,
                        feature_registry=feature_registry,
                        label_inputs=outcomes,
                        weights=unit_weights,
                        outer_year=outer_year,
                        learner_ids=learner_ids,
                        learner_settings=learner_settings,
                        target_id=evaluation_target_id,
                        measurement_id=evaluation_target_id,
                        columns=columns,
                        random_state=seed,
                        track_id="domain_transfer",
                        learner_search_spaces=learner_search_spaces,
                        maximum_valid_configurations=maximum_valid_configurations,
                        required_feature_groups=["observability_only", "full"],
                    )

                for learner_id in learner_ids:
                    candidate_id = f"track_l1:{learner_id}:full"
                    reference_id = f"track_l1:{learner_id}:observability_only"
                    candidate_ap: float | None = None
                    reference_ap: float | None = None
                    prediction_rows = 0
                    reason = fit_reason
                    if fit is not None:
                        paired = _paired_domain_predictions(
                            fit.outer_predictions,
                            observed_outcomes,
                            candidate_model_id=f"domain_transfer:{learner_id}:full",
                            reference_model_id=f"domain_transfer:{learner_id}:observability_only",
                            columns=columns,
                        )
                        prediction_rows = len(paired)
                        if paired.empty:
                            reason = "DOMAIN_REFIT_PREDICTIONS_UNAVAILABLE"
                        else:
                            truth = paired[outcome].astype(bool).tolist()
                            candidate_ap = average_precision(
                                truth, paired["candidate_prediction"].astype(float).tolist()
                            )
                            reference_ap = average_precision(
                                truth, paired["reference_prediction"].astype(float).tolist()
                            )
                            if positives == 0 or negatives == 0:
                                reason = "HELD_DOMAIN_TEST_CLASS_UNAVAILABLE"
                            elif candidate_ap is None or reference_ap is None:
                                reason = "HELD_DOMAIN_AP_UNAVAILABLE"
                            else:
                                reason = None

                    support_fraction = support.get("support_fraction")
                    estimable = bool(
                        reason is None
                        and isinstance(support_fraction, (int, float))
                        and candidate_ap is not None
                        and reference_ap is not None
                    )
                    support_pass = bool(
                        estimable and float(support_fraction) >= support_fraction_minimum
                    )
                    noninferior = bool(
                        support_pass
                        and candidate_ap is not None
                        and reference_ap is not None
                        and candidate_ap >= reference_ap * (1.0 - noninferiority_margin)
                    )
                    all_rows.append(
                        {
                            "domain_id": domain_id,
                            "domain_column": domain_column,
                            "level": level,
                            OUTER_FOLD: str(outer_year),
                            "evaluation_target_id": evaluation_target_id,
                            "candidate": candidate_id,
                            "reference": reference_id,
                            LEARNER_ID: learner_id,
                            "train_rows_other_domains": int(len(development)),
                            "test_rows_held_domain": int(len(held_test)),
                            "observed_test_rows": int(len(observed_test)),
                            "positives": positives,
                            "negatives": negatives,
                            "prediction_rows": prediction_rows,
                            "common_support_fraction": support_fraction,
                            "support_method": support.get("method"),
                            "support_bounds": [lower, upper],
                            "candidate_ap": candidate_ap,
                            "reference_ap": reference_ap,
                            "estimable": estimable,
                            "support_pass": support_pass,
                            "noninferior": noninferior,
                            "reason_code": reason or support.get("reason_code"),
                            "refit_executed": fit is not None,
                            "tuning_scope": "development_other_domains_only",
                            "weighting_mode": "unit_weight_transport_refit",
                            "outer_outcomes_used_in_fit": False,
                        }
                    )

    candidate_results: list[dict[str, object]] = []
    expected_count = len(expected_scenarios)
    for learner_id in learner_ids:
        candidate_id = f"track_l1:{learner_id}:full"
        rows = [row for row in all_rows if row["candidate"] == candidate_id]
        complete = bool(
            expected_count > 0
            and len(domain_levels) >= minimum_domains
            and len(rows) == expected_count
            and all(row["estimable"] is True for row in rows)
        )
        robust_fraction = (
            sum(row["noninferior"] is True for row in rows) / expected_count
            if expected_count
            else None
        )
        candidate_results.append(
            {
                "candidate": candidate_id,
                "reference": f"track_l1:{learner_id}:observability_only",
                "expected_scenarios": expected_count,
                "evaluated_scenarios": len(rows),
                "evidence_complete": complete,
                "robust_scenario_fraction": robust_fraction,
                "estimable_scenarios": sum(row["estimable"] is True for row in rows),
                "support_pass_scenarios": sum(row["support_pass"] is True for row in rows),
                "noninferior_scenarios": sum(row["noninferior"] is True for row in rows),
            }
        )

    all_complete = bool(candidate_results) and all(
        item["evidence_complete"] is True for item in candidate_results
    )
    summary_fractions = [
        float(item["robust_scenario_fraction"])
        for item in candidate_results
        if isinstance(item.get("robust_scenario_fraction"), (int, float))
    ]
    return {
        "status": "PASS" if all_complete else "SKIPPED",
        "reason_code": None if all_complete else "INCOMPLETE_DOMAIN_TRANSFER_GRID",
        "evaluation_target_id": evaluation_target_id,
        "configured_bindings": bindings,
        "domain_level_count": len(domain_levels),
        "minimum_domains": minimum_domains,
        "expected_scenario_count": expected_count,
        "scenario_unit": "domain_level_x_outer_fold",
        "support_method": "balanced_logistic_domain_propensity_held_test_fraction",
        "support_bounds": [lower, upper],
        "support_fraction_minimum": support_fraction_minimum,
        "support_uses_outcomes": False,
        "weighting_mode": "unit_weight_transport_refit",
        "candidate_results": candidate_results,
        "domains": all_rows,
        "robust_scenario_fraction": min(summary_fractions)
        if all_complete and summary_fractions
        else None,
        "leave_one_domain_out_refit_executed": any(
            row["refit_executed"] is True for row in all_rows
        ),
    }


def _normalize_bindings(raw_bindings: list[dict[str, object]]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for raw in raw_bindings:
        domain_id = raw.get("domain_id")
        column = raw.get("column")
        if not isinstance(domain_id, str) or not domain_id.strip():
            raise ValueError("P13 domain binding requires non-empty domain_id")
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"P13 domain={domain_id}: non-empty column is required")
        bindings.append({"domain_id": domain_id.strip(), "column": column.strip()})
    keys = [(item["domain_id"], item["column"]) for item in bindings]
    if len(keys) != len(set(keys)):
        raise ValueError("P13 domain bindings must be unique by domain_id and column")
    return bindings


def _held_test_propensity_support(
    *,
    development: pd.DataFrame,
    held_test: pd.DataFrame,
    feature_ids: list[str],
    lower: float,
    upper: float,
) -> dict[str, object]:
    method = "balanced_logistic_domain_propensity_held_test_fraction"
    if development.empty or held_test.empty:
        return {
            "method": method,
            "support_fraction": None,
            "reason_code": "DOMAIN_SUPPORT_SAMPLE_UNAVAILABLE",
        }
    missing = sorted(
        (set(feature_ids) - set(development.columns)) | (set(feature_ids) - set(held_test.columns))
    )
    if missing:
        raise ValueError(f"P13 support features are absent: {missing}")
    combined = pd.concat(
        [
            development.loc[:, feature_ids].assign(_held_domain=0),
            held_test.loc[:, feature_ids].assign(_held_domain=1),
        ],
        ignore_index=True,
    )
    labels = combined.pop("_held_domain").astype(int)
    if labels.nunique() < 2:
        return {
            "method": method,
            "support_fraction": None,
            "reason_code": "DOMAIN_SUPPORT_CLASSES_UNAVAILABLE",
        }
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    try:
        cast(Any, estimator).fit(combined, labels)
        held_probabilities = np.asarray(
            cast(Any, estimator).predict_proba(held_test.loc[:, feature_ids]), dtype=float
        )[:, 1]
    except ValueError:
        return {
            "method": method,
            "support_fraction": None,
            "reason_code": "DOMAIN_PROPENSITY_UNESTIMABLE",
        }
    supported = (
        np.isfinite(held_probabilities)
        & (held_probabilities >= lower)
        & (held_probabilities <= upper)
    )
    return {
        "method": method,
        "support_fraction": float(np.mean(supported)) if len(supported) else None,
        "reason_code": None,
        "held_propensity_min": float(np.min(held_probabilities)),
        "held_propensity_max": float(np.max(held_probabilities)),
        "held_propensity_mean": float(np.mean(held_probabilities)),
    }


def _paired_domain_predictions(
    predictions: pd.DataFrame,
    observed_outcomes: pd.DataFrame,
    *,
    candidate_model_id: str,
    reference_model_id: str,
    columns: dict[str, str],
) -> pd.DataFrame:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    prediction = columns[PREDICTION]
    outcome = columns[OUTCOME]
    candidate = predictions.loc[
        predictions[learner].astype("string").eq(candidate_model_id),
        [firm, year, prediction],
    ].rename(columns={prediction: "candidate_prediction"})
    reference = predictions.loc[
        predictions[learner].astype("string").eq(reference_model_id),
        [firm, year, prediction],
    ].rename(columns={prediction: "reference_prediction"})
    if candidate.empty or reference.empty:
        return pd.DataFrame(
            columns=[firm, year, "candidate_prediction", "reference_prediction", outcome]
        )
    paired = candidate.merge(reference, on=[firm, year], how="inner", validate="one_to_one")
    return paired.merge(observed_outcomes, on=[firm, year], how="inner", validate="one_to_one")


def _skipped(reason_code: str) -> dict[str, object]:
    return {
        "status": "SKIPPED",
        "reason_code": reason_code,
        "domains": [],
        "candidate_results": [],
        "robust_scenario_fraction": None,
        "leave_one_domain_out_refit_executed": False,
    }
