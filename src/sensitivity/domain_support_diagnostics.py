"""Post-hoc diagnostics for feature drivers of P13 domain common support.

This module is diagnostic only. It reproduces the P13 balanced logistic domain-score
construction, then quantifies how individual predictors contribute to domain
separation and support loss. It does not use outcomes and does not alter any gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.sklearn_support import DropAllMissingColumns, make_logistic_regression


@dataclass(frozen=True)
class DomainScoreResult:
    """Cross-fitted domain-score diagnostics aligned with the P13 support contract."""

    support_fraction: float | None
    domain_auc: float | None
    held_scores: np.ndarray
    held_supported: np.ndarray
    crossfit_folds_used: int | None
    reason_code: str | None


def domain_score_diagnostics(
    *,
    development: pd.DataFrame,
    held_test: pd.DataFrame,
    feature_ids: list[str],
    firm_column: str,
    lower: float,
    upper: float,
    crossfit_folds: int,
    random_state: int,
) -> DomainScoreResult:
    """Reproduce P13 cross-fitted balanced domain scores and expose OOF diagnostics."""

    empty_scores = np.asarray([], dtype=float)
    empty_supported = np.asarray([], dtype=bool)
    if development.empty or held_test.empty:
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            None,
            "DOMAIN_SUPPORT_SAMPLE_UNAVAILABLE",
        )
    if not feature_ids:
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            None,
            "DOMAIN_SUPPORT_FEATURES_UNAVAILABLE",
        )

    required = {firm_column, *feature_ids}
    missing = sorted((required - set(development.columns)) | (required - set(held_test.columns)))
    if missing:
        raise ValueError(f"domain support diagnostic fields are absent: {missing}")

    combined = pd.concat(
        [development.loc[:, feature_ids], held_test.loc[:, feature_ids]],
        ignore_index=True,
    )
    labels = pd.Series(
        np.concatenate([np.zeros(len(development), dtype=int), np.ones(len(held_test), dtype=int)]),
        dtype="int64",
    )
    groups = pd.concat(
        [
            development[firm_column].astype("string"),
            held_test[firm_column].astype("string"),
        ],
        ignore_index=True,
    )
    if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            None,
            "DOMAIN_SUPPORT_GROUP_ID_UNAVAILABLE",
        )
    if labels.nunique() < 2:
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            None,
            "DOMAIN_SUPPORT_CLASSES_UNAVAILABLE",
        )

    held_groups = int(groups.loc[labels.eq(1)].nunique())
    other_groups = int(groups.loc[labels.eq(0)].nunique())
    splits = min(crossfit_folds, held_groups, other_groups)
    if splits < 2:
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            splits,
            "DOMAIN_SUPPORT_GROUPS_UNAVAILABLE",
        )

    splitter = StratifiedGroupKFold(
        n_splits=splits,
        shuffle=True,
        random_state=random_state,
    )
    out_of_fold = np.full(len(combined), np.nan, dtype=float)
    try:
        for fold_index, (train_index, validation_index) in enumerate(
            splitter.split(combined, labels, groups)
        ):
            training_labels = labels.iloc[train_index]
            if training_labels.nunique() < 2:
                return DomainScoreResult(
                    None,
                    None,
                    empty_scores,
                    empty_supported,
                    splits,
                    "DOMAIN_PROPENSITY_TRAIN_CLASS_UNAVAILABLE",
                )
            estimator = Pipeline(
                [
                    ("drop_all_missing", DropAllMissingColumns()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        make_logistic_regression(
                            penalty_kind="l2",
                            C=1.0,
                            solver="lbfgs",
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=random_state + fold_index,
                        ),
                    ),
                ]
            )
            cast(Any, estimator).fit(combined.iloc[train_index], training_labels)
            out_of_fold[validation_index] = np.asarray(
                cast(Any, estimator).predict_proba(combined.iloc[validation_index]),
                dtype=float,
            )[:, 1]
    except ValueError:
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            splits,
            "DOMAIN_PROPENSITY_UNESTIMABLE",
        )

    if not np.isfinite(out_of_fold).all():
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            splits,
            "DOMAIN_PROPENSITY_OOF_INCOMPLETE",
        )
    held_scores = out_of_fold[labels.to_numpy(dtype=bool)]
    if len(held_scores) != len(held_test):
        return DomainScoreResult(
            None,
            None,
            empty_scores,
            empty_supported,
            splits,
            "DOMAIN_PROPENSITY_OOF_INCOMPLETE",
        )

    held_supported = (held_scores >= lower) & (held_scores <= upper)
    support_fraction = float(np.mean(held_supported)) if len(held_supported) else None
    domain_auc = float(roc_auc_score(labels.to_numpy(dtype=int), out_of_fold))
    return DomainScoreResult(
        support_fraction,
        domain_auc,
        held_scores,
        held_supported,
        splits,
        None,
    )


def feature_driver_row(
    *,
    feature_id: str,
    development: pd.DataFrame,
    held_test: pd.DataFrame,
    all_feature_ids: list[str],
    baseline: DomainScoreResult,
    firm_column: str,
    lower: float,
    upper: float,
    crossfit_folds: int,
    random_state: int,
    support_fraction_minimum: float,
) -> dict[str, object]:
    """Measure one feature using distribution shift and leave-one-feature-out support."""

    if feature_id not in all_feature_ids:
        raise ValueError(f"feature={feature_id}: absent from full feature set")
    remaining = [value for value in all_feature_ids if value != feature_id]
    leave_one_out = domain_score_diagnostics(
        development=development,
        held_test=held_test,
        feature_ids=remaining,
        firm_column=firm_column,
        lower=lower,
        upper=upper,
        crossfit_folds=crossfit_folds,
        random_state=random_state,
    )
    shift = _feature_shift(
        development=development,
        held_test=held_test,
        feature_id=feature_id,
        baseline=baseline,
    )
    baseline_support = baseline.support_fraction
    loo_support = leave_one_out.support_fraction
    baseline_auc = baseline.domain_auc
    loo_auc = leave_one_out.domain_auc
    support_gain = (
        float(loo_support - baseline_support)
        if loo_support is not None and baseline_support is not None
        else None
    )
    auc_drop = (
        float(baseline_auc - loo_auc) if baseline_auc is not None and loo_auc is not None else None
    )
    restores_support = bool(
        baseline_support is not None
        and baseline_support < support_fraction_minimum
        and loo_support is not None
        and loo_support >= support_fraction_minimum
    )
    return {
        "feature_id": feature_id,
        "baseline_support_fraction": baseline_support,
        "leave_one_out_support_fraction": loo_support,
        "support_gain_when_removed": support_gain,
        "restores_locked_support_when_removed": restores_support,
        "baseline_domain_auc": baseline_auc,
        "leave_one_out_domain_auc": loo_auc,
        "domain_auc_drop_when_removed": auc_drop,
        "leave_one_out_reason_code": leave_one_out.reason_code,
        **shift,
    }


def _feature_shift(
    *,
    development: pd.DataFrame,
    held_test: pd.DataFrame,
    feature_id: str,
    baseline: DomainScoreResult,
) -> dict[str, object]:
    development_values = pd.to_numeric(development[feature_id], errors="coerce")
    held_values = pd.to_numeric(held_test[feature_id], errors="coerce")
    development_observed = development_values.dropna().to_numpy(dtype=float)
    held_observed = held_values.dropna().to_numpy(dtype=float)

    development_missing = float(development_values.isna().mean())
    held_missing = float(held_values.isna().mean())
    mean_development = _mean_or_none(development_observed)
    mean_held = _mean_or_none(held_observed)
    median_development = _median_or_none(development_observed)
    median_held = _median_or_none(held_observed)
    standardized_mean_difference = _standardized_mean_difference(
        development_observed,
        held_observed,
    )
    univariate_auc = _univariate_separation_auc(development_values, held_values)

    unsupported_excess = None
    unsupported_median = None
    supported_median = None
    if len(baseline.held_supported) == len(held_test) and len(held_test) > 0:
        robust_z = _robust_z_against_development(development_values, held_values)
        supported_values = held_values.loc[baseline.held_supported].dropna().to_numpy(dtype=float)
        unsupported_values = (
            held_values.loc[~baseline.held_supported].dropna().to_numpy(dtype=float)
        )
        supported_median = _median_or_none(supported_values)
        unsupported_median = _median_or_none(unsupported_values)
        supported_z = robust_z[baseline.held_supported]
        unsupported_z = robust_z[~baseline.held_supported]
        supported_abs_z = _finite_mean(np.abs(supported_z))
        unsupported_abs_z = _finite_mean(np.abs(unsupported_z))
        if supported_abs_z is not None and unsupported_abs_z is not None:
            unsupported_excess = float(unsupported_abs_z - supported_abs_z)

    return {
        "development_missing_rate": development_missing,
        "held_missing_rate": held_missing,
        "held_minus_development_missing_rate": float(held_missing - development_missing),
        "development_mean": mean_development,
        "held_mean": mean_held,
        "development_median": median_development,
        "held_median": median_held,
        "supported_held_median": supported_median,
        "unsupported_held_median": unsupported_median,
        "standardized_mean_difference": standardized_mean_difference,
        "absolute_standardized_mean_difference": (
            abs(standardized_mean_difference) if standardized_mean_difference is not None else None
        ),
        "univariate_domain_separation_auc": univariate_auc,
        "unsupported_minus_supported_mean_abs_robust_z": unsupported_excess,
    }


def _standardized_mean_difference(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 2 or len(second) < 2:
        return None
    first_variance = float(np.var(first, ddof=1))
    second_variance = float(np.var(second, ddof=1))
    pooled_sd = float(np.sqrt((first_variance + second_variance) / 2.0))
    if not np.isfinite(pooled_sd) or pooled_sd <= 0.0:
        return None
    return float((np.mean(second) - np.mean(first)) / pooled_sd)


def _univariate_separation_auc(
    development_values: pd.Series,
    held_values: pd.Series,
) -> float | None:
    combined = pd.concat([development_values, held_values], ignore_index=True).astype(float)
    if combined.notna().sum() < 2:
        return None
    median = combined.median(skipna=True)
    if pd.isna(median):
        return None
    scores = combined.fillna(float(median)).to_numpy(dtype=float)
    if float(np.ptp(scores)) <= 0.0:
        return 0.5
    labels = np.concatenate(
        [np.zeros(len(development_values), dtype=int), np.ones(len(held_values), dtype=int)]
    )
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def _robust_z_against_development(
    development_values: pd.Series,
    held_values: pd.Series,
) -> np.ndarray:
    development_observed = development_values.dropna().to_numpy(dtype=float)
    held_numeric = held_values.to_numpy(dtype=float)
    if len(development_observed) < 2:
        return np.full(len(held_numeric), np.nan, dtype=float)
    median = float(np.median(development_observed))
    q25, q75 = np.quantile(development_observed, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(development_observed, ddof=1))
    if not np.isfinite(scale) or scale <= 0.0:
        return np.full(len(held_numeric), np.nan, dtype=float)
    return (held_numeric - median) / scale


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if len(finite) else None


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None
