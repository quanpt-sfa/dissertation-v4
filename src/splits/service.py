"""Rolling-origin split construction and development-history-only weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    OUTER_FOLD,
    WEIGHT,
)


@dataclass(frozen=True)
class SplitWeightResult:
    splits: list[dict[str, Any]]
    channel_splits: list[dict[str, Any]]
    weights: dict[str, pd.DataFrame]
    weight_diagnostics: dict[str, dict[str, Any]]


def build_splits_and_weights(
    *,
    feature_panel: pd.DataFrame,
    risk_sets: pd.DataFrame,
    matrices: dict[str, Any],
    observability: dict[str, Any],
    outer_years: list[int],
    fold_eligibility: list[dict[str, Any]],
    support_bounds: tuple[float, float],
    support_fraction_minimum: float,
    ess_fraction_minimum: float,
    ess_absolute_minimum: float,
    columns: dict[str, str],
) -> SplitWeightResult:
    """Build strict rolling splits; no outer-year row enters a weight fit."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    eligible = columns[ELIGIBLE]
    mature = columns[MATURE]
    needed = {firm, year}
    if not needed.issubset(feature_panel.columns) or not {
        firm,
        year,
        eligible,
        mature,
    }.issubset(risk_sets.columns):
        raise ValueError("P09 inputs do not satisfy firm-year contracts")
    risk = risk_sets.loc[risk_sets[eligible] & risk_sets[mature], [firm, year]].copy()
    eligible_panel = feature_panel.merge(risk, on=[firm, year], how="inner", validate="1:1")
    verification_channels = _verification_channels(observability)
    verified_by_key = _verified_keys(matrices, verification_channels)
    results: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    splits: list[dict[str, Any]] = []
    channel_splits: list[dict[str, Any]] = []
    role_by_fold = {
        str(item.get(OUTER_FOLD)): item.get("assigned_role") for item in fold_eligibility
    }
    for outer_year in outer_years:
        fold_id = str(outer_year)
        train = eligible_panel.loc[eligible_panel[year] < outer_year, [firm, year]].copy()
        test = eligible_panel.loc[eligible_panel[year] == outer_year, [firm, year]].copy()
        if not train.empty and int(train[year].max()) >= outer_year:
            raise ValueError(f"fold={fold_id}: temporal leakage detected")
        observed = [
            (str(row[firm]), int(row[year])) in verified_by_key
            for row in train.to_dict(orient="records")
        ]
        has_observed_verification = bool(verification_channels and any(observed))
        proposed_weights, propensities = _stabilized_year_weights(train, year, observed)
        proposed_ess = _effective_sample_size(proposed_weights)
        support_fraction = (
            sum(support_bounds[0] <= value <= support_bounds[1] for value in propensities)
            / len(propensities)
            if propensities
            else None
        )
        required_ess = max(ess_absolute_minimum, ess_fraction_minimum * len(train))
        ipw_diagnostics_pass = bool(
            has_observed_verification
            and proposed_ess >= required_ess
            and support_fraction is not None
            and support_fraction >= support_fraction_minimum
        )
        applied_weights = proposed_weights if ipw_diagnostics_pass else [1.0] * len(train)
        weight_method = "stabilized_year_ipw" if ipw_diagnostics_pass else "unweighted"
        reason_code = (
            None
            if ipw_diagnostics_pass
            else "NO_OBSERVED_VERIFICATION"
            if not has_observed_verification
            else "WEIGHT_DIAGNOSTICS_FAILED"
        )
        weights = train.copy()
        weights[columns[OUTER_FOLD]] = fold_id
        weights[columns[WEIGHT]] = applied_weights
        weights[firm] = weights[firm].astype("string")
        weights[year] = weights[year].astype("int16")
        weights[columns[OUTER_FOLD]] = weights[columns[OUTER_FOLD]].astype("string")
        weights[columns[WEIGHT]] = weights[columns[WEIGHT]].astype("float64")
        results[fold_id] = weights
        fit_max_year = int(train[year].max()) if not train.empty else None
        splits.append(
            {
                OUTER_FOLD: fold_id,
                "development_years": sorted(int(value) for value in train[year].unique()),
                "outer_test_year": outer_year,
                "development_rows": len(train),
                "outer_rows": len(test),
                "assigned_fold_role": role_by_fold.get(fold_id),
                "weight_fit_max_year": fit_max_year,
                "weight_method": weight_method,
                "weight_reason_code": reason_code,
                "outer_rows_used_in_weight_fit": 0,
            }
        )
        channel_splits.append(
            {
                OUTER_FOLD: fold_id,
                "verification_channels": sorted(verification_channels),
                "fit_scope": "development_history",
                "outer_outcomes_accessed": False,
            }
        )
        diagnostics[fold_id] = {
            OUTER_FOLD: fold_id,
            "fit_scope": "development_history",
            "development_years": sorted(int(value) for value in train[year].unique()),
            "outer_rows_used_in_fit": 0,
            "requested_weight_method": "stabilized_year_ipw",
            "applied_weight_method": weight_method,
            "verification_observed": has_observed_verification,
            "proposed_ipw_ess": proposed_ess,
            "required_ess": required_ess,
            "propensity_support_bounds": list(support_bounds),
            "propensity_support_fraction": support_fraction,
            "required_support_fraction": support_fraction_minimum,
            "ipw_diagnostics_pass": ipw_diagnostics_pass,
            "analytical_use_allowed": True,
            "reason_code": reason_code,
            "estimand": "target_population" if ipw_diagnostics_pass else "unweighted_mature_cohort",
        }
    return SplitWeightResult(splits, channel_splits, results, diagnostics)


def _stabilized_year_weights(
    train: pd.DataFrame, year: str, observed: list[bool]
) -> tuple[list[float], list[float]]:
    if not observed or not any(observed):
        return [1.0] * len(train), []
    overall = (sum(observed) + 1.0) / (len(observed) + 2.0)
    counts: dict[int, list[int]] = {}
    years = [int(value) for value in train[year].tolist()]
    for value, flag in zip(years, observed, strict=True):
        bucket = counts.setdefault(value, [0, 0])
        bucket[0] += int(flag)
        bucket[1] += 1
    propensity = {value: (count[0] + 1.0) / (count[1] + 2.0) for value, count in counts.items()}
    propensities = [propensity[value] for value in years]
    weights = [
        overall / propensity[value] if flag else (1.0 - overall) / (1.0 - propensity[value])
        for value, flag in zip(years, observed, strict=True)
    ]
    return weights, propensities


def _effective_sample_size(weights: list[float]) -> float:
    if not weights:
        return 0.0
    denominator = sum(value * value for value in weights)
    return sum(weights) ** 2 / denominator if denominator > 0 else 0.0


def _verification_channels(observability: dict[str, Any]) -> set[str]:
    raw = observability.get("channels")
    if not isinstance(raw, dict):
        raise ValueError("observability channels are required")
    raw = cast(dict[str, Any], raw)
    return {
        str(channel_id)
        for channel_id, metadata in raw.items()
        if isinstance(metadata, dict)
        and cast(dict[str, Any], metadata).get("verification_classification")
        == "observed_verification"
    }


def _verified_keys(matrices: dict[str, Any], channels: set[str]) -> set[tuple[str, int]]:
    raw = matrices.get("rows")
    if not isinstance(raw, list):
        raise ValueError("source-channel matrix rows are required")
    raw = cast(list[Any], raw)
    keys: set[tuple[str, int]] = set()
    for raw_row in raw:
        row = cast(dict[str, Any], raw_row) if isinstance(raw_row, dict) else {}
        if not isinstance(row.get("channel_outcomes"), dict):
            continue
        outcomes = row["channel_outcomes"]
        outcomes = cast(dict[str, Any], outcomes)
        if any(outcomes.get(channel) is not None for channel in channels):
            keys.add((str(row[FIRM_ID]), int(row[FISCAL_YEAR])))
    return keys
