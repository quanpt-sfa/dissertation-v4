"""Compute registered post-outer domain summaries and model-block ablations."""

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
    CHANNEL_ID,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    SOURCE_ID,
    TARGET_ID,
)
from labels.service import aggregate_l1
from modeling.service import fit_fold_models


def domain_transfer(
    *,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, object]],
    domain_bindings: list[dict[str, object]],
    noninferiority_margin: float,
    support_fraction_minimum: float,
    evaluation_target_id: str,
    columns: dict[str, str],
) -> dict[str, object]:
    bindings: list[dict[str, str]] = []
    for raw in domain_bindings:
        domain_id = raw.get("domain_id")
        domain_column = raw.get("column")
        if not isinstance(domain_id, str) or not domain_id.strip():
            raise ValueError("P13 domain binding requires non-empty domain_id")
        if not isinstance(domain_column, str) or not domain_column.strip():
            raise ValueError(f"P13 domain={domain_id}: non-empty column is required")
        bindings.append(
            {
                "domain_id": domain_id.strip(),
                "column": domain_column.strip(),
            }
        )
    if not bindings:
        return {
            "status": "SKIPPED",
            "reason_code": "DOMAIN_BINDINGS_UNAVAILABLE",
            "domains": [],
            "robust_scenario_fraction": None,
            "leave_one_domain_out_refit_executed": False,
        }
    binding_keys = [(item["domain_id"], item["column"]) for item in bindings]
    if len(binding_keys) != len(set(binding_keys)):
        raise ValueError("P13 domain bindings must be unique by domain_id and column")

    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    content = [
        str(item["feature_id"]) for item in feature_registry if item.get("role") == "content"
    ]
    observability = [
        str(item["feature_id"]) for item in feature_registry if item.get("role") == "observability"
    ]
    if not content or not observability:
        return {
            "status": "SKIPPED",
            "reason_code": "DOMAIN_REFIT_FEATURE_BLOCKS_UNAVAILABLE",
            "domains": [],
            "robust_scenario_fraction": None,
            "leave_one_domain_out_refit_executed": False,
        }

    predictor_ids = set(content) | set(observability)
    domain_columns = {item["column"] for item in bindings}
    overlap = sorted(domain_columns & predictor_ids)
    if overlap:
        raise ValueError(
            "P13 domain metadata must not enter predictor blocks: "
            f"{overlap}"
        )
    missing_domain_columns = sorted(domain_columns - set(feature_panel.columns))
    if missing_domain_columns:
        return {
            "status": "SKIPPED",
            "reason_code": "DOMAIN_COLUMN_UNAVAILABLE",
            "domains": [],
            "robust_scenario_fraction": None,
            "leave_one_domain_out_refit_executed": False,
            "configured_bindings": bindings,
            "missing_domain_columns": missing_domain_columns,
        }

    observed_outcomes = select_observed_target_outcomes(
        outcomes,
        target_id=evaluation_target_id,
        columns=columns,
        context="P13 domain transfer",
    )
    data_base = feature_panel.merge(
        observed_outcomes,
        on=[firm, year],
        how="inner",
        validate="1:1",
    )
    outer_years = sorted(int(value) for value in predictions[year].unique())
    rows: list[dict[str, object]] = []
    robust: list[bool] = []
    for binding in bindings:
        domain_column = binding["column"]
        level_values = cast(list[object], data_base[domain_column].dropna().tolist())
        for level in sorted(set(level_values), key=str):
            for outer_year in outer_years:
                train: pd.DataFrame = data_base.loc[
                    (data_base[year] < outer_year) & (data_base[domain_column] != level)
                ].dropna(subset=[outcome])
                test: pd.DataFrame = data_base.loc[
                    (data_base[year] == outer_year) & (data_base[domain_column] == level)
                ].dropna(subset=[outcome])
                if train.empty or test.empty or train[outcome].nunique() < 2:
                    continue
                support = _rectangular_common_support(
                    train, test, sorted(set(observability + content))
                )
                reference_scores = _domain_refit_predict(train, test, observability, outcome)
                candidate_scores = _domain_refit_predict(
                    train, test, sorted(set(observability + content)), outcome
                )
                truth = test[outcome].astype(bool).tolist()
                reference_ap = average_precision(truth, reference_scores.tolist())
                candidate_ap = average_precision(truth, candidate_scores.tolist())
                passed = bool(
                    support >= support_fraction_minimum
                    and candidate_ap is not None
                    and reference_ap is not None
                    and candidate_ap >= reference_ap * (1.0 - noninferiority_margin)
                )
                robust.append(passed)
                rows.append(
                    {
                        "domain_id": binding["domain_id"],
                        "domain_column": domain_column,
                        "level": str(level),
                        OUTER_FOLD: str(outer_year),
                        "evaluation_target_id": evaluation_target_id,
                        "train_rows_other_domains": len(train),
                        "test_rows_held_domain": len(test),
                        "positives": int(test[outcome].sum()),
                        "common_support_fraction": support,
                        "candidate_ap": candidate_ap,
                        "reference_ap": reference_ap,
                        "noninferior": passed,
                        "refit_executed": True,
                    }
                )
    domain_count = len({(str(item["domain_id"]), str(item["level"])) for item in rows})
    return {
        "status": "PASS" if robust and domain_count >= 2 else "SKIPPED",
        "reason_code": None if robust and domain_count >= 2 else "INSUFFICIENT_DOMAIN_REFITS",
        "evaluation_target_id": evaluation_target_id,
        "configured_bindings": bindings,
        "domains": rows,
        "robust_scenario_fraction": sum(robust) / len(robust) if robust else None,
        "leave_one_domain_out_refit_executed": bool(robust and domain_count >= 2),
    }


def _domain_refit_predict(
    train: pd.DataFrame, test: pd.DataFrame, feature_ids: list[str], outcome: str
) -> np.ndarray:
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )
    cast(Any, estimator).fit(train[feature_ids], train[outcome].astype(int))
    return np.asarray(cast(Any, estimator).predict_proba(test[feature_ids]), dtype=float)[:, 1]


def _rectangular_common_support(
    train: pd.DataFrame, test: pd.DataFrame, feature_ids: list[str]
) -> float:
    supported = np.ones(len(test), dtype=bool)
    for feature_id in feature_ids:
        train_values = pd.to_numeric(train[feature_id], errors="coerce")
        test_values = pd.to_numeric(test[feature_id], errors="coerce").to_numpy(dtype=float)
        finite = train_values.dropna().to_numpy(dtype=float)
        if len(finite) == 0:
            return 0.0
        supported &= (
            np.isfinite(test_values)
            & (test_values >= np.min(finite))
            & (test_values <= np.max(finite))
        )
    return float(np.mean(supported)) if len(supported) else 0.0


def ablation_summary(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        comparisons = evaluation.get("comparisons")
        if not isinstance(comparisons, list):
            continue
        for comparison in cast(list[Any], comparisons):
            if isinstance(comparison, dict):
                rows.append(
                    {
                        OUTER_FOLD: evaluation.get(OUTER_FOLD),
                        "ablation": "remove_content_block",
                        **comparison,
                    }
                )
    return rows


def hierarchical_pi_status(capability: dict[str, object]) -> dict[str, object]:
    available = capability.get("status") == "AVAILABLE" and capability.get("pilot_executed") is True
    return {
        "status": "SKIPPED" if not available else "EMPIRICALLY_PENDING",
        "reason_code": None if available else "CAPABILITY_UNAVAILABLE",
        "role": "sensitivity_only",
        "entered_primary_selection": False,
        "estimated_pi_metrics_allowed": available,
        "fixed_pi_bias_or_rmse_reported": False,
    }


def source_sensitivity_summary(
    evidence: pd.DataFrame,
    lag_decomposition: dict[str, Any],
    columns: dict[str, str],
) -> dict[str, object]:
    """Register source-set and lag sensitivity capability without redefining labels."""
    source = columns[SOURCE_ID]
    channel = columns[CHANNEL_ID]
    sources = sorted(str(value) for value in evidence[source].dropna().unique())
    channels = sorted(str(value) for value in evidence[channel].dropna().unique())
    lag_records = lag_decomposition.get("records")
    lag_count = len(cast(list[Any], lag_records)) if isinstance(lag_records, list) else 0
    return {
        "status": "AVAILABLE" if len(sources) >= 2 else "SKIPPED",
        "reason_code": None if len(sources) >= 2 else "INSUFFICIENT_SOURCES",
        "source_ids": sources,
        "channel_ids": channels,
        "leave_one_source_out_registered": len(sources) >= 2,
        "strict_channel_holdout_registered": len(channels) >= 2,
        "event_deduplication_applied": True,
        "lag_record_count": lag_count,
        "horizon_sensitivities_months": [12, 24],
        "reuses_locked_evidence_ledger": True,
    }


def select_observed_target_outcomes(
    outcomes: pd.DataFrame,
    *,
    target_id: str,
    columns: dict[str, str],
    context: str,
) -> pd.DataFrame:
    """Return one observed sealed outcome per firm-year for a single registered target."""
    if not target_id:
        raise ValueError(f"{context}: target_id is required")
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    target = columns[TARGET_ID]
    outcome = columns[OUTCOME]
    required = {firm, year, target, outcome}
    if not required.issubset(outcomes.columns):
        raise ValueError(f"{context}: target-aware sealed outcome contract is incomplete")
    selected = outcomes.loc[
        outcomes[target].astype("string").eq(target_id) & outcomes[outcome].notna(),
        [firm, year, outcome],
    ].copy()
    if selected.empty:
        raise RuntimeError(f"{context}: target={target_id} has no observed sealed outcomes")
    duplicates = selected.loc[selected.duplicated([firm, year], keep=False), [firm, year]]
    if not duplicates.empty:
        sample = duplicates.drop_duplicates().head(5).to_dict(orient="records")
        raise RuntimeError(
            f"{context}: target={target_id} outcomes are not unique by firm-year: {sample}"
        )
    return selected


def source_exclusion_refits(
    *,
    matrices: dict[str, Any],
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, Any]],
    weights_by_fold: dict[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    outer_folds: list[str],
    learner_ids: list[str],
    learner_settings: dict[str, Any],
    learner_search_spaces: dict[str, Any],
    maximum_valid_configurations: int,
    evaluation_target_id: str,
    columns: dict[str, str],
    seed_by_fold_and_exclusion: dict[tuple[str, str], int],
) -> dict[str, object]:
    """Actually refit Track A after each source and channel exclusion."""
    raw_rows = matrices.get("rows")
    expected_sources = matrices.get("expected_sources")
    if not isinstance(raw_rows, list) or not isinstance(expected_sources, dict):
        raise ValueError("source sensitivity requires source-channel matrices")
    observed_outcomes = select_observed_target_outcomes(
        outcomes,
        target_id=evaluation_target_id,
        columns=columns,
        context="P13 source exclusion",
    )
    source_channels = {
        str(key): str(value) for key, value in cast(dict[object, object], expected_sources).items()
    }
    exclusions: dict[str, set[str]] = {
        f"source:{source}": {source} for source in sorted(source_channels)
    }
    for channel in sorted(set(source_channels.values())):
        exclusions[f"channel:{channel}"] = {
            source
            for source, source_channel in source_channels.items()
            if source_channel == channel
        }
    rows: list[dict[str, object]] = []
    for exclusion_id, excluded_sources in exclusions.items():
        training_target_id = f"L1_without_{exclusion_id}"
        labels = _alternative_l1_labels(
            matrix_rows=cast(list[object], raw_rows),
            excluded_sources=excluded_sources,
            target_id=training_target_id,
            columns=columns,
        )
        for fold_id in outer_folds:
            weights = weights_by_fold.get(fold_id)
            if weights is None:
                raise ValueError(f"fold={fold_id}: sensitivity weights required")
            fit = fit_fold_models(
                feature_panel=feature_panel,
                feature_registry=feature_registry,
                label_inputs=labels,
                weights=weights,
                outer_year=int(fold_id),
                learner_ids=learner_ids,
                learner_settings=learner_settings,
                target_id=training_target_id,
                measurement_id=training_target_id,
                columns=columns,
                random_state=seed_by_fold_and_exclusion[(fold_id, exclusion_id)],
                track_id="source_sensitivity",
                learner_search_spaces=learner_search_spaces,
                maximum_valid_configurations=maximum_valid_configurations,
            )
            evaluated = fit.outer_predictions.merge(
                observed_outcomes,
                on=[columns[FIRM_ID], columns[FISCAL_YEAR]],
                how="inner",
                validate="m:1",
            )
            for model_id, frame in evaluated.groupby(columns[LEARNER_ID], sort=True):
                truth = frame[columns[OUTCOME]].astype(bool).tolist()
                scores = frame[columns[PREDICTION]].astype(float).tolist()
                rows.append(
                    {
                        "exclusion_id": exclusion_id,
                        "excluded_source_ids": sorted(excluded_sources),
                        OUTER_FOLD: fold_id,
                        "model_id": str(model_id),
                        "training_target_id": training_target_id,
                        "evaluation_target_id": evaluation_target_id,
                        "fit_status": fit.models["status"],
                        "rows": len(frame),
                        "positives": int(frame[columns[OUTCOME]].sum()),
                        "average_precision": average_precision(truth, scores),
                        "outer_outcomes_used_in_fit": False,
                        "tuning_scope": "development_history_only",
                        "tuning_budget_maximum": maximum_valid_configurations,
                    }
                )
    return {
        "status": "PASS" if rows else "SKIPPED",
        "reason_code": None if rows else "INSUFFICIENT_SOURCE_REFITS",
        "evaluation_target_id": evaluation_target_id,
        "refit_executed": bool(rows),
        "results": rows,
    }


def _alternative_l1_labels(
    *,
    matrix_rows: list[object],
    excluded_sources: set[str],
    target_id: str,
    columns: dict[str, str],
) -> pd.DataFrame:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    target = columns[TARGET_ID]
    outcome = columns[OUTCOME]
    rows: list[dict[str, object]] = []
    for raw in matrix_rows:
        if not isinstance(raw, dict):
            continue
        row = cast(dict[str, Any], raw)
        source_outcomes = row.get("source_outcomes")
        if not isinstance(source_outcomes, dict):
            continue
        remaining = {
            str(source): cast(bool | None, value)
            for source, value in cast(dict[object, object], source_outcomes).items()
            if str(source) not in excluded_sources
        }
        value = aggregate_l1(remaining) if row.get(MATURE) is True else None
        rows.append(
            {
                firm: str(row[FIRM_ID]),
                year: int(row[FISCAL_YEAR]),
                target: target_id,
                outcome: value,
            }
        )
    frame = pd.DataFrame(rows, columns=[firm, year, target, outcome])
    return frame.astype(
        {
            firm: "string",
            year: "int16",
            target: "string",
            outcome: "boolean",
        }
    )


def censoring_sensitivity_summary(
    diagnostics: list[dict[str, Any]], censoring_registry: list[dict[str, Any]]
) -> dict[str, object]:
    """Allow IPCW sensitivity only for folds whose development diagnostics pass."""
    eligible_folds = [
        str(item.get(OUTER_FOLD))
        for item in diagnostics
        if item.get("ipw_diagnostics_pass") is True
        and item.get("fit_scope") == "development_history"
        and item.get("outer_rows_used_in_fit") == 0
    ]
    prospective = sum(
        item.get("classification") == "prospective_immature" for item in censoring_registry
    )
    return {
        "status": "SKIPPED",
        "reason_code": "CENSORING_TIME_OR_EXIT_SOURCE_UNAVAILABLE"
        if eligible_folds
        else "IPCW_DIAGNOSTICS_UNAVAILABLE",
        "eligible_outer_folds": eligible_folds,
        "role": "sensitivity_only",
        "rerun_executed": False,
        "prospective_immature_count": prospective,
        "immature_or_exit_assigned_negative": False,
    }
