"""Fit registered learners with temporal OOF preprocessing and frozen models."""

from __future__ import annotations

import base64
import hashlib
import itertools
import math
import pickle
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.metrics import average_precision
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MEASUREMENT_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    TARGET_ID,
    TARGET_VALUE,
    WEIGHT,
)


@dataclass(frozen=True)
class ModelFitResult:
    models: dict[str, object]
    oof_predictions: pd.DataFrame
    outer_predictions: pd.DataFrame


def fit_anchor_pu_fold(
    *,
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, Any]],
    label_inputs: pd.DataFrame,
    weights: pd.DataFrame,
    anchor_source_id: str,
    outer_year: int,
    settings: dict[str, Any],
    columns: dict[str, str],
    random_state: int,
) -> ModelFitResult:
    """Fit a bagging-PU benchmark using only the registered anchor as positive."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    target = columns[TARGET_ID]
    outcome = columns[OUTCOME]
    weight = columns[WEIGHT]
    anchor_target = f"L0:{anchor_source_id}"
    positive_keys = label_inputs.loc[
        (label_inputs[target] == anchor_target) & (label_inputs[outcome] == True),  # noqa: E712
        [firm, year],
    ].assign(_anchor_positive=True)
    frame = feature_panel.merge(positive_keys, on=[firm, year], how="left", validate="1:1")
    frame["_anchor_positive"] = frame["_anchor_positive"].fillna(False).astype(bool)
    frame = frame.merge(weights[[firm, year, weight]], on=[firm, year], how="left", validate="1:1")
    frame[weight] = frame[weight].fillna(1.0)
    development = frame.loc[frame[year] < outer_year].copy()
    outer = frame.loc[frame[year] == outer_year].copy()
    features = _feature_groups(feature_registry)["full"]
    model_id = f"track_a:anchor_pu:{anchor_source_id}:full"
    oof_rows: list[dict[str, object]] = []
    for validation_year in sorted(int(value) for value in development[year].unique()):
        train = development.loc[development[year] < validation_year]
        validation = development.loc[development[year] == validation_year]
        ensemble = _fit_pu_ensemble(
            train=train,
            features=features,
            positive="_anchor_positive",
            weight=weight,
            settings=settings,
            random_state=random_state + validation_year,
        )
        if not ensemble or validation.empty:
            continue
        predictions = _predict_pu_ensemble(ensemble, validation[features])
        for (_, source), prediction_value in zip(validation.iterrows(), predictions, strict=True):
            oof_rows.append(
                _prediction_row(
                    source,
                    firm=firm,
                    year=year,
                    outer_year=outer_year,
                    learner_id=model_id,
                    measurement_id="L1",
                    target_id="L1",
                    prediction=float(prediction_value),
                    columns=columns,
                )
            )
    ensemble = _fit_pu_ensemble(
        train=development,
        features=features,
        positive="_anchor_positive",
        weight=weight,
        settings=settings,
        random_state=random_state,
    )
    outer_rows: list[dict[str, object]] = []
    if ensemble and not outer.empty:
        predictions = _predict_pu_ensemble(ensemble, outer[features])
        for (_, source), prediction_value in zip(outer.iterrows(), predictions, strict=True):
            outer_rows.append(
                _prediction_row(
                    source,
                    firm=firm,
                    year=year,
                    outer_year=outer_year,
                    learner_id=model_id,
                    measurement_id="L1",
                    target_id="L1",
                    prediction=float(prediction_value),
                    columns=columns,
                )
            )
    model_rows: list[dict[str, object]] = []
    if ensemble and outer_rows:
        payload = pickle.dumps(ensemble, protocol=pickle.HIGHEST_PROTOCOL)
        model_rows.append(
            {
                "model_id": model_id,
                "status": "PASS",
                "track_id": "track_a",
                "training_mechanism": "bagging_pu",
                "positive_source_id": anchor_source_id,
                "positive_target_id": anchor_target,
                "l1_union_used_as_clean_positive": False,
                "unlabeled_rows": int((~development["_anchor_positive"]).sum()),
                "positive_rows": int(development["_anchor_positive"].sum()),
                "bags": len(ensemble),
                "serialized_model_b64": base64.b64encode(payload).decode("ascii"),
                "serialized_model_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    else:
        model_rows.append(
            {
                "model_id": model_id,
                "status": "SKIPPED",
                "reason_code": "INSUFFICIENT_ANCHOR_POSITIVES_OR_UNLABELED_ROWS",
                "positive_source_id": anchor_source_id,
            }
        )
    return ModelFitResult(
        models={
            "status": "PASS" if ensemble and outer_rows else "SKIPPED",
            OUTER_FOLD: str(outer_year),
            "models": model_rows,
            "oof_training_targets": [],
        },
        oof_predictions=_prediction_frame(oof_rows, columns),
        outer_predictions=_prediction_frame(outer_rows, columns),
    )


def fit_fold_models(
    *,
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, Any]],
    label_inputs: pd.DataFrame,
    weights: pd.DataFrame,
    outer_year: int,
    learner_ids: list[str],
    learner_settings: dict[str, Any],
    target_id: str,
    measurement_id: str,
    columns: dict[str, str],
    random_state: int,
    target_values: pd.DataFrame | None = None,
    soft_target: bool = False,
    target_transform: str = "identity",
    track_id: str = "track_a",
    learner_search_spaces: dict[str, Any] | None = None,
    maximum_valid_configurations: int = 50,
) -> ModelFitResult:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    target = columns[TARGET_ID]
    weight = columns[WEIGHT]
    target_value = columns[TARGET_VALUE]
    if target_values is None:
        labels = label_inputs.loc[label_inputs[target] == target_id, [firm, year, outcome]]
        response = outcome
    else:
        required_targets = {firm, year, target_value}
        if not required_targets.issubset(target_values.columns):
            raise ValueError("soft target frame does not satisfy its contract")
        labels = target_values.loc[:, [firm, year, target_value]]
        response = target_value
    frame = feature_panel.merge(labels, on=[firm, year], how="left", validate="1:1")
    frame = frame.merge(weights[[firm, year, weight]], on=[firm, year], how="left", validate="1:1")
    frame[weight] = frame[weight].fillna(1.0)
    development = frame.loc[(frame[year] < outer_year) & frame[response].notna()].copy()
    outer = frame.loc[frame[year] == outer_year].copy()
    groups = _feature_groups(feature_registry)
    models: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    oof_target_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    for group_id, feature_ids in groups.items():
        if not feature_ids:
            continue
        for learner_id in learner_ids:
            model_id = f"{track_id}:{learner_id}:{group_id}"
            candidate_settings = _candidate_settings(
                learner_id=learner_id,
                base_settings=learner_settings,
                search_spaces=learner_search_spaces or {},
                maximum=maximum_valid_configurations,
            )
            tuning_started = time.perf_counter()
            valid_candidates: list[
                tuple[float, dict[str, Any], dict[int, tuple[float, float]]]
            ] = []
            for candidate_index, candidate in enumerate(candidate_settings):
                candidate_registry = dict(learner_settings)
                candidate_registry[learner_id] = candidate
                candidate_predictions = _temporal_oof(
                    development,
                    feature_ids,
                    learner_id,
                    candidate_registry,
                    response,
                    year,
                    weight,
                    random_state + candidate_index * 1009,
                    soft_target=soft_target,
                    target_transform=target_transform,
                )
                objective = _tuning_objective(candidate_predictions, soft_target=soft_target)
                if objective is not None:
                    valid_candidates.append((objective, candidate, candidate_predictions))
            tuning_seconds = time.perf_counter() - tuning_started
            if valid_candidates:
                selected_objective, selected_settings, inner_predictions = min(
                    valid_candidates, key=lambda item: item[0]
                )
            else:
                selected_objective = None
                selected_settings = candidate_settings[0]
                inner_predictions = {}
            for row_index, (prediction, validation_target) in inner_predictions.items():
                source = cast(pd.Series, development.loc[row_index, :])
                oof_rows.append(
                    _prediction_row(
                        source,
                        firm=firm,
                        year=year,
                        outer_year=outer_year,
                        learner_id=model_id,
                        measurement_id=measurement_id,
                        target_id=target_id,
                        prediction=prediction,
                        columns=columns,
                    )
                )
                oof_target_rows.append(
                    {
                        FIRM_ID: str(source[firm]),
                        FISCAL_YEAR: int(source[year]),
                        LEARNER_ID: model_id,
                        TARGET_ID: target_id,
                        TARGET_VALUE: validation_target,
                    }
                )
            transformed_target = _transform_target(
                development[response], development[response], target_transform
            )
            if (
                development.empty
                or outer.empty
                or not _target_is_estimable(transformed_target, soft_target)
                or not valid_candidates
            ):
                models.append(
                    {
                        "model_id": model_id,
                        "status": "SKIPPED",
                        "reason_code": "INSUFFICIENT_POSITIVES",
                        "track_id": track_id,
                        TARGET_ID: target_id,
                        "tuning_status": "SKIPPED_NO_VALID_INNER_CONFIGURATION",
                        "valid_configuration_count": 0,
                        "tuning_runtime_seconds": tuning_seconds,
                    }
                )
                continue
            selected_registry = dict(learner_settings)
            selected_registry[learner_id] = selected_settings
            estimator = _estimator(learner_id, selected_registry, random_state)
            _fit(
                estimator,
                development[feature_ids],
                transformed_target,
                development[weight],
                soft_target=soft_target,
            )
            predictions = np.asarray(
                cast(Any, estimator).predict_proba(outer[feature_ids]), dtype=float
            )[:, 1]
            for (_, source), prediction in zip(outer.iterrows(), predictions, strict=True):
                outer_rows.append(
                    _prediction_row(
                        source,
                        firm=firm,
                        year=year,
                        outer_year=outer_year,
                        learner_id=model_id,
                        measurement_id=measurement_id,
                        target_id=target_id,
                        prediction=float(prediction),
                        columns=columns,
                    )
                )
            payload = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
            models.append(
                {
                    "model_id": model_id,
                    "status": "PASS",
                    "feature_ids": feature_ids,
                    "fit_max_year": int(development[year].max()),
                    "outer_year": outer_year,
                    "track_id": track_id,
                    TARGET_ID: target_id,
                    MEASUREMENT_ID: measurement_id,
                    "training_objective": "soft_cross_entropy"
                    if soft_target
                    else "binary_cross_entropy",
                    "target_transform": target_transform,
                    "tuning_status": "PASS",
                    "valid_configuration_count": len(valid_candidates),
                    "evaluated_configuration_count": len(candidate_settings),
                    "maximum_valid_configurations": maximum_valid_configurations,
                    "tuning_runtime_seconds": tuning_seconds,
                    "tuning_objective": "soft_cross_entropy"
                    if soft_target
                    else "negative_average_precision",
                    "selected_tuning_objective": selected_objective,
                    "selected_settings": selected_settings,
                    "serialized_model_b64": base64.b64encode(payload).decode("ascii"),
                    "serialized_model_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return ModelFitResult(
        models={
            "status": "PASS" if any(item["status"] == "PASS" for item in models) else "SKIPPED",
            OUTER_FOLD: str(outer_year),
            "models": models,
            "oof_training_targets": oof_target_rows,
        },
        oof_predictions=_prediction_frame(oof_rows, columns),
        outer_predictions=_prediction_frame(outer_rows, columns),
    )


def _feature_groups(registry: list[dict[str, Any]]) -> dict[str, list[str]]:
    content = [str(item["feature_id"]) for item in registry if item.get("role") == "content"]
    observable = [
        str(item["feature_id"]) for item in registry if item.get("role") == "observability"
    ]
    ambiguous = [str(item["feature_id"]) for item in registry if item.get("role") == "ambiguous"]
    return {
        "observability_only": observable,
        "content_only": content,
        "full": observable + content + ambiguous,
    }


def _candidate_settings(
    *,
    learner_id: str,
    base_settings: dict[str, Any],
    search_spaces: dict[str, Any],
    maximum: int,
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum tuning configurations must be positive")
    raw_base = base_settings.get(learner_id)
    if not isinstance(raw_base, dict):
        raise ValueError(f"learner={learner_id}: base settings required")
    base = cast(dict[str, Any], raw_base)
    raw_space = search_spaces.get(learner_id)
    if raw_space is None:
        return [dict(base)]
    if not isinstance(raw_space, dict):
        raise ValueError(f"learner={learner_id}: tuning search space must be a mapping")
    space = cast(dict[str, Any], raw_space)
    keys = sorted(space)
    values: list[list[object]] = []
    for key in keys:
        options = space[key]
        if not isinstance(options, list) or not options:
            raise ValueError(f"learner={learner_id}, setting={key}: nonempty list required")
        values.append(cast(list[object], options))
    combinations = list(itertools.product(*values))
    if len(combinations) > maximum:
        raise ValueError(
            f"learner={learner_id}: search space has {len(combinations)} configurations; max={maximum}"
        )
    candidates: list[dict[str, Any]] = []
    for combination in combinations:
        candidate = dict(base)
        candidate.update(dict(zip(keys, combination, strict=True)))
        candidates.append(candidate)
    return candidates


def _tuning_objective(
    predictions: dict[int, tuple[float, float]], *, soft_target: bool
) -> float | None:
    if not predictions:
        return None
    score = [value[0] for value in predictions.values()]
    target = [value[1] for value in predictions.values()]
    if soft_target:
        clipped = np.clip(np.asarray(score, dtype=float), 1e-9, 1.0 - 1e-9)
        soft = np.asarray(target, dtype=float)
        return float(np.mean(-(soft * np.log(clipped) + (1.0 - soft) * np.log(1.0 - clipped))))
    average = average_precision([bool(value) for value in target], score)
    return -average if average is not None else None


def _temporal_oof(
    frame: pd.DataFrame,
    features: list[str],
    learner_id: str,
    settings: dict[str, Any],
    outcome: str,
    year: str,
    weight: str,
    random_state: int,
    *,
    soft_target: bool,
    target_transform: str,
) -> dict[int, tuple[float, float]]:
    predictions: dict[int, tuple[float, float]] = {}
    for validation_year in sorted(int(value) for value in frame[year].unique()):
        train = frame.loc[frame[year] < validation_year]
        validation = frame.loc[frame[year] == validation_year]
        transformed = _transform_target(train[outcome], train[outcome], target_transform)
        if train.empty or validation.empty or not _target_is_estimable(transformed, soft_target):
            continue
        estimator = _estimator(learner_id, settings, random_state + validation_year)
        _fit(
            estimator,
            train[features],
            transformed,
            train[weight],
            soft_target=soft_target,
        )
        values = np.asarray(cast(Any, estimator).predict_proba(validation[features]), dtype=float)[
            :, 1
        ]
        validation_targets = _transform_target(
            validation[outcome], train[outcome], target_transform
        ).to_numpy(dtype=float)
        predictions.update(
            {
                int(index): (float(value), float(target_value))
                for index, value, target_value in zip(
                    validation.index, values, validation_targets, strict=True
                )
            }
        )
    return predictions


def _estimator(learner_id: str, settings: dict[str, Any], random_state: int) -> Pipeline:
    raw = settings.get(learner_id)
    if not isinstance(raw, dict):
        raise ValueError(f"learner={learner_id}: settings required")
    raw = cast(dict[str, Any], raw)
    if learner_id == "elastic_net_logistic":
        model: ClassifierMixin = LogisticRegression(
            C=float(raw["inverse_regularization"]),
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(raw["l1_ratio"]),
            max_iter=int(raw["maximum_iterations"]),
            random_state=random_state,
        )
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    elif learner_id == "random_forest":
        maximum_features = raw["maximum_features"]
        if maximum_features not in {"sqrt", "log2"}:
            raise ValueError("random_forest.maximum_features must be sqrt or log2")
        model = RandomForestClassifier(
            n_estimators=int(raw["trees"]),
            min_samples_leaf=int(raw["minimum_leaf_size"]),
            max_features=maximum_features,
            random_state=random_state,
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    elif learner_id == "main_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=int(raw["maximum_iterations"]),
            learning_rate=float(raw["learning_rate"]),
            max_leaf_nodes=int(raw["maximum_leaf_nodes"]),
            random_state=random_state,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    else:
        raise ValueError(f"learner={learner_id}: unsupported confirmatory learner")
    return Pipeline(steps)


def _fit(
    estimator: Pipeline,
    features: pd.DataFrame,
    outcome: pd.Series,
    weights: pd.Series,
    *,
    soft_target: bool = False,
) -> None:
    if soft_target:
        probabilities = outcome.to_numpy(dtype=float)
        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("soft targets must be in [0, 1]")
        expanded_features = pd.concat([features, features], ignore_index=True)
        expanded_outcome = np.concatenate(
            [np.ones(len(probabilities), dtype=int), np.zeros(len(probabilities), dtype=int)]
        )
        base_weights = weights.to_numpy(dtype=float)
        expanded_weights = np.concatenate(
            [base_weights * probabilities, base_weights * (1.0 - probabilities)]
        )
        keep = expanded_weights > 0
        cast(Any, estimator).fit(
            expanded_features.loc[keep],
            expanded_outcome[keep],
            model__sample_weight=expanded_weights[keep],
        )
        return
    cast(Any, estimator).fit(
        features,
        outcome.astype(int),
        model__sample_weight=weights.to_numpy(dtype=float),
    )


def _fit_pu_ensemble(
    *,
    train: pd.DataFrame,
    features: list[str],
    positive: str,
    weight: str,
    settings: dict[str, Any],
    random_state: int,
) -> list[Pipeline]:
    raw = settings.get("anchor_pu")
    if not isinstance(raw, dict):
        raise ValueError("learners.settings.anchor_pu is required")
    configuration = cast(dict[str, Any], raw)
    positive_indices = train.index[train[positive]].to_numpy(dtype=int)
    unlabeled_indices = train.index[~train[positive]].to_numpy(dtype=int)
    if not features or len(positive_indices) < 2 or len(unlabeled_indices) < 2:
        return []
    bags = int(configuration["bags"])
    ratio = float(configuration["unlabeled_to_positive_ratio"])
    if bags < 1 or ratio <= 0:
        raise ValueError("anchor-PU bagging controls are invalid")
    negative_count = min(len(unlabeled_indices), max(1, math.ceil(len(positive_indices) * ratio)))
    rng = np.random.default_rng(random_state)
    ensemble: list[Pipeline] = []
    for bag in range(bags):
        negative_indices = rng.choice(unlabeled_indices, size=negative_count, replace=False)
        selected = np.concatenate([positive_indices, negative_indices])
        labels = np.concatenate(
            [np.ones(len(positive_indices), dtype=int), np.zeros(len(negative_indices), dtype=int)]
        )
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(configuration["inverse_regularization"]),
                        solver="lbfgs",
                        max_iter=int(configuration["maximum_iterations"]),
                        random_state=random_state + bag,
                    ),
                ),
            ]
        )
        cast(Any, model).fit(
            train.loc[selected, features],
            labels,
            model__sample_weight=train.loc[selected, weight].to_numpy(dtype=float),
        )
        ensemble.append(model)
    return ensemble


def _predict_pu_ensemble(ensemble: list[Pipeline], features: pd.DataFrame) -> np.ndarray:
    predictions = np.column_stack(
        [
            np.asarray(cast(Any, model).predict_proba(features), dtype=float)[:, 1]
            for model in ensemble
        ]
    )
    return np.mean(predictions, axis=1)


def _transform_target(values: pd.Series, reference: pd.Series, transform: str) -> pd.Series:
    numeric = values.astype(float)
    if transform == "identity":
        return numeric
    if transform != "training_ecdf":
        raise ValueError(f"target transform={transform}: unsupported")
    reference_values = np.sort(reference.dropna().to_numpy(dtype=float))
    if len(reference_values) == 0:
        return pd.Series(np.nan, index=values.index, dtype="float64")
    ranks = np.searchsorted(reference_values, numeric.to_numpy(dtype=float), side="right")
    transformed = (ranks - 0.5) / len(reference_values)
    return pd.Series(np.clip(transformed, 1e-6, 1.0 - 1e-6), index=values.index)


def _target_is_estimable(values: pd.Series, soft_target: bool) -> bool:
    observed = values.dropna().to_numpy(dtype=float)
    if len(observed) < 2:
        return False
    if soft_target:
        return bool(np.ptp(observed) > 1e-12)
    return len(np.unique(observed.astype(int))) >= 2


def _prediction_row(
    source: pd.Series,
    *,
    firm: str,
    year: str,
    outer_year: int,
    learner_id: str,
    measurement_id: str,
    target_id: str,
    prediction: float,
    columns: dict[str, str],
) -> dict[str, object]:
    return {
        firm: str(source[firm]),
        year: int(source[year]),
        columns[OUTER_FOLD]: str(outer_year),
        columns[LEARNER_ID]: learner_id,
        columns[MEASUREMENT_ID]: measurement_id,
        columns[TARGET_ID]: target_id,
        columns[PREDICTION]: prediction,
    }


def _prediction_frame(rows: list[dict[str, object]], columns: dict[str, str]) -> pd.DataFrame:
    names = [
        columns[FIRM_ID],
        columns[FISCAL_YEAR],
        columns[OUTER_FOLD],
        columns[LEARNER_ID],
        columns[MEASUREMENT_ID],
        columns[TARGET_ID],
        columns[PREDICTION],
    ]
    frame = pd.DataFrame(rows, columns=names)
    return frame.astype(
        {
            names[0]: "string",
            names[1]: "int16",
            names[2]: "string",
            names[3]: "string",
            names[4]: "string",
            names[5]: "string",
            names[6]: "float64",
        }
    )
