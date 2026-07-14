"""Fit registered learners with temporal OOF preprocessing and frozen models."""

from __future__ import annotations

import base64
import hashlib
import pickle
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

from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MEASUREMENT_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    TARGET_ID,
    WEIGHT,
)


@dataclass(frozen=True)
class ModelFitResult:
    models: dict[str, object]
    oof_predictions: pd.DataFrame
    outer_predictions: pd.DataFrame


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
) -> ModelFitResult:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    target = columns[TARGET_ID]
    weight = columns[WEIGHT]
    labels = label_inputs.loc[label_inputs[target] == target_id, [firm, year, outcome]]
    frame = feature_panel.merge(labels, on=[firm, year], how="left", validate="1:1")
    frame = frame.merge(weights[[firm, year, weight]], on=[firm, year], how="left", validate="1:1")
    frame[weight] = frame[weight].fillna(1.0)
    development = frame.loc[(frame[year] < outer_year) & frame[outcome].notna()].copy()
    outer = frame.loc[frame[year] == outer_year].copy()
    groups = _feature_groups(feature_registry)
    models: list[dict[str, object]] = []
    oof_rows: list[dict[str, object]] = []
    outer_rows: list[dict[str, object]] = []
    for group_id, feature_ids in groups.items():
        if not feature_ids:
            continue
        for learner_id in learner_ids:
            model_id = f"{learner_id}:{group_id}"
            inner_predictions = _temporal_oof(
                development,
                feature_ids,
                learner_id,
                learner_settings,
                outcome,
                year,
                weight,
                random_state,
            )
            for row_index, prediction in inner_predictions.items():
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
            if development.empty or outer.empty or development[outcome].nunique() < 2:
                models.append(
                    {
                        "model_id": model_id,
                        "status": "SKIPPED",
                        "reason_code": "INSUFFICIENT_POSITIVES",
                    }
                )
                continue
            estimator = _estimator(learner_id, learner_settings, random_state)
            _fit(
                estimator,
                development[feature_ids],
                development[outcome].astype(int),
                development[weight],
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
                    "serialized_model_b64": base64.b64encode(payload).decode("ascii"),
                    "serialized_model_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return ModelFitResult(
        models={
            "status": "PASS" if any(item["status"] == "PASS" for item in models) else "SKIPPED",
            OUTER_FOLD: str(outer_year),
            "models": models,
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


def _temporal_oof(
    frame: pd.DataFrame,
    features: list[str],
    learner_id: str,
    settings: dict[str, Any],
    outcome: str,
    year: str,
    weight: str,
    random_state: int,
) -> dict[int, float]:
    predictions: dict[int, float] = {}
    for validation_year in sorted(int(value) for value in frame[year].unique()):
        train = frame.loc[frame[year] < validation_year]
        validation = frame.loc[frame[year] == validation_year]
        if train.empty or validation.empty or train[outcome].nunique() < 2:
            continue
        estimator = _estimator(learner_id, settings, random_state + validation_year)
        _fit(estimator, train[features], train[outcome].astype(int), train[weight])
        values = np.asarray(cast(Any, estimator).predict_proba(validation[features]), dtype=float)[
            :, 1
        ]
        predictions.update(
            {
                int(index): float(value)
                for index, value in zip(validation.index, values, strict=True)
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
    estimator: Pipeline, features: pd.DataFrame, outcome: pd.Series, weights: pd.Series
) -> None:
    cast(Any, estimator).fit(features, outcome, model__sample_weight=weights.to_numpy(dtype=float))


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
