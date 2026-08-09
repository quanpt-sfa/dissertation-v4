"""Fold-safe scikit-learn compatibility helpers used by production stages."""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class DropAllMissingColumns:
    """Drop columns that are entirely missing in the fit sample.

    The fitted mask is reused at transform time, so later rows cannot reintroduce
    a feature that carried no information in the training sample. This makes the
    existing SimpleImputer behavior explicit without zero-imputing an all-missing
    training feature.
    """

    def __init__(self) -> None:
        self.n_features_in_: int = 0
        self.keep_mask_: np.ndarray = np.asarray([], dtype=bool)
        self.feature_names_in_: np.ndarray = np.asarray([], dtype=object)
        self.dropped_feature_names_: tuple[str, ...] = ()

    def fit(self, x: object, y: object = None) -> DropAllMissingColumns:
        del y
        array = self._array(x)
        keep_mask = ~pd.isna(array).all(axis=0)
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if not np.any(keep_mask):
            raise ValueError("all training features are missing")
        self.n_features_in_ = int(array.shape[1])
        self.keep_mask_ = keep_mask
        if isinstance(x, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(value) for value in x.columns], dtype=object)
            self.dropped_feature_names_ = tuple(
                str(name)
                for name, keep in zip(self.feature_names_in_, keep_mask, strict=True)
                if not keep
            )
        else:
            self.feature_names_in_ = np.asarray(
                [f"feature_{index}" for index in range(self.n_features_in_)], dtype=object
            )
            self.dropped_feature_names_ = tuple(
                str(name)
                for name, keep in zip(self.feature_names_in_, keep_mask, strict=True)
                if not keep
            )
        return self

    def transform(self, x: object) -> object:
        if self.n_features_in_ < 1 or len(self.keep_mask_) != self.n_features_in_:
            raise RuntimeError("DropAllMissingColumns must be fitted before transform")
        if isinstance(x, pd.DataFrame):
            if x.shape[1] != self.n_features_in_:
                raise ValueError("feature count changed after DropAllMissingColumns fit")
            return x.iloc[:, self.keep_mask_]
        array = self._array(x)
        if array.shape[1] != self.n_features_in_:
            raise ValueError("feature count changed after DropAllMissingColumns fit")
        return array[:, self.keep_mask_]

    @staticmethod
    def _array(x: object) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            array = x.to_numpy()
        else:
            array = np.asarray(x)
        if array.ndim != 2:
            raise ValueError("model feature input must be two-dimensional")
        return array


@dataclass(frozen=True)
class FitWarningSummary:
    convergence_warning: bool
    warning_names: tuple[str, ...]


def capture_fit_warnings(operation: Callable[[], object]) -> FitWarningSummary:
    """Run one estimator fit while converting convergence warnings into state."""

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        operation()
    names = tuple(item.category.__name__ for item in captured)
    convergence = any(issubclass(item.category, ConvergenceWarning) for item in captured)
    return FitWarningSummary(convergence_warning=convergence, warning_names=names)


def make_logistic_regression(
    *,
    penalty_kind: str = "l2",
    elasticnet_l1_ratio: float | None = None,
    **kwargs: Any,
) -> LogisticRegression:
    """Construct LogisticRegression without the sklearn 1.8 penalty deprecation.

    scikit-learn <1.8 receives the legacy ``penalty`` argument; 1.8+ receives
    the equivalent ``l1_ratio`` representation. This keeps the repo compatible
    with its historical environment while remaining forward-compatible with
    removal of ``penalty`` in 1.10.
    """

    if penalty_kind not in {"l1", "l2", "elasticnet"}:
        raise ValueError(f"unsupported logistic penalty kind: {penalty_kind}")
    match = re.match(r"^(\d+)\.(\d+)", sklearn.__version__)
    if match is None:
        raise RuntimeError(f"cannot parse scikit-learn version: {sklearn.__version__}")
    version = (int(match.group(1)), int(match.group(2)))
    parameters: dict[str, Any] = dict(kwargs)
    if version >= (1, 8):
        if penalty_kind == "l2":
            parameters["l1_ratio"] = 0.0
        elif penalty_kind == "l1":
            parameters["l1_ratio"] = 1.0
        else:
            if elasticnet_l1_ratio is None or not 0.0 <= elasticnet_l1_ratio <= 1.0:
                raise ValueError("elastic-net logistic requires l1_ratio in [0, 1]")
            parameters["l1_ratio"] = float(elasticnet_l1_ratio)
    else:
        parameters["penalty"] = penalty_kind
        if penalty_kind == "elasticnet":
            if elasticnet_l1_ratio is None or not 0.0 <= elasticnet_l1_ratio <= 1.0:
                raise ValueError("elastic-net logistic requires l1_ratio in [0, 1]")
            parameters["l1_ratio"] = float(elasticnet_l1_ratio)
    return LogisticRegression(**parameters)
