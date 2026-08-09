from __future__ import annotations

import re
from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def replace_once(text: str, old: str, new: str, *, context: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{context}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def replace_count(text: str, old: str, new: str, expected: int, *, context: str) -> str:
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{context}: expected {expected} matches, found {count}")
    return text.replace(old, new)


def regex_once(text: str, pattern: str, replacement: str, *, context: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{context}: expected exactly one regex match, found {count}")
    return updated


SKLEARN_SUPPORT = '''"""Fold-safe scikit-learn compatibility helpers used by production stages."""

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

    def fit(self, x: object, y: object = None) -> "DropAllMissingColumns":
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
                str(name) for name, keep in zip(self.feature_names_in_, keep_mask, strict=True) if not keep
            )
        else:
            self.feature_names_in_ = np.asarray(
                [f"feature_{index}" for index in range(self.n_features_in_)], dtype=object
            )
            self.dropped_feature_names_ = tuple(
                str(name) for name, keep in zip(self.feature_names_in_, keep_mask, strict=True) if not keep
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
    match = re.match(r"^(\\d+)\\.(\\d+)", sklearn.__version__)
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
'''

TEST_RUNTIME_HARDENING = '''# pyright: reportPrivateUsage=false

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from core.sklearn_support import (
    DropAllMissingColumns,
    capture_fit_warnings,
    make_logistic_regression,
)
from scripts.run_pipeline import _auto_p08_workers, _resolve_p08_workers
from simulation.service import _fit_pu_ensemble_cost_sensitive, diagnostic_capture, DiagnosticCollector


def test_p08_worker_auto_scale_is_independent_of_general_workers() -> None:
    assert _auto_p08_workers(8) == 6
    assert _auto_p08_workers(32) == 24
    assert _auto_p08_workers(64) == 32
    assert _resolve_p08_workers(None, None, logical_processors=32) == 24
    assert _resolve_p08_workers(12, "20", logical_processors=32) == 12
    assert _resolve_p08_workers(None, "20", logical_processors=32) == 20


def test_drop_all_missing_columns_prevents_median_imputer_warning() -> None:
    frame = pd.DataFrame({"empty": [np.nan, np.nan, np.nan], "usable": [1.0, np.nan, 3.0]})
    pipeline = Pipeline(
        [
            ("drop_all_missing", DropAllMissingColumns()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        transformed = pipeline.fit_transform(frame)
    assert transformed.shape == (3, 1)
    assert not any("Skipping features without any observed values" in str(item.message) for item in captured)
    dropper = pipeline.named_steps["drop_all_missing"]
    assert dropper.dropped_feature_names_ == ("empty",)


def test_logistic_compatibility_emits_no_penalty_future_warning() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    y = np.asarray([0, 0, 1, 1])
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = make_logistic_regression(
            penalty_kind="elasticnet",
            elasticnet_l1_ratio=0.5,
            solver="saga",
            C=1.0,
            max_iter=5000,
            random_state=1,
        )
        model.fit(x, y)
    assert not any(issubclass(item.category, FutureWarning) for item in captured)


def test_capture_fit_warnings_flags_convergence() -> None:
    from sklearn.exceptions import ConvergenceWarning

    def operation() -> None:
        warnings.warn("not converged", ConvergenceWarning, stacklevel=2)

    summary = capture_fit_warnings(operation)
    assert summary.convergence_warning is True
    assert "ConvergenceWarning" in summary.warning_names


def test_pu_precheck_records_one_insufficient_data_event() -> None:
    collector = DiagnosticCollector()
    x = np.arange(60, dtype=float).reshape(20, 3)
    with diagnostic_capture(collector, 7):
        scores, success, _ = _fit_pu_ensemble_cost_sensitive(
            learner_id="logistic_regression",
            x_train=x,
            predict_x=x[:2],
            positive_indices=np.asarray([0, 1, 2]),
            unlabeled_indices=np.arange(3, 20),
            bags=5,
            unlabeled_to_positive_ratio=1.0,
            training_cost={
                "false_negative_cost": 1.0,
                "true_positive_benefit": 0.0,
                "false_positive_cost": 1.0,
            },
            scenario={"cost_sensitive_settings": {"maximum_cost_weight": 20.0}},
            rng=np.random.default_rng(1),
        )
    assert success == 0.0
    assert scores.shape == (2,)
    assert collector.resampling_failures == {"InsufficientPUData": 1}
    assert collector.fit_failures == {}
'''


def patch_run_pipeline() -> None:
    path = "scripts/run_pipeline.py"
    text = read(path)
    marker = "\ndef _require_clean_tree(root: Path) -> None:\n"
    helpers = '''\ndef _auto_p08_workers(logical_processors: int | None = None) -> int:\n    logical = max(1, int(logical_processors or os.cpu_count() or 1))\n    if logical <= 4:\n        return logical\n    return min(32, max(4, (logical * 3 + 3) // 4))\n\n\ndef _resolve_p08_workers(\n    requested: int | None, env_value: str | None, *, logical_processors: int | None = None\n) -> int:\n    if requested is not None:\n        if requested < 1:\n            raise ValueError("--p08-workers must be positive")\n        return requested\n    if env_value not in {None, ""}:\n        try:\n            value = int(str(env_value))\n        except ValueError as exc:\n            raise ValueError("P08_WORKERS must be a positive integer") from exc\n        if value < 1:\n            raise ValueError("P08_WORKERS must be a positive integer")\n        return value\n    return _auto_p08_workers(logical_processors)\n\n'''
    text = replace_once(text, marker, helpers + marker, context="run_pipeline worker helpers")
    old = '''    parser.add_argument(\n        "--workers",\n        type=int,\n        default=1,\n        help="Parallel workers for partitioned stages (P01, P10, P11, P12).",\n    )\n    args = parser.parse_args()\n    workers = max(1, args.workers)\n'''
    new = '''    parser.add_argument(\n        "--workers",\n        type=int,\n        default=1,\n        help="Parallel workers for partitioned non-P08 stages (P01, P10, P11, P12).",\n    )\n    parser.add_argument(\n        "--p08-workers",\n        type=int,\n        help=(\n            "P08 subprocess workers. If omitted, use P08_WORKERS when set; "\n            "otherwise auto-size to 75% of logical CPUs, capped at 32."\n        ),\n    )\n    args = parser.parse_args()\n    workers = max(1, args.workers)\n    p08_workers = _resolve_p08_workers(args.p08_workers, os.environ.get("P08_WORKERS"))\n    print(\n        f"worker_plan general={workers} p08={p08_workers} "\n        f"logical_cpus={os.cpu_count() or 1}",\n        flush=True,\n    )\n'''
    text = replace_once(text, old, new, context="run_pipeline parser")
    text = replace_once(
        text,
        '        str(max(1, int(os.environ.get("P08_WORKERS", str(workers))))),\n',
        '        str(p08_workers),\n',
        context="run_pipeline P08 worker propagation",
    )
    write(path, text)


def patch_p08_worker() -> None:
    path = "scripts/p08b_run_batch.py"
    text = read(path)
    old = '''for _name in (\n    "OMP_NUM_THREADS",\n    "OPENBLAS_NUM_THREADS",\n    "MKL_NUM_THREADS",\n    "NUMEXPR_NUM_THREADS",\n):\n    os.environ.setdefault(_name, "1")\n'''
    new = '''for _name in (\n    "OMP_NUM_THREADS",\n    "OPENBLAS_NUM_THREADS",\n    "MKL_NUM_THREADS",\n    "NUMEXPR_NUM_THREADS",\n    "BLIS_NUM_THREADS",\n):\n    os.environ[_name] = "1"\n'''
    text = replace_once(text, old, new, context="P08 hard thread cap")
    write(path, text)


def patch_simulation() -> None:
    path = "src/simulation/service.py"
    text = read(path)
    text = replace_once(
        text,
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.exceptions import ConvergenceWarning\n",
        context="simulation sklearn imports",
    )
    text = replace_once(
        text,
        "from core.metrics import average_precision, roc_auc\n",
        "from core.metrics import average_precision, roc_auc\nfrom core.sklearn_support import DropAllMissingColumns, make_logistic_regression\n",
        context="simulation helper import",
    )
    text = replace_once(
        text,
        "_REQUIRED = {\n",
        "_MINIMUM_CLASSIFIER_ROWS = 10\n\n_REQUIRED = {\n",
        context="simulation minimum rows constant",
    )
    text = replace_count(
        text,
        '("imputer", SimpleImputer(strategy="median")),',
        '("drop_all_missing", DropAllMissingColumns()),\n                ("imputer", SimpleImputer(strategy="median")),',
        7,
        context="simulation fold-safe imputers",
    )
    text = replace_count(
        text,
        'LogisticRegression(\n                    penalty="l2",',
        'make_logistic_regression(\n                    penalty_kind="l2",',
        2,
        context="simulation L2 logistic compatibility",
    )
    text = replace_once(
        text,
        'LogisticRegression(\n                        penalty="elasticnet",\n                        solver="saga",\n                        C=1.0,\n                        l1_ratio=0.5,',
        'make_logistic_regression(\n                        penalty_kind="elasticnet",\n                        elasticnet_l1_ratio=0.5,\n                        solver="saga",\n                        C=1.0,',
        context="simulation elasticnet compatibility",
    )
    text = replace_once(
        text,
        "if len(y_train) < 10 or len(np.unique(y_train)) < 2:",
        "if len(y_train) < _MINIMUM_CLASSIFIER_ROWS or len(np.unique(y_train)) < 2:",
        context="simulation minimum fit rows",
    )
    old = '''        for w in captured_warnings or []:\n            record_model_warning(w.category.__name__)\n        scores = _predict_scores(estimator, x_test)\n'''
    new = '''        convergence_warning = False\n        for warning in captured_warnings or []:\n            record_model_warning(warning.category.__name__)\n            if issubclass(warning.category, ConvergenceWarning):\n                convergence_warning = True\n        if convergence_warning:\n            record_fit_failure("ConvergenceWarning")\n            return (\n                fallback_scores,\n                0.0,\n                {\n                    "failure_type": "ConvergenceWarning",\n                    "failure_message": "estimator did not converge within the locked iteration budget",\n                },\n            )\n        scores = _predict_scores(estimator, x_test)\n'''
    text = replace_once(text, old, new, context="simulation convergence policy")
    text = re.sub(
        r'\n\s*if success == 0\.0:\n\s*import sys\n\n\s*print\(f"Learner fit failure \(\{learner_id\}\): \{diag\}", file=sys\.stderr, flush=True\)\n',
        "\n",
        text,
        count=1,
    )
    text = re.sub(
        r'\n\s*if success == 0\.0:\n\s*import sys\n\n\s*print\(f"PU bag learner fit failure \(\{learner_id\}\): \{diag\}", file=sys\.stderr, flush=True\)\n',
        "\n",
        text,
        count=1,
    )
    old_sample = '''    sample_count = min(\n        len(unlabeled_indices),\n        max(2, int(math.ceil(len(positive_indices) * unlabeled_to_positive_ratio))),\n    )\n    for _ in range(bags):\n'''
    new_sample = '''    sample_count = min(\n        len(unlabeled_indices),\n        max(2, int(math.ceil(len(positive_indices) * unlabeled_to_positive_ratio))),\n    )\n    if len(positive_indices) + sample_count < _MINIMUM_CLASSIFIER_ROWS:\n        record_resampling_failure("InsufficientPUData")\n        prior = len(positive_indices) / max(1, len(positive_indices) + len(unlabeled_indices))\n        return np.full(len(predict_x), prior, dtype=float), 0.0, (1.0, 1.0)\n    for _ in range(bags):\n'''
    text = replace_once(text, old_sample, new_sample, context="simulation PU precheck")
    if "LogisticRegression(" in text:
        raise RuntimeError("simulation: legacy LogisticRegression constructor remains")
    write(path, text)


def patch_modeling() -> None:
    path = "src/modeling/service.py"
    text = read(path)
    text = replace_once(
        text,
        "from sklearn.linear_model import LogisticRegression\n",
        "",
        context="modeling remove direct logistic import",
    )
    text = replace_once(
        text,
        "from core.metrics import average_precision\n",
        "from core.metrics import average_precision\nfrom core.sklearn_support import (\n    DropAllMissingColumns,\n    capture_fit_warnings,\n    make_logistic_regression,\n)\n",
        context="modeling helper imports",
    )
    text = replace_count(
        text,
        '("imputer", SimpleImputer(strategy="median")),',
        '("drop_all_missing", DropAllMissingColumns()),\n            ("imputer", SimpleImputer(strategy="median")),',
        4,
        context="modeling fold-safe imputers",
    )
    text = replace_once(
        text,
        'LogisticRegression(\n            C=float(raw["inverse_regularization"]),\n            penalty="elasticnet",\n            solver="saga",\n            l1_ratio=float(raw["l1_ratio"]),',
        'make_logistic_regression(\n            penalty_kind="elasticnet",\n            elasticnet_l1_ratio=float(raw["l1_ratio"]),\n            C=float(raw["inverse_regularization"]),\n            solver="saga",',
        context="modeling elasticnet compatibility",
    )
    text = replace_once(
        text,
        'LogisticRegression(\n                        C=float(configuration["inverse_regularization"]),\n                        solver="lbfgs",',
        'make_logistic_regression(\n                        penalty_kind="l2",\n                        C=float(configuration["inverse_regularization"]),\n                        solver="lbfgs",',
        context="modeling anchor PU logistic compatibility",
    )
    fit_replacement = '''def _fit(\n    estimator: Pipeline,\n    features: pd.DataFrame,\n    outcome: pd.Series,\n    weights: pd.Series,\n    *,\n    soft_target: bool = False,\n) -> bool:\n    if soft_target:\n        probabilities = outcome.to_numpy(dtype=float)\n        if np.any((probabilities < 0) | (probabilities > 1)):\n            raise ValueError("soft targets must be in [0, 1]")\n        expanded_features = pd.concat([features, features], ignore_index=True)\n        expanded_outcome = np.concatenate(\n            [np.ones(len(probabilities), dtype=int), np.zeros(len(probabilities), dtype=int)]\n        )\n        base_weights = weights.to_numpy(dtype=float)\n        expanded_weights = np.concatenate(\n            [base_weights * probabilities, base_weights * (1.0 - probabilities)]\n        )\n        keep = expanded_weights > 0\n\n        def operation() -> object:\n            return cast(Any, estimator).fit(\n                expanded_features.loc[keep],\n                expanded_outcome[keep],\n                model__sample_weight=expanded_weights[keep],\n            )\n\n    else:\n\n        def operation() -> object:\n            return cast(Any, estimator).fit(\n                features,\n                outcome.astype(int),\n                model__sample_weight=weights.to_numpy(dtype=float),\n            )\n\n    summary = capture_fit_warnings(operation)\n    return not summary.convergence_warning\n\n\ndef _fit_pu_ensemble('''
    text = regex_once(
        text,
        r"def _fit\(\n.*?\n\n\ndef _fit_pu_ensemble\(",
        fit_replacement,
        context="modeling convergence-aware fit",
    )
    old_oof = '''        _fit(\n            estimator,\n            train[features],\n            transformed,\n            train[weight],\n            soft_target=soft_target,\n        )\n        values = np.asarray(cast(Any, estimator).predict_proba(validation[features]), dtype=float)[\n'''
    new_oof = '''        if not _fit(\n            estimator,\n            train[features],\n            transformed,\n            train[weight],\n            soft_target=soft_target,\n        ):\n            return {}\n        values = np.asarray(cast(Any, estimator).predict_proba(validation[features]), dtype=float)[\n'''
    text = replace_once(text, old_oof, new_oof, context="modeling OOF convergence gate")
    old_final = '''            _fit(\n                estimator,\n                development[feature_ids],\n                transformed_target,\n                development[weight],\n                soft_target=soft_target,\n            )\n            predictions = np.asarray(\n'''
    new_final = '''            if not _fit(\n                estimator,\n                development[feature_ids],\n                transformed_target,\n                development[weight],\n                soft_target=soft_target,\n            ):\n                models.append(\n                    {\n                        "model_id": model_id,\n                        "status": "SKIPPED",\n                        "reason_code": "MODEL_NONCONVERGENCE",\n                        "track_id": track_id,\n                        TARGET_ID: target_id,\n                        "tuning_status": "SKIPPED_FINAL_FIT_NONCONVERGENCE",\n                        "valid_configuration_count": len(valid_candidates),\n                        "tuning_runtime_seconds": tuning_seconds,\n                    }\n                )\n                continue\n            predictions = np.asarray(\n'''
    text = replace_once(text, old_final, new_final, context="modeling final convergence gate")
    old_pu_fit = '''        cast(Any, model).fit(\n            train.loc[selected, features],\n            labels,\n            model__sample_weight=train.loc[selected, weight].to_numpy(dtype=float),\n        )\n        ensemble.append(model)\n'''
    new_pu_fit = '''        summary = capture_fit_warnings(\n            lambda: cast(Any, model).fit(\n                train.loc[selected, features],\n                labels,\n                model__sample_weight=train.loc[selected, weight].to_numpy(dtype=float),\n            )\n        )\n        if summary.convergence_warning:\n            continue\n        ensemble.append(model)\n'''
    text = replace_once(text, old_pu_fit, new_pu_fit, context="modeling anchor PU convergence gate")
    if "LogisticRegression(" in text:
        raise RuntimeError("modeling: legacy LogisticRegression constructor remains")
    write(path, text)


def patch_splits() -> None:
    path = "src/splits/service.py"
    text = read(path)
    text = replace_once(text, "from sklearn.linear_model import LogisticRegression\n", "", context="splits direct logistic import")
    text = replace_once(
        text,
        "from core.semantic_keys import (\n",
        "from core.sklearn_support import (\n    DropAllMissingColumns,\n    capture_fit_warnings,\n    make_logistic_regression,\n)\nfrom core.semantic_keys import (\n",
        context="splits helper imports",
    )
    text = replace_once(
        text,
        '("imputer", SimpleImputer(strategy="median")),',
        '("drop_all_missing", DropAllMissingColumns()),\n            ("imputer", SimpleImputer(strategy="median")),',
        context="splits fold-safe imputer",
    )
    text = replace_once(
        text,
        '("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),',
        '(\n                "model",\n                make_logistic_regression(\n                    penalty_kind="l2", C=1.0, solver="lbfgs", max_iter=2000\n                ),\n            ),',
        context="splits logistic compatibility",
    )
    old_fit = '''    cast(Any, estimator).fit(train[feature_ids], outcome.to_numpy())\n    propensities = np.asarray(cast(Any, estimator).predict_proba(train[feature_ids]), dtype=float)[\n'''
    new_fit = '''    try:\n        fit_summary = capture_fit_warnings(\n            lambda: cast(Any, estimator).fit(train[feature_ids], outcome.to_numpy())\n        )\n    except ValueError as exc:\n        if "all training features are missing" not in str(exc):\n            raise\n        return (\n            [1.0] * len(train),\n            [],\n            {\n                "status": "SKIPPED",\n                "reason_code": "NO_ESTIMABLE_VERIFICATION_FEATURES",\n            },\n        )\n    if fit_summary.convergence_warning:\n        return (\n            [1.0] * len(train),\n            [],\n            {\n                "status": "SKIPPED",\n                "reason_code": "PROPENSITY_NONCONVERGENCE",\n            },\n        )\n    propensities = np.asarray(cast(Any, estimator).predict_proba(train[feature_ids]), dtype=float)[\n'''
    text = replace_once(text, old_fit, new_fit, context="splits convergence policy")
    text = replace_once(
        text,
        '            "feature_ids": feature_ids,\n',
        '            "feature_ids": feature_ids,\n            "dropped_all_missing_feature_ids": list(\n                cast(Any, estimator).named_steps["drop_all_missing"].dropped_feature_names_\n            ),\n',
        context="splits dropped feature diagnostics",
    )
    write(path, text)


def patch_domain_transfer() -> None:
    path = "src/sensitivity/domain_transfer.py"
    text = read(path)
    text = replace_once(text, "from sklearn.linear_model import LogisticRegression\n", "", context="domain direct logistic import")
    text = replace_once(
        text,
        "from core.metrics import average_precision\n",
        "from core.metrics import average_precision\nfrom core.sklearn_support import DropAllMissingColumns, make_logistic_regression\n",
        context="domain helper imports",
    )
    text = replace_once(
        text,
        '("imputer", SimpleImputer(strategy="median")),',
        '("drop_all_missing", DropAllMissingColumns()),\n                    ("imputer", SimpleImputer(strategy="median")),',
        context="domain fold-safe imputer",
    )
    text = replace_once(
        text,
        'LogisticRegression(\n                            C=1.0,\n                            solver="lbfgs",',
        'make_logistic_regression(\n                            penalty_kind="l2",\n                            C=1.0,\n                            solver="lbfgs",',
        context="domain logistic compatibility",
    )
    write(path, text)


def patch_learners() -> None:
    path = "config/execution/learners.yaml"
    text = read(path)
    text = replace_once(
        text,
        '''    elastic_net_logistic:\n      inverse_regularization: 1.0\n      l1_ratio: 0.5\n      maximum_iterations: 2000\n''',
        '''    elastic_net_logistic:\n      inverse_regularization: 1.0\n      l1_ratio: 0.5\n      maximum_iterations: 10000\n''',
        context="elastic-net convergence budget",
    )
    text = replace_once(
        text,
        '''    anchor_pu:\n      bags: 25\n      unlabeled_to_positive_ratio: 1.0\n      inverse_regularization: 1.0\n      maximum_iterations: 2000\n''',
        '''    anchor_pu:\n      bags: 25\n      unlabeled_to_positive_ratio: 1.0\n      inverse_regularization: 1.0\n      maximum_iterations: 5000\n''',
        context="anchor PU convergence budget",
    )
    write(path, text)


def patch_docs() -> None:
    path = "docs/AGENT_REPO_MAP.md"
    text = read(path)
    old = '''- `run_pipeline.py`: orchestration duy nhất cho luồng định kỳ; `--workers N` bật\n  song song cho các stage có partition (P01, P10, P11, P12); mặc định `--workers 1`\n  giữ hành vi tuần tự.\n'''
    new = '''- `run_pipeline.py`: orchestration duy nhất cho luồng định kỳ; `--workers N` điều khiển\n  các stage partition ngoài P08 (P01, P10, P11, P12). P08 dùng `--p08-workers N` nếu\n  được truyền, tiếp theo là `P08_WORKERS`, và nếu cả hai đều vắng thì tự chọn khoảng\n  75% logical CPU với trần 32 subprocess workers.\n'''
    text = replace_once(text, old, new, context="worker docs")
    text = replace_once(
        text,
        '- P08 có coordinator, worker theo batch và collector MCSE riêng; `P08_WORKERS` env\n  var ghi đè `--workers` cho riêng P08.\n',
        '- P08 có coordinator, worker theo batch và collector MCSE riêng; `--p08-workers`\n  ghi đè auto-sizing, còn `P08_WORKERS` là override môi trường khi CLI không được truyền.\n',
        context="P08 override docs",
    )
    write(path, text)


def main() -> None:
    Path("src/core/sklearn_support.py").write_text(SKLEARN_SUPPORT, encoding="utf-8", newline="\n")
    Path("tests/test_runtime_hardening.py").write_text(TEST_RUNTIME_HARDENING, encoding="utf-8", newline="\n")
    patch_run_pipeline()
    patch_p08_worker()
    patch_simulation()
    patch_modeling()
    patch_splits()
    patch_domain_transfer()
    patch_learners()
    patch_docs()
    Path("scripts/_agent_runtime_hardening_patch.py").unlink()
    Path(".github/workflows/agent-runtime-hardening.yml").unlink()
    print("runtime hardening patch applied")


if __name__ == "__main__":
    main()
