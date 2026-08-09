# pyright: reportPrivateUsage=false

from __future__ import annotations

import warnings
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from core.sklearn_support import (
    DropAllMissingColumns,
    capture_fit_warnings,
    make_logistic_regression,
)
from simulation.service import DiagnosticCollector, diagnostic_capture
from simulation.v24_service import PUFitResult, _fit_pu_ensemble_cost_sensitive_v24

_RUNNER = run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"))
_auto_p08_workers = _RUNNER["_auto_p08_workers"]
_resolve_p08_workers = _RUNNER["_resolve_p08_workers"]


def test_p08_worker_auto_scale_is_independent_of_general_workers() -> None:
    assert _auto_p08_workers(4) == 4
    assert _auto_p08_workers(8) == 6
    assert _auto_p08_workers(32) == 24
    assert _auto_p08_workers(64) == 32
    assert _resolve_p08_workers(None, None, logical_processors=32) == 24
    assert _resolve_p08_workers(12, "20", logical_processors=32) == 12
    assert _resolve_p08_workers(None, "20", logical_processors=32) == 20
    assert _resolve_p08_workers(7, "20", logical_processors=64) == 7


def test_p08_worker_overrides_fail_closed_when_nonpositive() -> None:
    with pytest.raises(ValueError, match="--p08-workers must be positive"):
        _resolve_p08_workers(0, None, logical_processors=32)
    with pytest.raises(ValueError, match="P08_WORKERS must be a positive integer"):
        _resolve_p08_workers(None, "0", logical_processors=32)


def test_drop_all_missing_columns_prevents_median_imputer_warning() -> None:
    frame = pd.DataFrame({"empty": [np.nan, np.nan, np.nan], "usable": [1.0, np.nan, 3.0]})
    dropper = DropAllMissingColumns()
    filtered = dropper.fit(frame).transform(frame)
    assert isinstance(filtered, pd.DataFrame)
    imputer = SimpleImputer(strategy="median")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        imputer.fit(filtered)
    assert not any(
        "Skipping features without any observed values" in str(item.message) for item in captured
    )
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

    def forbidden_delegate(**_: object) -> PUFitResult:
        raise AssertionError("non-estimable PU bag must be short-circuited")

    with diagnostic_capture(collector, 7):
        scores, success, _ = _fit_pu_ensemble_cost_sensitive_v24(
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
            delegate=forbidden_delegate,
        )
    assert success == 0.0
    assert scores.shape == (2,)
    assert collector.resampling_failures == {"InsufficientPUData": 1}
    assert collector.fit_failures == {}
