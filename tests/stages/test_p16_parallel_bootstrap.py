from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gates.parallel_bootstrap import parallel_shape_bootstrap


def _interaction(
    frame: pd.DataFrame,
    _monitoring: str,
    _outcome: str,
    _gate: dict[str, Any],
) -> float:
    return float(frame["_pressure_z"].mean())


def _breakpoint(
    frame: pd.DataFrame,
    _monitoring: str,
    _outcome: str,
    _gate: dict[str, Any],
) -> tuple[float | None, float]:
    return float(frame["_pressure_z"].median()), 0.5


def _legacy_sequential(
    *,
    frame: pd.DataFrame,
    replications: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    coefficients: list[float] = []
    breakpoints: list[float] = []
    for _ in range(replications):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        if sample["outcome"].nunique() < 2:
            continue
        coefficients.append(_interaction(sample, "monitoring", "outcome", {}))
        point, _ = _breakpoint(sample, "monitoring", "outcome", {})
        if point is not None:
            breakpoints.append(point)
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    coefficient_interval = (
        (
            float(np.quantile(coefficients, quantiles[0])),
            float(np.quantile(coefficients, quantiles[1])),
        )
        if coefficients
        else (-math.inf, math.inf)
    )
    breakpoint_interval = (
        (
            float(np.quantile(breakpoints, quantiles[0])),
            float(np.quantile(breakpoints, quantiles[1])),
        )
        if breakpoints
        else (-math.inf, math.inf)
    )
    return coefficient_interval, breakpoint_interval


def test_parallel_shape_bootstrap_matches_legacy_rng_stream() -> None:
    frame = pd.DataFrame(
        {
            "_pressure_z": np.linspace(-1.5, 1.5, 16),
            "monitoring": [0.0, 1.0] * 8,
            "outcome": [False, True, False, True] * 4,
        }
    )
    expected = _legacy_sequential(
        frame=frame,
        replications=40,
        alpha=0.05,
        rng=np.random.default_rng(20260804),
    )
    sequential = parallel_shape_bootstrap(
        frame=frame,
        monitoring="monitoring",
        outcome="outcome",
        gate={},
        replications=40,
        alpha=0.05,
        rng=np.random.default_rng(20260804),
        workers=1,
        interaction_coefficient=_interaction,
        breakpoint_estimator=_breakpoint,
    )
    parallel = parallel_shape_bootstrap(
        frame=frame,
        monitoring="monitoring",
        outcome="outcome",
        gate={},
        replications=40,
        alpha=0.05,
        rng=np.random.default_rng(20260804),
        workers=4,
        interaction_coefficient=_interaction,
        breakpoint_estimator=_breakpoint,
    )

    assert sequential == expected
    assert parallel == expected


def test_parallel_shape_bootstrap_requires_positive_workers() -> None:
    with pytest.raises(ValueError, match="P16 workers must be positive"):
        parallel_shape_bootstrap(
            frame=pd.DataFrame({"outcome": [False, True]}),
            monitoring="monitoring",
            outcome="outcome",
            gate={},
            replications=1,
            alpha=0.05,
            rng=np.random.default_rng(1),
            workers=0,
            interaction_coefficient=_interaction,
            breakpoint_estimator=_breakpoint,
        )
