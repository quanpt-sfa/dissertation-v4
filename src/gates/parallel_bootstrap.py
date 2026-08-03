"""Deterministic worker execution for Gate 3 bootstrap replications."""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

InteractionCoefficient = Callable[[pd.DataFrame, str, str, dict[str, Any]], float]
BreakpointEstimator = Callable[
    [pd.DataFrame, str, str, dict[str, Any]],
    tuple[float | None, float],
]


def parallel_shape_bootstrap(
    *,
    frame: pd.DataFrame,
    monitoring: str,
    outcome: str,
    gate: dict[str, Any],
    replications: int,
    alpha: float,
    rng: np.random.Generator,
    workers: int,
    interaction_coefficient: InteractionCoefficient,
    breakpoint_estimator: BreakpointEstimator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Evaluate bootstrap replications concurrently without changing the RNG stream."""
    if workers < 1:
        raise ValueError("P16 workers must be positive")
    if replications < 0:
        raise ValueError("P16 bootstrap replications must be nonnegative")

    # Generate every resample on the caller thread, in the same order as the
    # former sequential loop. Worker scheduling therefore cannot alter RNG use.
    resamples = [
        rng.integers(0, len(frame), size=len(frame), dtype=np.int64)
        for _ in range(replications)
    ]

    def evaluate(indices: NDArray[np.int64]) -> tuple[float | None, float | None]:
        sample = frame.iloc[indices]
        if sample[outcome].nunique() < 2:
            return None, None
        coefficient = interaction_coefficient(sample, monitoring, outcome, gate)
        point, _ = breakpoint_estimator(sample, monitoring, outcome, gate)
        return coefficient, point

    worker_count = min(workers, len(resamples)) if resamples else 1
    if worker_count == 1:
        results = [evaluate(indices) for indices in resamples]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # map preserves resample order, keeping reductions deterministic.
            results = list(executor.map(evaluate, resamples))

    coefficients = [coefficient for coefficient, _ in results if coefficient is not None]
    breakpoints = [point for _, point in results if point is not None]
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
