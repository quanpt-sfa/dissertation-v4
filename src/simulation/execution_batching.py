"""Execution-only batching controls for P08 simulation artifacts.

The configured replication plan continues to determine minimum and maximum
replications and MCSE stopping. This module only coalesces several configured
chunks into one subprocess/artifact so checkpoint granularity remains practical
for dissertation production runs.
"""

from __future__ import annotations

import math


DEFAULT_BATCH_MULTIPLIER = 5


def validate_batch_multiplier(value: int) -> int:
    """Require a positive execution batch multiplier."""
    if value < 1:
        raise ValueError("batch multiplier must be a positive integer")
    return value


def execution_batch_size(
    *,
    configured_batch_size: int,
    minimum_replications: int,
    maximum_replications: int,
    batch_multiplier: int,
) -> int:
    """Return an aligned artifact batch size without changing replication counts.

    The initial minimum must end on an execution-batch boundary so adaptive
    continuation can start from ``completed`` without changing coordinate
    semantics. The largest multiplier not exceeding the requested value that
    preserves that boundary is selected.
    """
    if configured_batch_size < 1:
        raise ValueError("configured batch size must be positive")
    if minimum_replications < 1:
        raise ValueError("minimum replications must be positive")
    if maximum_replications < minimum_replications:
        raise ValueError("maximum replications must be at least the minimum")
    if minimum_replications % configured_batch_size != 0:
        raise ValueError("minimum replications must align with configured batch size")

    requested = validate_batch_multiplier(batch_multiplier)
    maximum_factor = minimum_replications // configured_batch_size
    factor = min(requested, maximum_factor)
    while factor > 1 and minimum_replications % (configured_batch_size * factor) != 0:
        factor -= 1
    return min(maximum_replications, configured_batch_size * factor)


def planned_batch_count(*, replications: int, batch_size: int) -> int:
    """Count artifact batches needed for a replication total."""
    if replications < 0:
        raise ValueError("replications must be nonnegative")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    return math.ceil(replications / batch_size) if replications else 0
