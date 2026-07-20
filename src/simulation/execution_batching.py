"""Execution-only batching controls for P08 simulation artifacts.

The configured replication plan continues to determine minimum and maximum
replications and MCSE stopping.  This module only coalesces several configured
chunks into one subprocess/artifact so that checkpoint granularity remains
practical for dissertation production runs.
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
    maximum_replications: int,
    batch_multiplier: int,
) -> int:
    """Return artifact batch size without changing the replication budget."""
    if configured_batch_size < 1:
        raise ValueError("configured batch size must be positive")
    if maximum_replications < 1:
        raise ValueError("maximum replications must be positive")
    multiplier = validate_batch_multiplier(batch_multiplier)
    return min(maximum_replications, configured_batch_size * multiplier)


def planned_batch_count(*, replications: int, batch_size: int) -> int:
    """Count artifact batches needed for a replication total."""
    if replications < 0:
        raise ValueError("replications must be nonnegative")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    return math.ceil(replications / batch_size) if replications else 0
