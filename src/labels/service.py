"""Missingness-preserving production label functions."""

from __future__ import annotations

import math
from collections.abc import Mapping


def aggregate_l1(source_outcomes: Mapping[str, bool | None]) -> bool | None:
    """Union sources while preserving unknown coverage rather than coding it as zero."""
    values = list(source_outcomes.values())
    if any(value is True for value in values):
        return True
    if values and all(value is False for value in values):
        return False
    return None


def evidence_score_l2(channel_outcomes: Mapping[str, bool | None]) -> float | None:
    """Equal-channel evidence score normalized only over observed channels."""
    observed = [float(value) for value in channel_outcomes.values() if value is not None]
    if not observed:
        return None
    return sum(observed) / len(observed)


def posterior_l3_fixed_pi(
    source_outcomes: Mapping[str, bool | None],
    source_accuracy: Mapping[str, tuple[float, float]],
    fixed_prevalence: float,
) -> float:
    """Conditional-independence L3 posterior for a locked fixed prevalence scenario."""
    if not 0 < fixed_prevalence < 1:
        raise ValueError("fixed L3 prevalence must be in (0, 1)")
    unknown_sources = set(source_outcomes) - set(source_accuracy)
    if unknown_sources:
        raise ValueError(f"L3 accuracy missing for sources {sorted(unknown_sources)}")
    log_odds = math.log(fixed_prevalence / (1.0 - fixed_prevalence))
    for source_id, outcome in source_outcomes.items():
        if outcome is None:
            continue
        sensitivity, specificity = source_accuracy[source_id]
        if not 0 <= sensitivity <= 1 or not 0 <= specificity <= 1:
            raise ValueError("L3 source sensitivity and specificity must be in [0, 1]")
        sensitivity = min(1.0 - 1e-9, max(1e-9, sensitivity))
        specificity = min(1.0 - 1e-9, max(1e-9, specificity))
        if outcome:
            log_odds += math.log(sensitivity / (1.0 - specificity))
        else:
            log_odds += math.log((1.0 - sensitivity) / specificity)
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, log_odds))))
