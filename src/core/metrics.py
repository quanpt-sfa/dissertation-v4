"""Dependency-light prediction metrics used by evaluation and simulation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def average_precision(outcomes: Sequence[bool], scores: Sequence[float]) -> float | None:
    _same_length(outcomes, scores)
    positives = sum(bool(value) for value in outcomes)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, outcomes, strict=True), key=lambda item: item[0], reverse=True)
    hits = 0
    total = 0.0
    for rank, (_, outcome) in enumerate(ranked, start=1):
        if outcome:
            hits += 1
            total += hits / rank
    return total / positives


def roc_auc(outcomes: Sequence[bool], scores: Sequence[float]) -> float | None:
    _same_length(outcomes, scores)
    positive = [score for score, outcome in zip(scores, outcomes, strict=True) if outcome]
    negative = [score for score, outcome in zip(scores, outcomes, strict=True) if not outcome]
    if not positive or not negative:
        return None
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            wins += float(positive_score > negative_score) + 0.5 * float(
                positive_score == negative_score
            )
    return wins / (len(positive) * len(negative))


def brier_score(outcomes: Sequence[bool], probabilities: Sequence[float]) -> float:
    _same_length(outcomes, probabilities)
    if not outcomes:
        raise ValueError("metrics require at least one observation")
    return sum(
        (float(outcome) - probability) ** 2
        for outcome, probability in zip(outcomes, probabilities, strict=True)
    ) / len(outcomes)


def precision_at_fraction(
    outcomes: Sequence[bool], scores: Sequence[float], fraction: float
) -> float:
    _same_length(outcomes, scores)
    if not outcomes or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1] and data must be nonempty")
    count = max(1, math.ceil(len(outcomes) * fraction))
    ranked = sorted(zip(scores, outcomes, strict=True), key=lambda item: item[0], reverse=True)
    return sum(bool(outcome) for _, outcome in ranked[:count]) / count


def _same_length(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError("metric inputs must have equal length")
