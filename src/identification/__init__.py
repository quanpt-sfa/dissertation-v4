"""Partial-identification and robust model-selection utilities."""

from .service import (
    build_identification_summary,
    minimax_regret,
    model_selection_set,
    paired_difference_dominance,
    prevalence_bounds,
    separated_range_dominance,
)

__all__ = [
    "build_identification_summary",
    "minimax_regret",
    "model_selection_set",
    "paired_difference_dominance",
    "prevalence_bounds",
    "separated_range_dominance",
]
