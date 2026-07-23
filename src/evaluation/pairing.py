"""Key-level contracts for paired candidate-reference evaluation."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, LEARNER_ID, PREDICTION


def validate_paired_prediction_keys(
    predictions: pd.DataFrame,
    columns: dict[str, str],
) -> dict[str, object]:
    """Require exact one-to-one firm-year support for every locked ablation pair."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    prediction = columns[PREDICTION]
    required = {firm, year, learner, prediction}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"P12 prediction frame missing columns {missing}")
    if predictions.empty:
        raise RuntimeError("P12_PAIRED_COMPARISON_BLOCKED: OUTER_PREDICTIONS_EMPTY")
    if predictions.duplicated([learner, firm, year]).any():
        raise RuntimeError("P12_PAIRED_COMPARISON_BLOCKED: DUPLICATE_MODEL_FIRM_YEAR")

    model_ids = sorted(str(value) for value in predictions[learner].unique())
    pair_rows: list[dict[str, Any]] = []
    for candidate in (value for value in model_ids if value.endswith(":full")):
        reference = candidate.removesuffix(":full") + ":observability_only"
        if reference not in model_ids:
            raise RuntimeError(
                f"P12_PAIRED_COMPARISON_BLOCKED: REFERENCE_MODEL_MISSING:{candidate}"
            )
        candidate_frame = predictions.loc[
            predictions[learner].astype(str).eq(candidate), [firm, year, prediction]
        ].copy()
        reference_frame = predictions.loc[
            predictions[learner].astype(str).eq(reference), [firm, year, prediction]
        ].copy()
        candidate_keys = pd.MultiIndex.from_frame(candidate_frame[[firm, year]])
        reference_keys = pd.MultiIndex.from_frame(reference_frame[[firm, year]])
        candidate_only = candidate_keys.difference(reference_keys)
        reference_only = reference_keys.difference(candidate_keys)
        if len(candidate_only) or len(reference_only):
            raise RuntimeError(
                "P12_PAIRED_COMPARISON_BLOCKED: FIRM_YEAR_SUPPORT_MISMATCH:"
                f"{candidate}:candidate_only={len(candidate_only)}:reference_only={len(reference_only)}"
            )
        candidate_truth_order = candidate_frame.sort_values([firm, year], kind="stable")[[firm, year]]
        reference_truth_order = reference_frame.sort_values([firm, year], kind="stable")[[firm, year]]
        if not candidate_truth_order.reset_index(drop=True).equals(
            reference_truth_order.reset_index(drop=True)
        ):
            raise RuntimeError(
                f"P12_PAIRED_COMPARISON_BLOCKED: PAIR_ORDER_NOT_RECONCILABLE:{candidate}"
            )
        pair_rows.append(
            {
                "candidate": candidate,
                "reference": reference,
                "paired_firm_year_count": len(candidate_frame),
                "candidate_only_count": 0,
                "reference_only_count": 0,
                "key_contract": "exact_firm_id_fiscal_year_match",
            }
        )
    if not pair_rows:
        raise RuntimeError("P12_PAIRED_COMPARISON_BLOCKED: NO_FULL_VS_OBSERVABILITY_PAIRS")
    return {
        "status": "PASS",
        "pair_count": len(pair_rows),
        "pairs": pair_rows,
        "duplicate_model_firm_year_count": 0,
    }


def paired_candidate_ids(audit: dict[str, object]) -> list[str]:
    raw = audit.get("pairs")
    if not isinstance(raw, list):
        return []
    return [
        str(cast(dict[str, object], row).get("candidate"))
        for row in raw
        if isinstance(row, dict)
    ]
