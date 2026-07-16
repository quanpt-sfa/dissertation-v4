from __future__ import annotations

import pandas as pd

from features.literature_audit import (
    BENEISH_COMPONENTS,
    DECHOW_MODEL_1_COMPONENTS,
    canonical_item_lookup,
    classify_mapping,
    paired_coverage,
    stable_candidate_items,
)


def test_p07b_canonical_benchmark_lists_are_complete() -> None:
    assert len(BENEISH_COMPONENTS) == 8
    assert len(DECHOW_MODEL_1_COMPONENTS) == 7


def test_mapping_never_accepts_missing_or_prohibited_source() -> None:
    assert classify_mapping(None, authorised=True) == "CONCEPT_UNAVAILABLE"
    assert classify_mapping("profit_after_tax", authorised=False) == "SOURCE_NOT_AUTHORIZED"
    assert classify_mapping("accounts_receivable", authorised=True, approval=True).endswith(
        "REQUIRES_APPROVAL"
    )


def test_two_year_coverage_requires_consecutive_years() -> None:
    frame = pd.DataFrame(
        {
            "issuer_ticker": ["A", "A", "B"],
            "fiscal_year": [2020, 2021, 2021],
            "canonical_item": ["x", "x", "x"],
        }
    )
    assert paired_coverage(frame, "x") == (1, 1 / 3)


def test_prepost_eligibility_is_rule_based_not_target_based() -> None:
    frame = pd.DataFrame(
        {
            "canonical_item": ["b", "a", "a"],
            "pair_complete_flag": [1, 1, 0],
            "audit_delta_post_minus_pre": [1.0, 0.0, 2.0],
        }
    )
    result = stable_candidate_items(frame)
    assert result.canonical_item.tolist() == ["a", "b"]
    assert result.loc[result.canonical_item.eq("a"), "passes_coverage"].item() is False


def test_lookup_is_deterministic_and_does_not_impute() -> None:
    assert canonical_item_lookup(["current_assets"], ["cash", "current_assets"]) == "current_assets"
    assert canonical_item_lookup(["current_assets"], ["cash"]) is None
