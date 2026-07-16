# pyright: basic
"""Pure, non-supervised helpers for the P07B feasibility audit.

This module deliberately has no imports from labels, evaluation, known cases, or
pipeline artifacts.  It only describes what is observable in authorised inputs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

BENEISH_COMPONENTS = ("DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA")
DECHOW_MODEL_1_COMPONENTS = (
    "RSST_ACCRUALS",
    "CHANGE_RECEIVABLES",
    "CHANGE_INVENTORY",
    "SOFT_ASSETS",
    "CHANGE_CASH_SALES",
    "CHANGE_ROA",
    "ACTUAL_ISSUANCE",
)


def sha256(path: Path) -> str:
    """Return a content hash without loading a complete source into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_item_lookup(items: Iterable[object], candidates: Iterable[str]) -> str | None:
    """Return an exact canonical-item match from an ordered candidate vocabulary."""
    observed = {str(item) for item in items if item is not None and str(item) != "nan"}
    return next((candidate for candidate in candidates if candidate in observed), None)


def paired_coverage(frame: pd.DataFrame, item: str) -> tuple[int, float]:
    """Coverage of consecutive firm-years for a single production accounting item."""
    subset = frame.loc[frame["canonical_item"].eq(item), ["issuer_ticker", "fiscal_year"]]
    if subset.empty:
        return 0, 0.0
    unique = subset.drop_duplicates()
    previous = unique.assign(fiscal_year=unique["fiscal_year"] + 1)
    joined = unique.merge(
        previous,
        left_on=["issuer_ticker", "fiscal_year"],
        right_on=["issuer_ticker", "fiscal_year"],
        how="inner",
    )
    denominator = len(unique)
    return len(joined), 0.0 if denominator == 0 else len(joined) / denominator


def classify_mapping(exact_item: str | None, *, authorised: bool, approval: bool = False) -> str:
    """Classify evidence without accepting a near substitute automatically."""
    if not authorised:
        return "SOURCE_NOT_AUTHORIZED"
    if exact_item is None:
        return "CONCEPT_UNAVAILABLE"
    if approval:
        return "DEFENSIBLE_VAS_MAPPING_REQUIRES_APPROVAL"
    return "EXACT_MAPPING"


def stable_candidate_items(frame: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Rule-based pre/post eligibility; no labels, ranks, floors, or imputation."""
    grouped = frame.groupby("canonical_item", dropna=False).agg(
        paired_coverage=("pair_complete_flag", "mean"),
        nonzero_adjustments=("audit_delta_post_minus_pre", lambda values: int((values != 0).sum())),
    )
    grouped = grouped.reset_index()
    grouped["passes_coverage"] = grouped["paired_coverage"] >= threshold
    return grouped.sort_values("canonical_item", kind="stable").reset_index(drop=True)
