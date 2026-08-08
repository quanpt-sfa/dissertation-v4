"""Preserve locked domain metadata outside the predictive feature registry."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def restore_domain_metadata(
    *,
    feature_panel: pd.DataFrame,
    source_panel: pd.DataFrame,
    domain_columns: Sequence[str],
    firm_column: str,
    year_column: str,
) -> pd.DataFrame:
    """Restore configured domain columns after P07 feature-panel construction.

    Domain fields are metadata used only by post-outer transport analyses. They
    are keyed by the same firm-year spine, remain outside the feature registry,
    and are never allowed to enter the predictor matrix implicitly.
    """

    normalized = [str(column).strip() for column in domain_columns]
    if any(not column for column in normalized):
        raise ValueError("P07 domain metadata columns must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("P07 domain metadata columns must be unique")
    if not normalized:
        return feature_panel.copy()

    required = {firm_column, year_column, *normalized}
    missing = sorted(required - set(source_panel.columns))
    if missing:
        raise ValueError(f"P07 configured domain metadata is absent from the source panel: {missing}")
    if source_panel.duplicated([firm_column, year_column]).any():
        raise ValueError("P07 domain metadata source requires unique firm-year keys")
    if feature_panel.duplicated([firm_column, year_column]).any():
        raise ValueError("P07 feature panel requires unique firm-year keys")

    metadata = source_panel.loc[:, [firm_column, year_column, *normalized]].copy()
    result = feature_panel.drop(columns=normalized, errors="ignore").merge(
        metadata,
        on=[firm_column, year_column],
        how="left",
        validate="one_to_one",
    )
    if len(result) != len(feature_panel):
        raise ValueError("P07 domain metadata restoration changed the firm-year row count")
    for column in normalized:
        result[column] = result[column].astype("string")
    return result
