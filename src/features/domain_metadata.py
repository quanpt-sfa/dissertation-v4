"""Preserve locked domain metadata outside the predictive feature registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def restore_domain_metadata(
    *,
    feature_panel: pd.DataFrame,
    source_panel: pd.DataFrame,
    domain_columns: Sequence[str],
    firm_column: str,
    year_column: str,
    allowed_values_by_column: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Restore and validate configured domain metadata after P07 construction.

    Domain fields are metadata used only by post-outer transport analyses. They
    are keyed by the same firm-year spine, remain outside the feature registry,
    and are never allowed to enter the predictor matrix implicitly. A configured
    domain is required to be observed for every row that survives into the P07
    feature panel so leave-one-domain-out refits do not silently change the
    analysis population for reasons other than the held domain itself.
    """

    normalized = [str(column).strip() for column in domain_columns]
    if any(not column for column in normalized):
        raise ValueError("P07 domain metadata columns must be non-empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("P07 domain metadata columns must be unique")
    if not normalized:
        return feature_panel.copy()

    allowed_raw = allowed_values_by_column or {}
    unknown_allowed_columns = sorted(set(allowed_raw) - set(normalized))
    if unknown_allowed_columns:
        raise ValueError(
            f"P07 domain vocabulary was configured for unbound columns: {unknown_allowed_columns}"
        )
    allowed: dict[str, set[str]] = {}
    for column, values in allowed_raw.items():
        normalized_values = [str(value).strip() for value in values]
        if any(not value for value in normalized_values) or not normalized_values:
            raise ValueError(f"P07 domain column={column} requires non-empty allowed values")
        if len(normalized_values) != len(set(normalized_values)):
            raise ValueError(f"P07 domain column={column} allowed values must be unique")
        allowed[column] = set(normalized_values)

    required = {firm_column, year_column, *normalized}
    missing = sorted(required - set(source_panel.columns))
    if missing:
        raise ValueError(
            f"P07 configured domain metadata is absent from the source panel: {missing}"
        )
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
        missing_values = result[column].isna() | result[column].str.strip().eq("")
        if missing_values.any():
            raise ValueError(
                f"P07 configured domain metadata column={column} is incomplete for "
                f"{int(missing_values.sum())} feature-panel row(s)"
            )
        if column in allowed:
            observed = set(result[column].astype(str))
            unexpected = sorted(observed - allowed[column])
            if unexpected:
                raise ValueError(
                    f"P07 configured domain metadata column={column} contains values "
                    f"outside the locked vocabulary: {unexpected}"
                )
    return result
