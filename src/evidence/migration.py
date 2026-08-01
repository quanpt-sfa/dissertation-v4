"""Fail-closed migration of the evidence ledger from schema v6 to v7.

Schema v7 materialises the source-specific observation process ``O-V-D-R-S``.
The migration preserves the legacy ``source_opportunity`` and ``outcome`` aliases
for downstream compatibility, but never infers unobserved verification or
adjudication layers from an observed zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.semantic_keys import (
    DETERMINATION_DATE,
    DETERMINATION_STATUS,
    EVIDENCE_RECORD_KIND,
    LAYER_OBSERVED_MASK,
    OBSERVATION_OPPORTUNITY,
    OBSERVED_SOURCE_LABEL,
    OUTCOME,
    REASON_UNKNOWN,
    RECORDING_DATE,
    RECORDING_STATUS,
    SOURCE_OPPORTUNITY,
    VERIFICATION_DATE,
    VERIFICATION_STATUS,
)


@dataclass(frozen=True)
class EvidenceLedgerMigration:
    frame: pd.DataFrame
    audit: dict[str, object]


def migrate_evidence_ledger_v6_to_v7(
    frame: pd.DataFrame,
    *,
    ordered_columns: list[str] | None = None,
) -> EvidenceLedgerMigration:
    """Materialise O-V-D-R-S without promoting endpoint zero to latent negative.

    Public positive S3 endpoint rows provide evidence that the source-specific
    verification, determination, and recording stages occurred. All other legacy
    rows retain unknown V/D/R unless those fields were already present. No dates
    are fabricated during migration.
    """

    required = {SOURCE_OPPORTUNITY, OUTCOME, EVIDENCE_RECORD_KIND}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v6 evidence ledger missing required columns={missing}")

    result = frame.copy()
    _materialise_alias(result, OBSERVATION_OPPORTUNITY, SOURCE_OPPORTUNITY)
    _materialise_alias(result, OBSERVED_SOURCE_LABEL, OUTCOME)

    for column in (VERIFICATION_STATUS, DETERMINATION_STATUS, RECORDING_STATUS):
        if column not in result.columns:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        else:
            result[column] = result[column].astype("boolean")

    kind = result[EVIDENCE_RECORD_KIND].astype("string")
    source_label = result[OBSERVED_SOURCE_LABEL].astype("boolean")
    public_positive = kind.eq("firm_year_endpoint_result") & source_label.eq(True)
    for column in (VERIFICATION_STATUS, DETERMINATION_STATUS, RECORDING_STATUS):
        conflict = public_positive & result[column].notna() & ~result[column].eq(True)
        if bool(conflict.any()):
            raise ValueError(
                f"legacy row conflicts with positive public endpoint process at column={column}"
            )
        result.loc[public_positive, column] = True
        result[column] = result[column].astype("boolean")

    for column in (VERIFICATION_DATE, DETERMINATION_DATE, RECORDING_DATE):
        if column not in result.columns:
            result[column] = pd.NaT
        result[column] = pd.to_datetime(result[column]).astype("datetime64[ns]")

    expected_masks = result.apply(_row_mask, axis=1).astype("string")
    if LAYER_OBSERVED_MASK in result.columns:
        existing = result[LAYER_OBSERVED_MASK].astype("string")
        conflict = existing.notna() & ~existing.eq(expected_masks)
        if bool(conflict.any()):
            raise ValueError("legacy layer_observed_mask conflicts with materialised O-V-D-R-S")
    result[LAYER_OBSERVED_MASK] = expected_masks

    expected_reasons = pd.Series(
        "SOURCE_LABEL_UNKNOWN",
        index=result.index,
        dtype="string",
    )
    source_label_observed = result[OBSERVED_SOURCE_LABEL].notna()
    opportunity = result[OBSERVATION_OPPORTUNITY].astype("boolean")
    recording = result[RECORDING_STATUS].astype("boolean")
    expected_reasons = expected_reasons.mask(source_label_observed)
    expected_reasons.loc[~source_label_observed & opportunity.isna()] = (
        "OBSERVATION_OPPORTUNITY_UNKNOWN"
    )
    expected_reasons.loc[~source_label_observed & opportunity.eq(False)] = (
        "NO_OBSERVATION_OPPORTUNITY"
    )
    expected_reasons.loc[~source_label_observed & opportunity.eq(True) & recording.eq(False)] = (
        "DETERMINATION_NOT_RECORDED"
    )
    if REASON_UNKNOWN in result.columns:
        existing_reason = result[REASON_UNKNOWN].astype("string")
        observed_with_reason = source_label_observed & existing_reason.notna()
        if bool(observed_with_reason.any()):
            raise ValueError("reason_unknown cannot be populated when S is observed")
        expected_reasons = existing_reason.fillna(expected_reasons)
    result[REASON_UNKNOWN] = expected_reasons

    for column in (
        SOURCE_OPPORTUNITY,
        OBSERVATION_OPPORTUNITY,
        VERIFICATION_STATUS,
        DETERMINATION_STATUS,
        RECORDING_STATUS,
        OBSERVED_SOURCE_LABEL,
        OUTCOME,
    ):
        result[column] = result[column].astype("boolean")

    if not result[SOURCE_OPPORTUNITY].equals(result[OBSERVATION_OPPORTUNITY]):
        raise ValueError("legacy source_opportunity differs from canonical O")
    if not result[OUTCOME].equals(result[OBSERVED_SOURCE_LABEL]):
        raise ValueError("legacy outcome differs from canonical S")

    if ordered_columns is not None:
        missing_target = sorted(set(ordered_columns) - set(result.columns))
        unexpected = sorted(set(result.columns) - set(ordered_columns))
        if missing_target:
            raise ValueError(f"migration cannot satisfy v7 schema; missing={missing_target}")
        result = result.loc[:, ordered_columns].copy()
    else:
        unexpected = []

    return EvidenceLedgerMigration(
        frame=result,
        audit={
            "from_schema_version": 6,
            "to_schema_version": 7,
            "row_count": int(len(result)),
            "public_positive_rows_with_materialised_vdr": int(public_positive.sum()),
            "observed_endpoint_zero_rows": int(
                result[OBSERVED_SOURCE_LABEL].astype("boolean").eq(False).sum()
            ),
            "rows_with_unknown_v": int(result[VERIFICATION_STATUS].isna().sum()),
            "dates_fabricated": False,
            "legacy_aliases_retained": [SOURCE_OPPORTUNITY, OUTCOME],
            "unexpected_columns_ignored": unexpected,
        },
    )


def _materialise_alias(frame: pd.DataFrame, canonical: str, legacy: str) -> None:
    legacy_values = frame[legacy].astype("boolean")
    if canonical in frame.columns:
        canonical_values = frame[canonical].astype("boolean")
        if not canonical_values.equals(legacy_values):
            raise ValueError(f"canonical column={canonical} conflicts with legacy alias={legacy}")
    frame[legacy] = legacy_values
    frame[canonical] = legacy_values.copy()


def _row_mask(row: pd.Series) -> str:
    return "".join(
        name if not pd.isna(row.get(column)) else "-"
        for name, column in (
            ("O", OBSERVATION_OPPORTUNITY),
            ("V", VERIFICATION_STATUS),
            ("D", DETERMINATION_STATUS),
            ("R", RECORDING_STATUS),
            ("S", OBSERVED_SOURCE_LABEL),
        )
    )
