"""Adapters from internal evidence objects to physical artifact contracts."""

from __future__ import annotations

import pandas as pd

from core.semantic_keys import (
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    DOCUMENT_ID,
    FIRM_ID,
    HARD_POSITIVE,
    LEGACY_EVENT_ID,
    NORMALIZED_VIOLATION_CODE,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    ROW_INCLUSION,
    SOURCE_RECORD_REFS,
    TARGET_FISCAL_YEAR,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
)

_DECISION_LEDGER_TAIL = [
    TARGET_FISCAL_YEAR,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    NORMALIZED_VIOLATION_CODE,
    HARD_POSITIVE,
    ROW_INCLUSION,
    LEGACY_EVENT_ID,
    PERIOD_LINK_SOURCE,
    PERIOD_LINK_CONFIDENCE,
    SOURCE_RECORD_REFS,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
]


def bind_sanction_decision_ledger_columns(
    frame: pd.DataFrame,
    *,
    firm_column: str,
) -> pd.DataFrame:
    """Bind the internal firm key to the compiled physical ledger contract.

    The wide S3 builder uses the logical firm key internally. Production
    artifacts use the physical firm column compiled from the registry. This
    adapter performs only that deterministic rename and exact-order check.
    """

    if not isinstance(firm_column, str) or not firm_column:
        raise ValueError("sanction decision ledger requires a physical firm column")

    result = frame.copy()
    if firm_column != FIRM_ID:
        if firm_column in result.columns and FIRM_ID in result.columns:
            raise ValueError("sanction decision ledger contains both internal and physical firm keys")
        if firm_column not in result.columns:
            if FIRM_ID not in result.columns:
                raise ValueError("sanction decision ledger is missing its internal firm key")
            result = result.rename(columns={FIRM_ID: firm_column})
    elif firm_column not in result.columns:
        raise ValueError("sanction decision ledger is missing its physical firm key")

    ordered = [DOCUMENT_ID, firm_column, *_DECISION_LEDGER_TAIL]
    missing = [column for column in ordered if column not in result.columns]
    extras = [column for column in result.columns if column not in ordered]
    if missing or extras:
        raise ValueError(
            "sanction decision ledger columns differ from the production contract: "
            f"missing={missing}, extras={extras}"
        )
    return result.loc[:, ordered]
