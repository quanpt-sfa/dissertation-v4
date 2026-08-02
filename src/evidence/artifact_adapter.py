"""Adapters from internal evidence objects to physical artifact contracts."""

from __future__ import annotations

import pandas as pd


_DECISION_LEDGER_TAIL = [
    "target_fiscal_year",
    "primary_violation_l1",
    "primary_violation_l2",
    "construct_family",
    "construct_target",
    "normalized_violation_code",
    "hard_positive",
    "row_inclusion",
    "legacy_event_id",
    "period_link_source",
    "period_link_confidence",
    "source_record_refs",
    "taxonomy_codes",
    "taxonomy_reason_code",
]


def bind_sanction_decision_ledger_columns(
    frame: pd.DataFrame,
    *,
    firm_column: str,
) -> pd.DataFrame:
    """Bind the internal firm key to the compiled physical ledger contract.

    The wide S3 builder uses ``firm_id`` internally. Production artifacts use
    the physical firm column compiled from ``columns.yaml`` (currently
    ``firm_master_id``). This boundary adapter performs only that deterministic
    rename and then enforces the exact ordered artifact columns.
    """

    if not isinstance(firm_column, str) or not firm_column:
        raise ValueError("sanction decision ledger requires a physical firm column")

    result = frame.copy()
    internal_firm = "firm_id"
    if firm_column != internal_firm:
        if firm_column in result.columns and internal_firm in result.columns:
            raise ValueError("sanction decision ledger contains both internal and physical firm keys")
        if firm_column not in result.columns:
            if internal_firm not in result.columns:
                raise ValueError("sanction decision ledger is missing its internal firm key")
            result = result.rename(columns={internal_firm: firm_column})
    elif firm_column not in result.columns:
        raise ValueError("sanction decision ledger is missing its physical firm key")

    ordered = ["document_id", firm_column, *_DECISION_LEDGER_TAIL]
    missing = [column for column in ordered if column not in result.columns]
    extras = [column for column in result.columns if column not in ordered]
    if missing or extras:
        raise ValueError(
            "sanction decision ledger columns differ from the production contract: "
            f"missing={missing}, extras={extras}"
        )
    return result.loc[:, ordered]
