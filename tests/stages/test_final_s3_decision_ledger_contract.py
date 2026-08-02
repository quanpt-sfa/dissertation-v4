"""The wide S3 decision ledger must satisfy the compiled artifact contract."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any, cast

import pandas as pd

from core.pipeline import physical_columns
from core.registry_compiler import compile_registry
from core.schema_registry import contract_registry
from core.semantic_keys import FIRM_ID
from evidence.artifact_adapter import bind_sanction_decision_ledger_columns

ROOT = Path(__file__).resolve().parents[2]


@cache
def _registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


def _internal_ledger() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "document_id": "DOC-1",
                "firm_id": "E706",
                "target_fiscal_year": 2024,
                "primary_violation_l1": None,
                "primary_violation_l2": None,
                "construct_family": "AGGREGATED_PUBLIC_ENDPOINT",
                "construct_target": '["S3_BROAD"]',
                "normalized_violation_code": None,
                "hard_positive": True,
                "row_inclusion": True,
                "legacy_event_id": None,
                "period_link_source": "precomputed_fiscal_year_endpoint",
                "period_link_confidence": "deterministic_final_build",
                "source_record_refs": '["source-ref"]',
                "taxonomy_codes": '["DISCLOSURE_UNSPECIFIED"]',
                "taxonomy_reason_code": "FINAL_ENDPOINT_PROVENANCE_JSON",
            }
        ]
    )
    string_columns = [
        column
        for column in frame.columns
        if column not in {"target_fiscal_year", "hard_positive", "row_inclusion"}
    ]
    for column in string_columns:
        frame[column] = frame[column].astype("string")
    frame["target_fiscal_year"] = frame["target_fiscal_year"].astype("Int16")
    frame["hard_positive"] = frame["hard_positive"].astype("boolean")
    frame["row_inclusion"] = frame["row_inclusion"].astype("boolean")
    return frame


def test_wide_s3_ledger_binds_internal_firm_id_to_physical_contract() -> None:
    registry = _registry()
    columns = physical_columns(registry)
    firm_column = columns[FIRM_ID]

    ledger = bind_sanction_decision_ledger_columns(
        _internal_ledger(),
        firm_column=firm_column,
    )

    assert firm_column == "firm_master_id"
    assert "firm_id" not in ledger.columns
    assert list(ledger.columns) == [
        "document_id",
        firm_column,
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
    assert ledger.loc[0, firm_column] == "E706"
    contract_registry(registry).validate("sanction_decision_ledger", ledger)
