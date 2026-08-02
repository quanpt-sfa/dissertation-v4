"""Regression tests for serialized empty S3 document identifiers."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

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


def _row(
    *,
    document_id: str = "[]",
    target_fiscal_year: int = 2024,
    endpoint: str = "S3_BROAD",
    source_ref: object | None = None,
) -> dict[str, object]:
    provenance = (
        {"decision_number": "123/QD-XPVPHC", "publish_date": "2025-03-31"}
        if source_ref is None
        else source_ref
    )
    canonical_ref = (
        provenance
        if isinstance(provenance, str)
        else json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return {
        "document_id": document_id,
        "firm_id": "E706",
        "target_fiscal_year": target_fiscal_year,
        "primary_violation_l1": None,
        "primary_violation_l2": None,
        "construct_family": "AGGREGATED_PUBLIC_ENDPOINT",
        "construct_target": json.dumps([endpoint], ensure_ascii=False),
        "normalized_violation_code": None,
        "hard_positive": True,
        "row_inclusion": True,
        "legacy_event_id": None,
        "period_link_source": "precomputed_fiscal_year_endpoint",
        "period_link_confidence": "deterministic_final_build",
        "source_record_refs": json.dumps([canonical_ref], ensure_ascii=False),
        "taxonomy_codes": json.dumps(["FS_FALSE_MISLEADING"], ensure_ascii=False),
        "taxonomy_reason_code": "FINAL_ENDPOINT_PROVENANCE_JSON",
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
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


def test_placeholder_document_id_uses_stable_provenance_key_and_collapses() -> None:
    registry = _registry()
    firm_column = physical_columns(registry)[FIRM_ID]
    rows = [
        _row(target_fiscal_year=2022, endpoint="S3_REPORTING"),
        _row(target_fiscal_year=2024, endpoint="S3_CONTENT"),
    ]

    ledger = bind_sanction_decision_ledger_columns(
        _frame(rows),
        firm_column=firm_column,
    )

    assert len(ledger) == 1
    assert str(ledger.loc[0, "document_id"]).startswith("embedded-provenance-")
    assert not ledger.duplicated(["document_id", firm_column]).any()
    assert pd.isna(ledger.loc[0, "target_fiscal_year"])
    assert ledger.loc[0, "period_link_confidence"] == "unresolved_multiple_target_years"
    assert (
        ledger.loc[0, "taxonomy_reason_code"]
        == "MULTIPLE_TARGET_FISCAL_YEARS_IN_FINAL_PROVENANCE"
    )
    assert json.loads(str(ledger.loc[0, "construct_target"])) == [
        "S3_CONTENT",
        "S3_REPORTING",
    ]
    contract_registry(registry).validate("sanction_decision_ledger", ledger)


def test_placeholder_document_id_with_empty_provenance_fails_closed() -> None:
    registry = _registry()
    firm_column = physical_columns(registry)[FIRM_ID]

    with pytest.raises(
        ValueError,
        match="empty-container document identifier without non-empty provenance",
    ):
        bind_sanction_decision_ledger_columns(
            _frame([_row(source_ref="[]")]),
            firm_column=firm_column,
        )


def test_real_document_id_is_not_rewritten() -> None:
    registry = _registry()
    firm_column = physical_columns(registry)[FIRM_ID]

    ledger = bind_sanction_decision_ledger_columns(
        _frame([_row(document_id="DOC-123")]),
        firm_column=firm_column,
    )

    assert ledger.loc[0, "document_id"] == "DOC-123"
    contract_registry(registry).validate("sanction_decision_ledger", ledger)
