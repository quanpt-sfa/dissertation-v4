from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any, cast

from core.pipeline import physical_columns
from core.registry_compiler import compile_registry
from core.schema_registry import contract_registry
from core.semantic_keys import (
    DOCUMENT_ID,
    FIRM_ID,
    HARD_POSITIVE,
    ROW_INCLUSION,
    SOURCE_RECORD_REFS,
    TARGET_FISCAL_YEAR,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
)
from evidence.sanctions import SanctionDecisionInput, build_s3_evidence

ROOT = Path(__file__).resolve().parents[2]


@cache
def _registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


@cache
def _columns() -> dict[str, str]:
    return physical_columns(_registry())


@cache
def _taxonomy() -> dict[str, Any]:
    return cast(dict[str, Any], _registry()["s3_taxonomy"])


def _build(decisions: list[SanctionDecisionInput]) -> Any:
    return build_s3_evidence(
        panel_keys={("F1", 2021), ("F1", 2022), ("F1", 2023)},
        decisions=decisions,
        taxonomy=_taxonomy(),
        complete_through_year=2025,
        incomplete_years={2026},
        columns=_columns(),
    )


def test_multiple_excluded_rows_collapse_to_one_decision_firm_mapping() -> None:
    result = _build(
        [
            SanctionDecisionInput(
                document_id="DOC-X",
                firm_id="F1",
                target_fiscal_year=2021,
                normalized_violation_code="DISCLOSURE_UNSPECIFIED",
                row_included=False,
                hard_positive=False,
                source_ref="row-1",
            ),
            SanctionDecisionInput(
                document_id="DOC-X",
                firm_id="F1",
                target_fiscal_year=2022,
                normalized_violation_code="FS_FALSE_MISLEADING",
                row_included=False,
                hard_positive=False,
                source_ref="row-2",
            ),
        ]
    )
    ledger = result.decision_ledger
    columns = _columns()

    assert len(ledger) == 1
    assert not ledger.duplicated([columns[DOCUMENT_ID], columns[FIRM_ID]]).any()
    row = ledger.iloc[0]
    assert bool(row[columns[ROW_INCLUSION]]) is False
    assert bool(row[columns[HARD_POSITIVE]]) is False
    assert row[columns[TAXONOMY_REASON_CODE]] == "EXCLUDED_BY_SOURCE_RULE"
    assert json.loads(row[columns[SOURCE_RECORD_REFS]]) == ["row-1", "row-2"]
    assert json.loads(row[columns[TAXONOMY_CODES]]) == [
        "DISCLOSURE_UNSPECIFIED",
        "FS_FALSE_MISLEADING",
    ]
    assert result.audit["excluded_ledger_rows_collapsed"] == 1
    contract_registry(_registry()).validate("sanction_decision_ledger", ledger)


def test_mixed_group_keeps_included_mapping_and_excludes_other_rows_from_endpoints() -> None:
    result = _build(
        [
            SanctionDecisionInput(
                document_id="DOC-MIXED",
                firm_id="F1",
                sanction_year=2024,
                normalized_violation_code="FS_FALSE_MISLEADING",
                row_included=True,
                hard_positive=True,
                source_ref="included-row",
            ),
            SanctionDecisionInput(
                document_id="DOC-MIXED",
                firm_id="F1",
                target_fiscal_year=2021,
                normalized_violation_code="DISCLOSURE_UNSPECIFIED",
                row_included=False,
                hard_positive=False,
                source_ref="excluded-row",
            ),
        ]
    )
    ledger = result.decision_ledger
    columns = _columns()

    assert len(ledger) == 1
    row = ledger.iloc[0]
    assert bool(row[columns[ROW_INCLUSION]]) is True
    assert bool(row[columns[HARD_POSITIVE]]) is True
    assert int(row[columns[TARGET_FISCAL_YEAR]]) == 2023
    assert json.loads(row[columns[SOURCE_RECORD_REFS]]) == ["included-row"]

    records = {
        (item.source_id, item.firm_id, item.fiscal_year): item for item in result.endpoint_records
    }
    assert records[("S3_CONTENT", "F1", 2023)].outcome is True
    assert result.audit["excluded_source_rule_row_count"] == 1
    assert result.audit["excluded_ledger_rows_collapsed"] == 1
    contract_registry(_registry()).validate("sanction_decision_ledger", ledger)
