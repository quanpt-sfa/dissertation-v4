"""Regression coverage for production S2 vocabulary and outer-fold dispatch."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from core.evidence_registry import logical_evidence_sources
from core.registry_compiler import compile_registry
from evidence.annual import OpinionRow, build_audit_opinion_records
from scripts.run_pipeline import _fold_execution_sets

ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict[str, object]:
    return compile_registry(ROOT / "config" / "pipeline.yaml").registry


def _opinion_row(raw: str) -> OpinionRow:
    return OpinionRow(
        firm_id="F1",
        fiscal_year=2024,
        opinion_raw=raw,
        audit_indicator="audit_opinion",
        period_type="annual",
        statement_scope="consolidated",
        audit_status="audited",
        source_ref="final-parquet:F1:2024",
    )


def test_final_parquet_unmodified_opinion_is_an_explicit_clean_negative() -> None:
    source = logical_evidence_sources(_registry())["S2_audit_opinion"]
    result = build_audit_opinion_records(
        panel_anchors={("F1", 2024): datetime(2025, 3, 31)},
        rows=[_opinion_row("UNMODIFIED")],
        source=source,
    )

    record = result.records[0]
    assert record.outcome is False
    assert record.source_opportunity is True
    assert record.outcome_basis == "S2_CLEAN_OPINION"
    assert result.audit["explicit_negative_count"] == 1
    assert result.audit["unknown_count"] == 0


def test_unknown_opinion_is_not_coerced_to_clean() -> None:
    source = logical_evidence_sources(_registry())["S2_audit_opinion"]
    result = build_audit_opinion_records(
        panel_anchors={("F1", 2024): datetime(2025, 3, 31)},
        rows=[_opinion_row("UNKNOWN")],
        source=source,
    )

    record = result.records[0]
    assert record.outcome is None
    assert record.source_opportunity is False
    assert record.outcome_basis == "S2_OPINION_UNMAPPED"


def test_runner_keeps_initial_fold_for_p09_but_not_confirmatory_stages() -> None:
    p09_folds, confirmatory_folds = _fold_execution_sets(_registry())

    assert p09_folds == ["2020", "2021", "2022", "2023", "2024"]
    assert confirmatory_folds == ["2021", "2022", "2023", "2024"]
