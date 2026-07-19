"""Production-registry tests for the sequential primary track and S1 rules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from core.evidence_registry import logical_evidence_sources
from core.fold_control import require_primary_target
from core.registry_compiler import compile_registry
from evidence.annual import AdjustmentRow, build_audit_adjustment_records

ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


def test_production_primary_target_resolves_to_l1_annual() -> None:
    registry = _registry()
    measurement = cast(dict[str, Any], registry["measurement"])
    assert require_primary_target(measurement, "TEST") == "L1_ANNUAL"


def test_s1_materiality_rules_are_locked_in_registry() -> None:
    sources = logical_evidence_sources(_registry())
    profit = sources["S1_profit_adjustment"]
    revenue = sources["S1_revenue_adjustment"]

    assert profit.logical_config["materiality_threshold"] == 0.10
    assert revenue.logical_config["materiality_threshold"] == 0.01
    assert profit.processor_config["audit_adjustment"]["minimum_absolute_denominator"] == 0.0


def test_zero_floor_means_only_zero_denominator_is_invalid() -> None:
    source = logical_evidence_sources(_registry())["S1_profit_adjustment"]
    anchor = {("F1", 2020): datetime(2021, 3, 31)}
    common = {
        "firm_id": "F1",
        "fiscal_year": 2020,
        "canonical_item": source.logical_config["canonical_item"],
        "unit": "VND",
        "statement_scope": "consolidated",
        "statement_family": "income_statement",
    }

    valid = build_audit_adjustment_records(
        panel_anchors=anchor,
        rows=[
            AdjustmentRow(audit_status="unaudited", value=120.0, source_ref="pre", **common),
            AdjustmentRow(audit_status="audited", value=100.0, source_ref="post", **common),
        ],
        sources=[source],
    ).records[0]
    zero = build_audit_adjustment_records(
        panel_anchors=anchor,
        rows=[
            AdjustmentRow(audit_status="unaudited", value=100.0, source_ref="pre", **common),
            AdjustmentRow(audit_status="audited", value=0.0, source_ref="post", **common),
        ],
        sources=[source],
    ).records[0]

    assert valid.outcome is True
    assert valid.source_opportunity is True
    assert zero.outcome is None
    assert zero.source_opportunity is False
    assert zero.outcome_basis == "S1_INVALID_DENOMINATOR_ZERO"
