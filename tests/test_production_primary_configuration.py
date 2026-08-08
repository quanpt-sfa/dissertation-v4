"""Production-registry tests for the sequential primary track and S1 rules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

from core.evidence_registry import LogicalEvidenceSource
from core.fold_control import require_primary_target
from core.registry_compiler import compile_registry
from evidence.annual import AdjustmentRow, build_audit_adjustment_records

ROOT = Path(__file__).resolve().parents[1]
S1_PROFILE_ID = "financial_statement_core_long"


def _registry() -> dict[str, Any]:
    return compile_registry(ROOT / "config" / "pipeline.yaml").registry


def _s1_catalog_configuration() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    registry = _registry()
    catalog = cast(dict[str, Any], registry["source_catalog"])
    profiles = cast(dict[str, Any], catalog["profiles"])
    profile = cast(dict[str, Any], profiles[S1_PROFILE_ID])
    evidence = cast(dict[str, Any], profile["evidence_mapping"])
    logical_sources = {
        str(item["source_id"]): cast(dict[str, Any], item)
        for item in cast(list[dict[str, Any]], evidence["logical_sources"])
    }
    return profile, logical_sources


def _logical_s1_source(source_id: str) -> LogicalEvidenceSource:
    profile, logical_sources = _s1_catalog_configuration()
    evidence = cast(dict[str, Any], profile["evidence_mapping"])
    logical = logical_sources[source_id]
    return LogicalEvidenceSource(
        source_id=source_id,
        endpoint_id=str(logical["endpoint_id"]),
        channel_id=str(profile["channel_id"]),
        physical_source_id=S1_PROFILE_ID,
        profile_id=S1_PROFILE_ID,
        processor=str(evidence["processor"]),
        temporal_role=str(evidence["temporal_role"]),
        availability_rule=str(evidence["availability_rule"]),
        explicit_negative_allowed=bool(evidence["explicit_negative_allowed"]),
        verification_status=str(profile["verification_status"]),
        opportunity_semantic=str(evidence["opportunity_semantic"]),
        logical_config=dict(logical),
        processor_config=dict(evidence),
    )


def _adjustment_row(
    *,
    audit_status: str,
    value: float,
    source_ref: str,
    canonical_item: str,
) -> AdjustmentRow:
    return AdjustmentRow(
        firm_id="F1",
        fiscal_year=2020,
        canonical_item=canonical_item,
        unit="VND",
        statement_scope="consolidated",
        statement_family="income_statement",
        audit_status=audit_status,
        value=value,
        source_ref=source_ref,
    )


def test_production_primary_target_resolves_to_l1_annual() -> None:
    registry = _registry()
    measurement = cast(dict[str, Any], registry["measurement"])
    assert require_primary_target(measurement, "TEST") == "L1_ANNUAL"


def test_p13_exchange_domain_methodology_is_locked() -> None:
    registry = _registry()
    evaluation = cast(dict[str, Any], registry["evaluation"])
    domain = cast(dict[str, Any], evaluation["domain_transfer"])
    bindings = cast(list[dict[str, Any]], domain["bindings"])
    gate2 = cast(dict[str, Any], evaluation["gate2"])

    assert bindings == [
        {
            "domain_id": "exchange_or_board",
            "column": "exchange_or_board",
            "source_field_semantic": "exchange_at_fye",
            "allowed_levels": ["HOSE", "HNX"],
        }
    ]
    assert domain["time_basis"] == "fiscal_year_end"
    assert domain["migration_policy"] == "row_level_board_at_fiscal_year_end"
    assert domain["support_method"] == (
        "cross_fitted_balanced_logistic_domain_score_held_test_fraction"
    )
    assert domain["support_score_bounds"] == [0.05, 0.95]
    assert domain["support_fraction_minimum"] == 0.8
    assert domain["support_crossfit_folds"] == 5
    assert domain["support_crossfit_group"] == "firm_id"
    assert domain["candidate_specific"] is True
    assert len(bindings[0]["allowed_levels"]) * int(gate2["fold_count"]) == 8


def test_s1_materiality_rules_are_locked_in_registry() -> None:
    profile, logical_sources = _s1_catalog_configuration()
    evidence = cast(dict[str, Any], profile["evidence_mapping"])
    adjustment = cast(dict[str, Any], evidence["audit_adjustment"])

    assert logical_sources["S1_profit_adjustment"]["materiality_threshold"] == 0.10
    assert logical_sources["S1_revenue_adjustment"]["materiality_threshold"] == 0.01
    assert adjustment["minimum_absolute_denominator"] == 0.0


def test_zero_floor_means_only_zero_denominator_is_invalid() -> None:
    source = _logical_s1_source("S1_profit_adjustment")
    anchor = {("F1", 2020): datetime(2021, 3, 31)}
    canonical_item = str(source.logical_config["canonical_item"])

    valid = build_audit_adjustment_records(
        panel_anchors=anchor,
        rows=[
            _adjustment_row(
                audit_status="unaudited",
                value=120.0,
                source_ref="pre",
                canonical_item=canonical_item,
            ),
            _adjustment_row(
                audit_status="audited",
                value=100.0,
                source_ref="post",
                canonical_item=canonical_item,
            ),
        ],
        sources=[source],
    ).records[0]
    zero = build_audit_adjustment_records(
        panel_anchors=anchor,
        rows=[
            _adjustment_row(
                audit_status="unaudited",
                value=100.0,
                source_ref="pre",
                canonical_item=canonical_item,
            ),
            _adjustment_row(
                audit_status="audited",
                value=0.0,
                source_ref="post",
                canonical_item=canonical_item,
            ),
        ],
        sources=[source],
    ).records[0]

    assert valid.outcome is True
    assert valid.source_opportunity is True
    assert zero.outcome is None
    assert zero.source_opportunity is False
    assert zero.outcome_basis == "S1_INVALID_DENOMINATOR_ZERO"
