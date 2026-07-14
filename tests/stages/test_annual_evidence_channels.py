"""Regression tests for annual S1/S2 evidence and delayed S3 verification."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import pandas as pd
import pytest

from core.evidence_registry import LogicalEvidenceSource
from core.semantic_keys import (
    AVAILABILITY_BASIS,
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    PREDICTION_TIME,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    TARGET_ID,
    TEMPORAL_ROLE,
)
from evidence.annual import (
    AdjustmentRow,
    OpinionRow,
    build_audit_adjustment_records,
    build_audit_opinion_records,
)
from features.service import build_feature_panel
from labels.service import aggregate_l1
from measurement.service import build_measurement_inputs
from observability.service import build_observability_registry
from risksets.service import build_risk_set


def _s1_source(
    source_id: str = "S1_profit_adjustment",
    *,
    item: str = "profit_after_tax",
    threshold: float | None = 0.10,
    denominator_floor: float | None = 1.0,
) -> LogicalEvidenceSource:
    return LogicalEvidenceSource(
        source_id=source_id,
        endpoint_id=source_id.removeprefix("S1_"),
        channel_id="S1",
        physical_source_id="financial_statement_core_long",
        profile_id="financial_statement_core_long",
        processor="audit_adjustment",
        temporal_role="annual_measurement_at_anchor",
        availability_rule="common_annual_anchor",
        explicit_negative_allowed=True,
        verification_status="observed_opportunity_only",
        opportunity_semantic="derived_matched_pair",
        logical_config={
            "source_id": source_id,
            "endpoint_id": source_id.removeprefix("S1_"),
            "canonical_item": item,
            "statement_family": "income_statement",
            "materiality_threshold": threshold,
        },
        processor_config={
            "duplicate_representative_rule": "fail_on_duplicate_status_record",
            "audit_adjustment": {
                "unaudited_status": "unaudited",
                "audited_status": "audited",
                "expected_unit": "VND",
                "expected_scope": "consolidated",
                "minimum_absolute_denominator": denominator_floor,
            },
        },
    )


def _s2_source() -> LogicalEvidenceSource:
    return LogicalEvidenceSource(
        source_id="S2_audit_opinion",
        endpoint_id="non_clean_audit_opinion",
        channel_id="S2",
        physical_source_id="audit_annual_long",
        profile_id="audit_annual_long",
        processor="audit_opinion",
        temporal_role="annual_measurement_at_anchor",
        availability_rule="common_annual_anchor",
        explicit_negative_allowed=True,
        verification_status="observed_opportunity_only",
        opportunity_semantic="derived_valid_audit_report",
        logical_config={
            "source_id": "S2_audit_opinion",
            "endpoint_id": "non_clean_audit_opinion",
        },
        processor_config={
            "duplicate_representative_rule": "identical_normalized_opinion_or_fail",
            "audit_opinion": {
                "indicator_value": "audit_opinion",
                "period_type": "annual",
                "statement_scope": "consolidated",
                "audited_status": "audited",
                "raw_to_normalized": {
                    "clean raw": "clean",
                    "qualified raw": "qualified",
                    "disclaimer raw": "disclaimer",
                    "adverse raw": "adverse",
                },
                "positive_categories": ["qualified", "disclaimer", "adverse"],
                "explicit_negative_categories": ["clean"],
                "pending_categories": ["warning", "emphasis_of_matter", "other_matter"],
            },
        },
    )


def _adjustment(
    status: str,
    value: float | None,
    *,
    ref: str,
    firm: str = "F1",
    year: int = 2020,
) -> AdjustmentRow:
    return AdjustmentRow(
        firm_id=firm,
        fiscal_year=year,
        audit_status=status,
        canonical_item="profit_after_tax",
        value=value,
        unit="VND",
        statement_scope="consolidated",
        statement_family="income_statement",
        source_ref=ref,
    )


def _opinion(value: str | None, *, ref: str) -> OpinionRow:
    return OpinionRow(
        firm_id="F1",
        fiscal_year=2020,
        opinion_raw=value,
        audit_indicator="audit_opinion",
        period_type="annual",
        statement_scope="consolidated",
        audit_status="audited",
        source_ref=ref,
    )


def _annual_anchor() -> dict[tuple[str, int], datetime]:
    return {("F1", 2020): datetime(2021, 3, 31)}


@pytest.mark.parametrize(
    ("pre_value", "post_value", "expected"),
    [(120.0, 100.0, True), (105.0, 100.0, False)],
)
def test_s1_matched_pair_materiality_rule(
    pre_value: float, post_value: float, expected: bool
) -> None:
    result = build_audit_adjustment_records(
        panel_anchors=_annual_anchor(),
        rows=[
            _adjustment("unaudited", pre_value, ref="pre"),
            _adjustment("audited", post_value, ref="post"),
        ],
        sources=[_s1_source()],
    )
    assert result.records[0].outcome is expected
    assert result.records[0].source_opportunity is True


def test_s1_missing_pair_and_invalid_denominator_are_unknown() -> None:
    missing = build_audit_adjustment_records(
        panel_anchors=_annual_anchor(),
        rows=[_adjustment("unaudited", 100.0, ref="pre")],
        sources=[_s1_source()],
    ).records[0]
    invalid = build_audit_adjustment_records(
        panel_anchors=_annual_anchor(),
        rows=[
            _adjustment("unaudited", 100.0, ref="pre"),
            _adjustment("audited", 0.0, ref="post"),
        ],
        sources=[_s1_source()],
    ).records[0]
    assert (missing.outcome, missing.source_opportunity, missing.outcome_basis) == (
        None,
        False,
        "S1_PAIR_MISSING_POST_AUDIT",
    )
    assert (invalid.outcome, invalid.source_opportunity, invalid.outcome_basis) == (
        None,
        False,
        "S1_INVALID_DENOMINATOR_ZERO",
    )


def test_s1_conflicting_duplicate_is_order_independent_and_fail_closed() -> None:
    rows = [
        _adjustment("unaudited", 120.0, ref="pre-a"),
        _adjustment("unaudited", 125.0, ref="pre-b"),
        _adjustment("audited", 100.0, ref="post"),
    ]
    first = build_audit_adjustment_records(
        panel_anchors=_annual_anchor(), rows=rows, sources=[_s1_source()]
    ).records[0]
    second = build_audit_adjustment_records(
        panel_anchors=_annual_anchor(), rows=list(reversed(rows)), sources=[_s1_source()]
    ).records[0]
    assert first.outcome is second.outcome is None
    assert first.source_opportunity is second.source_opportunity is False
    assert first.outcome_basis == second.outcome_basis == "S1_DUPLICATE_CONFLICT"
    assert first.source_record_refs == second.source_record_refs


@pytest.mark.parametrize("raw", ["qualified raw", "disclaimer raw", "adverse raw"])
def test_s2_non_clean_opinions_are_positive(raw: str) -> None:
    record = build_audit_opinion_records(
        panel_anchors=_annual_anchor(), rows=[_opinion(raw, ref="opinion")], source=_s2_source()
    ).records[0]
    assert record.outcome is True
    assert record.source_opportunity is True


def test_s2_clean_missing_and_conflicting_opinions_preserve_three_states() -> None:
    clean = build_audit_opinion_records(
        panel_anchors=_annual_anchor(),
        rows=[_opinion("clean raw", ref="clean")],
        source=_s2_source(),
    ).records[0]
    missing = build_audit_opinion_records(
        panel_anchors=_annual_anchor(), rows=[_opinion(None, ref="missing")], source=_s2_source()
    ).records[0]
    conflict_rows = [_opinion("clean raw", ref="a"), _opinion("qualified raw", ref="b")]
    conflict = build_audit_opinion_records(
        panel_anchors=_annual_anchor(), rows=conflict_rows, source=_s2_source()
    ).records[0]
    reversed_conflict = build_audit_opinion_records(
        panel_anchors=_annual_anchor(), rows=list(reversed(conflict_rows)), source=_s2_source()
    ).records[0]
    assert (clean.outcome, clean.source_opportunity) == (False, True)
    assert (missing.outcome, missing.source_opportunity) == (None, False)
    assert conflict.outcome is reversed_conflict.outcome is None
    assert conflict.outcome_basis == reversed_conflict.outcome_basis == "S2_CONFLICTING_OPINIONS"


def test_annual_anchor_is_included_but_same_day_s3_is_not_delayed_verification() -> None:
    columns = {
        FIRM_ID: "firm",
        FISCAL_YEAR: "year",
        PREDICTION_TIME: "anchor",
        MATURE: "mature",
        ELIGIBLE: "eligible",
        SOURCE_ID: "source",
        CHANNEL_ID: "channel",
        AVAILABILITY_DATE: "available",
        OUTCOME: "outcome",
        SOURCE_OPPORTUNITY: "opportunity",
        TEMPORAL_ROLE: "temporal",
        TARGET_ID: "target",
    }
    risk = pd.DataFrame(
        [{"firm": "F1", "year": 2020, "anchor": "2021-03-31", "mature": True, "eligible": True}]
    )
    evidence = pd.DataFrame(
        [
            {
                "firm": "F1",
                "year": 2020,
                "source": "S1_profit_adjustment",
                "channel": "S1",
                "available": "2021-03-31",
                "outcome": False,
                "opportunity": True,
                "temporal": "annual_measurement_at_anchor",
            },
            {
                "firm": "F1",
                "year": 2020,
                "source": "S2_audit_opinion",
                "channel": "S2",
                "available": "2021-03-31",
                "outcome": False,
                "opportunity": True,
                "temporal": "annual_measurement_at_anchor",
            },
            {
                "firm": "F1",
                "year": 2020,
                "source": "S3_sanction_evidence",
                "channel": "S3",
                "available": "2021-03-31",
                "outcome": True,
                "opportunity": None,
                "temporal": "delayed_verification",
            },
        ]
    )
    result = build_measurement_inputs(
        risk_sets=risk,
        evidence=evidence,
        expected_sources={
            "S1_profit_adjustment": "S1",
            "S2_audit_opinion": "S2",
            "S3_sanction_evidence": "S3",
        },
        horizon_months=12,
        columns=columns,
        pending_status="EMPIRICALLY_PENDING",
        unavailable_status="UNAVAILABLE_BY_DESIGN",
        insufficient_channels_reason="INSUFFICIENT_CHANNELS",
        anchor_source_ids=["S3_sanction_evidence"],
        source_temporal_roles={
            "S1_profit_adjustment": "annual_measurement_at_anchor",
            "S2_audit_opinion": "annual_measurement_at_anchor",
            "S3_sanction_evidence": "delayed_verification",
        },
        explicit_negative_allowed={
            "S1_profit_adjustment": True,
            "S2_audit_opinion": True,
            "S3_sanction_evidence": False,
        },
    )
    row = cast(list[dict[str, Any]], result.matrices["rows"])[0]
    outcomes = cast(dict[str, bool | None], row["source_outcomes"])
    assert outcomes["S1_profit_adjustment"] is False
    assert outcomes["S2_audit_opinion"] is False
    assert outcomes["S3_sanction_evidence"] is None
    assert row["l1"] is None


def test_l1_requires_all_opportunities_for_explicit_negative() -> None:
    outcomes = {"S1": False, "S2": False, "S3": None}
    assert aggregate_l1(outcomes, {"S1": True, "S2": True, "S3": None}) is None
    assert aggregate_l1({**outcomes, "S3": True}, {"S1": True, "S2": True, "S3": None}) is True
    assert (
        aggregate_l1(
            {"S1": False, "S2": False, "S3": False},
            {"S1": True, "S2": True, "S3": True},
        )
        is False
    )


def test_annual_sources_are_excluded_from_delayed_maturity_curves() -> None:
    columns = {
        FIRM_ID: "firm",
        FISCAL_YEAR: "year",
        PREDICTION_TIME: "anchor",
        ELIGIBLE: "eligible",
        MATURE: "mature",
        SOURCE_ID: "source",
        CHANNEL_ID: "channel",
        AVAILABILITY_DATE: "available",
        TEMPORAL_ROLE: "temporal",
        SOURCE_OPPORTUNITY: "opportunity",
        AVAILABILITY_BASIS: "availability_basis",
    }
    panel = pd.DataFrame([{"firm": "F1", "year": 2020, "anchor": "2021-03-31"}])
    evidence = pd.DataFrame(
        [
            {
                "firm": "F1",
                "year": 2020,
                "source": "S1_profit_adjustment",
                "channel": "S1",
                "available": "2021-03-31",
                "temporal": "annual_measurement_at_anchor",
                "opportunity": True,
                "availability_basis": "common_annual_anchor",
            },
            {
                "firm": "F1",
                "year": 2020,
                "source": "S3_sanction_evidence",
                "channel": "S3",
                "available": "2021-06-30",
                "temporal": "delayed_verification",
                "opportunity": None,
                "availability_basis": "actual_publish_date",
            },
        ]
    )
    result = build_risk_set(
        panel=panel,
        data_cutoff=datetime(2022, 3, 31),
        horizon_months=12,
        columns=columns,
        evidence=evidence,
    )
    curves = cast(list[dict[str, Any]], result.maturity_audit["source_maturity_curves"])
    annual = cast(list[dict[str, Any]], result.maturity_audit["annual_measurement_availability"])
    assert [row[SOURCE_ID] for row in curves] == ["S3_sanction_evidence"]
    assert [row[SOURCE_ID] for row in annual] == ["S1_profit_adjustment"]
    assert annual[0]["included_in_delayed_maturity_curve"] is False


def test_label_derived_same_year_semantic_is_rejected_from_feature_registry() -> None:
    panel = pd.DataFrame([{"firm": "F1", "year": 2020, "anchor": "2021-03-31", "x": 1.0}])
    risk = pd.DataFrame([{"firm": "F1", "year": 2020, "eligible": True}])
    with pytest.raises(ValueError, match="label-derived same-year variable"):
        build_feature_panel(
            firm_year_panel=panel,
            risk_sets=risk,
            feature_definitions=[
                {
                    "feature_id": "same_year_opinion",
                    "physical_column": "x",
                    "role": "observability",
                    "allowed_in_label_model": True,
                    "availability_rule": "common_annual_anchor",
                    "theoretical_block": "Q",
                    "source_semantics": ["audit_opinion"],
                }
            ],
            columns={
                FIRM_ID: "firm",
                FISCAL_YEAR: "year",
                PREDICTION_TIME: "anchor",
                ELIGIBLE: "eligible",
            },
            blocked_label_semantics=["audit_opinion"],
        )


def test_observability_channel_overlap_counts_observed_channels_exactly() -> None:
    rows = [
        {
            ELIGIBLE: True,
            MATURE: True,
            "source_outcomes": {"s1": False, "s2": True, "s3": None},
            "source_opportunities": {"s1": True, "s2": True, "s3": None},
            "channel_outcomes": {"S1": False, "S2": True, "S3": None},
            "channel_opportunities": {"S1": True, "S2": True, "S3": None},
        },
        {
            ELIGIBLE: True,
            MATURE: True,
            "source_outcomes": {"s1": None, "s2": None, "s3": True},
            "source_opportunities": {"s1": False, "s2": False, "s3": None},
            "channel_outcomes": {"S1": None, "S2": None, "S3": True},
            "channel_opportunities": {"S1": False, "S2": False, "S3": None},
        },
    ]
    metadata = {
        "s1": {
            CHANNEL_ID: "S1",
            "verification_status": "observed",
            "evidence_mapping": {"opportunity_semantic": "pair"},
        },
        "s2": {
            CHANNEL_ID: "S2",
            "verification_status": "observed",
            "evidence_mapping": {"opportunity_semantic": "opinion"},
        },
        "s3": {CHANNEL_ID: "S3", "verification_status": "high_confirmation"},
    }
    result = build_observability_registry({"rows": rows}, metadata)
    overlap = cast(list[dict[str, object]], result["channel_overlap"])
    observed = {
        str(row["channel_pattern"]): int(cast(int, row["eligible_count"]))
        for row in overlap
        if row["basis"] == "observed_outcome"
    }
    assert observed == {"S1∩S2": 1, "S3": 1}
