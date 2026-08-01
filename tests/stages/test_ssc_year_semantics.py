"""SSC opinion evidence must not be hard-coded by calendar year."""

from __future__ import annotations

from datetime import datetime

from core.evidence_registry import LogicalEvidenceSource
from evidence.annual import OpinionRow, build_audit_opinion_records


def _source() -> LogicalEvidenceSource:
    return LogicalEvidenceSource(
        source_id="S2_audit_opinion",
        endpoint_id="non_clean_audit_opinion",
        channel_id="S2",
        physical_source_id="ssc_audit_opinion",
        profile_id="ssc_audit_opinion",
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
                    "UNMODIFIED": "clean",
                    "QUALIFIED": "qualified",
                    "DISCLAIMER": "disclaimer",
                    "ADVERSE": "adverse",
                },
                "positive_categories": ["qualified", "disclaimer", "adverse"],
                "explicit_negative_categories": ["clean"],
                "pending_categories": [],
            },
        },
    )


def test_missing_ssc_year_remains_unknown_without_year_specific_override() -> None:
    records = build_audit_opinion_records(
        panel_anchors={
            ("F1", 2015): datetime(2016, 3, 31),
            ("F1", 2018): datetime(2019, 3, 31),
        },
        rows=[
            OpinionRow(
                firm_id="F1",
                fiscal_year=2018,
                opinion_raw="QUALIFIED",
                audit_indicator="audit_opinion",
                period_type="annual",
                statement_scope="consolidated",
                audit_status="audited",
                source_ref="ssc-2018",
            )
        ],
        source=_source(),
    ).records
    by_year = {record.fiscal_year: record for record in records}

    assert by_year[2015].outcome is None
    assert by_year[2015].source_opportunity is False
    assert by_year[2015].outcome_basis == "S2_OPINION_RECORD_MISSING"

    assert by_year[2018].outcome is True
    assert by_year[2018].source_opportunity is True
    assert by_year[2018].outcome_basis == "S2_NON_CLEAN_OPINION"
