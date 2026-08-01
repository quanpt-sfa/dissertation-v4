from __future__ import annotations

from datetime import datetime

import pandas as pd

import core.semantic_keys as semantic_keys
from core.semantic_keys import (
    DETERMINATION_STATUS,
    FIRM_ID,
    FISCAL_YEAR,
    LAYER_OBSERVED_MASK,
    OBSERVATION_OPPORTUNITY,
    OBSERVED_SOURCE_LABEL,
    OUTCOME,
    PREDICTION_TIME,
    REASON_UNKNOWN,
    RECORDING_STATUS,
    SOURCE_OPPORTUNITY,
    VERIFICATION_STATUS,
)
from evidence.migration import migrate_evidence_ledger_v6_to_v7
from evidence.service import EvidenceRecord, build_evidence_ledger


def _columns() -> dict[str, str]:
    return {
        value: value
        for name, value in vars(semantic_keys).items()
        if name.isupper() and isinstance(value, str)
    }


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            FIRM_ID: pd.Series(["F1"], dtype="string"),
            FISCAL_YEAR: pd.Series([2021], dtype="int16"),
            PREDICTION_TIME: pd.Series([pd.Timestamp("2022-03-31")], dtype="datetime64[ns]"),
        }
    )


def test_v7_ledger_materialises_canonical_layers_and_retains_aliases() -> None:
    anchor = datetime(2022, 3, 31)
    records = [
        EvidenceRecord(
            source_id="S2_audit_opinion",
            source_profile_id="audit_annual_long",
            channel_id="S2",
            firm_id="F1",
            fiscal_year=2021,
            availability_date=anchor,
            outcome=False,
            event_id=None,
            event_cluster_id=None,
            evidence_record_id="S2_audit_opinion:F1:2021",
            evidence_record_kind="annual_source_result",
            temporal_role="annual_measurement_at_anchor",
            availability_basis="common_annual_anchor",
            source_opportunity=True,
            opportunity_basis="clean_audit_opinion",
            outcome_basis="clean_audit_opinion",
        )
    ]
    result = build_evidence_ledger(
        panel=_panel(),
        records=records,
        columns=_columns(),
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    row = result.ledger.iloc[0]
    assert bool(row[SOURCE_OPPORTUNITY]) is True
    assert bool(row[OBSERVATION_OPPORTUNITY]) is True
    assert bool(row[OUTCOME]) is False
    assert bool(row[OBSERVED_SOURCE_LABEL]) is False
    assert pd.isna(row[VERIFICATION_STATUS])
    assert pd.isna(row[DETERMINATION_STATUS])
    assert pd.isna(row[RECORDING_STATUS])
    assert row[LAYER_OBSERVED_MASK] == "O---S"
    assert pd.isna(row[REASON_UNKNOWN])


def test_s3_positive_materialises_public_vdr_but_zero_does_not() -> None:
    anchor = datetime(2022, 3, 31)
    positive = EvidenceRecord(
        source_id="S3_CONTENT",
        source_profile_id="sanction_evidence",
        channel_id="S3",
        firm_id="F1",
        fiscal_year=2021,
        availability_date=anchor,
        outcome=True,
        event_id=None,
        event_cluster_id=None,
        evidence_record_id="S3_CONTENT|F1|2021",
        evidence_record_kind="firm_year_endpoint_result",
        temporal_role="next_calendar_year_regulatory_event",
        availability_basis="sanction_calendar_year",
        source_opportunity=True,
        opportunity_basis="complete_source_year_and_panel_universe",
        verification_status=True,
        determination_status=True,
        recording_status=True,
        sanction_year=2022,
        target_fiscal_year=2021,
        outcome_basis="eligible_decision_in_required_sanction_year",
    )
    result = build_evidence_ledger(
        panel=_panel(),
        records=[positive],
        columns=_columns(),
        fiscal_year_end_month_day="12-31",
        lag_tolerance_days=0,
    )
    row = result.ledger.iloc[0]
    assert row[LAYER_OBSERVED_MASK] == "OVDRS"
    assert bool(row[VERIFICATION_STATUS])
    assert bool(row[DETERMINATION_STATUS])
    assert bool(row[RECORDING_STATUS])


def test_v6_migration_never_infers_vdr_from_observed_endpoint_zero() -> None:
    legacy = pd.DataFrame(
        {
            SOURCE_OPPORTUNITY: pd.Series([True, True, pd.NA], dtype="boolean"),
            OUTCOME: pd.Series([True, False, pd.NA], dtype="boolean"),
            "evidence_record_kind": pd.Series(
                [
                    "firm_year_endpoint_result",
                    "firm_year_endpoint_result",
                    "delayed_event",
                ],
                dtype="string",
            ),
        }
    )
    migrated = migrate_evidence_ledger_v6_to_v7(legacy)
    frame = migrated.frame
    assert frame.loc[0, LAYER_OBSERVED_MASK] == "OVDRS"
    assert frame.loc[1, LAYER_OBSERVED_MASK] == "O---S"
    assert pd.isna(frame.loc[1, VERIFICATION_STATUS])
    assert pd.isna(frame.loc[1, DETERMINATION_STATUS])
    assert pd.isna(frame.loc[1, RECORDING_STATUS])
    assert frame.loc[2, REASON_UNKNOWN] == "OBSERVATION_OPPORTUNITY_UNKNOWN"
    assert migrated.audit["dates_fabricated"] is False
    assert migrated.audit["observed_endpoint_zero_rows"] == 1
