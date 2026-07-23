"""Regression tests for unique-channel anchors and the S3 year audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    CHANNEL_ID,
    FIRM_ID,
    FISCAL_YEAR,
    OUTCOME,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load test module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPORT = _load_script_module(
    "test_unique_channel_calibration_report",
    "scripts/report_measurement_calibration.py",
)
_AUDIT = _load_script_module("test_s3_year_audit_script", "scripts/report_s3_year_audit.py")
_physical_channel_coverage = cast(Any, getattr(_REPORT, "_physical_channel_coverage"))
_empirical_anchors = cast(Any, getattr(_REPORT, "_empirical_anchors"))
_prior_worksheet = cast(Any, getattr(_REPORT, "_prior_worksheet"))
_reconcile_p03_p05 = cast(Any, getattr(_AUDIT, "_reconcile_p03_p05"))


def _sources() -> dict[str, object]:
    return {
        "S1_profit_adjustment": SimpleNamespace(
            profile_id="financial_statement_core_long", channel_id="S1"
        ),
        "S1_revenue_adjustment": SimpleNamespace(
            profile_id="financial_statement_core_long", channel_id="S1"
        ),
        "S2_audit_opinion": SimpleNamespace(profile_id="audit_annual_long", channel_id="S2"),
        "S3_BROAD": SimpleNamespace(profile_id="sanction_evidence", channel_id="S3"),
        "S3_REPORTING": SimpleNamespace(profile_id="sanction_evidence", channel_id="S3"),
        "S3_CONTENT": SimpleNamespace(profile_id="sanction_evidence", channel_id="S3"),
        "S3_TIMELINESS": SimpleNamespace(profile_id="sanction_evidence", channel_id="S3"),
    }


def test_prevalence_anchors_use_each_channel_once() -> None:
    source_coverage = pd.DataFrame(
        [
            {
                SOURCE_ID: "S1_profit_adjustment",
                "profile_id": "financial_statement_core_long",
                CHANNEL_ID: "S1",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.20,
            },
            {
                SOURCE_ID: "S1_revenue_adjustment",
                "profile_id": "financial_statement_core_long",
                CHANNEL_ID: "S1",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.10,
            },
            {
                SOURCE_ID: "S2_audit_opinion",
                "profile_id": "audit_annual_long",
                CHANNEL_ID: "S2",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.12,
            },
            {
                SOURCE_ID: "S3_BROAD",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.0,
            },
            {
                SOURCE_ID: "S3_REPORTING",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.0,
            },
            {
                SOURCE_ID: "S3_CONTENT",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.0,
            },
            {
                SOURCE_ID: "S3_TIMELINESS",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
                "observed_positive_rate": 0.0,
            },
        ]
    )
    channel_coverage = pd.DataFrame(
        [
            {
                CHANNEL_ID: "S1",
                "development_rows": 100,
                "observed_outcomes": 100,
                "observed_opportunities": 100,
                "positive": 25,
                "explicit_negative": 75,
                "unknown": 0,
                "coverage_fraction": 1.0,
                "observed_positive_rate": 0.25,
            },
            {
                CHANNEL_ID: "S2",
                "development_rows": 100,
                "observed_outcomes": 100,
                "observed_opportunities": 100,
                "positive": 12,
                "explicit_negative": 88,
                "unknown": 0,
                "coverage_fraction": 1.0,
                "observed_positive_rate": 0.12,
            },
            {
                CHANNEL_ID: "S3",
                "development_rows": 100,
                "observed_outcomes": 100,
                "observed_opportunities": 100,
                "positive": 0,
                "explicit_negative": 100,
                "unknown": 0,
                "coverage_fraction": 1.0,
                "observed_positive_rate": 0.0,
            },
        ]
    )

    physical = _physical_channel_coverage(channel_coverage, _sources())
    anchors = _empirical_anchors(source_coverage, physical)

    assert anchors["anchor_unit"] == "unique_measurement_channel"
    assert anchors["unique_channel_count"] == 3
    assert anchors["channel_observed_positive_rates"] == [0.25, 0.12, 0.0]
    assert anchors["observed_rate_quantiles"]["p50"] == 0.12
    assert len(anchors["logical_source_rates_for_diagnostics_only"]) == 7


def test_prior_worksheet_does_not_multiply_s3_endpoint_count() -> None:
    source_coverage = pd.DataFrame(
        [
            {
                SOURCE_ID: "S1_profit_adjustment",
                "profile_id": "financial_statement_core_long",
                CHANNEL_ID: "S1",
                "observed_outcomes": 90,
            },
            {
                SOURCE_ID: "S1_revenue_adjustment",
                "profile_id": "financial_statement_core_long",
                CHANNEL_ID: "S1",
                "observed_outcomes": 80,
            },
            {
                SOURCE_ID: "S2_audit_opinion",
                "profile_id": "audit_annual_long",
                CHANNEL_ID: "S2",
                "observed_outcomes": 95,
            },
            {
                SOURCE_ID: "S3_BROAD",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
            },
            {
                SOURCE_ID: "S3_REPORTING",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
            },
            {
                SOURCE_ID: "S3_CONTENT",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
            },
            {
                SOURCE_ID: "S3_TIMELINESS",
                "profile_id": "sanction_evidence",
                CHANNEL_ID: "S3",
                "observed_outcomes": 100,
            },
        ]
    )
    physical = pd.DataFrame(
        [
            {CHANNEL_ID: "S1", "observed_outcomes": 92},
            {CHANNEL_ID: "S2", "observed_outcomes": 95},
            {CHANNEL_ID: "S3", "observed_outcomes": 100},
        ]
    )

    worksheet = _prior_worksheet(
        source_coverage=source_coverage,
        physical_channel_coverage=physical,
        sources=_sources(),
    )
    sanction = next(
        row for row in worksheet["profiles"] if row["profile_id"] == "sanction_evidence"
    )

    assert sanction["development_observed_firm_year_channels"] == 100
    assert sanction["logical_endpoint_observed_outcome_sum"] == 400
    assert sanction["independent_prior_information_unit"] == "firm_year_channel"


def test_s3_reconciliation_surfaces_p03_to_p05_outcome_drift() -> None:
    columns = {
        FIRM_ID: FIRM_ID,
        FISCAL_YEAR: FISCAL_YEAR,
        SOURCE_ID: SOURCE_ID,
        OUTCOME: OUTCOME,
        SOURCE_OPPORTUNITY: SOURCE_OPPORTUNITY,
    }
    evidence = pd.DataFrame(
        [
            {
                FIRM_ID: "F1",
                FISCAL_YEAR: 2019,
                SOURCE_ID: "S3_BROAD",
                OUTCOME: True,
                SOURCE_OPPORTUNITY: True,
            },
            {
                FIRM_ID: "F2",
                FISCAL_YEAR: 2019,
                SOURCE_ID: "S3_BROAD",
                OUTCOME: False,
                SOURCE_OPPORTUNITY: True,
            },
        ]
    )
    matrices: dict[str, Any] = {
        "rows": [
            {
                FIRM_ID: "F1",
                FISCAL_YEAR: 2019,
                "source_outcomes": {"S3_BROAD": True},
                "source_opportunities": {"S3_BROAD": True},
            },
            {
                FIRM_ID: "F2",
                FISCAL_YEAR: 2019,
                "source_outcomes": {"S3_BROAD": None},
                "source_opportunities": {"S3_BROAD": True},
            },
        ]
    }

    summary, detail = _reconcile_p03_p05(evidence, matrices, columns)

    assert int(summary.iloc[0]["outcome_mismatch"]) == 1
    assert int(summary.iloc[0]["p03_missing_key"]) == 0
    assert int(summary.iloc[0]["p05_missing_key"]) == 0
    assert len(detail) == 1
    assert detail.iloc[0][FIRM_ID] == "F2"
