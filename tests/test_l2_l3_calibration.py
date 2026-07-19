"""Regression tests for neutral L2 and development-only L3 calibration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from core.evidence_registry import LogicalEvidenceSource
from core.registry_compiler import compile_registry
from core.semantic_keys import (
    CHANNEL_ID,
    DOCUMENT_ID,
    ELIGIBLE,
    FISCAL_YEAR,
    MATURE,
    SOURCE_ID,
    TEMPORAL_ROLE,
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


_P10 = _load_script_module("test_p10_select_measurement", "scripts/p10_select_measurement.py")
_REPORT = _load_script_module(
    "test_report_measurement_calibration",
    "scripts/report_measurement_calibration.py",
)
_l3_bindings = cast(Any, getattr(_P10, "_l3_bindings"))
_source_coverage = cast(Any, getattr(_REPORT, "_source_coverage"))


def test_neutral_l2_configuration_is_locked() -> None:
    registry = cast(
        dict[str, Any],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )
    measurement = cast(dict[str, Any], registry["measurement"])
    missingness = cast(dict[str, Any], measurement["l2_missingness"])
    scoring = cast(dict[str, Any], measurement["l2_scoring"])

    assert missingness["minimum_observed_channels"] == 1
    assert scoring["formula"] == "quality_delay_weighted_observed_source_mean"
    assert scoring["delay_half_life_days"] == 365
    assert scoring["channel_weights"] == "equal"
    assert scoring["source_quality_by_profile"] == {
        "financial_statement_core_long": 1.0,
        "audit_annual_long": 1.0,
        "sanction_evidence": 1.0,
    }


def test_l3_bindings_expand_physical_source_to_logical_endpoints() -> None:
    registry: dict[str, object] = {
        "data_sources": {
            "source_registry": {
                "sources": {
                    "sanction_evidence": {
                        "enabled": True,
                        "role": "evidence",
                        CHANNEL_ID: "S3",
                        "profile_id": "sanction_evidence",
                        "verification_status": "high_confirmation",
                        "evidence_mapping": {
                            "processor": "sanction_calendar_year",
                            TEMPORAL_ROLE: "next_calendar_year_regulatory_event",
                            "availability_rule": "sanction_calendar_year",
                            "explicit_negative_allowed": True,
                            "opportunity_semantic": "derived_source_year_completeness",
                            "target_year_rule": "sanction_year_minus_one",
                            "decision_key_semantic": DOCUMENT_ID,
                            "logical_sources": [
                                {SOURCE_ID: "S3_BROAD", "endpoint_id": "S3_BROAD"},
                                {SOURCE_ID: "S3_REPORTING", "endpoint_id": "S3_REPORTING"},
                            ],
                        },
                    }
                }
            }
        }
    }
    prior = {
        "sensitivity_alpha": 2.0,
        "sensitivity_beta": 2.0,
        "specificity_alpha": 4.0,
        "specificity_beta": 1.0,
    }

    channels, priors = _l3_bindings(
        registry=registry,
        priors_by_profile={"sanction_evidence": prior},
    )

    assert channels == {"S3_BROAD": "S3", "S3_REPORTING": "S3"}
    assert priors == {"S3_BROAD": prior, "S3_REPORTING": prior}
    assert "sanction_evidence" not in channels


def test_source_coverage_preserves_unknown_outcomes() -> None:
    source = LogicalEvidenceSource(
        source_id="S1_profit_adjustment",
        endpoint_id="profit_adjustment",
        channel_id="S1",
        physical_source_id="financial_statement_core_long",
        profile_id="financial_statement_core_long",
        processor="audit_adjustment",
        temporal_role="annual_measurement_at_anchor",
        availability_rule="common_annual_anchor",
        explicit_negative_allowed=True,
        verification_status="observed_opportunity_only",
        opportunity_semantic="derived_matched_pair",
        logical_config={},
        processor_config={},
    )
    rows = [
        {
            FISCAL_YEAR: 2019,
            ELIGIBLE: True,
            MATURE: True,
            "source_outcomes": {source.source_id: True},
            "source_opportunities": {source.source_id: True},
        },
        {
            FISCAL_YEAR: 2020,
            ELIGIBLE: True,
            MATURE: True,
            "source_outcomes": {source.source_id: None},
            "source_opportunities": {source.source_id: False},
        },
    ]

    report = _source_coverage(rows, {source.source_id: source}).iloc[0]

    assert report["observed_outcomes"] == 1
    assert report["positive"] == 1
    assert report["explicit_negative"] == 0
    assert report["unknown"] == 1
    assert report["coverage_fraction"] == 0.5
