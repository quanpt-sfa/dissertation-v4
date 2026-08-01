"""Regression tests for the P07 raw-computation decision report."""

from __future__ import annotations

import pytest

from features.reporting import build_raw_computation_decision_report


def _summary() -> dict[str, object]:
    return {
        "status": "PASS",
        "panel_features": 136,
        "unresolved_features": 15,
        "feature_store": {
            "stage_status": "PASS",
            "source": "financial_statement_core_long",
            "computation_mode": "raw_registry_formula",
            "feature_count": 151,
            "computed_pass": 136,
            "unsupported_features": ["beneish_dsri", "beneish_tata"],
            "raw_rows": 445456,
            "component_sum_incomplete_dropped": [],
        },
        "identifier_mapping": {
            "matched_identifiers": 558,
            "unmatched_identifiers": 0,
        },
    }


def test_decision_report_accepts_raw_computation_contract() -> None:
    report = build_raw_computation_decision_report(
        "# P07 feature-governance decision report\n",
        _summary(),
    )

    assert "P07 stage status: PASS" in report
    assert "Registered feature computations: 151" in report
    assert "Computed PASS: 136" in report
    assert "Unsupported feature computations: 2" in report
    assert "Operational panel features: 136" in report
    assert "beneish_dsri, beneish_tata" in report
    assert "Package status" not in report
    assert "Validated files" not in report


def test_decision_report_fails_with_explicit_contract_error() -> None:
    summary = _summary()
    feature_store = summary["feature_store"]
    assert isinstance(feature_store, dict)
    feature_store.pop("computed_pass")

    with pytest.raises(
        ValueError,
        match="P07 report contract requires nonnegative integer field computed_pass",
    ):
        build_raw_computation_decision_report("# P07\n", summary)
