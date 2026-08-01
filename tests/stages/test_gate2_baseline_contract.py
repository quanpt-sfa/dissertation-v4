"""Regression tests for Gate 2 pilot semantics."""

from __future__ import annotations

from gates.evidence_contract import (
    baseline_only_gate2_verdict,
    gate2_blocker_details,
    insufficient_gate2_verdict,
)


def test_baseline_only_is_not_reported_as_performance_failure() -> None:
    verdict = baseline_only_gate2_verdict()
    assert verdict["status"] == "NOT_APPLICABLE"
    assert verdict["verdict"] == "NOT_APPLICABLE"
    assert verdict["reason_code"] == "BASELINE_ONLY_NO_COMPARATIVE_CANDIDATE_FAMILY"
    assert verdict["evidence_blockers"] == []


def test_incomplete_comparison_preserves_structured_blocker_detail() -> None:
    blockers = [
        "GATE2_REQUIRED_PAIR_MISSING:2022:track_l1:main_boosting:full",
        "GATE2_BOOTSTRAP_TOO_SHORT:2023:track_l1:main_boosting:full",
    ]
    verdict = insufficient_gate2_verdict(blockers, ["track_l1:main_boosting:full"])
    assert verdict["status"] == "INSUFFICIENT_EVIDENCE"
    details = verdict["evidence_blocker_details"]
    assert isinstance(details, list)
    assert {row["outer_fold"] for row in details} == {"2022", "2023"}
    assert all(row["candidate_or_model"] == "track_l1" for row in details)


def test_blocker_detail_keeps_raw_code_for_lossless_reporting() -> None:
    raw = "GATE2_AP_MISSING:2024:track_l1:elastic_net_logistic:full"
    detail = gate2_blocker_details([raw])[0]
    assert detail["reason_code"] == "GATE2_AP_MISSING"
    assert detail["outer_fold"] == "2024"
    assert detail["raw_blocker"] == raw
