from __future__ import annotations

from typing import Any, cast

import gates.domain_transfer as gate_domain


def test_gate2_requires_domain_robustness_for_same_candidate(monkeypatch: Any) -> None:
    def fake_gate2_verdict(**_: Any) -> dict[str, Any]:
        return {
            "status": "PASS",
            "verdict": "PASS",
            "evidence_complete": True,
            "candidates": [
                {"candidate": "track_l1:a:full", "pass": True},
                {"candidate": "track_l1:b:full", "pass": True},
            ],
        }

    monkeypatch.setattr(gate_domain, "gate2_verdict", fake_gate2_verdict)
    result = gate_domain.gate2_verdict_with_candidate_domains(
        evaluations=[],
        bootstraps=[],
        domain_transfer={
            "status": "PASS",
            "support_method": "cross_fitted_balanced_logistic_domain_score_held_test_fraction",
            "candidate_results": [
                {
                    "candidate": "track_l1:a:full",
                    "evidence_complete": True,
                    "robust_scenario_fraction": 0.75,
                },
                {
                    "candidate": "track_l1:b:full",
                    "evidence_complete": True,
                    "robust_scenario_fraction": 1.0,
                },
            ],
        },
        gate={},
        common={"robust_scenario_fraction_min": 0.8},
        confirmatory_folds=[],
    )

    candidates = cast(list[dict[str, object]], result["candidates"])
    assert candidates[0]["domain_pass"] is False
    assert candidates[0]["pass"] is False
    assert candidates[1]["domain_pass"] is True
    assert candidates[1]["pass"] is True
    assert result["verdict"] == "PASS"
    assert (
        result["domain_transfer_method"]
        == "cross_fitted_balanced_logistic_domain_score_held_test_fraction"
    )
