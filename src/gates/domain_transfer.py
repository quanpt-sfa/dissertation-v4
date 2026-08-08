"""Bind Gate 2 to candidate-specific P13 domain-transfer evidence."""

from __future__ import annotations

from typing import Any, cast

from gates.service import gate2_verdict


def gate2_verdict_with_candidate_domains(
    *,
    evaluations: list[dict[str, Any]],
    bootstraps: list[list[dict[str, Any]]],
    domain_transfer: dict[str, Any],
    gate: dict[str, Any],
    common: dict[str, Any],
    confirmatory_folds: list[str],
) -> dict[str, Any]:
    """Evaluate Gate 2 and require transfer robustness for the same candidate.

    The legacy Gate 2 function is used for the locked within-fold comparison
    criteria. Its generic domain scalar is neutralized here, then each candidate
    is qualified only by its own leave-one-domain-out evidence from P13.
    """

    neutral_domain = {
        "status": "PASS",
        "robust_scenario_fraction": 1.0,
        "leave_one_domain_out_refit_executed": True,
    }
    result = gate2_verdict(
        evaluations=evaluations,
        bootstraps=bootstraps,
        domain_transfer=neutral_domain,
        gate=gate,
        common=common,
        confirmatory_folds=confirmatory_folds,
    )

    raw_candidates = domain_transfer.get("candidate_results")
    domain_candidates = (
        cast(list[dict[str, Any]], raw_candidates) if isinstance(raw_candidates, list) else []
    )
    by_candidate = {
        str(item.get("candidate")): item
        for item in domain_candidates
        if isinstance(item.get("candidate"), str)
    }
    required_fraction = float(common["robust_scenario_fraction_min"])
    blockers: list[str] = []
    fractions: dict[str, float | None] = {}

    raw_gate_candidates = result.get("candidates")
    gate_candidates = (
        cast(list[dict[str, Any]], raw_gate_candidates)
        if isinstance(raw_gate_candidates, list)
        else []
    )
    for candidate in gate_candidates:
        candidate_id = str(candidate.get("candidate"))
        domain = by_candidate.get(candidate_id)
        complete = bool(domain and domain.get("evidence_complete") is True)
        fraction_raw = domain.get("robust_scenario_fraction") if domain else None
        fraction = float(fraction_raw) if isinstance(fraction_raw, (int, float)) else None
        domain_pass = bool(
            complete and fraction is not None and fraction >= required_fraction
        )
        fractions[candidate_id] = fraction
        candidate["domain_evidence_complete"] = complete
        candidate["domain_robust_scenario_fraction"] = fraction
        candidate["domain_required_scenario_fraction"] = required_fraction
        candidate["domain_pass"] = domain_pass
        candidate["pass"] = bool(candidate.get("pass") is True and domain_pass)
        if not complete:
            blockers.append(f"DOMAIN_EVIDENCE_INCOMPLETE:{candidate_id}")

    base_complete = result.get("evidence_complete") is True
    evidence_complete = bool(gate_candidates and base_complete and not blockers)
    verdict = (
        "INSUFFICIENT_EVIDENCE"
        if not evidence_complete
        else "PASS"
        if any(candidate.get("pass") is True for candidate in gate_candidates)
        else "FAIL"
    )
    complete_fractions = [value for value in fractions.values() if value is not None]
    result.update(
        {
            "status": verdict,
            "verdict": verdict,
            "candidates": gate_candidates,
            "domain_robust_scenario_fraction": min(complete_fractions)
            if evidence_complete and complete_fractions
            else None,
            "domain_robust_scenario_fraction_by_candidate": fractions,
            "domain_required_scenario_fraction": required_fraction,
            "domain_transfer_status": domain_transfer.get("status"),
            "domain_transfer_method": domain_transfer.get("support_method"),
            "domain_evidence_blockers": sorted(blockers),
            "evidence_complete": evidence_complete,
            "reason_code": None if evidence_complete else "REQUIRED_GATE2_EVIDENCE_INCOMPLETE",
        }
    )
    return result
