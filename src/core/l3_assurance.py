"""Apply and validate the P0-locked D06 assurance amendment."""

from __future__ import annotations

from typing import Any, cast


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{context}: string keys required")
    return cast(dict[str, Any], value)


def apply_l3_preregistered_assurance(registry: dict[str, object]) -> None:
    """Replace legacy D06 metadata with the preregistered P0 decision.

    The amendment is a named manifest module, so its source hash is part of the
    protocol lock. Applying it before reference validation ensures generated
    traceability, Appendix B, and executable assurance tests use one decision.
    """

    raw = _mapping(
        registry.get("l3_preregistered_assurance"),
        "l3_preregistered_assurance",
    )
    if raw.get("version") != 1:
        raise ValueError("l3_preregistered_assurance.version must equal 1")
    if raw.get("decision_id") != "D06":
        raise ValueError("L3 assurance amendment must target D06")

    scenarios = _mapping(registry.get("l3_scenarios"), "l3_scenarios")
    if scenarios.get("status") != "LOCKED_AT_P0":
        raise ValueError("D06 amendment requires l3_scenarios.status=LOCKED_AT_P0")
    if scenarios.get("primary_scenario_id") != "neutral_pi_03":
        raise ValueError("D06 amendment requires neutral_pi_03 as primary scenario")
    policy = _mapping(scenarios.get("execution_policy"), "l3_scenarios.execution_policy")
    if policy.get("performance_based_scenario_selection_forbidden") is not True:
        raise ValueError("D06 amendment requires performance-based selection to be forbidden")
    if policy.get("modification_after_p00_forbidden") is not True:
        raise ValueError("D06 amendment requires modification after P00 to be forbidden")

    decision = _mapping(raw.get("decision"), "l3_preregistered_assurance.decision")
    appendix = _mapping(raw.get("appendix_b"), "l3_preregistered_assurance.appendix_b")
    if decision.get("lock_stage") != "P00":
        raise ValueError("D06 decision must be locked at P00")
    if decision.get("test_ids") != ["T006"]:
        raise ValueError("D06 decision must be enforced by T006")
    required_anchors = {
        "l3_scenarios",
        "measurement.prior_accuracy_domain",
        "measurement.l3_model",
    }
    if set(decision.get("config_anchors", [])) != required_anchors:
        raise ValueError("D06 amendment has incomplete config anchors")
    if appendix.get("canonical_title") != "Miền prior/accuracy":
        raise ValueError("D06 Appendix B title drift")
    if appendix.get("lock_status") != "Khóa tại P0":
        raise ValueError("D06 Appendix B lock status must be Khóa tại P0")

    decisions = _mapping(registry.get("decisions"), "decisions")
    appendix_b = _mapping(registry.get("appendix_b"), "appendix_b")
    if "D06" not in decisions or "D06" not in appendix_b:
        raise ValueError("base assurance registry must contain D06")
    decisions["D06"] = dict(decision)
    appendix_b["D06"] = dict(appendix)
