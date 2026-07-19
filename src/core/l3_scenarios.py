"""P0-locked L3 scenario registry validation and expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


REQUIRED_PROFILES = {
    "financial_statement_core_long",
    "audit_annual_long",
    "sanction_evidence",
}
PRIOR_FIELDS = {
    "sensitivity_alpha",
    "sensitivity_beta",
    "specificity_alpha",
    "specificity_beta",
}


@dataclass(frozen=True)
class L3ScenarioSpec:
    scenario_id: str
    role: str
    fixed_pi: float
    prior_set_id: str
    accuracy_priors_by_profile: dict[str, dict[str, float]]


@dataclass(frozen=True)
class L3ScenarioRegistry:
    primary_scenario_id: str
    scenarios: tuple[L3ScenarioSpec, ...]
    execution_policy: dict[str, Any]
    provenance: dict[str, Any]


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{context}: string keys required")
    return cast(dict[str, Any], value)


def _prior_set(value: object, context: str) -> dict[str, dict[str, float]]:
    raw_profiles = _mapping(value, context)
    missing = sorted(REQUIRED_PROFILES - set(raw_profiles))
    extra = sorted(set(raw_profiles) - REQUIRED_PROFILES)
    if missing or extra:
        raise ValueError(f"{context}: profile mismatch missing={missing} extra={extra}")
    result: dict[str, dict[str, float]] = {}
    for profile in sorted(REQUIRED_PROFILES):
        raw = _mapping(raw_profiles[profile], f"{context}.{profile}")
        if set(raw) != PRIOR_FIELDS:
            raise ValueError(
                f"{context}.{profile}: exactly {sorted(PRIOR_FIELDS)} are required"
            )
        parsed: dict[str, float] = {}
        for field in sorted(PRIOR_FIELDS):
            raw_value = raw[field]
            if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
                raise ValueError(f"{context}.{profile}.{field}: numeric value required")
            value = float(raw_value)
            if value <= 0:
                raise ValueError(f"{context}.{profile}.{field}: positive value required")
            parsed[field] = value
        result[profile] = parsed
    return result


def locked_l3_scenario_registry(registry: dict[str, object]) -> L3ScenarioRegistry:
    """Return the validated, pre-data scenario registry included in the P0 hash."""
    raw = _mapping(registry.get("l3_scenarios"), "l3_scenarios")
    if raw.get("version") != 1:
        raise ValueError("l3_scenarios.version must equal 1")
    if raw.get("status") != "LOCKED_AT_P0":
        raise ValueError("l3_scenarios.status must be LOCKED_AT_P0")

    policy = _mapping(raw.get("execution_policy"), "l3_scenarios.execution_policy")
    required_true = {
        "run_all_registered_scenarios",
        "performance_based_scenario_selection_forbidden",
        "modification_after_p00_forbidden",
    }
    for field in sorted(required_true):
        if policy.get(field) is not True:
            raise ValueError(f"l3_scenarios.execution_policy.{field} must be true")
    if policy.get("outer_outcomes_accessed") is not False:
        raise ValueError("L3 scenario registry may not access outer outcomes")
    if policy.get("known_cases_accessed") is not False:
        raise ValueError("L3 scenario registry may not access known cases")
    if policy.get("diagnostic_losses_role") != "capability_and_reporting_only":
        raise ValueError("L3 diagnostic losses must be capability-and-reporting only")

    primary_scenario_id = raw.get("primary_scenario_id")
    if not isinstance(primary_scenario_id, str) or not primary_scenario_id:
        raise ValueError("l3_scenarios.primary_scenario_id must be a nonempty string")

    raw_prior_sets = _mapping(raw.get("prior_sets"), "l3_scenarios.prior_sets")
    if not raw_prior_sets:
        raise ValueError("l3_scenarios.prior_sets must not be empty")
    prior_sets = {
        prior_set_id: _prior_set(value, f"l3_scenarios.prior_sets.{prior_set_id}")
        for prior_set_id, value in raw_prior_sets.items()
    }

    raw_scenarios = _mapping(raw.get("scenarios"), "l3_scenarios.scenarios")
    if not raw_scenarios:
        raise ValueError("l3_scenarios.scenarios must not be empty")
    specs: list[L3ScenarioSpec] = []
    fixed_pi_values: set[float] = set()
    primary_roles: list[str] = []
    for scenario_id, value in raw_scenarios.items():
        scenario = _mapping(value, f"l3_scenarios.scenarios.{scenario_id}")
        role = scenario.get("role")
        if role not in {"primary", "robustness"}:
            raise ValueError(f"scenario={scenario_id}: role must be primary or robustness")
        raw_pi = scenario.get("fixed_pi")
        if not isinstance(raw_pi, (int, float)) or isinstance(raw_pi, bool):
            raise ValueError(f"scenario={scenario_id}: fixed_pi must be numeric")
        fixed_pi = float(raw_pi)
        if not 0.0 < fixed_pi < 1.0:
            raise ValueError(f"scenario={scenario_id}: fixed_pi must lie in (0, 1)")
        if fixed_pi in fixed_pi_values:
            raise ValueError("registered L3 fixed_pi values must be unique")
        fixed_pi_values.add(fixed_pi)
        prior_set_id = scenario.get("prior_set_id")
        if not isinstance(prior_set_id, str) or prior_set_id not in prior_sets:
            raise ValueError(f"scenario={scenario_id}: registered prior_set_id required")
        if role == "primary":
            primary_roles.append(scenario_id)
        specs.append(
            L3ScenarioSpec(
                scenario_id=scenario_id,
                role=role,
                fixed_pi=fixed_pi,
                prior_set_id=prior_set_id,
                accuracy_priors_by_profile=prior_sets[prior_set_id],
            )
        )
    if primary_roles != [primary_scenario_id]:
        raise ValueError(
            "L3 registry requires exactly one primary role matching primary_scenario_id"
        )

    provenance = _mapping(raw.get("provenance"), "l3_scenarios.provenance")
    if provenance.get("decision_stage") != "P0_PROTOCOL_LOCK":
        raise ValueError("L3 scenario provenance must be decided at P0_PROTOCOL_LOCK")
    if provenance.get("outer_outcomes_accessed") is not False:
        raise ValueError("L3 scenario provenance may not access outer outcomes")
    if provenance.get("known_cases_accessed") is not False:
        raise ValueError("L3 scenario provenance may not access known cases")

    ordered = tuple(sorted(specs, key=lambda item: item.scenario_id))
    return L3ScenarioRegistry(
        primary_scenario_id=primary_scenario_id,
        scenarios=ordered,
        execution_policy=dict(policy),
        provenance=dict(provenance),
    )
