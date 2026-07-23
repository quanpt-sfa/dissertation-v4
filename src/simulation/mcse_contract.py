"""Locked metric-role and undefined-value contracts for P08 MCSE evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fnmatch import fnmatchcase
from typing import TypedDict, cast


class MetricPolicy(TypedDict):
    policy_id: str
    priority: int
    metric_pattern: str
    resolved_pattern: str
    role: str
    mcse_gate_required: bool
    undefined_policy: str
    target_rule: str
    minimum_finite_fraction: float


class PolicyCoverage(TypedDict):
    status: str
    metric_count: int
    gated_metric_count: int
    role_counts: dict[str, int]
    policies: dict[str, MetricPolicy]


_ALLOWED_ROLES = frozenset({"confirmatory", "diagnostic", "descriptive"})
_ALLOWED_UNDEFINED_POLICIES = frozenset({"forbid", "allow_nonblocking", "report_only"})
_ALLOWED_TARGET_RULES = frozenset({"absolute", "relative_to_mean", "mmi_scaled", "none"})
_PRIMARY_COST_TOKEN = "{primary_evaluation_cost_regime_id}"


def compile_metric_policies(
    registry: Mapping[str, object],
    simulation: Mapping[str, object],
) -> list[MetricPolicy]:
    """Validate and expand the locked policy registry."""
    if registry.get("contract_version") != 1:
        raise ValueError("simulation_mcse.contract_version must equal 1")
    if registry.get("require_complete_policy_coverage") is not True:
        raise ValueError("simulation_mcse must require complete metric-policy coverage")
    _validate_allowed_registry_values(registry)

    cost_settings_raw = simulation.get("cost_sensitive")
    if not isinstance(cost_settings_raw, dict):
        raise ValueError("simulation.cost_sensitive mapping is required for MCSE policy expansion")
    cost_settings = cast(dict[str, object], cost_settings_raw)
    primary_cost_id = cost_settings.get("primary_evaluation_cost_regime_id")
    if not isinstance(primary_cost_id, str) or not primary_cost_id:
        raise ValueError("simulation primary evaluation cost regime must be locked")

    raw_policies = registry.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise ValueError("simulation_mcse.policies must be a nonempty list")

    policies: list[MetricPolicy] = []
    seen_ids: set[str] = set()
    for index, raw_policy in enumerate(cast(list[object], raw_policies)):
        if not isinstance(raw_policy, dict):
            raise ValueError(f"simulation_mcse.policies[{index}] must be a mapping")
        policy = cast(dict[str, object], raw_policy)
        policy_id = _required_string(policy, "policy_id", index)
        if policy_id in seen_ids:
            raise ValueError(f"duplicate MCSE policy_id={policy_id}")
        seen_ids.add(policy_id)

        pattern = _required_string(policy, "metric_pattern", index)
        resolved_pattern = pattern.replace(_PRIMARY_COST_TOKEN, primary_cost_id)
        if "{" in resolved_pattern or "}" in resolved_pattern:
            raise ValueError(f"policy={policy_id}: unresolved metric-pattern template")
        priority = policy.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool) or priority < 0:
            raise ValueError(f"policy={policy_id}: priority must be a nonnegative integer")
        role = _required_string(policy, "role", index)
        undefined_policy = _required_string(policy, "undefined_policy", index)
        target_rule = _required_string(policy, "target_rule", index)
        gate_required = policy.get("mcse_gate_required")
        finite_fraction = policy.get("minimum_finite_fraction")
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"policy={policy_id}: unsupported role={role}")
        if undefined_policy not in _ALLOWED_UNDEFINED_POLICIES:
            raise ValueError(f"policy={policy_id}: unsupported undefined_policy={undefined_policy}")
        if target_rule not in _ALLOWED_TARGET_RULES:
            raise ValueError(f"policy={policy_id}: unsupported target_rule={target_rule}")
        if not isinstance(gate_required, bool):
            raise ValueError(f"policy={policy_id}: mcse_gate_required must be boolean")
        if (
            not isinstance(finite_fraction, (int, float))
            or isinstance(finite_fraction, bool)
            or not 0.0 <= float(finite_fraction) <= 1.0
        ):
            raise ValueError(f"policy={policy_id}: minimum_finite_fraction must be in [0,1]")
        if role == "confirmatory":
            if not gate_required or undefined_policy != "forbid" or target_rule == "none":
                raise ValueError(
                    f"policy={policy_id}: confirmatory metrics must gate, forbid undefined "
                    "values, and define a target rule"
                )
            if float(finite_fraction) <= 0.0:
                raise ValueError(
                    f"policy={policy_id}: confirmatory metrics require finite replications"
                )
        elif gate_required:
            raise ValueError(
                f"policy={policy_id}: only confirmatory metrics may enter the MCSE gate"
            )

        policies.append(
            {
                "policy_id": policy_id,
                "priority": priority,
                "metric_pattern": pattern,
                "resolved_pattern": resolved_pattern,
                "role": role,
                "mcse_gate_required": gate_required,
                "undefined_policy": undefined_policy,
                "target_rule": target_rule,
                "minimum_finite_fraction": float(finite_fraction),
            }
        )
    return policies


def resolve_metric_policy(metric_id: str, policies: Sequence[MetricPolicy]) -> MetricPolicy:
    """Resolve exactly one highest-precedence policy for a metric identifier."""
    matches = [policy for policy in policies if fnmatchcase(metric_id, policy["resolved_pattern"])]
    if not matches:
        raise ValueError(f"P08 MCSE policy missing for metric_id={metric_id}")
    ordered = sorted(
        matches,
        key=lambda policy: (
            policy["priority"],
            _pattern_specificity(policy["resolved_pattern"]),
            policy["policy_id"],
        ),
        reverse=True,
    )
    winner = ordered[0]
    winner_rank = (
        winner["priority"],
        _pattern_specificity(winner["resolved_pattern"]),
    )
    tied = [
        policy
        for policy in ordered
        if (
            policy["priority"],
            _pattern_specificity(policy["resolved_pattern"]),
        )
        == winner_rank
    ]
    if len(tied) != 1:
        ids = sorted(policy["policy_id"] for policy in tied)
        raise ValueError(f"P08 MCSE policy ambiguous for metric_id={metric_id}: {ids}")
    return dict(winner)


def resolve_metric_policies(
    metric_ids: Iterable[str],
    policies: Sequence[MetricPolicy],
) -> PolicyCoverage:
    """Resolve complete coverage and require at least one confirmatory gate."""
    resolved: dict[str, MetricPolicy] = {}
    role_counts = {role: 0 for role in sorted(_ALLOWED_ROLES)}
    for metric_id in sorted(set(metric_ids)):
        policy = resolve_metric_policy(metric_id, policies)
        resolved[metric_id] = policy
        role_counts[policy["role"]] += 1
    gated_count = sum(1 for policy in resolved.values() if policy["mcse_gate_required"] is True)
    if not resolved:
        raise ValueError("P08 MCSE metric set must be nonempty")
    if gated_count < 1:
        raise ValueError("P08 MCSE metric set must contain at least one confirmatory gate")
    return {
        "status": "PASS",
        "metric_count": len(resolved),
        "gated_metric_count": gated_count,
        "role_counts": role_counts,
        "policies": resolved,
    }


def _validate_allowed_registry_values(registry: Mapping[str, object]) -> None:
    expected = {
        "allowed_roles": _ALLOWED_ROLES,
        "allowed_undefined_policies": _ALLOWED_UNDEFINED_POLICIES,
        "allowed_target_rules": _ALLOWED_TARGET_RULES,
    }
    for key, required in expected.items():
        raw = registry.get(key)
        if not isinstance(raw, list):
            raise ValueError(f"simulation_mcse.{key} must be a list")
        actual = {str(item) for item in cast(list[object], raw)}
        if actual != set(required):
            raise ValueError(f"simulation_mcse.{key} must equal {sorted(required)}")


def _required_string(policy: Mapping[str, object], field: str, index: int) -> str:
    value = policy.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"simulation_mcse.policies[{index}].{field} must be nonempty")
    return value


def _pattern_specificity(pattern: str) -> int:
    return len(pattern.replace("*", "").replace("?", ""))
