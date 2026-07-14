"""Machine-readable access firewall compiled from step declarations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .errors import AccessPolicyError


def compile_access_matrix(steps: object) -> dict[str, Any]:
    """Compile declared access contracts without granting implicit permissions."""
    if not isinstance(steps, dict):
        raise AccessPolicyError("steps: expected mapping")
    matrix: dict[str, Any] = {}
    for step_id, value in steps.items():
        if not isinstance(value, dict):
            raise AccessPolicyError(f"step={step_id}: expected mapping")
        step = cast(dict[str, Any], value)
        matrix[str(step_id)] = {
            "reads": step.get("reads", []),
            "optional_reads": step.get("optional_reads", []),
            "writes": step.get("writes", []),
            "outer_access": step.get("outer_access"),
            "known_case_access": step.get("known_case_access"),
            "permitted_states": step.get("permitted_states", []),
            "allowed_next_states": step.get("allowed_next_states", []),
        }
    return matrix


def assert_access(
    matrix: Mapping[str, Any], step_id: str, artifact_id: str, mode: str, state: str
) -> None:
    """Enforce a compiled read/write permission for a future RunContext."""
    contract = matrix.get(step_id)
    if not isinstance(contract, dict):
        raise AccessPolicyError(f"step={step_id}: unknown step")
    typed = cast(dict[str, Any], contract)
    if state not in typed.get("permitted_states", []):
        raise AccessPolicyError(f"step={step_id}, state={state}: state is not permitted")
    field = "reads" if mode == "read" else "writes"
    if artifact_id not in typed.get(field, []):
        raise AccessPolicyError(f"step={step_id}, artifact={artifact_id}: {mode} is not declared")
