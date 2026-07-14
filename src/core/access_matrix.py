"""Machine-readable access firewall compiled from step declarations."""

from __future__ import annotations

from typing import Mapping

from .errors import AccessPolicyError


def compile_access_matrix(steps: object) -> dict[str, object]:
    """Compile declared access contracts without granting implicit permissions."""
    if not isinstance(steps, dict):
        raise AccessPolicyError("steps: expected mapping")
    matrix: dict[str, object] = {}
    for step_id, value in steps.items():
        if not isinstance(value, dict):
            raise AccessPolicyError(f"step={step_id}: expected mapping")
        matrix[str(step_id)] = {
            "reads": value.get("reads", []),
            "optional_reads": value.get("optional_reads", []),
            "writes": value.get("writes", []),
            "outer_access": value.get("outer_access"),
            "known_case_access": value.get("known_case_access"),
            "permitted_states": value.get("permitted_states", []),
            "allowed_next_states": value.get("allowed_next_states", []),
        }
    return matrix


def assert_access(matrix: Mapping[str, object], step_id: str, artifact_id: str, mode: str, state: str) -> None:
    """Enforce a compiled read/write permission for a future RunContext."""
    contract = matrix.get(step_id)
    if not isinstance(contract, dict):
        raise AccessPolicyError(f"step={step_id}: unknown step")
    if state not in contract.get("permitted_states", []):
        raise AccessPolicyError(f"step={step_id}, state={state}: state is not permitted")
    field = "reads" if mode == "read" else "writes"
    if artifact_id not in contract.get(field, []):
        raise AccessPolicyError(f"step={step_id}, artifact={artifact_id}: {mode} is not declared")
