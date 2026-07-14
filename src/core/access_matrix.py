"""Machine-enforced access policies using artifact sensitivity classes and receipts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .errors import AccessPolicyError


def compile_access_matrix(
    steps: object, artifacts: object, access_control: object
) -> dict[str, Any]:
    if (
        not isinstance(steps, dict)
        or not isinstance(artifacts, dict)
        or not isinstance(access_control, dict)
    ):
        raise AccessPolicyError("steps, artifacts, and access_control must be mappings")
    artifact_map = cast(dict[str, Any], artifacts)
    matrix: dict[str, Any] = {}
    for step_id, value in cast(dict[str, Any], steps).items():
        if not isinstance(value, dict):
            raise AccessPolicyError(f"step={step_id}: mapping required")
        step = cast(dict[str, Any], value)
        matrix[str(step_id)] = {
            "reads": list(step.get("reads", [])),
            "optional_reads": list(step.get("optional_reads", [])),
            "writes": list(step.get("writes", [])),
            "outer_access": step.get("outer_access"),
            "known_case_access": step.get("known_case_access"),
            "permitted_states": list(step.get("permitted_states", [])),
            "allowed_next_states": list(step.get("allowed_next_states", [])),
            "required_receipts": list(step.get("required_receipts", [])),
            "read_sensitivity_classes": sorted(
                {
                    artifact_map[artifact_id]["sensitivity_class"]
                    for artifact_id in [*step.get("reads", []), *step.get("optional_reads", [])]
                }
            ),
        }
    return matrix


def assert_access(
    matrix: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    access_control: Mapping[str, Any],
    step_id: str,
    artifact_id: str,
    mode: str,
    state: str,
) -> None:
    contract = matrix.get(step_id)
    artifact = artifacts.get(artifact_id)
    if not isinstance(contract, dict):
        raise AccessPolicyError(f"step={step_id}: unknown step")
    if not isinstance(artifact, dict):
        raise AccessPolicyError(f"artifact={artifact_id}: unknown artifact")
    typed = cast(dict[str, Any], contract)
    if state not in typed.get("permitted_states", []):
        raise AccessPolicyError(f"step={step_id}, state={state}: state is not permitted")

    allowed = (
        typed.get("writes", [])
        if mode == "write"
        else [*typed.get("reads", []), *typed.get("optional_reads", [])]
    )
    if artifact_id not in allowed:
        raise AccessPolicyError(f"step={step_id}, artifact={artifact_id}: {mode} is not declared")

    if mode == "read":
        artifact_spec = cast(dict[str, Any], artifact)
        sensitivity = artifact_spec.get("sensitivity_class")
        if sensitivity == "outer" and typed.get("outer_access") != "open":
            raise AccessPolicyError(
                f"step={step_id}, artifact={artifact_id}: outer outcome is sealed"
            )
        if sensitivity == "known_case" and typed.get("known_case_access") != "open":
            raise AccessPolicyError(
                f"step={step_id}, artifact={artifact_id}: known-case content is sealed"
            )

    outer_policy = cast(dict[str, Any], access_control.get("outer_outcomes", {}))
    post_open_states = {
        "OUTER_OPEN",
        "EVALUATED",
        "SENSITIVITY",
        "GATE2",
        "KNOWN_CASES_OPEN",
        "GATE3",
        "REPORTED",
    }
    if mode == "write" and state in post_open_states:
        if step_id in outer_policy.get("post_open_write_prohibited_steps", []):
            raise AccessPolicyError(f"step={step_id}: writes are revoked after outer opening")
