"""Validate canonical Appendix B decisions, crosswalks, and executable pytest nodes."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from .errors import DecisionTraceabilityError


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DecisionTraceabilityError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def path_exists(registry: object, dotted: str) -> bool:
    current: object = registry
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = cast(dict[str, Any], current)[part]
    return True


def validate_decisions(registry: dict[str, object]) -> list[dict[str, object]]:
    appendix_b = _mapping(registry.get("appendix_b"), "appendix_b")
    decisions = _mapping(registry.get("decisions"), "decisions")
    tests = _mapping(registry.get("tests"), "tests")
    steps = _mapping(registry.get("steps"), "steps")
    artifacts = _mapping(registry.get("artifacts"), "artifacts")
    controls = _mapping(registry.get("implementation_controls"), "implementation_controls")

    expected = {f"D{number:02d}" for number in range(1, 46)}
    if set(appendix_b) != expected or set(decisions) != expected:
        raise DecisionTraceabilityError("appendix_b and decisions must contain exactly D01-D45")

    rows: list[dict[str, object]] = []
    for decision_id in sorted(expected):
        canonical = _mapping(appendix_b[decision_id], f"appendix_b.{decision_id}")
        crosswalk = _mapping(decisions[decision_id], f"decisions.{decision_id}")
        if crosswalk.get("appendix_b_id") != decision_id:
            raise DecisionTraceabilityError(f"decision={decision_id}: appendix_b_id mismatch")
        if canonical.get("chapter_reference") != f"Appendix B, Table B.1, {decision_id}":
            raise DecisionTraceabilityError(
                f"decision={decision_id}: exact Table B.1 reference required"
            )
        for key in ("canonical_title", "main_specification", "sensitivity_or_rule", "lock_status"):
            if not isinstance(canonical.get(key), str) or not str(canonical[key]).strip():
                raise DecisionTraceabilityError(f"decision={decision_id}: canonical {key} required")

        anchors = crosswalk.get("config_anchors")
        test_ids = crosswalk.get("test_ids")
        evidence = crosswalk.get("output_evidence")
        enforced = crosswalk.get("enforced_by_steps")
        implementation_controls = crosswalk.get("implementation_controls")
        for name, value in (
            ("config_anchors", anchors),
            ("test_ids", test_ids),
            ("output_evidence", evidence),
            ("enforced_by_steps", enforced),
            ("implementation_controls", implementation_controls),
        ):
            if (
                not isinstance(value, list)
                or not value
                or not all(isinstance(item, str) for item in cast(list[object], value))
            ):
                raise DecisionTraceabilityError(
                    f"decision={decision_id}: non-empty {name} required"
                )

        for anchor in cast(list[str], anchors):
            if not path_exists(registry, anchor):
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, anchor={anchor}: unknown registry path"
                )
        for test_id in cast(list[str], test_ids):
            if test_id not in tests:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, test={test_id}: unknown test"
                )
            test_spec = _mapping(tests[test_id], f"tests.{test_id}")
            if decision_id not in test_spec.get("decision_ids", []):
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, test={test_id}: reciprocal link missing"
                )
        for step_id in cast(list[str], enforced):
            if step_id not in steps:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, step={step_id}: unknown step"
                )
        for artifact_id in cast(list[str], evidence):
            if artifact_id not in artifacts:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, artifact={artifact_id}: unknown artifact"
                )
        for control_id in cast(list[str], implementation_controls):
            if control_id not in controls:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, control={control_id}: unknown control"
                )

        rows.append(
            {
                "decision_id": decision_id,
                **canonical,
                "config_anchors": anchors,
                "lock_stage": crosswalk.get("lock_stage"),
                "enforced_by_steps": enforced,
                "test_ids": test_ids,
                "output_evidence": evidence,
                "implementation_controls": implementation_controls,
            }
        )
    return rows


def validate_test_registry(registry: dict[str, object], root: Path) -> None:
    tests = _mapping(registry.get("tests"), "tests")
    expected = {f"T{number:03d}" for number in range(1, 46)}
    if set(tests) != expected:
        raise DecisionTraceabilityError("tests must contain exactly T001-T045")

    declared_nodes: set[str] = set()
    for test_id in sorted(expected):
        item = _mapping(tests[test_id], f"tests.{test_id}")
        nodes = item.get("pytest_nodes")
        decision_ids = item.get("decision_ids")
        if (
            not isinstance(nodes, list)
            or not nodes
            or not all(isinstance(node, str) for node in cast(list[object], nodes))
        ):
            raise DecisionTraceabilityError(f"test={test_id}: pytest_nodes required")
        if not isinstance(decision_ids, list) or len(cast(list[object], decision_ids)) != 1:
            raise DecisionTraceabilityError(f"test={test_id}: exactly one decision_id required")
        for node in cast(list[str], nodes):
            if node in declared_nodes:
                raise DecisionTraceabilityError(
                    f"test={test_id}: pytest node reused by another required test"
                )
            declared_nodes.add(node)

    collected = _collect_pytest_nodes(root)
    missing = sorted(declared_nodes - collected)
    if missing:
        raise DecisionTraceabilityError(f"tests: missing pytest nodes {missing}")


def _collect_pytest_nodes(root: Path) -> set[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or getattr(exc, "stdout", "")
        raise DecisionTraceabilityError(f"tests: pytest collection failed: {detail}") from exc
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    }
