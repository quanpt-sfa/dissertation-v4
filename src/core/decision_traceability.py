"""D01--D45 traceability validation and machine-readable rows."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from .errors import DecisionTraceabilityError


def path_exists(registry: object, dotted: str) -> bool:
    """Resolve a dotted anchor against normalized registry mappings."""
    current: object = registry
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = cast(dict[str, Any], current)[part]
    return True


def validate_decisions(registry: dict[str, object]) -> list[dict[str, object]]:
    """Require exactly D01 through D45, anchors, tests, steps, and evidence."""
    decisions = registry.get("decisions")
    tests = registry.get("tests")
    steps = registry.get("steps")
    artifacts = registry.get("artifacts")
    if not all(isinstance(item, dict) for item in (decisions, tests, steps, artifacts)):
        raise DecisionTraceabilityError(
            "decisions: decisions, tests, steps, and artifacts must be mappings"
        )
    decisions = cast(dict[str, Any], decisions)
    tests = cast(dict[str, Any], tests)
    steps = cast(dict[str, Any], steps)
    artifacts = cast(dict[str, Any], artifacts)
    expected = {f"D{number:02d}" for number in range(1, 46)}
    if set(decisions) != expected:
        raise DecisionTraceabilityError(
            "decisions: require exactly D01 through D45; complete missing IDs"
        )
    rows: list[dict[str, object]] = []
    used_tests: set[str] = set()
    for decision_id in sorted(decisions):
        item = decisions[decision_id]
        if not isinstance(item, dict):
            raise DecisionTraceabilityError(f"decision={decision_id}: mapping required")
        item = cast(dict[str, Any], item)
        if (
            not isinstance(item.get("canonical_title"), str)
            or not isinstance(item.get("statement"), str)
            or item["statement"].startswith(("TODO", "TBD", "placeholder"))
        ):
            raise DecisionTraceabilityError(
                f"decision={decision_id}: canonical title and completed statement required"
            )
        anchors = item.get("config_anchors")
        test_ids = item.get("test_ids")
        evidence = item.get("output_evidence")
        if not isinstance(anchors, list) or not anchors:
            raise DecisionTraceabilityError(
                f"decision={decision_id}: at least one config anchor required"
            )
        if not isinstance(test_ids, list) or not test_ids:
            raise DecisionTraceabilityError(
                f"decision={decision_id}: at least one test ID required"
            )
        if not isinstance(evidence, list) or not evidence:
            raise DecisionTraceabilityError(
                f"decision={decision_id}: at least one output evidence artifact required"
            )
        if not all(isinstance(anchor, str) for anchor in anchors):
            raise DecisionTraceabilityError(f"decision={decision_id}: anchor strings required")
        if not all(isinstance(test_id, str) for test_id in test_ids):
            raise DecisionTraceabilityError(f"decision={decision_id}: test ID strings required")
        if not all(isinstance(artifact_id, str) for artifact_id in evidence):
            raise DecisionTraceabilityError(f"decision={decision_id}: evidence strings required")
        anchors = cast(list[str], anchors)
        test_ids = cast(list[str], test_ids)
        evidence = cast(list[str], evidence)
        for anchor in anchors:
            if not path_exists(registry, anchor):
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, anchor={anchor}: unknown normalized registry path"
                )
        for test_id in test_ids:
            if test_id not in tests:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, test={test_id}: unknown test ID"
                )
        for step_id in cast(list[str], item.get("enforced_by_steps", [])):
            if step_id not in steps:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, step={step_id}: unknown step ID"
                )
        for artifact_id in evidence:
            if artifact_id not in artifacts:
                raise DecisionTraceabilityError(
                    f"decision={decision_id}, artifact={artifact_id}: unknown output artifact"
                )
        expected_reference = f"Appendix B, {decision_id}"
        if item.get("chapter_reference") != expected_reference:
            raise DecisionTraceabilityError(
                f"decision={decision_id}: chapter_reference must be {expected_reference}"
            )
        controls = item.get("implementation_controls")
        if not isinstance(controls, list) or not controls:
            raise DecisionTraceabilityError(
                f"decision={decision_id}: implementation_controls required"
            )
        used_tests.update(test_ids)
        rows.append({"decision_id": decision_id, **item})
    if len(used_tests) < 10:
        raise DecisionTraceabilityError(
            "decisions: traceability cannot assign one generic test to all decisions"
        )
    return rows


def validate_test_registry(registry: dict[str, object], root: Path) -> None:
    """Require every declared pytest node to exist in pytest collection."""
    tests = registry.get("tests")
    if not isinstance(tests, dict):
        raise DecisionTraceabilityError("tests: mapping required")
    tests = cast(dict[str, Any], tests)
    expected = {f"T{number:03d}" for number in range(1, 46)}
    if set(tests) != expected:
        raise DecisionTraceabilityError("tests: require exactly T001 through T045")
    declared_nodes: set[str] = set()
    for test_id in sorted(tests):
        item = tests[test_id]
        if not isinstance(item, dict):
            raise DecisionTraceabilityError(f"test={test_id}: mapping required")
        item = cast(dict[str, Any], item)
        nodes = item.get("pytest_nodes")
        if not isinstance(nodes, list) or not nodes:
            raise DecisionTraceabilityError(f"test={test_id}: pytest_nodes required")
        if not all(isinstance(node, str) for node in nodes):
            raise DecisionTraceabilityError(f"test={test_id}: pytest node strings required")
        nodes = cast(list[str], nodes)
        for node in nodes:
            if not node.startswith("tests/"):
                raise DecisionTraceabilityError(f"test={test_id}: invalid pytest node")
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
        raise DecisionTraceabilityError("tests: pytest collection failed") from exc
    nodes: set[str] = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("tests/") and "::" in line:
            nodes.add(line)
    return nodes
