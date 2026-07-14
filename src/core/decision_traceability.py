"""D01--D45 traceability validation and machine-readable rows."""

from __future__ import annotations

from .errors import DecisionTraceabilityError


def path_exists(registry: object, dotted: str) -> bool:
    """Resolve a dotted anchor against normalized registry mappings."""
    current: object = registry
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def validate_decisions(registry: dict[str, object]) -> list[dict[str, object]]:
    """Require exactly D01 through D45, anchors, tests, steps, and evidence."""
    decisions = registry.get("decisions")
    tests = registry.get("tests")
    steps = registry.get("steps")
    artifacts = registry.get("artifacts")
    if not all(isinstance(item, dict) for item in (decisions, tests, steps, artifacts)):
        raise DecisionTraceabilityError("decisions: decisions, tests, steps, and artifacts must be mappings")
    expected = {f"D{number:02d}" for number in range(1, 46)}
    if set(decisions) != expected:
        raise DecisionTraceabilityError("decisions: require exactly D01 through D45; complete missing IDs")
    rows: list[dict[str, object]] = []
    for decision_id in sorted(decisions):
        item = decisions[decision_id]
        if not isinstance(item, dict) or not isinstance(item.get("statement"), str) or item["statement"].startswith("TODO"):
            raise DecisionTraceabilityError(f"decision={decision_id}: completed non-placeholder statement required")
        anchors = item.get("config_anchors")
        test_ids = item.get("test_ids")
        evidence = item.get("output_evidence")
        if not isinstance(anchors, list) or not anchors:
            raise DecisionTraceabilityError(f"decision={decision_id}: at least one config anchor required")
        if not isinstance(test_ids, list) or not test_ids:
            raise DecisionTraceabilityError(f"decision={decision_id}: at least one test ID required")
        if not isinstance(evidence, list) or not evidence:
            raise DecisionTraceabilityError(f"decision={decision_id}: at least one output evidence artifact required")
        for anchor in anchors:
            if not isinstance(anchor, str) or not path_exists(registry, anchor):
                raise DecisionTraceabilityError(f"decision={decision_id}, anchor={anchor}: unknown normalized registry path")
        for test_id in test_ids:
            if test_id not in tests:
                raise DecisionTraceabilityError(f"decision={decision_id}, test={test_id}: unknown test ID")
        for step_id in item.get("enforced_by_steps", []):
            if step_id not in steps:
                raise DecisionTraceabilityError(f"decision={decision_id}, step={step_id}: unknown step ID")
        for artifact_id in evidence:
            if artifact_id not in artifacts:
                raise DecisionTraceabilityError(f"decision={decision_id}, artifact={artifact_id}: unknown output artifact")
        rows.append({"decision_id": decision_id, **item})
    return rows
