"""Pure generated-document rendering from the compiled registry."""

from __future__ import annotations

from pathlib import Path

from .errors import GeneratedFileDriftError

HEADER = "GENERATED FILE — DO NOT EDIT\nSource: config/pipeline.yaml\n\n"


def render_documents(registry: dict[str, object]) -> dict[str, str]:
    """Render all repository documentation derived from the one manifest."""
    contract = registry["agent_contract"]
    artifacts = registry["artifacts"]
    schemas = registry["schemas"]
    steps = registry["steps"]
    decisions = registry["decisions"]
    if not all(isinstance(item, dict) for item in (contract, artifacts, schemas, steps, decisions)):
        raise ValueError("registry required to render documentation")
    agent_lines = [HEADER, "# Agent contract\n"]
    for item in contract["rules"]:
        agent_lines.append(f"- {item}\n")
    agent_lines.append(
        "\nCatalogs: docs/generated/ARTIFACT_CATALOG.md, SCHEMA_CATALOG.md, STEP_CARDS.md, D01_D45_TRACEABILITY.md\n"
    )
    artifact_lines = [
        HEADER,
        "# Artifact catalog\n",
        "| ID | Producer | Schema | Path |\n|---|---|---|---|\n",
    ]
    for artifact_id, value in sorted(artifacts.items()):
        if isinstance(value, dict):
            artifact_lines.append(
                f"| {artifact_id} | {value['producer_step']} | {value['schema_id']} | `{value['path_template']}` |\n"
            )
    schema_lines = [HEADER, "# Schema catalog\n"]
    for schema_id, value in sorted(schemas.items()):
        if isinstance(value, dict):
            schema_lines.append(f"## {schema_id}\n\nVersion: {value.get('version')}\n\n")
    step_lines = [HEADER, "# Step cards\n"]
    for step_id, value in sorted(steps.items()):
        if isinstance(value, dict):
            step_lines.append(
                f"## {step_id}\n\n{value['description']}\n\nReads: {value['reads']}\n\nWrites: {value['writes']}\n\n"
            )
    decision_lines = [
        HEADER,
        "# D01–D45 traceability\n",
        "| Decision | Statement | Tests |\n|---|---|---|\n",
    ]
    for decision_id, value in sorted(decisions.items()):
        if isinstance(value, dict):
            decision_lines.append(
                f"| {decision_id} | {value['statement']} | {', '.join(value['test_ids'])} |\n"
            )
    return {
        "AGENTS.md": "".join(agent_lines),
        "docs/generated/ARTIFACT_CATALOG.md": "".join(artifact_lines),
        "docs/generated/SCHEMA_CATALOG.md": "".join(schema_lines),
        "docs/generated/STEP_CARDS.md": "".join(step_lines),
        "docs/generated/D01_D45_TRACEABILITY.md": "".join(decision_lines),
    }


def write_or_check_documents(root: Path, documents: dict[str, str], check: bool) -> None:
    """Write generated docs, or fail when checked-in files drift."""
    stale = [
        relative
        for relative, content in documents.items()
        if not (root / relative).is_file()
        or (root / relative).read_text(encoding="utf-8") != content
    ]
    if check and stale:
        raise GeneratedFileDriftError(
            f"generated files are stale: {stale}; run bootstrap with --write"
        )
    if not check:
        for relative, content in documents.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
