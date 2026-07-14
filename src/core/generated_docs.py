"""Pure generated-document rendering from the compiled registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .errors import GeneratedFileDriftError

HEADER = "GENERATED FILE — DO NOT EDIT\nSource: config/pipeline.yaml\n\n"


def render_documents(registry: dict[str, object]) -> dict[str, str]:
    required = ("agent_contract", "artifacts", "schemas", "steps", "decisions", "implementation_controls", "tests", "access_matrix")
    values = {key: registry[key] for key in required}
    if not all(isinstance(value, dict) for value in values.values()):
        raise ValueError("registry required to render documentation")
    contract = cast(dict[str, Any], values["agent_contract"])
    artifacts = cast(dict[str, Any], values["artifacts"])
    schemas = cast(dict[str, Any], values["schemas"])
    steps = cast(dict[str, Any], values["steps"])
    decisions = cast(dict[str, Any], values["decisions"])
    controls = cast(dict[str, Any], values["implementation_controls"])
    tests = cast(dict[str, Any], values["tests"])
    access = cast(dict[str, Any], values["access_matrix"])
    agent = [HEADER, "# Agent contract\n"] + [f"- {line}\n" for line in contract["rules"]]
    artifact = [HEADER, "# Artifact catalog\n", "| ID | Producer | Schema | Path |\n|---|---|---|---|\n"]
    artifact += [f"| {key} | {value['producer_step']} | {value['schema_id']} | `{value['path_template']}` |\n" for key, value in sorted(artifacts.items()) if isinstance(value, dict)]
    schema = [HEADER, "# Schema catalog\n"] + [f"## {key}\n\nVersion: {value.get('version')}\n\n" for key, value in sorted(schemas.items()) if isinstance(value, dict)]
    step = [HEADER, "# Step cards\n"] + [f"## {key}\n\n{value['description']}\n\nReads: {value['reads']}\n\nWrites: {value['writes']}\n\n" for key, value in sorted(steps.items()) if isinstance(value, dict)]
    decision = [HEADER, "# D01-D45 traceability\n", "| Decision | Title | Chapter reference | Tests |\n|---|---|---|---|\n"]
    decision += [f"| {key} | {value.get('canonical_title', '')} | {value['chapter_reference']} | {', '.join(value['test_ids'])} |\n" for key, value in sorted(decisions.items()) if isinstance(value, dict)]
    control = [HEADER, "# Implementation controls\n"] + [f"## {key}\n\n{value['statement']}\n\n" for key, value in sorted(controls.items()) if isinstance(value, dict)]
    test = [HEADER, "# Test catalog\n"] + [f"## {key}\n\n{value['description']}\n\n" for key, value in sorted(tests.items()) if isinstance(value, dict)]
    access_doc = [HEADER, "# Access matrix\n\n```json\n", json.dumps(access, indent=2, sort_keys=True), "\n```\n"]
    return {"AGENTS.md": "".join(agent), "docs/generated/ARTIFACT_CATALOG.md": "".join(artifact), "docs/generated/SCHEMA_CATALOG.md": "".join(schema), "docs/generated/STEP_CARDS.md": "".join(step), "docs/generated/D01_D45_TRACEABILITY.md": "".join(decision), "docs/generated/IMPLEMENTATION_CONTROLS.md": "".join(control), "docs/generated/TEST_CATALOG.md": "".join(test), "docs/generated/ACCESS_MATRIX.md": "".join(access_doc)}


def write_or_check_documents(root: Path, documents: dict[str, str], check: bool) -> None:
    stale = [relative for relative, content in documents.items() if not (root / relative).is_file() or (root / relative).read_text(encoding="utf-8") != content]
    if check and stale:
        raise GeneratedFileDriftError(f"generated files are stale: {stale}; run bootstrap with --write")
    if not check:
        for relative, content in documents.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
