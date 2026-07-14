"""Generate assurance documentation from the compiled registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .errors import GeneratedFileDriftError

HEADER = "GENERATED FILE — DO NOT EDIT\nSource: config/pipeline.yaml\n\n"


def render_documents(
    registry: dict[str, object],
) -> dict[str, str]:
    contract = cast(dict[str, Any], registry["agent_contract"])
    artifacts = cast(dict[str, Any], registry["artifacts"])
    schemas = cast(dict[str, Any], registry["schemas"])
    steps = cast(dict[str, Any], registry["steps"])
    rows = cast(list[dict[str, Any]], registry["decision_traceability"])
    controls = cast(dict[str, Any], registry["implementation_controls"])
    tests = cast(dict[str, Any], registry["tests"])
    access = cast(dict[str, Any], registry["access_matrix"])

    agent = [
        HEADER,
        "# Agent contract\n",
        *[f"- {rule}\n" for rule in contract["rules"]],
    ]
    artifact = [
        HEADER,
        "# Artifact catalog\n",
        "| ID | Producer | Schema | Path |\n|---|---|---|---|\n",
    ]
    artifact += [
        f"| {key} | {value['producer_step']} | {value['schema_id']} | `{value['path_template']}` |\n"
        for key, value in sorted(artifacts.items())
    ]
    schema = [HEADER, "# Schema catalog\n"] + [
        f"## {key}\n\nVersion: {value.get('version')}\n\n" for key, value in sorted(schemas.items())
    ]
    step = [HEADER, "# Step cards\n"] + [
        (
            f"## {key}\n\n{value['description']}\n\n"
            f"Reads: {value['reads']}\n\n"
            f"Writes: {value['writes']}\n\n"
            f"Required receipts: {value.get('required_receipts', [])}\n\n"
        )
        for key, value in sorted(steps.items())
    ]
    decision = [
        HEADER,
        "# D01–D45 traceability\n",
        "| ID | Canonical decision | Main specification | Test | Evidence |\n"
        "|---|---|---|---|---|\n",
    ]
    for row in rows:
        decision.append(
            f"| {row['decision_id']} | {row['canonical_title']} | "
            f"{row['main_specification']} | {', '.join(row['test_ids'])} | "
            f"{', '.join(row['output_evidence'])} |\n"
        )
    control = [HEADER, "# Implementation controls\n"] + [
        f"## {key}\n\n{value['statement']}\n\n" for key, value in sorted(controls.items())
    ]
    test = [HEADER, "# Test catalog\n"] + [
        f"## {key}\n\n{value['description']}\n\nNodes: {value['pytest_nodes']}\n\n"
        for key, value in sorted(tests.items())
    ]
    access_doc = [
        HEADER,
        "# Access matrix\n\n```json\n",
        json.dumps(access, indent=2, sort_keys=True),
        "\n```\n",
    ]
    return {
        "AGENTS.md": "".join(agent),
        "docs/generated/ARTIFACT_CATALOG.md": "".join(artifact),
        "docs/generated/SCHEMA_CATALOG.md": "".join(schema),
        "docs/generated/STEP_CARDS.md": "".join(step),
        "docs/generated/D01_D45_TRACEABILITY.md": "".join(decision),
        "docs/generated/IMPLEMENTATION_CONTROLS.md": "".join(control),
        "docs/generated/TEST_CATALOG.md": "".join(test),
        "docs/generated/ACCESS_MATRIX.md": "".join(access_doc),
    }


def write_or_check_documents(root: Path, documents: dict[str, str], check: bool) -> None:
    stale = [
        relative
        for relative, content in documents.items()
        if not (root / relative).is_file()
        or (root / relative).read_text(encoding="utf-8") != content
    ]
    if check and stale:
        raise GeneratedFileDriftError(f"generated files are stale: {stale}; run bootstrap --write")
    if not check:
        for relative, content in documents.items():
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
