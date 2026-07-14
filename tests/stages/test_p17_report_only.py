"""P17 must stay a report-only consumer of finalized artifacts."""

from __future__ import annotations

import ast
from pathlib import Path

from reporting.service import (
    build_chapter4_tables,
    build_verdict_matrix,
    render_gate_figure,
)

ROOT = Path(__file__).resolve().parents[2]


def test_p17_does_not_import_training_or_label_modules() -> None:
    tree = ast.parse((ROOT / "scripts" / "p17_build_outputs.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        str(node.module).split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported.isdisjoint({"modeling", "labels", "selection", "simulation"})


def test_p17_outputs_are_machine_readable_and_known_cases_cannot_upgrade() -> None:
    verdicts = build_verdict_matrix(
        {"verdict": "FAIL", "criteria_source": "locked_registry"},
        {"verdict": "INELIGIBLE_PARENT_GATE_FAILED", "criteria_source": "locked_registry"},
        [{"record_type": "summary", "soft_veto": False}],
    )
    known = next(item for item in verdicts if item["gate_id"] == "KNOWN_CASE_SOFT_VETO")
    assert known["may_upgrade_parent"] is False
    tables = build_chapter4_tables([], verdicts, [])
    assert tables["report_only"] is True
    assert render_gate_figure(verdicts).startswith("<svg")
