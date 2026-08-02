"""Regression coverage for the P08-to-P09 access boundary."""

from pathlib import Path
from typing import Any, cast

from core.access_matrix import assert_access
from core.registry_compiler import compile_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_p09_declares_mcse_report_as_a_required_read() -> None:
    """P09 must be authorized to enforce the P08 completion gate it reads."""
    registry = compile_registry(PROJECT_ROOT / "config" / "pipeline.yaml").registry
    steps = cast(dict[str, dict[str, Any]], registry["steps"])
    access_matrix = cast(dict[str, dict[str, Any]], registry["access_matrix"])
    artifacts = cast(dict[str, dict[str, Any]], registry["artifacts"])
    access_control = cast(dict[str, Any], registry["access_control"])

    assert "mcse_report" in steps["P09"]["reads"]
    assert "mcse_report" in access_matrix["P09"]["reads"]
    assert "mcse_report" not in access_matrix["P09"]["optional_reads"]

    assert_access(
        access_matrix,
        artifacts,
        access_control,
        step_id="P09",
        artifact_id="mcse_report",
        mode="read",
        state="SIMULATED",
    )
