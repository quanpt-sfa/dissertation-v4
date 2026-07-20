from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


def _module() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "p07_features.py"
    spec = importlib.util.spec_from_file_location("p07_features_cli", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load P07 CLI module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p07_production_source_has_no_external_feature_store_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "scripts" / "p07_features.py"
    ).read_text(encoding="utf-8")
    assert "assemble_feature_input_panel" not in source
    assert "from features.store" not in source
    assert "external_feature_store_used" in source


def test_pipeline_generated_receipts_preserve_legacy_artifact_contract() -> None:
    module = _module()
    panel = pd.DataFrame(
        {
            "firm_master_id": pd.Series(["A", "B"], dtype="string"),
            "fiscal_year": pd.Series([2022, 2022], dtype="int16"),
            "feature_a": pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    definitions = [
        {
            "feature_id": "feature_a",
            "research_decision_status": "LOCKED",
            "confirmatory_status": "confirmatory",
            "model_eligibility": "eligible",
        },
        {
            "feature_id": "feature_b",
            "research_decision_status": "RESEARCH_DECISION_REQUIRED",
            "confirmatory_status": "blocked",
            "model_eligibility": "blocked_until_locked",
        },
    ]
    receipts = module._pipeline_generated_feature_receipts(
        panel=panel,
        definitions=definitions,
        firm_column="firm_master_id",
    )
    report = receipts["report"]
    assert report["status"] == "PIPELINE_GENERATED_FEATURES_VALID"
    assert report["external_feature_store_used"] is False
    assert report["external_manifest_required"] is False
    assert report["external_crosswalk_required"] is False
    assert report["locked_feature_count"] == 1
    assert report["unresolved_feature_count"] == 1
    assert list(receipts["file_audit"]["feature_id"]) == ["feature_a", "feature_b"]
    assert set(receipts["identifier_audit"]["mapping_status"]) == {
        "CANONICAL_PIPELINE_ID"
    }
    assert receipts["availability_violations"].empty
