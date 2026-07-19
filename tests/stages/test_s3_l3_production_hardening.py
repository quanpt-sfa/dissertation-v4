from __future__ import annotations

from pathlib import Path

import yaml


def test_primary_source_set_uses_only_s3_content() -> None:
    config = yaml.safe_load(Path("config/methodology/measurement.yaml").read_text(encoding="utf-8"))
    measurement = config["measurement"]
    primary = measurement["source_sets"][measurement["primary_source_set_id"]]["sources"]
    assert measurement["primary_s3_endpoint"] == "S3_CONTENT"
    assert sorted(source for source in primary if source.startswith("S3_")) == ["S3_CONTENT"]


def test_known_case_registry_contract_is_external_validation_only() -> None:
    config = yaml.safe_load(Path("config/methodology/source_catalog.yaml").read_text(encoding="utf-8"))
    known = config["source_catalog"]["profiles"]["known_cases"]
    assert known["discovery"]["globs"] == ["data/source/known_case_registry.csv"]
    required = set(known["required_semantic_fields"])
    assert {
        "case_construct",
        "case_role",
        "training_include_flag",
        "calibration_include_flag",
        "model_selection_include_flag",
        "external_validation_include_flag",
    }.issubset(required)
