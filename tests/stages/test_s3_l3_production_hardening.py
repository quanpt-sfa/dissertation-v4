from __future__ import annotations

from pathlib import Path

import yaml


def test_primary_source_set_excludes_next_year_s3_enforcement() -> None:
    config = yaml.safe_load(Path("config/methodology/measurement.yaml").read_text(encoding="utf-8"))
    measurement = config["measurement"]
    primary = measurement["source_sets"][measurement["primary_source_set_id"]]
    assert primary["role"] == "primary_measurement"
    assert primary["temporal_alignment"] == "fiscal_year_annual_reporting"
    assert sorted(primary["sources"]) == [
        "S1_profit_adjustment",
        "S1_revenue_adjustment",
        "S2_audit_opinion",
    ]
    assert not any(source.startswith("S3_") for source in primary["sources"])
    boundary = measurement["source_sets"]["s3_content_boundary"]
    assert measurement["boundary_s3_endpoint"] == "S3_CONTENT"
    assert boundary["sources"] == ["S3_CONTENT"]
    assert boundary["role"] == "prospective_enforcement_boundary"


def test_known_case_registry_contract_is_external_validation_only() -> None:
    config = yaml.safe_load(
        Path("config/methodology/source_catalog.yaml").read_text(encoding="utf-8")
    )
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
