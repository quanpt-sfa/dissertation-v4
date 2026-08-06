"""Production uses one modeling Parquet plus one sealed validation registry."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
FINAL_PATH = "data/source/vn_pipeline_final_firm_year_2015_2025.parquet"
KNOWN_CASE_PATH = "data/source/known_case_registry.csv"


def _profiles() -> dict[str, dict[str, object]]:
    raw = yaml.safe_load(
        (ROOT / "config/methodology/source_catalog.yaml").read_text(encoding="utf-8")
    )
    catalog = cast(dict[str, object], raw)["source_catalog"]
    assert isinstance(catalog, dict)
    profiles = cast(dict[str, object], catalog)["profiles"]
    assert isinstance(profiles, dict)
    return {
        str(source_id): cast(dict[str, object], value)
        for source_id, value in cast(dict[object, object], profiles).items()
    }


def test_modeling_views_share_one_parquet_and_known_cases_remain_separate() -> None:
    profiles = _profiles()
    enabled = {name: value for name, value in profiles.items() if value.get("enabled") is True}

    assert set(enabled) == {
        "financial_statement_core_long",
        "audit_annual_long",
        "sanction_evidence",
        "known_cases",
    }
    for source_id in (
        "financial_statement_core_long",
        "audit_annual_long",
        "sanction_evidence",
    ):
        profile = enabled[source_id]
        discovery = cast(dict[str, object], profile["discovery"])
        assert discovery["globs"] == [FINAL_PATH]
        assert profile["format"] == "parquet"
        assert profile["required"] is True

    known = enabled["known_cases"]
    known_discovery = cast(dict[str, object], known["discovery"])
    assert known_discovery["globs"] == [KNOWN_CASE_PATH]
    assert known["format"] == "csv"
    assert known["required"] is True
    assert known["role"] == "known_case"


def test_physical_separation_preserves_measurement_and_validation_roles() -> None:
    profiles = _profiles()
    processors: dict[str, str] = {}
    panel_sources: list[str] = []
    for source_id, profile in profiles.items():
        evidence = profile.get("evidence_mapping")
        if isinstance(evidence, dict):
            processor = cast(dict[str, object], evidence).get("processor")
            if isinstance(processor, str):
                processors[source_id] = processor
        panel = profile.get("panel_mapping")
        if isinstance(panel, dict) and cast(dict[str, object], panel).get("enabled") is True:
            panel_sources.append(source_id)

    assert processors == {
        "financial_statement_core_long": "audit_adjustment",
        "audit_annual_long": "audit_opinion",
        "sanction_evidence": "sanction_calendar_year",
    }
    assert panel_sources == ["financial_statement_core_long"]
    assert profiles["known_cases"]["measurement_process_role"] == (
        "external_falsification_positive_evidence"
    )
