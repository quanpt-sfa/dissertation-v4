"""Production input is one physical Parquet exposed through semantic views."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
FINAL_PATH = "data/source/vn_pipeline_final_firm_year_2015_2025.parquet"


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


def test_all_enabled_views_resolve_to_one_physical_parquet() -> None:
    profiles = _profiles()
    enabled = {name: value for name, value in profiles.items() if value.get("enabled") is True}

    assert set(enabled) == {
        "financial_statement_core_long",
        "audit_annual_long",
        "sanction_evidence",
        "known_cases",
    }
    paths: set[str] = set()
    for profile in enabled.values():
        discovery = profile["discovery"]
        assert isinstance(discovery, dict)
        globs = cast(dict[str, object], discovery)["globs"]
        assert globs == [FINAL_PATH]
        assert profile["format"] == "parquet"
        assert profile["required"] is True
        paths.update(cast(list[str], globs))

    assert paths == {FINAL_PATH}


def test_single_file_preserves_separate_measurement_roles() -> None:
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
    assert profiles["known_cases"]["role"] == "known_case"
