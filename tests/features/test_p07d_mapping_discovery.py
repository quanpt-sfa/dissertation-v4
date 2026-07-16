# pyright: basic
from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_p07d_preserves_authoritative_catalogue_universe() -> None:
    catalogue = pd.read_csv("methodology/catalogue/FSF_Literature_Backed_Feature_Catalogue.csv")
    assert len(catalogue) == 211
    assert catalogue.feature_id.is_unique


def test_p07d_discovery_contract_keeps_researcher_decisions_blank() -> None:
    script = Path("scripts/p07d_dataset_mapping_discovery.py").read_text(encoding="utf-8")
    assert "researcher_decision" in script
    assert "no_outcomes_read" in script
    assert "no_known_cases_read" in script


def test_p07d_validation_only_and_future_sources_remain_nonproduction() -> None:
    script = Path("scripts/p07d_dataset_mapping_discovery.py").read_text(encoding="utf-8")
    assert "VALIDATION_ONLY" in script
    assert "VALIDATION_ONLY_SOURCE" in script
    assert "EXTERNAL_UNTRACKED" in script
