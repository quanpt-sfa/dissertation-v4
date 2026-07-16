# pyright: basic
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT = Path("scripts/p07c_feature_catalogue_audit.py")


def _module() -> Any:
    spec = importlib.util.spec_from_file_location("p07c_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authoritative_catalogue_has_unchanged_unique_feature_ids() -> None:
    frame = pd.read_csv("methodology/catalogue/FSF_Literature_Backed_Feature_Catalogue.csv")
    assert len(frame) == 211
    assert frame.feature_id.is_unique


def test_prohibited_and_transcription_rows_fail_closed() -> None:
    module = _module()
    panel = pd.DataFrame(columns=["issuer_ticker", "fiscal_year", "canonical_item"])
    blocked = pd.Series(
        {
            "feature_id": "direct_identity",
            "formula_or_definition": "identity",
            "required_raw_concepts": "",
            "specification_tier": "BLOCK",
        }
    )
    transcription = pd.Series(
        {
            "feature_id": "ratio",
            "formula_or_definition": "SOURCE_FORMULA_TRANSCRIPTION_REQUIRED",
            "required_raw_concepts": "",
            "specification_tier": "C3",
        }
    )
    assert module.audit_row(blocked, panel, set())["final_researcher_decision"] == "BLOCK"
    assert (
        module.audit_row(transcription, panel, set())["decision_reason"]
        == "SOURCE_FORMULA_TRANSCRIPTION_REQUIRED"
    )


def test_validation_only_prepost_never_becomes_production_predictor() -> None:
    module = _module()
    panel = pd.DataFrame(columns=["issuer_ticker", "fiscal_year", "canonical_item"])
    prepost = pd.Series(
        {
            "feature_id": "prepost_x",
            "formula_or_definition": "pre/post audit adjustment",
            "required_raw_concepts": "",
            "specification_tier": "N1",
        }
    )
    result = module.audit_row(prepost, panel, set())
    assert result["source_authorization"] == "validation_only"
    assert result["final_researcher_decision"] == "BLOCK"


def test_canonical_score_is_not_recommended_with_unavailable_components() -> None:
    module = _module()
    panel = pd.DataFrame(columns=["issuer_ticker", "fiscal_year", "canonical_item"])
    score = pd.Series(
        {
            "feature_id": "beneish_m_score",
            "formula_or_definition": "M score",
            "required_raw_concepts": "DSRI; GMI; DEPI",
            "specification_tier": "C1",
        }
    )
    result = module.audit_row(score, panel, set())
    assert result["mapping_status"] == "CONCEPT_UNAVAILABLE"
    assert result["final_researcher_decision"] == "RESEARCHER_APPROVAL_REQUIRED"


def test_audit_uses_no_outcome_or_known_case_inputs() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "sealed_outcome_store" not in text
    assert "known_cases.csv" not in text
