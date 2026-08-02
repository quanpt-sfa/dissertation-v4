"""Regression coverage for production S2 vocabulary and outer-fold dispatch."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import yaml

from core.evidence_registry import LogicalEvidenceSource
from core.registry_compiler import compile_registry
from evidence.annual import OpinionRow, build_audit_opinion_records

ROOT = Path(__file__).resolve().parents[2]
FoldExecutionSets = Callable[[dict[str, object]], tuple[list[str], list[str]]]


def _registry() -> dict[str, object]:
    return cast(
        dict[str, object],
        compile_registry(ROOT / "config" / "pipeline.yaml").registry,
    )


def _s2_source() -> LogicalEvidenceSource:
    raw = cast(
        dict[str, Any],
        yaml.safe_load(
            (ROOT / "config" / "methodology" / "source_catalog.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )
    catalog = cast(dict[str, Any], raw["source_catalog"])
    profiles = cast(dict[str, Any], catalog["profiles"])
    profile = cast(dict[str, Any], profiles["audit_annual_long"])
    evidence = cast(dict[str, Any], profile["evidence_mapping"])
    logical_sources = cast(list[dict[str, Any]], evidence["logical_sources"])
    logical = logical_sources[0]
    return LogicalEvidenceSource(
        source_id=str(logical["source_id"]),
        endpoint_id=str(logical["endpoint_id"]),
        channel_id=str(profile["channel_id"]),
        physical_source_id="audit_annual_long",
        profile_id="audit_annual_long",
        processor=str(evidence["processor"]),
        temporal_role=str(evidence["temporal_role"]),
        availability_rule=str(evidence["availability_rule"]),
        explicit_negative_allowed=bool(evidence["explicit_negative_allowed"]),
        verification_status=str(profile["verification_status"]),
        opportunity_semantic=str(evidence["opportunity_semantic"]),
        logical_config=dict(logical),
        processor_config=dict(evidence),
    )


def _load_runner() -> ModuleType:
    path = ROOT / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("production_run_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load production runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _opinion_row(raw: str) -> OpinionRow:
    return OpinionRow(
        firm_id="F1",
        fiscal_year=2024,
        opinion_raw=raw,
        audit_indicator="audit_opinion",
        period_type="annual",
        statement_scope="consolidated",
        audit_status="audited",
        source_ref="final-parquet:F1:2024",
    )


def test_final_parquet_unmodified_opinion_is_an_explicit_clean_negative() -> None:
    result = build_audit_opinion_records(
        panel_anchors={("F1", 2024): datetime(2025, 3, 31)},
        rows=[_opinion_row("UNMODIFIED")],
        source=_s2_source(),
    )

    record = result.records[0]
    assert record.outcome is False
    assert record.source_opportunity is True
    assert record.outcome_basis == "S2_CLEAN_OPINION"
    assert result.audit["explicit_negative_count"] == 1
    assert result.audit["unknown_count"] == 0


def test_unknown_opinion_is_not_coerced_to_clean() -> None:
    result = build_audit_opinion_records(
        panel_anchors={("F1", 2024): datetime(2025, 3, 31)},
        rows=[_opinion_row("UNKNOWN")],
        source=_s2_source(),
    )

    record = result.records[0]
    assert record.outcome is None
    assert record.source_opportunity is False
    assert record.outcome_basis == "S2_OPINION_UNMAPPED"


def test_runner_keeps_initial_fold_for_p09_but_not_confirmatory_stages() -> None:
    fold_execution_sets = cast(FoldExecutionSets, getattr(_load_runner(), "_fold_execution_sets"))
    p09_folds, confirmatory_folds = fold_execution_sets(_registry())

    assert p09_folds == ["2020", "2021", "2022", "2023", "2024"]
    assert confirmatory_folds == ["2021", "2022", "2023", "2024"]
