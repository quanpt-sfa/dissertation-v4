from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from core.errors import MethodologicalInvariantError
from core.registry_compiler import compile_registry

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AssertionError("expected YAML mapping")
    return cast(dict[str, Any], raw)


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _source(panel_mapping: dict[str, object] | None) -> dict[str, object]:
    source: dict[str, object] = {
        "enabled": True,
        "channel_id": "S1",
        "source_type": "official",
        "source_agency": "Agency",
        "original_unit": "firm-year",
        "related_period_field": "year",
        "availability_date_field": "available",
        "availability_date_source": "publication",
        "coverage_dimensions": ["year"],
        "role": "predictor",
        "verification_status": "observed",
        "data_risks": [],
        "relative_path": "source.csv",
        "format": "csv",
        "encoding": "utf-8",
        "delimiter": ",",
        "locked_sha256": "a" * 64,
        "schema": {
            "required_columns": ["entity", "year", "available"],
            "optional_columns": ["ticker"],
            "key_columns": ["entity", "year"],
            "date_columns": ["available"],
            "required_date_columns": ["available"],
            "numeric_columns": {},
            "allow_extra_columns": False,
            "key_unique": True,
            "row_count_min": 1,
        },
    }
    if panel_mapping is not None:
        source["panel_mapping"] = panel_mapping
    return source


def _copy_config(tmp_path: Path) -> Path:
    target = tmp_path / "config"
    shutil.copytree(ROOT / "config", target)
    return target / "pipeline.yaml"


def test_enabled_source_requires_panel_mapping(tmp_path: Path) -> None:
    pipeline = _copy_config(tmp_path)
    path = pipeline.parent / "methodology" / "data_sources.yaml"
    document = _load(path)
    document["data_sources"]["source_registry"]["sources"]["source_a"] = _source(None)
    _write(path, document)
    with pytest.raises(MethodologicalInvariantError, match="panel_mapping"):
        compile_registry(pipeline)


def test_ticker_identity_is_rejected_during_protocol_lock(tmp_path: Path) -> None:
    pipeline = _copy_config(tmp_path)
    path = pipeline.parent / "methodology" / "data_sources.yaml"
    document = _load(path)
    document["data_sources"]["source_registry"]["sources"]["source_a"] = _source(
        {
            "firm_id_field": "ticker",
            "fiscal_year_field": "year",
            "availability_date_field": "available",
            "fiscal_year_end_field": None,
            "ticker_field": "ticker",
            "contributes_to_firm_master": True,
            "core_predictor": True,
        }
    )
    _write(path, document)
    with pytest.raises(MethodologicalInvariantError, match="ticker cannot"):
        compile_registry(pipeline)


def test_enabled_sources_require_at_least_one_core_predictor(tmp_path: Path) -> None:
    pipeline = _copy_config(tmp_path)
    path = pipeline.parent / "methodology" / "data_sources.yaml"
    document = _load(path)
    document["data_sources"]["source_registry"]["sources"]["source_a"] = _source(
        {
            "firm_id_field": "entity",
            "fiscal_year_field": "year",
            "availability_date_field": "available",
            "fiscal_year_end_field": None,
            "ticker_field": "ticker",
            "contributes_to_firm_master": True,
            "core_predictor": False,
        }
    )
    _write(path, document)
    with pytest.raises(MethodologicalInvariantError, match="core predictor"):
        compile_registry(pipeline)
