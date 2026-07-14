# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import openpyxl
import pandas as pd

from snapshot.builder import build_snapshot, load_snapshot, write_snapshot
from snapshot.injection import inject_snapshot


def _catalog() -> dict[str, object]:
    return {
        "root_environment_variable": "DISSERTATION_RAW_ROOT",
        "snapshot_schema_version": 1,
        "profiles": {
            "panel": {
                "enabled": True,
                "required": True,
                "discovery": {
                    "globs": ["data/panel.parquet"],
                    "excludes": [],
                    "cardinality": "one",
                },
                "format": "parquet",
                "channel_id": "S1",
                "source_type": "panel",
                "source_agency": "test",
                "original_unit": "firm-year",
                "role": "predictor",
                "verification_status": "observed",
                "coverage_dimensions": ["year"],
                "data_risks": [],
                "semantic_fields": {
                    "firm_id": ["firm_id"],
                    "fiscal_year": ["fiscal_year"],
                    "availability_date": ["publication_date"],
                },
                "required_semantic_fields": ["firm_id", "fiscal_year", "availability_date"],
                "key_semantics": ["firm_id", "fiscal_year"],
                "date_semantics": ["availability_date"],
                "panel_mapping": {
                    "enabled": True,
                    "core_predictor": True,
                    "contributes_to_firm_master": True,
                },
            },
            "reference": {
                "enabled": True,
                "required": True,
                "discovery": {
                    "globs": ["data/reference.xlsx"],
                    "excludes": [],
                    "cardinality": "one",
                },
                "format": "xlsx",
                "reader": {"sheet_name": "Sheet1", "header_row": 2},
                "channel_id": "S3",
                "source_type": "reference",
                "source_agency": "test",
                "original_unit": "firm",
                "role": "reference",
                "verification_status": "observed",
                "coverage_dimensions": [],
                "data_risks": [],
                "semantic_fields": {"firm_id": ["Mã"]},
                "required_semantic_fields": ["firm_id"],
                "key_semantics": ["firm_id"],
                "date_semantics": [],
                "panel_mapping": {"enabled": False},
            },
        },
    }


def _raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    data = root / "data"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "firm_id": ["A", "B"],
            "fiscal_year": [2023, 2023],
            "publication_date": ["2024-03-31", "2024-04-01"],
            "value": [1.0, 2.0],
        }
    ).to_parquet(data / "panel.parquet", index=False)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Sheet1"
    sheet.append(["title"])
    sheet.append(["Mã", "Tên"])
    sheet.append(["A", "Firm A"])
    workbook.save(data / "reference.xlsx")
    return root


def test_snapshot_discovers_real_files_and_locks_schema(tmp_path: Path) -> None:
    root = _raw_root(tmp_path)
    snapshot = build_snapshot(
        registry={"source_catalog": _catalog()},
        raw_root=root,
        snapshot_id="snapshot-a",
    )
    sources = snapshot["sources"]
    assert isinstance(sources, list)
    assert len(sources) == 2
    panel = next(
        item for item in sources if isinstance(item, dict) and item["source_id"] == "panel"
    )
    assert panel["resolved_semantics"]["availability_date"] == "publication_date"
    assert panel["panel_mapping"]["core_predictor"] is True
    assert panel["schema"]["allow_extra_columns"] is False


def test_snapshot_is_immutable_and_hash_verified(tmp_path: Path) -> None:
    root = _raw_root(tmp_path)
    snapshot = build_snapshot(
        registry={"source_catalog": _catalog()},
        raw_root=root,
        snapshot_id="snapshot-a",
    )
    path = tmp_path / "snapshot.json"
    write_snapshot(path, snapshot)
    assert load_snapshot(path)["snapshot_hash"] == snapshot["snapshot_hash"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["snapshot_id"] = "tampered"
    path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        load_snapshot(path)
    except ValueError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered snapshot was accepted")


def test_snapshot_injection_populates_runtime_source_registry(tmp_path: Path) -> None:
    root = _raw_root(tmp_path)
    snapshot = build_snapshot(
        registry={"source_catalog": _catalog()},
        raw_root=root,
        snapshot_id="snapshot-a",
    )
    path = tmp_path / "snapshot.json"
    write_snapshot(path, snapshot)
    registry: dict[str, object] = {"data_sources": {}}
    inject_snapshot(registry, path)
    data_sources = registry["data_sources"]
    assert isinstance(data_sources, dict)
    source_registry_value = data_sources["source_registry"]
    assert isinstance(source_registry_value, dict)
    source_registry = cast(dict[str, object], source_registry_value)
    sources_value = source_registry["sources"]
    assert isinstance(sources_value, dict)
    assert set(cast(dict[str, object], sources_value)) == {"panel", "reference"}
