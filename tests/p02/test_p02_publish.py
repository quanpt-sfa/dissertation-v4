from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from core.errors import ConfigurationError
from core.registry_compiler import compile_registry
from p02.builder import PanelBuildResult
from p02.publish import publish_p02_outputs

ROOT = Path(__file__).resolve().parents[2]


def _result() -> PanelBuildResult:
    return PanelBuildResult(
        firm_master=pd.DataFrame({"firm_master_id": pd.Series(["A"], dtype="string")}),
        firm_year_panel=pd.DataFrame(
            {
                "firm_master_id": pd.Series(["A"], dtype="string"),
                "fiscal_year": pd.Series([2023], dtype="int16"),
                "prediction_time": pd.Series([pd.Timestamp("2024-03-31")], dtype="datetime64[ns]"),
            }
        ),
        resolution_report={
            "run_id": "run-1",
            "protocol_hash": "a" * 64,
            "entity_resolution": [],
            "duplicate_groups": [],
            "excluded_rows": [],
            "excluded_firm_years": [],
            "summary": {},
        },
    )


def _registry() -> tuple[dict[str, Any], str]:
    compiled = compile_registry(ROOT / "config" / "pipeline.yaml")
    return cast(dict[str, Any], compiled.registry), compiled.protocol_hash


def test_p02_group_publish_is_complete_and_immutable(tmp_path: Path) -> None:
    registry, protocol_hash = _registry()
    result = _result()
    result.resolution_report["protocol_hash"] = protocol_hash
    publish_p02_outputs(
        run_root=tmp_path,
        registry=registry,
        protocol_hash=protocol_hash,
        recovery_policy="quarantine_incomplete",
        result=result,
    )
    expected = [
        tmp_path / "P02" / "firm_master.parquet",
        tmp_path / "P02" / "firm_year_panel.parquet",
        tmp_path / "P02" / "duplicate_map.json",
    ]
    assert all(path.is_file() for path in expected)
    assert all(path.with_name(path.name + ".manifest.json").is_file() for path in expected)
    with pytest.raises(ConfigurationError, match="immutable"):
        publish_p02_outputs(
            run_root=tmp_path,
            registry=registry,
            protocol_hash=protocol_hash,
            recovery_policy="quarantine_incomplete",
            result=result,
        )


def test_incomplete_p02_directory_is_quarantined(tmp_path: Path) -> None:
    registry, protocol_hash = _registry()
    incomplete = tmp_path / "P02"
    incomplete.mkdir()
    (incomplete / "firm_master.parquet").write_bytes(b"partial")
    result = _result()
    result.resolution_report["protocol_hash"] = protocol_hash
    publish_p02_outputs(
        run_root=tmp_path,
        registry=registry,
        protocol_hash=protocol_hash,
        recovery_policy="quarantine_incomplete",
        result=result,
    )
    assert (tmp_path / "P02" / "firm_year_panel.parquet").is_file()
    quarantines = list(tmp_path.glob("P02.quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "firm_master.parquet").read_bytes() == b"partial"
