from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.artifact_store import ArtifactStore
from core.errors import ConfigurationError
from core.registry_compiler import compile_registry
from core.schema_registry import contract_registry

ROOT = Path(__file__).resolve().parents[2]


def store(tmp_path: Path) -> ArtifactStore:
    compiled = compile_registry(ROOT / "config" / "pipeline.yaml")
    registry = dict(compiled.registry)
    return ArtifactStore(
        tmp_path,
        registry["artifacts"],
        contract_registry(registry),
        compiled.protocol_hash,
    )


def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_master_id": pd.Series(["A"], dtype="string"),
            "fiscal_year": pd.Series([2021], dtype="int16"),
            "outer_fold": pd.Series(["2021"], dtype="string"),
            "learner_id": pd.Series(["elastic_net_logistic"], dtype="string"),
            "measurement_id": pd.Series(["L1"], dtype="string"),
            "target_id": pd.Series(["L1"], dtype="string"),
            "prediction": pd.Series([0.25], dtype="float64"),
        }
    )


def test_parquet_round_trip_and_manifest(tmp_path: Path) -> None:
    artifact_store = store(tmp_path)
    coordinates = {"outer_fold": "2021"}
    artifact_store.write("raw_outer_predictions", predictions(), coordinates, "P11")
    observed = artifact_store.read("raw_outer_predictions", coordinates)
    assert isinstance(observed, pd.DataFrame)
    assert abs(float(observed["prediction"].iloc[0]) - 0.25) < 1e-12


def test_missing_manifest_fails_closed(tmp_path: Path) -> None:
    artifact_store = store(tmp_path)
    coordinates = {"outer_fold": "2021"}
    artifact_store.write("raw_outer_predictions", predictions(), coordinates, "P11")
    target = artifact_store.path("raw_outer_predictions", coordinates)
    target.with_name(target.name + ".manifest.json").unlink()
    with pytest.raises(ConfigurationError):
        artifact_store.read("raw_outer_predictions", coordinates)


def test_wrong_producer_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        store(tmp_path).write(
            "raw_outer_predictions",
            predictions(),
            {"outer_fold": "2021"},
            "P12",
        )


def test_probability_outside_unit_interval_fails(
    tmp_path: Path,
) -> None:
    frame = predictions()
    frame.loc[0, "prediction"] = 1.5
    with pytest.raises(Exception):
        store(tmp_path).write(
            "raw_outer_predictions",
            frame,
            {"outer_fold": "2021"},
            "P11",
        )
