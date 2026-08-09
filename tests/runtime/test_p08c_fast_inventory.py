from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.artifact_store import ArtifactStore
from core.errors import ConfigurationError
from core.registry_compiler import compile_registry
from core.schema_registry import contract_registry

ROOT = Path(__file__).resolve().parents[2]


def _store(tmp_path: Path) -> ArtifactStore:
    compiled = compile_registry(ROOT / "config" / "pipeline.yaml")
    registry = dict(compiled.registry)
    return ArtifactStore(
        tmp_path,
        registry["artifacts"],
        contract_registry(registry),
        compiled.protocol_hash,
    )


def _predictions() -> pd.DataFrame:
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


def test_filtered_verified_inventory_skips_unselected_payloads(tmp_path: Path) -> None:
    artifact_store = _store(tmp_path)
    coordinates = {"outer_fold": "2021"}
    artifact_store.write("raw_outer_predictions", _predictions(), coordinates, "P11")
    artifact_store.write("availability_registry", [], {}, "P03")

    unrelated = artifact_store.path("availability_registry", {})
    unrelated.write_text("corrupted-but-unselected", encoding="utf-8")

    selected = list(artifact_store.iter_verified_inventory({"raw_outer_predictions"}))
    assert len(selected) == 1
    manifest, value = selected[0]
    assert manifest["artifact_id"] == "raw_outer_predictions"
    assert isinstance(value, pd.DataFrame)
    assert value.equals(_predictions())

    filtered_inventory = artifact_store.inventory({"raw_outer_predictions"})
    assert [item["artifact_id"] for item in filtered_inventory] == ["raw_outer_predictions"]

    with pytest.raises(ConfigurationError, match="content hash mismatch"):
        artifact_store.inventory()


def test_p08c_reuses_verified_batch_values_instead_of_reading_twice() -> None:
    source = (ROOT / "scripts" / "p08c_aggregate_batches.py").read_text(encoding="utf-8")
    assert "iter_verified_inventory(_P08_CONTROL_ARTIFACT_IDS)" in source
    assert 'context.read("simulation_batches"' not in source
    assert (
        '_P08_CONTROL_ARTIFACT_IDS = frozenset({"simulation_batches", "model_diagnostics"})'
        in source
    )
