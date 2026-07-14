from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from core.artifact_store import ArtifactStore
from core.errors import AccessPolicyError
from core.registry_compiler import compile_registry
from core.runtime import RunContext
from core.schema_registry import contract_registry

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def foundations(
    tmp_path: Path,
) -> tuple[dict[str, Any], ArtifactStore, str]:
    compiled = compile_registry(ROOT / "config" / "pipeline.yaml")
    registry = cast(dict[str, Any], compiled.registry)
    store = ArtifactStore(
        tmp_path,
        registry["artifacts"],
        contract_registry(registry),
        compiled.protocol_hash,
    )
    return registry, store, compiled.protocol_hash


def test_p10_cannot_read_sealed_outer_outcomes(
    foundations: tuple[dict[str, Any], ArtifactStore, str],
) -> None:
    registry, store, protocol_hash = foundations
    context = RunContext(
        "P10",
        "SPLIT",
        protocol_hash,
        registry,
        store,
        {"outer_fold": "2021"},
    )
    with pytest.raises(AccessPolicyError):
        context.read("sealed_outcome_store", {})


def test_p12_requires_freeze_receipt(
    foundations: tuple[dict[str, Any], ArtifactStore, str],
) -> None:
    registry, store, protocol_hash = foundations
    context = RunContext(
        "P12",
        "FROZEN",
        protocol_hash,
        registry,
        store,
        {"outer_fold": "2021"},
    )
    with pytest.raises(Exception):
        context.read("sealed_outcome_store", {})


def test_p12_cannot_write_raw_outer_predictions(
    foundations: tuple[dict[str, Any], ArtifactStore, str],
) -> None:
    registry, store, protocol_hash = foundations
    context = RunContext(
        "P12",
        "FROZEN",
        protocol_hash,
        registry,
        store,
        {"outer_fold": "2021"},
    )
    with pytest.raises(AccessPolicyError):
        context.write(
            "raw_outer_predictions",
            pd.DataFrame(),
            {"outer_fold": "2021"},
        )


def test_p11_write_is_revoked_after_outer_open(
    foundations: tuple[dict[str, Any], ArtifactStore, str],
) -> None:
    registry, store, protocol_hash = foundations
    context = RunContext(
        "P11",
        "OUTER_OPEN",
        protocol_hash,
        registry,
        store,
        {"outer_fold": "2021"},
    )
    with pytest.raises(AccessPolicyError):
        context.write("model_artifacts", {}, {"outer_fold": "2021"})
