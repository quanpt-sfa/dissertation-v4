from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from core.artifact_store import ArtifactStore
from core.atomic_io import publish_p00
from core.errors import ConfigurationError
from core.registry_compiler import compile_registry, environment_observation, source_manifest
from core.runtime import RunContext
from core.schema_registry import contract_registry

ROOT = Path(__file__).resolve().parents[2]


def _published_p00(tmp_path: Path) -> tuple[ArtifactStore, dict[str, Any]]:
    compiled = compile_registry(ROOT / "config" / "pipeline.yaml")
    registry = dict(compiled.registry)
    contracts = contract_registry(registry)
    vocabulary = cast(dict[str, Any], registry["vocabulary"])
    seal_states = cast(dict[str, Any], vocabulary["seal_states"])
    files: dict[str, object] = {
        "source_config_manifest.json": source_manifest(compiled, ROOT),
        "environment_observation.json": environment_observation(ROOT),
        "known_cases_seal.json": {
            "status": seal_states["sealed"],
            "protocol_hash": compiled.protocol_hash,
            "opens_at_step": "P15",
        },
        "job_manifest.json": {
            "run_id": "detached-runtime-test",
            "protocol_hash": compiled.protocol_hash,
            "output_hashes": {},
        },
    }
    artifact_ids = {
        "source_config_manifest.json": "source_config_manifest",
        "environment_observation.json": "environment_observation",
        "known_cases_seal.json": "known_cases_seal",
        "job_manifest.json": "job_manifest",
    }

    def validate(path: str, value: object) -> None:
        artifact_id = "success_receipt" if path == "_SUCCESS.json" else artifact_ids[path]
        contracts.validate(artifact_id, value)

    publication_statuses = cast(dict[str, Any], vocabulary["publication_statuses"])
    publish_p00(
        tmp_path / "P00",
        files,
        validate,
        {
            "status": publication_statuses["success"],
            "protocol_hash": compiled.protocol_hash,
        },
    )
    store = ArtifactStore(
        tmp_path,
        cast(dict[str, object], registry["artifacts"]),
        contracts,
        compiled.protocol_hash,
    )
    return store, registry


def _p11_context(store: ArtifactStore, registry: dict[str, Any]) -> RunContext:
    return RunContext(
        "P11",
        "SELECTED",
        store.protocol_hash,
        registry,
        store,
        {"outer_fold": "2024"},
    )


def _p15_context(store: ArtifactStore, registry: dict[str, Any]) -> RunContext:
    return RunContext(
        "P15",
        "GATE2",
        store.protocol_hash,
        registry,
        store,
        {},
    )


def _write_gate2_receipt(store: ArtifactStore) -> None:
    store.write(
        "gate2_verdict",
        {
            "status": "INSUFFICIENT_EVIDENCE",
            "protocol_hash": store.protocol_hash,
            "gate_id": "GATE2",
            "verdict": "INSUFFICIENT_EVIDENCE",
        },
        {},
        "P14",
    )


def test_runtime_reads_p00_artifacts_under_detached_manifest(tmp_path: Path) -> None:
    store, registry = _published_p00(tmp_path)
    context = _p11_context(store, registry)

    source = context.read("source_config_manifest", {})
    environment = context.read("environment_observation", {})

    assert isinstance(source, dict)
    assert isinstance(environment, dict)
    dependencies = {str(item["artifact_id"]): item for item in context.dependency_records()}
    assert dependencies["source_config_manifest"]["producer_step"] == "P00"
    assert dependencies["environment_observation"]["protocol_hash"] == store.protocol_hash


def test_required_p00_receipt_uses_detached_manifest(tmp_path: Path) -> None:
    store, registry = _published_p00(tmp_path)
    _write_gate2_receipt(store)
    seal_path = store.path("known_cases_seal", {})

    assert seal_path.is_file()
    assert not seal_path.with_name(seal_path.name + ".manifest.json").exists()

    gate2 = _p15_context(store, registry).read("gate2_verdict", {})

    assert isinstance(gate2, dict)
    assert gate2["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_required_p00_receipt_tampering_fails_closed(tmp_path: Path) -> None:
    store, registry = _published_p00(tmp_path)
    _write_gate2_receipt(store)
    store.path("known_cases_seal", {}).write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="P00 content hash mismatch"):
        _p15_context(store, registry).read("gate2_verdict", {})


def test_p00_artifact_tampering_fails_closed(tmp_path: Path) -> None:
    store, registry = _published_p00(tmp_path)
    target = store.path("source_config_manifest", {})
    target.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="P00 content hash mismatch"):
        _p11_context(store, registry).read("source_config_manifest", {})


def test_p00_job_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    store, registry = _published_p00(tmp_path)
    manifest_path = store.path("job_manifest", {})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="detached manifest hash mismatch"):
        _p11_context(store, registry).read("source_config_manifest", {})
