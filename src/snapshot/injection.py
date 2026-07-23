"""Inject an external immutable snapshot into the compiled protocol registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from .builder import load_snapshot


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def inject_snapshot(registry: dict[str, object], snapshot_path: Path) -> dict[str, object]:
    snapshot = load_snapshot(snapshot_path)
    sources_raw = snapshot.get("sources")
    if not isinstance(sources_raw, list):
        raise ValueError("snapshot.sources must be a list")
    source_registry: dict[str, object] = {}
    for raw in cast(list[object], sources_raw):
        source = _mapping(raw, "snapshot.source")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("snapshot source_id required")
        if source_id in source_registry:
            raise ValueError(f"duplicate snapshot source_id={source_id}")
        source_registry[source_id] = source

    data_sources = _mapping(registry.get("data_sources"), "data_sources")
    protocol_source_registry = _mapping(
        data_sources.get("source_registry"), "data_sources.source_registry"
    )
    root_env = snapshot.get("root_environment_variable")
    if not isinstance(root_env, str):
        raise ValueError("snapshot.root_environment_variable required")

    injected_registry: dict[str, object] = {
        str(key): value for key, value in protocol_source_registry.items()
    }
    injected_registry.update(
        {
            "root_environment_variable": root_env,
            "schema_evolution_policy": "fail_on_unregistered_fields",
            "hash_policy": "locked_sha256_required",
            "snapshot_id": snapshot.get("snapshot_id"),
            "snapshot_hash": snapshot.get("snapshot_hash"),
            "snapshot_content_hash": snapshot.get("snapshot_content_hash")
            or _legacy_content_hash(snapshot),
            "extract_provenance": snapshot.get("extract_provenance", {}),
            "sources": source_registry,
        }
    )
    data_sources["source_registry"] = injected_registry
    registry["data_snapshot"] = snapshot
    return registry


def _legacy_content_hash(snapshot: dict[str, object]) -> str:
    """Derive a semantic content hash for schema-v1 manifests created before this field existed."""
    payload = {
        "snapshot_schema_version": snapshot.get("snapshot_schema_version"),
        "root_environment_variable": snapshot.get("root_environment_variable"),
        "raw_root_recorded": snapshot.get("raw_root_recorded"),
        "profiles": snapshot.get("profiles"),
        "sources": snapshot.get("sources"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
