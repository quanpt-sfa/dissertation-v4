"""Inject an external immutable snapshot into the compiled protocol registry."""

from __future__ import annotations

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
    root_env = snapshot.get("root_environment_variable")
    if not isinstance(root_env, str):
        raise ValueError("snapshot.root_environment_variable required")
    data_sources["source_registry"] = {
        "root_environment_variable": root_env,
        "schema_evolution_policy": "fail_on_unregistered_fields",
        "hash_policy": "locked_sha256_required",
        "snapshot_id": snapshot.get("snapshot_id"),
        "snapshot_hash": snapshot.get("snapshot_hash"),
        "sources": source_registry,
    }
    registry["data_snapshot"] = snapshot
    return registry
