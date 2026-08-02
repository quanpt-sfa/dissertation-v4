"""Load the immutable P00 registry and resolve a registered raw source."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from core.hashing import protocol_hash

from .models import SourceSpec


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def load_locked_registry(registry_path: Path) -> tuple[dict[str, Any], str, Path]:
    registry_path = registry_path.resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(registry_path)
    raw: object = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = _mapping(raw, "registry.lock.json")
    computed_hash = protocol_hash(registry)
    protocol_path = registry_path.with_name("protocol_hash.txt")
    if not protocol_path.is_file():
        raise ValueError("protocol_hash.txt is required beside registry.lock.json")
    locked_hash = protocol_path.read_text(encoding="utf-8").strip()
    if locked_hash != computed_hash:
        raise ValueError("registry.lock.json does not match protocol_hash.txt")
    run_root = registry_path.parent.parent
    return registry, locked_hash, run_root


def resolve_source(
    registry: dict[str, Any],
    source_id: str,
) -> tuple[SourceSpec, Path]:
    data_sources = _mapping(registry.get("data_sources"), "data_sources")
    source_registry = _mapping(data_sources.get("source_registry"), "data_sources.source_registry")
    root_env = source_registry.get("root_environment_variable")
    if not isinstance(root_env, str) or not root_env:
        raise ValueError("data_sources.source_registry.root_environment_variable is required")
    policy = source_registry.get("schema_evolution_policy")
    if policy != "fail_on_unregistered_fields":
        raise ValueError("P01 requires schema_evolution_policy=fail_on_unregistered_fields")
    hash_policy = source_registry.get("hash_policy")
    if hash_policy != "locked_sha256_required":
        raise ValueError("P01 requires hash_policy=locked_sha256_required")
    sources = _mapping(source_registry.get("sources"), "data_sources.source_registry.sources")
    resolved_id = _resolve_compatibility_alias(source_id, sources)
    spec = SourceSpec.from_mapping(resolved_id, sources[resolved_id])
    if not spec.enabled:
        raise ValueError(f"source_id={resolved_id} is disabled")
    root_value = os.environ.get(root_env)
    if not root_value:
        raise OSError(f"environment variable {root_env} is not set")
    root = Path(root_value).expanduser().resolve()
    path = (root / spec.relative_path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"source_id={resolved_id}: relative_path escapes source root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return spec, path


def _resolve_compatibility_alias(source_id: str, sources: dict[str, Any]) -> str:
    """Map the retired long-file feature source to the final S1 semantic view."""

    if source_id in sources:
        return source_id
    if source_id != "financial_statement_core_long":
        raise KeyError(f"source_id={source_id} is not registered in the locked protocol")
    matches: list[str] = []
    for candidate_id, raw in sources.items():
        candidate = _mapping(raw, f"source={candidate_id}")
        evidence = candidate.get("evidence_mapping")
        if not isinstance(evidence, dict):
            continue
        if cast(dict[str, object], evidence).get("processor") == "audit_adjustment":
            matches.append(str(candidate_id))
    if len(matches) != 1:
        raise KeyError(
            "legacy feature source requires exactly one registered audit-adjustment final view"
        )
    return matches[0]
