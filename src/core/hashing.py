"""Canonical serialization and deterministic protocol hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast


def normalize(value: object) -> object:
    """Recursively sort mappings while preserving semantically ordered lists."""
    if isinstance(value, Mapping):
        mapping = cast(Mapping[Any, Any], value)
        return {str(key): normalize(mapping[key]) for key in sorted(mapping, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        sequence = cast(Sequence[Any], value)
        return [normalize(item) for item in sequence]
    return value


def canonical_json(value: object) -> str:
    """Return stable JSON suitable for a cryptographic protocol hash."""
    return json.dumps(normalize(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def protocol_hash(value: object) -> str:
    """Hash semantic protocol content with SHA-256.

    Snapshot IDs, creation timestamps, and the detached integrity hash identify an
    operational capture, not a methodological/data-content change.  The source file,
    schema, and mapping hashes remain in the semantic view through ``sources`` and
    ``snapshot_content_hash``.
    """
    return hashlib.sha256(
        canonical_json(_semantic_protocol_view(value)).encode("utf-8")
    ).hexdigest()


def _semantic_protocol_view(value: object) -> object:
    normalized = normalize(value)
    if not isinstance(normalized, dict):
        return normalized
    result = dict(cast(dict[str, object], normalized))
    snapshot = result.get("data_snapshot")
    if isinstance(snapshot, dict):
        semantic_snapshot = dict(cast(dict[str, object], snapshot))
        for key in ("snapshot_id", "created_at_utc", "snapshot_hash"):
            semantic_snapshot.pop(key, None)
        result["data_snapshot"] = semantic_snapshot
    data_sources = result.get("data_sources")
    if isinstance(data_sources, dict):
        semantic_sources = dict(cast(dict[str, object], data_sources))
        source_registry = semantic_sources.get("source_registry")
        if isinstance(source_registry, dict):
            registry_view = dict(cast(dict[str, object], source_registry))
            for key in ("snapshot_id", "snapshot_hash"):
                registry_view.pop(key, None)
            semantic_sources["source_registry"] = registry_view
        result["data_sources"] = semantic_sources
    return result


def file_hash(path_bytes: bytes) -> str:
    """Hash source bytes for provenance (not protocol semantics)."""
    return hashlib.sha256(path_bytes).hexdigest()
