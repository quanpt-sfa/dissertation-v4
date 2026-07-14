"""Canonical serialization and deterministic protocol hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence


def normalize(value: object) -> object:
    """Recursively sort mappings while preserving semantically ordered lists."""
    if isinstance(value, Mapping):
        return {str(key): normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Return stable JSON suitable for a cryptographic protocol hash."""
    return json.dumps(normalize(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def protocol_hash(value: object) -> str:
    """Hash canonical protocol content with SHA-256."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path_bytes: bytes) -> str:
    """Hash source bytes for provenance (not protocol semantics)."""
    return hashlib.sha256(path_bytes).hexdigest()
