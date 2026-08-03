"""Verified reads for P00 artifacts published under a detached directory manifest."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from .artifact_store import ArtifactStore
from .errors import ConfigurationError


def read_p00_artifact(
    store: ArtifactStore,
    artifact_id: str,
    coordinates: Mapping[str, str],
) -> tuple[object, dict[str, object]]:
    """Read a P00 artifact after verifying the directory-level detached manifest."""
    item_raw = store.artifacts.get(artifact_id)
    if not isinstance(item_raw, dict):
        raise ConfigurationError(f"artifact={artifact_id}: undeclared artifact")
    item = cast(dict[str, Any], item_raw)
    if item.get("producer_step") != "P00":
        raise ConfigurationError(f"artifact={artifact_id}: detached P00 read is not permitted")

    target = store.path(artifact_id, coordinates)
    job_manifest_path = store.path("job_manifest", {})
    success_path = store.path("success_receipt", {})
    if not target.is_file():
        raise ConfigurationError(f"artifact={artifact_id}: P00 artifact is missing")
    if not job_manifest_path.is_file() or not success_path.is_file():
        raise ConfigurationError(
            f"artifact={artifact_id}: P00 job manifest and success receipt are required"
        )

    job_manifest = _json_object(job_manifest_path, "P00 job manifest")
    success = _json_object(success_path, "P00 success receipt")
    store.contracts.validate("job_manifest", job_manifest)
    store.contracts.validate("success_receipt", success)

    if job_manifest.get("protocol_hash") != store.protocol_hash:
        raise ConfigurationError("P00 job manifest protocol hash mismatch")
    if success.get("protocol_hash") != store.protocol_hash:
        raise ConfigurationError("P00 success receipt protocol hash mismatch")
    if success.get("manifest_hash") != _hash_file(job_manifest_path):
        raise ConfigurationError("P00 detached manifest hash mismatch")

    content_hash = _hash_file(target)
    output_hashes_raw = job_manifest.get("output_hashes")
    if not isinstance(output_hashes_raw, dict):
        raise ConfigurationError("P00 job manifest output_hashes must be an object")
    output_hashes = cast(dict[str, object], output_hashes_raw)
    relative = target.relative_to(job_manifest_path.parent)
    expected_raw = output_hashes.get(relative.as_posix())
    if expected_raw is None:
        expected_raw = output_hashes.get(str(relative))
    expected_hash = expected_raw if isinstance(expected_raw, str) else None
    if expected_hash != content_hash:
        raise ConfigurationError(f"artifact={artifact_id}: P00 content hash mismatch")

    value = _read_format(target, str(item.get("format")))
    store.contracts.validate(artifact_id, value)
    manifest: dict[str, object] = {
        "artifact_id": artifact_id,
        "producer_step": "P00",
        "schema_id": item["schema_id"],
        "schema_version": store.contracts.schema_version(str(item["schema_id"])),
        "format": item["format"],
        "coordinates": dict(coordinates),
        "protocol_hash": store.protocol_hash,
        "content_hash": content_hash,
        "dependencies": [],
        "provenance_mode": "p00_detached_directory_manifest",
    }
    return value, manifest


def _json_object(path: Path, context: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{context}: object required")
    return cast(dict[str, Any], raw)


def _read_format(path: Path, format_name: str) -> object:
    if format_name == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if format_name in {"text", "markdown"}:
        return path.read_text(encoding="utf-8")
    raise ConfigurationError(
        f"artifact={path.name}: P00 detached reader does not support format={format_name}"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
