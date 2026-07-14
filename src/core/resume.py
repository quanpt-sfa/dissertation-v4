"""Hash-verified helpers for resuming immutable pipeline runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path, context: str) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], raw)


def artifact_complete(
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifact_id: str,
    coordinates: dict[str, str],
) -> bool:
    raw_artifacts = registry.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("locked artifact catalog is unavailable")
    raw = cast(dict[str, object], raw_artifacts).get(artifact_id)
    if not isinstance(raw, dict):
        raise ValueError(f"artifact={artifact_id}: absent from locked catalog")
    artifact = cast(dict[str, object], raw)
    expected_coordinates = artifact.get("coordinates")
    if not isinstance(expected_coordinates, list) or set(coordinates) != {
        str(value) for value in cast(list[object], expected_coordinates)
    }:
        raise ValueError(f"artifact={artifact_id}: runner coordinates do not match catalog")
    template = artifact.get("path_template")
    if not isinstance(template, str):
        raise ValueError(f"artifact={artifact_id}: path template required")
    target = run_root / template.format(**coordinates)
    manifest_path = target.with_name(target.name + ".manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        return False
    manifest = json_object(manifest_path, f"artifact={artifact_id} manifest")
    expected = {
        "artifact_id": artifact_id,
        "producer_step": artifact.get("producer_step"),
        "coordinates": coordinates,
        "protocol_hash": protocol_hash,
    }
    return all(manifest.get(key) == value for key, value in expected.items()) and manifest.get(
        "content_hash"
    ) == hash_file(target)


def verify_resume_inputs(
    project_root: Path,
    raw_root: Path,
    snapshot: dict[str, object],
    run_root: Path,
) -> None:
    sources = snapshot.get("sources")
    if not isinstance(sources, list):
        raise ValueError("snapshot.sources must be a list")
    for raw in cast(list[object], sources):
        if not isinstance(raw, dict):
            raise ValueError("snapshot source entry must be an object")
        source = cast(dict[str, object], raw)
        relative = source.get("relative_path")
        expected = source.get("locked_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("snapshot source path/hash is incomplete")
        path = raw_root / relative
        if not path.is_file() or hash_file(path) != expected:
            raise RuntimeError(f"resume refused: raw source drifted or is missing: {relative}")
    manifest = json_object(run_root / "P00" / "source_config_manifest.json", "source manifest")
    hashes = manifest.get("source_code_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("source manifest code hashes are unavailable")
    for relative, expected in cast(dict[object, object], hashes).items():
        path = project_root / str(relative)
        if not path.is_file() or hash_file(path) != str(expected):
            raise RuntimeError(f"resume refused: implementation drifted: {relative}")
    config_hashes = manifest.get("source_hashes")
    if not isinstance(config_hashes, dict):
        raise ValueError("source manifest config hashes are unavailable")
    for relative, expected in cast(dict[object, object], config_hashes).items():
        path = (
            run_root / "SNAPSHOT" / "data_snapshot.json"
            if str(relative) == "external/data_snapshot.json"
            else project_root / "config" / str(relative)
        )
        if not path.is_file() or hash_file(path) != str(expected):
            raise RuntimeError(f"resume refused: locked config/snapshot drifted: {relative}")
