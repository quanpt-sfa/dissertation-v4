"""Fail-closed, catalog-only artifact persistence with Parquet support and manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .errors import ConfigurationError
from .path_policy import RelativePathError, resolve_relative_path
from .schema_registry import ContractRegistry


def dataframe_to_csv(value: pd.DataFrame) -> str:
    """Render a CSV sidecar through the approved core I/O boundary."""
    return value.to_csv(index=False)


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        artifacts: Mapping[str, object],
        contracts: ContractRegistry,
        protocol_hash: str,
        recovery_policy: str = "quarantine_incomplete",
    ) -> None:
        self.root = root.expanduser().resolve(strict=False)
        self.artifacts = artifacts
        self.contracts = contracts
        self.protocol_hash = protocol_hash
        self.recovery_policy = recovery_policy

    def path(self, artifact_id: str, coordinates: Mapping[str, str]) -> Path:
        item = self._artifact(artifact_id)
        expected = item.get("coordinates")
        if not isinstance(expected, list):
            raise ConfigurationError(
                f"artifact={artifact_id}: coordinates do not match catalog"
            )
        expected_coordinates = cast(list[object], expected)
        if set(coordinates) != {str(name) for name in expected_coordinates}:
            raise ConfigurationError(
                f"artifact={artifact_id}: coordinates do not match catalog"
            )
        try:
            rendered = str(item["path_template"]).format(**coordinates)
            return resolve_relative_path(
                self.root,
                rendered,
                context=f"artifact={artifact_id}",
            )
        except (KeyError, RelativePathError) as exc:
            raise ConfigurationError(
                f"artifact={artifact_id}: invalid relative path contract: {exc}"
            ) from exc

    def write(
        self,
        artifact_id: str,
        value: object,
        coordinates: Mapping[str, str],
        producer_step: str,
        dependencies: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        item = self._artifact(artifact_id)
        if item.get("producer_step") != producer_step:
            raise ConfigurationError(
                f"artifact={artifact_id}: producer {producer_step} does not match catalog"
            )
        self.contracts.validate(artifact_id, value)
        target = self.path(artifact_id, coordinates)
        manifest_path = self._manifest_path(target)
        if target.is_file() and manifest_path.is_file() and item.get("immutability") == "immutable":
            existing = self.read(artifact_id, coordinates)
            manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_raw, dict):
                raise ConfigurationError(f"artifact={artifact_id}: manifest must be an object")
            manifest = cast(dict[str, Any], manifest_raw)
            expected_dependencies = _sorted_dependencies(dependencies or [])
            if (
                _values_equal(existing, value)
                and manifest.get("dependencies", []) == expected_dependencies
            ):
                return cast(dict[str, object], manifest)
            raise ConfigurationError(
                f"artifact={target}: immutable artifact already exists with different content or provenance"
            )
        self._recover_or_reject_existing(target, manifest_path, item)

        target.parent.mkdir(parents=True, exist_ok=True)
        stage_root = self.root / ".t"
        stage_root.mkdir(exist_ok=True)
        stage_dir = Path(tempfile.mkdtemp(prefix="s-", dir=stage_root))
        staged_artifact = stage_dir / target.name
        staged_manifest = stage_dir / manifest_path.name
        try:
            self._write_format(staged_artifact, str(item["format"]), value)
            persisted = self._read_format(staged_artifact, str(item["format"]))
            self.contracts.validate(artifact_id, persisted)
            manifest = self._manifest(
                artifact_id,
                item,
                staged_artifact,
                coordinates,
                producer_step,
                dependencies or [],
            )
            staged_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            _fsync_file(staged_artifact)
            _fsync_file(staged_manifest)
            _safe_replace(staged_artifact, target)
            _safe_replace(staged_manifest, manifest_path)
            _fsync_directory(target.parent)
            return manifest
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        item = self._artifact(artifact_id)
        target = self.path(artifact_id, coordinates)
        manifest_path = self._manifest_path(target)
        if not target.is_file() or not manifest_path.is_file():
            raise ConfigurationError(
                f"artifact={artifact_id}: artifact and manifest are both required"
            )
        manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest_raw, dict):
            raise ConfigurationError(f"artifact={artifact_id}: manifest must be an object")
        manifest = cast(dict[str, Any], manifest_raw)
        expected = {
            "artifact_id": artifact_id,
            "producer_step": item["producer_step"],
            "schema_id": item["schema_id"],
            "schema_version": self.contracts.schema_version(str(item["schema_id"])),
            "format": item["format"],
            "coordinates": dict(coordinates),
            "protocol_hash": self.protocol_hash,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ConfigurationError(f"artifact={artifact_id}: manifest {key} mismatch")
        if manifest.get("content_hash") != _hash_file(target):
            raise ConfigurationError(f"artifact={artifact_id}: content hash mismatch")
        value = self._read_format(target, str(item["format"]))
        self.contracts.validate(artifact_id, value)
        return value

    def receipt_hash(self, artifact_id: str, coordinates: Mapping[str, str]) -> str:
        value = self.read(artifact_id, coordinates)
        if not isinstance(value, dict):
            raise ConfigurationError(f"artifact={artifact_id}: receipt object required")
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def manifest(self, artifact_id: str, coordinates: Mapping[str, str]) -> dict[str, Any]:
        """Return a verified manifest for provenance chaining."""
        self.read(artifact_id, coordinates)
        target = self.path(artifact_id, coordinates)
        raw = json.loads(self._manifest_path(target).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError(f"artifact={artifact_id}: manifest must be an object")
        return cast(dict[str, Any], raw)

    def inventory(self) -> list[dict[str, object]]:
        """Return a verified manifest inventory without bypassing the core I/O layer."""
        rows: list[dict[str, object]] = []
        for manifest_path in sorted(self.root.rglob("*.manifest.json")):
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ConfigurationError(f"manifest={manifest_path}: object required")
            manifest = cast(dict[str, Any], raw)
            artifact_id = manifest.get("artifact_id")
            coordinates_raw = manifest.get("coordinates")
            if not isinstance(artifact_id, str) or not isinstance(coordinates_raw, dict):
                raise ConfigurationError(f"manifest={manifest_path}: identity is incomplete")
            coordinates = {
                str(key): str(value)
                for key, value in cast(dict[object, object], coordinates_raw).items()
            }
            target = self.path(artifact_id, coordinates)
            if self._manifest_path(target) != manifest_path.resolve():
                raise ConfigurationError(
                    f"manifest={manifest_path}: path does not match artifact catalog"
                )
            self.read(artifact_id, coordinates)
            rows.append(
                {
                    **manifest,
                    "relative_path": target.relative_to(self.root).as_posix(),
                    "manifest_relative_path": manifest_path.relative_to(self.root).as_posix(),
                }
            )
        return rows

    def _recover_or_reject_existing(
        self, target: Path, manifest_path: Path, item: Mapping[str, object]
    ) -> None:
        target_exists = target.exists()
        manifest_exists = manifest_path.exists()
        if target_exists and manifest_exists:
            if item.get("immutability") == "immutable":
                raise ConfigurationError(f"artifact={target}: immutable artifact already exists")
            return
        if not target_exists and not manifest_exists:
            return
        if self.recovery_policy != "quarantine_incomplete":
            raise ConfigurationError(f"artifact={target}: incomplete artifact-manifest pair")
        quarantine = target.parent / ".quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        for path in (target, manifest_path):
            if path.exists():
                destination = quarantine / f"{path.name}.{os.getpid()}"
                _safe_replace(path, destination)

    @staticmethod
    def _manifest_path(target: Path) -> Path:
        return target.with_name(target.name + ".manifest.json")

    def _artifact(self, artifact_id: str) -> dict[str, Any]:
        item = self.artifacts.get(artifact_id)
        if not isinstance(item, dict):
            raise ConfigurationError(f"artifact={artifact_id}: undeclared artifact")
        item_spec = cast(dict[str, Any], item)
        if not isinstance(item_spec.get("path_template"), str):
            raise ConfigurationError(f"artifact={artifact_id}: undeclared artifact")
        return item_spec

    def _manifest(
        self,
        artifact_id: str,
        item: Mapping[str, object],
        path: Path,
        coordinates: Mapping[str, str],
        producer_step: str,
        dependencies: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "artifact_id": artifact_id,
            "producer_step": producer_step,
            "schema_id": item["schema_id"],
            "schema_version": self.contracts.schema_version(str(item["schema_id"])),
            "format": item["format"],
            "coordinates": dict(coordinates),
            "protocol_hash": self.protocol_hash,
            "content_hash": _hash_file(path),
            "dependencies": _sorted_dependencies(dependencies),
        }

    @staticmethod
    def _write_format(path: Path, format_name: str, value: object) -> None:
        if format_name == "json":
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        elif format_name in {"text", "markdown"}:
            if not isinstance(value, str):
                raise ConfigurationError(f"format={format_name}: string required")
            path.write_text(value, encoding="utf-8")
        elif format_name == "parquet":
            if not isinstance(value, pd.DataFrame):
                raise ConfigurationError("format=parquet: pandas DataFrame required")
            value.to_parquet(path, index=False, engine="pyarrow")
        elif format_name == "csv":
            if not isinstance(value, pd.DataFrame):
                raise ConfigurationError("format=csv: pandas DataFrame required")
            value.to_csv(path, index=False)
        else:
            raise ConfigurationError(f"format={format_name}: unsupported")

    @staticmethod
    def _read_format(path: Path, format_name: str) -> object:
        if format_name == "json":
            return json.loads(path.read_text(encoding="utf-8"))
        if format_name in {"text", "markdown"}:
            return path.read_text(encoding="utf-8")
        if format_name == "parquet":
            return pd.read_parquet(path)
        if format_name == "csv":
            return pd.read_csv(path).convert_dtypes(dtype_backend="pyarrow")
        raise ConfigurationError(f"format={format_name}: unsupported")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_dependencies(dependencies: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        dependencies,
        key=lambda item: (
            str(item.get("artifact_id")),
            json.dumps(item.get("coordinates", {}), sort_keys=True),
        ),
    )


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
        return left.equals(right)
    if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
        return False
    return left == right


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _safe_replace(src: Path, dst: Path) -> None:
    import time

    for i in range(15):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == 14:
                raise
            time.sleep(0.1)
