"""Catalog-only artifact paths and contract-checked persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .errors import ConfigurationError
from .schema_registry import ContractRegistry


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        artifacts: Mapping[str, object],
        contracts: ContractRegistry,
        protocol_hash: str,
    ) -> None:
        self.root = root.resolve()
        self.artifacts = artifacts
        self.contracts = contracts
        self.protocol_hash = protocol_hash

    def path(self, artifact_id: str, coordinates: Mapping[str, str]) -> Path:
        item = self._artifact(artifact_id)
        expected = item.get("coordinates")
        if not isinstance(expected, list) or set(coordinates) != set(expected):
            raise ConfigurationError(f"artifact={artifact_id}: coordinates do not match catalog")
        relative = Path(str(item["path_template"]).format(**coordinates))
        result = (self.root / relative).resolve()
        if self.root not in result.parents:
            raise ConfigurationError(f"artifact={artifact_id}: path escapes run root")
        return result

    def write(
        self,
        artifact_id: str,
        value: object,
        coordinates: Mapping[str, str],
        producer_step: str,
    ) -> dict[str, object]:
        self.contracts.validate(artifact_id, value)
        item = self._artifact(artifact_id)
        target = self.path(artifact_id, coordinates)
        if target.exists() and item.get("immutability") == "immutable":
            raise ConfigurationError(f"artifact={artifact_id}: immutable artifact already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._temporary_path(target)
        try:
            self._write_format(temp_path, str(item["format"]), value)
            observed = self._read_format(temp_path, str(item["format"]))
            self.contracts.validate(artifact_id, observed)
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        manifest = self._manifest(artifact_id, item, target, coordinates, producer_step)
        manifest_path = target.with_suffix(target.suffix + ".manifest.json")
        self._write_json_atomic(manifest_path, manifest)
        return manifest

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        item = self._artifact(artifact_id)
        target = self.path(artifact_id, coordinates)
        manifest_path = target.with_suffix(target.suffix + ".manifest.json")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("protocol_hash") != self.protocol_hash:
                raise ConfigurationError(f"artifact={artifact_id}: protocol hash mismatch")
            if manifest.get("content_hash") != _hash_file(target):
                raise ConfigurationError(f"artifact={artifact_id}: content hash mismatch")
        value = self._read_format(target, str(item["format"]))
        self.contracts.validate(artifact_id, value)
        return value

    def _artifact(self, artifact_id: str) -> dict[str, object]:
        item = self.artifacts.get(artifact_id)
        if not isinstance(item, dict) or not isinstance(item.get("path_template"), str):
            raise ConfigurationError(f"artifact={artifact_id}: undeclared artifact")
        return item

    def _manifest(
        self,
        artifact_id: str,
        item: Mapping[str, object],
        target: Path,
        coordinates: Mapping[str, str],
        producer_step: str,
    ) -> dict[str, object]:
        return {
            "artifact_id": artifact_id,
            "producer_step": producer_step,
            "schema_id": item["schema_id"],
            "schema_version": self.contracts.schema_version(str(item["schema_id"])),
            "format": item["format"],
            "coordinates": dict(coordinates),
            "protocol_hash": self.protocol_hash,
            "content_hash": _hash_file(target),
        }

    @staticmethod
    def _temporary_path(target: Path) -> Path:
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        )
        handle.close()
        return Path(handle.name)

    @staticmethod
    def _write_format(path: Path, format_name: str, value: object) -> None:
        if format_name == "json":
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")
        elif format_name in {"text", "markdown"}:
            if not isinstance(value, str):
                raise ConfigurationError(f"format={format_name}: string value required")
            path.write_text(value, "utf-8")
        elif format_name == "parquet":
            if not isinstance(value, pd.DataFrame):
                raise ConfigurationError("format=parquet: pandas DataFrame required")
            value.to_parquet(path, index=False)
        else:
            raise ConfigurationError(f"format={format_name}: unsupported artifact format")

    @staticmethod
    def _read_format(path: Path, format_name: str) -> object:
        if format_name == "json":
            return json.loads(path.read_text("utf-8"))
        if format_name in {"text", "markdown"}:
            return path.read_text("utf-8")
        if format_name == "parquet":
            return pd.read_parquet(path)
        raise ConfigurationError(f"format={format_name}: unsupported artifact format")

    @staticmethod
    def _write_json_atomic(target: Path, value: Mapping[str, object]) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = ArtifactStore._temporary_path(target)
        try:
            temp_path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
