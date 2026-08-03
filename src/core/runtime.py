"""Stateful runtime enforcing access, receipts, sensitivity classes, and protocol hashes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .access_matrix import assert_access
from .artifact_store import ArtifactStore
from .errors import AccessPolicyError
from .p00_artifacts import read_p00_artifact


class RunContext:
    def __init__(
        self,
        step_id: str,
        state: str,
        protocol_hash: str,
        registry: Mapping[str, Any],
        store: ArtifactStore,
        unit_coordinates: Mapping[str, str] | None = None,
    ) -> None:
        if protocol_hash != store.protocol_hash:
            raise AccessPolicyError("runtime: context and ArtifactStore protocol hashes differ")
        self.step_id = step_id
        self.state = state
        self.protocol_hash = protocol_hash
        self.registry = registry
        self.store = store
        self.unit_coordinates = dict(unit_coordinates or {})
        self._dependencies: dict[tuple[str, str], dict[str, object]] = {}

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        self._access(artifact_id, "read")
        self._verify_required_receipts()
        artifact_spec = self._artifact_spec(artifact_id)
        if artifact_spec.get("producer_step") == "P00":
            value, manifest = read_p00_artifact(self.store, artifact_id, coordinates)
        else:
            value = self.store.read(artifact_id, coordinates)
            manifest = self.store.manifest(artifact_id, coordinates)
        self._record_dependency(artifact_id, coordinates, manifest)
        return value

    def read_p00(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        """Read a P00 lock artifact under its detached directory-manifest contract."""
        self._access(artifact_id, "read")
        self._verify_required_receipts()
        value, manifest = read_p00_artifact(self.store, artifact_id, coordinates)
        self._record_dependency(artifact_id, coordinates, manifest)
        return value

    def write(
        self, artifact_id: str, value: object, coordinates: Mapping[str, str]
    ) -> dict[str, object]:
        self._access(artifact_id, "write")
        self._verify_required_receipts()
        return self.store.write(
            artifact_id,
            value,
            coordinates,
            self.step_id,
            dependencies=list(self._dependencies.values()),
        )

    def dependency_records(self) -> list[dict[str, object]]:
        """Export verified upstream identities for an atomic multi-artifact publisher."""
        return list(self._dependencies.values())

    def inherit_dependencies(self, dependencies: list[dict[str, object]]) -> None:
        """Seed provenance when computation and atomic publication use separate stores."""
        for dependency in dependencies:
            artifact_id = dependency.get("artifact_id")
            coordinates = dependency.get("coordinates")
            if not isinstance(artifact_id, str) or not isinstance(coordinates, dict):
                raise AccessPolicyError("runtime: inherited dependency identity is incomplete")
            key = (artifact_id, str(sorted(cast(dict[object, object], coordinates).items())))
            self._dependencies[key] = dependency

    def _record_dependency(
        self,
        artifact_id: str,
        coordinates: Mapping[str, str],
        manifest: Mapping[str, object],
    ) -> None:
        dependency: dict[str, object] = {
            "artifact_id": artifact_id,
            "coordinates": dict(coordinates),
            "content_hash": manifest["content_hash"],
            "producer_step": manifest["producer_step"],
            "protocol_hash": manifest["protocol_hash"],
        }
        key = (artifact_id, str(sorted(coordinates.items())))
        self._dependencies[key] = dependency

    def _artifact_spec(self, artifact_id: str) -> dict[str, Any]:
        artifacts = self.registry.get("artifacts")
        if not isinstance(artifacts, dict):
            raise AccessPolicyError("runtime: artifact catalog unavailable")
        item = cast(dict[str, Any], artifacts).get(artifact_id)
        if not isinstance(item, dict):
            raise AccessPolicyError(f"artifact={artifact_id}: unknown")
        return cast(dict[str, Any], item)

    def _access(self, artifact_id: str, mode: str) -> None:
        matrix = self.registry.get("access_matrix")
        artifacts = self.registry.get("artifacts")
        policy = self.registry.get("access_control")
        if (
            not isinstance(matrix, dict)
            or not isinstance(artifacts, dict)
            or not isinstance(policy, dict)
        ):
            raise AccessPolicyError("runtime: access foundations unavailable")
        assert_access(
            cast(dict[str, Any], matrix),
            cast(dict[str, Any], artifacts),
            cast(dict[str, Any], policy),
            self.step_id,
            artifact_id,
            mode,
            self.state,
        )

    def _verify_required_receipts(self) -> None:
        matrix = self.registry.get("access_matrix")
        artifacts = self.registry.get("artifacts")
        if not isinstance(matrix, dict) or not isinstance(artifacts, dict):
            raise AccessPolicyError("runtime: access matrix unavailable")
        matrix_map = cast(dict[str, Any], matrix)
        artifact_map = cast(dict[str, Any], artifacts)
        step = matrix_map.get(self.step_id)
        if not isinstance(step, dict):
            raise AccessPolicyError(f"step={self.step_id}: unknown")
        step_spec = cast(dict[str, Any], step)
        receipts = step_spec.get("required_receipts", [])
        if not isinstance(receipts, list):
            raise AccessPolicyError(f"step={self.step_id}: required_receipts must be a list")
        for receipt_id in cast(list[object], receipts):
            if not isinstance(receipt_id, str):
                raise AccessPolicyError(f"step={self.step_id}: receipt id must be a string")
            receipt = artifact_map.get(receipt_id)
            if not isinstance(receipt, dict):
                raise AccessPolicyError(f"receipt={receipt_id}: unknown")
            receipt_spec = cast(dict[str, Any], receipt)
            coordinate_names = receipt_spec.get("coordinates", [])
            if not isinstance(coordinate_names, list) or not all(
                isinstance(name, str) for name in cast(list[object], coordinate_names)
            ):
                raise AccessPolicyError(f"receipt={receipt_id}: coordinate names invalid")
            required_names = set(cast(list[str], coordinate_names))
            coordinates = {
                name: self.unit_coordinates[name]
                for name in required_names
                if name in self.unit_coordinates
            }
            if set(coordinates) != required_names:
                raise AccessPolicyError(f"receipt={receipt_id}: coordinates unavailable")
            self.store.read(receipt_id, coordinates)
