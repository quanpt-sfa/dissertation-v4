"""Stateful runtime enforcing access, receipts, sensitivity classes, and protocol hashes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .access_matrix import assert_access
from .artifact_store import ArtifactStore
from .errors import AccessPolicyError


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

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        self._access(artifact_id, "read")
        self._verify_required_receipts()
        return self.store.read(artifact_id, coordinates)

    def write(
        self, artifact_id: str, value: object, coordinates: Mapping[str, str]
    ) -> dict[str, object]:
        self._access(artifact_id, "write")
        self._verify_required_receipts()
        return self.store.write(artifact_id, value, coordinates, self.step_id)

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
        assert_access(matrix, artifacts, policy, self.step_id, artifact_id, mode, self.state)

    def _verify_required_receipts(self) -> None:
        matrix = self.registry.get("access_matrix")
        artifacts = self.registry.get("artifacts")
        if not isinstance(matrix, dict) or not isinstance(artifacts, dict):
            raise AccessPolicyError("runtime: access matrix unavailable")
        step = matrix.get(self.step_id)
        if not isinstance(step, dict):
            raise AccessPolicyError(f"step={self.step_id}: unknown")
        for receipt_id in step.get("required_receipts", []):
            receipt = artifacts.get(receipt_id)
            if not isinstance(receipt, dict):
                raise AccessPolicyError(f"receipt={receipt_id}: unknown")
            required_names = set(receipt.get("coordinates", []))
            coordinates = {
                name: self.unit_coordinates[name]
                for name in required_names
                if name in self.unit_coordinates
            }
            if set(coordinates) != required_names:
                raise AccessPolicyError(f"receipt={receipt_id}: coordinates unavailable")
            self.store.read(receipt_id, coordinates)
