"""Minimal stateful runtime foundation; no scientific computation."""

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
    ) -> None:
        self.step_id, self.state, self.protocol_hash, self.registry, self.store = (
            step_id,
            state,
            protocol_hash,
            registry,
            store,
        )

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        self._access(artifact_id, "read")
        return self.store.read(artifact_id, coordinates)

    def write(self, artifact_id: str, value: object, coordinates: Mapping[str, str]) -> None:
        self._access(artifact_id, "write")
        self.store.write(artifact_id, value, coordinates, self.step_id)

    def _access(self, artifact_id: str, mode: str) -> None:
        matrix = self.registry.get("access_matrix")
        if not isinstance(matrix, dict):
            raise AccessPolicyError("runtime: access matrix unavailable")
        assert_access(matrix, self.step_id, artifact_id, mode, self.state)
