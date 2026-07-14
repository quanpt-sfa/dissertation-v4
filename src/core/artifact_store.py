"""Catalog-only artifact paths and contract-checked persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .errors import ConfigurationError
from .schema_registry import ContractRegistry


class ArtifactStore:
    def __init__(
        self, root: Path, artifacts: Mapping[str, object], contracts: ContractRegistry
    ) -> None:
        self.root, self.artifacts, self.contracts = root, artifacts, contracts

    def path(self, artifact_id: str, coordinates: Mapping[str, str]) -> Path:
        item = self.artifacts.get(artifact_id)
        if not isinstance(item, dict) or not isinstance(item.get("path_template"), str):
            raise ConfigurationError(f"artifact={artifact_id}: undeclared artifact")
        expected = item.get("coordinates")
        if not isinstance(expected, list) or set(coordinates) != set(expected):
            raise ConfigurationError(f"artifact={artifact_id}: coordinates do not match catalog")
        relative = Path(item["path_template"].format(**coordinates))
        result = (self.root / relative).resolve()
        if self.root.resolve() not in result.parents:
            raise ConfigurationError(f"artifact={artifact_id}: path escapes run root")
        return result

    def write(self, artifact_id: str, value: object, coordinates: Mapping[str, str]) -> None:
        self.contracts.validate(artifact_id, value)
        target = self.path(artifact_id, coordinates)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
            encoding="utf-8",
        )

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object:
        target = self.path(artifact_id, coordinates)
        value: object = target.read_text(encoding="utf-8")
        item = self.artifacts[artifact_id]
        if isinstance(item, dict) and item.get("format") == "json":
            value = json.loads(str(value))
        self.contracts.validate(artifact_id, value)
        return value
