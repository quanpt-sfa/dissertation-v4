"""Strict modular YAML loading; there is no include or merge mechanism."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from .config_model import Manifest
from .errors import (
    ConfigurationError,
    DuplicateOwnershipError,
    MissingModuleError,
    UndeclaredConfigurationFileError,
)
from .hashing import file_hash


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigurationError(f"{context}: expected a mapping; correct the YAML structure")
    return cast(dict[str, Any], value)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"source={path}: invalid YAML; correct syntax") from exc
    return _mapping(loaded if loaded is not None else {}, f"source={path}")


def load_manifest(path: Path) -> Manifest:
    """Load and validate the intentionally small entry-point manifest."""
    raw = _read_yaml(path)
    permitted = {"config_format_version", "modules", "schema_modules"}
    extras = set(raw) - permitted
    if extras or raw.get("config_format_version") != 1:
        raise ConfigurationError(
            f"source={path}: manifest may contain only {sorted(permitted)} and version 1; remove {sorted(extras)}"
        )
    modules = _mapping(raw.get("modules"), f"source={path}, key=modules")
    if not modules or not all(isinstance(value, str) for value in modules.values()):
        raise ConfigurationError(
            f"source={path}, key=modules: non-empty ID-to-path mapping required"
        )
    module_paths = list(cast(dict[str, str], modules).values())
    if len(module_paths) != len(set(module_paths)):
        raise DuplicateOwnershipError(
            f"source={path}: duplicate module path; each namespace needs one owner"
        )
    schemas_raw = raw.get("schema_modules", [])
    if not isinstance(schemas_raw, list) or not all(isinstance(item, str) for item in schemas_raw):
        raise ConfigurationError(f"source={path}, key=schema_modules: list of paths required")
    return Manifest(
        path=path, modules=cast(dict[str, str], modules), schema_modules=tuple(schemas_raw)
    )


def load_modules(manifest: Manifest) -> tuple[dict[str, Any], dict[str, str]]:
    """Load every declared module into its own namespace and reject stray YAML."""
    root = manifest.path.parent.resolve()
    declared = {manifest.path.resolve()}
    registry: dict[str, Any] = {}
    hashes: dict[str, str] = capture_hash({}, root, manifest.path.resolve())
    for module_id, relative in manifest.modules.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ConfigurationError(
                f"module={module_id}, source={relative}: path escapes config root"
            )
        if not candidate.is_file():
            raise MissingModuleError(
                f"module={module_id}, source={candidate}: create the declared module"
            )
        data = _read_yaml(candidate)
        if set(data) != {module_id}:
            raise DuplicateOwnershipError(
                f"module={module_id}, source={candidate}: module must contain exactly its owned namespace"
            )
        registry[module_id] = data[module_id]
        declared.add(candidate)
        hashes = capture_hash(hashes, root, candidate)
    schemas: dict[str, object] = {}
    for relative in manifest.schema_modules:
        candidate = (root / relative).resolve()
        if not candidate.is_file():
            raise MissingModuleError(
                f"schema source={candidate}: create the declared schema module"
            )
        data = _read_yaml(candidate)
        if set(data) != {"schemas"}:
            raise DuplicateOwnershipError(
                f"source={candidate}: schema modules own only the schemas namespace"
            )
        section = _mapping(data["schemas"], f"source={candidate}, key=schemas")
        overlap = set(schemas) & set(section)
        if overlap:
            raise DuplicateOwnershipError(
                f"source={candidate}: schema IDs already owned: {sorted(overlap)}"
            )
        schemas.update(section)
        declared.add(candidate)
        hashes = capture_hash(hashes, root, candidate)
    registry["schemas"] = schemas
    actual = {item.resolve() for item in root.rglob("*.yaml")}
    undeclared = actual - declared
    if undeclared:
        raise UndeclaredConfigurationFileError(
            f"source={root}: undeclared YAML files {sorted(str(item.relative_to(root)) for item in undeclared)}; list them in pipeline.yaml"
        )
    return registry, hashes


def capture_hash(hashes: dict[str, str], root: Path, path: Path) -> dict[str, str]:
    """Record portable source path provenance."""
    hashes[path.relative_to(root).as_posix()] = file_hash(path.read_bytes())
    return hashes
