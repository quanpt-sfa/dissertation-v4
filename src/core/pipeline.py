"""Shared post-P0 runtime construction and registry-only lookup helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from p01.registry import load_locked_registry

from .artifact_store import ArtifactStore
from .runtime import RunContext
from .schema_registry import contract_registry


@dataclass(frozen=True)
class LoadedRun:
    registry: dict[str, Any]
    protocol_hash: str
    run_root: Path
    context: RunContext


def load_run(
    *,
    registry_path: Path,
    run_id: str,
    step_id: str,
    state: str,
    coordinates: Mapping[str, str] | None = None,
) -> LoadedRun:
    """Open a locked run and bind a catalog-enforced step context."""
    registry, locked_hash, run_root = load_locked_registry(registry_path)
    if run_root.name != run_id:
        raise ValueError(f"run-id={run_id} does not match registry run directory {run_root.name}")
    artifacts = mapping(registry.get("artifacts"), "artifacts")
    reproducibility = mapping(registry.get("reproducibility"), "reproducibility")
    recovery = reproducibility.get("recovery_policy", "quarantine_incomplete")
    if not isinstance(recovery, str):
        raise ValueError("reproducibility.recovery_policy must be a string")
    store = ArtifactStore(
        run_root,
        artifacts,
        contract_registry(registry),
        locked_hash,
        recovery_policy=recovery,
    )
    context = RunContext(
        step_id,
        state,
        locked_hash,
        registry,
        store,
        coordinates,
    )
    return LoadedRun(registry, locked_hash, run_root, context)


def mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def sequence(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: list required")
    return cast(list[Any], value)


def physical_columns(registry: Mapping[str, Any]) -> dict[str, str]:
    raw = mapping(registry.get("columns"), "columns")
    result: dict[str, str] = {}
    for logical, value in raw.items():
        entry = mapping(value, f"columns.{logical}")
        physical = entry.get("physical_name")
        if not isinstance(physical, str) or not physical:
            raise ValueError(f"columns.{logical}.physical_name required")
        result[logical] = physical
    return result


def artifact_exists(context: RunContext, artifact_id: str, coordinates: Mapping[str, str]) -> bool:
    target = context.store.path(artifact_id, coordinates)
    manifest = target.with_name(target.name + ".manifest.json")
    return target.is_file() and manifest.is_file()


def read_optional(
    context: RunContext, artifact_id: str, coordinates: Mapping[str, str]
) -> object | None:
    if not artifact_exists(context, artifact_id, coordinates):
        return None
    return context.read(artifact_id, coordinates)


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def registered_value(registry: Mapping[str, Any], dotted_path: str) -> object:
    current: object = registry
    for part in dotted_path.split("."):
        current = mapping(current, dotted_path).get(part)
    return current


def outer_fold_ids(registry: Mapping[str, Any], *, include_initial: bool = True) -> list[str]:
    folds = mapping(registry.get("folds"), "folds")
    values: list[int] = []
    if include_initial:
        initial = folds.get("initial_outer_year")
        if not isinstance(initial, int):
            raise ValueError("folds.initial_outer_year must be an integer")
        values.append(initial)
    nested = sequence(folds.get("fully_nested_outer_years"), "folds.fully_nested_outer_years")
    if not all(isinstance(item, int) for item in nested):
        raise ValueError("folds.fully_nested_outer_years must contain integers")
    values.extend(cast(list[int], nested))
    return [str(value) for value in values]
