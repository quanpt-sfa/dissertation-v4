"""Orchestrate strict load, validation, normalization, and P0 outputs."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from .access_matrix import compile_access_matrix
from .config_loader import load_manifest, load_modules
from .config_model import CompiledRegistry
from .config_validation import validate_methodology, validate_references
from .decision_traceability import validate_decisions
from .hashing import protocol_hash
from .schema_registry import compile_schemas

COMPILER_VERSION = "0.1.0"


def compile_registry(config_path: Path) -> CompiledRegistry:
    """Compile a validated immutable normalized registry from the sole manifest."""
    manifest = load_manifest(config_path)
    registry, source_hashes = load_modules(manifest)
    columns = registry.get("columns")
    schemas = registry.get("schemas")
    registry["schemas"] = dict(compile_schemas(schemas, columns).schemas)
    validate_references(registry)
    validate_methodology(registry)
    traceability = validate_decisions(registry)
    registry["access_matrix"] = compile_access_matrix(registry["steps"])
    registry["compiler_version"] = COMPILER_VERSION
    registry["decision_traceability"] = traceability
    return CompiledRegistry(registry=registry, protocol_hash=protocol_hash(registry), source_hashes=source_hashes)


def source_manifest(compiled: CompiledRegistry, root: Path) -> dict[str, object]:
    """Build non-semantic provenance manifest without polluting protocol hash."""
    return {"source_hashes": compiled.source_hashes, "compiler_version": COMPILER_VERSION, "python_version": platform.python_version(), "git_commit": _git_commit(root)}


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
