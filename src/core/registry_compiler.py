"""Orchestrate strict load, validation, normalization, and P0 outputs."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

from .access_matrix import compile_access_matrix
from .config_loader import load_manifest, load_modules
from .config_model import CompiledRegistry
from .config_validation import validate_methodology, validate_references
from .decision_traceability import validate_decisions, validate_test_registry
from .hashing import protocol_hash
from .schema_registry import compile_schemas

COMPILER_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    validate_test_registry(registry, PROJECT_ROOT)
    registry["access_matrix"] = compile_access_matrix(registry["steps"])
    registry["compiler_version"] = COMPILER_VERSION
    registry["decision_traceability"] = traceability
    return CompiledRegistry(
        registry=registry, protocol_hash=protocol_hash(registry), source_hashes=source_hashes
    )


def source_manifest(compiled: CompiledRegistry, root: Path) -> dict[str, object]:
    """Build non-semantic provenance manifest without polluting protocol hash."""
    package_lock = root / "uv.lock"
    return {
        "source_hashes": compiled.source_hashes,
        "compiler_version": COMPILER_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(root),
        "git_dirty": _git_dirty(root),
        "git_diff_hash": _git_diff_hash(root),
        "package_lock": {
            "path": "uv.lock",
            "sha256": _file_sha256(package_lock),
        },
        "environment_observation": environment_observation(root),
    }


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_dirty(root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _git_diff_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=root,
            text=False,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    import hashlib

    return hashlib.sha256(result.stdout).hexdigest()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_observation(root: Path) -> dict[str, object]:
    return {
        "compiler_version": COMPILER_VERSION,
        "hash_algorithm": "sha256",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "uv_lock_hash": _file_sha256(root / "uv.lock"),
    }
