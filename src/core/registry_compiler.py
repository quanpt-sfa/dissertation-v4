"""Compile the locked registry and produce complete non-semantic provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

from snapshot.injection import inject_snapshot

from .access_matrix import compile_access_matrix
from .config_loader import load_manifest, load_modules
from .config_model import CompiledRegistry
from .config_validation import validate_methodology, validate_references
from .decision_traceability import validate_decisions, validate_test_registry
from .hashing import protocol_hash
from .l3_assurance import apply_l3_preregistered_assurance
from .l3_scenarios import locked_l3_scenario_registry
from .schema_registry import compile_schemas

COMPILER_VERSION = "0.2.0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compile_registry(config_path: Path, snapshot_path: Path | None = None) -> CompiledRegistry:
    manifest = load_manifest(config_path)
    registry, source_hashes = load_modules(manifest)
    if snapshot_path is not None:
        registry = inject_snapshot(registry, snapshot_path)
        source_hashes["external/data_snapshot.json"] = _file_sha256(snapshot_path) or ""
    apply_l3_preregistered_assurance(registry)
    registry["schemas"] = dict(
        compile_schemas(registry.get("schemas"), registry.get("columns")).schemas
    )
    validate_references(registry)
    validate_methodology(registry)
    locked_l3_scenario_registry(registry)
    registry["decision_traceability"] = validate_decisions(registry)
    validate_test_registry(registry, PROJECT_ROOT)
    registry["access_matrix"] = compile_access_matrix(
        registry["steps"], registry["artifacts"], registry["access_control"]
    )
    registry["compiler_version"] = COMPILER_VERSION
    return CompiledRegistry(
        registry=registry,
        protocol_hash=protocol_hash(registry),
        source_hashes=source_hashes,
    )


def source_manifest(compiled: CompiledRegistry, root: Path) -> dict[str, object]:
    return {
        "source_hashes": compiled.source_hashes,
        "source_code_hashes": _tree_hashes(root, ("src/**/*.py", "scripts/*.py")),
        "generated_document_hashes": _tree_hashes(root, ("AGENTS.md", "docs/generated/*.md")),
        "compiler_version": COMPILER_VERSION,
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git(root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(root, ["status", "--porcelain"])),
        "git_diff_hash": _git_diff_hash(root),
        "untracked_hashes": _untracked_hashes(root),
        "package_lock": {
            "path": "uv.lock",
            "sha256": _file_sha256(root / "uv.lock"),
        },
        "environment_observation": environment_observation(root),
    }


def environment_observation(root: Path) -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for name in ("pandas", "pandera", "pyarrow", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "compiler_version": COMPILER_VERSION,
        "hash_algorithm": "sha256",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "uv_lock_hash": _file_sha256(root / "uv.lock"),
        "packages": packages,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "container": {
            "available": False,
            "reason": "NO_CONTAINER_METADATA_PROVIDED",
        },
    }


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("chapter3-protocol-lock")
    except importlib.metadata.PackageNotFoundError:
        return None


def _git(root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_diff_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--binary"],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _untracked_hashes(root: Path) -> dict[str, str]:
    output = _git(root, ["ls-files", "--others", "--exclude-standard"])
    if not output:
        return {}
    result: dict[str, str] = {}
    for relative in output.splitlines():
        path = root / relative
        if path.is_file():
            result[Path(relative).as_posix()] = _file_sha256(path) or ""
    return result


def _tree_hashes(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return {path.relative_to(root).as_posix(): _file_sha256(path) or "" for path in sorted(paths)}


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
