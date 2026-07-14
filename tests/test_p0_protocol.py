"""Focused P0 regression tests using the complete, data-free protocol fixture."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core.config_loader import load_manifest
from core.errors import (
    ArtifactPathCollisionError,
    DecisionTraceabilityError,
    DuplicateOwnershipError,
    GeneratedFileDriftError,
    MethodologicalInvariantError,
    MissingModuleError,
    ConfigurationError,
)
from core.generated_docs import render_documents, write_or_check_documents
from core.forbidden_patterns import validate_source_patterns
from core.registry_compiler import compile_registry


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def configuration(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    shutil.copytree(ROOT / "config", destination)
    return destination / "pipeline.yaml"


def replace(path: Path, old: str, new: str) -> None:
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def test_valid_manifest_and_repeated_hash_are_deterministic(configuration: Path) -> None:
    first = compile_registry(configuration)
    second = compile_registry(configuration)
    assert first.protocol_hash == second.protocol_hash
    assert first.registry == second.registry


def test_formatting_does_not_change_hash_but_semantics_do(configuration: Path) -> None:
    first = compile_registry(configuration).protocol_hash
    with (configuration.parent / "methodology" / "study.yaml").open("a", encoding="utf-8") as stream:
        stream.write("\n# formatting-only comment\n")
    assert compile_registry(configuration).protocol_hash == first
    replace(configuration.parent / "methodology" / "study.yaml", "observed_endpoint", "different_endpoint")
    assert compile_registry(configuration).protocol_hash != first


def test_missing_module_and_duplicate_owner_fail(configuration: Path) -> None:
    (configuration.parent / "foundation" / "metadata.yaml").unlink()
    with pytest.raises(MissingModuleError):
        compile_registry(configuration)
    configuration = ROOT / "config" / "pipeline.yaml"
    raw = configuration.read_text(encoding="utf-8").replace(
        "  metadata: foundation/metadata.yaml", "  duplicate: foundation/metadata.yaml\n  metadata: foundation/metadata.yaml"
    )
    local = ROOT / "work" / "duplicate-pipeline.yaml"
    local.write_text(raw, encoding="utf-8")
    try:
        with pytest.raises(DuplicateOwnershipError):
            load_manifest(local)
    finally:
        local.unlink()


def test_path_collision_and_traceability_fail(configuration: Path) -> None:
    replace(configuration.parent / "foundation" / "artifacts.yaml", "P17/report.json", "P00/registry.lock.json")
    with pytest.raises(ArtifactPathCollisionError):
        compile_registry(configuration)
    configuration = ROOT / "config" / "pipeline.yaml"


def test_incomplete_decision_traceability_fails(configuration: Path) -> None:
    replace(configuration.parent / "assurance" / "decisions.yaml", "D45:", "DX45:")
    with pytest.raises(DecisionTraceabilityError):
        compile_registry(configuration)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("[L2, L3_fixed_pi, none]", "[L2, L3_hierarchical_pi, none]"),
        ("label_model_allows_content: false", "label_model_allows_content: true"),
        ("ap_soft_targets: false", "ap_soft_targets: true"),
        ("ipcw_role: sensitivity_only", "ipcw_role: confirmatory"),
    ],
)
def test_methodological_invariants_fail_closed(configuration: Path, old: str, new: str) -> None:
    for path in configuration.parent.rglob("*.yaml"):
        if old in path.read_text(encoding="utf-8"):
            replace(path, old, new)
            break
    with pytest.raises(MethodologicalInvariantError):
        compile_registry(configuration)


def test_generated_docs_have_header_and_drift_is_detected(configuration: Path, tmp_path: Path) -> None:
    compiled = compile_registry(configuration)
    documents = render_documents(dict(compiled.registry))
    assert all(value.startswith("GENERATED FILE — DO NOT EDIT") for value in documents.values())
    with pytest.raises(GeneratedFileDriftError):
        write_or_check_documents(tmp_path, documents, check=True)
    write_or_check_documents(tmp_path, documents, check=False)
    write_or_check_documents(tmp_path, documents, check=True)


def test_forbidden_pattern_guard_uses_compiled_columns(configuration: Path, tmp_path: Path) -> None:
    compiled = compile_registry(configuration)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bad.py").write_text("pd.read_csv('x')\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        validate_source_patterns(tmp_path, dict(compiled.registry))


def test_p00_cli_is_atomic_and_non_overwriting(tmp_path: Path) -> None:
    command = [sys.executable, str(ROOT / "scripts" / "p00_lock_protocol.py"), "--config", str(ROOT / "config" / "pipeline.yaml"), "--run-id", "test", "--output-root", str(tmp_path)]
    assert subprocess.run(command, check=False).returncode == 0
    assert (tmp_path / "test" / "P00" / "_SUCCESS.json").is_file()
    assert subprocess.run(command, check=False).returncode != 0
