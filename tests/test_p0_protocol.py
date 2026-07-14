"""Focused P0 regression tests independent of empirical data."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from core.config_loader import load_manifest
from core.errors import (
    ArtifactPathCollisionError,
    ConfigurationError,
    DecisionTraceabilityError,
    DuplicateOwnershipError,
    GeneratedFileDriftError,
    MethodologicalInvariantError,
    MissingModuleError,
    UndeclaredConfigurationFileError,
)
from core.forbidden_patterns import validate_source_patterns
from core.generated_docs import render_documents, write_or_check_documents
from core.registry_compiler import compile_registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def configuration(tmp_path: Path) -> Path:
    destination = tmp_path / "config"
    shutil.copytree(ROOT / "config", destination)
    return destination / "pipeline.yaml"


def read_yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def write_yaml(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def test_valid_manifest_and_repeated_hash_are_deterministic(
    configuration: Path,
) -> None:
    first = compile_registry(configuration)
    second = compile_registry(configuration)
    assert first.protocol_hash == second.protocol_hash
    assert first.registry == second.registry


def test_formatting_does_not_change_hash_but_semantics_do(
    configuration: Path,
) -> None:
    first = compile_registry(configuration).protocol_hash
    study = configuration.parent / "methodology" / "study.yaml"
    study.write_text(
        study.read_text(encoding="utf-8") + "\n# formatting-only\n",
        encoding="utf-8",
    )
    assert compile_registry(configuration).protocol_hash == first
    value = read_yaml(study)
    value["study"]["horizons_months"]["primary"] = 13  # type: ignore[index]
    write_yaml(study, value)
    with pytest.raises(MethodologicalInvariantError):
        compile_registry(configuration)


def test_missing_module_duplicate_owner_and_undeclared_yaml_fail(
    configuration: Path,
) -> None:
    metadata = configuration.parent / "foundation" / "metadata.yaml"
    metadata.unlink()
    with pytest.raises(MissingModuleError):
        compile_registry(configuration)

    manifest = ROOT / "config" / "pipeline.yaml"
    raw = manifest.read_text(encoding="utf-8").replace(
        "  metadata: foundation/metadata.yaml",
        "  duplicate: foundation/metadata.yaml\n  metadata: foundation/metadata.yaml",
    )
    local = ROOT / "work" / "duplicate-pipeline.yaml"
    local.parent.mkdir(exist_ok=True)
    local.write_text(raw, encoding="utf-8")
    try:
        with pytest.raises(DuplicateOwnershipError):
            load_manifest(local)
    finally:
        local.unlink(missing_ok=True)

    configuration = configuration.parent / "pipeline.yaml"
    stray = configuration.parent / "stray.yaml"
    stray.write_text("stray: true\n", encoding="utf-8")
    with pytest.raises(UndeclaredConfigurationFileError):
        compile_registry(configuration)


def test_path_collision_and_incomplete_traceability_fail(
    configuration: Path,
) -> None:
    artifacts = configuration.parent / "foundation" / "artifacts.yaml"
    value = read_yaml(artifacts)
    value["artifacts"]["final_report"]["path_template"] = "P00/registry.lock.json"  # type: ignore[index]
    write_yaml(artifacts, value)
    with pytest.raises(ArtifactPathCollisionError):
        compile_registry(configuration)

    configuration = ROOT / "config" / "pipeline.yaml"
    copied = configuration.parent.parent / "work" / "invalid-config"
    if copied.exists():
        shutil.rmtree(copied)
    shutil.copytree(configuration.parent, copied)
    decisions = copied / "assurance" / "decisions.yaml"
    value = read_yaml(decisions)
    value["decisions"]["DX45"] = value["decisions"].pop("D45")  # type: ignore[index]
    write_yaml(decisions, value)
    try:
        with pytest.raises(DecisionTraceabilityError):
            compile_registry(copied / "pipeline.yaml")
    finally:
        shutil.rmtree(copied)


@pytest.mark.parametrize(
    ("path", "mutator"),
    [
        (
            "methodology/measurement.yaml",
            lambda value: value["measurement"].update(
                {"selection_candidates": ["L2", "L3_hierarchical_pi", "none"]}
            ),
        ),
        (
            "methodology/features.yaml",
            lambda value: value["features"].update({"label_model_allows_content": True}),
        ),
        (
            "methodology/evaluation.yaml",
            lambda value: value["evaluation"].update({"ap_soft_targets": True}),
        ),
        (
            "execution/weighting.yaml",
            lambda value: value["weighting"].update({"ipcw_role": "confirmatory"}),
        ),
    ],
)
def test_methodological_invariants_fail_closed(
    configuration: Path, path: str, mutator: object
) -> None:
    target = configuration.parent / path
    value = read_yaml(target)
    mutator(value)  # type: ignore[operator]
    write_yaml(target, value)
    with pytest.raises(MethodologicalInvariantError):
        compile_registry(configuration)


def test_generated_docs_have_header_and_drift_is_detected(
    configuration: Path, tmp_path: Path
) -> None:
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
