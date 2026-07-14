"""Atomic publication of the three P02 artifacts as one stage."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from core.artifact_store import ArtifactStore
from core.errors import ConfigurationError
from core.runtime import RunContext
from core.schema_registry import contract_registry

from .builder import PanelBuildResult

P02_ARTIFACTS = ("firm_master", "firm_year_panel", "duplicate_map")


def publish_p02_outputs(
    *,
    run_root: Path,
    registry: dict[str, Any],
    protocol_hash: str,
    recovery_policy: str,
    result: PanelBuildResult,
    dependencies: list[dict[str, object]] | None = None,
) -> None:
    artifacts_raw = registry.get("artifacts")
    if not isinstance(artifacts_raw, dict):
        raise ConfigurationError("registry artifacts are unavailable")
    artifacts = cast(dict[str, object], artifacts_raw)
    final_directory = run_root / "P02"
    _prepare_final_directory(final_directory, run_root, artifacts)

    stage_root = Path(tempfile.mkdtemp(prefix=".p02-stage-", dir=run_root))
    try:
        stage_store = ArtifactStore(
            stage_root,
            artifacts,
            contract_registry(cast(dict[str, object], registry)),
            protocol_hash,
            recovery_policy=recovery_policy,
        )
        stage_context = RunContext(
            "P02",
            "AUDITED",
            protocol_hash,
            registry,
            stage_store,
        )
        stage_context.inherit_dependencies(dependencies or [])
        stage_context.write("firm_master", result.firm_master, {})
        stage_context.write("firm_year_panel", result.firm_year_panel, {})
        stage_context.write("duplicate_map", result.resolution_report, {})
        staged_directory = stage_root / "P02"
        if not staged_directory.is_dir():
            raise ConfigurationError("P02 staging directory was not created")
        os.replace(staged_directory, final_directory)
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def _prepare_final_directory(
    final_directory: Path,
    run_root: Path,
    artifacts: dict[str, object],
) -> None:
    if not final_directory.exists():
        return
    if _is_complete(final_directory, run_root, artifacts):
        raise ConfigurationError(f"output={final_directory}: successful P02 stage is immutable")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantine = final_directory.with_name(f"P02.quarantine-{timestamp}")
    os.replace(final_directory, quarantine)


def _is_complete(
    final_directory: Path,
    run_root: Path,
    artifacts: dict[str, object],
) -> bool:
    for artifact_id in P02_ARTIFACTS:
        raw = artifacts.get(artifact_id)
        if not isinstance(raw, dict):
            return False
        artifact = cast(dict[str, object], raw)
        path_template = artifact.get("path_template")
        if not isinstance(path_template, str):
            return False
        artifact_path = run_root / path_template
        manifest_path = artifact_path.with_name(artifact_path.name + ".manifest.json")
        if not artifact_path.is_file() or not manifest_path.is_file():
            return False
    return final_directory.is_dir()
