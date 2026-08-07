from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from core.path_policy import portable_relative_path

ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return cast(dict[str, Any], raw)


def test_pipeline_module_paths_are_portable_relative_paths() -> None:
    manifest = _yaml(ROOT / "config" / "pipeline.yaml")
    paths = [
        *cast(dict[str, str], manifest["modules"]).values(),
        *cast(list[str], manifest["schema_modules"]),
    ]

    for path in paths:
        assert "\\" not in path
        portable_relative_path(path, context="pipeline module")


def test_artifact_templates_are_portable_relative_paths() -> None:
    document = _yaml(ROOT / "config" / "foundation" / "artifacts.yaml")
    artifacts = cast(dict[str, dict[str, object]], document["artifacts"])

    for artifact_id, artifact in artifacts.items():
        template = artifact["path_template"]
        assert isinstance(template, str)
        assert "\\" not in template
        portable_relative_path(template, context=f"artifact={artifact_id}")


def test_source_discovery_and_provenance_paths_are_portable() -> None:
    catalog_document = _yaml(ROOT / "config" / "methodology" / "source_catalog.yaml")
    profiles = cast(
        dict[str, dict[str, object]],
        catalog_document["source_catalog"]["profiles"],
    )
    for profile_id, profile in profiles.items():
        discovery = cast(dict[str, object], profile["discovery"])
        for path in [
            *cast(list[str], discovery.get("globs", [])),
            *cast(list[str], discovery.get("excludes", [])),
        ]:
            assert "\\" not in path
            portable_relative_path(path, context=f"profile={profile_id}")

    data_sources_document = _yaml(ROOT / "config" / "methodology" / "data_sources.yaml")
    source_registry = cast(
        dict[str, object],
        data_sources_document["data_sources"]["source_registry"],
    )
    provenance = cast(dict[str, object], source_registry["provenance_contract"])
    manifest_path = provenance["manifest_relative_path"]
    assert isinstance(manifest_path, str)
    assert "\\" not in manifest_path
    portable_relative_path(manifest_path, context="extract provenance manifest")
