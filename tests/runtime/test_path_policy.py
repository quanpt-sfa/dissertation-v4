from __future__ import annotations

from pathlib import Path

import pytest

from core.artifact_store import ArtifactStore
from core.errors import ConfigurationError
from core.path_policy import RelativePathError, portable_relative_path, resolve_relative_path


def _windows_absolute() -> str:
    return "D:" + "\\machine\\data\\source.parquet"


def _windows_drive_relative() -> str:
    return "D:" + "machine\\data\\source.parquet"


def _unc_path() -> str:
    return "\\\\" + "server\\share\\source.parquet"


def test_nonexistent_p01_partition_path_stays_beneath_run_root(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    target = resolve_relative_path(
        run_root,
        "P01/raw_audit/financial_statement_core_long.json",
        context="raw_audit",
    )

    expected = run_root.resolve() / "P01" / "raw_audit" / "financial_statement_core_long.json"
    assert target == expected
    assert not target.exists()


def test_artifact_store_formats_safe_coordinates_without_resolving_missing_leaf(
    tmp_path: Path,
) -> None:
    artifacts: dict[str, object] = {
        "raw_audit": {
            "producer_step": "P01",
            "path_template": "P01/raw_audit/{source_id}.json",
            "schema_id": "raw_audit_schema",
            "format": "json",
            "coordinates": ["source_id"],
            "immutability": "immutable",
        }
    }
    store = ArtifactStore(
        tmp_path / "run",
        artifacts,
        contracts=None,  # type: ignore[arg-type]
        protocol_hash="p" * 64,
    )

    target = store.path("raw_audit", {"source_id": "financial_statement_core_long"})

    expected = (
        tmp_path.resolve() / "run" / "P01" / "raw_audit" / ("financial_statement_core_long.json")
    )
    assert target == expected


@pytest.mark.parametrize(
    "value",
    [
        _windows_absolute(),
        _windows_drive_relative(),
        _unc_path(),
        str(Path("/") / "machine" / "source.parquet"),
        "P01/../outside.json",
        "P01//raw_audit.json",
        "./P01/raw_audit.json",
    ],
)
def test_nonportable_or_escaping_paths_fail_closed(value: str) -> None:
    with pytest.raises(RelativePathError):
        portable_relative_path(value, context="test")


def test_coordinate_separator_injection_fails_closed(tmp_path: Path) -> None:
    artifacts: dict[str, object] = {
        "raw_audit": {
            "producer_step": "P01",
            "path_template": "P01/raw_audit/{source_id}.json",
            "schema_id": "raw_audit_schema",
            "format": "json",
            "coordinates": ["source_id"],
            "immutability": "immutable",
        }
    }
    store = ArtifactStore(
        tmp_path / "run",
        artifacts,
        contracts=None,  # type: ignore[arg-type]
        protocol_hash="p" * 64,
    )

    with pytest.raises(ConfigurationError, match="invalid relative path contract"):
        store.path("raw_audit", {"source_id": "../outside"})


def test_existing_symlink_parent_cannot_escape_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "P01"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(RelativePathError, match="existing path component escapes root"):
        resolve_relative_path(
            root,
            "P01/raw_audit/source.json",
            context="artifact=raw_audit",
        )
