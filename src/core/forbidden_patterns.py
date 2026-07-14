"""Static guard for direct I/O and copied registry literals."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

from .errors import ConfigurationError

FORBIDDEN_IO = (
    "pd.read_csv(",
    "pd.read_parquet(",
    ".to_csv(",
    ".to_parquet(",
    "np.random.seed(",
)
FORBIDDEN_APPEND = (
    'mode="a"',
    "mode='a'",
)
APPROVED_CORE_FILES = {
    "artifact_store.py",
    "config_loader.py",
    "forbidden_patterns.py",
}

StringMap = dict[str, Any]


def _mapping(value: object, context: str) -> StringMap:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context}: mapping required")
    return cast(StringMap, value)


def _is_p01_boundary(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return relative.startswith("src/p01/") or relative in {
        "scripts/p01_audit_raw.py",
        "scripts/p01_raw_audit.py",
    }


def validate_source_patterns(
    root: Path,
    registry: dict[str, object],
) -> None:
    columns = _mapping(registry.get("columns"), "columns")
    artifacts = _mapping(registry.get("artifacts"), "artifacts")

    physical_names: set[str] = set()
    artifact_paths: set[str] = set()

    for column_id, raw_column in columns.items():
        column = _mapping(raw_column, f"column={column_id}")
        physical_name = column.get("physical_name")
        if not isinstance(physical_name, str):
            raise ConfigurationError(f"column={column_id}: physical_name must be a string")
        physical_names.add(physical_name)

    for artifact_id, raw_artifact in artifacts.items():
        artifact = _mapping(raw_artifact, f"artifact={artifact_id}")
        path_template = artifact.get("path_template")
        if not isinstance(path_template, str):
            raise ConfigurationError(f"artifact={artifact_id}: path_template must be a string")
        artifact_paths.add(path_template)

    source_files = [
        *root.glob("scripts/*.py"),
        *root.glob("src/**/*.py"),
    ]
    for path in source_files:
        if path.name in APPROVED_CORE_FILES and path.parent.name == "core":
            continue

        text = path.read_text(encoding="utf-8")
        for pattern in (*FORBIDDEN_IO, *FORBIDDEN_APPEND):
            if pattern in text:
                raise ConfigurationError(
                    f"source={path}: forbidden pattern {pattern}; "
                    "use the registered raw-reader or core runtime layer"
                )

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise ConfigurationError(f"source={path}: Python syntax error") from exc

        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        if not _is_p01_boundary(root, path):
            copied_columns = sorted(physical_names & literals)
            if copied_columns:
                raise ConfigurationError(
                    f"source={path}: registered physical columns "
                    f"copied into source: {copied_columns}"
                )

        copied_paths = sorted(artifact_paths & literals)
        if copied_paths:
            raise ConfigurationError(
                f"source={path}: registered artifact paths copied into source: {copied_paths}"
            )

        direct_config_paths = sorted(
            value
            for value in literals
            if (value.startswith("config/") or value.startswith("config\\"))
            and value
            not in {
                "config/pipeline.yaml",
                "config\\pipeline.yaml",
            }
        )
        if direct_config_paths:
            raise ConfigurationError(
                f"source={path}: direct source-config paths are forbidden: {direct_config_paths}"
            )
