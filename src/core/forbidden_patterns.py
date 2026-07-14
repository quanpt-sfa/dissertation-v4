"""Static guard for direct artifact I/O and copied semantic literals."""

from __future__ import annotations

import ast
from pathlib import Path

from .errors import ConfigurationError

FORBIDDEN_IO = (
    "pd.read_csv(",
    "pd.read_parquet(",
    ".to_csv(",
    ".to_parquet(",
    "np.random.seed(",
)
FORBIDDEN_APPEND = ('mode="a"', "mode='a'")
APPROVED_CORE_FILES = {
    "artifact_store.py",
    "config_loader.py",
    "forbidden_patterns.py",
}


def validate_source_patterns(root: Path, registry: dict[str, object]) -> None:
    columns = registry.get("columns")
    artifacts = registry.get("artifacts")
    physical_names: set[str] = set()
    artifact_paths: set[str] = set()
    if isinstance(columns, dict):
        for value in columns.values():
            if isinstance(value, dict) and isinstance(value.get("physical_name"), str):
                physical_names.add(value["physical_name"])
    if isinstance(artifacts, dict):
        for value in artifacts.values():
            if isinstance(value, dict) and isinstance(value.get("path_template"), str):
                artifact_paths.add(value["path_template"])

    for path in [*root.glob("scripts/*.py"), *root.glob("src/**/*.py")]:
        if path.name in APPROVED_CORE_FILES and path.parent.name == "core":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in (*FORBIDDEN_IO, *FORBIDDEN_APPEND):
            if pattern in text:
                raise ConfigurationError(
                    f"source={path}: forbidden pattern {pattern}; use the core runtime layer"
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
        copied_columns = sorted(physical_names & literals)
        if copied_columns:
            raise ConfigurationError(
                f"source={path}: registered physical columns copied into source: {copied_columns}"
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
