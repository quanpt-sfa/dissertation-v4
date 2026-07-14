"""Static P0 guard for forbidden direct I/O and copied semantic literals."""

from __future__ import annotations

from pathlib import Path

from .errors import ConfigurationError

FORBIDDEN = ("pd.read_csv(", "pd.read_parquet(", ".to_csv(", ".to_parquet(", "np.random.seed(")


def validate_source_patterns(root: Path, registry: dict[str, object]) -> None:
    """Reject direct artifact I/O, global RNG seeding, and copied physical columns."""
    columns = registry.get("columns")
    physical_names: set[str] = set()
    if isinstance(columns, dict):
        for value in columns.values():
            if isinstance(value, dict) and isinstance(value.get("physical_name"), str):
                physical_names.add(value["physical_name"])
    for path in [*root.glob("scripts/*.py"), *root.glob("src/**/*.py")]:
        if path.parts[-2:] in {
            ("core", "forbidden_patterns.py"),
            ("core", "artifact_store.py"),
        }:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if pattern in text:
                raise ConfigurationError(
                    f"source={path}: forbidden pattern {pattern}; use the core runtime layer"
                )
        for physical_name in physical_names:
            if physical_name in text:
                raise ConfigurationError(
                    f"source={path}: registered physical column copied into source"
                )
