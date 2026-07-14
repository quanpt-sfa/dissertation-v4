"""Typed immutable models at the configuration parsing boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Manifest:
    """Validated sole-entry-point manifest."""

    path: Path
    modules: Mapping[str, str]
    schema_modules: tuple[str, ...]


@dataclass(frozen=True)
class CompiledRegistry:
    """Canonical registry and provenance produced by P0 compilation."""

    registry: Mapping[str, object]
    protocol_hash: str
    source_hashes: Mapping[str, str]
