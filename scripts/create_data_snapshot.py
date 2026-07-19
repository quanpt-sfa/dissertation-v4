"""Discover real data files and create an immutable operational snapshot."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.registry_compiler import compile_registry
from snapshot.builder import (
    _snapshot_content_hash,
    _snapshot_hash,
    build_snapshot,
    write_snapshot,
)

_ZERO_FLOOR_PARSE_SENTINEL = 1e-300


def _snapshot_compatible_registry(registry: dict[str, object]) -> dict[str, object]:
    """Bridge the legacy positive-only typed parser while preserving protocol semantics."""
    compatible = deepcopy(registry)
    catalog = cast(dict[str, Any], compatible["source_catalog"])
    profiles = cast(dict[str, Any], catalog["profiles"])
    for raw_profile in profiles.values():
        profile = cast(dict[str, Any], raw_profile)
        evidence = profile.get("evidence_mapping")
        if not isinstance(evidence, dict) or evidence.get("processor") != "audit_adjustment":
            continue
        adjustment = evidence.get("audit_adjustment")
        if not isinstance(adjustment, dict):
            continue
        if adjustment.get("minimum_absolute_denominator") == 0.0:
            adjustment["minimum_absolute_denominator"] = _ZERO_FLOOR_PARSE_SENTINEL
    return compatible


def _restore_zero_floor(snapshot: dict[str, object]) -> None:
    """Restore the locked zero-only guard before immutable snapshot hashes are computed."""
    sources = snapshot.get("sources")
    if not isinstance(sources, list):
        raise ValueError("snapshot.sources must be a list")
    for raw_source in cast(list[object], sources):
        if not isinstance(raw_source, dict):
            continue
        evidence = raw_source.get("evidence_mapping")
        if not isinstance(evidence, dict) or evidence.get("processor") != "audit_adjustment":
            continue
        adjustment = evidence.get("audit_adjustment")
        if not isinstance(adjustment, dict):
            continue
        if adjustment.get("minimum_absolute_denominator") == _ZERO_FLOOR_PARSE_SENTINEL:
            adjustment["minimum_absolute_denominator"] = 0.0
    snapshot["snapshot_content_hash"] = _snapshot_content_hash(snapshot)
    snapshot["snapshot_hash"] = _snapshot_hash(snapshot)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compiled = compile_registry(args.config)
    snapshot = build_snapshot(
        registry=_snapshot_compatible_registry(dict(compiled.registry)),
        raw_root=args.raw_root,
        snapshot_id=args.snapshot_id,
    )
    _restore_zero_floor(snapshot)
    write_snapshot(args.output, snapshot)
    print(
        f"snapshot_id={snapshot['snapshot_id']} "
        f"sources={len(cast(list[object], snapshot['sources']))} "
        f"snapshot_hash={snapshot['snapshot_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
