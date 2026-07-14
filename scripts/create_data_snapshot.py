"""Discover real data files and create an immutable operational snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.registry_compiler import compile_registry
from snapshot.builder import build_snapshot, write_snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compiled = compile_registry(args.config)
    snapshot = build_snapshot(
        registry=dict(compiled.registry),
        raw_root=args.raw_root,
        snapshot_id=args.snapshot_id,
    )
    write_snapshot(args.output, snapshot)
    print(
        f"snapshot_id={snapshot['snapshot_id']} "
        f"sources={len(cast(list[object], snapshot['sources']))} "
        f"snapshot_hash={snapshot['snapshot_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
