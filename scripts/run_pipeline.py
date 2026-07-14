"""Production runner for snapshot -> P00 -> all P01 units -> P02."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _require_clean_tree(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    )
    if result.stdout.strip():
        raise RuntimeError("operational pipeline requires a clean committed Git tree")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--through", choices=["P00", "P01", "P02"], default="P02")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    if not args.allow_dirty:
        _require_clean_tree(project_root)
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / args.run_id
    if run_root.exists():
        raise FileExistsError(f"run root already exists and is immutable: {run_root}")

    snapshot_path = run_root / "SNAPSHOT" / "data_snapshot.json"
    env = dict(os.environ)
    env["DISSERTATION_RAW_ROOT"] = str(raw_root)
    python = sys.executable

    _run(
        [
            python,
            "scripts/create_data_snapshot.py",
            "--config",
            str(args.config),
            "--raw-root",
            str(raw_root),
            "--snapshot-id",
            args.run_id,
            "--output",
            str(snapshot_path),
        ],
        cwd=project_root,
        env=env,
    )
    _run(
        [
            python,
            "scripts/p00_lock_protocol.py",
            "--config",
            str(args.config),
            "--run-id",
            args.run_id,
            "--output-root",
            str(output_root),
            "--snapshot-manifest",
            str(snapshot_path),
        ],
        cwd=project_root,
        env=env,
    )
    if args.through == "P00":
        return 0

    snapshot_raw: object = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot_raw, dict):
        raise ValueError("snapshot must be an object")
    snapshot = cast(dict[str, object], snapshot_raw)
    sources_raw = snapshot.get("sources")
    if not isinstance(sources_raw, list):
        raise ValueError("snapshot.sources must be a list")
    source_ids: list[str] = []
    for raw in cast(list[object], sources_raw):
        if not isinstance(raw, dict):
            raise ValueError("snapshot source entry must be an object")
        source = cast(dict[str, object], raw)
        source_id = source.get("source_id")
        if not isinstance(source_id, str):
            raise ValueError("snapshot source_id required")
        source_ids.append(source_id)

    registry_path = run_root / "P00" / "registry.lock.json"
    for source_id in sorted(source_ids):
        _run(
            [
                python,
                "scripts/p01_audit_raw.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
                "--source-id",
                source_id,
            ],
            cwd=project_root,
            env=env,
        )
    if args.through == "P01":
        return 0

    _run(
        [
            python,
            "scripts/p02_build_firm_panel.py",
            "--registry",
            str(registry_path),
            "--run-id",
            args.run_id,
        ],
        cwd=project_root,
        env=env,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
