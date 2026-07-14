"""Register and hash one raw source before creating the final P00 lock."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from p01.readers import hash_file


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-sources-config",
        type=Path,
        default=Path("config/methodology/data_sources.yaml"),
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--relative-path", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.data_sources_config.resolve()
    root = args.source_root.resolve()
    source_path = (root / args.relative_path).resolve()
    if source_path == root or root not in source_path.parents:
        raise ValueError("relative-path escapes source-root")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    profile_raw: object = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    profile = _mapping(profile_raw, "profile")
    profile["relative_path"] = args.relative_path.as_posix()
    profile["locked_sha256"] = hash_file(source_path)

    config_raw: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = _mapping(config_raw, "data_sources config")
    data_sources = _mapping(config.get("data_sources"), "data_sources")
    source_registry = _mapping(data_sources.get("source_registry"), "data_sources.source_registry")
    sources = _mapping(source_registry.get("sources"), "data_sources.source_registry.sources")
    sources[args.source_id] = profile

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    print(
        f"Registered source_id={args.source_id} "
        f"relative_path={args.relative_path.as_posix()} "
        f"sha256={profile['locked_sha256']}"
    )
    print("Re-run bootstrap, commit the configuration, and create a new clean P00 lock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
