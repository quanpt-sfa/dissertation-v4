"""P06 CLI: build descriptive observability and verification registries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping
from observability.service import build_observability_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P06",
        state="MEASURED",
    )
    if args.dry_run:
        print("P06 dry-run: descriptive observability only; analytical weights prohibited")
        return 0
    matrices = mapping(loaded.context.read("source_channel_matrices", {}), "source matrices")
    loaded.context.read("l3_pilot_capability", {})
    sources = mapping(
        mapping(
            mapping(loaded.registry.get("data_sources"), "data_sources").get("source_registry"),
            "source_registry",
        ).get("sources"),
        "source_registry.sources",
    )
    metadata = {
        source_id: mapping(value, f"source={source_id}")
        for source_id, value in sources.items()
        if mapping(value, f"source={source_id}").get("role") == "evidence"
    }
    result = build_observability_registry(matrices, metadata)
    if args.validate_only:
        return 0
    loaded.context.write("observability_registry", result, {})
    print(f"P06 status=PASS channels={len(mapping(result['channels'], 'channels'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
