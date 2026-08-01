"""P06 CLI: build descriptive observability and verification registries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.evidence_registry import logical_evidence_sources
from core.pipeline import load_run, mapping
from core.semantic_keys import CHANNEL_ID, TEMPORAL_ROLE, VERIFICATION_STATUS
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
    sources = logical_evidence_sources(loaded.registry)
    metadata = {
        source_id: {
            CHANNEL_ID: value.channel_id,
            VERIFICATION_STATUS: value.verification_status,
            TEMPORAL_ROLE: value.temporal_role,
            "availability_rule": value.availability_rule,
            "explicit_negative_allowed": value.explicit_negative_allowed,
            "evidence_mapping": {"opportunity_semantic": value.opportunity_semantic},
        }
        for source_id, value in sources.items()
    }
    result = build_observability_registry(matrices, metadata)
    if args.validate_only:
        return 0
    loaded.context.write("observability_registry", result, {})
    print(f"P06 status=PASS channels={len(mapping(result['channels'], 'channels'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
