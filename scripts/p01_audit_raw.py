"""P01 CLI: audit one registered raw source or partition."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.artifact_store import ArtifactStore
from core.runtime import RunContext
from core.schema_registry import contract_registry
from p01.audit import audit_source
from p01.registry import load_locked_registry, resolve_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()

    registry, locked_hash, run_root = load_locked_registry(args.registry)
    if run_root.name != args.run_id:
        raise ValueError(
            f"run-id={args.run_id} does not match registry run directory {run_root.name}"
        )

    spec, source_path = resolve_source(registry, args.source_id)
    artifacts = registry.get("artifacts")
    reproducibility = registry.get("reproducibility")
    if not isinstance(artifacts, dict):
        raise ValueError("registry artifacts are unavailable")
    if not isinstance(reproducibility, dict):
        raise ValueError("registry reproducibility settings are unavailable")

    reproducibility_map = cast(dict[str, object], reproducibility)
    recovery_value = reproducibility_map.get("recovery_policy", "quarantine_incomplete")
    if not isinstance(recovery_value, str):
        raise ValueError("reproducibility.recovery_policy must be a string")
    recovery_policy = recovery_value

    store = ArtifactStore(
        run_root,
        cast(dict[str, object], artifacts),
        contract_registry(cast(dict[str, object], registry)),
        locked_hash,
        recovery_policy=recovery_policy,
    )
    context = RunContext(
        "P01",
        "LOCKED",
        locked_hash,
        registry,
        store,
        {"source_id": args.source_id},
    )

    report = audit_source(
        run_id=args.run_id,
        protocol_hash=locked_hash,
        source_path=source_path,
        spec=spec,
    )
    context.write(
        "raw_audit",
        report.to_dict(),
        {"source_id": args.source_id},
    )

    print(
        f"P01 source={args.source_id} status={report.status} "
        f"fatal={report.decision['fatal_issue_count']} "
        f"isolated={report.decision['isolated_issue_count']}"
    )
    return 0 if report.status != "FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
