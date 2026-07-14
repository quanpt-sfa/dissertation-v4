"""P02 CLI: construct canonical firm master and as-of panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.artifact_store import ArtifactStore
from core.runtime import RunContext
from core.schema_registry import contract_registry
from p01.registry import load_locked_registry, resolve_source
from p02.builder import SourceInput, build_firm_panel
from p02.publish import publish_p02_outputs
from p02.registry import load_p02_specs, output_column_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    registry, locked_hash, run_root = load_locked_registry(args.registry)
    if run_root.name != args.run_id:
        raise ValueError(
            f"run-id={args.run_id} does not match registry run directory {run_root.name}"
        )
    artifacts = registry.get("artifacts")
    reproducibility = registry.get("reproducibility")
    if not isinstance(artifacts, dict) or not isinstance(reproducibility, dict):
        raise ValueError("registry artifact/reproducibility namespaces are unavailable")
    recovery_policy = cast(dict[str, object], reproducibility).get(
        "recovery_policy", "quarantine_incomplete"
    )
    if not isinstance(recovery_policy, str):
        raise ValueError("reproducibility.recovery_policy must be a string")

    store = ArtifactStore(
        run_root,
        cast(dict[str, object], artifacts),
        contract_registry(cast(dict[str, object], registry)),
        locked_hash,
        recovery_policy=recovery_policy,
    )
    context = RunContext("P02", "AUDITED", locked_hash, registry, store)

    source_specs, entity_spec, sample_start, sample_end = load_p02_specs(registry)
    inputs: list[SourceInput] = []
    for source_spec, panel_spec in source_specs:
        audit_value = context.read("raw_audit", {"source_id": source_spec.source_id})
        if not isinstance(audit_value, dict):
            raise ValueError(f"source={source_spec.source_id}: raw_audit must be a JSON object")
        resolved_spec, source_path = resolve_source(registry, source_spec.source_id)
        inputs.append(
            SourceInput(
                source_spec=resolved_spec,
                panel_spec=panel_spec,
                path=source_path,
                audit=cast(dict[str, Any], audit_value),
            )
        )

    result = build_firm_panel(
        run_id=args.run_id,
        protocol_hash=locked_hash,
        source_inputs=inputs,
        entity_spec=entity_spec,
        sample_start=sample_start,
        sample_end=sample_end,
        output_columns=output_column_names(registry),
    )
    publish_p02_outputs(
        run_root=run_root,
        registry=registry,
        protocol_hash=locked_hash,
        recovery_policy=recovery_policy,
        result=result,
    )

    summary = result.resolution_report["summary"]
    if not isinstance(summary, dict):
        raise ValueError("P02 summary missing after build")
    print(
        "P02 status=PASS "
        f"firms={summary['firm_count']} "
        f"firm_years={summary['panel_row_count']} "
        f"excluded={summary['excluded_firm_year_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
