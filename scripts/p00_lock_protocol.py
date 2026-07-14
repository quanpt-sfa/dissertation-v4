"""Compile and atomically publish P00."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.atomic_io import publish_p00
from core.registry_compiler import (
    compile_registry,
    environment_observation,
    source_manifest,
)
from core.schema_registry import contract_registry


def _relative(artifacts: dict[str, Any], artifact_id: str, stage: Path) -> str:
    return str(Path(artifacts[artifact_id]["path_template"]).relative_to(stage))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.snapshot_manifest is None:
        raise ValueError("--snapshot-manifest is required for a published operational P00 lock")
    compiled = compile_registry(args.config, args.snapshot_manifest)
    if args.validate_only:
        return 0

    registry = dict(compiled.registry)
    artifacts = cast(dict[str, Any], registry["artifacts"])
    reproducibility = cast(dict[str, Any], registry["reproducibility"])
    vocabulary = cast(dict[str, Any], registry["vocabulary"])
    root = args.config.resolve().parent.parent

    if args.run_id.startswith("p0-final") and reproducibility.get(
        "require_clean_tree_for_acceptance"
    ):
        manifest_preview = source_manifest(compiled, root)
        if manifest_preview["git_dirty"]:
            raise RuntimeError("final acceptance run requires a clean committed working tree")

    run_root = (
        args.output_root / args.run_id
        if args.output_root
        else root / reproducibility["output_root_template"].format(run_id=args.run_id)
    )
    stage = Path(artifacts["registry_lock"]["path_template"]).parent
    target = run_root / stage
    contracts = contract_registry(registry)

    def p(artifact_id: str) -> str:
        return _relative(artifacts, artifact_id, stage)

    capabilities = cast(dict[str, Any], registry["capabilities"])
    files: dict[str, object] = {
        p("registry_lock"): registry,
        p("protocol_hash"): compiled.protocol_hash + "\n",
        p("source_config_manifest"): source_manifest(compiled, root),
        p("capability_seed"): {key: value["initial_status"] for key, value in capabilities.items()},
        p("decision_traceability"): registry["decision_traceability"],
        p("artifact_catalog"): artifacts,
        p("schema_catalog"): registry["schemas"],
        p("step_catalog"): registry["steps"],
        p("access_matrix"): registry["access_matrix"],
        p("known_cases_seal"): {
            "status": vocabulary["seal_states"]["sealed"],
            "protocol_hash": compiled.protocol_hash,
            "opens_at_step": registry["known_cases"]["opening_step"],
        },
        p("environment_expectation"): reproducibility["environment_expectations"],
        p("environment_observation"): environment_observation(root),
        p("job_manifest"): {
            "run_id": args.run_id,
            "protocol_hash": compiled.protocol_hash,
            "output_hashes": {},
        },
        p("p00_audit_report"): _report(args.run_id, compiled.protocol_hash, registry),
    }
    names = {
        p(name): name
        for name in (
            "registry_lock",
            "protocol_hash",
            "source_config_manifest",
            "capability_seed",
            "decision_traceability",
            "artifact_catalog",
            "schema_catalog",
            "step_catalog",
            "access_matrix",
            "known_cases_seal",
            "environment_expectation",
            "environment_observation",
            "job_manifest",
            "p00_audit_report",
        )
    }

    def validate(path: str, value: object) -> None:
        contracts.validate(
            "success_receipt" if path == "_SUCCESS.json" else names[path],
            value,
        )

    publish_p00(
        target,
        files,
        validate,
        {
            "status": vocabulary["publication_statuses"]["success"],
            "protocol_hash": compiled.protocol_hash,
        },
    )
    return 0


def _report(run_id: str, protocol_hash: str, registry: dict[str, object]) -> str:
    return "\n".join(
        (
            "# P00 audit report",
            "",
            f"- Run ID: `{run_id}`",
            f"- Protocol hash: `{protocol_hash}`",
            f"- Compiler version: `{registry['compiler_version']}`",
            f"- Canonical Appendix B decisions: {len(cast(Sized, registry['appendix_b']))}",
            f"- Traceability rows: {len(cast(Sized, registry['decision_traceability']))}",
            f"- Schemas: {len(cast(Sized, registry['schemas']))}",
            f"- Artifacts: {len(cast(Sized, registry['artifacts']))}",
            f"- Steps: {len(cast(Sized, registry['steps']))}",
            f"- Executable test controls: {len(cast(Sized, registry['tests']))}",
            f"- Data snapshot ID: `{cast(dict[str, object], registry.get('data_snapshot', {})).get('snapshot_id')}`",
            f"- Data snapshot hash: `{cast(dict[str, object], registry.get('data_snapshot', {})).get('snapshot_hash')}`",
            "- Methodological invariants: PASS",
            "- Artifact/access references: PASS",
            "- Test-node collection: PASS",
            "- Final status: SUCCESS",
            "",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
