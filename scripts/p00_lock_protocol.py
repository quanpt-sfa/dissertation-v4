"""Lock a fully validated P0 protocol; performs no scientific computation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.atomic_io import publish_p00
from core.registry_compiler import compile_registry, source_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-new-run", action="store_true")
    arguments = parser.parse_args()
    compiled = compile_registry(arguments.config)
    if arguments.validate_only:
        return 0
    root = arguments.config.resolve().parent.parent
    output_root = arguments.output_root or root / "artifacts" / "runs"
    registry = dict(compiled.registry)
    artifacts = registry["artifacts"]
    if not isinstance(artifacts, dict):
        raise TypeError("compiled artifacts must be a mapping")
    lock = artifacts["registry_lock"]
    if not isinstance(lock, dict) or not isinstance(lock.get("path_template"), str):
        raise TypeError("registry_lock must declare path_template")
    stage_directory = Path(lock["path_template"]).parent
    target = output_root / arguments.run_id / stage_directory
    if target.exists() and arguments.force_new_run:
        arguments.run_id = f"{arguments.run_id}-{compiled.protocol_hash[:8]}"
        target = output_root / arguments.run_id / stage_directory
    if arguments.dry_run:
        print(target)
        return 0
    def p00_path(artifact_id: str) -> str:
        declaration = artifacts[artifact_id]
        if not isinstance(declaration, dict) or not isinstance(declaration.get("path_template"), str):
            raise TypeError(f"{artifact_id} must declare path_template")
        return str(Path(declaration["path_template"]).relative_to(stage_directory))

    files: dict[str, object] = {
        p00_path("registry_lock"): registry,
        p00_path("protocol_hash"): compiled.protocol_hash + "\n",
        p00_path("source_config_manifest"): source_manifest(compiled, root),
        p00_path("capability_seed"): {"L3": "EMPIRICALLY_PENDING", "strict_channel_holdout": "STRUCTURALLY_ELIGIBLE", "observed_verification": "UNAVAILABLE_BY_DESIGN"},
        p00_path("decision_traceability"): registry["decision_traceability"],
        p00_path("artifact_catalog"): registry["artifacts"],
        p00_path("schema_catalog"): registry["schemas"],
        p00_path("step_catalog"): registry["steps"],
        p00_path("access_matrix"): registry["access_matrix"],
        p00_path("known_cases_seal"): {"state": "SEALED", "opens_at_step": "P15"},
        p00_path("environment_expectation"): {"compiler_version": registry["compiler_version"]},
        p00_path("p00_audit_report"): f"# P00 audit report\n\nProtocol hash: `{compiled.protocol_hash}`\n",
    }
    publish_p00(target, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
