"""Compile and atomically publish the P00 protocol lock only."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.atomic_io import publish_p00
from core.registry_compiler import compile_registry, environment_observation, source_manifest
from core.schema_registry import contract_registry


def _path(artifacts: dict[str, object], artifact_id: str, stage: Path) -> str:
    item = artifacts.get(artifact_id)
    if not isinstance(item, dict) or not isinstance(item.get("path_template"), str):
        raise TypeError(f"artifact={artifact_id}: path template required")
    return str(Path(item["path_template"]).relative_to(stage))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    compiled = compile_registry(arguments.config)
    if arguments.validate_only:
        return 0
    registry = dict(compiled.registry)
    artifacts = registry.get("artifacts")
    reproducibility = registry.get("reproducibility")
    vocabulary = registry.get("vocabulary")
    if (
        not isinstance(artifacts, dict)
        or not isinstance(reproducibility, dict)
        or not isinstance(vocabulary, dict)
    ):
        raise TypeError("compiled registry lacks P00 foundations")
    artifacts = cast(dict[str, Any], artifacts)
    reproducibility = cast(dict[str, Any], reproducibility)
    vocabulary = cast(dict[str, Any], vocabulary)
    output_template = reproducibility.get("output_root_template")
    lock = artifacts.get("registry_lock")
    if (
        not isinstance(output_template, str)
        or not isinstance(lock, dict)
        or not isinstance(lock.get("path_template"), str)
    ):
        raise TypeError("configured output policy is incomplete")
    root = arguments.config.resolve().parent.parent
    run_root = (
        (arguments.output_root / arguments.run_id)
        if arguments.output_root
        else root / output_template.format(run_id=arguments.run_id)
    )
    stage = Path(lock["path_template"]).parent
    target = run_root / stage
    contracts = contract_registry(registry)
    capabilities = registry.get("capabilities")
    publication_statuses = vocabulary.get("publication_statuses")
    seal_states = vocabulary.get("seal_states")
    if (
        not isinstance(capabilities, dict)
        or not isinstance(publication_statuses, dict)
        or not isinstance(seal_states, dict)
        or not isinstance(vocabulary.get("reason_codes"), list)
    ):
        raise TypeError("configured vocabulary is incomplete")
    capabilities = cast(dict[str, Any], capabilities)
    publication_statuses = cast(dict[str, str], publication_statuses)
    seal_states = cast(dict[str, str], seal_states)

    def p(artifact_id: str) -> str:
        return _path(artifacts, artifact_id, stage)

    environment = environment_observation(root)
    files: dict[str, object] = {
        p("registry_lock"): registry,
        p("protocol_hash"): compiled.protocol_hash + "\n",
        p("source_config_manifest"): source_manifest(compiled, root),
        p("capability_seed"): {
            capability_id: specification["initial_status"]
            for capability_id, specification in capabilities.items()
            if isinstance(specification, dict) and isinstance(specification.get("initial_status"), str)
        },
        p("decision_traceability"): registry["decision_traceability"],
        p("artifact_catalog"): artifacts,
        p("schema_catalog"): registry["schemas"],
        p("step_catalog"): registry["steps"],
        p("access_matrix"): registry["access_matrix"],
        p("known_cases_seal"): {
            "status": seal_states["sealed"],
            "protocol_hash": compiled.protocol_hash,
            "opens_at_step": registry["known_cases"]["opening_step"],
        },
        p("environment_expectation"): environment,
        p("job_manifest"): {
            "run_id": arguments.run_id,
            "protocol_hash": compiled.protocol_hash,
            "output_hashes": {},
        },
        p("p00_audit_report"): _report(arguments.run_id, compiled.protocol_hash, registry),
    }
    p00_names = (
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
        "job_manifest",
        "p00_audit_report",
    )
    names = {p(name): name for name in p00_names}

    def validate(path: str, value: object) -> None:
        if path == "_SUCCESS.json":
            contracts.validate("success_receipt", value)
        else:
            contracts.validate(names[path], value)

    publish_p00(
        target,
        files,
        validate,
        {
            "status": publication_statuses["success"],
            "protocol_hash": compiled.protocol_hash,
        },
    )
    return 0


def _report(run_id: str, protocol_hash: str, registry: dict[str, object]) -> str:
    decisions = registry["decision_traceability"]
    schemas = registry["schemas"]
    artifacts = registry["artifacts"]
    steps = registry["steps"]
    tests = registry["tests"]
    if not all(isinstance(item, dict) or isinstance(item, list) for item in (decisions, schemas, artifacts, steps, tests)):
        raise TypeError("compiled registry lacks reportable catalogs")
    decisions = cast(Sized, decisions)
    schemas = cast(Sized, schemas)
    artifacts = cast(Sized, artifacts)
    steps = cast(Sized, steps)
    tests = cast(Sized, tests)
    return "\n".join(
        (
            "# P00 audit report",
            "",
            f"- Run ID: `{run_id}`",
            f"- Protocol hash: `{protocol_hash}`",
            f"- Compiler version: `{registry['compiler_version']}`",
            f"- Source module count: {len(registry) - 4}",
            f"- Schema count: {len(schemas)}",
            f"- Artifact count: {len(artifacts)}",
            f"- Step count: {len(steps)}",
            f"- Test count: {len(tests)}",
            f"- D01-D45 completeness: {len(decisions) == 45}",
            "- Methodological invariants: passed during compilation",
            "- Access policy: passed during compilation",
            "- Path collisions: passed during compilation",
            "- Generated-document drift: checked by bootstrap",
            "- Forbidden patterns: checked by bootstrap",
            "- Final status: SUCCESS",
            "",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
