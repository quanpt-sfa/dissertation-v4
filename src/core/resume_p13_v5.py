"""Compatibility verifier for process/checkpoint P13 continuation from head 3510eae."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import cast

from core.resume import (
    hash_file,
    json_object,
    p11_boundary_complete,
    p12_boundary_complete,
    p13_resume_boundary_ready,
    read_compatibility_receipt,
    write_compatibility_receipt,
)

P13_PROCESS_COMPATIBILITY_PATCH_ID = "P13_PROCESS_CHECKPOINT_EXECUTION_V5"
P13_PROCESS_COMPATIBILITY_BASE_COMMIT = "3510eae3681381c655e0b3501f7700e5aa10a5fa"
P13_PROCESS_REQUIRED_SOURCE_DRIFT = frozenset(
    {
        "scripts/p13_sensitivity.py",
        "src/sensitivity/parallel_refits.py",
    }
)
P13_PROCESS_REQUIRED_CHANGED_PATHS = frozenset(
    {
        *P13_PROCESS_REQUIRED_SOURCE_DRIFT,
        "scripts/run_pipeline_p13_process_resume.py",
        "src/core/resume_p13_v5.py",
    }
)
P13_PROCESS_ALLOWED_CHANGED_PATHS = frozenset(
    {
        *P13_PROCESS_REQUIRED_CHANGED_PATHS,
        "tests/runtime/test_p13_process_resume.py",
        "tests/stages/test_p13_parallel_refits.py",
    }
)


def _git_output(root: Path, args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"resume refused: git {' '.join(args)} failed") from exc
    return result.stdout.strip()


def _git_lines(root: Path, args: list[str]) -> list[str]:
    output = _git_output(root, args)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _require_git_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "resume refused: current implementation is not descended from the locked commit"
        ) from exc


def _verify_raw_sources(
    raw_root: Path,
    snapshot: dict[str, object],
) -> None:
    sources = snapshot.get("sources")
    if not isinstance(sources, list):
        raise ValueError("snapshot.sources must be a list")
    for raw in cast(list[object], sources):
        if not isinstance(raw, dict):
            raise ValueError("snapshot source entry must be an object")
        source = cast(dict[str, object], raw)
        relative = source.get("relative_path")
        expected = source.get("locked_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("snapshot source path/hash is incomplete")
        path = raw_root / relative
        if not path.is_file() or hash_file(path) != expected:
            raise RuntimeError(f"resume refused: raw source drifted or is missing: {relative}")


def _verify_locked_config(
    project_root: Path,
    run_root: Path,
    manifest: dict[str, object],
) -> None:
    config_hashes_raw = manifest.get("source_hashes")
    if not isinstance(config_hashes_raw, dict):
        raise ValueError("source manifest config hashes are unavailable")
    for relative_raw, expected_raw in cast(dict[object, object], config_hashes_raw).items():
        relative = str(relative_raw)
        expected = str(expected_raw)
        path = (
            run_root / "SNAPSHOT" / "data_snapshot.json"
            if relative == "external/data_snapshot.json"
            else project_root / "config" / relative
        )
        if not path.is_file() or hash_file(path) != expected:
            raise RuntimeError(f"resume refused: locked config/snapshot drifted: {relative}")


def verify_resume_inputs_v5(
    project_root: Path,
    raw_root: Path,
    snapshot: dict[str, object],
    run_root: Path,
) -> dict[str, object] | None:
    """Authorize only the non-analytical P13 process/checkpoint execution patch."""
    _verify_raw_sources(raw_root, snapshot)
    manifest = json_object(run_root / "P00" / "source_config_manifest.json", "source manifest")
    _verify_locked_config(project_root, run_root, manifest)

    source_code_hashes_raw = manifest.get("source_code_hashes")
    if not isinstance(source_code_hashes_raw, dict):
        raise ValueError("source manifest code hashes are unavailable")
    source_code_hashes = {
        str(key): str(value)
        for key, value in cast(dict[object, object], source_code_hashes_raw).items()
    }
    drifted_source_paths = sorted(
        relative
        for relative, expected in source_code_hashes.items()
        if not (project_root / relative).is_file() or hash_file(project_root / relative) != expected
    )
    if not drifted_source_paths:
        return None

    locked_commit = manifest.get("git_commit")
    if locked_commit != P13_PROCESS_COMPATIBILITY_BASE_COMMIT:
        raise RuntimeError(
            "resume refused: P13 process/checkpoint compatibility applies only to runs locked at "
            f"{P13_PROCESS_COMPATIBILITY_BASE_COMMIT}"
        )
    if set(drifted_source_paths) != set(P13_PROCESS_REQUIRED_SOURCE_DRIFT):
        raise RuntimeError(
            "resume refused: source drift exceeds the registered P13 process patch: "
            f"{drifted_source_paths}"
        )

    current_commit = _git_output(project_root, ["rev-parse", "HEAD"])
    _require_git_ancestor(project_root, P13_PROCESS_COMPATIBILITY_BASE_COMMIT, current_commit)
    changed_paths = sorted(
        _git_lines(
            project_root,
            [
                "diff",
                "--name-only",
                f"{P13_PROCESS_COMPATIBILITY_BASE_COMMIT}..{current_commit}",
            ],
        )
    )
    changed_set = set(changed_paths)
    if not P13_PROCESS_REQUIRED_CHANGED_PATHS.issubset(changed_set) or not changed_set.issubset(
        P13_PROCESS_ALLOWED_CHANGED_PATHS
    ):
        raise RuntimeError(
            "resume refused: repository diff exceeds the registered P13 process patch: "
            f"{changed_paths}"
        )
    if not p11_boundary_complete(run_root):
        raise RuntimeError(
            "resume refused: P13 process compatibility requires complete hash-verified P11 outputs"
        )
    if not p12_boundary_complete(run_root):
        raise RuntimeError(
            "resume refused: P13 process compatibility requires complete hash-verified P12 outputs"
        )
    if not p13_resume_boundary_ready(run_root):
        raise RuntimeError(
            "resume refused: P13 outputs exceed the registered pre-refit partial state"
        )

    protocol_hash = (run_root / "P00" / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    drift: dict[str, object] = {
        relative: {
            "locked_sha256": source_code_hashes[relative],
            "current_sha256": hash_file(project_root / relative),
        }
        for relative in drifted_source_paths
    }
    existing = read_compatibility_receipt(run_root, P13_PROCESS_COMPATIBILITY_PATCH_ID)
    if existing is not None:
        if (
            existing.get("locked_git_commit") != locked_commit
            or existing.get("current_git_commit") != current_commit
            or existing.get("changed_paths") != changed_paths
            or existing.get("source_code_drift") != drift
            or existing.get("resume_from") != "P13"
            or existing.get("p11_boundary_verified") is not True
            or existing.get("p12_boundary_verified") is not True
            or existing.get("p13_partial_boundary_verified") is not True
        ):
            raise RuntimeError(
                "resume refused: existing P13 process compatibility receipt does not match current state"
            )
        print(
            "Compatibility resume reused "
            f"patch={P13_PROCESS_COMPATIBILITY_PATCH_ID} resume_from=P13 "
            f"commit={current_commit}",
            flush=True,
        )
        return existing

    receipt: dict[str, object] = {
        "schema_version": 5,
        "patch_id": P13_PROCESS_COMPATIBILITY_PATCH_ID,
        "resume_from": "P13",
        "reason_code": "THREAD_EXECUTION_FAILED_TO_PARALLELIZE_P13_NESTED_REFITS",
        "protocol_hash": protocol_hash,
        "locked_git_commit": locked_commit,
        "current_git_commit": current_commit,
        "changed_paths": changed_paths,
        "source_code_drift": drift,
        "p11_boundary_verified": True,
        "p12_boundary_verified": True,
        "p13_partial_boundary_verified": True,
        "execution_backend_before": "thread",
        "execution_backend_after": "process",
        "checkpoint_grain": "exclusion_id_x_outer_fold",
        "deterministic_task_order_preserved": True,
        "seed_contract_preserved": True,
        "learner_contract_preserved": True,
        "tuning_contract_preserved": True,
        "evaluation_target_contract_preserved": True,
        "analytical_contract_change": False,
        "access_matrix_relaxation": False,
        "registry_lock_modified": False,
    }
    receipt_hash = write_compatibility_receipt(run_root, receipt)
    receipt["receipt_hash"] = receipt_hash
    print(
        "Compatibility resume authorized "
        f"patch={P13_PROCESS_COMPATIBILITY_PATCH_ID} resume_from=P13 "
        f"commit={current_commit}",
        flush=True,
    )
    return receipt
