"""Hash-verified helpers for resuming immutable pipeline runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

P12_COMPATIBILITY_PATCH_ID = "P12_SELECTION_HASH_READ_V1"
P12_COMPATIBILITY_BASE_COMMIT = "09116ea5dea68236f46b2466eb50fbfff5c2bd0a"
P12_COMPATIBILITY_REQUIRED_PATHS = frozenset(
    {
        "scripts/p12_evaluate.py",
        "src/core/resume.py",
    }
)
P12_COMPATIBILITY_ALLOWED_PATHS = frozenset(
    {
        *P12_COMPATIBILITY_REQUIRED_PATHS,
        "tests/runtime/test_p12_compatible_resume.py",
    }
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path, context: str) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be an object")
    return cast(dict[str, object], raw)


def artifact_complete(
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifact_id: str,
    coordinates: dict[str, str],
) -> bool:
    raw_artifacts = registry.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        raise ValueError("locked artifact catalog is unavailable")
    raw = cast(dict[str, object], raw_artifacts).get(artifact_id)
    if not isinstance(raw, dict):
        raise ValueError(f"artifact={artifact_id}: absent from locked catalog")
    artifact = cast(dict[str, object], raw)
    expected_coordinates = artifact.get("coordinates")
    if not isinstance(expected_coordinates, list) or set(coordinates) != {
        str(value) for value in cast(list[object], expected_coordinates)
    }:
        raise ValueError(f"artifact={artifact_id}: runner coordinates do not match catalog")
    template = artifact.get("path_template")
    if not isinstance(template, str):
        raise ValueError(f"artifact={artifact_id}: path template required")
    target = run_root / template.format(**coordinates)
    manifest_path = target.with_name(target.name + ".manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        return False
    manifest = json_object(manifest_path, f"artifact={artifact_id} manifest")
    expected = {
        "artifact_id": artifact_id,
        "producer_step": artifact.get("producer_step"),
        "coordinates": coordinates,
        "protocol_hash": protocol_hash,
    }
    return all(manifest.get(key) == value for key, value in expected.items()) and manifest.get(
        "content_hash"
    ) == hash_file(target)


def verify_resume_inputs(
    project_root: Path,
    raw_root: Path,
    snapshot: dict[str, object],
    run_root: Path,
) -> dict[str, object] | None:
    """Verify a strict resume or authorize the registered P12 compatibility continuation."""
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

    manifest = json_object(run_root / "P00" / "source_config_manifest.json", "source manifest")
    source_code_hashes_raw = manifest.get("source_code_hashes")
    if not isinstance(source_code_hashes_raw, dict):
        raise ValueError("source manifest code hashes are unavailable")
    source_code_hashes = cast(dict[object, object], source_code_hashes_raw)
    drifted_source_paths: list[str] = []
    for relative_raw, expected_raw in source_code_hashes.items():
        relative = str(relative_raw)
        expected = str(expected_raw)
        path = project_root / relative
        if not path.is_file() or hash_file(path) != expected:
            drifted_source_paths.append(relative)

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

    if not drifted_source_paths:
        return None
    return _authorize_p12_compatibility_resume(
        project_root=project_root,
        run_root=run_root,
        manifest=manifest,
        source_code_hashes={str(key): str(value) for key, value in source_code_hashes.items()},
        drifted_source_paths=sorted(drifted_source_paths),
    )


def verify_p12_implementation(run_root: Path, project_root: Path) -> dict[str, object] | None:
    """Require a valid compatibility receipt when P12 differs from the locked source tree."""
    manifest = json_object(run_root / "P00" / "source_config_manifest.json", "source manifest")
    hashes_raw = manifest.get("source_code_hashes")
    if not isinstance(hashes_raw, dict):
        raise RuntimeError("P12 source manifest code hashes are unavailable")
    hashes = cast(dict[object, object], hashes_raw)
    expected_raw = hashes.get("scripts/p12_evaluate.py")
    if not isinstance(expected_raw, str):
        raise RuntimeError("P12 locked implementation hash is unavailable")
    current_hash = hash_file(project_root / "scripts" / "p12_evaluate.py")
    if current_hash == expected_raw:
        return None

    receipt = read_compatibility_receipt(run_root)
    if receipt is None:
        raise RuntimeError(
            "P12 implementation drift requires a verified compatibility resume receipt; "
            "run scripts/run_pipeline.py with --resume"
        )
    drift_raw = receipt.get("source_code_drift")
    if not isinstance(drift_raw, dict):
        raise RuntimeError("P12 compatibility receipt source drift is unavailable")
    drift = cast(dict[object, object], drift_raw)
    p12_raw = drift.get("scripts/p12_evaluate.py")
    if not isinstance(p12_raw, dict):
        raise RuntimeError("P12 compatibility receipt does not bind the evaluation script")
    p12 = cast(dict[object, object], p12_raw)
    if p12.get("locked_sha256") != expected_raw or p12.get("current_sha256") != current_hash:
        raise RuntimeError("P12 compatibility receipt implementation hash mismatch")
    current_commit = _git_output(project_root, ["rev-parse", "HEAD"])
    if receipt.get("current_git_commit") != current_commit:
        raise RuntimeError("P12 compatibility receipt current commit mismatch")
    return receipt


def read_compatibility_receipt(run_root: Path) -> dict[str, object] | None:
    directory = run_root / "COMPATIBILITY"
    receipt_path = directory / f"{P12_COMPATIBILITY_PATCH_ID}.json"
    hash_path = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    if not receipt_path.exists() and not hash_path.exists():
        return None
    if not receipt_path.is_file() or not hash_path.is_file():
        raise RuntimeError("compatibility receipt and hash sidecar are both required")
    expected_hash = hash_path.read_text(encoding="utf-8").strip()
    observed_hash = hash_file(receipt_path)
    if expected_hash != observed_hash:
        raise RuntimeError("compatibility receipt hash mismatch")
    receipt = json_object(receipt_path, "compatibility receipt")
    if receipt.get("patch_id") != P12_COMPATIBILITY_PATCH_ID:
        raise RuntimeError("compatibility receipt patch id mismatch")
    protocol_hash = (run_root / "P00" / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    if receipt.get("protocol_hash") != protocol_hash:
        raise RuntimeError("compatibility receipt protocol hash mismatch")
    receipt["receipt_hash"] = observed_hash
    return receipt


def _authorize_p12_compatibility_resume(
    *,
    project_root: Path,
    run_root: Path,
    manifest: dict[str, object],
    source_code_hashes: dict[str, str],
    drifted_source_paths: list[str],
) -> dict[str, object]:
    locked_commit = manifest.get("git_commit")
    if locked_commit != P12_COMPATIBILITY_BASE_COMMIT:
        raise RuntimeError(
            "resume refused: implementation drift is not covered by a registered compatibility patch"
        )
    if set(drifted_source_paths) != set(P12_COMPATIBILITY_REQUIRED_PATHS):
        raise RuntimeError(
            "resume refused: source drift exceeds the registered P12 compatibility patch: "
            f"{drifted_source_paths}"
        )

    current_commit = _git_output(project_root, ["rev-parse", "HEAD"])
    _require_git_ancestor(project_root, P12_COMPATIBILITY_BASE_COMMIT, current_commit)
    changed_paths = sorted(
        _git_lines(project_root, ["diff", "--name-only", f"{locked_commit}..{current_commit}"])
    )
    changed_set = set(changed_paths)
    if not P12_COMPATIBILITY_REQUIRED_PATHS.issubset(changed_set) or not changed_set.issubset(
        P12_COMPATIBILITY_ALLOWED_PATHS
    ):
        raise RuntimeError(
            "resume refused: repository diff exceeds the registered P12 compatibility patch: "
            f"{changed_paths}"
        )
    if not p11_boundary_complete(run_root):
        raise RuntimeError(
            "resume refused: registered P12 compatibility patch requires every P11 fold artifact "
            "to be complete and hash-verified"
        )

    drift: dict[str, object] = {}
    for relative in drifted_source_paths:
        current_path = project_root / relative
        drift[relative] = {
            "locked_sha256": source_code_hashes[relative],
            "current_sha256": hash_file(current_path),
        }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "patch_id": P12_COMPATIBILITY_PATCH_ID,
        "resume_from": "P12",
        "reason_code": "REDUNDANT_UNDECLARED_MEASUREMENT_SELECTION_READ",
        "protocol_hash": (run_root / "P00" / "protocol_hash.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "locked_git_commit": locked_commit,
        "current_git_commit": current_commit,
        "changed_paths": changed_paths,
        "source_code_drift": drift,
        "p11_boundary_verified": True,
        "analytical_contract_change": False,
        "access_matrix_relaxation": False,
        "registry_lock_modified": False,
    }
    receipt_hash = write_compatibility_receipt(run_root, receipt)
    receipt["receipt_hash"] = receipt_hash
    print(
        "Compatibility resume authorized "
        f"patch={P12_COMPATIBILITY_PATCH_ID} resume_from=P12 commit={current_commit}",
        flush=True,
    )
    return receipt


def p11_boundary_complete(run_root: Path) -> bool:
    registry = json_object(run_root / "P00" / "registry.lock.json", "locked registry")
    protocol_hash = (run_root / "P00" / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    folds_raw = registry.get("folds")
    steps_raw = registry.get("steps")
    artifacts_raw = registry.get("artifacts")
    if (
        not isinstance(folds_raw, dict)
        or not isinstance(steps_raw, dict)
        or not isinstance(artifacts_raw, dict)
    ):
        return False
    folds = cast(dict[object, object], folds_raw)
    nested_raw = folds.get("fully_nested_outer_years")
    if not isinstance(nested_raw, list):
        return False
    confirmatory_folds = [str(value) for value in cast(list[object], nested_raw)]
    if folds.get("initial_in_confirmatory_pool") is True:
        initial = folds.get("initial_outer_year")
        if initial is None or isinstance(initial, bool):
            return False
        confirmatory_folds.insert(0, str(initial))
    if not confirmatory_folds:
        return False

    p11_raw = cast(dict[object, object], steps_raw).get("P11")
    if not isinstance(p11_raw, dict):
        return False
    writes_raw = cast(dict[object, object], p11_raw).get("writes")
    if not isinstance(writes_raw, list):
        return False
    artifact_map = cast(dict[object, object], artifacts_raw)
    for artifact_raw in cast(list[object], writes_raw):
        artifact_id = str(artifact_raw)
        spec_raw = artifact_map.get(artifact_id)
        if not isinstance(spec_raw, dict):
            return False
        coordinates_raw = cast(dict[object, object], spec_raw).get("coordinates")
        if coordinates_raw == ["outer_fold"]:
            for fold_id in confirmatory_folds:
                if not artifact_complete(
                    registry,
                    run_root,
                    protocol_hash,
                    artifact_id,
                    {"outer_fold": fold_id},
                ):
                    return False
        elif coordinates_raw == []:
            if not artifact_complete(registry, run_root, protocol_hash, artifact_id, {}):
                return False
        else:
            return False
    return True


def write_compatibility_receipt(run_root: Path, receipt: dict[str, object]) -> str:
    directory = run_root / "COMPATIBILITY"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{P12_COMPATIBILITY_PATCH_ID}.json"
    hash_target = target.with_suffix(target.suffix + ".sha256")
    rendered = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    encoded = rendered.encode("utf-8")
    receipt_hash = hashlib.sha256(encoded).hexdigest()
    if target.exists() or hash_target.exists():
        if not target.is_file() or not hash_target.is_file():
            raise RuntimeError("compatibility receipt publication is incomplete")
        if (
            target.read_bytes() != encoded
            or hash_target.read_text(encoding="utf-8").strip() != receipt_hash
        ):
            raise RuntimeError("compatibility receipt already exists with different content")
        return receipt_hash
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary_hash = hash_target.with_suffix(hash_target.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary_hash.write_text(receipt_hash + "\n", encoding="utf-8")
    temporary.replace(target)
    temporary_hash.replace(hash_target)
    return receipt_hash


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
