"""Hash-verified helpers for resuming immutable pipeline runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import cast

P12_SELECTION_COMPATIBILITY_PATCH_ID = "P12_SELECTION_HASH_READ_V1"
P12_SELECTION_COMPATIBILITY_COMMIT = "3c261c5a387d83582b9cc3becbad59a6c1a7cae8"
P12_COMPATIBILITY_PATCH_ID = "P12_PAIRED_SCOPE_V2"
P12_COMPATIBILITY_BASE_COMMIT = "09116ea5dea68236f46b2466eb50fbfff5c2bd0a"
P12_COMPATIBILITY_REQUIRED_PATHS = frozenset(
    {
        "scripts/p12_evaluate.py",
        "src/core/resume.py",
        "src/evaluation/pairing.py",
    }
)
P12_COMPATIBILITY_ALLOWED_PATHS = frozenset(
    {
        *P12_COMPATIBILITY_REQUIRED_PATHS,
        "tests/runtime/test_p12_compatible_resume.py",
        "tests/stages/test_pipeline_audit_remediation.py",
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


def _artifact_target(
    registry: dict[str, object],
    run_root: Path,
    artifact_id: str,
    coordinates: dict[str, str],
) -> Path:
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
    return run_root / template.format(**coordinates)


def artifact_complete(
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifact_id: str,
    coordinates: dict[str, str],
) -> bool:
    target = _artifact_target(registry, run_root, artifact_id, coordinates)
    manifest_path = target.with_name(target.name + ".manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        return False
    raw_artifacts = cast(dict[str, object], registry["artifacts"])
    artifact = cast(dict[str, object], raw_artifacts[artifact_id])
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
    """Require a current compatibility receipt when P12-related code differs from the lock."""
    manifest = json_object(run_root / "P00" / "source_config_manifest.json", "source manifest")
    hashes_raw = manifest.get("source_code_hashes")
    if not isinstance(hashes_raw, dict):
        raise RuntimeError("P12 source manifest code hashes are unavailable")
    hashes = {str(key): str(value) for key, value in cast(dict[object, object], hashes_raw).items()}
    drifted = [
        relative
        for relative in P12_COMPATIBILITY_REQUIRED_PATHS
        if relative not in hashes
        or not (project_root / relative).is_file()
        or hash_file(project_root / relative) != hashes[relative]
    ]
    if not drifted:
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
    if set(str(key) for key in drift) != set(P12_COMPATIBILITY_REQUIRED_PATHS):
        raise RuntimeError("P12 compatibility receipt drift scope mismatch")
    for relative in P12_COMPATIBILITY_REQUIRED_PATHS:
        entry_raw = drift.get(relative)
        if not isinstance(entry_raw, dict):
            raise RuntimeError(f"P12 compatibility receipt does not bind {relative}")
        entry = cast(dict[object, object], entry_raw)
        current_hash = hash_file(project_root / relative)
        if (
            entry.get("locked_sha256") != hashes.get(relative)
            or entry.get("current_sha256") != current_hash
        ):
            raise RuntimeError(
                f"P12 compatibility receipt implementation hash mismatch: {relative}"
            )
    current_commit = _git_output(project_root, ["rev-parse", "HEAD"])
    if receipt.get("current_git_commit") != current_commit:
        raise RuntimeError("P12 compatibility receipt current commit mismatch")
    return receipt


def read_compatibility_receipt(
    run_root: Path,
    patch_id: str = P12_COMPATIBILITY_PATCH_ID,
) -> dict[str, object] | None:
    directory = run_root / "COMPATIBILITY"
    receipt_path = directory / f"{patch_id}.json"
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
    if receipt.get("patch_id") != patch_id:
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

    previous = read_compatibility_receipt(run_root, P12_SELECTION_COMPATIBILITY_PATCH_ID)
    previous_hash: str | None = None
    if previous is not None:
        if (
            previous.get("locked_git_commit") != P12_COMPATIBILITY_BASE_COMMIT
            or previous.get("current_git_commit") != P12_SELECTION_COMPATIBILITY_COMMIT
        ):
            raise RuntimeError(
                "resume refused: prior P12 compatibility receipt does not match its lock"
            )
        previous_hash = str(previous["receipt_hash"])

    quarantined = quarantine_partial_p12_outputs(run_root, previous)
    drift: dict[str, object] = {}
    for relative in drifted_source_paths:
        current_path = project_root / relative
        drift[relative] = {
            "locked_sha256": source_code_hashes[relative],
            "current_sha256": hash_file(current_path),
        }
    receipt: dict[str, object] = {
        "schema_version": 2,
        "patch_id": P12_COMPATIBILITY_PATCH_ID,
        "resume_from": "P12",
        "reason_code": "PAIRING_SCOPE_INFERRED_FROM_SUFFIX_INSTEAD_OF_LOCKED_GATE2_PAIRS",
        "protocol_hash": (run_root / "P00" / "protocol_hash.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "locked_git_commit": locked_commit,
        "current_git_commit": current_commit,
        "changed_paths": changed_paths,
        "source_code_drift": drift,
        "previous_patch_id": P12_SELECTION_COMPATIBILITY_PATCH_ID if previous else None,
        "previous_receipt_hash": previous_hash,
        "quarantined_partial_outputs": quarantined,
        "p11_boundary_verified": True,
        "pairing_scope_source": "evaluation.gate2.reference_by_candidate",
        "pu_branch_preserved_separate": True,
        "analytical_contract_change": False,
        "access_matrix_relaxation": False,
        "registry_lock_modified": False,
    }
    receipt_hash = write_compatibility_receipt(run_root, receipt)
    receipt["receipt_hash"] = receipt_hash
    print(
        "Compatibility resume authorized "
        f"patch={P12_COMPATIBILITY_PATCH_ID} resume_from=P12 commit={current_commit} "
        f"quarantined={len(quarantined)}",
        flush=True,
    )
    return receipt


def _confirmatory_folds(registry: dict[str, object]) -> list[str]:
    folds_raw = registry.get("folds")
    if not isinstance(folds_raw, dict):
        raise ValueError("locked fold registry is unavailable")
    folds = cast(dict[object, object], folds_raw)
    nested_raw = folds.get("fully_nested_outer_years")
    if not isinstance(nested_raw, list):
        raise ValueError("fully nested outer years are unavailable")
    result = [str(value) for value in cast(list[object], nested_raw)]
    if folds.get("initial_in_confirmatory_pool") is True:
        initial = folds.get("initial_outer_year")
        if initial is None or isinstance(initial, bool):
            raise ValueError("initial outer year is unavailable")
        result.insert(0, str(initial))
    if not result:
        raise ValueError("confirmatory folds are unavailable")
    return result


def quarantine_partial_p12_outputs(
    run_root: Path,
    previous_receipt: dict[str, object] | None,
) -> list[dict[str, object]]:
    """Quarantine only the known receipt-only residue from the failed P12 preflight."""
    registry = json_object(run_root / "P00" / "registry.lock.json", "locked registry")
    protocol_hash = (run_root / "P00" / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    steps_raw = registry.get("steps")
    if not isinstance(steps_raw, dict):
        raise RuntimeError("P12 quarantine requires the locked step registry")
    p12_raw = cast(dict[str, object], steps_raw).get("P12")
    if not isinstance(p12_raw, dict):
        raise RuntimeError("P12 quarantine requires the locked P12 contract")
    writes_raw = cast(dict[str, object], p12_raw).get("writes")
    if not isinstance(writes_raw, list):
        raise RuntimeError("P12 quarantine requires the locked P12 writes")
    writes = [str(value) for value in cast(list[object], writes_raw)]
    records: list[dict[str, object]] = []
    quarantine_root = run_root / "COMPATIBILITY" / "quarantine" / P12_COMPATIBILITY_PATCH_ID

    for fold_id in _confirmatory_folds(registry):
        coordinates = {"outer_fold": fold_id}
        present: list[str] = []
        complete: list[str] = []
        for artifact_id in writes:
            target = _artifact_target(registry, run_root, artifact_id, coordinates)
            manifest_path = target.with_name(target.name + ".manifest.json")
            if target.exists() or manifest_path.exists():
                present.append(artifact_id)
            if artifact_complete(
                registry,
                run_root,
                protocol_hash,
                artifact_id,
                coordinates,
            ):
                complete.append(artifact_id)
        if len(complete) == len(writes):
            continue
        if not present:
            existing = _existing_quarantine_record(
                run_root,
                quarantine_root,
                registry,
                fold_id,
                protocol_hash,
            )
            if existing is not None:
                records.append(existing)
            continue
        if present != ["outer_open_receipt"] or complete != ["outer_open_receipt"]:
            raise RuntimeError(
                "resume refused: P12 has an unknown partial-output state: "
                f"fold={fold_id}, present={present}, complete={complete}"
            )

        target = _artifact_target(registry, run_root, "outer_open_receipt", coordinates)
        manifest_path = target.with_name(target.name + ".manifest.json")
        value = json_object(target, f"fold={fold_id} partial outer-open receipt")
        if previous_receipt is not None:
            if (
                value.get("compatibility_patch_id") != P12_SELECTION_COMPATIBILITY_PATCH_ID
                or value.get("compatibility_receipt_hash") != previous_receipt.get("receipt_hash")
                or value.get("evaluation_git_commit") != P12_SELECTION_COMPATIBILITY_COMMIT
            ):
                raise RuntimeError(
                    f"resume refused: fold={fold_id} partial P12 receipt is not from the known failure"
                )
        destination = quarantine_root / target.relative_to(run_root)
        destination_manifest = quarantine_root / manifest_path.relative_to(run_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_manifest.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination_manifest.exists():
            raise RuntimeError(f"resume refused: fold={fold_id} P12 quarantine destination exists")
        target_hash = hash_file(target)
        manifest_hash = hash_file(manifest_path)
        target.replace(destination)
        manifest_path.replace(destination_manifest)
        records.append(
            {
                "outer_fold": fold_id,
                "artifact_id": "outer_open_receipt",
                "original_path": target.relative_to(run_root).as_posix(),
                "quarantine_path": destination.relative_to(run_root).as_posix(),
                "content_hash": target_hash,
                "manifest_hash": manifest_hash,
                "reason_code": "P12_PREFLIGHT_FAILED_AFTER_PREMATURE_OUTER_OPEN_RECEIPT",
            }
        )
    return records


def _existing_quarantine_record(
    run_root: Path,
    quarantine_root: Path,
    registry: dict[str, object],
    fold_id: str,
    protocol_hash: str,
) -> dict[str, object] | None:
    coordinates = {"outer_fold": fold_id}
    original = _artifact_target(registry, run_root, "outer_open_receipt", coordinates)
    original_manifest = original.with_name(original.name + ".manifest.json")
    target = quarantine_root / original.relative_to(run_root)
    manifest_path = quarantine_root / original_manifest.relative_to(run_root)
    if not target.exists() and not manifest_path.exists():
        return None
    if not target.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"resume refused: fold={fold_id} P12 quarantine is incomplete")
    manifest = json_object(manifest_path, f"fold={fold_id} quarantined manifest")
    if (
        manifest.get("artifact_id") != "outer_open_receipt"
        or manifest.get("producer_step") != "P12"
        or manifest.get("coordinates") != coordinates
        or manifest.get("protocol_hash") != protocol_hash
        or manifest.get("content_hash") != hash_file(target)
    ):
        raise RuntimeError(f"resume refused: fold={fold_id} quarantined P12 receipt is invalid")
    return {
        "outer_fold": fold_id,
        "artifact_id": "outer_open_receipt",
        "original_path": original.relative_to(run_root).as_posix(),
        "quarantine_path": target.relative_to(run_root).as_posix(),
        "content_hash": hash_file(target),
        "manifest_hash": hash_file(manifest_path),
        "reason_code": "P12_PREFLIGHT_FAILED_AFTER_PREMATURE_OUTER_OPEN_RECEIPT",
    }


def p11_boundary_complete(run_root: Path) -> bool:
    registry = json_object(run_root / "P00" / "registry.lock.json", "locked registry")
    protocol_hash = (run_root / "P00" / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    steps_raw = registry.get("steps")
    artifacts_raw = registry.get("artifacts")
    if not isinstance(steps_raw, dict) or not isinstance(artifacts_raw, dict):
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
            for fold_id in _confirmatory_folds(registry):
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
    patch_id = receipt.get("patch_id")
    if not isinstance(patch_id, str) or not patch_id:
        raise ValueError("compatibility receipt patch_id is required")
    directory = run_root / "COMPATIBILITY"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{patch_id}.json"
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
