from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import core.resume as resume
from core.resume import (
    P12_COMPATIBILITY_ALLOWED_PATHS,
    P12_COMPATIBILITY_BASE_COMMIT,
    P12_COMPATIBILITY_PATCH_ID,
    P12_COMPATIBILITY_REQUIRED_PATHS,
    _p11_boundary_complete,
    _write_compatibility_receipt,
    read_compatibility_receipt,
    verify_p12_implementation,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_artifact(
    run_root: Path,
    *,
    artifact_id: str,
    fold_id: str,
    protocol_hash: str,
) -> None:
    target = run_root / "P11" / artifact_id / f"{fold_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({"artifact_id": artifact_id, "fold_id": fold_id}).encode("utf-8")
    target.write_bytes(content)
    manifest = {
        "artifact_id": artifact_id,
        "producer_step": "P11",
        "coordinates": {"outer_fold": fold_id},
        "protocol_hash": protocol_hash,
        "content_hash": _sha256(content),
    }
    target.with_name(target.name + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _p11_fixture(tmp_path: Path) -> tuple[Path, str]:
    run_root = tmp_path / "run"
    p00 = run_root / "P00"
    p00.mkdir(parents=True)
    protocol_hash = "p" * 64
    (p00 / "protocol_hash.txt").write_text(protocol_hash, encoding="utf-8")
    registry: dict[str, object] = {
        "folds": {
            "initial_outer_year": 2020,
            "fully_nested_outer_years": [2021, 2022],
            "initial_in_confirmatory_pool": False,
        },
        "steps": {
            "P11": {
                "writes": ["model_artifacts", "model_freeze_receipt"],
            }
        },
        "artifacts": {
            "model_artifacts": {
                "producer_step": "P11",
                "coordinates": ["outer_fold"],
                "path_template": "P11/model_artifacts/{outer_fold}.json",
            },
            "model_freeze_receipt": {
                "producer_step": "P11",
                "coordinates": ["outer_fold"],
                "path_template": "P11/model_freeze_receipt/{outer_fold}.json",
            },
        },
    }
    (p00 / "registry.lock.json").write_text(json.dumps(registry), encoding="utf-8")
    return run_root, protocol_hash


def test_p12_uses_frozen_selection_hash_without_undeclared_read() -> None:
    source = (ROOT / "scripts" / "p12_evaluate.py").read_text(encoding="utf-8")
    assert 'loaded.context.read("measurement_selection_registry"' not in source
    assert 'freeze.get("measurement_selection_hash")' in source


def test_registered_patch_scope_is_exact_and_nonanalytical() -> None:
    assert P12_COMPATIBILITY_PATCH_ID == "P12_SELECTION_HASH_READ_V1"
    assert P12_COMPATIBILITY_BASE_COMMIT == "09116ea5dea68236f46b2466eb50fbfff5c2bd0a"
    assert P12_COMPATIBILITY_REQUIRED_PATHS == {
        "scripts/p12_evaluate.py",
        "src/core/resume.py",
    }
    assert P12_COMPATIBILITY_REQUIRED_PATHS.issubset(P12_COMPATIBILITY_ALLOWED_PATHS)
    assert "config/foundation/steps.yaml" not in P12_COMPATIBILITY_ALLOWED_PATHS


def test_p11_boundary_requires_every_fold_artifact(tmp_path: Path) -> None:
    run_root, protocol_hash = _p11_fixture(tmp_path)
    for artifact_id in ("model_artifacts", "model_freeze_receipt"):
        _write_artifact(
            run_root,
            artifact_id=artifact_id,
            fold_id="2021",
            protocol_hash=protocol_hash,
        )
    assert not _p11_boundary_complete(run_root)

    for artifact_id in ("model_artifacts", "model_freeze_receipt"):
        _write_artifact(
            run_root,
            artifact_id=artifact_id,
            fold_id="2022",
            protocol_hash=protocol_hash,
        )
    assert _p11_boundary_complete(run_root)

    target = run_root / "P11" / "model_artifacts" / "2022.json"
    target.write_text("tampered", encoding="utf-8")
    assert not _p11_boundary_complete(run_root)


def test_compatibility_receipt_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "P00").mkdir(parents=True)
    protocol_hash = "p" * 64
    (run_root / "P00" / "protocol_hash.txt").write_text(protocol_hash, encoding="utf-8")
    receipt: dict[str, object] = {
        "patch_id": P12_COMPATIBILITY_PATCH_ID,
        "protocol_hash": protocol_hash,
        "locked_git_commit": P12_COMPATIBILITY_BASE_COMMIT,
        "current_git_commit": "new-commit",
    }
    receipt_hash = _write_compatibility_receipt(run_root, receipt)
    observed = read_compatibility_receipt(run_root)
    assert observed is not None
    assert observed["receipt_hash"] == receipt_hash

    receipt_path = run_root / "COMPATIBILITY" / f"{P12_COMPATIBILITY_PATCH_ID}.json"
    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        read_compatibility_receipt(run_root)


def test_p12_direct_execution_requires_matching_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    project_root = tmp_path / "project"
    (run_root / "P00").mkdir(parents=True)
    (project_root / "scripts").mkdir(parents=True)
    protocol_hash = "p" * 64
    current_content = b"patched p12"
    current_hash = _sha256(current_content)
    locked_hash = "a" * 64
    (project_root / "scripts" / "p12_evaluate.py").write_bytes(current_content)
    (run_root / "P00" / "protocol_hash.txt").write_text(protocol_hash, encoding="utf-8")
    (run_root / "P00" / "source_config_manifest.json").write_text(
        json.dumps(
            {
                "source_code_hashes": {"scripts/p12_evaluate.py": locked_hash},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="compatibility resume receipt"):
        verify_p12_implementation(run_root, project_root)

    receipt: dict[str, object] = {
        "patch_id": P12_COMPATIBILITY_PATCH_ID,
        "protocol_hash": protocol_hash,
        "locked_git_commit": P12_COMPATIBILITY_BASE_COMMIT,
        "current_git_commit": "new-commit",
        "source_code_drift": {
            "scripts/p12_evaluate.py": {
                "locked_sha256": locked_hash,
                "current_sha256": current_hash,
            }
        },
    }
    _write_compatibility_receipt(run_root, receipt)
    monkeypatch.setattr(resume, "_git_output", lambda _root, _args: "new-commit")
    observed = verify_p12_implementation(run_root, project_root)
    assert observed is not None
    assert observed["current_git_commit"] == "new-commit"
