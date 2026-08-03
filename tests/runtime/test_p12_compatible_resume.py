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
    P12_PAIRED_COMPATIBILITY_COMMIT,
    P12_PAIRED_COMPATIBILITY_PATCH_ID,
    P12_SELECTION_COMPATIBILITY_COMMIT,
    P12_SELECTION_COMPATIBILITY_PATCH_ID,
    P13_CONFIRMATORY_COMPATIBILITY_COMMIT,
    P13_CONFIRMATORY_COMPATIBILITY_PATCH_ID,
    p11_boundary_complete,
    p12_boundary_complete,
    p13_resume_boundary_ready,
    quarantine_partial_p12_outputs,
    read_compatibility_receipt,
    verify_p12_implementation,
    write_compatibility_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _fixed_git_commit(_root: Path, _args: list[str]) -> str:
    return "new-commit"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_artifact(
    run_root: Path,
    *,
    target: Path,
    artifact_id: str,
    producer_step: str,
    coordinates: dict[str, str],
    protocol_hash: str,
    value: dict[str, object] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value or {"artifact_id": artifact_id, **coordinates}).encode("utf-8")
    target.write_bytes(content)
    manifest = {
        "artifact_id": artifact_id,
        "producer_step": producer_step,
        "coordinates": coordinates,
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
        "steps": {"P11": {"writes": ["model_artifacts", "model_freeze_receipt"]}},
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


def _p12_fixture(tmp_path: Path) -> tuple[Path, str]:
    run_root = tmp_path / "run"
    p00 = run_root / "P00"
    p00.mkdir(parents=True)
    protocol_hash = "p" * 64
    (p00 / "protocol_hash.txt").write_text(protocol_hash, encoding="utf-8")
    writes = [
        "outer_open_receipt",
        "calibration_outputs",
        "evaluation_metrics",
        "bootstrap_batches",
        "utility_scenarios",
    ]
    registry: dict[str, object] = {
        "folds": {
            "initial_outer_year": 2020,
            "fully_nested_outer_years": [2021, 2022],
            "initial_in_confirmatory_pool": False,
        },
        "steps": {"P12": {"writes": writes}},
        "artifacts": {
            artifact_id: {
                "producer_step": "P12",
                "coordinates": ["outer_fold"],
                "path_template": f"P12/{artifact_id}/{{outer_fold}}.json",
            }
            for artifact_id in writes
        },
    }
    (p00 / "registry.lock.json").write_text(json.dumps(registry), encoding="utf-8")
    return run_root, protocol_hash


def _p13_fixture(tmp_path: Path) -> tuple[Path, str]:
    run_root = tmp_path / "run"
    p00 = run_root / "P00"
    p00.mkdir(parents=True)
    protocol_hash = "p" * 64
    (p00 / "protocol_hash.txt").write_text(protocol_hash, encoding="utf-8")
    writes = [
        "domain_transfer_outputs",
        "source_sensitivity_outputs",
        "censoring_sensitivity_outputs",
        "hierarchical_pi_sensitivity",
        "ablation_results",
    ]
    registry: dict[str, object] = {
        "steps": {"P13": {"writes": writes}},
        "artifacts": {
            artifact_id: {
                "producer_step": "P13",
                "coordinates": [],
                "path_template": f"P13/{artifact_id}.json",
            }
            for artifact_id in writes
        },
    }
    (p00 / "registry.lock.json").write_text(json.dumps(registry), encoding="utf-8")
    return run_root, protocol_hash


def test_p12_uses_locked_pairs_before_outer_open() -> None:
    source = (ROOT / "scripts" / "p12_evaluate.py").read_text(encoding="utf-8")
    assert 'loaded.context.read("measurement_selection_registry"' not in source
    assert 'freeze.get("measurement_selection_hash")' in source
    assert "reference_by_candidate," in source
    assert source.index("pairing_audit = validate_paired_prediction_keys") < source.index(
        'loaded.context.write("outer_open_receipt"'
    )


def test_registered_patch_scope_is_exact_and_nonanalytical() -> None:
    assert P12_COMPATIBILITY_PATCH_ID == "P13_PRIMARY_TARGET_OUTCOME_V4"
    assert P13_CONFIRMATORY_COMPATIBILITY_PATCH_ID == "P13_CONFIRMATORY_SCOPE_V3"
    assert P13_CONFIRMATORY_COMPATIBILITY_COMMIT == "42c9fa7549002c361464b6cb1285c8a4793c9615"
    assert P12_PAIRED_COMPATIBILITY_PATCH_ID == "P12_PAIRED_SCOPE_V2"
    assert P12_PAIRED_COMPATIBILITY_COMMIT == "67857db9751e9005168c33ee49c956bc7e1340ef"
    assert P12_SELECTION_COMPATIBILITY_PATCH_ID == "P12_SELECTION_HASH_READ_V1"
    assert P12_SELECTION_COMPATIBILITY_COMMIT == "3c261c5a387d83582b9cc3becbad59a6c1a7cae8"
    assert P12_COMPATIBILITY_BASE_COMMIT == "09116ea5dea68236f46b2466eb50fbfff5c2bd0a"
    assert P12_COMPATIBILITY_REQUIRED_PATHS == {
        "scripts/p12_evaluate.py",
        "src/core/resume.py",
        "src/evaluation/pairing.py",
        "scripts/p13_sensitivity.py",
        "scripts/p15_open_known_cases.py",
        "scripts/p16_gate3.py",
        "scripts/p17_build_outputs.py",
        "src/sensitivity/service.py",
    }
    assert P12_COMPATIBILITY_REQUIRED_PATHS.issubset(P12_COMPATIBILITY_ALLOWED_PATHS)
    assert "tests/stages/test_p13_target_outcome_scope.py" in P12_COMPATIBILITY_ALLOWED_PATHS
    assert "config/foundation/steps.yaml" not in P12_COMPATIBILITY_ALLOWED_PATHS
    assert "config/methodology/evaluation.yaml" not in P12_COMPATIBILITY_ALLOWED_PATHS


def test_p11_boundary_requires_every_fold_artifact(tmp_path: Path) -> None:
    run_root, protocol_hash = _p11_fixture(tmp_path)
    for artifact_id in ("model_artifacts", "model_freeze_receipt"):
        _write_artifact(
            run_root,
            target=run_root / "P11" / artifact_id / "2021.json",
            artifact_id=artifact_id,
            producer_step="P11",
            coordinates={"outer_fold": "2021"},
            protocol_hash=protocol_hash,
        )
    assert not p11_boundary_complete(run_root)

    for artifact_id in ("model_artifacts", "model_freeze_receipt"):
        _write_artifact(
            run_root,
            target=run_root / "P11" / artifact_id / "2022.json",
            artifact_id=artifact_id,
            producer_step="P11",
            coordinates={"outer_fold": "2022"},
            protocol_hash=protocol_hash,
        )
    assert p11_boundary_complete(run_root)

    target = run_root / "P11" / "model_artifacts" / "2022.json"
    target.write_text("tampered", encoding="utf-8")
    assert not p11_boundary_complete(run_root)


def test_p12_boundary_requires_every_confirmatory_output(tmp_path: Path) -> None:
    run_root, protocol_hash = _p12_fixture(tmp_path)
    writes = [
        "outer_open_receipt",
        "calibration_outputs",
        "evaluation_metrics",
        "bootstrap_batches",
        "utility_scenarios",
    ]
    for fold_id in ("2021", "2022"):
        for artifact_id in writes:
            _write_artifact(
                run_root,
                target=run_root / "P12" / artifact_id / f"{fold_id}.json",
                artifact_id=artifact_id,
                producer_step="P12",
                coordinates={"outer_fold": fold_id},
                protocol_hash=protocol_hash,
            )
    assert p12_boundary_complete(run_root)

    (run_root / "P12" / "evaluation_metrics" / "2022.json").unlink()
    assert not p12_boundary_complete(run_root)


def test_p13_boundary_allows_only_domain_transfer_prefix(tmp_path: Path) -> None:
    run_root, protocol_hash = _p13_fixture(tmp_path)
    assert p13_resume_boundary_ready(run_root)

    _write_artifact(
        run_root,
        target=run_root / "P13" / "domain_transfer_outputs.json",
        artifact_id="domain_transfer_outputs",
        producer_step="P13",
        coordinates={},
        protocol_hash=protocol_hash,
    )
    assert p13_resume_boundary_ready(run_root)

    _write_artifact(
        run_root,
        target=run_root / "P13" / "source_sensitivity_outputs.json",
        artifact_id="source_sensitivity_outputs",
        producer_step="P13",
        coordinates={},
        protocol_hash=protocol_hash,
    )
    assert not p13_resume_boundary_ready(run_root)


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
    receipt_hash = write_compatibility_receipt(run_root, receipt)
    assert write_compatibility_receipt(run_root, receipt) == receipt_hash
    observed = read_compatibility_receipt(run_root)
    assert observed is not None
    assert observed["receipt_hash"] == receipt_hash

    receipt_path = run_root / "COMPATIBILITY" / f"{P12_COMPATIBILITY_PATCH_ID}.json"
    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        read_compatibility_receipt(run_root)


def test_partial_outer_open_receipts_are_quarantined_idempotently(tmp_path: Path) -> None:
    run_root, protocol_hash = _p12_fixture(tmp_path)
    previous_receipt: dict[str, object] = {
        "patch_id": P12_SELECTION_COMPATIBILITY_PATCH_ID,
        "protocol_hash": protocol_hash,
        "locked_git_commit": P12_COMPATIBILITY_BASE_COMMIT,
        "current_git_commit": P12_SELECTION_COMPATIBILITY_COMMIT,
    }
    previous_hash = write_compatibility_receipt(run_root, previous_receipt)
    previous = read_compatibility_receipt(run_root, P12_SELECTION_COMPATIBILITY_PATCH_ID)
    assert previous is not None
    assert previous["receipt_hash"] == previous_hash

    for fold_id in ("2021", "2022"):
        target = run_root / "P12" / "outer_open_receipt" / f"{fold_id}.json"
        _write_artifact(
            run_root,
            target=target,
            artifact_id="outer_open_receipt",
            producer_step="P12",
            coordinates={"outer_fold": fold_id},
            protocol_hash=protocol_hash,
            value={
                "status": "PASS",
                "protocol_hash": protocol_hash,
                "compatibility_patch_id": P12_SELECTION_COMPATIBILITY_PATCH_ID,
                "compatibility_receipt_hash": previous_hash,
                "evaluation_git_commit": P12_SELECTION_COMPATIBILITY_COMMIT,
            },
        )

    records = quarantine_partial_p12_outputs(run_root, previous)
    assert [record["outer_fold"] for record in records] == ["2021", "2022"]
    for fold_id in ("2021", "2022"):
        original = run_root / "P12" / "outer_open_receipt" / f"{fold_id}.json"
        quarantined = (
            run_root
            / "COMPATIBILITY"
            / "quarantine"
            / P12_PAIRED_COMPATIBILITY_PATCH_ID
            / "P12"
            / "outer_open_receipt"
            / f"{fold_id}.json"
        )
        assert not original.exists()
        assert quarantined.is_file()
        assert quarantined.with_name(quarantined.name + ".manifest.json").is_file()

    repeated = quarantine_partial_p12_outputs(run_root, previous)
    assert repeated == records


def test_p12_direct_execution_requires_matching_latest_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    project_root = tmp_path / "project"
    (run_root / "P00").mkdir(parents=True)
    protocol_hash = "p" * 64
    locked_hashes: dict[str, str] = {}
    current_hashes: dict[str, str] = {}
    for index, relative in enumerate(sorted(P12_COMPATIBILITY_REQUIRED_PATHS)):
        content = f"patched source {index}".encode()
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        locked_hashes[relative] = f"{index + 1}" * 64
        current_hashes[relative] = _sha256(content)
    (run_root / "P00" / "protocol_hash.txt").write_text(protocol_hash, encoding="utf-8")
    (run_root / "P00" / "source_config_manifest.json").write_text(
        json.dumps({"source_code_hashes": locked_hashes}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="compatibility resume receipt"):
        verify_p12_implementation(run_root, project_root)

    receipt: dict[str, object] = {
        "patch_id": P12_COMPATIBILITY_PATCH_ID,
        "protocol_hash": protocol_hash,
        "locked_git_commit": P12_COMPATIBILITY_BASE_COMMIT,
        "current_git_commit": "new-commit",
        "source_code_drift": {
            relative: {
                "locked_sha256": locked_hashes[relative],
                "current_sha256": current_hashes[relative],
            }
            for relative in P12_COMPATIBILITY_REQUIRED_PATHS
        },
    }
    write_compatibility_receipt(run_root, receipt)
    monkeypatch.setattr(resume, "_git_output", _fixed_git_commit)
    observed = verify_p12_implementation(run_root, project_root)
    assert observed is not None
    assert observed["current_git_commit"] == "new-commit"
