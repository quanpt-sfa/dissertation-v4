from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.resume import artifact_complete, verify_resume_inputs


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_resume_skips_only_a_complete_hash_verified_unit(tmp_path: Path) -> None:
    protocol_hash = "p" * 64
    target = tmp_path / "P10" / "selection" / "2021.json"
    target.parent.mkdir(parents=True)
    content = b'{"selected_measurement":"none"}'
    target.write_bytes(content)
    manifest = {
        "artifact_id": "measurement_selection_registry",
        "producer_step": "P10",
        "coordinates": {"outer_fold": "2021"},
        "protocol_hash": protocol_hash,
        "content_hash": _sha256(content),
    }
    target.with_name(target.name + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    registry: dict[str, object] = {
        "artifacts": {
            "measurement_selection_registry": {
                "producer_step": "P10",
                "coordinates": ["outer_fold"],
                "path_template": "P10/selection/{outer_fold}.json",
            }
        }
    }
    coordinates = {"outer_fold": "2021"}
    assert artifact_complete(
        registry,
        tmp_path,
        protocol_hash,
        "measurement_selection_registry",
        coordinates,
    )
    target.write_bytes(b"tampered")
    assert not artifact_complete(
        registry,
        tmp_path,
        protocol_hash,
        "measurement_selection_registry",
        coordinates,
    )


def test_resume_refuses_raw_source_drift(tmp_path: Path) -> None:
    project = tmp_path / "project"
    raw_root = tmp_path / "raw"
    run_root = tmp_path / "run"
    (run_root / "P00").mkdir(parents=True)
    raw_root.mkdir()
    source = raw_root / "source.csv"
    source.write_bytes(b"locked")
    (run_root / "P00" / "source_config_manifest.json").write_text(
        json.dumps({"source_code_hashes": {}, "source_hashes": {}}), encoding="utf-8"
    )
    snapshot: dict[str, object] = {
        "sources": [{"relative_path": "source.csv", "locked_sha256": _sha256(b"locked")}]
    }
    verify_resume_inputs(project, raw_root, snapshot, run_root)
    source.write_bytes(b"drifted")
    with pytest.raises(RuntimeError, match="raw source drifted"):
        verify_resume_inputs(project, raw_root, snapshot, run_root)
