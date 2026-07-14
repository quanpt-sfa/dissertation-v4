from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_p00_detached_manifest_hash_and_output_hashes(
    tmp_path: Path,
) -> None:
    run_id = "test-p00"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "p00_lock_protocol.py"),
            "--config",
            str(ROOT / "config" / "pipeline.yaml"),
            "--run-id",
            run_id,
            "--output-root",
            str(tmp_path),
        ],
        check=False,
    )
    assert result.returncode == 0
    directory = tmp_path / run_id / "P00"
    manifest_text = (directory / "job_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    success = json.loads((directory / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["manifest_hash"] == hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    assert "job_manifest.json" not in manifest["output_hashes"]
    for relative, expected in manifest["output_hashes"].items():
        assert hashlib.sha256((directory / relative).read_bytes()).hexdigest() == expected


def test_source_manifest_includes_pipeline_lock_and_source_code(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "p00_lock_protocol.py"),
            "--config",
            str(ROOT / "config" / "pipeline.yaml"),
            "--run-id",
            "test-source",
            "--output-root",
            str(tmp_path),
        ],
        check=False,
    )
    assert result.returncode == 0
    manifest = json.loads(
        (tmp_path / "test-source" / "P00" / "source_config_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "pipeline.yaml" in manifest["source_hashes"]
    assert manifest["package_lock"]["path"] == "uv.lock"
    assert manifest["source_code_hashes"]
    assert "generated_document_hashes" in manifest
