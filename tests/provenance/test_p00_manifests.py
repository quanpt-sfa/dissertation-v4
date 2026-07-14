from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _snapshot(path: Path, snapshot_id: str) -> Path:
    snapshot: dict[str, object] = {
        "snapshot_schema_version": 1,
        "snapshot_id": snapshot_id,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "root_environment_variable": "DISSERTATION_RAW_ROOT",
        "raw_root_recorded": False,
        "profiles": {"fixture": {"matched_file_count": 1}},
        "sources": [
            {
                "source_id": "fixture",
                "profile_id": "fixture",
                "enabled": True,
                "channel_id": "S1",
                "source_type": "fixture",
                "source_agency": "tests",
                "original_unit": "firm-year",
                "related_period_field": "fiscal_year",
                "availability_date_field": "availability_date",
                "availability_date_source": "fixture",
                "coverage_dimensions": ["year"],
                "role": "predictor",
                "verification_status": "observed",
                "data_risks": [],
                "relative_path": "fixture.parquet",
                "format": "parquet",
                "encoding": "utf-8",
                "delimiter": None,
                "sheet_name": None,
                "header_row": None,
                "locked_sha256": "0" * 64,
                "file_size_bytes": 0,
                "row_count_snapshot": 1,
                "schema_hash": "1" * 64,
                "resolved_semantics": {
                    "firm_id": "firm_id",
                    "fiscal_year": "fiscal_year",
                    "availability_date": "availability_date",
                },
                "schema": {
                    "required_columns": ["firm_id", "fiscal_year", "availability_date"],
                    "optional_columns": [],
                    "key_columns": ["firm_id", "fiscal_year"],
                    "date_columns": ["availability_date"],
                    "required_date_columns": ["availability_date"],
                    "numeric_columns": {},
                    "allow_extra_columns": False,
                    "key_unique": True,
                    "row_count_min": 1,
                },
                "panel_mapping": {
                    "enabled": True,
                    "firm_id_field": "firm_id",
                    "fiscal_year_field": "fiscal_year",
                    "availability_date_field": "availability_date",
                    "fiscal_year_end_field": None,
                    "ticker_field": None,
                    "contributes_to_firm_master": True,
                    "core_predictor": True,
                },
            }
        ],
    }
    payload = dict(snapshot)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    snapshot["snapshot_hash"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    return path


def test_p00_detached_manifest_hash_and_output_hashes(
    tmp_path: Path,
) -> None:
    run_id = "test-p00"
    snapshot = _snapshot(tmp_path / "snapshot.json", run_id)
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
            "--snapshot-manifest",
            str(snapshot),
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
    snapshot = _snapshot(tmp_path / "snapshot-source.json", "test-source")
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
            "--snapshot-manifest",
            str(snapshot),
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
