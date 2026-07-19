"""Run one fail-closed production workflow from P00 through P17 with L3 required."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping  # noqa: E402


def _run(command: list[str], *, cwd: Path, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=True,
    )
    if capture:
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
        return result.stdout
    return ""


def _require_clean_tree(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError("final production run requires a clean committed Git tree")


def _load_yaml(path: Path) -> dict[str, Any]:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return raw


def _load_json(path: Path) -> dict[str, Any]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"JSON object required: {path}")
    return raw


def _parse_prefixed_json(output: str, prefix: str) -> dict[str, Any]:
    matches = [line[len(prefix) :] for line in output.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {prefix} line, found {len(matches)}")
    raw: object = json.loads(matches[0])
    if not isinstance(raw, dict):
        raise ValueError(f"{prefix} payload must be a JSON object")
    return raw


def _validate_locked_config(config_path: Path) -> None:
    pipeline = _load_yaml(config_path)
    imports = pipeline.get("imports")
    if not isinstance(imports, list):
        raise ValueError("pipeline config imports are required")
    measurement_path: Path | None = None
    for value in imports:
        candidate = (config_path.parent / str(value)).resolve()
        if candidate.name == "measurement.yaml":
            measurement_path = candidate
            break
    if measurement_path is None:
        measurement_path = config_path.parent / "methodology" / "measurement.yaml"
    raw = _load_yaml(measurement_path)
    measurement = raw.get("measurement")
    if not isinstance(measurement, dict):
        raise ValueError("measurement mapping is required")
    if measurement.get("primary_s3_endpoint") != "S3_CONTENT":
        raise ValueError("final run requires primary_s3_endpoint=S3_CONTENT")
    source_set_id = measurement.get("primary_source_set_id")
    source_sets = measurement.get("source_sets")
    if not isinstance(source_set_id, str) or not isinstance(source_sets, dict):
        raise ValueError("primary source set is not locked")
    primary = source_sets.get(source_set_id)
    if not isinstance(primary, dict) or not isinstance(primary.get("sources"), list):
        raise ValueError("primary source set is malformed")
    s3_sources = sorted(str(value) for value in primary["sources"] if str(value).startswith("S3_"))
    if s3_sources != ["S3_CONTENT"]:
        raise ValueError("final primary source set must contain S3_CONTENT and no other S3 endpoint")
    l3 = measurement.get("l3_model")
    operational = l3.get("operational") if isinstance(l3, dict) else None
    if not isinstance(operational, dict):
        raise ValueError("L3 operational configuration is required")
    grid = operational.get("fixed_pi_grid")
    priors = operational.get("accuracy_priors_by_profile")
    lock = operational.get("parameter_lock")
    if not isinstance(grid, list) or not grid:
        raise ValueError("fixed_pi_grid must be locked before final run")
    if not isinstance(priors, dict) or not priors:
        raise ValueError("accuracy_priors_by_profile must be locked before final run")
    if not isinstance(lock, dict) or lock.get("status") != "LOCKED":
        raise ValueError("L3 parameter_lock receipt is required")
    if lock.get("outer_outcomes_accessed") is not False or lock.get("known_cases_accessed") is not False:
        raise ValueError("L3 lock receipt violates sealed-evidence constraints")


def _audit_blockers(s3: dict[str, Any], calibration: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in (
        "p03_p05_outcome_mismatch_count",
        "p03_p05_missing_key_count",
        "eligible_unresolved_sanction_year_mapping_count",
        "sanction_year_unresolved_firm_year_count",
    ):
        if int(s3.get(key) or 0) != 0:
            result.append(key.upper())
    positives = s3.get("development_positive_count_by_endpoint")
    if not isinstance(positives, dict) or int(positives.get("S3_CONTENT") or 0) <= 0:
        result.append("NO_DEVELOPMENT_S3_CONTENT_POSITIVES")
    if s3.get("outer_outcomes_accessed") is not False:
        result.append("S3_AUDIT_OUTER_OUTCOME_ACCESS")
    if calibration.get("outer_outcomes_accessed") is not False:
        result.append("CALIBRATION_OUTER_OUTCOME_ACCESS")
    if calibration.get("l2_status") != "AVAILABLE":
        result.append("L2_NOT_AVAILABLE")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    config = args.config.resolve()
    project_root = config.parent.parent
    _require_clean_tree(project_root)
    _validate_locked_config(config)
    python = sys.executable
    if not args.skip_tests:
        _run([python, "-m", "pytest", "-q"], cwd=project_root)

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / args.run_id
    base_command = [
        python,
        "scripts/run_pipeline.py",
        "--run-id",
        args.run_id,
        "--raw-root",
        str(raw_root),
        "--output-root",
        str(output_root),
        "--config",
        str(config),
    ]
    _run([*base_command, "--through", "P06"], cwd=project_root)

    registry = run_root / "P00" / "registry.lock.json"
    s3_dir = run_root / "S3_AUDIT"
    calibration_dir = run_root / "CALIBRATION"
    _run(
        [
            python,
            "scripts/report_s3_year_audit.py",
            "--registry",
            str(registry),
            "--run-id",
            args.run_id,
            "--output-dir",
            str(s3_dir),
        ],
        cwd=project_root,
    )
    calibration_output = _run(
        [
            python,
            "scripts/report_measurement_calibration.py",
            "--registry",
            str(registry),
            "--run-id",
            args.run_id,
            "--output-dir",
            str(calibration_dir),
        ],
        cwd=project_root,
        capture=True,
    )
    calibration = _parse_prefixed_json(calibration_output, "MEASUREMENT_CALIBRATION_JSON=")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "measurement_calibration_summary.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    s3 = _load_json(s3_dir / "s3_year_audit_summary.json")
    blockers = _audit_blockers(s3, calibration)

    p06 = load_run(registry_path=registry, run_id=args.run_id, step_id="P06", state="MEASURED")
    capability = mapping(p06.context.read("l3_pilot_capability", {}), "L3 capability")
    matrices = mapping(p06.context.read("source_channel_matrices", {}), "source matrices")
    if capability.get("status") != "AVAILABLE" or capability.get("pilot_executed") is not True:
        blockers.append(f"L3_PILOT_{capability.get('status', 'UNKNOWN')}_{capability.get('reason_code')}")
    primary_sources = matrices.get("primary_measurement_sources")
    if not isinstance(primary_sources, list):
        blockers.append("PRIMARY_MEASUREMENT_SOURCE_RECEIPT_MISSING")
    else:
        s3_sources = sorted(str(value) for value in primary_sources if str(value).startswith("S3_"))
        if s3_sources != ["S3_CONTENT"]:
            blockers.append("PRIMARY_MEASUREMENT_S3_ENDPOINT_DRIFT")

    preflight_dir = run_root / "PREFLIGHT"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "run_id": args.run_id,
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "protocol_hash": s3.get("protocol_hash"),
        "l3_capability": capability,
        "primary_measurement_sources": primary_sources,
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
    }
    (preflight_dir / "l3_preflight_receipt.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if blockers:
        print("L3_PREFLIGHT_JSON=" + json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 4

    p10_output = _run(
        [*base_command, "--through", "P10", "--resume"], cwd=project_root, capture=True
    )
    p10_lines = [line for line in p10_output.splitlines() if line.startswith("P10 status=PASS")]
    if not p10_lines or any("L3_fixed_pi=AVAILABLE" not in line for line in p10_lines):
        preflight["status"] = "FAIL"
        preflight["blockers"] = ["L3_NOT_AVAILABLE_IN_EVERY_P10_FOLD"]
        preflight["p10_status_lines"] = p10_lines
        (preflight_dir / "l3_preflight_receipt.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("L3_PREFLIGHT_JSON=" + json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 5

    _run([*base_command, "--through", "P17", "--resume"], cwd=project_root)
    preflight["status"] = "PASS_P00_P17"
    preflight["p10_status_lines"] = p10_lines
    (preflight_dir / "l3_preflight_receipt.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("L3_PRODUCTION_JSON=" + json.dumps(preflight, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
