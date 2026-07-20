"""Run P00-P17 with P0-registered, capability-gated L3 scenarios."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.l3_scenarios import locked_l3_scenario_registry  # noqa: E402
from core.pipeline import load_run, mapping  # noqa: E402
from core.registry_compiler import compile_registry  # noqa: E402


def _run(command: list[str], *, cwd: Path, capture: bool = False) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=False,
    )
    if capture:
        if result.stdout:
            print(result.stdout, end="", flush=True)
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout if capture else None,
            stderr=result.stderr if capture else None,
        )
    return result.stdout if capture else ""


def _run_tests(project_root: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is required to run the test suite")
    _run(
        [uv, "run", "--extra", "dev", "python", "-m", "pytest", "-q"],
        cwd=project_root,
    )


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


def _validate_preregistered_config(config_path: Path) -> dict[str, Any]:
    registry = compile_registry(config_path).registry
    measurement = mapping(registry.get("measurement"), "measurement")
    if measurement.get("primary_s3_endpoint") != "S3_CONTENT":
        raise ValueError("final run requires primary_s3_endpoint=S3_CONTENT")
    source_set_id = measurement.get("primary_source_set_id")
    source_sets = mapping(measurement.get("source_sets"), "measurement.source_sets")
    if not isinstance(source_set_id, str):
        raise ValueError("primary source set is not locked")
    primary = mapping(source_sets.get(source_set_id), f"measurement.source_sets.{source_set_id}")
    raw_sources = primary.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("primary source set is malformed")
    s3_sources = sorted(str(value) for value in raw_sources if str(value).startswith("S3_"))
    if s3_sources != ["S3_CONTENT"]:
        raise ValueError("final primary source set must contain S3_CONTENT and no other S3 endpoint")
    l3_model = mapping(measurement.get("l3_model"), "measurement.l3_model")
    if l3_model.get("scenario_registry_module") != "l3_scenarios":
        raise ValueError("L3 model must bind the l3_scenarios protocol module")
    if l3_model.get("scenario_selection_rule") != "preregistered_primary_only":
        raise ValueError("L3 scenario selection must be preregistered_primary_only")
    if l3_model.get("performance_based_scenario_selection_forbidden") is not True:
        raise ValueError("performance-based L3 scenario selection must be forbidden")
    scenario_registry = locked_l3_scenario_registry(registry)
    return {
        "primary_scenario_id": scenario_registry.primary_scenario_id,
        "registered_scenario_ids": [item.scenario_id for item in scenario_registry.scenarios],
        "scenario_registry_status": "LOCKED_AT_P0",
        "scenario_selection_rule": "PRE_REGISTERED_PRIMARY",
        "performance_based_scenario_selection": False,
    }


def _audit_blockers(s3: dict[str, Any], calibration: dict[str, Any]) -> list[str]:
    result: list[str] = []
    required_s3_fields = {
        "development_positive_count_by_endpoint",
        "sanction_year_unresolved_firm_year_count",
        "excluded_source_rule_row_count",
    }
    missing_s3_fields = sorted(field for field in required_s3_fields if field not in s3)
    if missing_s3_fields:
        result.append("S3_AUDIT_SCHEMA_NOT_HARDENED:" + ",".join(missing_s3_fields))
    for key in (
        "p03_p05_outcome_mismatch_count",
        "p03_p05_missing_key_count",
        "eligible_unresolved_sanction_year_mapping_count",
        "sanction_year_unresolved_firm_year_count",
    ):
        if int(s3.get(key) or 0) != 0:
            result.append(key.upper())
    if s3.get("outer_outcomes_accessed") is not False:
        result.append("S3_AUDIT_OUTER_OUTCOME_ACCESS")
    if calibration.get("outer_outcomes_accessed") is not False:
        result.append("CALIBRATION_OUTER_OUTCOME_ACCESS")
    return result


def _l3_unavailable_reasons(
    s3: dict[str, Any],
    capability: dict[str, Any],
) -> list[str]:
    result: list[str] = []
    positives = s3.get("development_positive_count_by_endpoint")
    if not isinstance(positives, dict):
        result.append("S3_AUDIT_ENDPOINT_COUNTS_MALFORMED")
    elif int(positives.get("S3_CONTENT") or 0) <= 0:
        result.append("NO_DEVELOPMENT_S3_CONTENT_POSITIVES")
    if capability.get("status") != "AVAILABLE" or capability.get("pilot_executed") is not True:
        result.append(
            f"L3_CAPABILITY_{capability.get('status', 'UNKNOWN')}_{capability.get('reason_code')}"
        )
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
    scenario_receipt = _validate_preregistered_config(config)
    python = sys.executable
    if not args.skip_tests:
        _run_tests(project_root)

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

    registry_path = run_root / "P00" / "registry.lock.json"
    s3_dir = run_root / "S3_AUDIT"
    calibration_dir = run_root / "CALIBRATION"
    _run(
        [
            python,
            "scripts/report_s3_year_audit.py",
            "--registry",
            str(registry_path),
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
            str(registry_path),
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

    p06 = load_run(registry_path=registry_path, run_id=args.run_id, step_id="P06", state="MEASURED")
    capability = mapping(p06.context.read("l3_pilot_capability", {}), "L3 capability")
    matrices = mapping(p06.context.read("source_channel_matrices", {}), "source matrices")
    unavailable_reasons = _l3_unavailable_reasons(s3, capability)
    if capability.get("status") not in {"AVAILABLE", "UNAVAILABLE_BY_DESIGN"}:
        blockers.append(
            f"L3_CAPABILITY_NOT_RESOLVED_{capability.get('status', 'UNKNOWN')}_{capability.get('reason_code')}"
        )
    primary_sources = matrices.get("primary_measurement_sources")
    if not isinstance(primary_sources, list):
        blockers.append("PRIMARY_MEASUREMENT_SOURCE_RECEIPT_MISSING")
    else:
        s3_sources = sorted(str(value) for value in primary_sources if str(value).startswith("S3_"))
        if s3_sources != ["S3_CONTENT"]:
            blockers.append("PRIMARY_MEASUREMENT_S3_ENDPOINT_DRIFT")

    l3_available = not unavailable_reasons and capability.get("status") == "AVAILABLE"
    preflight_dir = run_root / "PREFLIGHT"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "run_id": args.run_id,
        "status": "FAIL"
        if blockers
        else "PASS_L3_AVAILABLE"
        if l3_available
        else "PASS_L3_UNAVAILABLE_BY_DESIGN",
        "blockers": blockers,
        "l3_unavailable_reasons": unavailable_reasons,
        "protocol_hash": s3.get("protocol_hash"),
        "l3_capability": capability,
        "l3_execution_status": "PENDING" if l3_available else "SKIPPED_UNAVAILABLE_BY_DESIGN",
        "primary_measurement_sources": primary_sources,
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
        **scenario_receipt,
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
    if l3_available and (
        not p10_lines or any("L3_fixed_pi=AVAILABLE" not in line for line in p10_lines)
    ):
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
    preflight["l3_execution_status"] = (
        "EXECUTED_ALL_REGISTERED_SCENARIOS"
        if l3_available
        else "SKIPPED_UNAVAILABLE_BY_DESIGN"
    )
    preflight["p10_status_lines"] = p10_lines
    (preflight_dir / "l3_preflight_receipt.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("L3_PRODUCTION_JSON=" + json.dumps(preflight, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
