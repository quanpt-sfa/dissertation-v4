"""Run P00-P06 and report capability for the P0-registered L3 scenarios."""

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


def _run_tests(project_root: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv executable is required to run the test suite")
    _run(
        [uv, "run", "--extra", "dev", "python", "-m", "pytest", "-q"],
        cwd=project_root,
    )


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def _require_clean_tree(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "L3 preparation requires a clean committed Git tree; review and commit the protocol first"
        )


def _data_contract_blockers(s3: dict[str, Any], calibration: dict[str, Any]) -> list[str]:
    result: list[str] = []
    required_s3_fields = {
        "development_positive_count_by_endpoint",
        "sanction_year_unresolved_firm_year_count",
        "excluded_source_rule_row_count",
    }
    missing_s3_fields = sorted(field for field in required_s3_fields if field not in s3)
    if missing_s3_fields:
        result.append("S3_AUDIT_SCHEMA_NOT_HARDENED:" + ",".join(missing_s3_fields))
    checks = {
        "P03_P05_OUTCOME_MISMATCH": int(s3.get("p03_p05_outcome_mismatch_count") or 0),
        "P03_P05_MISSING_KEY": int(s3.get("p03_p05_missing_key_count") or 0),
        "ELIGIBLE_SANCTION_YEAR_UNRESOLVED": int(
            s3.get("eligible_unresolved_sanction_year_mapping_count") or 0
        ),
        "SANCTION_YEAR_UNRESOLVED_FIRM_YEAR": int(
            s3.get("sanction_year_unresolved_firm_year_count") or 0
        ),
    }
    result.extend(name for name, value in checks.items() if value != 0)
    if s3.get("outer_outcomes_accessed") is not False:
        result.append("S3_AUDIT_ACCESSED_OUTER_OUTCOMES")
    if calibration.get("outer_outcomes_accessed") is not False:
        result.append("CALIBRATION_ACCESSED_OUTER_OUTCOMES")
    return result


def _unavailable_reasons(
    s3: dict[str, Any],
    capability: dict[str, Any],
) -> list[str]:
    result: list[str] = []
    by_endpoint = s3.get("development_positive_count_by_endpoint")
    if not isinstance(by_endpoint, dict):
        result.append("S3_AUDIT_ENDPOINT_COUNTS_MALFORMED")
    elif int(by_endpoint.get("S3_CONTENT") or 0) <= 0:
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

    project_root = args.config.resolve().parent.parent
    _require_clean_tree(project_root)
    python = sys.executable
    if not args.skip_tests:
        _run_tests(project_root)

    run_root = args.output_root.expanduser().resolve() / args.run_id
    _run(
        [
            python,
            "scripts/run_pipeline.py",
            "--run-id",
            args.run_id,
            "--raw-root",
            str(args.raw_root.expanduser().resolve()),
            "--output-root",
            str(args.output_root.expanduser().resolve()),
            "--config",
            str(args.config.resolve()),
            "--through",
            "P06",
        ],
        cwd=project_root,
    )
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

    s3 = _json(s3_dir / "s3_year_audit_summary.json")
    calibration = _parse_prefixed_json(
        calibration_output, "MEASUREMENT_CALIBRATION_JSON="
    )
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "measurement_calibration_summary.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    p06 = load_run(
        registry_path=registry_path,
        run_id=args.run_id,
        step_id="P06",
        state="MEASURED",
    )
    scenario_registry = locked_l3_scenario_registry(p06.registry)
    capability = mapping(p06.context.read("l3_pilot_capability", {}), "L3 capability")
    blockers = _data_contract_blockers(s3, calibration)
    unavailable = _unavailable_reasons(s3, capability)
    status = (
        "L3_BLOCKED_BY_DATA_CONTRACT"
        if blockers
        else "L3_UNAVAILABLE_BY_DESIGN"
        if unavailable
        else "L3_AVAILABLE"
    )
    receipt = {
        "run_id": args.run_id,
        "protocol_hash": s3.get("protocol_hash"),
        "status": status,
        "blockers": blockers,
        "unavailable_reasons": unavailable,
        "scenario_registry_status": "LOCKED_AT_P0",
        "primary_scenario_id": scenario_registry.primary_scenario_id,
        "registered_scenario_ids": [
            scenario.scenario_id for scenario in scenario_registry.scenarios
        ],
        "scenario_selection_rule": "PRE_REGISTERED_PRIMARY",
        "performance_based_scenario_selection": False,
        "l3_capability": capability,
        "development_positive_count_by_endpoint": s3.get(
            "development_positive_count_by_endpoint", {}
        ),
        "eligible_unresolved_sanction_year_mapping_count": s3.get(
            "eligible_unresolved_sanction_year_mapping_count"
        ),
        "excluded_source_rule_mapping_count": s3.get("excluded_source_rule_mapping_count"),
        "excluded_source_rule_row_count": s3.get("excluded_source_rule_row_count"),
        "excluded_unresolved_mapping_count": s3.get("excluded_unresolved_mapping_count"),
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
    }
    receipt_dir = run_root / "PREPARATION"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    (receipt_dir / "l3_preparation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("L3_PREPARATION_JSON=" + json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 3 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
