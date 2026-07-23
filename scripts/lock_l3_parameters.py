"""Lock reviewed fixed-pi scenarios and source-profile accuracy priors into measurement.yaml."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_PROFILES = {
    "financial_statement_core_long",
    "audit_annual_long",
    "sanction_evidence",
}
PRIOR_FIELDS = {
    "sensitivity_alpha",
    "sensitivity_beta",
    "specificity_alpha",
    "specificity_beta",
}


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_preparation(receipt: dict[str, Any]) -> None:
    if receipt.get("status") != "READY_FOR_L3_PARAMETER_LOCK":
        raise ValueError("preparation receipt is not ready for L3 parameter lock")
    if receipt.get("blockers") not in ([], None):
        raise ValueError("preparation receipt still contains blockers")
    if receipt.get("outer_outcomes_accessed") is not False:
        raise ValueError("preparation receipt must certify no outer-outcome access")
    if receipt.get("known_cases_accessed") is not False:
        raise ValueError("preparation receipt must certify no known-case access")
    positives = receipt.get("development_positive_count_by_endpoint")
    if not isinstance(positives, dict) or int(positives.get("S3_CONTENT") or 0) <= 0:
        raise ValueError("preparation receipt requires development S3_CONTENT positives")
    if int(receipt.get("eligible_unresolved_sanction_year_mapping_count") or 0) != 0:
        raise ValueError("eligible unresolved sanction mappings must equal zero")


def _validate_lock(
    lock: dict[str, Any],
) -> tuple[list[float], dict[str, dict[str, float]], dict[str, Any]]:
    raw_grid = lock.get("fixed_pi_grid")
    if not isinstance(raw_grid, list) or not raw_grid:
        raise ValueError("fixed_pi_grid must be a nonempty list")
    grid = [float(value) for value in raw_grid]
    if any(not 0.0 < value < 1.0 for value in grid):
        raise ValueError("each fixed-pi value must lie strictly between zero and one")
    if grid != sorted(set(grid)):
        raise ValueError("fixed_pi_grid must be sorted and unique")

    raw_priors = lock.get("accuracy_priors_by_profile")
    if not isinstance(raw_priors, dict):
        raise ValueError("accuracy_priors_by_profile mapping is required")
    missing = sorted(REQUIRED_PROFILES - set(raw_priors))
    if missing:
        raise ValueError(f"L3 priors are missing required profiles: {missing}")
    priors: dict[str, dict[str, float]] = {}
    for profile in sorted(REQUIRED_PROFILES):
        raw = raw_priors[profile]
        if not isinstance(raw, dict) or set(raw) != PRIOR_FIELDS:
            raise ValueError(f"profile={profile}: exactly four beta-prior fields are required")
        parsed = {field: float(raw[field]) for field in sorted(PRIOR_FIELDS)}
        if any(value <= 0 for value in parsed.values()):
            raise ValueError(f"profile={profile}: prior hyperparameters must be positive")
        priors[profile] = parsed

    provenance = lock.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("lock provenance mapping is required")
    if provenance.get("status") != "LOCKED":
        raise ValueError("provenance.status must be LOCKED after researcher review")
    if provenance.get("outer_outcomes_accessed") is not False:
        raise ValueError("L3 parameter lock may not use outer outcomes")
    if provenance.get("known_cases_accessed") is not False:
        raise ValueError("L3 parameter lock may not use known cases")
    encoded = json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    if "REPLACE_ME" in encoded or "illustrative" in encoded.lower():
        raise ValueError("replace all template placeholders and illustrative notes before locking")
    return grid, priors, provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measurement-config", type=Path, default=Path("config/methodology/measurement.yaml")
    )
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--preparation-receipt", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    measurement_path = args.measurement_config.resolve()
    lock_path = args.lock_file.resolve()
    preparation_path = args.preparation_receipt.resolve()
    measurement = _load_yaml(measurement_path)
    lock = _load_yaml(lock_path)
    preparation = _load_json(preparation_path)
    _validate_preparation(preparation)
    grid, priors, provenance = _validate_lock(lock)

    measurement_section = measurement.get("measurement")
    if not isinstance(measurement_section, dict):
        raise ValueError("measurement config is missing measurement mapping")
    if measurement_section.get("primary_s3_endpoint") != "S3_CONTENT":
        raise ValueError("primary_s3_endpoint must be S3_CONTENT before L3 locking")
    source_set_id = measurement_section.get("primary_source_set_id")
    source_sets = measurement_section.get("source_sets")
    if not isinstance(source_set_id, str) or not isinstance(source_sets, dict):
        raise ValueError("primary measurement source set is not locked")
    primary = source_sets.get(source_set_id)
    if not isinstance(primary, dict) or not isinstance(primary.get("sources"), list):
        raise ValueError("primary measurement source set is malformed")
    s3_sources = sorted(str(value) for value in primary["sources"] if str(value).startswith("S3_"))
    if s3_sources != ["S3_CONTENT"]:
        raise ValueError("primary source set must contain only S3_CONTENT among S3 endpoints")

    l3_model = measurement_section.get("l3_model")
    if not isinstance(l3_model, dict) or not isinstance(l3_model.get("operational"), dict):
        raise ValueError("measurement.l3_model.operational mapping is required")
    operational = l3_model["operational"]
    operational["fixed_pi_grid"] = grid
    operational["accuracy_priors_by_profile"] = priors
    operational["parameter_lock"] = {
        "status": "LOCKED",
        "preparation_run_id": preparation.get("run_id"),
        "preparation_protocol_hash": preparation.get("protocol_hash"),
        "lock_file_sha256": _sha256(lock_path),
        "preparation_receipt_sha256": _sha256(preparation_path),
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
        "provenance": provenance,
    }

    rendered = yaml.safe_dump(measurement, sort_keys=False, allow_unicode=True)
    receipt = {
        "status": "LOCKED",
        "measurement_config": str(measurement_path),
        "write_applied": bool(args.write),
        "fixed_pi_grid": grid,
        "profiles": sorted(priors),
        "preparation_run_id": preparation.get("run_id"),
        "preparation_protocol_hash": preparation.get("protocol_hash"),
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
    }
    if args.write:
        measurement_path.write_text(rendered, encoding="utf-8")
        receipt_path = measurement_path.with_name("l3_parameter_lock.receipt.json")
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print("L3_PARAMETER_LOCK_JSON=" + json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
