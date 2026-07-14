"""Idempotently add P01 source-registry and raw-audit contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _load(path: Path) -> dict[str, Any]:
    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(raw, str(path))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]

    data_sources_path = root / "config" / "methodology" / "data_sources.yaml"
    data_sources_doc = _load(data_sources_path)
    data_sources = _mapping(data_sources_doc.get("data_sources"), "data_sources")
    registry = data_sources.get("source_registry")
    if registry is None:
        data_sources["source_registry"] = {
            "root_environment_variable": "DISSERTATION_RAW_ROOT",
            "schema_evolution_policy": "fail_on_unregistered_fields",
            "hash_policy": "locked_sha256_required",
            "sources": {},
        }
    else:
        registry_map = _mapping(registry, "data_sources.source_registry")
        registry_map.setdefault("root_environment_variable", "DISSERTATION_RAW_ROOT")
        registry_map.setdefault("schema_evolution_policy", "fail_on_unregistered_fields")
        registry_map.setdefault("hash_policy", "locked_sha256_required")
        registry_map.setdefault("sources", {})
    _write(data_sources_path, data_sources_doc)

    schema_path = root / "config" / "schemas" / "core.yaml"
    schema_doc = _load(schema_path)
    schemas = _mapping(schema_doc.get("schemas"), "schemas")
    schemas["raw_audit_schema"] = {
        "contract_type": "json_object",
        "version": 1,
        "required_keys": [
            "run_id",
            "source_id",
            "protocol_hash",
            "audited_at_utc",
            "status",
            "source_metadata",
            "file_signature",
            "schema_audit",
            "duplicate_key_audit",
            "date_audit",
            "unit_audit",
            "coverage",
            "issues",
            "decision",
        ],
        "additional_properties": False,
        "compatibility_policy": "strict",
    }
    _write(schema_path, schema_doc)

    artifacts_path = root / "config" / "foundation" / "artifacts.yaml"
    artifacts_doc = _load(artifacts_path)
    artifacts = _mapping(artifacts_doc.get("artifacts"), "artifacts")
    raw_audit = _mapping(artifacts.get("raw_audit"), "artifacts.raw_audit")
    if raw_audit.get("producer_step") != "P01":
        raise ValueError("raw_audit must remain owned by P01")
    raw_audit["schema_id"] = "raw_audit_schema"
    _write(artifacts_path, artifacts_doc)

    steps_path = root / "config" / "foundation" / "steps.yaml"
    steps_doc = _load(steps_path)
    steps = _mapping(steps_doc.get("steps"), "steps")
    p01 = _mapping(steps.get("P01"), "steps.P01")
    p01["cli_module"] = "scripts.p01_audit_raw"
    if p01.get("reads") != [] or p01.get("writes") != ["raw_audit"]:
        raise ValueError("P01 scope drift: expected no artifact reads and only raw_audit writes")
    _write(steps_path, steps_doc)

    print("P01 configuration migration completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
