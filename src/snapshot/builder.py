"""Build an immutable operational snapshot from the committed source catalog."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from core.path_policy import resolve_relative_path

from .inspector import file_sha256, inspect_file, resolve_semantics
from .models import SourceCatalog, SourceProfile

_EMBEDDED_PROVENANCE_INPUT = "data/source/vn_pipeline_final_firm_year_2015_2025.parquet"
_UNAVAILABLE_DERIVED_METADATA = "UNAVAILABLE_NOT_RETAINED_IN_DERIVED_INPUT"
_EMBEDDED_LINEAGE_COLUMNS = (
    "source_provenance_json",
    "source_protocol_hash",
    "source_run_id",
    "source_unified_population_sha256",
)


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def build_snapshot(
    *,
    registry: dict[str, object],
    raw_root: Path,
    snapshot_id: str,
) -> dict[str, object]:
    raw_root = raw_root.expanduser().resolve(strict=True)
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    catalog = SourceCatalog.from_mapping(registry.get("source_catalog"))
    extract_provenance = _load_extract_provenance(registry, raw_root)
    entries: list[dict[str, object]] = []
    profile_summary: dict[str, object] = {}

    for profile in catalog.profiles:
        if not profile.enabled:
            continue
        matches = _discover(raw_root, profile)
        _validate_cardinality(profile, matches)
        profile_summary[profile.profile_id] = {
            "matched_file_count": len(matches),
            "required": profile.required,
            "cardinality": profile.discovery.cardinality,
        }
        for path in matches:
            inspection = inspect_file(path, profile.format, profile.reader)
            semantics = resolve_semantics(inspection.columns, profile.semantic_fields)
            derived_semantics = _derived_semantics(profile)
            missing_semantics = sorted(
                set(profile.required_semantic_fields) - set(semantics) - set(derived_semantics)
            )
            if missing_semantics:
                raise ValueError(
                    f"profile={profile.profile_id}, file={path}: required semantic fields "
                    f"not resolved: {missing_semantics}; columns={list(inspection.columns)}"
                )
            source_id = _source_id(profile, path, matches)
            relative = path.relative_to(raw_root).as_posix()
            entry = _build_entry(
                profile=profile,
                source_id=source_id,
                relative_path=relative,
                file_hash=file_sha256(path),
                file_size=path.stat().st_size,
                inspection=inspection,
                semantics=semantics,
                derived_semantics=derived_semantics,
            )
            entries.append(entry)

    if not entries:
        raise ValueError("snapshot contains no source files")
    if not any(_panel_enabled(entry) for entry in entries):
        raise ValueError("snapshot contains no P02-enabled source")
    if not any(_panel_core(entry) for entry in entries):
        raise ValueError("snapshot contains no core predictor source")

    snapshot: dict[str, object] = {
        "snapshot_schema_version": catalog.snapshot_schema_version,
        "snapshot_id": snapshot_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "root_environment_variable": catalog.root_environment_variable,
        "raw_root_recorded": True,
        "raw_root": str(raw_root),
        "extract_provenance": extract_provenance,
        "profiles": profile_summary,
        "sources": sorted(entries, key=lambda item: str(item["source_id"])),
    }
    snapshot["snapshot_content_hash"] = _snapshot_content_hash(snapshot)
    snapshot["snapshot_hash"] = _snapshot_hash(snapshot)
    return snapshot


def _load_extract_provenance(registry: dict[str, object], raw_root: Path) -> dict[str, object]:
    data_sources = registry.get("data_sources")
    if not isinstance(data_sources, dict):
        return {}
    source_registry = cast(dict[str, object], data_sources).get("source_registry")
    if not isinstance(source_registry, dict):
        return {}
    contract = cast(dict[str, object], source_registry).get("provenance_contract")
    if not isinstance(contract, dict):
        return {}
    typed = cast(dict[str, object], contract)
    relative = typed.get("manifest_relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("data source provenance manifest path must be registered")
    path = resolve_relative_path(
        raw_root,
        relative,
        context="data source provenance manifest",
    )
    required = typed.get("required") is True
    if path.is_file():
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        provenance = _mapping(raw, "extract provenance")
        return _validate_extract_provenance(provenance, typed)

    embedded = _load_embedded_input_provenance(raw_root)
    if embedded is not None:
        return _validate_extract_provenance(embedded, typed)

    if required:
        raise FileNotFoundError(
            "required provenance evidence is unavailable: external manifest is missing "
            f"at {path} and {_EMBEDDED_PROVENANCE_INPUT} does not provide the locked "
            "embedded lineage contract"
        )
    return {}


def _validate_extract_provenance(
    provenance: dict[str, Any],
    contract: dict[str, object],
) -> dict[str, object]:
    raw_fields = contract.get("required_fields")
    if not isinstance(raw_fields, list) or not all(
        isinstance(value, str) and value for value in cast(list[object], raw_fields)
    ):
        raise ValueError("extract provenance required_fields must be registered")
    required_fields = cast(list[str], raw_fields)
    missing = [
        field
        for field in required_fields
        if field not in provenance or provenance[field] in {None, ""}
    ]
    if missing:
        raise ValueError(f"extract provenance is missing required fields: {missing}")
    return {str(key): value for key, value in provenance.items()}


def _load_embedded_input_provenance(raw_root: Path) -> dict[str, Any] | None:
    input_path = resolve_relative_path(
        raw_root,
        _EMBEDDED_PROVENANCE_INPUT,
        context="embedded provenance input",
    )
    if not input_path.is_file():
        return None

    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema_arrow.names)
    if "source_provenance_json" not in available:
        return None

    columns = [column for column in _EMBEDDED_LINEAGE_COLUMNS if column in available]
    table = pq.read_table(input_path, columns=columns)
    embedded_text = _single_nonblank_value(table, "source_provenance_json", required=True)
    if embedded_text is None:
        return None

    raw_embedded: object = json.loads(embedded_text)
    embedded = _mapping(raw_embedded, "embedded source provenance")

    provenance: dict[str, Any] = {
        "vendor": _UNAVAILABLE_DERIVED_METADATA,
        "vendor_product": _UNAVAILABLE_DERIVED_METADATA,
        "pull_date": _UNAVAILABLE_DERIVED_METADATA,
        "vendor_version": _UNAVAILABLE_DERIVED_METADATA,
        "extract_query": _UNAVAILABLE_DERIVED_METADATA,
        "revision_policy": _UNAVAILABLE_DERIVED_METADATA,
        "point_in_time_vintages_available": False,
        "provenance_origin": "embedded_derived_input_lineage",
        "vendor_metadata_status": "not_retained_in_derived_input",
        "derived_input_relative_path": _EMBEDDED_PROVENANCE_INPUT,
        "derived_input_sha256": file_sha256(input_path),
        "embedded_source_provenance": {str(key): value for key, value in embedded.items()},
    }

    for column in _EMBEDDED_LINEAGE_COLUMNS[1:]:
        if column not in columns:
            continue
        value = _single_nonblank_value(table, column, required=False)
        if value is not None:
            provenance[column] = value

    return provenance


def _single_nonblank_value(table: Any, column: str, *, required: bool) -> str | None:
    values = {
        str(value).strip()
        for value in cast(list[object], table[column].to_pylist())
        if value is not None and str(value).strip()
    }
    if not values:
        if required:
            raise ValueError(f"embedded provenance column={column}: no nonblank value")
        return None
    if len(values) != 1:
        raise ValueError(
            f"embedded provenance column={column}: expected one immutable value, "
            f"found {len(values)}"
        )
    return next(iter(values))


def write_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    path = path.expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"snapshot already exists and is immutable: {path}")
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_snapshot(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    snapshot = _mapping(raw, "snapshot")
    recorded = snapshot.get("snapshot_hash")
    if not isinstance(recorded, str) or recorded != _snapshot_hash(snapshot):
        raise ValueError("snapshot hash mismatch")
    return cast(dict[str, object], snapshot)


def _discover(raw_root: Path, profile: SourceProfile) -> list[Path]:
    included: set[Path] = set()
    for pattern in profile.discovery.globs:
        resolve_relative_path(
            raw_root,
            pattern,
            context=f"profile={profile.profile_id}, discovery glob",
        )
        included.update(
            path.resolve(strict=True) for path in raw_root.glob(pattern) if path.is_file()
        )
    excluded: set[Path] = set()
    for pattern in profile.discovery.excludes:
        resolve_relative_path(
            raw_root,
            pattern,
            context=f"profile={profile.profile_id}, discovery exclude",
        )
        excluded.update(
            path.resolve(strict=True) for path in raw_root.glob(pattern) if path.is_file()
        )
    return sorted(included - excluded, key=lambda path: path.relative_to(raw_root).as_posix())


def _validate_cardinality(profile: SourceProfile, matches: list[Path]) -> None:
    count = len(matches)
    if profile.discovery.cardinality == "one" and count != 1:
        raise ValueError(f"profile={profile.profile_id}: expected exactly one file, found {count}")
    if profile.discovery.cardinality == "one_or_none" and count > 1:
        raise ValueError(f"profile={profile.profile_id}: expected at most one file, found {count}")
    if profile.discovery.cardinality == "many" and profile.required and count < 1:
        raise ValueError(f"profile={profile.profile_id}: expected at least one file")
    if profile.required and count == 0:
        raise ValueError(f"profile={profile.profile_id}: required source not found")


def _source_id(profile: SourceProfile, path: Path, matches: list[Path]) -> str:
    if len(matches) == 1 and profile.discovery.cardinality != "many":
        return profile.profile_id
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.casefold()).strip("_")
    suffix = file_sha256(path)[:8]
    return f"{profile.profile_id}__{stem}__{suffix}"


def _build_entry(
    *,
    profile: SourceProfile,
    source_id: str,
    relative_path: str,
    file_hash: str,
    file_size: int,
    inspection: object,
    semantics: dict[str, str],
    derived_semantics: dict[str, dict[str, str]],
) -> dict[str, object]:
    columns = tuple(cast(Any, inspection).columns)
    required_columns = sorted(
        {semantics[name] for name in profile.required_semantic_fields if name in semantics}
    )
    key_columns = [semantics[name] for name in profile.key_semantics if name in semantics]
    date_columns = [semantics[name] for name in profile.date_semantics if name in semantics]
    optional_columns = [column for column in columns if column not in required_columns]
    resolved_semantics = dict(semantics)
    resolved_semantics.update(
        {semantic: value["field"] for semantic, value in derived_semantics.items()}
    )
    availability_field = resolved_semantics.get("availability_date")
    availability_source = (
        f"derived:{profile.panel_mapping.availability_date_rule}:"
        f"{profile.panel_mapping.availability_month_day}"
        if "availability_date" in derived_semantics
        else f"snapshot:{source_id}"
    )

    source: dict[str, object] = {
        "source_id": source_id,
        "profile_id": profile.profile_id,
        "enabled": True,
        "channel_id": profile.channel_id,
        "source_type": profile.source_type,
        "source_agency": profile.source_agency,
        "original_unit": profile.original_unit,
        "related_period_field": semantics.get("fiscal_year"),
        "availability_date_field": availability_field,
        "availability_date_source": availability_source,
        "coverage_dimensions": list(profile.coverage_dimensions),
        "role": profile.role,
        "verification_status": profile.verification_status,
        "data_risks": list(profile.data_risks),
        "relative_path": relative_path,
        "format": profile.format,
        "encoding": profile.reader.encoding,
        "delimiter": profile.reader.delimiter,
        "sheet_name": cast(Any, inspection).reader.get("sheet_name"),
        "header_row": cast(Any, inspection).reader.get("header_row"),
        "locked_sha256": file_hash,
        "file_size_bytes": file_size,
        "row_count_snapshot": cast(Any, inspection).row_count,
        "schema_hash": cast(Any, inspection).schema_hash,
        "resolved_semantics": resolved_semantics,
        "derived_semantics": derived_semantics,
        "schema": {
            "required_columns": required_columns,
            "optional_columns": optional_columns,
            "key_columns": key_columns,
            "date_columns": date_columns,
            "required_date_columns": [semantics["availability_date"]]
            if "availability_date" in semantics
            else [],
            "numeric_columns": {},
            "allow_extra_columns": False,
            "key_unique": bool(key_columns),
            "row_count_min": 1,
        },
    }
    if profile.evidence_mapping is not None:
        source["evidence_mapping"] = deepcopy(profile.evidence_mapping.raw_mapping)
    if profile.panel_mapping.enabled:
        source["panel_mapping"] = {
            "enabled": True,
            "firm_id_field": semantics["firm_id"],
            "fiscal_year_field": semantics["fiscal_year"],
            "availability_date_field": resolved_semantics["availability_date"],
            "fiscal_year_end_field": semantics.get("fiscal_year_end"),
            "ticker_field": semantics.get("ticker"),
            "contributes_to_firm_master": profile.panel_mapping.contributes_to_firm_master,
            "core_predictor": profile.panel_mapping.core_predictor,
            "availability_date_rule": profile.panel_mapping.availability_date_rule,
            "availability_month_day": profile.panel_mapping.availability_month_day,
            "row_aggregation": profile.panel_mapping.row_aggregation,
        }
    else:
        source["panel_mapping"] = {"enabled": False}
    return source


def _derived_semantics(profile: SourceProfile) -> dict[str, dict[str, str]]:
    panel = profile.panel_mapping
    if not panel.enabled or panel.availability_date_rule == "physical_column":
        return {}
    if panel.availability_date_rule != "fiscal_year_plus_one_month_day":
        raise ValueError(
            f"profile={profile.profile_id}: unsupported availability rule "
            f"{panel.availability_date_rule}"
        )
    if panel.availability_month_day is None:
        raise ValueError(f"profile={profile.profile_id}: availability month/day is required")
    return {
        "availability_date": {
            "field": "__derived_availability_date__",
            "rule": panel.availability_date_rule,
            "month_day": panel.availability_month_day,
        }
    }


def _snapshot_hash(snapshot: dict[str, object]) -> str:
    import hashlib

    payload = dict(snapshot)
    payload.pop("snapshot_hash", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_content_hash(snapshot: dict[str, object]) -> str:
    """Hash only source-discovery semantics, not run-local snapshot metadata."""
    import hashlib

    payload = {
        "snapshot_schema_version": snapshot.get("snapshot_schema_version"),
        "root_environment_variable": snapshot.get("root_environment_variable"),
        "raw_root_recorded": snapshot.get("raw_root_recorded"),
        "extract_provenance": snapshot.get("extract_provenance"),
        "profiles": snapshot.get("profiles"),
        "sources": snapshot.get("sources"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _panel_enabled(entry: dict[str, object]) -> bool:
    panel = entry.get("panel_mapping")
    if not isinstance(panel, dict):
        return False
    return cast(dict[str, object], panel).get("enabled") is True


def _panel_core(entry: dict[str, object]) -> bool:
    panel = entry.get("panel_mapping")
    if not isinstance(panel, dict):
        return False
    typed = cast(dict[str, object], panel)
    return typed.get("enabled") is True and typed.get("core_predictor") is True
