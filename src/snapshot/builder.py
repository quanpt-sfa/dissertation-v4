"""Build an immutable operational snapshot from the committed source catalog."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .inspector import file_sha256, inspect_file, resolve_semantics
from .models import SourceCatalog, SourceProfile


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
    raw_root = raw_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)
    catalog = SourceCatalog.from_mapping(registry.get("source_catalog"))
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
            missing_semantics = sorted(set(profile.required_semantic_fields) - set(semantics))
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
        "raw_root_recorded": False,
        "profiles": profile_summary,
        "sources": sorted(entries, key=lambda item: str(item["source_id"])),
    }
    snapshot["snapshot_content_hash"] = _snapshot_content_hash(snapshot)
    snapshot["snapshot_hash"] = _snapshot_hash(snapshot)
    return snapshot


def write_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    path = path.resolve()
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
        included.update(path.resolve() for path in raw_root.glob(pattern) if path.is_file())
    excluded: set[Path] = set()
    for pattern in profile.discovery.excludes:
        excluded.update(path.resolve() for path in raw_root.glob(pattern) if path.is_file())
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
) -> dict[str, object]:
    columns = tuple(cast(Any, inspection).columns)
    required_columns = sorted({semantics[name] for name in profile.required_semantic_fields})
    key_columns = [semantics[name] for name in profile.key_semantics if name in semantics]
    date_columns = [semantics[name] for name in profile.date_semantics if name in semantics]
    optional_columns = [column for column in columns if column not in required_columns]
    availability_field = semantics.get("availability_date")

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
        "availability_date_source": f"snapshot:{source_id}",
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
        "resolved_semantics": semantics,
        "schema": {
            "required_columns": required_columns,
            "optional_columns": optional_columns,
            "key_columns": key_columns,
            "date_columns": date_columns,
            "required_date_columns": [availability_field] if availability_field else [],
            "numeric_columns": {},
            "allow_extra_columns": False,
            "key_unique": bool(key_columns),
            "row_count_min": 1,
        },
    }
    if profile.panel_mapping.enabled:
        source["panel_mapping"] = {
            "enabled": True,
            "firm_id_field": semantics["firm_id"],
            "fiscal_year_field": semantics["fiscal_year"],
            "availability_date_field": semantics["availability_date"],
            "fiscal_year_end_field": semantics.get("fiscal_year_end"),
            "ticker_field": semantics.get("ticker"),
            "contributes_to_firm_master": profile.panel_mapping.contributes_to_firm_master,
            "core_predictor": profile.panel_mapping.core_predictor,
        }
    else:
        source["panel_mapping"] = {"enabled": False}
    return source


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
