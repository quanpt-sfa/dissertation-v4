"""P03 CLI: build the deduplicated evidence ledger and lag audit."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns
from core.runtime import RunContext
from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    FIRM_ID,
    FISCAL_YEAR,
    OUTCOME,
    SOURCE_ID,
)
from evidence.service import EvidenceRecord, build_evidence_ledger
from p01.readers import iter_rows
from p01.registry import resolve_source
from p02.builder import normalize_entity_field, resolve_entity_link
from p02.models import EntityResolutionSpec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P03",
        state="PANELLED",
    )
    if args.dry_run:
        print("P03 dry-run: evidence sources will be resolved from the locked registry")
        return 0
    panel = loaded.context.read("firm_year_panel", {})
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("firm_year_panel must be a DataFrame")
    records = _records(loaded.registry, loaded.context)
    entity = EntityResolutionSpec.from_mapping(loaded.registry.get("entity_resolution"))
    evidence = mapping(loaded.registry.get("evidence"), "evidence")
    tolerance = evidence.get("lag_identity_tolerance_days")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("evidence.lag_identity_tolerance_days must be nonnegative")
    result = build_evidence_ledger(
        panel=panel,
        records=records,
        columns=physical_columns(loaded.registry),
        fiscal_year_end_month_day=entity.reporting_calendar.default_fiscal_year_end_month_day,
        lag_tolerance_days=tolerance,
    )
    if args.validate_only:
        return 0
    loaded.context.write("evidence_ledger", result.ledger, {})
    loaded.context.write("availability_registry", result.availability_registry, {})
    loaded.context.write("lag_decomposition", result.lag_decomposition, {})
    print(f"P03 status=PASS evidence_rows={len(result.ledger)}")
    return 0


def _records(registry: dict[str, Any], context: RunContext) -> list[EvidenceRecord]:
    data_sources = mapping(registry.get("data_sources"), "data_sources")
    source_registry = mapping(data_sources.get("source_registry"), "source_registry")
    sources = mapping(source_registry.get("sources"), "source_registry.sources")
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    result: list[EvidenceRecord] = []
    for source_id, raw in sorted(sources.items()):
        source = mapping(raw, f"source={source_id}")
        if source.get("role") != "evidence" or source.get("enabled") is not True:
            continue
        audit = mapping(context.read("raw_audit", {SOURCE_ID: source_id}), "raw_audit")
        if mapping(audit.get("decision"), "audit.decision").get("pipeline_may_advance") is not True:
            raise ValueError(f"source={source_id}: passing P01 audit required")
        spec, path = resolve_source(registry, source_id)
        semantics = mapping(source.get("resolved_semantics"), f"source={source_id}.semantics")
        required = {FIRM_ID, FISCAL_YEAR, AVAILABILITY_DATE, OUTCOME}
        if not required.issubset(semantics):
            raise ValueError(
                f"source={source_id}: unresolved evidence semantics {sorted(required - set(semantics))}"
            )
        for row_number, row in enumerate(iter_rows(path, spec), start=1):
            firm_raw = _required(row.get(str(semantics[FIRM_ID])), FIRM_ID)
            normalized = normalize_entity_field(str(firm_raw), entity)
            canonical, _ = resolve_entity_link(source_id, str(firm_raw), normalized, entity)
            fiscal_year = int(str(_required(row.get(str(semantics[FISCAL_YEAR])), FISCAL_YEAR)))
            availability = _datetime(
                _required(row.get(str(semantics[AVAILABILITY_DATE])), AVAILABILITY_DATE)
            )
            outcome = _boolean(row.get(str(semantics[OUTCOME])))
            event_id = _optional_text(row, semantics, "event_id") or _derived_event_id(
                source_id, canonical, fiscal_year, availability, row_number
            )
            cluster = _optional_text(row, semantics, "event_cluster_id") or event_id
            result.append(
                EvidenceRecord(
                    source_id=source_id,
                    channel_id=str(source[CHANNEL_ID]),
                    firm_id=canonical,
                    fiscal_year=fiscal_year,
                    availability_date=availability,
                    outcome=outcome,
                    event_id=event_id,
                    event_cluster_id=cluster,
                )
            )
    if not result:
        raise ValueError("P03 requires at least one registered evidence record")
    return result


def _required(value: object, field: str) -> object:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"evidence row missing {field}")
    return value


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    parsed = pd.Timestamp(str(value))
    return parsed.to_pydatetime()


def _boolean(value: object) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"evidence outcome is not a registered binary representation: {value}")


def _optional_text(row: dict[str, object], semantics: dict[str, Any], name: str) -> str | None:
    column = semantics.get(name)
    if not isinstance(column, str):
        return None
    value = row.get(column)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _derived_event_id(
    source_id: str, firm_id: str, fiscal_year: int, availability: datetime, row_number: int
) -> str:
    value = f"{source_id}|{firm_id}|{fiscal_year}|{availability.isoformat()}|{row_number}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
