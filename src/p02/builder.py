"""Deterministic entity resolution and as-of panel construction."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from p01.models import SourceSpec
from p01.readers import hash_file, iter_rows

from .models import EntityResolutionSpec, PanelSourceSpec


@dataclass(frozen=True)
class SourceInput:
    source_spec: SourceSpec
    panel_spec: PanelSourceSpec
    path: Path
    audit: dict[str, Any]


@dataclass(frozen=True)
class PanelBuildResult:
    firm_master: pd.DataFrame
    firm_year_panel: pd.DataFrame
    resolution_report: dict[str, object]


@dataclass(frozen=True)
class _Observation:
    source_id: str
    canonical_firm_id: str
    fiscal_year: int
    availability_date: datetime
    fiscal_year_end: date
    raw_firm_id: str
    normalized_firm_id: str
    resolution_method: str


def build_firm_panel(
    *,
    run_id: str,
    protocol_hash: str,
    source_inputs: list[SourceInput],
    entity_spec: EntityResolutionSpec,
    sample_start: int,
    sample_end: int,
    output_columns: dict[str, str],
) -> PanelBuildResult:
    if sample_start > sample_end:
        raise ValueError("sample fiscal-year range is inverted")
    if not source_inputs:
        raise ValueError("P02 requires at least one enabled source")

    core_sources = {
        item.panel_spec.source_id for item in source_inputs if item.panel_spec.core_predictor
    }
    if not core_sources:
        raise ValueError("P02 requires at least one core predictor source")

    observations: list[_Observation] = []
    resolution_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    exclusions: list[dict[str, object]] = []
    seen_source_keys: set[tuple[str, str, int]] = set()
    normalized_origins: defaultdict[str, set[tuple[str, str]]] = defaultdict(set)
    collapsed_source_rows = 0

    for source_input in source_inputs:
        _validate_audit(source_input)
        if hash_file(source_input.path) != source_input.source_spec.locked_sha256:
            raise ValueError(
                f"source={source_input.source_spec.source_id}: raw hash changed after P01"
            )
        for row_number, row in enumerate(
            iter_rows(source_input.path, source_input.source_spec), start=1
        ):
            panel_spec = source_input.panel_spec
            raw_identifier = row.get(panel_spec.firm_id_field)
            if _missing(raw_identifier):
                exclusions.append(
                    {
                        "record_type": "row_exclusion",
                        "reason": "MISSING_FIRM_ID",
                        "source_id": panel_spec.source_id,
                        "row_number": row_number,
                    }
                )
                continue
            raw_firm_id = str(raw_identifier)
            normalized = normalize_entity_field(raw_firm_id, entity_spec)
            canonical, method = resolve_entity_link(
                panel_spec.source_id, raw_firm_id, normalized, entity_spec
            )
            normalized_origins[canonical].add((panel_spec.source_id, normalized))

            fiscal_year = _parse_fiscal_year(row.get(panel_spec.fiscal_year_field))
            if fiscal_year is None:
                exclusions.append(
                    {
                        "record_type": "row_exclusion",
                        "reason": "INVALID_FISCAL_YEAR",
                        "source_id": panel_spec.source_id,
                        "row_number": row_number,
                        "canonical_firm_id": canonical,
                    }
                )
                continue
            if fiscal_year < sample_start or fiscal_year > sample_end:
                exclusions.append(
                    {
                        "record_type": "row_exclusion",
                        "reason": "OUTSIDE_SAMPLE_WINDOW",
                        "source_id": panel_spec.source_id,
                        "row_number": row_number,
                        "canonical_firm_id": canonical,
                        "fiscal_year": fiscal_year,
                    }
                )
                continue

            availability = _resolve_availability_date(
                row=row,
                panel_spec=panel_spec,
                fiscal_year=fiscal_year,
            )
            if availability is None:
                exclusions.append(
                    {
                        "record_type": "row_exclusion",
                        "reason": "MISSING_OR_INVALID_AVAILABILITY_DATE",
                        "source_id": panel_spec.source_id,
                        "row_number": row_number,
                        "canonical_firm_id": canonical,
                        "fiscal_year": fiscal_year,
                    }
                )
                continue

            fiscal_year_end = _resolve_fiscal_year_end(
                row=row,
                panel_spec=panel_spec,
                canonical_firm_id=canonical,
                fiscal_year=fiscal_year,
                entity_spec=entity_spec,
            )
            if (
                availability.date() < fiscal_year_end
                and (
                    canonical,
                    fiscal_year,
                    panel_spec.source_id,
                )
                not in entity_spec.reporting_calendar.early_report_exceptions
            ):
                raise ValueError(
                    "report/availability date precedes fiscal-year end without a registered "
                    f"exception: source={panel_spec.source_id}, firm={canonical}, "
                    f"fiscal_year={fiscal_year}"
                )

            source_key = (panel_spec.source_id, canonical, fiscal_year)
            if source_key in seen_source_keys:
                if panel_spec.row_aggregation == "firm_year_presence":
                    collapsed_source_rows += 1
                    continue
                raise ValueError(
                    f"duplicate source firm-year after entity resolution: {source_key}"
                )
            seen_source_keys.add(source_key)
            observations.append(
                _Observation(
                    source_id=panel_spec.source_id,
                    canonical_firm_id=canonical,
                    fiscal_year=fiscal_year,
                    availability_date=availability,
                    fiscal_year_end=fiscal_year_end,
                    raw_firm_id=raw_firm_id,
                    normalized_firm_id=normalized,
                    resolution_method=method,
                )
            )
            resolution_counts[
                (
                    panel_spec.source_id,
                    raw_firm_id,
                    normalized,
                    canonical,
                    method,
                )
            ] += 1

    _validate_collisions(normalized_origins, entity_spec)

    master_ids = sorted(
        {
            observation.canonical_firm_id
            for observation in observations
            if next(
                item.panel_spec.contributes_to_firm_master
                for item in source_inputs
                if item.panel_spec.source_id == observation.source_id
            )
        }
    )
    if not master_ids:
        raise ValueError("P02 produced an empty firm master")

    availability_by_key: defaultdict[tuple[str, int], dict[str, datetime]] = defaultdict(dict)
    for observation in observations:
        if observation.source_id in core_sources:
            availability_by_key[(observation.canonical_firm_id, observation.fiscal_year)][
                observation.source_id
            ] = observation.availability_date

    panel_rows: list[dict[str, object]] = []
    incomplete: list[dict[str, object]] = []
    for (firm_id, fiscal_year), source_dates in sorted(availability_by_key.items()):
        missing_sources = sorted(core_sources - set(source_dates))
        if missing_sources:
            incomplete.append(
                {
                    "record_type": "firm_year_exclusion",
                    "reason": "INCOMPLETE_CORE_PREDICTOR_AVAILABILITY",
                    "canonical_firm_id": firm_id,
                    "fiscal_year": fiscal_year,
                    "missing_source_ids": missing_sources,
                }
            )
            continue
        row: dict[str, object] = {
            output_columns["firm_id"]: firm_id,
            output_columns["fiscal_year"]: fiscal_year,
            output_columns["prediction_time"]: max(source_dates.values()),
        }
        panel_rows.append(row)

    if not panel_rows:
        raise ValueError("P02 produced no complete firm-year panel rows")

    firm_column = output_columns["firm_id"]
    year_column = output_columns["fiscal_year"]
    prediction_column = output_columns["prediction_time"]
    firm_master = pd.DataFrame({firm_column: pd.Series(master_ids, dtype="string")})
    panel = pd.DataFrame(
        panel_rows,
        columns=[firm_column, year_column, prediction_column],
    )
    panel[firm_column] = panel[firm_column].astype("string")
    panel[year_column] = panel[year_column].astype("int16")
    panel[prediction_column] = (
        pd.to_datetime(panel[prediction_column]).dt.tz_localize(None).astype("datetime64[ns]")
    )
    panel = panel.sort_values([firm_column, year_column], kind="stable").reset_index(drop=True)

    resolution_entries = [
        {
            "source_id": source_id,
            "raw_firm_id": raw,
            "normalized_firm_id": normalized,
            "canonical_firm_id": canonical,
            "resolution_method": method,
            "occurrence_count": count,
        }
        for (source_id, raw, normalized, canonical, method), count in sorted(
            resolution_counts.items()
        )
    ]
    report: dict[str, object] = {
        "run_id": run_id,
        "protocol_hash": protocol_hash,
        "entity_resolution": resolution_entries,
        "duplicate_groups": [],
        "excluded_rows": exclusions,
        "excluded_firm_years": incomplete,
        "summary": {
            "enabled_source_count": len(source_inputs),
            "core_source_ids": sorted(core_sources),
            "observation_count": len(observations),
            "firm_count": len(firm_master),
            "panel_row_count": len(panel),
            "excluded_row_count": len(exclusions),
            "excluded_firm_year_count": len(incomplete),
            "collapsed_source_row_count": collapsed_source_rows,
        },
    }
    return PanelBuildResult(
        firm_master=firm_master,
        firm_year_panel=panel,
        resolution_report=report,
    )


def normalize_entity_field(value: str, spec: EntityResolutionSpec) -> str:
    result = value
    if spec.normalization.unicode_nfkc:
        result = unicodedata.normalize("NFKC", result)
    if spec.normalization.trim:
        result = result.strip()
    if spec.normalization.collapse_internal_whitespace:
        result = re.sub(r"\s+", " ", result)
    if spec.normalization.uppercase:
        result = result.upper()
    if not result:
        raise ValueError("firm identity becomes empty after normalization")
    return result


def resolve_entity_link(
    source_id: str,
    raw_value: str,
    normalized_value: str,
    spec: EntityResolutionSpec,
) -> tuple[str, str]:
    aliases = spec.aliases.get(source_id, {})
    if raw_value in aliases:
        return normalize_entity_field(aliases[raw_value], spec), "registered_raw_alias"
    if normalized_value in aliases:
        return normalize_entity_field(
            aliases[normalized_value], spec
        ), "registered_normalized_alias"
    if spec.allow_identity_mapping:
        return normalized_value, "normalized_identity"
    raise ValueError(f"source={source_id}, firm_id={raw_value}: no registered entity mapping")


def _validate_audit(source_input: SourceInput) -> None:
    audit = source_input.audit
    if audit.get("source_id") != source_input.source_spec.source_id:
        raise ValueError("raw audit source_id does not match source registry")
    decision = audit.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("raw audit decision is missing")
    typed_decision = cast(dict[str, object], decision)
    if typed_decision.get("pipeline_may_advance") is not True:
        raise ValueError(f"source={source_input.source_spec.source_id}: P01 audit blocks P02")
    status = audit.get("status")
    if status not in {"PASS", "PASS_WITH_ISSUES"}:
        raise ValueError(
            f"source={source_input.source_spec.source_id}: invalid P01 audit status {status}"
        )


def _validate_collisions(
    origins: dict[str, set[tuple[str, str]]], spec: EntityResolutionSpec
) -> None:
    if spec.collision_policy != "fail":
        return
    for canonical, source_values in origins.items():
        normalized_values = {value for _, value in source_values}
        if len(normalized_values) <= 1:
            continue
        explicitly_registered = all(
            value == canonical
            or any(
                normalize_entity_field(alias, spec) == value
                and normalize_entity_field(target, spec) == canonical
                for alias, target in spec.aliases.get(source_id, {}).items()
            )
            for source_id, value in source_values
        )
        if not explicitly_registered:
            raise ValueError(
                f"canonical firm {canonical} receives multiple unregistered identifiers: "
                f"{sorted(source_values)}"
            )


def _parse_fiscal_year(value: object) -> int | None:
    if _missing(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return None
    return parsed if 1900 <= parsed <= 2200 else None


def _parse_datetime(value: object) -> datetime | None:
    if _missing(value):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for date_format in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d"):
            try:
                return datetime.strptime(text, date_format)
            except ValueError:
                continue
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def _resolve_availability_date(
    *,
    row: dict[str, object],
    panel_spec: PanelSourceSpec,
    fiscal_year: int,
) -> datetime | None:
    if panel_spec.availability_date_rule == "physical_column":
        return _parse_datetime(row.get(panel_spec.availability_date_field))
    if panel_spec.availability_date_rule != "fiscal_year_plus_one_month_day":
        raise ValueError(
            f"source={panel_spec.source_id}: unsupported availability rule "
            f"{panel_spec.availability_date_rule}"
        )
    if panel_spec.availability_month_day is None:
        raise ValueError(f"source={panel_spec.source_id}: availability month/day is not registered")
    month, day = (int(part) for part in panel_spec.availability_month_day.split("-"))
    return datetime(fiscal_year + 1, month, day)


def _resolve_fiscal_year_end(
    *,
    row: dict[str, object],
    panel_spec: PanelSourceSpec,
    canonical_firm_id: str,
    fiscal_year: int,
    entity_spec: EntityResolutionSpec,
) -> date:
    if panel_spec.fiscal_year_end_field:
        parsed = _parse_datetime(row.get(panel_spec.fiscal_year_end_field))
        if parsed is None:
            raise ValueError(
                f"source={panel_spec.source_id}: invalid fiscal-year-end value for "
                f"firm={canonical_firm_id}, fiscal_year={fiscal_year}"
            )
        return parsed.date()
    month_day = entity_spec.reporting_calendar.firm_exceptions.get(
        canonical_firm_id,
        entity_spec.reporting_calendar.default_fiscal_year_end_month_day,
    )
    month, day = (int(part) for part in month_day.split("-"))
    return date(fiscal_year, month, day)


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()
