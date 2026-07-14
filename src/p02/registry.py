"""Resolve P02 source and entity-resolution configuration."""

from __future__ import annotations

from typing import Any, cast

from p01.models import SourceSpec

from .models import EntityResolutionSpec, PanelSourceSpec


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def load_p02_specs(
    registry: dict[str, Any],
) -> tuple[list[tuple[SourceSpec, PanelSourceSpec]], EntityResolutionSpec, int, int]:
    data_sources = _mapping(registry.get("data_sources"), "data_sources")
    source_registry = _mapping(data_sources.get("source_registry"), "data_sources.source_registry")
    sources = _mapping(source_registry.get("sources"), "data_sources.source_registry.sources")
    enabled: list[tuple[SourceSpec, PanelSourceSpec]] = []
    for source_id, raw_source in sorted(sources.items()):
        source_spec = SourceSpec.from_mapping(source_id, raw_source)
        if not source_spec.enabled:
            continue
        if not isinstance(raw_source, dict):
            raise ValueError(f"source={source_id}: mapping required")
        source_mapping = cast(dict[str, object], raw_source)
        panel_mapping = source_mapping.get("panel_mapping")
        if not isinstance(panel_mapping, dict):
            continue
        panel = cast(dict[str, object], panel_mapping)
        panel_enabled = panel.get("enabled", True)
        if not isinstance(panel_enabled, bool):
            raise ValueError(f"source={source_id}.panel_mapping.enabled: bool required")
        if not panel_enabled:
            continue
        enabled.append(
            (source_spec, PanelSourceSpec.from_source_mapping(source_id, source_mapping))
        )
    if not enabled:
        raise ValueError("no enabled source is registered for P02")
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    study = _mapping(registry.get("study"), "study")
    sample = _mapping(study.get("sample_fiscal_years"), "study.sample_fiscal_years")
    start = sample.get("start")
    end = sample.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("study.sample_fiscal_years start/end must be integers")
    return enabled, entity, start, end


def output_column_names(registry: dict[str, Any]) -> dict[str, str]:
    columns = _mapping(registry.get("columns"), "columns")
    result: dict[str, str] = {}
    for logical in ("firm_id", "fiscal_year", "prediction_time"):
        column = _mapping(columns.get(logical), f"columns.{logical}")
        physical = column.get("physical_name")
        if not isinstance(physical, str) or not physical:
            raise ValueError(f"columns.{logical}.physical_name is required")
        result[logical] = physical
    return result
