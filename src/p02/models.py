"""Typed P02 configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: non-empty string required")
    return value.strip()


def _bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context}: bool required")
    return value


@dataclass(frozen=True)
class PanelSourceSpec:
    source_id: str
    firm_id_field: str
    fiscal_year_field: str
    availability_date_field: str
    fiscal_year_end_field: str | None
    ticker_field: str | None
    contributes_to_firm_master: bool
    core_predictor: bool
    availability_date_rule: str
    availability_month_day: str | None
    row_aggregation: str

    @classmethod
    def from_source_mapping(cls, source_id: str, value: object) -> PanelSourceSpec:
        source = _mapping(value, f"source={source_id}")
        panel = _mapping(source.get("panel_mapping"), f"source={source_id}.panel_mapping")
        fiscal_year_end = panel.get("fiscal_year_end_field")
        ticker = panel.get("ticker_field")
        if fiscal_year_end is not None and not isinstance(fiscal_year_end, str):
            raise ValueError(
                f"source={source_id}.panel_mapping.fiscal_year_end_field: string or null required"
            )
        if ticker is not None and not isinstance(ticker, str):
            raise ValueError(
                f"source={source_id}.panel_mapping.ticker_field: string or null required"
            )
        firm_id_field = _string(
            panel.get("firm_id_field"), f"source={source_id}.panel_mapping.firm_id_field"
        )
        if ticker is not None and firm_id_field == ticker:
            raise ValueError(
                f"source={source_id}: ticker cannot be used as immutable firm identity"
            )
        registered_availability = _string(
            source.get("availability_date_field"), f"source={source_id}.availability_date_field"
        )
        panel_availability = _string(
            panel.get("availability_date_field", registered_availability),
            f"source={source_id}.panel_mapping.availability_date_field",
        )
        if panel_availability != registered_availability:
            raise ValueError(
                f"source={source_id}: panel availability field must equal the P01 registered field"
            )
        availability_rule = _string(
            panel.get("availability_date_rule", "physical_column"),
            f"source={source_id}.panel_mapping.availability_date_rule",
        )
        if availability_rule not in {"physical_column", "fiscal_year_plus_one_month_day"}:
            raise ValueError(
                f"source={source_id}: unsupported availability rule {availability_rule}"
            )
        availability_month_day = panel.get("availability_month_day")
        if availability_month_day is not None and not isinstance(availability_month_day, str):
            raise ValueError(
                f"source={source_id}.panel_mapping.availability_month_day: string or null required"
            )
        if availability_rule == "fiscal_year_plus_one_month_day":
            if not availability_month_day:
                raise ValueError(
                    f"source={source_id}: derived availability requires availability_month_day"
                )
            _validate_month_day(availability_month_day, "availability_month_day")
        row_aggregation = _string(
            panel.get("row_aggregation", "one_row_per_firm_year"),
            f"source={source_id}.panel_mapping.row_aggregation",
        )
        if row_aggregation not in {"one_row_per_firm_year", "firm_year_presence"}:
            raise ValueError(f"source={source_id}: unsupported row aggregation {row_aggregation}")
        return cls(
            source_id=source_id,
            firm_id_field=firm_id_field,
            fiscal_year_field=_string(
                panel.get("fiscal_year_field"),
                f"source={source_id}.panel_mapping.fiscal_year_field",
            ),
            availability_date_field=panel_availability,
            fiscal_year_end_field=fiscal_year_end,
            ticker_field=ticker,
            contributes_to_firm_master=_bool(
                panel.get("contributes_to_firm_master", True),
                f"source={source_id}.panel_mapping.contributes_to_firm_master",
            ),
            core_predictor=_bool(
                panel.get("core_predictor", False),
                f"source={source_id}.panel_mapping.core_predictor",
            ),
            availability_date_rule=availability_rule,
            availability_month_day=availability_month_day,
            row_aggregation=row_aggregation,
        )


@dataclass(frozen=True)
class NormalizationSpec:
    unicode_nfkc: bool
    trim: bool
    uppercase: bool
    collapse_internal_whitespace: bool


@dataclass(frozen=True)
class ReportingCalendarSpec:
    default_fiscal_year_end_month_day: str
    firm_exceptions: dict[str, str]
    early_report_exceptions: frozenset[tuple[str, int, str]]


@dataclass(frozen=True)
class EntityResolutionSpec:
    policy: str
    allow_identity_mapping: bool
    collision_policy: str
    normalization: NormalizationSpec
    aliases: dict[str, dict[str, str]]
    reporting_calendar: ReportingCalendarSpec

    @classmethod
    def from_mapping(cls, value: object) -> EntityResolutionSpec:
        raw = _mapping(value, "entity_resolution")
        policy = _string(raw.get("policy"), "entity_resolution.policy")
        if policy != "registered_only":
            raise ValueError("P02 requires entity_resolution.policy=registered_only")
        collision_policy = _string(
            raw.get("collision_policy", "fail"), "entity_resolution.collision_policy"
        )
        if collision_policy != "fail":
            raise ValueError("P02 currently requires collision_policy=fail")
        normalization_raw = _mapping(
            raw.get("normalization", {}), "entity_resolution.normalization"
        )
        normalization = NormalizationSpec(
            unicode_nfkc=_bool(
                normalization_raw.get("unicode_nfkc", True),
                "entity_resolution.normalization.unicode_nfkc",
            ),
            trim=_bool(
                normalization_raw.get("trim", True),
                "entity_resolution.normalization.trim",
            ),
            uppercase=_bool(
                normalization_raw.get("uppercase", True),
                "entity_resolution.normalization.uppercase",
            ),
            collapse_internal_whitespace=_bool(
                normalization_raw.get("collapse_internal_whitespace", True),
                "entity_resolution.normalization.collapse_internal_whitespace",
            ),
        )
        aliases_raw = _mapping(raw.get("aliases", {}), "entity_resolution.aliases")
        aliases: dict[str, dict[str, str]] = {}
        for source_id, raw_aliases in aliases_raw.items():
            alias_map = _mapping(raw_aliases, f"entity_resolution.aliases.{source_id}")
            aliases[source_id] = {
                _string(alias, f"entity_resolution.aliases.{source_id}.alias"): _string(
                    canonical, f"entity_resolution.aliases.{source_id}.{alias}"
                )
                for alias, canonical in alias_map.items()
            }
        calendar_raw = _mapping(
            raw.get("reporting_calendar", {}), "entity_resolution.reporting_calendar"
        )
        default_month_day = _string(
            calendar_raw.get("default_fiscal_year_end_month_day", "12-31"),
            "entity_resolution.reporting_calendar.default_fiscal_year_end_month_day",
        )
        _validate_month_day(default_month_day, "default_fiscal_year_end_month_day")
        firm_exceptions_raw = _mapping(
            calendar_raw.get("firm_exceptions", {}),
            "entity_resolution.reporting_calendar.firm_exceptions",
        )
        firm_exceptions: dict[str, str] = {}
        for firm_id, month_day in firm_exceptions_raw.items():
            canonical = _string(firm_id, "entity_resolution.reporting_calendar.firm_exceptions")
            parsed = _string(
                month_day,
                f"entity_resolution.reporting_calendar.firm_exceptions.{canonical}",
            )
            _validate_month_day(parsed, canonical)
            firm_exceptions[canonical] = parsed
        early_raw = calendar_raw.get("early_report_exceptions", [])
        if not isinstance(early_raw, list):
            raise ValueError(
                "entity_resolution.reporting_calendar.early_report_exceptions: list required"
            )
        early: set[tuple[str, int, str]] = set()
        for index, item in enumerate(cast(list[object], early_raw)):
            entry = _mapping(item, f"early_report_exceptions[{index}]")
            firm_id = _string(entry.get("firm_id"), f"early_report_exceptions[{index}].firm_id")
            fiscal_year = entry.get("fiscal_year")
            source_id = _string(
                entry.get("source_id"), f"early_report_exceptions[{index}].source_id"
            )
            if not isinstance(fiscal_year, int):
                raise ValueError(f"early_report_exceptions[{index}].fiscal_year: int required")
            early.add((firm_id, fiscal_year, source_id))
        allow_identity = raw.get("allow_identity_mapping", True)
        if not isinstance(allow_identity, bool):
            raise ValueError("entity_resolution.allow_identity_mapping: bool required")
        return cls(
            policy=policy,
            allow_identity_mapping=allow_identity,
            collision_policy=collision_policy,
            normalization=normalization,
            aliases=aliases,
            reporting_calendar=ReportingCalendarSpec(
                default_fiscal_year_end_month_day=default_month_day,
                firm_exceptions=firm_exceptions,
                early_report_exceptions=frozenset(early),
            ),
        )


def _validate_month_day(value: str, context: str) -> None:
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError(f"{context}: expected MM-DD")
    try:
        month, day = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{context}: expected numeric MM-DD") from exc
    try:
        from datetime import date

        date(2000, month, day)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid MM-DD") from exc
