"""Typed source-catalog and snapshot models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, cast

Cardinality = Literal["one", "one_or_none", "many"]
SourceFormat = Literal["csv", "tsv", "parquet", "json", "jsonl", "xlsx"]
AvailabilityDateRule = Literal["physical_column", "fiscal_year_plus_one_month_day"]
RowAggregation = Literal["one_row_per_firm_year", "firm_year_presence"]
EvidenceOutcomeMode = Literal["direct_outcome", "positive_indicator"]
EvidenceAbsencePolicy = Literal["unknown"]
DuplicateRepresentativeRule = Literal["identical_signature_then_source_event_id"]


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: non-empty string required")
    return value.strip()


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: list required")
    items = cast(list[object], value)
    if not all(isinstance(item, str) and item.strip() for item in items):
        raise ValueError(f"{context}: non-empty strings required")
    return [cast(str, item).strip() for item in items]


@dataclass(frozen=True)
class DiscoverySpec:
    globs: tuple[str, ...]
    excludes: tuple[str, ...]
    cardinality: Cardinality

    @classmethod
    def from_mapping(cls, value: object, context: str) -> DiscoverySpec:
        raw = _mapping(value, context)
        cardinality = _string(raw.get("cardinality"), f"{context}.cardinality")
        if cardinality not in {"one", "one_or_none", "many"}:
            raise ValueError(f"{context}.cardinality: unsupported {cardinality}")
        return cls(
            globs=tuple(_string_list(raw.get("globs", []), f"{context}.globs")),
            excludes=tuple(_string_list(raw.get("excludes", []), f"{context}.excludes")),
            cardinality=cast(Cardinality, cardinality),
        )


@dataclass(frozen=True)
class ReaderSpec:
    encoding: str
    delimiter: str | None
    sheet_name: str | None
    header_row: int | Literal["auto"] | None

    @classmethod
    def from_mapping(cls, value: object, context: str) -> ReaderSpec:
        raw = _mapping(value if value is not None else {}, context)
        encoding = raw.get("encoding", "utf-8")
        if not isinstance(encoding, str):
            raise ValueError(f"{context}.encoding: string required")
        delimiter = raw.get("delimiter")
        if delimiter is not None and (not isinstance(delimiter, str) or len(delimiter) != 1):
            raise ValueError(f"{context}.delimiter: one character or null required")
        sheet_name = raw.get("sheet_name")
        if sheet_name is not None and not isinstance(sheet_name, str):
            raise ValueError(f"{context}.sheet_name: string or null required")
        header_row = raw.get("header_row")
        if (
            header_row is not None
            and header_row != "auto"
            and (not isinstance(header_row, int) or header_row < 1)
        ):
            raise ValueError(f"{context}.header_row: positive integer, auto, or null required")
        return cls(
            encoding=encoding,
            delimiter=delimiter,
            sheet_name=sheet_name,
            header_row=header_row,
        )


@dataclass(frozen=True)
class PanelProfile:
    enabled: bool
    core_predictor: bool
    contributes_to_firm_master: bool
    availability_date_rule: AvailabilityDateRule
    availability_month_day: str | None
    row_aggregation: RowAggregation

    @classmethod
    def from_mapping(cls, value: object, context: str) -> PanelProfile:
        raw = _mapping(value if value is not None else {}, context)
        enabled = raw.get("enabled", False)
        core = raw.get("core_predictor", False)
        master = raw.get("contributes_to_firm_master", False)
        if not all(isinstance(item, bool) for item in (enabled, core, master)):
            raise ValueError(f"{context}: boolean panel flags required")
        availability_rule = raw.get("availability_date_rule", "physical_column")
        if availability_rule not in {"physical_column", "fiscal_year_plus_one_month_day"}:
            raise ValueError(f"{context}.availability_date_rule: unsupported {availability_rule}")
        availability_month_day = raw.get("availability_month_day")
        if availability_month_day is not None and not isinstance(availability_month_day, str):
            raise ValueError(f"{context}.availability_month_day: string or null required")
        if availability_rule == "fiscal_year_plus_one_month_day":
            if not availability_month_day:
                raise ValueError(
                    f"{context}.availability_month_day: required for derived availability"
                )
            _validate_month_day(availability_month_day, f"{context}.availability_month_day")
        elif availability_month_day is not None:
            raise ValueError(
                f"{context}.availability_month_day: only valid for derived availability"
            )
        row_aggregation = raw.get("row_aggregation", "one_row_per_firm_year")
        if row_aggregation not in {"one_row_per_firm_year", "firm_year_presence"}:
            raise ValueError(f"{context}.row_aggregation: unsupported {row_aggregation}")
        return cls(
            enabled=cast(bool, enabled),
            core_predictor=cast(bool, core),
            contributes_to_firm_master=cast(bool, master),
            availability_date_rule=cast(AvailabilityDateRule, availability_rule),
            availability_month_day=availability_month_day,
            row_aggregation=cast(RowAggregation, row_aggregation),
        )


@dataclass(frozen=True)
class EvidenceProfile:
    outcome_mode: EvidenceOutcomeMode
    row_inclusion_semantic: str | None
    positive_semantic: str | None
    false_indicator_policy: EvidenceAbsencePolicy
    absence_policy: EvidenceAbsencePolicy
    opportunity_semantic: str | None
    duplicate_representative_rule: DuplicateRepresentativeRule

    @classmethod
    def from_mapping(cls, value: object, context: str) -> EvidenceProfile:
        raw = _mapping(value, context)
        outcome_mode = raw.get("outcome_mode")
        if outcome_mode not in {"direct_outcome", "positive_indicator"}:
            raise ValueError(f"{context}.outcome_mode: unsupported {outcome_mode}")
        row_inclusion = raw.get("row_inclusion_semantic")
        if row_inclusion is not None:
            row_inclusion = _string(row_inclusion, f"{context}.row_inclusion_semantic")
        positive_semantic = raw.get("positive_semantic")
        if positive_semantic is not None:
            positive_semantic = _string(positive_semantic, f"{context}.positive_semantic")
        false_indicator_policy = raw.get("false_indicator_policy", "unknown")
        if false_indicator_policy != "unknown":
            raise ValueError(f"{context}.false_indicator_policy must be unknown")
        absence_policy = raw.get("absence_policy")
        if absence_policy != "unknown":
            raise ValueError(f"{context}.absence_policy must be unknown")
        opportunity = raw.get("opportunity_semantic")
        if opportunity is not None:
            opportunity = _string(opportunity, f"{context}.opportunity_semantic")
        duplicate_rule = raw.get("duplicate_representative_rule")
        if duplicate_rule != "identical_signature_then_source_event_id":
            raise ValueError(f"{context}.duplicate_representative_rule: unsupported rule")
        return cls(
            outcome_mode=cast(EvidenceOutcomeMode, outcome_mode),
            row_inclusion_semantic=row_inclusion,
            positive_semantic=positive_semantic,
            false_indicator_policy="unknown",
            absence_policy="unknown",
            opportunity_semantic=opportunity,
            duplicate_representative_rule="identical_signature_then_source_event_id",
        )


def _validate_month_day(value: str, context: str) -> None:
    parts = value.split("-")
    if len(parts) != 2:
        raise ValueError(f"{context}: expected MM-DD")
    try:
        month, day = (int(part) for part in parts)
        date(2000, month, day)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid MM-DD") from exc


@dataclass(frozen=True)
class SourceProfile:
    profile_id: str
    enabled: bool
    required: bool
    discovery: DiscoverySpec
    format: SourceFormat
    reader: ReaderSpec
    channel_id: str
    source_type: str
    source_agency: str
    original_unit: str
    role: str
    verification_status: str
    coverage_dimensions: tuple[str, ...]
    data_risks: tuple[str, ...]
    semantic_fields: dict[str, tuple[str, ...]]
    required_semantic_fields: tuple[str, ...]
    key_semantics: tuple[str, ...]
    date_semantics: tuple[str, ...]
    evidence_mapping: EvidenceProfile | None
    panel_mapping: PanelProfile

    @classmethod
    def from_mapping(cls, profile_id: str, value: object) -> SourceProfile:
        raw = _mapping(value, f"source_catalog.profiles.{profile_id}")
        enabled = raw.get("enabled", True)
        required = raw.get("required", False)
        if not isinstance(enabled, bool) or not isinstance(required, bool):
            raise ValueError(f"profile={profile_id}: enabled/required must be bool")
        format_value = _string(raw.get("format"), f"profile={profile_id}.format").lower()
        if format_value not in {"csv", "tsv", "parquet", "json", "jsonl", "xlsx"}:
            raise ValueError(f"profile={profile_id}: unsupported format {format_value}")
        semantics_raw = _mapping(
            raw.get("semantic_fields", {}), f"profile={profile_id}.semantic_fields"
        )
        semantics = {
            semantic: tuple(
                _string_list(candidates, f"profile={profile_id}.semantic_fields.{semantic}")
            )
            for semantic, candidates in semantics_raw.items()
        }
        role = _string(raw.get("role"), f"profile={profile_id}.role")
        evidence_mapping = (
            EvidenceProfile.from_mapping(
                raw.get("evidence_mapping"), f"profile={profile_id}.evidence_mapping"
            )
            if role == "evidence"
            else None
        )
        if role != "evidence" and raw.get("evidence_mapping") is not None:
            raise ValueError(f"profile={profile_id}: evidence_mapping is only valid for evidence")
        if evidence_mapping is not None:
            if (
                evidence_mapping.row_inclusion_semantic is not None
                and evidence_mapping.row_inclusion_semantic not in semantics
            ):
                raise ValueError(f"profile={profile_id}: row inclusion semantic is not registered")
            if (
                evidence_mapping.opportunity_semantic is not None
                and evidence_mapping.opportunity_semantic not in semantics
            ):
                raise ValueError(f"profile={profile_id}: opportunity semantic is not registered")
            if (
                evidence_mapping.positive_semantic is not None
                and evidence_mapping.positive_semantic not in semantics
            ):
                raise ValueError(f"profile={profile_id}: positive semantic is not registered")
            if evidence_mapping.outcome_mode == "direct_outcome" and "outcome" not in semantics:
                raise ValueError(f"profile={profile_id}: direct outcome semantic is required")
            if (
                evidence_mapping.outcome_mode == "positive_indicator"
                and evidence_mapping.positive_semantic is None
            ):
                raise ValueError(f"profile={profile_id}: positive indicator semantic is required")
        return cls(
            profile_id=profile_id,
            enabled=enabled,
            required=required,
            discovery=DiscoverySpec.from_mapping(
                raw.get("discovery"), f"profile={profile_id}.discovery"
            ),
            format=cast(SourceFormat, format_value),
            reader=ReaderSpec.from_mapping(raw.get("reader", {}), f"profile={profile_id}.reader"),
            channel_id=_string(raw.get("channel_id"), f"profile={profile_id}.channel_id"),
            source_type=_string(raw.get("source_type"), f"profile={profile_id}.source_type"),
            source_agency=_string(raw.get("source_agency"), f"profile={profile_id}.source_agency"),
            original_unit=_string(raw.get("original_unit"), f"profile={profile_id}.original_unit"),
            role=role,
            verification_status=_string(
                raw.get("verification_status"), f"profile={profile_id}.verification_status"
            ),
            coverage_dimensions=tuple(
                _string_list(
                    raw.get("coverage_dimensions", []), f"profile={profile_id}.coverage_dimensions"
                )
            ),
            data_risks=tuple(
                _string_list(raw.get("data_risks", []), f"profile={profile_id}.data_risks")
            ),
            semantic_fields=semantics,
            required_semantic_fields=tuple(
                _string_list(
                    raw.get("required_semantic_fields", []),
                    f"profile={profile_id}.required_semantic_fields",
                )
            ),
            key_semantics=tuple(
                _string_list(raw.get("key_semantics", []), f"profile={profile_id}.key_semantics")
            ),
            date_semantics=tuple(
                _string_list(raw.get("date_semantics", []), f"profile={profile_id}.date_semantics")
            ),
            evidence_mapping=evidence_mapping,
            panel_mapping=PanelProfile.from_mapping(
                raw.get("panel_mapping", {}), f"profile={profile_id}.panel_mapping"
            ),
        )


@dataclass(frozen=True)
class SourceCatalog:
    root_environment_variable: str
    snapshot_schema_version: int
    profiles: tuple[SourceProfile, ...]

    @classmethod
    def from_mapping(cls, value: object) -> SourceCatalog:
        raw = _mapping(value, "source_catalog")
        version = raw.get("snapshot_schema_version")
        if not isinstance(version, int) or version != 1:
            raise ValueError("source_catalog.snapshot_schema_version must equal 1")
        profiles_raw = _mapping(raw.get("profiles"), "source_catalog.profiles")
        profiles = tuple(
            SourceProfile.from_mapping(profile_id, profile)
            for profile_id, profile in sorted(profiles_raw.items())
        )
        return cls(
            root_environment_variable=_string(
                raw.get("root_environment_variable"), "source_catalog.root_environment_variable"
            ),
            snapshot_schema_version=version,
            profiles=profiles,
        )
