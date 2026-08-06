"""Each production source profile must resolve against its actual physical schema."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from snapshot.inspector import resolve_semantics

ROOT = Path(__file__).resolve().parents[2]
FINAL_PATH = "data/source/vn_pipeline_final_firm_year_2015_2025.parquet"
KNOWN_CASE_PATH = "data/source/known_case_registry.csv"

ACTUAL_FINAL_COLUMNS = (
    "firm_master_id",
    "fiscal_year",
    "prediction_time",
    "fs_aud_net_revenue",
    "fs_aud_profit_after_tax",
    "fs_unaud_net_revenue",
    "fs_unaud_profit_after_tax",
    "source_unified_population_sha256",
    "s2_audit_indicator",
    "s2_audit_opinion",
    "s2_audit_firm",
    "s3_broad_endpoint_evidence",
    "s3_broad_source_opportunity",
    "s3_broad_observation_opportunity",
    "s3_broad_provenance_json",
    "s3_reporting_endpoint_evidence",
    "s3_reporting_source_opportunity",
    "s3_reporting_observation_opportunity",
    "s3_reporting_provenance_json",
    "s3_content_endpoint_evidence",
    "s3_content_source_opportunity",
    "s3_content_observation_opportunity",
    "s3_content_provenance_json",
    "s3_timeliness_endpoint_evidence",
    "s3_timeliness_source_opportunity",
    "s3_timeliness_observation_opportunity",
    "s3_timeliness_provenance_json",
)
ACTUAL_KNOWN_CASE_COLUMNS = (
    "case_id",
    "firm_id",
    "fiscal_year",
    "case_construct",
    "role",
    "training_include_flag",
    "calibration_include_flag",
    "model_selection_include_flag",
    "external_validation_include_flag",
    "case_group_id",
    "user_case_label",
    "confirmation_status",
    "known_case_positive",
    "known_case_role",
    "source_presence_status",
    "source_row_count",
    "source_document_ids",
    "source_firm_event_ids",
    "source_row_numbers",
    "source_datasets",
    "move_reason",
)


def _profiles() -> dict[str, dict[str, object]]:
    raw = yaml.safe_load(
        (ROOT / "config/methodology/source_catalog.yaml").read_text(encoding="utf-8")
    )
    catalog = cast(dict[str, object], raw)["source_catalog"]
    assert isinstance(catalog, dict)
    profiles = cast(dict[str, object], catalog)["profiles"]
    assert isinstance(profiles, dict)
    return {
        str(profile_id): cast(dict[str, object], profile)
        for profile_id, profile in cast(dict[object, object], profiles).items()
    }


def _physical_columns(profile: dict[str, object]) -> tuple[str, ...]:
    discovery = cast(dict[str, object], profile["discovery"])
    globs = cast(list[object], discovery["globs"])
    assert len(globs) == 1
    path = str(globs[0])
    if path == FINAL_PATH:
        return ACTUAL_FINAL_COLUMNS
    if path == KNOWN_CASE_PATH:
        return ACTUAL_KNOWN_CASE_COLUMNS
    raise AssertionError(f"unregistered physical schema fixture: {path}")


def test_actual_source_columns_resolve_all_required_semantics() -> None:
    for profile_id, profile in _profiles().items():
        raw_semantics = profile["semantic_fields"]
        assert isinstance(raw_semantics, dict)
        candidates = {
            str(name): tuple(str(value) for value in cast(list[object], options))
            for name, options in cast(dict[object, object], raw_semantics).items()
        }
        required = {str(value) for value in cast(list[object], profile["required_semantic_fields"])}
        resolved = resolve_semantics(_physical_columns(profile), candidates)
        assert required <= set(resolved), (
            profile_id,
            sorted(required - set(resolved)),
        )


def test_final_modeling_parquet_contains_no_known_case_identifiers() -> None:
    assert not any(column.startswith("known_case_") for column in ACTUAL_FINAL_COLUMNS)
