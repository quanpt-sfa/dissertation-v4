"""The production catalog must resolve the columns emitted by the final data build."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from snapshot.inspector import resolve_semantics

ROOT = Path(__file__).resolve().parents[2]

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
    "known_case_id",
    "known_case_construct",
    "known_case_role",
    "known_case_external_validation_include_flag",
    "known_case_seal_status",
    "known_case_opens_at_step",
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


def test_actual_final_build_columns_resolve_all_required_semantics() -> None:
    for profile_id, profile in _profiles().items():
        raw_semantics = profile["semantic_fields"]
        assert isinstance(raw_semantics, dict)
        candidates = {
            str(name): tuple(str(value) for value in cast(list[object], options))
            for name, options in cast(dict[object, object], raw_semantics).items()
        }
        required = {str(value) for value in cast(list[object], profile["required_semantic_fields"])}
        resolved = resolve_semantics(ACTUAL_FINAL_COLUMNS, candidates)
        assert required <= set(resolved), (
            profile_id,
            sorted(required - set(resolved)),
        )
