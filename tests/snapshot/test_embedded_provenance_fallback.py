from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from snapshot.builder import _load_extract_provenance  # pyright: ignore[reportPrivateUsage]


def _registry() -> dict[str, object]:
    return {
        "data_sources": {
            "source_registry": {
                "provenance_contract": {
                    "manifest_relative_path": "extract_provenance.json",
                    "required": True,
                    "required_fields": [
                        "vendor",
                        "vendor_product",
                        "pull_date",
                        "vendor_version",
                        "extract_query",
                        "revision_policy",
                        "point_in_time_vintages_available",
                    ],
                }
            }
        }
    }


def _write_final_input(raw_root: Path) -> None:
    target = raw_root / "data" / "source" / "vn_pipeline_final_firm_year_2015_2025.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    embedded = {
        "protocol_hash": "old-protocol",
        "unified_population_source": {
            "sha256": "abc123",
        },
    }
    frame = pd.DataFrame(
        {
            "source_provenance_json": [json.dumps(embedded), json.dumps(embedded)],
            "source_protocol_hash": ["old-protocol", "old-protocol"],
            "source_run_id": ["old-run", "old-run"],
            "source_unified_population_sha256": ["abc123", "abc123"],
        }
    )
    frame.to_parquet(target, index=False)


def test_missing_external_manifest_uses_embedded_derived_input_lineage(tmp_path: Path) -> None:
    _write_final_input(tmp_path)

    provenance = _load_extract_provenance(_registry(), tmp_path)

    assert provenance["provenance_origin"] == "embedded_derived_input_lineage"
    assert provenance["vendor_metadata_status"] == "not_retained_in_derived_input"
    assert provenance["vendor"] == "UNAVAILABLE_NOT_RETAINED_IN_DERIVED_INPUT"
    assert provenance["source_protocol_hash"] == "old-protocol"
    assert provenance["source_run_id"] == "old-run"
    assert provenance["source_unified_population_sha256"] == "abc123"
    assert provenance["embedded_source_provenance"] == {
        "protocol_hash": "old-protocol",
        "unified_population_source": {"sha256": "abc123"},
    }
    assert isinstance(provenance["derived_input_sha256"], str)


def test_external_manifest_takes_precedence_over_embedded_lineage(tmp_path: Path) -> None:
    _write_final_input(tmp_path)
    external = {
        "vendor": "FiinPro",
        "vendor_product": "annual",
        "pull_date": "2026-08-01",
        "vendor_version": "unknown",
        "extract_query": "controlled-export",
        "revision_policy": "as-exported",
        "point_in_time_vintages_available": False,
    }
    (tmp_path / "extract_provenance.json").write_text(
        json.dumps(external),
        encoding="utf-8",
    )

    provenance = _load_extract_provenance(_registry(), tmp_path)

    assert provenance == external


def test_missing_all_provenance_evidence_still_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="required provenance evidence is unavailable"):
        _load_extract_provenance(_registry(), tmp_path)
