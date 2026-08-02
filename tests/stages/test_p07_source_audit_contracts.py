"""Regression tests for P07 unified-source audit artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from core.schema_registry import ContractRegistry, compile_schemas
from features.panel_source import _build_file_audit, _build_identifier_audit

ROOT = Path(__file__).resolve().parents[2]


def _contracts() -> ContractRegistry:
    columns_raw = yaml.safe_load(
        (ROOT / "config/foundation/columns.yaml").read_text(encoding="utf-8")
    )
    schemas_raw = yaml.safe_load((ROOT / "config/schemas/core.yaml").read_text(encoding="utf-8"))
    artifacts_raw = yaml.safe_load(
        (ROOT / "config/foundation/artifacts.yaml").read_text(encoding="utf-8")
    )
    columns = cast(dict[str, object], columns_raw)["columns"]
    schemas = cast(dict[str, object], schemas_raw)["schemas"]
    artifacts = cast(dict[str, object], artifacts_raw)["artifacts"]
    return ContractRegistry(
        schemas=compile_schemas(schemas, columns),
        artifacts=cast(dict[str, object], artifacts),
    )


def test_unified_file_audit_matches_feature_keyed_artifact_contract(tmp_path: Path) -> None:
    status_frame = pd.DataFrame(
        [
            {"feature_id": "feature_a", "status": "PASS", "reason_code": "computed"},
            {
                "feature_id": "feature_b",
                "status": "UNSUPPORTED",
                "reason_code": "missing_crosswalk",
            },
        ]
    )
    audit = _build_file_audit(
        status_frame=status_frame,
        raw_source_path=tmp_path / "vn_annual_unified_long_2015_2025.parquet",
        raw_rows=445_456,
        usable_rows=445_456,
    )

    assert list(audit.columns) == [
        "feature_id",
        "source_id",
        "relative_path",
        "raw_rows",
        "usable_rows",
        "status",
        "reason_code",
    ]
    assert len(audit) == 2
    assert audit["feature_id"].is_unique
    assert set(audit["source_id"]) == {"financial_statement_core_long"}
    _contracts().validate("feature_store_file_audit", audit)


def test_identifier_audit_leads_with_canonical_physical_key() -> None:
    audit = _build_identifier_audit(
        firm_ids=["F001", "F002"],
        panel_firms={"F001"},
        firm_column="firm_master_id",
    )

    assert list(audit.columns) == [
        "firm_master_id",
        "feature_store_firm_id",
        "canonical_firm_id",
        "mapping_status",
        "ambiguity_flag",
    ]
    assert audit["firm_master_id"].tolist() == ["F001", "F002"]
    assert audit["mapping_status"].tolist() == ["MATCHED", "UNMATCHED"]
    _contracts().validate("feature_store_identifier_crosswalk_audit", audit)
