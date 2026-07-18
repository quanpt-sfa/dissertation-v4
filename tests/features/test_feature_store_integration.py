from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from features.store import (
    assemble_feature_input_panel,
    load_feature_manifest,
    validate_feature_file,
    validate_feature_manifest,
)


def _definition(feature_id: str = "feature_a", **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "feature_id": feature_id,
        "version": 1,
        "value_column": "value_numeric",
        "expected_dtype": "float64",
        "research_decision_status": "LOCKED",
        "confirmatory_status": "confirmatory",
        "model_eligibility": "eligible",
        "availability_reason_code": "SYNTHETIC_ANNUAL_ANCHOR",
        "theoretical_block": "accounting_content",
    }
    return {**value, **overrides}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package(
    tmp_path: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    definitions: list[dict[str, object]] | None = None,
) -> tuple[Path, dict[str, Any], list[dict[str, object]]]:
    repository = tmp_path / "repo"
    store = repository / "store"
    values = store / "values" / "atomic"
    manifests = store / "manifests"
    mappings = store / "mappings"
    values.mkdir(parents=True)
    manifests.mkdir()
    mappings.mkdir()
    definitions = definitions or [_definition()]
    rows = rows or [
        {
            "firm_master_id": " A ",
            "fiscal_year": 2023,
            "feature_id": "feature_a",
            "value_numeric": 1.5,
            "available_date": "2024-03-31",
            "availability_basis": "synthetic_annual_anchor",
            "source_snapshot_hash": "a" * 64,
            "quality_status": "PASS",
        }
    ]
    manifest_rows: list[dict[str, object]] = []
    for definition in definitions:
        feature_id = str(definition["feature_id"])
        feature_rows = [row for row in rows if row["feature_id"] == feature_id]
        path = values / f"{feature_id}.csv.gz"
        pd.DataFrame(feature_rows).to_csv(path, index=False, compression="gzip")
        manifest_rows.append(
            {
                "feature_id": feature_id,
                "feature_version": str(definition["version"]),
                "relative_path": f"values/atomic/{feature_id}.csv.gz",
                "row_count": len(feature_rows),
                "firm_count": len({row["firm_master_id"] for row in feature_rows}),
                "min_fiscal_year": min(int(str(row["fiscal_year"])) for row in feature_rows),
                "max_fiscal_year": max(int(str(row["fiscal_year"])) for row in feature_rows),
                "duplicate_key_count": 0,
                "file_sha256": _sha(path),
                "source_snapshot_hash": "a" * 64,
                "build_config_hash": "b" * 64,
                "review_status": "LOCKED"
                if definition["research_decision_status"] == "LOCKED"
                else "UNRESOLVED",
                "build_status": "PASS",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(manifests / "feature_files_manifest.csv", index=False)
    (manifests / "feature_build_manifest.json").write_text(
        json.dumps(
            {
                "feature_file_count": len(definitions),
                "dataset_sha256": "a" * 64,
                "config_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    store_ids = sorted({str(row["firm_master_id"]).strip() for row in rows})
    pd.DataFrame(
        [
            {
                "issuer_ticker": identifier,
                "firm_master_id": identifier,
                "valid_from_year": 2015,
                "valid_to_year": 2025,
                "mapping_method": "issuer_ticker_identity",
                "review_status": "LOCKED",
            }
            for identifier in store_ids
        ]
    ).to_csv(mappings / "firm_ticker_crosswalk.csv", index=False)
    config: dict[str, Any] = {
        "store": {
            "root": "store",
            "manifest": "manifests/feature_files_manifest.csv",
            "build_manifest": "manifests/feature_build_manifest.json",
            "values_root": "values",
            "strict_hash_validation": True,
            "allow_unregistered_files": False,
            "allowed_fiscal_year_min": 2015,
            "allowed_fiscal_year_max": 2025,
            "availability_violation_tolerance": 0,
            "expected_feature_count": len(definitions),
            "expected_locked_feature_count": sum(
                value["research_decision_status"] == "LOCKED" for value in definitions
            ),
            "expected_unresolved_feature_count": sum(
                value["research_decision_status"] == "RESEARCH_DECISION_REQUIRED"
                for value in definitions
            ),
        }
    }
    return repository, config, definitions


def _base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "firm_master_id": pd.Series(["A", "B"], dtype="string"),
            "fiscal_year": pd.Series([2023, 2023], dtype="int16"),
            "prediction_time": pd.to_datetime(["2024-03-31", "2024-03-31"]),
        }
    )


def _assemble(
    repository: Path,
    config: dict[str, Any],
    definitions: list[dict[str, object]],
    *,
    base: pd.DataFrame | None = None,
):
    locked = [value for value in definitions if value["research_decision_status"] == "LOCKED"]
    intended = [value for value in definitions if value not in locked]
    return assemble_feature_input_panel(
        base_panel=_base() if base is None else base,
        feature_definitions=locked,
        intended_definitions=intended,
        features_config=config,
        repository_root=repository,
        firm_column="firm_master_id",
        year_column="fiscal_year",
        prediction_time_column="prediction_time",
    )


def test_exact_normalized_mapping_and_missingness_are_preserved(tmp_path: Path) -> None:
    repository, config, definitions = _package(tmp_path)
    result = _assemble(repository, config, definitions)
    assert result.panel["feature_a"].tolist()[0] == pytest.approx(1.5)
    assert pd.isna(result.panel["feature_a"].tolist()[1])
    assert result.identifier_audit["mapping_status"].eq("MATCHED").any()
    assert result.availability_violations.empty
    assert result.validation_report["status"] == "FEATURE_STORE_PACKAGE_VALID"


def test_missing_manifest_fails(tmp_path: Path) -> None:
    repository, config, _ = _package(tmp_path)
    (repository / "store/manifests/feature_files_manifest.csv").unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        load_feature_manifest(repository / "store", config["store"])


def test_missing_file_and_hash_mismatch_fail(tmp_path: Path) -> None:
    repository, config, definitions = _package(tmp_path)
    store = repository / "store"
    manifest = load_feature_manifest(store, config["store"])
    row = {str(key): str(value) for key, value in manifest.iloc[0].to_dict().items()}
    path = store / row["relative_path"]
    path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_feature_file(
            store_root=store,
            manifest_row=row,
            definition=definitions[0],
            config=config["store"],
        )
    repository, config, definitions = _package(tmp_path / "second")
    store = repository / "store"
    manifest = load_feature_manifest(store, config["store"])
    row = {str(key): str(value) for key, value in manifest.iloc[0].to_dict().items()}
    (store / row["relative_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_feature_file(
            store_root=store,
            manifest_row=row,
            definition=definitions[0],
            config=config["store"],
        )


@pytest.mark.parametrize("field", ["feature_id", "relative_path"])
def test_duplicate_manifest_identity_fails(tmp_path: Path, field: str) -> None:
    repository, config, definitions = _package(tmp_path)
    store = repository / "store"
    manifest = load_feature_manifest(store, config["store"])
    duplicate = pd.concat([manifest, manifest], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_feature_manifest(
            duplicate,
            definitions,
            store_root=store,
            config=config["store"],
        )


def test_unregistered_feature_file_fails(tmp_path: Path) -> None:
    repository, config, definitions = _package(tmp_path)
    store = repository / "store"
    (store / "values/atomic/unknown.csv.gz").write_bytes(b"unknown")
    manifest = load_feature_manifest(store, config["store"])
    with pytest.raises(ValueError, match="unregistered"):
        validate_feature_manifest(
            manifest,
            definitions,
            store_root=store,
            config=config["store"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("feature_id", "inconsistent feature_id"),
        ("version", "version mismatch"),
        ("dtype", "numeric dtype"),
        ("year", "fiscal year"),
        ("duplicate", "duplicate firm-year"),
    ],
)
def test_file_contract_failures(tmp_path: Path, mutation: str, message: str) -> None:
    repository, config, definitions = _package(tmp_path)
    store = repository / "store"
    manifest = load_feature_manifest(store, config["store"])
    row = {str(key): str(value) for key, value in manifest.iloc[0].to_dict().items()}
    definition = dict(definitions[0])
    if mutation == "version":
        definition["version"] = 2
    else:
        value_path = store / row["relative_path"]
        values = pd.read_csv(value_path)
        if mutation == "feature_id":
            values.loc[0, "feature_id"] = "other"
        elif mutation == "dtype":
            values["value_numeric"] = values["value_numeric"].astype("object")
            values.loc[0, "value_numeric"] = "not-a-number"
        elif mutation == "year":
            values.loc[0, "fiscal_year"] = 1800
        elif mutation == "duplicate":
            values = pd.concat([values, values], ignore_index=True)
            row["row_count"] = "2"
        values.to_csv(value_path, index=False, compression="gzip")
        row["file_sha256"] = _sha(value_path)
    with pytest.raises(ValueError, match=message):
        validate_feature_file(
            store_root=store,
            manifest_row=row,
            definition=definition,
            config=config["store"],
        )


def test_post_anchor_value_is_blocked(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {
            "firm_master_id": "A",
            "fiscal_year": 2023,
            "feature_id": "feature_a",
            "value_numeric": 1.0,
            "available_date": "2024-04-01",
            "availability_basis": "synthetic_annual_anchor",
            "source_snapshot_hash": "a" * 64,
            "quality_status": "PASS",
        }
    ]
    repository, config, definitions = _package(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="availability violations"):
        _assemble(repository, config, definitions)


def test_lagged_feature_requires_prior_year_history(tmp_path: Path) -> None:
    definitions = [
        _definition(
            fiscal_period_reference="fiscal_year_t_and_t_minus_1",
            lag_structure="one_fiscal_year_history_required",
        )
    ]
    repository, config, definitions = _package(tmp_path, definitions=definitions)
    result = _assemble(repository, config, definitions)
    assert result.panel["feature_a"].isna().all()
    assert set(result.availability_violations["violation_type"]) == {"MISSING_REQUIRED_HISTORY"}


def test_unresolved_feature_is_validated_but_not_joined(tmp_path: Path) -> None:
    definitions = [
        _definition(),
        _definition(
            "beneish_dsri",
            research_decision_status="RESEARCH_DECISION_REQUIRED",
            confirmatory_status="blocked",
            model_eligibility="blocked_until_locked",
        ),
    ]
    rows: list[dict[str, object]] = [
        {
            "firm_master_id": "A",
            "fiscal_year": 2023,
            "feature_id": feature,
            "value_numeric": 1.0,
            "available_date": "2024-03-31",
            "availability_basis": "synthetic_annual_anchor",
            "source_snapshot_hash": "a" * 64,
            "quality_status": "PASS",
        }
        for feature in ("feature_a", "beneish_dsri")
    ]
    repository, config, definitions = _package(tmp_path, rows=rows, definitions=definitions)
    result = _assemble(repository, config, definitions)
    assert "feature_a" in result.panel
    assert "beneish_dsri" not in result.panel
    decision = result.research_decision_audit.set_index("feature_id").loc["beneish_dsri"]
    assert str(decision["source_of_status"]) == "registry_config"
    assert str(decision["technical_validation_status"]) == "PASS"


def test_ambiguous_or_overlapping_crosswalk_fails(tmp_path: Path) -> None:
    repository, config, definitions = _package(tmp_path)
    path = repository / "store/mappings/firm_ticker_crosswalk.csv"
    crosswalk = pd.read_csv(path)
    duplicate = crosswalk.copy()
    duplicate["firm_master_id"] = "OTHER"
    pd.concat([crosswalk, duplicate], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlapping or ambiguous"):
        _assemble(repository, config, definitions)


def test_deterministic_panel_order_and_hash(tmp_path: Path) -> None:
    repository, config, definitions = _package(tmp_path)
    first = _assemble(repository, config, definitions).panel
    second = _assemble(repository, config, definitions).panel
    pd.testing.assert_frame_equal(first, second)
    first_hash = hashlib.sha256(first.to_csv(index=False).encode()).hexdigest()
    second_hash = hashlib.sha256(second.to_csv(index=False).encode()).hexdigest()
    assert first_hash == second_hash
    assert not first.duplicated(["firm_master_id", "fiscal_year"]).any()
