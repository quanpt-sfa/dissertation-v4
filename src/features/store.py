"""Validated, deterministic ingestion of the immutable P07 feature store."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

_COMMON_VALUE_COLUMNS = {
    "firm_master_id",
    "fiscal_year",
    "feature_id",
    "available_date",
    "availability_basis",
    "source_snapshot_hash",
    "quality_status",
}
_FORBIDDEN_COLUMN_TOKENS = (
    "outcome",
    "known_case",
    "sanction",
    "outer_fold",
    "post_outcome",
)
_VALID_AVAILABILITY_BASES = {
    "observed_publication_date",
    "synthetic_annual_anchor",
    "prior_year_available",
    "unknown",
}


@dataclass(frozen=True)
class FeatureStoreLoadResult:
    """Validated panel and all ingestion-boundary audits."""

    panel: pd.DataFrame
    validation_report: dict[str, object]
    file_audit: pd.DataFrame
    identifier_audit: pd.DataFrame
    availability_violations: pd.DataFrame
    coverage_audit: pd.DataFrame
    research_decision_audit: pd.DataFrame


@dataclass(frozen=True)
class _ValidatedFeature:
    values: pd.DataFrame
    audit: dict[str, object]


def load_feature_store_config(
    features_config: dict[str, Any], *, repository_root: Path
) -> tuple[dict[str, Any], Path]:
    """Resolve the one canonical store configuration from the compiled registry."""
    raw = features_config.get("store")
    if not isinstance(raw, dict):
        raise ValueError("features.store must be a mapping")
    config = cast(dict[str, Any], raw)
    root_value = config.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("features.store.root is required")
    repository_root = repository_root.resolve()
    store_root = (repository_root / root_value).resolve()
    if store_root == repository_root or repository_root not in store_root.parents:
        raise ValueError("features.store.root escapes the repository")
    if not store_root.is_dir():
        raise FileNotFoundError(f"feature-store root not found: {store_root}")
    return config, store_root


def load_feature_manifest(store_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Read the registered manifest with all values preserved as strings."""
    manifest_value = config.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("features.store.manifest is required")
    manifest_path = _contained_path(store_root, manifest_value)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path, dtype="string", keep_default_na=False)
    required = {
        "feature_id",
        "feature_version",
        "relative_path",
        "row_count",
        "firm_count",
        "min_fiscal_year",
        "max_fiscal_year",
        "duplicate_key_count",
        "file_sha256",
        "source_snapshot_hash",
        "build_config_hash",
        "review_status",
        "build_status",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"feature manifest missing columns: {sorted(missing)}")
    return manifest


def validate_feature_manifest(
    manifest: pd.DataFrame,
    definitions: list[dict[str, object]],
    *,
    store_root: Path,
    config: dict[str, Any],
) -> None:
    """Validate manifest identity, registered paths, and package build metadata."""
    if manifest.empty:
        raise ValueError("feature manifest is empty")
    if manifest["feature_id"].eq("").any() or manifest["feature_id"].duplicated().any():
        raise ValueError("feature manifest contains blank or duplicate feature IDs")
    if manifest["relative_path"].eq("").any() or manifest["relative_path"].duplicated().any():
        raise ValueError("feature manifest contains blank or duplicate paths")
    definition_ids = [str(item.get("feature_id", "")) for item in definitions]
    if any(not value for value in definition_ids) or len(definition_ids) != len(
        set(definition_ids)
    ):
        raise ValueError("feature definitions contain blank or duplicate feature IDs")
    manifest_ids = set(manifest["feature_id"].astype(str))
    if manifest_ids != set(definition_ids):
        missing = sorted(set(definition_ids) - manifest_ids)
        unknown = sorted(manifest_ids - set(definition_ids))
        raise ValueError(
            f"manifest/registry feature mismatch: missing={missing}, unknown={unknown}"
        )
    for relative in manifest["relative_path"].astype(str):
        _contained_path(store_root, relative)

    if not bool(config.get("allow_unregistered_files", False)):
        values_root = config.get("values_root", "values")
        if not isinstance(values_root, str) or not values_root:
            raise ValueError("features.store.values_root must be a non-empty string")
        values_path = _contained_path(store_root, values_root)
        observed = {
            path.relative_to(store_root).as_posix()
            for path in values_path.rglob("*")
            if path.is_file()
        }
        registered = set(manifest["relative_path"].astype(str).str.replace("\\", "/"))
        unknown_files = sorted(observed - registered)
        if unknown_files:
            raise ValueError(f"unregistered feature files present: {unknown_files}")

    build_value = config.get("build_manifest")
    if not isinstance(build_value, str) or not build_value:
        raise ValueError("features.store.build_manifest is required")
    build_path = _contained_path(store_root, build_value)
    if not build_path.is_file():
        raise FileNotFoundError(f"feature build manifest not found: {build_path}")
    raw: object = json.loads(build_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("feature build manifest must be an object")
    build = cast(dict[str, object], raw)
    if build.get("feature_file_count") != len(manifest):
        raise ValueError("feature build manifest file count mismatch")
    snapshot_hashes = set(manifest["source_snapshot_hash"].astype(str))
    config_hashes = set(manifest["build_config_hash"].astype(str))
    if snapshot_hashes != {str(build.get("dataset_sha256", ""))}:
        raise ValueError("source snapshot hash differs across package manifests")
    if config_hashes != {str(build.get("config_sha256", ""))}:
        raise ValueError("build configuration hash differs across package manifests")


def validate_feature_file(
    *,
    store_root: Path,
    manifest_row: dict[str, str],
    definition: dict[str, object],
    config: dict[str, Any],
) -> _ValidatedFeature:
    """Validate and project one scalar feature file without changing source values."""
    feature_id = str(definition["feature_id"])
    path = _contained_path(store_root, manifest_row["relative_path"])
    if not path.is_file():
        raise FileNotFoundError(f"feature={feature_id}: file not found: {path}")
    observed_hash = _sha256(path)
    if (
        bool(config.get("strict_hash_validation", True))
        and observed_hash != manifest_row["file_sha256"]
    ):
        raise ValueError(f"feature={feature_id}: SHA-256 mismatch")
    if str(definition.get("version")) != manifest_row["feature_version"]:
        raise ValueError(f"feature={feature_id}: version mismatch")
    if manifest_row["build_status"] != "PASS":
        raise ValueError(f"feature={feature_id}: manifest build status is not PASS")

    value_column = definition.get("value_column")
    if not isinstance(value_column, str) or not value_column:
        raise ValueError(f"feature={feature_id}: value_column is required")
    projection = pd.read_csv(path, compression="infer", nrows=0).columns.tolist()
    required = _COMMON_VALUE_COLUMNS | {value_column}
    missing = required - set(projection)
    if missing:
        raise ValueError(f"feature={feature_id}: missing columns {sorted(missing)}")
    forbidden = sorted(
        column
        for column in projection
        if any(token in column.lower() for token in _FORBIDDEN_COLUMN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"feature={feature_id}: forbidden columns {forbidden}")

    selected_columns = [
        "firm_master_id",
        "fiscal_year",
        "feature_id",
        value_column,
        "available_date",
        "availability_basis",
        "source_snapshot_hash",
        "quality_status",
    ]
    values = pd.read_csv(
        path,
        compression="infer",
        usecols=selected_columns,
        dtype={
            "firm_master_id": "string",
            "feature_id": "string",
            "available_date": "string",
            "availability_basis": "string",
            "source_snapshot_hash": "string",
            "quality_status": "string",
            value_column: "string",
        },
        keep_default_na=False,
    )
    if len(values) != int(manifest_row["row_count"]):
        raise ValueError(f"feature={feature_id}: row count differs from manifest")
    if set(values["feature_id"].astype(str).unique()) != {feature_id}:
        raise ValueError(f"feature={feature_id}: inconsistent feature_id values")
    years = pd.to_numeric(values["fiscal_year"], errors="coerce")
    if years.isna().any() or not (years % 1 == 0).all():
        raise ValueError(f"feature={feature_id}: invalid fiscal year encoding")
    values["fiscal_year"] = years.astype("int16")
    minimum = int(config.get("allowed_fiscal_year_min", 1900))
    maximum = int(config.get("allowed_fiscal_year_max", 2200))
    if not values["fiscal_year"].between(minimum, maximum).all():
        raise ValueError(f"feature={feature_id}: fiscal year outside registered range")
    if int(values["fiscal_year"].min()) != int(manifest_row["min_fiscal_year"]) or int(
        values["fiscal_year"].max()
    ) != int(manifest_row["max_fiscal_year"]):
        raise ValueError(f"feature={feature_id}: fiscal-year range differs from manifest")
    if values.duplicated(["firm_master_id", "fiscal_year"]).any():
        raise ValueError(f"feature={feature_id}: duplicate firm-year keys")
    if values["firm_master_id"].eq("").any():
        raise ValueError(f"feature={feature_id}: missing firm identifier")
    values["firm_master_id"] = values["firm_master_id"].map(_normalize_identifier)
    if values.duplicated(["firm_master_id", "fiscal_year"]).any():
        raise ValueError(f"feature={feature_id}: duplicate firm-year keys after normalization")
    row_hashes = {value for value in values["source_snapshot_hash"].astype(str).unique() if value}
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in row_hashes):
        raise ValueError(f"feature={feature_id}: invalid row-level lineage hash encoding")
    bases = set(values["availability_basis"].astype(str).unique())
    if not bases.issubset(_VALID_AVAILABILITY_BASES):
        raise ValueError(f"feature={feature_id}: invalid availability basis {sorted(bases)}")

    expected_dtype = str(definition.get("expected_dtype"))
    raw_value = values[value_column].astype("string")
    nonempty = raw_value.ne("")
    if expected_dtype in {"float64", "float32", "int64", "int32", "int16"}:
        parsed = pd.to_numeric(raw_value.mask(~nonempty), errors="coerce")
        if parsed[nonempty].isna().any():
            raise ValueError(f"feature={feature_id}: invalid numeric dtype")
        if expected_dtype.startswith("int") and ((parsed[nonempty] % 1) != 0).any():
            raise ValueError(f"feature={feature_id}: non-integral value for integer feature")
        usable: pd.Series[Any] = parsed.astype("float64")
    elif expected_dtype in {"string", "category"}:
        usable = raw_value.mask(~nonempty).astype("string")
    elif expected_dtype in {"bool", "boolean"}:
        normalized = raw_value.str.lower()
        valid = normalized.isin({"true", "false", "1", "0"}) | ~nonempty
        if not valid.all():
            raise ValueError(f"feature={feature_id}: invalid boolean dtype")
        usable = normalized.map({"true": True, "1": True, "false": False, "0": False})
    else:
        raise ValueError(f"feature={feature_id}: unsupported expected dtype {expected_dtype}")
    values["usable_value"] = usable
    values.loc[values["quality_status"] != "PASS", "usable_value"] = pd.NA
    values["available_date"] = pd.to_datetime(
        values["available_date"].mask(values["available_date"].eq("")), errors="coerce"
    )
    pass_missing_date = values["quality_status"].eq("PASS") & values["available_date"].isna()
    if pass_missing_date.any():
        raise ValueError(f"feature={feature_id}: PASS values have unresolved availability")
    requires_prior = "t_minus_1" in str(
        definition.get("fiscal_period_reference", "")
    ) or "history" in str(definition.get("lag_structure", ""))
    lagged_first_year_usable_count = 0
    if requires_prior:
        first_year = values.groupby("firm_master_id", observed=True)["fiscal_year"].transform("min")
        invalid_first_year = values["usable_value"].notna() & values["fiscal_year"].eq(first_year)
        lagged_first_year_usable_count = int(invalid_first_year.sum())

    audit: dict[str, object] = {
        "feature_id": feature_id,
        "relative_path": manifest_row["relative_path"],
        "file_sha256": observed_hash,
        "feature_version": manifest_row["feature_version"],
        "row_count": len(values),
        "firm_count": int(values["firm_master_id"].nunique()),
        "min_fiscal_year": int(values["fiscal_year"].min()),
        "max_fiscal_year": int(values["fiscal_year"].max()),
        "duplicate_key_count": 0,
        "constant_feature_id": True,
        "expected_dtype": expected_dtype,
        "availability_basis_values": "|".join(sorted(bases)),
        "row_lineage_hash_count": len(row_hashes),
        "blank_row_lineage_hash_count": int(values["source_snapshot_hash"].eq("").sum()),
        "lagged_first_year_usable_count": lagged_first_year_usable_count,
        "technical_validation_status": "PASS",
        "review_status": manifest_row["review_status"],
    }
    return _ValidatedFeature(values=values, audit=audit)


def assemble_feature_input_panel(
    *,
    base_panel: pd.DataFrame,
    feature_definitions: list[dict[str, object]],
    intended_definitions: list[dict[str, object]],
    features_config: dict[str, Any],
    repository_root: Path,
    firm_column: str,
    year_column: str,
    prediction_time_column: str,
) -> FeatureStoreLoadResult:
    """Validate all files and join only LOCKED features to the P02 spine."""
    config, store_root = load_feature_store_config(features_config, repository_root=repository_root)
    definitions = [*feature_definitions, *intended_definitions]
    manifest = load_feature_manifest(store_root, config)
    validate_feature_manifest(manifest, definitions, store_root=store_root, config=config)
    required_base = {firm_column, year_column, prediction_time_column}
    if not required_base.issubset(base_panel.columns):
        raise ValueError(
            f"P02 panel missing feature-store join columns: {sorted(required_base - set(base_panel.columns))}"
        )
    panel = base_panel.copy()
    panel[firm_column] = panel[firm_column].astype("string")
    panel[year_column] = pd.to_numeric(panel[year_column], errors="raise").astype("int16")
    panel[prediction_time_column] = pd.to_datetime(
        panel[prediction_time_column], errors="raise"
    ).dt.tz_localize(None)
    if panel.duplicated([firm_column, year_column]).any():
        raise ValueError("P02 panel has duplicate firm-year keys")
    panel = panel.sort_values([firm_column, year_column], kind="stable").reset_index(drop=True)

    crosswalk, identifier_audit = _identifier_crosswalk(
        store_root=store_root,
        panel=panel,
        firm_column=firm_column,
        year_column=year_column,
    )
    manifest_rows: dict[str, dict[str, str]] = {
        str(row["feature_id"]): {str(key): str(value) for key, value in row.items()}
        for row in manifest.to_dict(orient="records")
    }
    definition_by_id = {str(item["feature_id"]): item for item in definitions}
    file_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    technical_status: dict[str, str] = {}
    feature_columns: dict[str, pd.Series] = {}
    base_key_index = pd.MultiIndex.from_frame(panel[[firm_column, year_column]])
    for feature_id in sorted(manifest_rows):
        definition = definition_by_id[feature_id]
        validated = validate_feature_file(
            store_root=store_root,
            manifest_row=manifest_rows[feature_id],
            definition=definition,
            config=config,
        )
        file_rows.append(validated.audit)
        technical_status[feature_id] = "PASS"
        if definition.get("research_decision_status") != "LOCKED":
            continue

        mapped = validated.values.merge(
            crosswalk,
            on=["firm_master_id", "fiscal_year"],
            how="left",
            validate="many_to_one",
        )
        if mapped["canonical_firm_id"].isna().any():
            raise ValueError(f"feature={feature_id}: identifier absent from registered crosswalk")
        projected = mapped.loc[
            :,
            [
                "canonical_firm_id",
                "fiscal_year",
                "usable_value",
                "available_date",
                "availability_basis",
            ],
        ].rename(columns={"canonical_firm_id": firm_column})
        joined = panel.loc[:, [firm_column, year_column, prediction_time_column]].merge(
            projected,
            on=[firm_column, year_column],
            how="left",
            validate="one_to_one",
        )
        post_anchor = joined["available_date"].notna() & (
            joined["available_date"] > joined[prediction_time_column]
        )
        unknown_availability = joined["usable_value"].notna() & joined["available_date"].isna()
        requires_prior = "t_minus_1" in str(
            definition.get("fiscal_period_reference", "")
        ) or "history" in str(definition.get("lag_structure", ""))
        if requires_prior:
            prior_keys = joined.loc[:, [firm_column, year_column]].copy()
            prior_keys[year_column] = prior_keys[year_column] - 1
            missing_history = joined["usable_value"].notna() & ~pd.MultiIndex.from_frame(
                prior_keys
            ).isin(base_key_index)
        else:
            missing_history = pd.Series(False, index=joined.index)
        for violation_type, mask in (
            ("POST_ANCHOR", post_anchor),
            ("UNRESOLVED_AVAILABILITY", unknown_availability),
        ):
            if not mask.any():
                continue
            for row in joined.loc[mask].itertuples(index=False):
                availability_rows.append(
                    {
                        "feature_id": feature_id,
                        "canonical_firm_id": getattr(row, firm_column),
                        "fiscal_year": int(getattr(row, year_column)),
                        "available_date": getattr(row, "available_date"),
                        "prediction_time": getattr(row, prediction_time_column),
                        "availability_basis": getattr(row, "availability_basis"),
                        "violation_type": violation_type,
                        "action_taken": "VALUE_MASKED_AND_STAGE_BLOCKED",
                    }
                )
            joined.loc[mask, "usable_value"] = pd.NA
        if missing_history.any():
            for row in joined.loc[missing_history].itertuples(index=False):
                availability_rows.append(
                    {
                        "feature_id": feature_id,
                        "canonical_firm_id": getattr(row, firm_column),
                        "fiscal_year": int(getattr(row, year_column)),
                        "available_date": getattr(row, "available_date"),
                        "prediction_time": getattr(row, prediction_time_column),
                        "availability_basis": getattr(row, "availability_basis"),
                        "violation_type": "MISSING_REQUIRED_HISTORY",
                        "action_taken": "VALUE_MASKED_EXPECTED_FIRST_YEAR",
                    }
                )
            joined.loc[missing_history, "usable_value"] = pd.NA
        violation_count = int((post_anchor | unknown_availability).sum())
        if violation_count > int(config.get("availability_violation_tolerance", 0)):
            raise ValueError(
                f"feature={feature_id}: {violation_count} availability violations exceed tolerance"
            )
        feature_columns[feature_id] = joined["usable_value"].reset_index(drop=True)
        valid = joined["usable_value"].notna()
        coverage_rows.append(
            {
                "feature_id": feature_id,
                "row_count": len(panel),
                "nonmissing_count": int(valid.sum()),
                "missing_count": int((~valid).sum()),
                "missing_rate": float((~valid).mean()),
                "firm_count": int(joined.loc[valid, firm_column].nunique()),
                "first_fiscal_year": int(joined.loc[valid, year_column].min())
                if valid.any()
                else pd.NA,
                "last_fiscal_year": int(joined.loc[valid, year_column].max())
                if valid.any()
                else pd.NA,
            }
        )

    if feature_columns:
        panel = pd.concat([panel.reset_index(drop=True), pd.DataFrame(feature_columns)], axis=1)
    panel = panel.sort_values([firm_column, year_column], kind="stable").reset_index(drop=True)
    if panel.duplicated([firm_column, year_column]).any():
        raise ValueError("assembled feature panel has duplicate firm-year keys")
    locked_count = sum(item.get("research_decision_status") == "LOCKED" for item in definitions)
    unresolved_count = sum(
        item.get("research_decision_status") == "RESEARCH_DECISION_REQUIRED" for item in definitions
    )
    expected_total = int(config.get("expected_feature_count", len(definitions)))
    expected_locked = int(config.get("expected_locked_feature_count", locked_count))
    expected_unresolved = int(config.get("expected_unresolved_feature_count", unresolved_count))
    if (
        locked_count != expected_locked
        or unresolved_count != expected_unresolved
        or len(definitions) != expected_total
    ):
        raise ValueError(
            "feature decision counts differ from package contract: "
            f"locked={locked_count}, unresolved={unresolved_count}, total={len(definitions)}"
        )
    research_audit = _research_decision_audit(definitions, technical_status)
    report: dict[str, object] = {
        "status": "FEATURE_STORE_PACKAGE_VALID_WITH_RESEARCH_DECISIONS"
        if unresolved_count
        else "FEATURE_STORE_PACKAGE_VALID",
        "feature_file_count": len(file_rows),
        "locked_feature_count": locked_count,
        "unresolved_feature_count": unresolved_count,
        "validated_feature_count": len(file_rows),
        "failed_feature_count": 0,
        "manifest_sha256": _sha256(_contained_path(store_root, str(config["manifest"]))),
        "build_manifest_sha256": _sha256(
            _contained_path(store_root, str(config["build_manifest"]))
        ),
        "identifier_mapping_method": "registered_year_valid_crosswalk_then_canonical_normalization",
        "synthetic_anchor_is_observed_publication_date": False,
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
        "preprocessing_fit_at_p07": False,
    }
    return FeatureStoreLoadResult(
        panel=panel,
        validation_report=report,
        file_audit=pd.DataFrame(file_rows),
        identifier_audit=identifier_audit,
        availability_violations=_availability_frame(availability_rows),
        coverage_audit=pd.DataFrame(coverage_rows),
        research_decision_audit=research_audit,
    )


def _identifier_crosswalk(
    *, store_root: Path, panel: pd.DataFrame, firm_column: str, year_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = _contained_path(store_root, "mappings/firm_ticker_crosswalk.csv")
    if not path.is_file():
        raise FileNotFoundError(f"feature-store identifier crosswalk not found: {path}")
    raw = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {
        "issuer_ticker",
        "firm_master_id",
        "valid_from_year",
        "valid_to_year",
        "mapping_method",
        "review_status",
    }
    if not required.issubset(raw.columns):
        raise ValueError("feature-store identifier crosswalk schema is incomplete")
    if not raw["review_status"].eq("LOCKED").all():
        raise ValueError("feature-store identifier crosswalk contains unlocked mappings")
    expanded_rows: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        start, end = int(str(row.valid_from_year)), int(str(row.valid_to_year))
        if start > end:
            raise ValueError("identifier crosswalk contains an inverted year range")
        source = _normalize_identifier(str(row.issuer_ticker))
        canonical = _normalize_identifier(str(row.firm_master_id))
        for fiscal_year in range(start, end + 1):
            expanded_rows.append(
                {
                    "firm_master_id": source,
                    "fiscal_year": fiscal_year,
                    "canonical_firm_id": canonical,
                    "mapping_method": str(row.mapping_method),
                }
            )
    expanded = pd.DataFrame(expanded_rows)
    conflicts = expanded.groupby(["firm_master_id", "fiscal_year"])["canonical_firm_id"].nunique()
    if (conflicts > 1).any() or expanded.duplicated(["firm_master_id", "fiscal_year"]).any():
        raise ValueError("identifier crosswalk contains overlapping or ambiguous mappings")
    base = panel.loc[:, [firm_column, year_column]].copy()
    base["normalized_canonical"] = base[firm_column].map(_normalize_identifier)
    if base.duplicated(["normalized_canonical", year_column]).any():
        raise ValueError("P02 identifiers collide after registered normalization")
    base_keys = set(
        zip(base["normalized_canonical"].astype(str), base[year_column].astype(int), strict=True)
    )
    audit = expanded.rename(
        columns={
            "firm_master_id": "feature_store_firm_id",
            "mapping_method": "source_mapping_method",
        }
    ).copy()
    audit["mapping_method"] = "registered_crosswalk_normalized_identity"
    audit["mapping_status"] = [
        "MATCHED" if (str(firm), int(year)) in base_keys else "UNMATCHED_P02"
        for firm, year in zip(audit["canonical_firm_id"], audit["fiscal_year"], strict=True)
    ]
    audit["ambiguity_flag"] = False
    audit["source_registry"] = "feature_store/mappings/firm_ticker_crosswalk.csv"
    audit.insert(0, firm_column, audit["canonical_firm_id"])
    result = expanded.loc[:, ["firm_master_id", "fiscal_year", "canonical_firm_id"]]
    result["fiscal_year"] = result["fiscal_year"].astype("int16")
    return result, audit.loc[
        :,
        [
            firm_column,
            "feature_store_firm_id",
            "canonical_firm_id",
            "fiscal_year",
            "mapping_method",
            "mapping_status",
            "ambiguity_flag",
            "source_registry",
        ],
    ]


def _research_decision_audit(
    definitions: list[dict[str, object]], technical_status: dict[str, str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in sorted(definitions, key=lambda value: str(value["feature_id"])):
        feature_id = str(item["feature_id"])
        unresolved = item.get("research_decision_status") != "LOCKED"
        rows.append(
            {
                "feature_id": feature_id,
                "feature_group": item.get("theoretical_block"),
                "research_decision_status": item.get("research_decision_status"),
                "confirmatory_status": item.get("confirmatory_status"),
                "model_eligibility": item.get("model_eligibility"),
                "availability_reason_code": item.get("availability_reason_code"),
                "source_of_status": "registry_config",
                "technical_validation_status": technical_status.get(feature_id, "NOT_VALIDATED"),
                "permitted_views": "technical_validation|coverage_audit|descriptive_mapping_audit"
                if unresolved
                else "target_specific_registry_views",
                "unresolved_issue": "BENEISH_MAPPING_AND_DENOMINATOR_POLICY"
                if feature_id.startswith("beneish_")
                else "none",
                "pipeline_may_proceed_without_feature": unresolved,
            }
        )
    return pd.DataFrame(rows)


def _availability_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "feature_id",
        "canonical_firm_id",
        "fiscal_year",
        "available_date",
        "prediction_time",
        "availability_basis",
        "violation_type",
        "action_taken",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["feature_id"] = frame["feature_id"].astype("string")
    frame["canonical_firm_id"] = frame["canonical_firm_id"].astype("string")
    frame["fiscal_year"] = frame["fiscal_year"].astype("int16")
    frame["available_date"] = pd.to_datetime(frame["available_date"])
    frame["prediction_time"] = pd.to_datetime(frame["prediction_time"])
    for column in ("availability_basis", "violation_type", "action_taken"):
        frame[column] = frame[column].astype("string")
    return frame


def _normalize_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    normalized = re.sub(r"\s+", " ", normalized)
    if not normalized:
        raise ValueError("firm identifier becomes empty after normalization")
    return normalized


def _contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    root = root.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"feature-store path escapes registered root: {relative}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
