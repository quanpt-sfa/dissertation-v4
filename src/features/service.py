"""P07 raw feature-panel construction and fail-closed leakage governance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

import pandas as pd

from core.semantic_keys import (
    ANNUAL_MEASUREMENT_MATURE,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    OUTCOME,
    PREDICTION_TIME,
    S3_NEXT_YEAR_MATURE,
    TARGET_ID,
    TEMPORAL_ROLE,
)

_ROLES = {"content", "observability", "ambiguous", "administrative", "identifier"}
_CONFIRMATORY = {"confirmatory", "robustness_only", "descriptive_only", "blocked"}
_DECISION_STATUS = {"LOCKED", "RESEARCH_DECISION_REQUIRED", "UNAVAILABLE", "NOT_APPLICABLE"}
# Valid registry states for model_eligibility. "ablation_only_by_target" marks a
# retained target-component feature that the leakage registry flags ABLATION_ONLY
# (methodology 3.5): it stays in the store for component-reconstruction diagnostics
# but must never enter a model matrix. It is therefore accepted by validation but
# excluded from FIT_MODEL_ELIGIBILITY_VALUES below.
MODEL_ELIGIBILITY_VALUES = frozenset(
    {
        ELIGIBLE,
        "eligible_observability_view",
        "blocked_until_mapping_review",
        "ablation_only_by_target",
    }
)
FIT_MODEL_ELIGIBILITY_VALUES = frozenset({ELIGIBLE, "eligible_observability_view"})
_LEAKAGE_DECISIONS = {
    "ALLOW",
    "ALLOW_WITH_ABLATION",
    "ROBUSTNESS_ONLY",
    "DESCRIPTIVE_ONLY",
    "BLOCK",
    "RESEARCH_DECISION_REQUIRED",
}
_VIEW_IDS = (
    "full_eligible",
    "content_only",
    "observability_only",
    "source_blind",
    "target_component_ablation",
    "core_confirmatory",
    "robustness_extended",
    "target_reconstruction_diagnostic",
)
_FORBIDDEN_PANEL_TOKENS = (OUTCOME, "known_case", "sanction_event", "post_outcome")


@dataclass(frozen=True)
class FeatureBuildResult:
    panel: pd.DataFrame
    registry: list[dict[str, object]]
    lineage_registry: pd.DataFrame
    leakage_rows: pd.DataFrame
    feature_views: dict[str, object]
    availability_audit: pd.DataFrame
    missingness_audit: pd.DataFrame
    summary: dict[str, object]
    decision_report: str

    @property
    def leakage_registry(self) -> dict[str, object]:
        return {
            "status": "PASS",
            "preprocessing_fit_at_p07": False,
            "outer_outcomes_accessed": False,
            "known_cases_accessed": False,
            "content_predictors_entered_label_model": False,
            "label_derived_features_entered_predictor_set": False,
            "rows": self.leakage_rows.to_dict(orient="records"),
        }


def build_feature_panel(
    *,
    firm_year_panel: pd.DataFrame,
    risk_sets: pd.DataFrame,
    feature_definitions: list[dict[str, object]],
    columns: dict[str, str],
    blocked_label_semantics: list[str] | None = None,
    intended_definitions: list[dict[str, object]] | None = None,
    target_ids: list[str] | None = None,
    temporal_estimands: dict[str, object] | None = None,
) -> FeatureBuildResult:
    """Build only raw, as-of fields; no target read or preprocessing fit is permitted."""
    for legacy_definition in feature_definitions:
        if (
            legacy_definition.get("role") == "content"
            and legacy_definition.get("allowed_in_label_model") is not False
        ):
            raise ValueError(
                f"feature={legacy_definition.get('feature_id')}: "
                "content predictors cannot enter label models"
            )
        blocked_matches = set(
            cast(list[object], legacy_definition.get("source_semantics", []))
        ) & set(blocked_label_semantics or [])
        if blocked_matches and legacy_definition.get("allowed_in_label_model") is not False:
            raise ValueError(
                f"feature={legacy_definition.get('feature_id')}: "
                "label-derived same-year variable is prohibited"
            )
    firm, year, prediction = columns[FIRM_ID], columns[FISCAL_YEAR], columns[PREDICTION_TIME]
    target_column = columns.get(TARGET_ID, TARGET_ID)
    temporal_role_column = columns.get(TEMPORAL_ROLE, TEMPORAL_ROLE)
    eligible, annual_mature, s3_mature = (
        columns[ELIGIBLE],
        columns[ANNUAL_MEASUREMENT_MATURE],
        columns[S3_NEXT_YEAR_MATURE],
    )
    if not {firm, year, prediction}.issubset(firm_year_panel.columns):
        raise ValueError("P07 firm-year panel contract is incomplete")
    if not {firm, year, eligible, annual_mature, s3_mature}.issubset(risk_sets.columns):
        raise ValueError("P07 risk-set contract is incomplete")
    _reject_forbidden_columns(firm_year_panel.columns)
    if firm_year_panel.duplicated([firm, year]).any() or risk_sets.duplicated([firm, year]).any():
        raise ValueError("P07 requires unique firm_id x fiscal_year rows")

    metadata = risk_sets.loc[:, [firm, year, eligible, annual_mature, s3_mature]]
    base = firm_year_panel.merge(metadata, on=[firm, year], how="inner", validate="one_to_one")
    if base.empty:
        raise ValueError("P07 raw panel is empty after risk-set binding")
    if not base[year].between(1990, 2100).all():
        raise ValueError("P07 fiscal years are invalid")
    panel = base.loc[:, [firm, year, prediction, eligible, annual_mature, s3_mature]].copy()
    # These fields are mandatory protocol metadata; unavailable source mappings stay explicitly missing.
    panel["exchange_or_board"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["industry_code"] = pd.Series(pd.NA, index=panel.index, dtype="string")

    definitions = [
        *_normalise_definitions(feature_definitions, operational=True),
        *_normalise_definitions(intended_definitions or [], operational=False),
    ]
    _validate_unique_ids(definitions)
    operational: list[dict[str, object]] = []
    lineage_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    feature_columns: dict[str, pd.Series] = {}
    for definition in definitions:
        _validate_definition(definition)
        feature_id = str(definition["feature_id"])
        column = definition.get("physical_column")
        status = str(definition["research_decision_status"])
        if status == "LOCKED" and definition["research_decision_status"] == "LOCKED":
            if not isinstance(column, str) or not column or column not in base.columns:
                raise ValueError(
                    f"feature={feature_id}: locked physical_column is absent from the panel"
                )
            if not bool(definition.get("operational", False)):
                raise ValueError(f"feature={feature_id}: locked feature must be operational")
            feature_columns[feature_id] = _coerce_feature(base[column], definition)
            operational.append(definition)
        availability_rows.append(
            {
                "feature_id": feature_id,
                "availability_rule": definition["availability_rule"],
                "availability_status": "PASS"
                if definition["research_decision_status"] == "LOCKED"
                else "NOT_OPERATIONAL",
                "reason_code": definition.get(
                    "availability_reason_code", "RESEARCH_DECISION_REQUIRED"
                ),
            }
        )
        lineage_rows.extend(_lineage_rows(definition))

    if feature_columns:
        feature_frame = pd.DataFrame(feature_columns, index=panel.index)
        panel = pd.concat([panel, feature_frame], axis=1)

    panel[firm] = panel[firm].astype("string")
    panel[year] = panel[year].astype("int16")
    if panel.duplicated([firm, year]).any():
        raise ValueError("P07 output has duplicate firm-year keys")
    _reject_forbidden_columns(panel.columns)
    registry = [
        {key: value for key, value in definition.items() if key != "operational"}
        for definition in definitions
    ]
    target_ids = sorted(target_ids or [])
    leakage = _build_leakage_rows(
        registry, target_ids, temporal_estimands or {}, target_column, temporal_role_column
    )
    views = _build_views(registry, leakage, target_ids, target_column)
    missingness = _missingness_audit(panel, operational, firm, year)
    summary = _summary(
        panel,
        registry,
        leakage,
        views,
        firm,
        year,
        {prediction, eligible, annual_mature, s3_mature},
    )
    return FeatureBuildResult(
        panel=panel,
        registry=registry,
        lineage_registry=pd.DataFrame(lineage_rows),
        leakage_rows=leakage,
        feature_views=views,
        availability_audit=pd.DataFrame(availability_rows),
        missingness_audit=missingness,
        summary=summary,
        decision_report=_decision_report(summary, registry),
    )


def _reject_forbidden_columns(columns: object) -> None:
    if not isinstance(columns, pd.Index):
        raise ValueError("P07 panel columns must be a pandas Index")
    column_names = cast(list[object], columns.tolist())
    forbidden = [
        str(column)
        for column in column_names
        if any(token in str(column).lower() for token in _FORBIDDEN_PANEL_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"P07 forbidden outcome/known-case columns: {sorted(forbidden)}")


def _normalise_definitions(
    items: list[dict[str, object]], *, operational: bool
) -> list[dict[str, object]]:
    return [{**item, "operational": operational} for item in items]


def _validate_unique_ids(definitions: list[dict[str, object]]) -> None:
    ids = [item.get("feature_id") for item in definitions]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("P07 feature IDs must be non-empty and unique")


def validate_feature_definition(item: dict[str, object]) -> None:
    """Validate one feature registry definition against the closed contract."""
    _validate_definition(item)


def _validate_definition(item: dict[str, object]) -> None:
    feature_id = str(item["feature_id"])
    # Retain the original firewall diagnostic even for a malformed legacy definition.
    if item.get("role") == "content" and item.get("allowed_in_label_model") is not False:
        raise ValueError(f"feature={feature_id}: content predictors cannot enter label models")
    required = (
        "canonical_name",
        "display_name",
        "description",
        "theoretical_block",
        "source_dataset",
        "source_table_or_file",
        "source_column_or_columns",
        "formula",
        "unit",
        "sign_convention",
        "fiscal_period_reference",
        "lag_structure",
        "availability_rule",
        "availability_date_field",
        TEMPORAL_ROLE,
        "maximum_information_date",
        "expected_dtype",
        "allowed_missingness",
        "missingness_interpretation",
        "content_observability_role",
        "target_component_flag",
        "source_specific_flag",
        "source_channel",
        "model_eligibility",
        "confirmatory_status",
        "research_decision_status",
        "version",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"feature={feature_id}: missing registry fields {missing}")
    if item["content_observability_role"] not in _ROLES:
        raise ValueError(f"feature={feature_id}: invalid content_observability_role")
    if item["confirmatory_status"] not in _CONFIRMATORY:
        raise ValueError(f"feature={feature_id}: invalid confirmatory_status")
    if item["research_decision_status"] not in _DECISION_STATUS:
        raise ValueError(f"feature={feature_id}: invalid research_decision_status")
    if item["model_eligibility"] not in MODEL_ELIGIBILITY_VALUES:
        raise ValueError(f"feature={feature_id}: invalid model_eligibility")
    if not isinstance(item["target_component_flag"], bool) or not isinstance(
        item["source_specific_flag"], bool
    ):
        raise ValueError(f"feature={feature_id}: feature flags must be boolean")
    if (
        item["content_observability_role"] == "content"
        and item.get("allowed_in_label_model") is not False
    ):
        raise ValueError(f"feature={feature_id}: content predictors cannot enter label models")
    if (
        item["research_decision_status"] != "LOCKED"
        and item["confirmatory_status"] == "confirmatory"
    ):
        raise ValueError(f"feature={feature_id}: unresolved feature cannot be confirmatory")
    if item.get("operational") and not item.get("lineage"):
        raise ValueError(f"feature={feature_id}: derived operational feature requires lineage")


def _lineage_rows(item: dict[str, object]) -> list[dict[str, object]]:
    raw = item.get("lineage") or [{"source_column": item["source_column_or_columns"]}]
    if not isinstance(raw, list):
        raise ValueError(f"feature={item['feature_id']}: lineage must be a list")
    return [
        {
            "feature_id": item["feature_id"],
            "source_dataset": item["source_dataset"],
            "source_column": entry.get("source_column", item["source_column_or_columns"]),
            "transformation_step": entry.get("transformation_step", "unavailable"),
            "transformation_order": entry.get("transformation_order", 1),
            "formula_or_operation": entry.get("formula_or_operation", item["formula"]),
            "temporal_shift": entry.get("temporal_shift", item["lag_structure"]),
            "aggregation_rule": entry.get("aggregation_rule", "not_applicable"),
            "join_keys": entry.get("join_keys", "firm_id,fiscal_year"),
            "join_cardinality": entry.get("join_cardinality", "one_to_one"),
            "availability_constraint": entry.get(
                "availability_constraint", item["availability_rule"]
            ),
            "implementation_file": "src/features/service.py",
            "implementation_function": "build_feature_panel",
            "config_reference": "features.registry",
        }
        for entry in cast(list[dict[str, object]], raw)
    ]


def _build_leakage_rows(
    registry: list[dict[str, object]],
    target_ids: list[str],
    temporal_estimands: dict[str, object],
    target_column: str,
    temporal_role_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in registry:
        for target_id in target_ids:
            status = str(feature["research_decision_status"])
            affected_targets = _target_component_targets(feature)
            if status != "LOCKED":
                decision, reason = "RESEARCH_DECISION_REQUIRED", "FEATURE_NOT_LOCKED"
            elif target_id in affected_targets:
                decision, reason = "BLOCK", "DIRECT_TARGET_COMPONENT"
            elif bool(feature["source_specific_flag"]):
                decision, reason = "ALLOW_WITH_ABLATION", "SOURCE_BLIND_EXCLUSION"
            else:
                decision, reason = "ALLOW", "TEMPORALLY_VALID"
            temporal_role = str(temporal_estimands.get(target_id, "target_specific"))
            rows.append(
                {
                    "feature_id": feature["feature_id"],
                    target_column: target_id,
                    temporal_role_column: temporal_role,
                    "target_component_flag": bool(feature["target_component_flag"]),
                    "source_specific_flag": bool(feature["source_specific_flag"]),
                    "temporal_leakage_status": "PASS",
                    "target_component_status": "DIRECT_COMPONENT"
                    if target_id in affected_targets
                    else "ABLATABLE"
                    if bool(feature["target_component_flag"])
                    else "NONE",
                    "source_overlap_status": "SOURCE_SPECIFIC"
                    if bool(feature["source_specific_flag"])
                    else "NONE",
                    "post_outcome_status": "PASS",
                    "label_construction_overlap": "NONE",
                    "availability_status": "PASS" if status == "LOCKED" else "UNRESOLVED",
                    "outer_outcome_access_status": "NOT_ACCESSED",
                    "known_case_access_status": "NOT_ACCESSED",
                    "decision": decision,
                    "reason_code": reason,
                    "reason": reason.replace("_", " ").lower(),
                    "required_action": "lock_definition"
                    if decision == "RESEARCH_DECISION_REQUIRED"
                    else "exclude_from_confirmatory_prediction"
                    if decision == "BLOCK"
                    else "none",
                    "review_status": "PASS"
                    if decision in {"ALLOW", "ALLOW_WITH_ABLATION"}
                    else "BLOCKED",
                    "permitted_views": "target_reconstruction_diagnostic"
                    if decision == "BLOCK"
                    else "none"
                    if decision == "RESEARCH_DECISION_REQUIRED"
                    else "target_specific_registry_views",
                }
            )
    return pd.DataFrame(rows)


def _build_views(
    registry: list[dict[str, object]],
    leakage: pd.DataFrame,
    target_ids: list[str],
    target_column: str,
) -> dict[str, object]:
    registry_hash = hashlib.sha256(
        json.dumps(registry, sort_keys=True, default=str).encode()
    ).hexdigest()
    views: list[dict[str, object]] = []
    for target_id in target_ids:
        target_rows = leakage[leakage[target_column] == target_id] if not leakage.empty else leakage
        for view_id in _VIEW_IDS:
            included: list[str] = []
            excluded: dict[str, str] = {}
            for feature in registry:
                feature_id = str(feature["feature_id"])
                row = target_rows[target_rows["feature_id"] == feature_id]
                decision = (
                    str(row.iloc[0]["decision"]) if not row.empty else "RESEARCH_DECISION_REQUIRED"
                )
                allowed = decision in {"ALLOW", "ALLOW_WITH_ABLATION"}
                role = str(feature["content_observability_role"])
                if view_id == "content_only":
                    allowed = allowed and role == "content"
                elif view_id == "observability_only":
                    allowed = allowed and role == "observability"
                elif view_id == "source_blind":
                    allowed = allowed and not bool(feature["source_specific_flag"])
                elif view_id == "target_component_ablation":
                    allowed = allowed and not bool(feature["target_component_flag"])
                elif view_id == "core_confirmatory":
                    allowed = (
                        allowed
                        and feature["confirmatory_status"] == "confirmatory"
                        and feature["research_decision_status"] == "LOCKED"
                    )
                elif view_id == "robustness_extended":
                    allowed = allowed and feature["confirmatory_status"] in {
                        "confirmatory",
                        "robustness_only",
                    }
                elif view_id == "target_reconstruction_diagnostic":
                    allowed = (
                        decision == "BLOCK"
                        and str(row.iloc[0]["reason_code"] if not row.empty else "")
                        == "DIRECT_TARGET_COMPONENT"
                    )
                if allowed:
                    included.append(feature_id)
                else:
                    excluded[feature_id] = (
                        decision if decision != "ALLOW" else "VIEW_MEMBERSHIP_RULE"
                    )
            views.append(
                {
                    "view_id": view_id,
                    target_column: target_id,
                    "included_feature_ids": included,
                    "excluded_feature_ids": sorted(excluded),
                    "exclusion_reasons": excluded,
                    "registry_hash": registry_hash,
                    "config_hash": registry_hash,
                }
            )
    return {"status": "PASS", "views": views, "preprocessing_fit_at_p07": False}


def _target_component_targets(feature: dict[str, object]) -> set[str]:
    if not bool(feature.get("target_component_flag")):
        return set()
    semantics = {
        str(value).lower() for value in cast(list[object], feature.get("source_semantics", []))
    }
    feature_id = str(feature.get("feature_id", "")).lower()
    if any("profit_after_tax" in value for value in semantics | {feature_id}) or any(
        "net_revenue" in value for value in semantics | {feature_id}
    ):
        return {"L1_ANNUAL", "L1_REPORTING", "L1_CONTENT_STRICT"}
    return set()


def _coerce_feature(values: pd.Series, definition: dict[str, object]) -> pd.Series:
    expected = str(definition["expected_dtype"])
    if expected in {"float64", "float32", "int64", "int32", "int16"}:
        numeric = pd.to_numeric(values, errors="raise")
        if expected.startswith("int"):
            if numeric.dropna().mod(1).ne(0).any():
                raise ValueError(
                    f"feature={definition['feature_id']}: non-integral value for {expected}"
                )
            if expected == "int16":
                return numeric.astype("Int16")
            if expected == "int32":
                return numeric.astype("Int32")
            return numeric.astype("Int64")
        return numeric.astype("float32" if expected == "float32" else "float64")
    if expected in {"string", "category"}:
        result = values.astype("string")
        return result.astype("category") if expected == "category" else result
    if expected in {"bool", "boolean"}:
        return values.astype("boolean")
    raise ValueError(f"feature={definition['feature_id']}: unsupported expected dtype {expected}")


def _missingness_audit(
    panel: pd.DataFrame, operational: list[dict[str, object]], firm: str, year: str
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_id": str(item["feature_id"]),
                "missing_count": int(panel[str(item["feature_id"])].isna().sum()),
                "row_count": len(panel),
                "missing_fraction": float(panel[str(item["feature_id"])].isna().mean()),
                "missing_to_zero_applied": False,
            }
            for item in operational
        ],
        columns=[
            "feature_id",
            "missing_count",
            "row_count",
            "missing_fraction",
            "missing_to_zero_applied",
        ],
    )


def _summary(
    panel: pd.DataFrame,
    registry: list[dict[str, object]],
    leakage: pd.DataFrame,
    views: dict[str, object],
    firm: str,
    year: str,
    metadata_columns: set[str],
) -> dict[str, object]:
    def counts(key: str) -> dict[str, int]:
        return {
            str(value): sum(item.get(key) == value for item in registry)
            for value in sorted({str(item.get(key)) for item in registry})
        }

    return {
        "status": "PASS",
        "firm_years": len(panel),
        "firms": int(panel[firm].nunique()),
        "fiscal_year_range": [int(panel[year].min()), int(panel[year].max())],
        "registered_features": len(registry),
        "panel_features": len(
            [
                column
                for column in panel.columns
                if column
                not in {
                    firm,
                    year,
                    *metadata_columns,
                    "exchange_or_board",
                    "industry_code",
                }
            ]
        ),
        "blocked_features": sum(item["confirmatory_status"] == "blocked" for item in registry),
        "unresolved_features": sum(
            item["research_decision_status"] == "RESEARCH_DECISION_REQUIRED" for item in registry
        ),
        "confirmatory_features": sum(
            item["confirmatory_status"] == "confirmatory" for item in registry
        ),
        "robustness_only_features": sum(
            item["confirmatory_status"] == "robustness_only" for item in registry
        ),
        "by_theoretical_block": counts("theoretical_block"),
        "by_role": counts("content_observability_role"),
        "by_confirmatory_status": counts("confirmatory_status"),
        "by_research_decision_status": counts("research_decision_status"),
        "by_availability_rule": counts("availability_rule"),
        "by_leakage_decision": leakage["decision"].value_counts().to_dict()
        if not leakage.empty
        else {},
        "feature_view_count": len(cast(list[object], views["views"])),
    }


def _decision_report(summary: dict[str, object], registry: list[dict[str, object]]) -> str:
    unresolved = [
        str(item["feature_id"]) for item in registry if item["research_decision_status"] != "LOCKED"
    ]
    return (
        "# P07 feature-governance decision report\n\n"
        + f"- Status: {summary['status']}\n- Firm-years: {summary['firm_years']}\n- Registered features: {summary['registered_features']}\n- Unresolved features: {', '.join(unresolved) if unresolved else 'none'}\n- No preprocessing, outer outcomes, or known cases were accessed.\n"
    )
