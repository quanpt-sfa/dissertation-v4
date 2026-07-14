"""Pure feature-registry validation; no preprocessing is fit at P07."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from core.semantic_keys import ELIGIBLE, FIRM_ID, FISCAL_YEAR, PREDICTION_TIME


@dataclass(frozen=True)
class FeatureBuildResult:
    panel: pd.DataFrame
    registry: list[dict[str, object]]
    leakage_registry: dict[str, object]


def build_feature_panel(
    *,
    firm_year_panel: pd.DataFrame,
    risk_sets: pd.DataFrame,
    feature_definitions: list[dict[str, object]],
    columns: dict[str, str],
    blocked_label_semantics: list[str] | None = None,
) -> FeatureBuildResult:
    """Validate feature metadata and return an as-of key panel without fitting transforms."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    required = {firm, year, columns[PREDICTION_TIME]}
    if not required.issubset(firm_year_panel.columns):
        raise ValueError("P07 firm-year panel contract is incomplete")
    if not {firm, year, columns[ELIGIBLE]}.issubset(risk_sets.columns):
        raise ValueError("P07 risk-set contract is incomplete")
    eligible_keys = risk_sets.loc[risk_sets[columns[ELIGIBLE]], [firm, year]]
    base = firm_year_panel.merge(eligible_keys, on=[firm, year], how="inner", validate="one_to_one")
    output = base.loc[:, [firm, year]].copy()
    validated: list[dict[str, object]] = []
    blocked = set(blocked_label_semantics or [])
    for definition in feature_definitions:
        feature_id = definition.get("feature_id")
        physical_column = definition.get("physical_column")
        role = definition.get("role")
        allowed = definition.get("allowed_in_label_model")
        availability_rule = definition.get("availability_rule")
        theoretical_block = definition.get("theoretical_block")
        target_component = definition.get("target_component", False)
        raw_semantics = definition.get("source_semantics", [])
        if not isinstance(feature_id, str) or not feature_id:
            raise ValueError("feature definition requires feature_id")
        if not isinstance(physical_column, str) or not physical_column:
            raise ValueError(f"feature={feature_id}: physical_column binding required")
        if physical_column not in base.columns:
            raise ValueError(f"feature={feature_id}: bound column is absent from the panel")
        if feature_id in output.columns:
            raise ValueError(f"feature={feature_id}: feature ID collides with a key column")
        if role not in {"content", "observability", "ambiguous"}:
            raise ValueError(f"feature={feature_id}: invalid role")
        if role == "content" and allowed is not False:
            raise ValueError(f"feature={feature_id}: content predictors cannot enter label models")
        if not isinstance(availability_rule, str) or not availability_rule:
            raise ValueError(f"feature={feature_id}: availability_rule required")
        if not isinstance(theoretical_block, str) or not theoretical_block:
            raise ValueError(f"feature={feature_id}: theoretical_block required")
        if not isinstance(target_component, bool):
            raise ValueError(f"feature={feature_id}: target_component must be boolean")
        if not isinstance(raw_semantics, list) or not all(
            isinstance(value, str) and value for value in cast(list[object], raw_semantics)
        ):
            raise ValueError(f"feature={feature_id}: source_semantics must be strings")
        source_semantics = {str(value) for value in cast(list[object], raw_semantics)}
        blocked_matches = sorted(source_semantics & blocked)
        if target_component or blocked_matches:
            raise ValueError(
                f"feature={feature_id}: label-derived same-year variable is prohibited; "
                f"blocked_semantics={blocked_matches}"
            )
        validated.append(dict(definition))
        output[feature_id] = pd.to_numeric(base[physical_column], errors="raise").astype("float64")
    output[firm] = output[firm].astype("string")
    output[year] = output[year].astype("int16")
    audits = [
        {
            "feature_id": str(item["feature_id"]),
            "availability_rule": item["availability_rule"],
            "role": item["role"],
            "allowed_in_label_model": item["allowed_in_label_model"],
            "content_label_firewall_pass": item["role"] != "content"
            or item["allowed_in_label_model"] is False,
            "label_derived_feature_firewall_pass": item.get("target_component", False) is False
            and not bool(set(cast(list[str], item.get("source_semantics", []))) & blocked),
        }
        for item in validated
    ]
    return FeatureBuildResult(
        output,
        validated,
        {
            "status": "PASS"
            if all(
                item["content_label_firewall_pass"] and item["label_derived_feature_firewall_pass"]
                for item in audits
            )
            else "FAIL",
            "features": audits,
            "preprocessing_fit_at_p07": False,
            "outer_outcomes_accessed": False,
            "content_predictors_entered_label_model": False,
            "label_derived_features_entered_predictor_set": False,
            "blocked_same_year_semantics": sorted(blocked),
        },
    )
