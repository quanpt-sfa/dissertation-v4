"""Pilot-safe feature roster preparation for P11 model fitting.

The production feature registry remains immutable.  This module derives an
*effective* model roster for one outer fold and excludes, rather than imputes or
silently accepts, any feature that is marked fit-eligible but is not physically
usable in the panel.  Every exclusion is returned for the P11 freeze receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from features.service import FIT_MODEL_ELIGIBILITY_VALUES


@dataclass(frozen=True)
class ModelFeatureContract:
    """Effective registry plus auditable fold-local feature diagnostics."""

    registry: list[dict[str, Any]]
    rows: list[dict[str, object]]
    exclusions: list[dict[str, object]]
    summary: dict[str, object]


def prepare_model_feature_contract(
    *,
    panel: pd.DataFrame,
    registry: list[dict[str, Any]],
    year_column: str,
    outer_year: int,
) -> ModelFeatureContract:
    """Derive a fold-local model roster without adding information or features.

    A registry item can enter a model only when all of the following hold:

    * it is locked (or uses the legacy null-as-locked state);
    * its ``model_eligibility`` is one of the fit-eligible values;
    * its physical feature column exists;
    * its computation status, when supplied, is ``PASS``;
    * it has at least one non-missing development-history value; and
    * it is not a direct target component.

    Items failing these conditions are copied into the effective registry with
    ``model_eligibility=blocked_until_mapping_review``.  The upstream registry
    artifact itself is never modified.
    """

    if year_column not in panel.columns:
        raise ValueError(f"P11 model feature contract: missing year column {year_column!r}")
    if panel.empty:
        raise ValueError("P11 model feature contract: feature panel is empty")
    if not registry:
        raise ValueError("P11 model feature contract: feature registry is empty")

    development_mask = panel[year_column] < outer_year
    outer_mask = panel[year_column] == outer_year
    effective: list[dict[str, Any]] = []
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    seen: set[str] = set()

    for original in registry:
        item = dict(original)
        feature_id = str(item.get("feature_id", "")).strip()
        if not feature_id:
            raise ValueError("P11 model feature contract: feature_id must be non-empty")
        if feature_id in seen:
            raise ValueError(f"P11 model feature contract: duplicate feature_id={feature_id}")
        seen.add(feature_id)

        decision_status = item.get("research_decision_status")
        model_eligibility = item.get("model_eligibility")
        requested_for_fit = decision_status in {None, "LOCKED"} and model_eligibility in {
            None,
            *FIT_MODEL_ELIGIBILITY_VALUES,
        }
        column_exists = feature_id in panel.columns
        computation_status = item.get("computation_status")
        computation_passed = computation_status in {None, "PASS"}
        target_component = bool(item.get("target_component_flag", False))
        development_non_null = (
            int(panel.loc[development_mask, feature_id].notna().sum()) if column_exists else 0
        )
        outer_non_null = int(panel.loc[outer_mask, feature_id].notna().sum()) if column_exists else 0

        reason: str | None = None
        if requested_for_fit and not column_exists:
            reason = "MODEL_FEATURE_COLUMN_MISSING"
        elif requested_for_fit and not computation_passed:
            reason = "MODEL_FEATURE_COMPUTATION_NOT_PASS"
        elif requested_for_fit and development_non_null == 0:
            reason = "MODEL_FEATURE_ZERO_DEVELOPMENT_COVERAGE"
        elif requested_for_fit and target_component:
            reason = "DIRECT_TARGET_COMPONENT_FORBIDDEN"

        included = requested_for_fit and reason is None
        if requested_for_fit and not included:
            item["model_eligibility"] = "blocked_until_mapping_review"
            item["runtime_exclusion_reason"] = reason
            exclusions.append(
                {
                    "feature_id": feature_id,
                    "reason_code": reason,
                    "column_exists": column_exists,
                    "computation_status": computation_status,
                    "development_non_null_count": development_non_null,
                    "outer_non_null_count": outer_non_null,
                }
            )

        rows.append(
            {
                "feature_id": feature_id,
                "registry_model_eligibility": model_eligibility,
                "effective_model_eligibility": item.get("model_eligibility"),
                "research_decision_status": decision_status,
                "computation_status": computation_status,
                "column_exists": column_exists,
                "development_non_null_count": development_non_null,
                "outer_non_null_count": outer_non_null,
                "target_component_flag": target_component,
                "included_in_effective_roster": included,
                "exclusion_reason": reason,
            }
        )
        effective.append(item)

    included_count = sum(bool(row["included_in_effective_roster"]) for row in rows)
    if included_count == 0:
        raise RuntimeError("P11_MODEL_FEATURE_ROSTER_EMPTY_AFTER_FAIL_CLOSED_FILTER")

    summary: dict[str, object] = {
        "outer_year": outer_year,
        "registry_feature_count": len(registry),
        "effective_feature_count": included_count,
        "runtime_exclusion_count": len(exclusions),
        "development_row_count": int(development_mask.sum()),
        "outer_row_count": int(outer_mask.sum()),
        "policy": "exclude_unusable_features_without_imputation_or_registry_mutation",
    }
    return ModelFeatureContract(
        registry=effective,
        rows=rows,
        exclusions=exclusions,
        summary=summary,
    )
