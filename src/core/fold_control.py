"""Fail-closed production controls for fold-specific confirmatory stages."""

from __future__ import annotations

from typing import Any, cast

from core.semantic_keys import OUTER_FOLD


def confirmatory_fold_blockers(
    fold_eligibility: object,
    outer_fold: str,
) -> list[str]:
    if not isinstance(fold_eligibility, list):
        return ["FOLD_ELIGIBILITY_ARTIFACT_INVALID"]
    matches = [
        cast(dict[str, Any], row)
        for row in cast(list[object], fold_eligibility)
        if isinstance(row, dict) and str(cast(dict[str, Any], row).get(OUTER_FOLD)) == outer_fold
    ]
    if len(matches) != 1:
        return ["FOLD_ELIGIBILITY_BINDING_MISSING"]
    row = matches[0]
    blockers: list[str] = []
    if row.get("assigned_role") != "confirmatory":
        blockers.append("FOLD_ROLE_NOT_CONFIRMATORY")
    positive = row.get("positive_count")
    negative = row.get("explicit_negative_count")
    classes = row.get("observed_binary_class_count")
    if not isinstance(positive, int) or positive <= 0:
        blockers.append("POSITIVE_CLASS_MISSING")
    if not isinstance(negative, int) or negative <= 0:
        blockers.append("EXPLICIT_NEGATIVE_CLASS_MISSING")
    if classes != 2 or row.get("both_binary_classes_observed") is not True:
        blockers.append("TWO_OBSERVED_CLASSES_REQUIRED")
    return sorted(set(blockers))


def require_confirmatory_fold(fold_eligibility: object, outer_fold: str, stage_id: str) -> None:
    blockers = confirmatory_fold_blockers(fold_eligibility, outer_fold)
    if blockers:
        raise RuntimeError(
            f"{stage_id}_FOLD_NOT_CONFIRMATORY: fold={outer_fold} blockers={','.join(blockers)}"
        )


def production_path_blockers(
    *,
    fold_eligibility: object,
    outer_fold: str,
    feature_registry: object,
    mcse_report: object,
) -> list[str]:
    blockers = confirmatory_fold_blockers(fold_eligibility, outer_fold)
    if not isinstance(feature_registry, list) or not feature_registry:
        blockers.append("FEATURE_REGISTRY_EMPTY")
    if not isinstance(mcse_report, dict):
        blockers.append("P08_MCSE_REPORT_INVALID")
    else:
        report = cast(dict[str, Any], mcse_report)
        if report.get("status") != "PASS" or report.get("precision_target_met") is not True:
            blockers.append("P08_MCSE_NOT_PASS")
    return sorted(set(blockers))
