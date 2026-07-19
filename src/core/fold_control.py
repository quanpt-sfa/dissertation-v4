"""Fail-closed production controls for fold-specific confirmatory stages."""

from __future__ import annotations

from typing import Any, cast

from core.semantic_keys import OUTER_FOLD, TARGET_ID


def require_primary_target(measurement: object, stage_id: str) -> str:
    if not isinstance(measurement, dict):
        raise RuntimeError(f"{stage_id}_PRODUCTION_PATH_BLOCKED: PRIMARY_TARGET_NOT_LOCKED")
    configuration = cast(dict[str, Any], measurement)
    primary = configuration.get("primary_target_id")
    if not isinstance(primary, str) or not primary.strip():
        execution = configuration.get("execution_tracks")
        if isinstance(execution, dict):
            primary = cast(dict[str, Any], execution).get("primary_target_id")
    if not isinstance(primary, str) or not primary.strip():
        raise RuntimeError(f"{stage_id}_PRODUCTION_PATH_BLOCKED: PRIMARY_TARGET_NOT_LOCKED")
    candidates = configuration.get("candidate_targets")
    if not isinstance(candidates, dict) or primary.strip() not in candidates:
        raise RuntimeError(f"{stage_id}_PRODUCTION_PATH_BLOCKED: PRIMARY_TARGET_NOT_REGISTERED")
    return primary.strip()


def confirmatory_fold_blockers(
    fold_eligibility: object,
    outer_fold: str,
    target_id: str | None = None,
) -> list[str]:
    if not isinstance(fold_eligibility, list):
        return ["FOLD_ELIGIBILITY_ARTIFACT_INVALID"]
    matches = [
        cast(dict[str, Any], row)
        for row in cast(list[object], fold_eligibility)
        if isinstance(row, dict)
        and str(cast(dict[str, Any], row).get(OUTER_FOLD)) == outer_fold
        and (target_id is None or str(cast(dict[str, Any], row).get(TARGET_ID)) == target_id)
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


def require_confirmatory_fold(
    fold_eligibility: object,
    outer_fold: str,
    stage_id: str,
    target_id: str | None = None,
) -> None:
    blockers = confirmatory_fold_blockers(fold_eligibility, outer_fold, target_id)
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
    target_id: str | None = None,
) -> list[str]:
    blockers = confirmatory_fold_blockers(fold_eligibility, outer_fold, target_id)
    if not isinstance(feature_registry, list) or not feature_registry:
        blockers.append("FEATURE_REGISTRY_EMPTY")
    if not isinstance(mcse_report, dict):
        blockers.append("P08_MCSE_REPORT_INVALID")
    else:
        report = cast(dict[str, Any], mcse_report)
        if report.get("status") != "PASS" or report.get("precision_target_met") is not True:
            blockers.append("P08_MCSE_NOT_PASS")
    return sorted(set(blockers))
