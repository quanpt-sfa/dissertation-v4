"""Fail-closed completion contract for the P08 simulation gate."""

from __future__ import annotations

from typing import Any, cast


def p08_completion_blockers(report: object) -> list[str]:
    blockers: list[str] = []
    if not isinstance(report, dict):
        return ["P08_MCSE_REPORT_INVALID"]
    payload = cast(dict[str, Any], report)
    if payload.get("status") != "PASS":
        blockers.append("P08_STATUS_NOT_PASS")
    if payload.get("precision_target_met") is not True:
        blockers.append("P08_PRECISION_TARGET_NOT_MET")
    return blockers


def require_p08_completion(report: object, stage_id: str) -> None:
    blockers = p08_completion_blockers(report)
    if blockers:
        raise RuntimeError(
            f"{stage_id}_PRODUCTION_PATH_BLOCKED: blockers={','.join(blockers)}"
        )
