"""Render final outputs exclusively from immutable upstream artifacts."""

from __future__ import annotations

from html import escape
from typing import Any, cast

from core.semantic_keys import LEARNER_ID, OUTER_FOLD


def build_final_ledger(
    *,
    gate2: dict[str, Any] | None,
    gate3: dict[str, Any],
    evaluations: list[dict[str, Any]],
    domain: dict[str, Any] | None,
    hierarchical_pi: dict[str, Any] | None,
    known_cases: list[dict[str, Any]] | None,
    threshold: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "result_id": "gate2",
            "result_type": "gate",
            "value": None if gate2 is None else gate2.get("verdict"),
        },
        {"result_id": "gate3", "result_type": "gate", "value": gate3.get("verdict")},
    ]
    for evaluation in evaluations:
        models = evaluation.get("models")
        if not isinstance(models, list):
            continue
        for raw_model in cast(list[Any], models):
            if not isinstance(raw_model, dict):
                continue
            model = cast(dict[str, Any], raw_model)
            rows.append(
                {
                    "result_id": f"outer:{evaluation.get(OUTER_FOLD)}:{model.get(LEARNER_ID)}",
                    "result_type": "outer_metric",
                    "average_precision": model.get("average_precision"),
                    "brier_score": model.get("brier_score"),
                    "positives": model.get("positives"),
                }
            )
    rows.extend(
        [
            {
                "result_id": "domain_transfer",
                "result_type": "sensitivity",
                "value": None if domain is None else domain.get("status"),
            },
            {
                "result_id": "hierarchical_pi",
                "result_type": "sensitivity",
                "value": None if hierarchical_pi is None else hierarchical_pi.get("status"),
            },
            {
                "result_id": "threshold_interaction",
                "result_type": "gate_component",
                "value": None if threshold is None else threshold.get("status"),
            },
        ]
    )
    if known_cases is not None:
        summary = next((item for item in known_cases if item.get("record_type") == "summary"), None)
        rows.append(
            {
                "result_id": "known_cases",
                "result_type": "post_gate_validation",
                "value": None if summary is None else summary.get("status"),
                "case_count": None if summary is None else summary.get("case_count"),
                "soft_veto": None if summary is None else summary.get("soft_veto"),
            }
        )
    return rows


def render_final_report(
    ledger: list[dict[str, Any]], gate2: dict[str, Any] | None, gate3: dict[str, Any]
) -> str:
    gate2_value = "UNAVAILABLE" if gate2 is None else str(gate2.get("verdict"))
    gate3_value = str(gate3.get("verdict"))
    fold_rows = [item for item in ledger if item.get("result_type") == "outer_metric"]
    sensitivity = [item for item in ledger if item.get("result_type") == "sensitivity"]
    lines = [
        "# Chapter 3 final pipeline report",
        "",
        "## Locked decisions",
        "",
        f"- Gate 2: `{gate2_value}`",
        f"- Gate 3: `{gate3_value}`",
        f"- Outer model-metric rows: {len(fold_rows)}",
        "",
        "## Sensitivity status",
        "",
    ]
    lines.extend(f"- {item['result_id']}: `{item.get('value')}`" for item in sensitivity)
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This document is report-only. It does not refit models, reopen seals, tune thresholds, or convert missing/immature evidence into negative outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def build_verdict_matrix(
    gate2: dict[str, Any] | None,
    gate3: dict[str, Any],
    known_cases: list[dict[str, Any]] | None,
) -> list[dict[str, object]]:
    """Create the report-only gate matrix with explicit non-upgrade semantics."""
    known_summary = next(
        (item for item in known_cases or [] if item.get("record_type") == "summary"),
        None,
    )
    return [
        {
            "gate_id": "GATE2",
            "verdict": None if gate2 is None else gate2.get("verdict"),
            "criteria_source": None if gate2 is None else gate2.get("criteria_source"),
            "parent_gate": None,
        },
        {
            "gate_id": "KNOWN_CASE_SOFT_VETO",
            "verdict": "DOWNGRADE"
            if known_summary is not None and known_summary.get("soft_veto") is True
            else "NO_DOWNGRADE",
            "criteria_source": "locked_registry",
            "parent_gate": "GATE2",
            "may_upgrade_parent": False,
        },
        {
            "gate_id": "GATE3",
            "verdict": gate3.get("verdict"),
            "criteria_source": gate3.get("criteria_source"),
            "parent_gate": "GATE2",
        },
    ]


def build_decision_log(traceability: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Reduce the locked traceability registry to a stable reporting ledger."""
    rows: list[dict[str, object]] = []
    for item in traceability:
        rows.append(
            {
                "decision_id": item.get("decision_id"),
                "lock_stage": item.get("lock_stage"),
                "test_ids": item.get("test_ids", []),
                "output_evidence": item.get("output_evidence", []),
            }
        )
    return sorted(rows, key=lambda item: str(item.get("decision_id")))


def build_chapter4_tables(
    ledger: list[dict[str, Any]],
    verdict_matrix: list[dict[str, object]],
    evaluations: list[dict[str, Any]],
) -> dict[str, object]:
    """Publish stable, machine-readable inputs; manuscript formatting stays downstream."""
    return {
        "table_gate_verdicts": verdict_matrix,
        "table_outer_performance": [
            item for item in ledger if item.get("result_type") == "outer_metric"
        ],
        "table_sensitivity_status": [
            item for item in ledger if item.get("result_type") == "sensitivity"
        ],
        "source_evaluation_fold_count": len(evaluations),
        "report_only": True,
    }


def render_gate_figure(verdict_matrix: list[dict[str, object]]) -> str:
    """Render a deterministic SVG summary without importing any training module."""
    rows = [(str(item.get("gate_id")), str(item.get("verdict"))) for item in verdict_matrix]
    height = 70 + 74 * len(rows)
    boxes: list[str] = []
    for index, (gate_id, verdict) in enumerate(rows):
        y = 40 + 74 * index
        color = (
            "#2e7d32"
            if verdict in {"PASS", "NO_DOWNGRADE"}
            else "#c62828"
            if verdict in {"FAIL", "DOWNGRADE"}
            else "#6d4c41"
        )
        boxes.extend(
            [
                f'<rect x="24" y="{y}" width="472" height="48" rx="8" fill="{color}"/>',
                f'<text x="42" y="{y + 30}" fill="white" font-family="Arial" font-size="16">{escape(gate_id)}: {escape(verdict)}</text>',
            ]
        )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="520" height="{height}" viewBox="0 0 520 {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            *boxes,
            "</svg>",
        ]
    )


def render_audit_report(
    *,
    protocol_hash: str,
    ledger: list[dict[str, Any]],
    verdict_matrix: list[dict[str, object]],
    upstream_inventory: list[dict[str, object]],
) -> str:
    """Summarize final integrity checks without recalculating analytical results."""
    missing_hashes = sum(not item.get("content_hash") for item in upstream_inventory)
    lines = [
        "# Final audit report",
        "",
        f"- Protocol hash: `{protocol_hash}`",
        f"- Verified upstream artifacts: {len(upstream_inventory)}",
        f"- Manifests missing content hashes: {missing_hashes}",
        f"- Final ledger rows: {len(ledger)}",
        f"- Verdict rows: {len(verdict_matrix)}",
        "- Models refit by P17: 0",
        "- Labels recalculated by P17: 0",
        "- New data opened by P17: 0",
        "",
        "The final artifact manifest is written after this report and therefore excludes only its own recursive hash.",
        "",
    ]
    return "\n".join(lines)
