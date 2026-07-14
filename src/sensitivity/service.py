"""Compute registered post-outer domain summaries and model-block ablations."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from core.metrics import average_precision
from core.semantic_keys import (
    CHANNEL_ID,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    SOURCE_ID,
)


def domain_transfer(
    *,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, object]],
    noninferiority_margin: float,
    columns: dict[str, str],
) -> dict[str, object]:
    bindings = [item for item in feature_registry if isinstance(item.get("domain_id"), str)]
    if not bindings:
        return {
            "status": "SKIPPED",
            "reason_code": "DOMAIN_BINDINGS_UNAVAILABLE",
            "domains": [],
            "robust_scenario_fraction": None,
        }
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    prediction = columns[PREDICTION]
    outcome = columns[OUTCOME]
    joined = predictions.merge(outcomes, on=[firm, year], how="inner", validate="m:1")
    rows: list[dict[str, object]] = []
    robust: list[bool] = []
    for binding in bindings:
        feature_id = str(binding["feature_id"])
        if feature_id not in feature_panel.columns:
            raise ValueError(f"domain feature={feature_id}: absent from feature panel")
        data = joined.merge(
            feature_panel[[firm, year, feature_id]], on=[firm, year], how="inner", validate="m:1"
        )
        for level, level_frame in data.groupby(feature_id, dropna=False, sort=True):
            model_ap: dict[str, float | None] = {}
            for model_id, model_frame in level_frame.groupby(learner, sort=True):
                model_ap[str(model_id)] = average_precision(
                    model_frame[outcome].astype(bool).tolist(),
                    model_frame[prediction].astype(float).tolist(),
                )
            comparisons: list[dict[str, object]] = []
            for candidate, candidate_ap in model_ap.items():
                if not candidate.endswith(":full"):
                    continue
                reference = candidate.removesuffix(":full") + ":observability_only"
                reference_ap = model_ap.get(reference)
                passed = (
                    candidate_ap is not None
                    and reference_ap is not None
                    and candidate_ap >= reference_ap * (1.0 - noninferiority_margin)
                )
                robust.append(passed)
                comparisons.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "candidate_ap": candidate_ap,
                        "reference_ap": reference_ap,
                        "noninferior": passed,
                    }
                )
            rows.append(
                {
                    "domain_id": binding["domain_id"],
                    "level": None if pd.isna(level) else str(level),
                    "rows": int(level_frame[[firm, year]].drop_duplicates().shape[0]),
                    "positives": int(
                        level_frame[[firm, year, outcome]].drop_duplicates()[outcome].sum()
                    ),
                    "comparisons": comparisons,
                }
            )
    return {
        "status": "PASS" if robust else "SKIPPED",
        "reason_code": None if robust else "INSUFFICIENT_POSITIVES",
        "domains": rows,
        "robust_scenario_fraction": sum(robust) / len(robust) if robust else None,
    }


def ablation_summary(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        comparisons = evaluation.get("comparisons")
        if not isinstance(comparisons, list):
            continue
        for comparison in cast(list[Any], comparisons):
            if isinstance(comparison, dict):
                rows.append(
                    {
                        OUTER_FOLD: evaluation.get(OUTER_FOLD),
                        "ablation": "remove_content_block",
                        **comparison,
                    }
                )
    return rows


def hierarchical_pi_status(capability: dict[str, object]) -> dict[str, object]:
    available = capability.get("status") == "AVAILABLE" and capability.get("pilot_executed") is True
    return {
        "status": "SKIPPED" if not available else "EMPIRICALLY_PENDING",
        "reason_code": None if available else "CAPABILITY_UNAVAILABLE",
        "role": "sensitivity_only",
        "entered_primary_selection": False,
        "estimated_pi_metrics_allowed": available,
        "fixed_pi_bias_or_rmse_reported": False,
    }


def source_sensitivity_summary(
    evidence: pd.DataFrame,
    lag_decomposition: dict[str, Any],
    columns: dict[str, str],
) -> dict[str, object]:
    """Register source-set and lag sensitivity capability without redefining labels."""
    source = columns[SOURCE_ID]
    channel = columns[CHANNEL_ID]
    sources = sorted(str(value) for value in evidence[source].dropna().unique())
    channels = sorted(str(value) for value in evidence[channel].dropna().unique())
    lag_records = lag_decomposition.get("records")
    lag_count = len(cast(list[Any], lag_records)) if isinstance(lag_records, list) else 0
    return {
        "status": "AVAILABLE" if len(sources) >= 2 else "SKIPPED",
        "reason_code": None if len(sources) >= 2 else "INSUFFICIENT_SOURCES",
        "source_ids": sources,
        "channel_ids": channels,
        "leave_one_source_out_registered": len(sources) >= 2,
        "strict_channel_holdout_registered": len(channels) >= 2,
        "event_deduplication_applied": True,
        "lag_record_count": lag_count,
        "horizon_sensitivities_months": [12, 24],
        "reuses_locked_evidence_ledger": True,
    }


def censoring_sensitivity_summary(
    diagnostics: list[dict[str, Any]], censoring_registry: list[dict[str, Any]]
) -> dict[str, object]:
    """Allow IPCW sensitivity only for folds whose development diagnostics pass."""
    eligible_folds = [
        str(item.get(OUTER_FOLD))
        for item in diagnostics
        if item.get("ipw_diagnostics_pass") is True
        and item.get("fit_scope") == "development_history"
        and item.get("outer_rows_used_in_fit") == 0
    ]
    prospective = sum(
        item.get("classification") == "prospective_immature" for item in censoring_registry
    )
    return {
        "status": "AVAILABLE" if eligible_folds else "SKIPPED",
        "reason_code": None if eligible_folds else "IPCW_DIAGNOSTICS_UNAVAILABLE",
        "eligible_outer_folds": eligible_folds,
        "role": "sensitivity_only",
        "prospective_immature_count": prospective,
        "immature_or_exit_assigned_negative": False,
    }
