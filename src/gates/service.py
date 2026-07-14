"""Evaluate immutable gate criteria without tuning them on outer results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, LEARNER_ID, OUTCOME, OUTER_FOLD


def gate2_verdict(
    *,
    evaluations: list[dict[str, Any]],
    bootstraps: list[list[dict[str, Any]]],
    domain_transfer: dict[str, Any],
    gate: dict[str, Any],
    common: dict[str, Any],
    confirmatory_folds: list[str],
) -> dict[str, Any]:
    comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    for evaluation in evaluations:
        fold = str(evaluation.get(OUTER_FOLD))
        if fold not in confirmatory_folds:
            continue
        raw_models = evaluation.get("models")
        if isinstance(raw_models, list):
            for raw_model in cast(list[Any], raw_models):
                model = cast(dict[str, Any], raw_model) if isinstance(raw_model, dict) else None
                if model is not None:
                    metrics[(fold, str(model.get(LEARNER_ID)))] = model
        raw_comparisons = evaluation.get("comparisons")
        if isinstance(raw_comparisons, list):
            for raw_comparison in cast(list[Any], raw_comparisons):
                comparison = (
                    cast(dict[str, Any], raw_comparison)
                    if isinstance(raw_comparison, dict)
                    else None
                )
                if comparison is not None:
                    comparisons[str(comparison.get("candidate"))].append(
                        {"fold": fold, **comparison}
                    )
    samples: dict[tuple[str, str], list[float]] = {}
    for batch in bootstraps:
        for item in batch:
            fold = str(item.get(OUTER_FOLD, ""))
            candidate = str(item.get("candidate"))
            raw = item.get("delta_ap_samples")
            if fold in confirmatory_folds and isinstance(raw, list):
                samples[(fold, candidate)] = [float(value) for value in cast(list[Any], raw)]
    candidate_results: list[dict[str, Any]] = []
    family_count = max(1, len(comparisons))
    interval_level = float(gate["simultaneous_interval_level"])
    alpha = (1.0 - interval_level) / (2.0 * family_count)
    absolute = float(_mapping(gate["minimum_meaningful_improvement"])["absolute"])
    relative = float(_mapping(gate["minimum_meaningful_improvement"])["relative_to_reference_ap"])
    for candidate, rows in sorted(comparisons.items()):
        reference = str(rows[0].get("reference")) if rows else ""
        deltas = [
            float(row["delta_average_precision"])
            for row in rows
            if row.get("delta_average_precision") is not None
        ]
        reference_aps = [
            float(metrics[(str(row["fold"]), reference)]["average_precision"])
            for row in rows
            if metrics.get((str(row["fold"]), reference), {}).get("average_precision") is not None
        ]
        paired_samples = [
            samples[(str(row["fold"]), candidate)]
            for row in rows
            if (str(row["fold"]), candidate) in samples
        ]
        minimum_length = min((len(value) for value in paired_samples), default=0)
        mean_samples = [
            float(np.mean([value[index] for value in paired_samples]))
            for index in range(minimum_length)
        ]
        lower = float(np.quantile(mean_samples, alpha)) if mean_samples else None
        full_precisions = [
            metrics.get((str(row["fold"]), candidate), {}).get("precision_at_primary_budget")
            for row in rows
        ]
        reference_precisions = [
            metrics.get((str(row["fold"]), reference), {}).get("precision_at_primary_budget")
            for row in rows
        ]
        declines: list[float] = []
        for full_value, reference_value in zip(full_precisions, reference_precisions, strict=True):
            if (
                isinstance(full_value, (int, float))
                and isinstance(reference_value, (int, float))
                and reference_value != 0
            ):
                declines.append(
                    max(
                        0.0,
                        (float(reference_value) - float(full_value)) / float(reference_value),
                    )
                )
        positives = sum(
            int(metrics.get((str(row["fold"]), candidate), {}).get("positives", 0)) for row in rows
        )
        reference_mean = float(np.mean(reference_aps)) if reference_aps else 0.0
        required = max(absolute, relative * reference_mean)
        point = float(np.mean(deltas)) if deltas else None
        complete = len(rows) == int(gate["fold_count"])
        passed = bool(
            complete
            and point is not None
            and point >= required
            and lower is not None
            and lower > 0.0
            and sum(value > 0 for value in deltas) >= int(gate["same_direction_min_folds"])
            and (max(declines) if declines else 0.0)
            <= float(gate["yield_at_primary_budget_relative_decline_max"])
            and positives >= int(gate["confirmatory_positive_count_min"])
        )
        candidate_results.append(
            {
                "candidate": candidate,
                "reference": reference,
                "folds": len(rows),
                "same_direction_folds": sum(value > 0 for value in deltas),
                "mean_delta_ap": point,
                "simultaneous_interval_lower": lower,
                "required_improvement": required,
                "maximum_relative_yield_decline": max(declines) if declines else 0.0,
                "positive_count": positives,
                "pass": passed,
            }
        )
    robust_fraction = domain_transfer.get("robust_scenario_fraction")
    evidence_complete = bool(candidate_results) and robust_fraction is not None
    robust_pass = (
        evidence_complete
        and isinstance(robust_fraction, (int, float))
        and float(robust_fraction) >= float(common["robust_scenario_fraction_min"])
    )
    verdict = (
        "INSUFFICIENT_EVIDENCE"
        if not evidence_complete
        else "PASS"
        if robust_pass and any(item["pass"] for item in candidate_results)
        else "FAIL"
    )
    return {
        "status": "PASS",
        "gate_id": "GATE2",
        "verdict": verdict,
        "candidates": candidate_results,
        "domain_robust_scenario_fraction": robust_fraction,
        "criteria_source": "locked_registry",
    }


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("gate configuration mapping required")
    return cast(dict[str, Any], value)


def gate3_verdict(
    *,
    gate2: dict[str, Any],
    known_case_results: list[dict[str, Any]],
    feature_panel: pd.DataFrame,
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    bindings: dict[str, Any],
    gate: dict[str, Any],
    confirmatory_folds: list[int],
    columns: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if gate2.get("verdict") != "PASS":
        threshold: dict[str, Any] = {
            "status": "SKIPPED",
            "reason_code": "PARENT_GATE_FAILED",
        }
        return threshold, _gate3_receipt("INELIGIBLE_PARENT_GATE_FAILED", threshold)
    empty_summary: dict[str, Any] = {}
    summary = next(
        (item for item in known_case_results if item.get("record_type") == "summary"),
        empty_summary,
    )
    required = {
        "pressure_feature_id",
        "monitoring_feature_id",
        "parent_model_id",
        "domain_feature_id",
    }
    if any(bindings.get(key) in {None, ""} for key in required):
        threshold = {
            "status": "SKIPPED",
            "reason_code": "INTERACTION_BINDINGS_UNAVAILABLE",
            "known_case_soft_veto": bool(summary.get("soft_veto")),
        }
        return threshold, _gate3_receipt("INSUFFICIENT_EVIDENCE", threshold)
    pressure = str(bindings["pressure_feature_id"])
    monitoring = str(bindings["monitoring_feature_id"])
    domain = str(bindings["domain_feature_id"])
    parent = str(bindings["parent_model_id"])
    if not {pressure, monitoring, domain}.issubset(feature_panel.columns):
        raise ValueError("Gate 3 feature bindings are not present in the feature panel")
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    outcome = columns[OUTCOME]
    keys = predictions.loc[predictions[learner] == parent, [firm, year]].drop_duplicates()
    data = keys.merge(
        feature_panel[[firm, year, pressure, monitoring, domain]], on=[firm, year], validate="1:1"
    )
    data = data.merge(outcomes[[firm, year, outcome]], on=[firm, year], validate="1:1")
    data = data.dropna(subset=[pressure, monitoring, domain, outcome]).copy()
    if data.empty or float(data[pressure].std(ddof=0)) == 0.0:
        threshold = {"status": "SKIPPED", "reason_code": "INSUFFICIENT_POSITIVES"}
        return threshold, _gate3_receipt("INSUFFICIENT_EVIDENCE", threshold)
    data["_pressure_z"] = (data[pressure] - data[pressure].mean()) / data[pressure].std(ddof=0)
    directions: list[float] = []
    breakpoints: list[float] = []
    side_fractions: list[float] = []
    for fold in confirmatory_folds:
        frame = data.loc[data[year].astype(int) == fold]
        if len(frame) < 8 or frame[outcome].nunique() < 2:
            continue
        directions.append(_interaction_coefficient(frame, monitoring, outcome))
        point, side = _breakpoint(frame, monitoring, outcome)
        if point is not None:
            breakpoints.append(point)
            side_fractions.append(side)
    domain_points: dict[str, float] = {}
    ranges: list[tuple[float, float]] = []
    for level, frame in data.groupby(domain, sort=True):
        point, _ = _breakpoint(frame, monitoring, outcome)
        if point is not None:
            domain_points[str(level)] = point
            ranges.append((float(frame["_pressure_z"].min()), float(frame["_pressure_z"].max())))
    support = _common_support(data, ranges)
    direction_count = (
        max(sum(value > 0 for value in directions), sum(value < 0 for value in directions))
        if directions
        else 0
    )
    stability = float(np.std(breakpoints, ddof=0)) if breakpoints else None
    domain_difference = (
        max(domain_points.values()) - min(domain_points.values())
        if len(domain_points) >= 2
        else None
    )
    passed = bool(
        not summary.get("soft_veto")
        and stability is not None
        and stability <= float(gate["breakpoint_tolerance_training_sd"])
        and domain_difference is not None
        and domain_difference <= float(gate["cross_domain_breakpoint_difference_max_sd"])
        and support >= float(gate["common_support_min_fraction_each_domain"])
        and side_fractions
        and min(side_fractions) >= float(gate["minimum_fraction_each_side"])
        and direction_count >= int(gate["same_direction_min_folds"])
        and len(domain_points) >= int(gate["minimum_domains"])
    )
    threshold = {
        "status": "PASS",
        "parent_model_id": parent,
        "fold_breakpoints_training_sd": breakpoints,
        "breakpoint_stability_sd": stability,
        "domain_breakpoints_training_sd": domain_points,
        "maximum_cross_domain_difference_sd": domain_difference,
        "common_support_fraction": support,
        "minimum_fraction_each_side": min(side_fractions) if side_fractions else None,
        "same_direction_folds": direction_count,
        "known_case_soft_veto": bool(summary.get("soft_veto")),
        "pass": passed,
    }
    return threshold, _gate3_receipt("PASS" if passed else "FAIL", threshold)


def _interaction_coefficient(frame: pd.DataFrame, monitoring: str, outcome: str) -> float:
    pressure = frame["_pressure_z"].to_numpy(dtype=float)
    monitor = frame[monitoring].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), pressure, monitor, pressure * monitor])
    coefficients, *_ = np.linalg.lstsq(design, frame[outcome].to_numpy(dtype=float), rcond=None)
    return float(coefficients[-1])


def _breakpoint(frame: pd.DataFrame, monitoring: str, outcome: str) -> tuple[float | None, float]:
    if len(frame) < 8:
        return None, 0.0
    pressure = frame["_pressure_z"].to_numpy(dtype=float)
    monitor = frame[monitoring].to_numpy(dtype=float)
    target = frame[outcome].to_numpy(dtype=float)
    candidates = np.unique(np.quantile(pressure, np.linspace(0.1, 0.9, 17)))
    best: tuple[float, float, float] | None = None
    for point in candidates:
        point_value = float(point)
        below = float(np.asarray(pressure <= point_value, dtype=float).mean())
        side = min(below, 1.0 - below)
        hinge = np.maximum(pressure - point_value, 0.0)
        design = np.column_stack(
            [np.ones(len(frame)), pressure, monitor, pressure * monitor, hinge, hinge * monitor]
        )
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        loss = float(np.mean((target - design @ coefficients) ** 2))
        if best is None or loss < best[0]:
            best = (loss, point_value, side)
    return (best[1], best[2]) if best is not None else (None, 0.0)


def _common_support(data: pd.DataFrame, ranges: list[tuple[float, float]]) -> float:
    if len(ranges) < 2:
        return 0.0
    lower = max(item[0] for item in ranges)
    upper = min(item[1] for item in ranges)
    if lower >= upper:
        return 0.0
    return float(data["_pressure_z"].between(lower, upper).mean())


def _gate3_receipt(verdict: str, threshold: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "gate_id": "GATE3",
        "verdict": verdict,
        "criteria_source": "locked_registry",
        "threshold_status": threshold.get("status"),
    }
