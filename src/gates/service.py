"""Evaluate immutable gate criteria without tuning them on outer results."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

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
        positive_counts = [
            int(metrics.get((str(row["fold"]), candidate), {}).get("positives", 0)) for row in rows
        ]
        reference_mean = float(np.mean(reference_aps)) if reference_aps else 0.0
        required = max(absolute, relative * reference_mean)
        point = float(np.mean(deltas)) if deltas else None
        observed_folds = {str(row["fold"]) for row in rows}
        complete = observed_folds == set(confirmatory_folds)
        bootstrap_complete = len(paired_samples) == int(gate["fold_count"]) and minimum_length > 0
        candidate_evidence_complete = bool(
            complete
            and bootstrap_complete
            and len(positive_counts) == int(gate["fold_count"])
            and len(reference_aps) == int(gate["fold_count"])
        )
        passed = bool(
            candidate_evidence_complete
            and point is not None
            and point >= required
            and lower is not None
            and lower > 0.0
            and sum(value > 0 for value in deltas) >= int(gate["same_direction_min_folds"])
            and (max(declines) if declines else 0.0)
            <= float(gate["yield_at_primary_budget_relative_decline_max"])
            and len(positive_counts) == int(gate["fold_count"])
            and min(positive_counts) >= int(gate["confirmatory_positive_count_min"])
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
                "positive_count_by_fold": positive_counts,
                "minimum_positive_count": min(positive_counts) if positive_counts else 0,
                "bootstrap_fold_count": len(paired_samples),
                "bootstrap_replications_used": minimum_length,
                "evidence_complete": candidate_evidence_complete,
                "pass": passed,
            }
        )
    robust_fraction = domain_transfer.get("robust_scenario_fraction")
    evidence_complete = bool(
        candidate_results
        and all(item["evidence_complete"] is True for item in candidate_results)
        and robust_fraction is not None
        and domain_transfer.get("status") == "PASS"
        and domain_transfer.get("leave_one_domain_out_refit_executed") is True
    )
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
        "status": verdict,
        "gate_id": "GATE2",
        "verdict": verdict,
        "candidates": candidate_results,
        "domain_robust_scenario_fraction": robust_fraction,
        "evidence_complete": evidence_complete,
        "reason_code": None if evidence_complete else "REQUIRED_GATE2_EVIDENCE_INCOMPLETE",
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
    bootstrap_replications: int = 500,
    familywise_alpha: float = 0.05,
    confirmatory_threshold_claims: int = 2,
    measurement_selections: list[dict[str, Any]] | None = None,
    measurement_stability_minimum: int = 3,
    measurement_stability_denominator: int = 4,
    rng: np.random.Generator | None = None,
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
    stability = _measurement_stability(
        measurement_selections or [],
        required_count=measurement_stability_minimum,
        required_denominator=measurement_stability_denominator,
    )
    if stability["pass"] is not True:
        threshold = {
            "status": "SKIPPED",
            "reason_code": "MEASUREMENT_SELECTION_UNSTABLE",
            "measurement_stability": stability,
            "known_case_soft_veto": bool(summary.get("soft_veto")),
        }
        return threshold, _gate3_receipt("INSUFFICIENT_EVIDENCE", threshold)
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
    raw_threshold_features = bindings.get("threshold_feature_ids")
    if (
        not isinstance(raw_threshold_features, list)
        or len(cast(list[object], raw_threshold_features)) != confirmatory_threshold_claims
        or any(
            not isinstance(value, str) or not value
            for value in cast(list[object], raw_threshold_features)
        )
    ):
        threshold = {
            "status": "SKIPPED",
            "reason_code": "TWO_THRESHOLD_BINDINGS_REQUIRED",
            "required_threshold_claims": confirmatory_threshold_claims,
            "known_case_soft_veto": bool(summary.get("soft_veto")),
        }
        return threshold, _gate3_receipt("INSUFFICIENT_EVIDENCE", threshold)
    threshold_features = [str(value) for value in cast(list[object], raw_threshold_features)]
    if len(set(threshold_features)) != len(threshold_features):
        raise ValueError("Gate 3 threshold feature bindings must be distinct")
    pressure = str(bindings["pressure_feature_id"])
    monitoring = str(bindings["monitoring_feature_id"])
    domain = str(bindings["domain_feature_id"])
    parent = str(bindings["parent_model_id"])
    required_features = {pressure, monitoring, domain, *threshold_features}
    if not required_features.issubset(feature_panel.columns):
        raise ValueError("Gate 3 feature bindings are not present in the feature panel")
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    outcome = columns[OUTCOME]
    keys = predictions.loc[predictions[learner] == parent, [firm, year]].drop_duplicates()
    data = keys.merge(
        feature_panel[[firm, year, *sorted(required_features)]],
        on=[firm, year],
        validate="1:1",
    )
    data = data.merge(outcomes[[firm, year, outcome]], on=[firm, year], validate="1:1")
    data = data.dropna(subset=[*sorted(required_features), outcome]).copy()
    if data.empty or float(data[pressure].std(ddof=0)) == 0.0:
        threshold = {"status": "SKIPPED", "reason_code": "INSUFFICIENT_POSITIVES"}
        return threshold, _gate3_receipt("INSUFFICIENT_EVIDENCE", threshold)
    data["_pressure_z"] = (data[pressure] - data[pressure].mean()) / data[pressure].std(ddof=0)
    directions: list[float] = []
    direction_intervals: list[tuple[float, float]] = []
    for fold in confirmatory_folds:
        frame = data.loc[data[year].astype(int) == fold]
        if len(frame) < 8 or frame[outcome].nunique() < 2:
            continue
        directions.append(_interaction_coefficient(frame, monitoring, outcome))
        if rng is not None:
            direction_interval, _ = _shape_bootstrap(
                frame=frame,
                monitoring=monitoring,
                outcome=outcome,
                replications=bootstrap_replications,
                alpha=familywise_alpha / 3.0,
                rng=rng,
            )
            direction_intervals.append(direction_interval)
    direction_count = (
        max(sum(value > 0 for value in directions), sum(value < 0 for value in directions))
        if directions
        else 0
    )
    breakpoint_tolerance = float(gate["breakpoint_tolerance_training_sd"])
    threshold_claim_results = [
        _threshold_claim(
            data=data,
            feature_id=feature_id,
            monitoring=monitoring,
            domain=domain,
            year=year,
            outcome=outcome,
            confirmatory_folds=confirmatory_folds,
            gate=gate,
            bootstrap_replications=bootstrap_replications,
            familywise_alpha=familywise_alpha,
            rng=rng,
        )
        for feature_id in threshold_features
    ]
    interaction_uncertainty_pass = bool(direction_intervals) and all(
        lower > 0 or upper < 0 for lower, upper in direction_intervals
    )
    passed = bool(
        len(threshold_claim_results) == confirmatory_threshold_claims
        and all(item["method_pass"] is True for item in threshold_claim_results)
        and direction_count >= int(gate["same_direction_min_folds"])
        and interaction_uncertainty_pass
    )
    threshold = {
        "status": "PASS",
        "parent_model_id": parent,
        "threshold_claims": threshold_claim_results,
        "required_threshold_claims": confirmatory_threshold_claims,
        "breakpoint_tolerance_training_sd": breakpoint_tolerance,
        "same_direction_folds": direction_count,
        "simultaneous_interaction_intervals": direction_intervals,
        "familywise_alpha": familywise_alpha,
        "family_claim_count": confirmatory_threshold_claims + 1,
        "interaction_uncertainty_pass": interaction_uncertainty_pass,
        "known_case_soft_veto": bool(summary.get("soft_veto")),
        "measurement_stability": stability,
        "method_pass": passed,
        "interpretation_downgraded": bool(passed and summary.get("soft_veto")),
    }
    verdict = (
        "PASS_WITH_SOFT_VETO"
        if passed and summary.get("soft_veto")
        else "PASS"
        if passed
        else "FAIL"
    )
    return threshold, _gate3_receipt(verdict, threshold)


def _measurement_stability(
    selections: list[dict[str, Any]], *, required_count: int, required_denominator: int
) -> dict[str, Any]:
    values = [
        str(item.get("selected_measurement"))
        for item in selections
        if item.get("selected_measurement") not in {None, "", "none"}
    ]
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    selected, count = max(counts.items(), key=lambda item: item[1]) if counts else (None, 0)
    complete = len(selections) == required_denominator
    return {
        "pass": bool(complete and count >= required_count),
        "selected_measurement": selected,
        "selected_fold_count": count,
        "required_selected_fold_count": required_count,
        "observed_fold_count": len(selections),
        "required_fold_count": required_denominator,
        "selection_counts": counts,
        "none_or_missing_count": len(selections) - len(values),
    }


def _threshold_claim(
    *,
    data: pd.DataFrame,
    feature_id: str,
    monitoring: str,
    domain: str,
    year: str,
    outcome: str,
    confirmatory_folds: list[int],
    gate: dict[str, Any],
    bootstrap_replications: int,
    familywise_alpha: float,
    rng: np.random.Generator | None,
) -> dict[str, Any]:
    """Evaluate one prespecified threshold claim over outer folds and domains."""
    frame = data.copy()
    standard_deviation = float(frame[feature_id].std(ddof=0))
    if standard_deviation <= 0:
        return {
            "feature_id": feature_id,
            "status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "ZERO_VARIANCE_THRESHOLD_FEATURE",
            "method_pass": False,
        }
    frame["_pressure_z"] = (
        frame[feature_id] - float(frame[feature_id].mean())
    ) / standard_deviation
    breakpoints: list[float] = []
    breakpoint_intervals: list[tuple[float, float]] = []
    side_fractions: list[float] = []
    for fold in confirmatory_folds:
        fold_frame = frame.loc[frame[year].astype(int) == fold]
        if len(fold_frame) < 8 or fold_frame[outcome].nunique() < 2:
            continue
        point, side = _breakpoint(fold_frame, monitoring, outcome)
        if point is None:
            continue
        breakpoints.append(point)
        side_fractions.append(side)
        if rng is not None:
            _, interval = _shape_bootstrap(
                frame=fold_frame,
                monitoring=monitoring,
                outcome=outcome,
                replications=bootstrap_replications,
                alpha=familywise_alpha / 3.0,
                rng=rng,
            )
            breakpoint_intervals.append(interval)
    domain_points: dict[str, float] = {}
    ranges: list[tuple[float, float]] = []
    for level, domain_frame in frame.groupby(domain, sort=True):
        point, _ = _breakpoint(domain_frame, monitoring, outcome)
        if point is not None:
            domain_points[str(level)] = point
            ranges.append(
                (
                    float(domain_frame["_pressure_z"].min()),
                    float(domain_frame["_pressure_z"].max()),
                )
            )
    support = _common_support(frame, ranges)
    stability = float(np.std(breakpoints, ddof=0)) if breakpoints else None
    domain_difference = (
        max(domain_points.values()) - min(domain_points.values())
        if len(domain_points) >= 2
        else None
    )
    passed = bool(
        stability is not None
        and breakpoints
        and all(
            abs(value) <= float(gate["breakpoint_tolerance_training_sd"]) for value in breakpoints
        )
        and domain_difference is not None
        and domain_difference <= float(gate["cross_domain_breakpoint_difference_max_sd"])
        and support >= float(gate["common_support_min_fraction_each_domain"])
        and side_fractions
        and min(side_fractions) >= float(gate["minimum_fraction_each_side"])
        and len(domain_points) >= int(gate["minimum_domains"])
        and len(breakpoint_intervals) == len(breakpoints)
    )
    return {
        "feature_id": feature_id,
        "status": "PASS" if passed else "FAIL",
        "fold_breakpoints_training_sd": breakpoints,
        "simultaneous_breakpoint_intervals": breakpoint_intervals,
        "breakpoint_stability_sd": stability,
        "domain_breakpoints_training_sd": domain_points,
        "maximum_cross_domain_difference_sd": domain_difference,
        "common_support_fraction": support,
        "minimum_fraction_each_side": min(side_fractions) if side_fractions else None,
        "method_pass": passed,
    }


def _interaction_coefficient(frame: pd.DataFrame, monitoring: str, outcome: str) -> float:
    pressure = frame["_pressure_z"].to_numpy(dtype=float)
    monitor = frame[monitoring].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), pressure, monitor, pressure * monitor])
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, fit_intercept=False)
    cast(Any, model).fit(design, frame[outcome].astype(int).to_numpy())
    coefficients = np.asarray(cast(Any, model).coef_, dtype=float)
    return float(coefficients[0, -1])


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
        if len(np.unique(target)) < 2:
            continue
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, fit_intercept=False)
        cast(Any, model).fit(design, target.astype(int))
        probability = np.asarray(cast(Any, model).predict_proba(design), dtype=float)[:, 1]
        loss = float(log_loss(target.astype(int), probability, labels=[0, 1]))
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
    fractions = [
        max(0.0, upper - lower) / max(upper_bound - lower_bound, 1e-12)
        for lower_bound, upper_bound in ranges
    ]
    return min(fractions, default=0.0)


def _shape_bootstrap(
    *,
    frame: pd.DataFrame,
    monitoring: str,
    outcome: str,
    replications: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], tuple[float, float]]:
    coefficients: list[float] = []
    breakpoints: list[float] = []
    for _ in range(replications):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        if sample[outcome].nunique() < 2:
            continue
        coefficients.append(_interaction_coefficient(sample, monitoring, outcome))
        point, _ = _breakpoint(sample, monitoring, outcome)
        if point is not None:
            breakpoints.append(point)
    quantiles = (alpha / 2.0, 1.0 - alpha / 2.0)
    coefficient_interval = (
        (
            float(np.quantile(coefficients, quantiles[0])),
            float(np.quantile(coefficients, quantiles[1])),
        )
        if coefficients
        else (-math.inf, math.inf)
    )
    breakpoint_interval = (
        (
            float(np.quantile(breakpoints, quantiles[0])),
            float(np.quantile(breakpoints, quantiles[1])),
        )
        if breakpoints
        else (-math.inf, math.inf)
    )
    return coefficient_interval, breakpoint_interval


def _gate3_receipt(verdict: str, threshold: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": verdict,
        "gate_id": "GATE3",
        "verdict": verdict,
        "criteria_source": "locked_registry",
        "threshold_status": threshold.get("status"),
    }
