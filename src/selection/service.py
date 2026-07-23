"""Nested time/channel measurement selection without outer-outcome access."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from core.semantic_keys import ELIGIBLE, FIRM_ID, FISCAL_YEAR, MATURE, OUTER_FOLD, TARGET_VALUE
from labels.latent_class import LatentClassResult, fit_fixed_pi_latent_class


@dataclass(frozen=True)
class SelectionResult:
    candidates: list[dict[str, Any]]
    selection: dict[str, Any]
    channel_selection: dict[str, Any]


def select_measurement(
    *,
    matrices: dict[str, Any],
    outer_year: int,
    candidates: list[str],
    l3_capability: dict[str, Any],
    l3_fold_result: dict[str, Any] | None = None,
    minimum_observed_channels: int | None = None,
    nested_candidate_results: list[dict[str, Any]] | None = None,
) -> SelectionResult:
    """Select using years before the outer fold and held-channel predictions only."""
    raw_rows = matrices.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("measurement matrices require rows")
    raw_rows = cast(list[Any], raw_rows)
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        row = cast(dict[str, Any], raw_row)
        if int(row[FISCAL_YEAR]) < outer_year:
            rows.append(row)
    expected = matrices.get("expected_channels")
    if not isinstance(expected, list):
        raise ValueError("measurement matrices require expected_channels")
    expected = cast(list[Any], expected)
    raw_l2_scoring = matrices.get("l2_scoring")
    l2_scoring = cast(dict[str, Any], raw_l2_scoring) if isinstance(raw_l2_scoring, dict) else {}
    l2_available = l2_scoring.get("status") == "AVAILABLE"
    channel_results: list[dict[str, Any]] = []
    for heldout in sorted(str(value) for value in expected):
        losses: list[float] = []
        for row in rows:
            outcomes = row.get("channel_outcomes")
            if not isinstance(outcomes, dict):
                continue
            outcomes = cast(dict[str, Any], outcomes)
            raw_scores = row.get("channel_evidence_scores")
            if not isinstance(raw_scores, dict) or not l2_available:
                continue
            scores = cast(dict[str, Any], raw_scores)
            if outcomes.get(heldout) is None:
                continue
            remaining = [
                float(value)
                for channel, value in scores.items()
                if channel != heldout and value is not None
            ]
            if minimum_observed_channels is None or len(remaining) < minimum_observed_channels:
                continue
            probability = min(1.0 - 1e-6, max(1e-6, sum(remaining) / len(remaining)))
            outcome = float(bool(outcomes[heldout]))
            losses.append(
                -(outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability))
            )
        channel_results.append(
            {
                "heldout_channel": heldout,
                "candidate": "L2",
                "rows": len(losses),
                "soft_cross_entropy": sum(losses) / len(losses) if losses else None,
                "heldout_removed_from_target_and_measurement": True,
            }
        )
    complete = bool(channel_results) and all(item["rows"] > 0 for item in channel_results)
    results: list[dict[str, Any]] = []
    if "L2" in candidates:
        losses = [
            float(item["soft_cross_entropy"])
            for item in channel_results
            if item["soft_cross_entropy"] is not None
        ]
        results.append(
            {
                "candidate": "L2",
                ELIGIBLE: complete and minimum_observed_channels is not None and l2_available,
                "objective": sum(losses) / len(losses) if complete and losses else None,
                "reason_code": None
                if complete and minimum_observed_channels is not None and l2_available
                else str(l2_scoring.get("reason_code"))
                if not l2_available
                else "L2_MINIMUM_COVERAGE_NOT_LOCKED"
                if minimum_observed_channels is None
                else "INSUFFICIENT_CHANNELS",
                "minimum_observed_channels": minimum_observed_channels,
            }
        )
    if "L3_fixed_pi" in candidates:
        pilot_available = l3_capability.get("status") == "AVAILABLE" and bool(
            l3_capability.get("pilot_executed") is True
        )
        fold_result = l3_fold_result or {}
        available = bool(
            pilot_available
            and fold_result.get("status") == "PASS"
            and fold_result.get("selection_objective") is not None
        )
        results.append(
            {
                "candidate": "L3_fixed_pi",
                ELIGIBLE: available,
                "objective": fold_result.get("selection_objective") if available else None,
                "reason_code": None
                if available
                else str(
                    fold_result.get(
                        "reason_code",
                        "L3_PILOT_UNAVAILABLE"
                        if not pilot_available
                        else "L3_FOLD_LOCAL_SELECTION_UNAVAILABLE",
                    )
                ),
                "selected_fixed_pi": fold_result.get("selected_fixed_pi"),
                "fit_scope": fold_result.get("fit_scope"),
            }
        )
    if nested_candidate_results is not None:
        proxy_by_candidate = {str(item["candidate"]): item for item in results}
        nested_by_candidate = {
            str(item["candidate"]): item
            for item in nested_candidate_results
            if isinstance(item.get("candidate"), str)
        }
        results = [
            {**proxy_by_candidate.get(candidate, {}), **nested_by_candidate[candidate]}
            for candidate in candidates
            if candidate != "none" and candidate in nested_by_candidate
        ]
    eligible = [item for item in results if item[ELIGIBLE] and item["objective"] is not None]
    selected = (
        min(eligible, key=lambda item: float(item["objective"]))["candidate"]
        if eligible
        else "none"
    )
    reason = None if eligible else "NO_ELIGIBLE_CANDIDATE"
    return SelectionResult(
        candidates=results,
        selection={
            OUTER_FOLD: str(outer_year),
            "selected_measurement": selected,
            "reason_code": reason,
            "selection_scope": "development_history_only",
            "outer_outcomes_accessed": False,
            "fit_max_year": max((int(row[FISCAL_YEAR]) for row in rows), default=None),
        },
        channel_selection={
            OUTER_FOLD: str(outer_year),
            "strict_channel_results": channel_results,
            "heldout_channel_removed_from_all_selection_inputs": True,
            "l3_strict_channel_results": (l3_fold_result or {}).get("strict_channel_results", []),
            "l3_scenario_results": (l3_fold_result or {}).get("scenario_results", []),
            "l3_target_rows": (l3_fold_result or {}).get("target_rows", [])
            if selected == "L3_fixed_pi"
            else [],
            "l3_selected_fixed_pi": (l3_fold_result or {}).get("selected_fixed_pi")
            if selected == "L3_fixed_pi"
            else None,
            "l3_selection_objective": (l3_fold_result or {}).get("selection_objective")
            if selected == "L3_fixed_pi"
            else None,
            "l3_full_fit_diagnostics": (l3_fold_result or {}).get("full_fit_diagnostics")
            if selected == "L3_fixed_pi"
            else None,
            "l3_source_accuracy": (l3_fold_result or {}).get("source_accuracy")
            if selected == "L3_fixed_pi"
            else None,
            "l3_channel_random_effect_sd": (l3_fold_result or {}).get("channel_random_effect_sd")
            if selected == "L3_fixed_pi"
            else None,
            "l3_fit_scope": (l3_fold_result or {}).get("fit_scope"),
            "nested_candidate_results": nested_candidate_results or [],
            "outer_outcomes_accessed": False,
        },
    )


def fit_l3_fold_candidate(
    *,
    matrices: dict[str, Any],
    outer_year: int,
    source_channels: dict[str, str],
    accuracy_priors: dict[str, dict[str, float]],
    fixed_pi_grid: list[float],
    mcmc: dict[str, Any],
    minimum_observed_channels: int,
    robust_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Refit L3 inside one time fold and score it with strict channel holdouts."""
    if minimum_observed_channels < 1:
        raise ValueError("L3 fold selection requires positive minimum channel coverage")
    if not fixed_pi_grid:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "L3_FIXED_PI_GRID_NOT_LOCKED",
            "selection_objective": None,
            "target_rows": [],
            "outer_outcomes_accessed": False,
        }
    if not 0 < robust_fraction <= 1:
        raise ValueError("L3 robust scenario fraction must be in (0, 1]")
    raw_rows = matrices.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("L3 fold selection requires matrix rows")
    rows: list[dict[str, Any]] = []
    for raw in cast(list[object], raw_rows):
        if not isinstance(raw, dict):
            continue
        row = cast(dict[str, Any], raw)
        row_year = row.get(FISCAL_YEAR)
        if not isinstance(row_year, int):
            raise ValueError("L3 matrix row requires an integer fiscal year")
        if row_year < outer_year and row.get(MATURE) is True:
            rows.append(row)
    if not rows:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "NO_MATURE_DEVELOPMENT_ROWS_FOR_L3",
            "selection_objective": None,
            "target_rows": [],
            "outer_outcomes_accessed": False,
        }
    expected_channels = sorted(set(source_channels.values()))
    scenario_results: list[dict[str, Any]] = []
    all_strict_results: list[dict[str, Any]] = []
    for fixed_pi in fixed_pi_grid:
        heldout_results: list[dict[str, Any]] = []
        for heldout in expected_channels:
            remaining_sources = {
                source: channel for source, channel in source_channels.items() if channel != heldout
            }
            if not remaining_sources:
                heldout_results.append(
                    {
                        "fixed_pi": fixed_pi,
                        "heldout_channel": heldout,
                        "rows": 0,
                        "soft_cross_entropy": None,
                        "diagnostics_eligible": False,
                        "reason_code": "NO_SOURCES_AFTER_CHANNEL_HOLDOUT",
                        "heldout_removed_from_target_and_measurement": True,
                        "target_rows": [],
                    }
                )
                continue
            fit_rows = [
                {
                    "source_outcomes": {
                        source: value
                        for source, value in _source_outcomes(row).items()
                        if source in remaining_sources
                    }
                }
                for row in rows
            ]
            try:
                fit = _fit_l3(
                    rows=fit_rows,
                    source_channels=remaining_sources,
                    accuracy_priors={
                        source: accuracy_priors[source] for source in remaining_sources
                    },
                    fixed_pi=float(fixed_pi),
                    mcmc=mcmc,
                    rng=_child_rng(rng),
                )
            except ValueError as error:
                heldout_results.append(
                    {
                        "fixed_pi": fixed_pi,
                        "heldout_channel": heldout,
                        "rows": 0,
                        "soft_cross_entropy": None,
                        "diagnostics_eligible": False,
                        "reason_code": f"L3_FIT_FAILED:{error}",
                        "heldout_removed_from_target_and_measurement": True,
                        "target_rows": [],
                    }
                )
                continue
            losses: list[float] = []
            heldout_target_rows: list[dict[str, object]] = []
            for row, probability in zip(rows, fit.posterior_mean, strict=True):
                channel_outcomes = _channel_outcomes(row)
                heldout_value = channel_outcomes.get(heldout)
                remaining_observed = sum(
                    value is not None
                    for channel, value in channel_outcomes.items()
                    if channel != heldout
                )
                if heldout_value is None or remaining_observed < minimum_observed_channels:
                    continue
                heldout_target_rows.append(
                    {
                        "heldout_channel": heldout,
                        FIRM_ID: str(row[FIRM_ID]),
                        FISCAL_YEAR: int(row[FISCAL_YEAR]),
                        TARGET_VALUE: float(probability),
                    }
                )
                clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
                outcome = float(bool(heldout_value))
                losses.append(
                    -(outcome * math.log(clipped) + (1.0 - outcome) * math.log(1.0 - clipped))
                )
            diagnostics_eligible = fit.diagnostics.get("eligible_for_gate1") is True
            heldout_results.append(
                {
                    "fixed_pi": float(fixed_pi),
                    "heldout_channel": heldout,
                    "rows": len(losses),
                    "soft_cross_entropy": float(np.mean(losses)) if losses else None,
                    "diagnostics_eligible": diagnostics_eligible,
                    "reason_code": None
                    if diagnostics_eligible and losses
                    else "L3_DIAGNOSTICS_FAILED"
                    if not diagnostics_eligible
                    else "INSUFFICIENT_HELDOUT_CHANNEL_ROWS",
                    "heldout_removed_from_target_and_measurement": True,
                    "fit_diagnostics": fit.diagnostics,
                    "target_rows": heldout_target_rows,
                }
            )
        complete = bool(heldout_results) and all(
            item["diagnostics_eligible"] is True and item["soft_cross_entropy"] is not None
            for item in heldout_results
        )
        losses = [
            float(item["soft_cross_entropy"])
            for item in heldout_results
            if item["soft_cross_entropy"] is not None
        ]
        scenario_results.append(
            {
                "fixed_pi": float(fixed_pi),
                ELIGIBLE: complete,
                "selection_objective": float(np.mean(losses)) if complete and losses else None,
                "heldout_channel_count": len(heldout_results),
            }
        )
        all_strict_results.extend(heldout_results)
    full_fits_by_pi: dict[str, LatentClassResult] = {}
    full_fit_rows = [{"source_outcomes": _source_outcomes(row)} for row in rows]
    for scenario in scenario_results:
        if scenario[ELIGIBLE] is not True:
            continue
        fixed_pi = float(scenario["fixed_pi"])
        try:
            full_fit = _fit_l3(
                rows=full_fit_rows,
                source_channels=source_channels,
                accuracy_priors=accuracy_priors,
                fixed_pi=fixed_pi,
                mcmc=mcmc,
                rng=_child_rng(rng),
            )
        except ValueError as error:
            scenario[ELIGIBLE] = False
            scenario["reason_code"] = f"L3_FULL_REFIT_FAILED:{error}"
            continue
        scenario["full_fit_diagnostics"] = full_fit.diagnostics
        scenario["source_accuracy"] = full_fit.source_accuracy
        scenario["channel_random_effect_sd"] = full_fit.channel_random_effect_sd
        scenario["posterior_parameter_draws"] = full_fit.parameter_draws
        if full_fit.diagnostics.get("eligible_for_gate1") is not True:
            scenario[ELIGIBLE] = False
            scenario["reason_code"] = "L3_FULL_REFIT_DIAGNOSTICS_FAILED"
            continue
        full_fits_by_pi[_pi_key(fixed_pi)] = full_fit
    eligible_scenarios = [
        item
        for item in scenario_results
        if item[ELIGIBLE] is True and item["selection_objective"] is not None
    ]
    eligible_fraction = len(eligible_scenarios) / len(scenario_results)
    if not eligible_scenarios or eligible_fraction < robust_fraction:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason_code": "L3_FOLD_DIAGNOSTICS_OR_ROBUSTNESS_FAILED",
            "selection_objective": None,
            "scenario_results": scenario_results,
            "strict_channel_results": all_strict_results,
            "eligible_scenario_fraction": eligible_fraction,
            "required_robust_fraction": robust_fraction,
            "target_rows": [],
            "fit_scope": "development_history_only",
            "fit_max_year": max(int(row[FISCAL_YEAR]) for row in rows),
            "outer_outcomes_accessed": False,
        }
    selected_scenario = min(eligible_scenarios, key=lambda item: float(item["selection_objective"]))
    selected_pi = float(selected_scenario["fixed_pi"])
    full_fit = full_fits_by_pi[_pi_key(selected_pi)]
    target_rows = [
        {
            FIRM_ID: str(row[FIRM_ID]),
            FISCAL_YEAR: int(row[FISCAL_YEAR]),
            TARGET_VALUE: float(probability),
        }
        for row, probability in zip(rows, full_fit.posterior_mean, strict=True)
        if int(row.get("observed_channel_count", 0)) >= minimum_observed_channels
    ]
    return {
        "status": "PASS",
        "reason_code": None,
        "selection_objective": float(selected_scenario["selection_objective"]),
        "selected_fixed_pi": selected_pi,
        "scenario_results": scenario_results,
        "strict_channel_results": all_strict_results,
        "eligible_scenario_fraction": eligible_fraction,
        "required_robust_fraction": robust_fraction,
        "full_fit_diagnostics": full_fit.diagnostics,
        "source_accuracy": full_fit.source_accuracy,
        "channel_random_effect_sd": full_fit.channel_random_effect_sd,
        "target_rows": target_rows,
        "heldout_target_rows": [
            target
            for item in all_strict_results
            if item.get("fixed_pi") == selected_pi
            for target in cast(list[dict[str, object]], item.get("target_rows", []))
        ],
        "fit_scope": "development_history_only",
        "fit_max_year": max(int(row[FISCAL_YEAR]) for row in rows),
        "outer_outcomes_accessed": False,
    }


def _fit_l3(
    *,
    rows: list[dict[str, Any]],
    source_channels: dict[str, str],
    accuracy_priors: dict[str, dict[str, float]],
    fixed_pi: float,
    mcmc: dict[str, Any],
    rng: np.random.Generator,
) -> LatentClassResult:
    return fit_fixed_pi_latent_class(
        rows=rows,
        source_channels=source_channels,
        accuracy_priors=accuracy_priors,
        fixed_pi=fixed_pi,
        chains=int(mcmc["chains"]),
        warmup=int(mcmc["warmup_per_chain"]),
        draws=int(mcmc["draws_per_chain"]),
        alpha_step=float(mcmc["alpha_proposal_sd"]),
        random_effect_step=float(mcmc["random_effect_proposal_sd"]),
        rhat_maximum=float(mcmc["rhat_max"]),
        ess_minimum=float(mcmc["ess_min"]),
        ppc_rate_error_maximum=float(mcmc["posterior_predictive_source_rate_error_max"]),
        minimum_observations_per_source=int(mcmc["minimum_observations_per_source"]),
        rng=rng,
    )


def _source_outcomes(row: dict[str, Any]) -> dict[str, bool | None]:
    raw = row.get("source_outcomes")
    if not isinstance(raw, dict):
        raise ValueError("L3 matrix row requires source_outcomes")
    return cast(dict[str, bool | None], raw)


def _channel_outcomes(row: dict[str, Any]) -> dict[str, bool | None]:
    raw = row.get("channel_outcomes")
    if not isinstance(raw, dict):
        raise ValueError("L3 matrix row requires channel_outcomes")
    return cast(dict[str, bool | None], raw)


def _child_rng(rng: np.random.Generator) -> np.random.Generator:
    return np.random.default_rng(int(rng.integers(0, np.iinfo(np.uint32).max)))


def _pi_key(value: float) -> str:
    return format(value, ".17g")
