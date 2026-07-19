"""Execute every P0-registered L3 scenario without performance-based selection."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from core.l3_scenarios import L3ScenarioRegistry, L3ScenarioSpec
from core.semantic_keys import ELIGIBLE, FIRM_ID, FISCAL_YEAR, MATURE, TARGET_VALUE
from labels.latent_class import LatentClassResult, fit_fixed_pi_latent_class


def fit_l3_preregistered_scenarios(
    *,
    matrices: dict[str, Any],
    outer_year: int,
    source_channels: dict[str, str],
    source_profiles: dict[str, str],
    scenario_registry: L3ScenarioRegistry,
    mcmc: dict[str, Any],
    minimum_observed_channels: int,
    robust_fraction: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run all registered scenarios and use only the predeclared primary scenario."""
    if minimum_observed_channels < 1:
        raise ValueError("L3 execution requires positive minimum channel coverage")
    if not 0 < robust_fraction <= 1:
        raise ValueError("L3 robust scenario fraction must be in (0, 1]")
    if set(source_channels) != set(source_profiles):
        raise ValueError("L3 source channel/profile bindings must cover the same sources")

    raw_rows = matrices.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("L3 execution requires matrix rows")
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
        return _unavailable(
            reason_code="NO_MATURE_DEVELOPMENT_ROWS_FOR_L3",
            scenario_registry=scenario_registry,
            outer_year=outer_year,
        )

    expected_channels = sorted(set(source_channels.values()))
    scenario_results: list[dict[str, Any]] = []
    strict_results: list[dict[str, Any]] = []
    full_fits: dict[str, LatentClassResult] = {}
    target_rows_by_scenario: dict[str, list[dict[str, Any]]] = {}

    for scenario in scenario_registry.scenarios:
        source_priors = _source_priors(scenario, source_profiles)
        heldout_results = _strict_channel_diagnostics(
            scenario=scenario,
            rows=rows,
            expected_channels=expected_channels,
            source_channels=source_channels,
            source_priors=source_priors,
            mcmc=mcmc,
            minimum_observed_channels=minimum_observed_channels,
            rng=rng,
        )
        strict_results.extend(heldout_results)
        complete = bool(heldout_results) and all(
            item["diagnostics_eligible"] is True
            and item["soft_cross_entropy"] is not None
            for item in heldout_results
        )
        losses = [
            float(item["soft_cross_entropy"])
            for item in heldout_results
            if item["soft_cross_entropy"] is not None
        ]
        scenario_result: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "role": scenario.role,
            "fixed_pi": scenario.fixed_pi,
            "prior_set_id": scenario.prior_set_id,
            ELIGIBLE: complete,
            "diagnostic_objective": float(np.mean(losses)) if complete and losses else None,
            "diagnostic_objective_role": "capability_and_reporting_only",
            "heldout_channel_count": len(heldout_results),
            "performance_based_selection_forbidden": True,
        }
        if not complete:
            scenario_result["reason_code"] = "L3_STRICT_CHANNEL_DIAGNOSTICS_FAILED"
            scenario_results.append(scenario_result)
            continue

        try:
            full_fit = _fit_l3(
                rows=[{"source_outcomes": _source_outcomes(row)} for row in rows],
                source_channels=source_channels,
                accuracy_priors=source_priors,
                fixed_pi=scenario.fixed_pi,
                mcmc=mcmc,
                rng=_child_rng(rng),
            )
        except ValueError as error:
            scenario_result[ELIGIBLE] = False
            scenario_result["reason_code"] = f"L3_FULL_REFIT_FAILED:{error}"
            scenario_results.append(scenario_result)
            continue

        scenario_result["full_fit_diagnostics"] = full_fit.diagnostics
        scenario_result["source_accuracy"] = full_fit.source_accuracy
        scenario_result["channel_random_effect_sd"] = full_fit.channel_random_effect_sd
        scenario_result["posterior_parameter_draws"] = full_fit.parameter_draws
        if full_fit.diagnostics.get("eligible_for_gate1") is not True:
            scenario_result[ELIGIBLE] = False
            scenario_result["reason_code"] = "L3_FULL_REFIT_DIAGNOSTICS_FAILED"
            scenario_results.append(scenario_result)
            continue

        full_fits[scenario.scenario_id] = full_fit
        target_rows_by_scenario[scenario.scenario_id] = [
            {
                FIRM_ID: str(row[FIRM_ID]),
                FISCAL_YEAR: int(row[FISCAL_YEAR]),
                TARGET_VALUE: float(probability),
                "l3_scenario_id": scenario.scenario_id,
            }
            for row, probability in zip(rows, full_fit.posterior_mean, strict=True)
            if int(row.get("observed_channel_count", 0)) >= minimum_observed_channels
        ]
        scenario_results.append(scenario_result)

    eligible_scenarios = [item for item in scenario_results if item[ELIGIBLE] is True]
    eligible_fraction = len(eligible_scenarios) / len(scenario_results)
    primary_result = next(
        item
        for item in scenario_results
        if item["scenario_id"] == scenario_registry.primary_scenario_id
    )
    primary_available = (
        primary_result[ELIGIBLE] is True
        and scenario_registry.primary_scenario_id in full_fits
    )
    if not primary_available or eligible_fraction < robust_fraction:
        reason = (
            "L3_PRIMARY_SCENARIO_DIAGNOSTICS_FAILED"
            if not primary_available
            else "L3_REGISTERED_SCENARIO_ROBUSTNESS_FAILED"
        )
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason_code": reason,
            "selection_objective": None,
            "primary_scenario_id": scenario_registry.primary_scenario_id,
            "scenario_results": scenario_results,
            "strict_channel_results": strict_results,
            "eligible_scenario_fraction": eligible_fraction,
            "required_robust_fraction": robust_fraction,
            "target_rows": [],
            "target_rows_by_scenario": target_rows_by_scenario,
            "fit_scope": "development_history_only",
            "fit_max_year": max(int(row[FISCAL_YEAR]) for row in rows),
            "scenario_selection_rule": "PRE_REGISTERED_PRIMARY",
            "performance_based_scenario_selection": False,
            "outer_outcomes_accessed": False,
            "known_cases_accessed": False,
        }

    primary_fit = full_fits[scenario_registry.primary_scenario_id]
    primary_fixed_pi = float(primary_result["fixed_pi"])
    return {
        "status": "PASS",
        "reason_code": None,
        # Compatibility field for the existing optional-candidate diagnostic layer.
        # It is the pre-registered primary scenario loss, never an optimization result.
        "selection_objective": primary_result["diagnostic_objective"],
        "primary_scenario_id": scenario_registry.primary_scenario_id,
        "primary_fixed_pi": primary_fixed_pi,
        "selected_fixed_pi": primary_fixed_pi,
        "selected_fixed_pi_semantic": "compatibility_alias_for_preregistered_primary",
        "scenario_results": scenario_results,
        "strict_channel_results": strict_results,
        "eligible_scenario_fraction": eligible_fraction,
        "required_robust_fraction": robust_fraction,
        "full_fit_diagnostics": primary_fit.diagnostics,
        "source_accuracy": primary_fit.source_accuracy,
        "channel_random_effect_sd": primary_fit.channel_random_effect_sd,
        "target_rows": target_rows_by_scenario[scenario_registry.primary_scenario_id],
        "target_rows_by_scenario": target_rows_by_scenario,
        "fit_scope": "development_history_only",
        "fit_max_year": max(int(row[FISCAL_YEAR]) for row in rows),
        "scenario_selection_rule": "PRE_REGISTERED_PRIMARY",
        "performance_based_scenario_selection": False,
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
    }


def _strict_channel_diagnostics(
    *,
    scenario: L3ScenarioSpec,
    rows: list[dict[str, Any]],
    expected_channels: list[str],
    source_channels: dict[str, str],
    source_priors: dict[str, dict[str, float]],
    mcmc: dict[str, Any],
    minimum_observed_channels: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for heldout in expected_channels:
        remaining_sources = {
            source: channel
            for source, channel in source_channels.items()
            if channel != heldout
        }
        if not remaining_sources:
            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "fixed_pi": scenario.fixed_pi,
                    "heldout_channel": heldout,
                    "rows": 0,
                    "soft_cross_entropy": None,
                    "diagnostics_eligible": False,
                    "reason_code": "NO_SOURCES_AFTER_CHANNEL_HOLDOUT",
                    "heldout_removed_from_target_and_measurement": True,
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
                accuracy_priors={source: source_priors[source] for source in remaining_sources},
                fixed_pi=scenario.fixed_pi,
                mcmc=mcmc,
                rng=_child_rng(rng),
            )
        except ValueError as error:
            results.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "fixed_pi": scenario.fixed_pi,
                    "heldout_channel": heldout,
                    "rows": 0,
                    "soft_cross_entropy": None,
                    "diagnostics_eligible": False,
                    "reason_code": f"L3_FIT_FAILED:{error}",
                    "heldout_removed_from_target_and_measurement": True,
                }
            )
            continue

        losses: list[float] = []
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
            clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
            outcome = float(bool(heldout_value))
            losses.append(
                -(outcome * math.log(clipped) + (1.0 - outcome) * math.log(1.0 - clipped))
            )
        diagnostics_eligible = fit.diagnostics.get("eligible_for_gate1") is True
        results.append(
            {
                "scenario_id": scenario.scenario_id,
                "fixed_pi": scenario.fixed_pi,
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
            }
        )
    return results


def _source_priors(
    scenario: L3ScenarioSpec,
    source_profiles: dict[str, str],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for source_id, profile_id in source_profiles.items():
        prior = scenario.accuracy_priors_by_profile.get(profile_id)
        if prior is None:
            raise ValueError(
                f"scenario={scenario.scenario_id}: missing prior profile={profile_id}"
            )
        result[source_id] = dict(prior)
    return result


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


def _unavailable(
    *,
    reason_code: str,
    scenario_registry: L3ScenarioRegistry,
    outer_year: int,
) -> dict[str, Any]:
    return {
        "status": "INSUFFICIENT_EVIDENCE",
        "reason_code": reason_code,
        "selection_objective": None,
        "primary_scenario_id": scenario_registry.primary_scenario_id,
        "scenario_results": [],
        "strict_channel_results": [],
        "target_rows": [],
        "target_rows_by_scenario": {},
        "fit_scope": "development_history_only",
        "fit_max_year": None,
        "outer_fold": str(outer_year),
        "scenario_selection_rule": "PRE_REGISTERED_PRIMARY",
        "performance_based_scenario_selection": False,
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
    }
