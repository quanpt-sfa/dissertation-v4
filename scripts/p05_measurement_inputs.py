"""P05 CLI: construct L0/L1 inputs, source matrices, and L3 capability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.rng import generator
from core.semantic_keys import CHANNEL_ID, FISCAL_YEAR, MATURE
from labels.latent_class import (
    attach_l3_pilot_posterior,
    finalize_l3_pilot_posteriors,
    fit_fixed_pi_latent_class,
)
from measurement.service import (
    build_measurement_inputs,
    l3_channel_capability_allows_pilot,
    summarize_fold_eligibility,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P05",
        state="RISK_SET",
    )
    if args.dry_run:
        print("P05 dry-run: L0/L1 and missingness-preserving L2 inputs")
        return 0
    risk_sets = loaded.context.read("risk_sets", {})
    evidence = loaded.context.read("evidence_ledger", {})
    if not isinstance(risk_sets, pd.DataFrame) or not isinstance(evidence, pd.DataFrame):
        raise ValueError("P05 input artifacts must be DataFrames")
    expected_sources, anchor_sources, source_profiles = _evidence_sources(loaded.registry)
    study = mapping(loaded.registry.get("study"), "study")
    horizons = mapping(study.get("horizons_months"), "study.horizons_months")
    horizon = horizons.get("primary")
    if not isinstance(horizon, int):
        raise ValueError("study.horizons_months.primary must be an integer")
    vocabulary = mapping(loaded.registry.get("vocabulary"), "vocabulary")
    statuses = sequence(vocabulary.get("capability_statuses"), "capability_statuses")
    if "EMPIRICALLY_PENDING" not in statuses or "UNAVAILABLE_BY_DESIGN" not in statuses:
        raise ValueError("required capability statuses are not registered")
    reasons = sequence(vocabulary.get("reason_codes"), "reason_codes")
    if "INSUFFICIENT_CHANNELS" not in reasons:
        raise ValueError("INSUFFICIENT_CHANNELS reason is not registered")
    result = build_measurement_inputs(
        risk_sets=risk_sets,
        evidence=evidence,
        expected_sources=expected_sources,
        horizon_months=horizon,
        columns=physical_columns(loaded.registry),
        pending_status="EMPIRICALLY_PENDING",
        unavailable_status="UNAVAILABLE_BY_DESIGN",
        insufficient_channels_reason="INSUFFICIENT_CHANNELS",
        anchor_source_ids=anchor_sources,
        source_profiles=source_profiles,
        l2_scoring=mapping(
            mapping(loaded.registry.get("measurement"), "measurement").get("l2_scoring"),
            "measurement.l2_scoring",
        ),
    )
    _execute_l3_pilot(
        matrices=result.matrices,
        capability=result.l3_capability,
        source_channels=expected_sources,
        source_profiles=source_profiles,
        measurement=mapping(loaded.registry.get("measurement"), "measurement"),
        initial_outer_year=int(
            mapping(loaded.registry.get("folds"), "folds")["initial_outer_year"]
        ),
        robust_fraction=float(
            mapping(
                mapping(loaded.registry.get("evaluation"), "evaluation").get("gate_common"),
                "evaluation.gate_common",
            )["robust_scenario_fraction_min"]
        ),
        rng=generator(loaded.protocol_hash, "P05", {}, "l3_feasibility_pilot"),
    )
    if args.validate_only:
        return 0
    loaded.context.write("source_channel_matrices", result.matrices, {})
    loaded.context.write("l0_l1_inputs", result.inputs, {})
    loaded.context.write("l3_pilot_capability", result.l3_capability, {})
    loaded.context.write("measurement_variable_registry", result.measurement_variables, {})
    loaded.context.write("channel_capability", result.channel_capability, {})
    loaded.context.write("anchor_capability", result.anchor_capability, {})
    folds = mapping(loaded.registry.get("folds"), "folds")
    nested = [
        int(value)
        for value in sequence(
            folds.get("fully_nested_outer_years"), "folds.fully_nested_outer_years"
        )
    ]
    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")
    sensitivity_range = sequence(
        gate2.get("sensitivity_positive_count_range"),
        "evaluation.gate2.sensitivity_positive_count_range",
    )
    if len(sensitivity_range) != 2:
        raise ValueError("sensitivity positive-count range requires two values")
    fold_eligibility = summarize_fold_eligibility(
        sealed_outcomes=result.sealed_outcomes,
        risk_sets=risk_sets,
        initial_outer_year=int(folds["initial_outer_year"]),
        confirmatory_years=nested,
        prospective_year=int(folds["prospective_year"]),
        confirmatory_positive_minimum=int(gate2["confirmatory_positive_count_min"]),
        sensitivity_positive_range=(
            int(sensitivity_range[0]),
            int(sensitivity_range[1]),
        ),
        columns=physical_columns(loaded.registry),
    )
    loaded.context.write("fold_eligibility", fold_eligibility, {})
    loaded.context.write("sealed_outcome_store", result.sealed_outcomes, {})
    print(
        f"P05 status=PASS targets={len(result.inputs)} "
        f"sealed_outcomes={len(result.sealed_outcomes)}"
    )
    return 0


def _evidence_sources(
    registry: dict[str, object],
) -> tuple[dict[str, str], list[str], dict[str, str]]:
    data_sources = mapping(registry.get("data_sources"), "data_sources")
    source_registry = mapping(data_sources.get("source_registry"), "source_registry")
    sources = mapping(source_registry.get("sources"), "source_registry.sources")
    result: dict[str, str] = {}
    anchors: list[str] = []
    profiles: dict[str, str] = {}
    for source_id, raw in sources.items():
        source = mapping(raw, f"source={source_id}")
        if source.get("role") == "evidence" and source.get("enabled") is True:
            channel = source.get(CHANNEL_ID)
            if not isinstance(channel, str):
                raise ValueError(f"source={source_id}: channel_id required")
            result[source_id] = channel
            profile_id = source.get("profile_id")
            if not isinstance(profile_id, str):
                raise ValueError(f"source={source_id}: profile_id required for L3 prior binding")
            profiles[source_id] = profile_id
            if source.get("verification_status") == "high_confirmation":
                anchors.append(source_id)
    if not result:
        raise ValueError("P05 requires registered evidence sources")
    return result, anchors, profiles


def _execute_l3_pilot(
    *,
    matrices: dict[str, object],
    capability: dict[str, object],
    source_channels: dict[str, str],
    source_profiles: dict[str, str],
    measurement: dict[str, object],
    initial_outer_year: int,
    robust_fraction: float,
    rng: object,
) -> None:
    import numpy as np

    if not isinstance(rng, np.random.Generator):
        raise TypeError("L3 pilot RNG must be a numpy Generator")
    if not l3_channel_capability_allows_pilot(capability):
        return
    model = mapping(measurement.get("l3_model"), "measurement.l3_model")
    operational = mapping(model.get("operational"), "measurement.l3_model.operational")
    grid = sequence(operational.get("fixed_pi_grid"), "l3 fixed_pi_grid")
    priors_by_profile = mapping(
        operational.get("accuracy_priors_by_profile"), "L3 priors by profile"
    )
    if not grid:
        capability.update(
            {
                "status": "EMPIRICALLY_PENDING",
                "pilot_executed": False,
                "reason_code": "L3_FIXED_PI_GRID_NOT_LOCKED",
                "required_config_key": "measurement.l3_model.operational.fixed_pi_grid",
            }
        )
        return
    missing_profiles = sorted(set(source_profiles.values()) - set(priors_by_profile))
    if missing_profiles:
        capability.update(
            {
                "status": "EMPIRICALLY_PENDING",
                "pilot_executed": False,
                "reason_code": "L3_ACCURACY_PRIORS_NOT_LOCKED",
                "missing_profile_ids": missing_profiles,
                "required_config_key": "measurement.l3_model.operational.accuracy_priors_by_profile",
            }
        )
        return
    raw_rows = matrices.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("L3 pilot requires matrix rows")
    eligible_rows: list[dict[str, object]] = []
    for raw_value in cast(list[object], raw_rows):
        row = mapping(raw_value, "L3 matrix row")
        if (
            int(row.get(FISCAL_YEAR, initial_outer_year)) < initial_outer_year
            and row.get(MATURE) is True
        ):
            eligible_rows.append(row)
    maximum_rows = int(operational["pilot_max_rows"])
    eligible_rows = eligible_rows[:maximum_rows]
    if not eligible_rows:
        capability.update(
            {
                "status": "UNAVAILABLE_BY_DESIGN",
                "pilot_executed": False,
                "reason_code": "NO_MATURE_DEVELOPMENT_ROWS_FOR_L3_PILOT",
                "fit_scope": f"mature_development_years_before_{initial_outer_year}",
                "outer_outcomes_accessed": False,
            }
        )
        return
    mcmc = mapping(operational.get("mcmc"), "L3 MCMC controls")
    priors: dict[str, dict[str, float]] = {}
    for source_id, profile_id in source_profiles.items():
        raw_prior = mapping(priors_by_profile[profile_id], f"L3 prior profile={profile_id}")
        priors[source_id] = {str(key): float(value) for key, value in raw_prior.items()}
    scenario_results: list[dict[str, object]] = []
    for index, raw_pi in enumerate(grid):
        fixed_pi = float(raw_pi)
        fit = fit_fixed_pi_latent_class(
            rows=eligible_rows,
            source_channels=source_channels,
            accuracy_priors=priors,
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
            rng=np.random.default_rng(int(rng.integers(0, np.iinfo(np.uint32).max)) + index),
        )
        attach_l3_pilot_posterior(
            rows=eligible_rows,
            fixed_pi=fixed_pi,
            posterior_mean=fit.posterior_mean,
        )
        scenario_results.append(
            {
                "fixed_pi": fixed_pi,
                "diagnostics": fit.diagnostics,
                "source_accuracy": fit.source_accuracy,
                "channel_random_effect_sd": fit.channel_random_effect_sd,
                "posterior_summary": {
                    "mean_min": min(fit.posterior_mean),
                    "mean_max": max(fit.posterior_mean),
                    "retained_draws": len(fit.posterior_draws),
                },
            }
        )
    eligible_fixed_pi: list[float] = []
    for item in scenario_results:
        if mapping(item["diagnostics"], "L3 diagnostics").get("eligible_for_gate1") is not True:
            continue
        raw_fixed_pi = item["fixed_pi"]
        if not isinstance(raw_fixed_pi, (int, float)) or isinstance(raw_fixed_pi, bool):
            raise ValueError("L3 fixed-pi scenario must be numeric")
        eligible_fixed_pi.append(float(raw_fixed_pi))
    finalize_l3_pilot_posteriors(
        rows=eligible_rows,
        eligible_fixed_pi=eligible_fixed_pi,
    )
    eligible_count = sum(
        mapping(item["diagnostics"], "L3 diagnostics").get("eligible_for_gate1") is True
        for item in scenario_results
    )
    eligible_fraction = eligible_count / len(scenario_results)
    capability.update(
        {
            "status": "AVAILABLE"
            if eligible_fraction >= robust_fraction
            else "UNAVAILABLE_BY_DESIGN",
            "pilot_executed": True,
            "reason_code": None
            if eligible_fraction >= robust_fraction
            else "L3_DIAGNOSTICS_OR_ROBUSTNESS_FAILED",
            "fit_scope": f"mature_development_years_before_{initial_outer_year}",
            "outer_outcomes_accessed": False,
            "fixed_pi_scenarios": scenario_results,
            "eligible_scenario_fraction": eligible_fraction,
            "required_robust_fraction": robust_fraction,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
