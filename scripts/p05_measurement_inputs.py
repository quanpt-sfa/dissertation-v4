"""P05 CLI: construct L0/L1 inputs, source matrices, and L3 capability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.evidence_registry import logical_evidence_sources
from core.pipeline import load_run, mapping, physical_columns, sequence
from core.rng import generator
from core.semantic_keys import FISCAL_YEAR, MATURE
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
    (
        expected_sources,
        anchor_sources,
        source_profiles,
        source_temporal_roles,
        explicit_negative_allowed,
    ) = _evidence_sources(loaded.registry)
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
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    candidate_registry = mapping(
        measurement.get("candidate_targets"), "measurement.candidate_targets"
    )
    candidate_targets = {
        str(target_id): [
            str(value)
            for value in sequence(
                mapping(raw, f"measurement.candidate_targets.{target_id}").get("sources"),
                f"measurement.candidate_targets.{target_id}.sources",
            )
        ]
        for target_id, raw in candidate_registry.items()
    }
    primary_measurement_sources = _primary_measurement_sources(measurement)
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
        source_temporal_roles=source_temporal_roles,
        explicit_negative_allowed=explicit_negative_allowed,
        l2_scoring=mapping(
            measurement.get("l2_scoring"),
            "measurement.l2_scoring",
        ),
        candidate_targets=candidate_targets,
        primary_measurement_sources=primary_measurement_sources,
    )
    primary_expected_sources = {
        source_id: expected_sources[source_id] for source_id in primary_measurement_sources
    }
    primary_source_profiles = {
        source_id: source_profiles[source_id] for source_id in primary_measurement_sources
    }
    _execute_l3_pilot(
        matrices=result.matrices,
        capability=result.l3_capability,
        source_channels=primary_expected_sources,
        source_profiles=primary_source_profiles,
        measurement=measurement,
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
        target_maturity=result.target_maturity,
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


def _primary_measurement_sources(measurement: dict[str, object]) -> list[str]:
    source_set_id = measurement.get("primary_source_set_id")
    if not isinstance(source_set_id, str) or not source_set_id:
        raise ValueError("measurement.primary_source_set_id must be locked")
    source_sets = mapping(measurement.get("source_sets"), "measurement.source_sets")
    source_set = mapping(source_sets.get(source_set_id), f"measurement.source_sets.{source_set_id}")
    if source_set.get("role") != "primary_measurement":
        raise ValueError("primary source set must have role=primary_measurement")
    sources = [
        str(value)
        for value in sequence(
            source_set.get("sources"), f"measurement.source_sets.{source_set_id}.sources"
        )
    ]
    if not sources:
        raise ValueError("primary measurement source set must not be empty")
    s3_sources = sorted(source for source in sources if source.startswith("S3_"))
    if s3_sources:
        raise ValueError(
            "primary annual measurement source set cannot contain next-calendar-year S3 endpoints"
        )
    return sources


def _evidence_sources(
    registry: dict[str, object],
) -> tuple[dict[str, str], list[str], dict[str, str], dict[str, str], dict[str, bool]]:
    sources = logical_evidence_sources(registry)
    result: dict[str, str] = {}
    anchors: list[str] = []
    profiles: dict[str, str] = {}
    temporal_roles: dict[str, str] = {}
    negative_policy: dict[str, bool] = {}
    for source_id, source in sources.items():
        result[source_id] = source.channel_id
        profiles[source_id] = source.profile_id
        temporal_roles[source_id] = source.temporal_role
        negative_policy[source_id] = source.explicit_negative_allowed
        if source.verification_status == "high_confirmation":
            anchors.append(source_id)
    if not result:
        raise ValueError("P05 requires registered evidence sources")
    return result, anchors, profiles, temporal_roles, negative_policy


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
                "missing_prior_profiles": missing_profiles,
            }
        )
        return
    minimum_channels = int(
        mapping(measurement.get("l2_missingness"), "measurement.l2_missingness")[
            "minimum_observed_channels"
        ]
    )
    rows = [
        row
        for row in sequence(matrices.get("rows"), "source-channel matrix rows")
        if int(mapping(row, "source-channel matrix row")[FISCAL_YEAR]) < initial_outer_year
        and mapping(row, "source-channel matrix row").get(MATURE) is True
    ]
    if not rows:
        capability.update(
            {
                "status": "EMPIRICALLY_PENDING",
                "pilot_executed": False,
                "reason_code": "L3_DEVELOPMENT_ROWS_UNAVAILABLE",
            }
        )
        return
    max_rows = int(operational.get("pilot_max_rows", len(rows)))
    if len(rows) > max_rows:
        selected = rng.choice(len(rows), size=max_rows, replace=False)
        rows = [rows[int(index)] for index in sorted(selected)]
    source_outcomes = {
        str(source_id): [
            mapping(row, "source-channel matrix row").get("source_outcomes", {}).get(source_id)
            for row in rows
        ]
        for source_id in source_channels
    }
    channel_count = len(set(source_channels.values()))
    if not l3_channel_capability_allows_pilot(
        {
            "status": capability.get("status"),
            "channel_count": channel_count,
            "minimum_channels": minimum_channels,
        }
    ):
        return
    mcmc = mapping(operational.get("mcmc"), "measurement.l3_model.operational.mcmc")
    fits = []
    for fixed_pi in [float(value) for value in grid]:
        fit = fit_fixed_pi_latent_class(
            source_outcomes=source_outcomes,
            source_channels=source_channels,
            accuracy_priors_by_source={
                source_id: mapping(
                    priors_by_profile[source_profiles[source_id]],
                    f"L3 prior profile={source_profiles[source_id]}",
                )
                for source_id in source_channels
            },
            fixed_pi=fixed_pi,
            chains=int(mcmc["chains"]),
            warmup_per_chain=int(mcmc["warmup_per_chain"]),
            draws_per_chain=int(mcmc["draws_per_chain"]),
            alpha_proposal_sd=float(mcmc["alpha_proposal_sd"]),
            random_effect_proposal_sd=float(mcmc["random_effect_proposal_sd"]),
            rng=rng,
        )
        fits.append(fit)
        attach_l3_pilot_posterior(rows, fit, fixed_pi=fixed_pi)
    finalize_l3_pilot_posteriors(
        capability=capability,
        fits=fits,
        robust_fraction=robust_fraction,
        rhat_maximum=float(mcmc["rhat_max"]),
        ess_minimum=float(mcmc["ess_min"]),
        source_rate_error_maximum=float(mcmc["posterior_predictive_source_rate_error_max"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
