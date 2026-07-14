"""P10 CLI: fold-specific development-only measurement selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from core.rng import generator
from core.semantic_keys import CHANNEL_ID, OUTER_FOLD
from selection.service import fit_l3_fold_candidate, select_measurement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--outer-fold", required=True)
    args = parser.parse_args()
    coordinates = {OUTER_FOLD: args.outer_fold}
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P10",
        state="SPLIT",
        coordinates=coordinates,
    )
    matrices = mapping(loaded.context.read("source_channel_matrices", {}), "source matrices")
    capability = mapping(loaded.context.read("l3_pilot_capability", {}), "L3 capability")
    loaded.context.read("temporal_split_registry", {})
    loaded.context.read("channel_time_split_registry", {})
    loaded.context.read("fold_aware_weights", {"fold_id": args.outer_fold})
    diagnostics = mapping(
        loaded.context.read("weight_diagnostics", {"fold_id": args.outer_fold}),
        "weight diagnostics",
    )
    if diagnostics.get("fit_scope") != "development_history":
        raise ValueError("P10 requires development-history weight diagnostics")
    mcse = mapping(loaded.context.read("mcse_report", {}), "MCSE report")
    if mcse.get("status") != "PASS" or mcse.get("precision_target_met") is not True:
        raise RuntimeError("P10 requires a PASS P08 simulation/MCSE report")
    loaded.context.read("fold_eligibility", {})
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    raw_candidates = sequence(
        measurement.get("selection_candidates"), "measurement.selection_candidates"
    )
    candidates = [str(value) for value in raw_candidates]
    missingness = mapping(measurement.get("l2_missingness"), "measurement.l2_missingness")
    minimum_channels = missingness.get("minimum_observed_channels")
    if minimum_channels is not None and (
        not isinstance(minimum_channels, int) or minimum_channels < 1
    ):
        raise ValueError("measurement.l2_missingness.minimum_observed_channels must be positive")
    l3_fold_result: dict[str, Any] | None = None
    if (
        "L3_fixed_pi" in candidates
        and capability.get("status") == "AVAILABLE"
        and capability.get("pilot_executed") is True
        and isinstance(minimum_channels, int)
    ):
        l3_model = mapping(measurement.get("l3_model"), "measurement.l3_model")
        operational = mapping(l3_model.get("operational"), "measurement.l3_model.operational")
        fixed_pi_grid = [
            float(value) for value in sequence(operational.get("fixed_pi_grid"), "L3 fixed-pi grid")
        ]
        source_channels, accuracy_priors = _l3_bindings(
            registry=loaded.registry,
            priors_by_profile=mapping(
                operational.get("accuracy_priors_by_profile"),
                "measurement.l3_model.operational.accuracy_priors_by_profile",
            ),
        )
        l3_fold_result = fit_l3_fold_candidate(
            matrices=matrices,
            outer_year=int(args.outer_fold),
            source_channels=source_channels,
            accuracy_priors=accuracy_priors,
            fixed_pi_grid=fixed_pi_grid,
            mcmc=mapping(operational.get("mcmc"), "L3 MCMC controls"),
            minimum_observed_channels=minimum_channels,
            robust_fraction=float(
                mapping(
                    mapping(loaded.registry.get("evaluation"), "evaluation").get("gate_common"),
                    "evaluation.gate_common",
                )["robust_scenario_fraction_min"]
            ),
            rng=generator(
                loaded.protocol_hash,
                "P10",
                coordinates,
                "l3_fold_local_channel_selection",
            ),
        )
    result = select_measurement(
        matrices=matrices,
        outer_year=int(args.outer_fold),
        candidates=candidates,
        l3_capability=capability,
        l3_fold_result=l3_fold_result,
        minimum_observed_channels=minimum_channels,
    )
    loaded.context.write("measurement_candidate_results", result.candidates, coordinates)
    loaded.context.write("measurement_selection_registry", result.selection, coordinates)
    loaded.context.write("channel_measurement_selection", result.channel_selection, coordinates)
    print(
        f"P10 status=PASS fold={args.outer_fold} selected={result.selection['selected_measurement']}"
    )
    return 0


def _l3_bindings(
    *,
    registry: dict[str, object],
    priors_by_profile: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    data_sources = mapping(registry.get("data_sources"), "data_sources")
    source_registry = mapping(data_sources.get("source_registry"), "source_registry")
    sources = mapping(source_registry.get("sources"), "source_registry.sources")
    source_channels: dict[str, str] = {}
    accuracy_priors: dict[str, dict[str, float]] = {}
    for source_id, raw_source in sources.items():
        source = mapping(raw_source, f"source={source_id}")
        if source.get("role") != "evidence" or source.get("enabled") is not True:
            continue
        channel = source.get(CHANNEL_ID)
        profile_id = source.get("profile_id")
        if not isinstance(channel, str) or not isinstance(profile_id, str):
            raise ValueError(f"source={source_id}: L3 channel and profile bindings required")
        raw_prior: object = priors_by_profile.get(profile_id)
        prior = mapping(raw_prior, f"L3 accuracy prior profile={profile_id}")
        source_channels[source_id] = channel
        parsed_prior: dict[str, float] = {}
        for key, value in prior.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"profile={profile_id}: L3 prior values must be numeric")
            parsed_prior[str(key)] = float(value)
        accuracy_priors[source_id] = parsed_prior
    if not source_channels:
        raise ValueError("P10 L3 selection requires registered evidence sources")
    return source_channels, accuracy_priors


if __name__ == "__main__":
    raise SystemExit(main())
