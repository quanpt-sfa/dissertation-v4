"""P10 CLI: build fold-specific sequential measurement-track plans."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.evidence_registry import logical_evidence_sources
from core.fold_control import production_path_blockers, require_primary_target
from core.pipeline import load_run, mapping, physical_columns, sequence
from core.rng import derive_seed, generator
from core.semantic_keys import MEASUREMENT_ID, OUTER_FOLD
from selection.nested_refit import run_nested_channel_refit
from selection.service import fit_l3_fold_candidate, select_measurement
from selection.track_plan import build_sequential_track_plan


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
    measurement = mapping(loaded.registry.get("measurement"), "measurement")
    primary_target_id = require_primary_target(measurement, "P10")
    matrices = mapping(loaded.context.read("source_channel_matrices", {}), "source matrices")
    capability = mapping(loaded.context.read("l3_pilot_capability", {}), "L3 capability")
    loaded.context.read("temporal_split_registry", {})
    loaded.context.read("channel_time_split_registry", {})
    weights = loaded.context.read("fold_aware_weights", {"fold_id": args.outer_fold})
    feature_panel = loaded.context.read("feature_panel", {})
    label_inputs = loaded.context.read("l0_l1_inputs", {})
    diagnostics = mapping(
        loaded.context.read("weight_diagnostics", {"fold_id": args.outer_fold}),
        "weight diagnostics",
    )
    if diagnostics.get("fit_scope") != "development_history":
        raise ValueError("P10 requires development-history weight diagnostics")
    mcse = mapping(loaded.context.read("mcse_report", {}), "MCSE report")
    fold_eligibility = loaded.context.read("fold_eligibility", {})
    feature_registry = loaded.context.read("feature_registry", {})
    if not all(isinstance(value, pd.DataFrame) for value in (weights, feature_panel, label_inputs)):
        raise ValueError("P10 nested-refit inputs must be DataFrames")
    feature_registry_rows = [
        mapping(item, "feature registry item")
        for item in sequence(feature_registry, "feature registry")
    ]
    blockers = production_path_blockers(
        fold_eligibility=fold_eligibility,
        outer_fold=args.outer_fold,
        feature_registry=feature_registry,
        mcse_report=mcse,
        target_id=primary_target_id,
    )
    if blockers:
        raise RuntimeError(
            f"P10_PRODUCTION_PATH_BLOCKED: fold={args.outer_fold} blockers={','.join(blockers)}"
        )

    candidates = [
        str(value)
        for value in sequence(
            measurement.get("selection_candidates"), "measurement.selection_candidates"
        )
    ]
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
            allowed_source_ids=_primary_measurement_sources(measurement),
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

    selection_config = mapping(measurement.get("selection"), "measurement.selection")
    if selection_config.get("nested_refit_required") is not True:
        raise RuntimeError("P10_NESTED_REFIT_NOT_LOCKED")
    learners = mapping(loaded.registry.get("learners"), "learners")
    tuning = mapping(learners.get("tuning"), "learners.tuning")
    gate1_learner_id = str(selection_config["gate1_learner_id"])
    gate1_feature_group = str(selection_config["gate1_feature_group"])
    nested = run_nested_channel_refit(
        matrices=matrices,
        feature_panel=cast(pd.DataFrame, feature_panel),
        feature_registry=feature_registry_rows,
        label_inputs=cast(pd.DataFrame, label_inputs),
        weights=cast(pd.DataFrame, weights),
        outer_year=int(args.outer_fold),
        candidates=candidates,
        l3_fold_result=l3_fold_result,
        minimum_observed_channels=minimum_channels if isinstance(minimum_channels, int) else 1,
        gate1_learner_id=gate1_learner_id,
        gate1_feature_group=gate1_feature_group,
        learner_settings=mapping(learners.get("settings"), "learners.settings"),
        learner_search_spaces=mapping(tuning.get("search_spaces"), "learners.tuning.search_spaces"),
        maximum_valid_configurations=int(tuning["max_valid_configurations_per_learner_inner_fold"]),
        columns=physical_columns(loaded.registry),
        random_state=derive_seed(
            loaded.protocol_hash,
            "P10",
            {OUTER_FOLD: args.outer_fold},
            "nested_channel_refit",
        )
        % (2**32 - 1),
    )

    result = select_measurement(
        matrices=matrices,
        outer_year=int(args.outer_fold),
        candidates=candidates,
        l3_capability=capability,
        l3_fold_result=l3_fold_result,
        minimum_observed_channels=minimum_channels,
        nested_candidate_results=cast(list[dict[str, Any]], nested.candidate_results),
    )
    execution = mapping(measurement.get("execution_tracks"), "measurement.execution_tracks")
    order = [str(value) for value in sequence(execution.get("order"), "execution_tracks.order")]
    track_ids = {
        str(key): str(value)
        for key, value in mapping(execution.get("track_ids"), "execution_tracks.track_ids").items()
    }
    plan = build_sequential_track_plan(
        primary_target_id=primary_target_id,
        outer_fold=args.outer_fold,
        candidate_results=result.candidates,
        execution_order=order,
        track_ids=track_ids,
    )
    selection_registry = {
        **plan,
        "selected_measurement": primary_target_id,
        "optional_candidate_selected_for_diagnostics": result.selection.get("selected_measurement"),
        "optional_selection_scope": result.selection.get("selection_scope"),
    }
    nested_cells_raw = nested.receipt.get("cell_results")
    if not isinstance(nested_cells_raw, list):
        raise RuntimeError("P10_NESTED_REFIT_CELL_RESULTS_MISSING")
    nested_cells = cast(list[object], nested_cells_raw)
    proxy_raw = result.channel_selection.get("strict_channel_results", [])
    proxy_channel_diagnostics = cast(list[object], proxy_raw) if isinstance(proxy_raw, list) else []
    channel_selection: dict[str, object] = {
        str(key): cast(object, value) for key, value in result.channel_selection.items()
    }
    channel_selection["proxy_only_channel_diagnostics"] = proxy_channel_diagnostics
    channel_selection["strict_channel_results"] = nested_cells
    channel_selection["nested_refit_receipt"] = nested.receipt
    if l3_fold_result is not None and l3_fold_result.get("status") == "PASS":
        channel_selection.update(
            {
                "l3_target_rows": l3_fold_result.get("target_rows", []),
                "l3_selected_fixed_pi": l3_fold_result.get("selected_fixed_pi"),
                "l3_selection_objective": l3_fold_result.get("selection_objective"),
                "l3_full_fit_diagnostics": l3_fold_result.get("full_fit_diagnostics"),
                "l3_source_accuracy": l3_fold_result.get("source_accuracy"),
                "l3_channel_random_effect_sd": l3_fold_result.get("channel_random_effect_sd"),
            }
        )

    loaded.context.write("measurement_candidate_results", result.candidates, coordinates)
    loaded.context.write("measurement_selection_registry", selection_registry, coordinates)
    loaded.context.write("channel_measurement_selection", channel_selection, coordinates)
    statuses = ",".join(f"{item[MEASUREMENT_ID]}={item['status']}" for item in plan["tracks"])
    print(f"P10 status=PASS fold={args.outer_fold} tracks={statuses}")
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
        raise ValueError("primary L3 source set must not be empty")
    s3_sources = sorted(source for source in sources if source.startswith("S3_"))
    if s3_sources:
        raise ValueError(
            "primary annual L3 source set cannot contain next-calendar-year S3 enforcement endpoints"
        )
    return sources


def _l3_bindings(
    *,
    registry: dict[str, object],
    priors_by_profile: dict[str, Any],
    allowed_source_ids: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Bind L3 to logical endpoints, not physical snapshot file identifiers."""
    sources = logical_evidence_sources(registry)
    source_channels: dict[str, str] = {}
    accuracy_priors: dict[str, dict[str, float]] = {}
    allowed = set(sources) if allowed_source_ids is None else set(allowed_source_ids)
    unknown = sorted(allowed - set(sources))
    if unknown:
        raise ValueError(f"primary L3 source set contains unknown sources {unknown}")
    for logical_source_id, source in sorted(sources.items()):
        if logical_source_id not in allowed:
            continue
        raw_prior: object = priors_by_profile.get(source.profile_id)
        prior = mapping(raw_prior, f"L3 accuracy prior profile={source.profile_id}")
        parsed_prior: dict[str, float] = {}
        for key, value in prior.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"profile={source.profile_id}: L3 prior values must be numeric")
            parsed_prior[str(key)] = float(value)
        source_channels[logical_source_id] = source.channel_id
        accuracy_priors[logical_source_id] = parsed_prior
    if not source_channels:
        raise ValueError("P10 L3 selection requires registered logical evidence sources")
    return source_channels, accuracy_priors


if __name__ == "__main__":
    raise SystemExit(main())
