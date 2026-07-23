"""P08A coordinator: lock scenarios, cost design, and one execution profile."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import FIRM_ID, FISCAL_YEAR, METHOD_ID, SCENARIO_ID
from simulation.coordinate_contract import (
    METHOD_KEY_PREFIX,
    SCENARIO_KEY_PREFIX,
    SHORT_KEY_HEX_LENGTH,
)
from simulation.empirical_calibration import attach_empirical_panel_calibration
from simulation.l3_variants import (
    L3_VARIANT_IDS,
    extend_method_registry,
)
from simulation.method_contract import (
    apply_execution_profile,
    build_cost_sensitive_contract,
    build_execution_profile_contract,
    build_full_method_registry,
    build_imbalance_treatment_contract,
    expected_counts,
    profile_expected_counts,
    validate_optional_dependencies,
)
from simulation.scenario_contract import validate_scenario_target_identity
from simulation.service import validate_scenarios


def build_short_key_map(
    ids: list[str],
    *,
    prefix: str,
    hex_length: int = SHORT_KEY_HEX_LENGTH,
) -> dict[str, str]:
    """Build a mapping from canonical IDs to short hash keys with a collision guard."""
    result: dict[str, str] = {}
    reverse: dict[str, str] = {}

    for canonical_id in sorted(ids):
        digest = hashlib.sha256(canonical_id.encode("utf-8")).hexdigest()[:hex_length]

        key = f"{prefix}{digest}"

        existing = reverse.get(key)
        if existing is not None and existing != canonical_id:
            raise ValueError(
                f"short-key collision: {existing} and {canonical_id} both map to {key}"
            )

        result[canonical_id] = key
        reverse[key] = canonical_id

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--profile",
        help=(
            "Optional assertion only. It must equal simulation.active_profile; "
            "the CLI cannot override the protocol-locked profile."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P08",
        state="FEATURED",
    )
    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    cost_contract = build_cost_sensitive_contract(simulation)
    imbalance_contract = build_imbalance_treatment_contract(simulation)
    full_registry = build_full_method_registry(
        simulation,
        cost_contract,
        imbalance_contract,
    )
    full_registry = extend_method_registry(simulation, full_registry)
    profile_contract = build_execution_profile_contract(
        simulation,
        full_registry,
    )
    active_profile = str(profile_contract["active_profile"])
    if args.profile is not None and args.profile != active_profile:
        raise ValueError(
            f"--profile={args.profile} differs from protocol-locked "
            f"simulation.active_profile={active_profile}"
        )
    method_registry = apply_execution_profile(
        full_registry,
        profile_contract,
    )
    required_dependencies = validate_optional_dependencies(
        method_registry,
        active_only=True,
    )
    full_counts = expected_counts(
        cost_contract,
        imbalance_contract,
    )
    full_counts = dict(full_counts)
    full_counts["standalone"] += len(L3_VARIANT_IDS)
    full_counts["method_total"] += len(L3_VARIANT_IDS)
    active_counts = profile_expected_counts(
        method_registry,
        active_only=True,
    )

    raw_scenarios = [
        mapping(item, "simulation operational scenario")
        for item in sequence(
            simulation.get("operational_scenarios"),
            "simulation.operational_scenarios",
        )
    ]
    firm_year_panel = loaded.context.read("firm_year_panel", {})
    feature_panel = loaded.context.read("feature_panel", {})
    source_channel_matrices = loaded.context.read(
        "source_channel_matrices",
        {},
    )
    feature_registry = [
        mapping(item, "feature registry item")
        for item in sequence(
            loaded.context.read("feature_registry", {}),
            "feature registry",
        )
    ]
    if not isinstance(firm_year_panel, pd.DataFrame):
        raise ValueError("P08 P02 firm-year panel must be a DataFrame")
    if not isinstance(feature_panel, pd.DataFrame):
        raise ValueError("P08 feature panel must be a DataFrame")
    if not isinstance(source_channel_matrices, dict):
        raise ValueError("P08 P05 source-channel matrices must be a JSON object")

    folds = mapping(loaded.registry.get("folds"), "folds")
    empirical_settings = mapping(
        simulation.get("empirical_calibration"),
        "simulation.empirical_calibration",
    )
    l3_variant_settings = mapping(
        simulation.get("l3_variant_settings"),
        "simulation.l3_variant_settings",
    )
    columns = physical_columns(loaded.registry)
    scenarios = attach_empirical_panel_calibration(
        scenarios=raw_scenarios,
        firm_year_panel=firm_year_panel,
        feature_panel=feature_panel,
        feature_registry=feature_registry,
        source_channel_matrices=source_channel_matrices,
        firm_column=columns[FIRM_ID],
        year_column=columns[FISCAL_YEAR],
        development_year_maximum=int(folds["initial_outer_year"]) - 1,
        calibration_repeats=int(empirical_settings.get("calibration_repeats", 64)),
    )
    scenarios = validate_scenarios(scenarios)

    scenario_ids = [str(s[SCENARIO_ID]) for s in scenarios]
    scenario_keys = build_short_key_map(
        scenario_ids,
        prefix=SCENARIO_KEY_PREFIX,
        hex_length=SHORT_KEY_HEX_LENGTH,
    )

    method_ids = [str(m[METHOD_ID]) for m in method_registry]
    method_keys = build_short_key_map(
        method_ids,
        prefix=METHOD_KEY_PREFIX,
        hex_length=SHORT_KEY_HEX_LENGTH,
    )

    locked: list[dict[str, object]] = []
    for scenario in scenarios:
        item = dict(scenario)
        validate_scenario_target_identity(item)
        scenario_id = str(item[SCENARIO_ID])
        item["scenario_" + "key"] = scenario_keys[scenario_id]
        item["l3_variant_settings"] = dict(l3_variant_settings)

        updated_method_registry = []
        for value in method_registry:
            method_spec = dict(value)
            method_id = str(method_spec[METHOD_ID])
            method_spec["method_" + "key"] = method_keys[method_id]
            updated_method_registry.append(method_spec)
        item["method_registry"] = updated_method_registry
        item["active_method_ids"] = list(profile_contract["active_method_ids"])
        item["execution_profile"] = active_profile
        item["execution_profile_registry"] = [dict(value) for value in profile_contract["profiles"]]
        item["cost_regime_registry"] = [
            dict(value) for value in cost_contract["cost_regime_registry"]
        ]
        item["review_budget_registry"] = [
            dict(value) for value in cost_contract["review_budget_registry"]
        ]
        item["imbalance_treatment_registry"] = [
            dict(value) for value in imbalance_contract["treatment_registry"]
        ]
        item["imbalance_treatment_settings"] = {
            key: value for key, value in imbalance_contract.items() if key != "treatment_registry"
        }
        item["cost_sensitive_settings"] = {
            key: value
            for key, value in cost_contract.items()
            if key
            not in {
                "cost_regime_registry",
                "review_budget_registry",
            }
        }
        item["methodology_contract"] = {
            "comparison_design": ("profiled_blocked_label_strategy_by_learner_by_training_cost"),
            "active_profile": active_profile,
            "full_library_counts": full_counts,
            "active_profile_counts": active_counts,
            "full_method_library_retained": True,
            "inactive_methods_executable": False,
            "cost_sensitive_learning": True,
            "cost_sensitive_threshold_selection": True,
            "cost_sensitive_evaluation": True,
            "smote_adasyn_robustness_separate_from_pu_ipw": True,
            "paired_dgp_across_methods_required": True,
            "latent_truth_used_for_evaluation_not_training": True,
            "profile_selection_locked_by_config": True,
            "active_optional_dependencies": required_dependencies,
            "contract_version": 8,
            "sample_size_derived_from_P02": True,
            "observed_label_rate_derived_from_P05": True,
            "latent_prevalence_calibrated_to_empirical_rate": True,
            "exact_empirical_rows_no_bootstrap": True,
            "l3_correct_and_misspecified_variants": True,
        }
        locked.append(item)

    if args.dry_run:
        print(
            "P08A registry dry-run: "
            f"scenarios={len(locked)} profile={active_profile} "
            f"active_methods={active_counts['method_total']} "
            f"available_methods={full_counts['method_total']}"
        )
        return 0

    loaded.context.write("simulation_scenario_registry", locked, {})
    status = "PASS" if locked else "SKIPPED reason=NO_OPERATIONAL_SCENARIOS"
    print(
        f"P08A registry status={status} scenarios={len(locked)} "
        f"profile={active_profile} "
        f"active_methods={active_counts['method_total']} "
        f"available_methods={full_counts['method_total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
