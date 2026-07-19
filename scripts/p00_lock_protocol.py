"""Compile, validate, and atomically publish the P00 protocol lock.

P00 is the fail-closed boundary for the P08 empirical-calibration architecture.
Semi-synthetic P08 scenarios must derive their population from P02, observed-label
rate from P05, and covariates from P07. Researcher-specified sample size or latent
prevalence is forbidden for those scenarios.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence, Sized
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.atomic_io import publish_p00
from core.registry_compiler import (
    compile_registry,
    environment_observation,
    source_manifest,
)
from core.schema_registry import contract_registry
from core.semantic_keys import SCENARIO_ID, SCENARIO_KEY, METHOD_KEY, BATCH_KEY
from simulation.coordinate_contract import (
    SCENARIO_KEY_PREFIX,
    METHOD_KEY_PREFIX,
    SHORT_KEY_HEX_LENGTH,
)
from simulation.scenario_contract import validate_scenario_target_identity


P08_EMPIRICAL_ARCHITECTURE_VERSION = "p02-p05-p07-calibrated-replication-paired-v2"
P08_REQUIRED_READS = frozenset(
    {
        "firm_year_panel",
        "feature_panel",
        "feature_registry",
        "source_channel_matrices",
        "simulation_scenario_registry",
        "simulation_batches",
        "model_diagnostics",
    }
)
P08_REQUIRED_WRITES = frozenset(
    {
        "simulation_scenario_registry",
        "simulation_batches",
        "mcse_report",
        "model_diagnostics",
    }
)
P08_EXPECTED_PRODUCERS = {
    "firm_year_panel": "P02",
    "source_channel_matrices": "P05",
    "feature_panel": "P07",
    "feature_registry": "P07",
}
P08_FORBIDDEN_READS = frozenset(
    {
        "sealed_outcome_store",
        "prospective_set",
        "known_cases_seal",
        "known_case_results",
    }
)
SEMI_SYNTHETIC_TIER = "semi_synthetic_development_covariates"
FULLY_SYNTHETIC_TIER = "fully_synthetic"
SEMI_SYNTHETIC_FORBIDDEN_FIELDS = frozenset(
    {
        "sample_size",
        "prevalence",
        "fixed_pi",
        "sample_size_source",
        "prevalence_source",
        "calibrated_latent_prevalence",
        "semi_synthetic_covariate_pool",
        "semi_synthetic_feature_count",
        "semi_synthetic_feature_columns",
        "semi_synthetic_pool_rows",
        "semi_synthetic_development_year_maximum",
        "semi_synthetic_preprocessing",
        "empirical_panel_key_hash",
        "empirical_calibration_target_id",
        "empirical_panel_rows",
        "empirical_positive_count",
        "empirical_known_label_count",
        "empirical_mature_target_count",
        "empirical_observed_positive_rate",
        "empirical_positive_rate_known_labels",
        "empirical_known_label_rate",
        "empirical_target_maturity_rate",
        "ascertainment_probability_if_latent_positive",
        "ascertainment_probability_if_latent_negative",
        "ascertainment_calibration_mcse",
        "outer_rows_used_in_pool",
    }
)


def _validate_path_budgets(registry: dict[str, Any], output_root: Path | None, run_id: str) -> None:
    artifacts = registry.get("artifacts", {})
    reproducibility = registry.get("reproducibility", {})
    output_root_template = reproducibility.get("output_root_template", "artifacts/runs/{run_id}")
    run_root = output_root / run_id if output_root else Path(output_root_template.format(run_id=run_id))
    run_root = run_root.resolve()

    mock_values = {
        SCENARIO_KEY: SCENARIO_KEY_PREFIX + "x" * SHORT_KEY_HEX_LENGTH,
        METHOD_KEY: METHOD_KEY_PREFIX + "x" * SHORT_KEY_HEX_LENGTH,
        BATCH_KEY: "b0000",
        "source_" + "id": "firm_identity_master",
        "fold_" + "id": "fold_00",
        "outer_" + "fold": "fold_00",
        "channel_" + "id": "S1",
        "feature_" + "id": "fs_aud_accounts_receivable",
    }

    violations = []
    max_path_limit = 220

    for artifact_id, spec in artifacts.items():
        if not isinstance(spec, dict):
            continue
        path_template = spec.get("path_template")
        if not isinstance(path_template, str):
            continue
        coords = spec.get("coordinates", [])

        fill = {}
        for c in coords:
            c_str = str(c)
            fill[c_str] = mock_values.get(c_str, "coord_val")

        try:
            rendered_rel = Path(path_template.format(**fill))
            rendered_abs = run_root / rendered_rel

            length = len(str(rendered_abs))
            if length > max_path_limit:
                violations.append((artifact_id, str(rendered_abs), length))

            manifest_path = rendered_abs.with_name(rendered_abs.name + ".manifest.json")
            m_length = len(str(manifest_path))
            if m_length > max_path_limit:
                violations.append((artifact_id + " (manifest)", str(manifest_path), m_length))

            stage_path = run_root / ".t" / "s-123456" / rendered_abs.name
            s_length = len(str(stage_path))
            if s_length > max_path_limit:
                violations.append((artifact_id + " (staging)", str(stage_path), s_length))

        except Exception as exc:
            raise ValueError(
                f"artifact={artifact_id}: cannot render path for budget validation"
            ) from exc

    if violations:
        details = "\n".join(
            f"  - {art_id} ({l} chars): {p}"
            for art_id, p, l in violations
        )
        raise ValueError(
            f"P00 path budget exceeded (limit {max_path_limit} characters):\n{details}\n"
            "Please check config/foundation/artifacts.yaml or run_id length."
        )


def _relative(artifacts: dict[str, Any], artifact_id: str, stage: Path) -> str:
    return str(Path(artifacts[artifact_id]["path_template"]).relative_to(stage))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only and args.snapshot_manifest is None:
        raise ValueError("--snapshot-manifest is required for a published operational P00 lock")

    compiled = compile_registry(args.config, args.snapshot_manifest)
    registry = dict(compiled.registry)
    p08_architecture = _validate_p08_empirical_architecture(registry)
    _validate_path_budgets(registry, args.output_root, args.run_id)

    if args.validate_only:
        print(
            "P00 validate-only: PASS "
            f"p08_architecture={p08_architecture['architecture_version']} "
            f"profile={p08_architecture['active_profile']} "
            f"semi_synthetic_scenarios={p08_architecture['semi_synthetic_scenarios']}"
        )
        return 0

    artifacts = cast(dict[str, Any], registry["artifacts"])
    reproducibility = cast(dict[str, Any], registry["reproducibility"])
    vocabulary = cast(dict[str, Any], registry["vocabulary"])
    root = args.config.resolve().parent.parent

    if args.run_id.startswith("p0-final") and reproducibility.get(
        "require_clean_tree_for_acceptance"
    ):
        manifest_preview = source_manifest(compiled, root)
        if manifest_preview["git_dirty"]:
            raise RuntimeError("final acceptance run requires a clean committed working tree")

    run_root = (
        args.output_root / args.run_id
        if args.output_root
        else root / reproducibility["output_root_template"].format(run_id=args.run_id)
    )
    stage = Path(artifacts["registry_lock"]["path_template"]).parent
    target = run_root / stage
    contracts = contract_registry(registry)

    def p(artifact_id: str) -> str:
        return _relative(artifacts, artifact_id, stage)

    capabilities = cast(dict[str, Any], registry["capabilities"])
    files: dict[str, object] = {
        p("registry_lock"): registry,
        p("protocol_hash"): compiled.protocol_hash + "\n",
        p("source_config_manifest"): source_manifest(compiled, root),
        p("capability_seed"): {
            key: value["initial_status"] for key, value in capabilities.items()
        },
        p("decision_traceability"): registry["decision_traceability"],
        p("artifact_catalog"): artifacts,
        p("schema_catalog"): registry["schemas"],
        p("step_catalog"): registry["steps"],
        p("access_matrix"): registry["access_matrix"],
        p("known_cases_seal"): {
            "status": vocabulary["seal_states"]["sealed"],
            "protocol_hash": compiled.protocol_hash,
            "opens_at_step": registry["known_cases"]["opening_step"],
        },
        p("environment_expectation"): reproducibility["environment_expectations"],
        p("environment_observation"): environment_observation(root),
        p("job_manifest"): {
            "run_id": args.run_id,
            "protocol_hash": compiled.protocol_hash,
            "output_hashes": {},
        },
        p("p00_audit_report"): _report(
            args.run_id,
            compiled.protocol_hash,
            registry,
            p08_architecture,
        ),
    }
    names = {
        p(name): name
        for name in (
            "registry_lock",
            "protocol_hash",
            "source_config_manifest",
            "capability_seed",
            "decision_traceability",
            "artifact_catalog",
            "schema_catalog",
            "step_catalog",
            "access_matrix",
            "known_cases_seal",
            "environment_expectation",
            "environment_observation",
            "job_manifest",
            "p00_audit_report",
        )
    }

    def validate(path: str, value: object) -> None:
        contracts.validate(
            "success_receipt" if path == "_SUCCESS.json" else names[path],
            value,
        )

    publish_p00(
        target,
        files,
        validate,
        {
            "status": vocabulary["publication_statuses"]["success"],
            "protocol_hash": compiled.protocol_hash,
        },
    )
    return 0


def _validate_p08_empirical_architecture(registry: Mapping[str, object]) -> dict[str, object]:
    """Fail closed unless P08 is locked to the P02/P05/P07 empirical architecture."""

    simulation = _mapping(registry.get("simulation"), "simulation")
    steps = _mapping(registry.get("steps"), "steps")
    artifacts = _mapping(registry.get("artifacts"), "artifacts")
    access_matrix = _mapping(registry.get("access_matrix"), "access_matrix")
    measurement = _mapping(registry.get("measurement"), "measurement")
    features = _mapping(registry.get("features"), "features")

    _validate_empirical_calibration_contract(simulation, artifacts)
    profile_summary = _validate_execution_profiles(simulation)
    access_summary = _validate_p08_access_contract(steps, artifacts, access_matrix)
    scenario_summary = _validate_operational_scenarios(
        simulation=simulation,
        measurement=measurement,
        features=features,
    )

    return {
        "architecture_version": P08_EMPIRICAL_ARCHITECTURE_VERSION,
        "active_profile": profile_summary["active_profile"],
        "registered_profiles": profile_summary["registered_profiles"],
        "operational_scenarios": scenario_summary["operational_scenarios"],
        "semi_synthetic_scenarios": scenario_summary["semi_synthetic_scenarios"],
        "fully_synthetic_scenarios": scenario_summary["fully_synthetic_scenarios"],
        "calibration_targets": scenario_summary["calibration_targets"],
        "required_reads": access_summary["required_reads"],
        "population_source": "P02.firm_year_panel",
        "observed_label_source": "P05.source_channel_matrices",
        "covariate_source": "P07.feature_panel",
        "sample_size_policy": "derived_from_exact_P02_development_rows",
        "prevalence_policy": "calibrated_from_P05_observed_rate_through_locked_DGP",
    }


def _validate_empirical_calibration_contract(
    simulation: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> None:
    calibration = _mapping(
        simulation.get("empirical_calibration"), "simulation.empirical_calibration"
    )
    expected = {
        "enabled": True,
        "population_artifact": "firm_year_panel",
        "covariate_artifact": "feature_panel",
        "label_artifact": "source_channel_matrices",
        "sample_design": "p02_development_panel",
        "empirical_row_mode": "exact_rows_no_bootstrap",
        "prevalence_calibration": "match_P05_observed_positive_rate_through_locked_DGP",
        "incompatible_detection_model": "fail_closed",
        "calibration_target_must_be_explicit": True,
    }
    drift = {
        key: {"expected": expected_value, "observed": calibration.get(key)}
        for key, expected_value in expected.items()
        if calibration.get(key) != expected_value
    }
    if drift:
        raise ValueError(f"P08 empirical calibration contract drift: {drift}")

    repeats = calibration.get("calibration_repeats")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 32:
        raise ValueError(
            "simulation.empirical_calibration.calibration_repeats must be an integer >= 32"
        )

    for key in ("population_artifact", "covariate_artifact", "label_artifact"):
        artifact_id = str(calibration[key])
        if artifact_id not in artifacts:
            raise ValueError(
                f"simulation.empirical_calibration.{key} references unknown artifact={artifact_id}"
            )

    if simulation.get("imports_production_labels") is not True:
        raise ValueError("simulation.imports_production_labels must be true")
    if simulation.get("outer_outcomes_allowed") is not False:
        raise ValueError("simulation.outer_outcomes_allowed must be false")
    if simulation.get("known_cases_allowed") is not False:
        raise ValueError("simulation.known_cases_allowed must be false")
    if simulation.get("full_cartesian_product") is not False:
        raise ValueError("simulation.full_cartesian_product must remain false")

    objective = _mapping(simulation.get("objective"), "simulation.objective")
    if objective.get("ground_truth_known") is not True:
        raise ValueError("simulation.objective.ground_truth_known must be true")

    legacy_prevalence = simulation.get("prevalence")
    if legacy_prevalence not in (None, {}):
        raise ValueError(
            "simulation.prevalence is a forbidden legacy grid under empirical calibration; "
            "semi-synthetic prevalence is derived from P05 and fully synthetic prevalence "
            "must be declared per scenario"
        )


def _validate_execution_profiles(simulation: Mapping[str, Any]) -> dict[str, object]:
    profiles = _mapping(simulation.get("execution_profiles"), "simulation.execution_profiles")
    active_profile = simulation.get("active_profile")
    if not isinstance(active_profile, str) or not active_profile:
        raise ValueError("simulation.active_profile must be a nonempty string")
    if active_profile not in profiles:
        raise ValueError(
            f"simulation.active_profile={active_profile} is not a registered execution profile"
        )

    learner_ids = set(_string_sequence(simulation.get("learners"), "simulation.learners"))
    method_ids = set(_string_sequence(simulation.get("methods"), "simulation.methods"))
    predictive_strategy_ids = method_ids - {"latent_bayesian", "partial_identification"}

    for profile_id, raw_profile in profiles.items():
        profile = _mapping(raw_profile, f"simulation.execution_profiles.{profile_id}")
        modes = sum(
            (
                profile.get("include_all_registered_methods") is True,
                profile.get("union_profiles") is not None,
                profile.get("blocks") is not None,
            )
        )
        if modes != 1:
            raise ValueError(
                f"execution profile={profile_id}: exactly one of include_all_registered_methods, "
                "union_profiles, or blocks is required"
            )
        if profile.get("union_profiles") is not None:
            members = _string_sequence(
                profile.get("union_profiles"),
                f"simulation.execution_profiles.{profile_id}.union_profiles",
            )
            unknown = sorted(set(members) - set(profiles))
            if unknown:
                raise ValueError(f"execution profile={profile_id}: unknown union profiles {unknown}")
            if profile_id in members:
                raise ValueError(f"execution profile={profile_id}: self-reference is forbidden")
        if profile.get("blocks") is not None:
            blocks = _sequence(
                profile.get("blocks"), f"simulation.execution_profiles.{profile_id}.blocks"
            )
            for index, raw_block in enumerate(blocks):
                block = _mapping(raw_block, f"profile={profile_id}.blocks[{index}]")
                block_learners = set(
                    _string_sequence(block.get("learners"), f"profile={profile_id}.learners")
                )
                block_strategies = set(
                    _string_sequence(block.get("strategies"), f"profile={profile_id}.strategies")
                )
                _string_sequence(
                    block.get("training_cost_regimes"),
                    f"profile={profile_id}.training_cost_regimes",
                )
                _string_sequence(
                    block.get("imbalance_treatments"),
                    f"profile={profile_id}.imbalance_treatments",
                )
                unknown_learners = sorted(block_learners - learner_ids)
                unknown_strategies = sorted(block_strategies - predictive_strategy_ids)
                if unknown_learners:
                    raise ValueError(
                        f"execution profile={profile_id}: unknown learners {unknown_learners}"
                    )
                if unknown_strategies:
                    raise ValueError(
                        f"execution profile={profile_id}: unknown strategies {unknown_strategies}"
                    )

    core = _mapping(profiles.get("core"), "simulation.execution_profiles.core")
    core_learners = _profile_block_learners(core, "core")
    matrix_design = _mapping(simulation.get("matrix_design"), "simulation.matrix_design")
    locked_core = set(
        _string_sequence(
            matrix_design.get("confirmatory_vietnamese_learners"),
            "simulation.matrix_design.confirmatory_vietnamese_learners",
        )
    )
    if core_learners != locked_core:
        raise ValueError(
            "core execution profile learners must exactly match "
            "matrix_design.confirmatory_vietnamese_learners; "
            f"profile={sorted(core_learners)}, matrix={sorted(locked_core)}"
        )

    return {
        "active_profile": active_profile,
        "registered_profiles": len(profiles),
    }


def _validate_p08_access_contract(
    steps: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    access_matrix: Mapping[str, Any],
) -> dict[str, object]:
    p08 = _mapping(steps.get("P08"), "steps.P08")
    reads = set(_string_sequence(p08.get("reads"), "steps.P08.reads"))
    writes = set(_string_sequence(p08.get("writes"), "steps.P08.writes"))

    missing_reads = sorted(P08_REQUIRED_READS - reads)
    missing_writes = sorted(P08_REQUIRED_WRITES - writes)
    forbidden_reads = sorted(P08_FORBIDDEN_READS & reads)
    if missing_reads:
        raise ValueError(f"steps.P08.reads is missing empirical inputs {missing_reads}")
    if missing_writes:
        raise ValueError(f"steps.P08.writes is missing required outputs {missing_writes}")
    if forbidden_reads:
        raise ValueError(f"steps.P08.reads contains sealed/forbidden inputs {forbidden_reads}")
    if p08.get("outer_access") != "sealed":
        raise ValueError("steps.P08.outer_access must remain sealed")
    if p08.get("known_case_access") not in {"none", "sealed"}:
        raise ValueError("steps.P08.known_case_access must be none or sealed")
    if "FEATURED" not in _string_sequence(
        p08.get("permitted_states"), "steps.P08.permitted_states"
    ):
        raise ValueError("steps.P08.permitted_states must include FEATURED")
    if "SIMULATED" not in _string_sequence(
        p08.get("allowed_next_states"), "steps.P08.allowed_next_states"
    ):
        raise ValueError("steps.P08.allowed_next_states must include SIMULATED")

    matrix_p08 = _mapping(access_matrix.get("P08"), "access_matrix.P08")
    matrix_reads = set(
        _string_sequence(matrix_p08.get("reads"), "access_matrix.P08.reads")
    )
    matrix_writes = set(
        _string_sequence(matrix_p08.get("writes"), "access_matrix.P08.writes")
    )
    if matrix_reads != reads or matrix_writes != writes:
        raise ValueError("compiled access_matrix.P08 does not match steps.P08")

    for artifact_id in P08_REQUIRED_READS | P08_REQUIRED_WRITES:
        if artifact_id not in artifacts:
            raise ValueError(f"P08 references unknown artifact={artifact_id}")

    for artifact_id in P08_REQUIRED_READS - {"simulation_scenario_registry", "simulation_batches"}:
        artifact = _mapping(artifacts.get(artifact_id), f"artifacts.{artifact_id}")
        if artifact.get("sensitivity_class") in {"outer", "known_case"}:
            raise ValueError(
                f"P08 empirical input artifact={artifact_id} has forbidden sensitivity_class="
                f"{artifact.get('sensitivity_class')}"
            )

    producers: dict[str, list[str]] = {}
    for step_id, raw_step in steps.items():
        step = _mapping(raw_step, f"steps.{step_id}")
        for artifact_id in _string_sequence_allow_empty(
            step.get("writes"), f"steps.{step_id}.writes"
        ):
            producers.setdefault(artifact_id, []).append(str(step_id))
    for artifact_id, expected_step in P08_EXPECTED_PRODUCERS.items():
        observed = producers.get(artifact_id, [])
        if observed != [expected_step]:
            raise ValueError(
                f"artifact={artifact_id}: expected sole producer={expected_step}, observed={observed}"
            )

    return {"required_reads": sorted(P08_REQUIRED_READS)}


def _validate_operational_scenarios(
    *,
    simulation: Mapping[str, Any],
    measurement: Mapping[str, Any],
    features: Mapping[str, Any],
) -> dict[str, object]:
    scenarios = _sequence(
        simulation.get("operational_scenarios"), "simulation.operational_scenarios"
    )
    candidate_targets = set(
        _mapping(measurement.get("candidate_targets"), "measurement.candidate_targets")
    )
    feature_columns = _registered_feature_columns(features)

    seen_ids: set[str] = set()
    semi_count = 0
    fully_count = 0
    calibration_targets: set[str] = set()
    for index, raw_scenario in enumerate(scenarios):
        scenario = _mapping(raw_scenario, f"simulation.operational_scenarios[{index}]")
        scenario_id = scenario.get(SCENARIO_ID)
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(f"operational scenario index={index}: nonempty scenario_id required")
        if scenario_id in seen_ids:
            raise ValueError(f"duplicate operational scenario_id={scenario_id}")
        seen_ids.add(scenario_id)

        tier = scenario.get("tier")
        if tier == SEMI_SYNTHETIC_TIER:
            semi_count += 1
            present_forbidden = sorted(SEMI_SYNTHETIC_FORBIDDEN_FIELDS & set(scenario))
            if present_forbidden:
                raise ValueError(
                    f"scenario={scenario_id}: semi-synthetic derived fields are forbidden at P00 "
                    f"{present_forbidden}"
                )
            if scenario.get("sample_design") != "p02_development_panel":
                raise ValueError(
                    f"scenario={scenario_id}: sample_design must be p02_development_panel"
                )
            validate_scenario_target_identity(scenario)
            target_id = scenario.get("calibration_target_id")
            if target_id not in candidate_targets:
                raise ValueError(
                    f"scenario={scenario_id}: calibration_target_id={target_id} is not registered "
                    "under measurement.candidate_targets"
                )
            calibration_targets.add(target_id)
            selected_features = set(
                _string_sequence(
                    scenario.get("semi_synthetic_feature_ids"),
                    f"scenario={scenario_id}.semi_synthetic_feature_ids",
                )
            )
            unknown_features = sorted(selected_features - feature_columns)
            if unknown_features:
                raise ValueError(
                    f"scenario={scenario_id}: unknown P07 physical feature columns {unknown_features}"
                )
            risk_strength = scenario.get("latent_risk_strength")
            if not isinstance(risk_strength, (int, float)) or isinstance(risk_strength, bool):
                raise ValueError(
                    f"scenario={scenario_id}: latent_risk_strength must be numeric"
                )
            if float(risk_strength) <= 0.0:
                raise ValueError(
                    f"scenario={scenario_id}: latent_risk_strength must be positive"
                )
        elif tier == FULLY_SYNTHETIC_TIER:
            fully_count += 1
            sample_size = scenario.get("sample_size")
            prevalence = scenario.get("prevalence")
            if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
                raise ValueError(
                    f"scenario={scenario_id}: fully synthetic sample_size must be a positive integer"
                )
            if not isinstance(prevalence, (int, float)) or isinstance(prevalence, bool):
                raise ValueError(
                    f"scenario={scenario_id}: fully synthetic prevalence must be numeric"
                )
            if not 0.0 < float(prevalence) < 1.0:
                raise ValueError(
                    f"scenario={scenario_id}: fully synthetic prevalence must be in (0, 1)"
                )
        else:
            raise ValueError(
                f"scenario={scenario_id}: unsupported tier={tier}; expected "
                f"{SEMI_SYNTHETIC_TIER} or {FULLY_SYNTHETIC_TIER}"
            )

    if semi_count == 0:
        raise ValueError(
            "simulation.operational_scenarios must contain at least one "
            "semi_synthetic_development_covariates scenario"
        )

    return {
        "operational_scenarios": len(scenarios),
        "semi_synthetic_scenarios": semi_count,
        "fully_synthetic_scenarios": fully_count,
        "calibration_targets": sorted(calibration_targets),
    }


def _registered_feature_columns(features: Mapping[str, Any]) -> set[str]:
    registry = _sequence(features.get("registry"), "features.registry")
    columns: set[str] = set()
    for index, raw_item in enumerate(registry):
        item = _mapping(raw_item, f"features.registry[{index}]")
        column = item.get("physical_column")
        if not isinstance(column, str) or not column:
            raise ValueError(f"features.registry[{index}].physical_column must be nonempty")
        if column in columns:
            raise ValueError(f"duplicate feature physical_column={column}")
        columns.add(column)
    return columns


def _profile_block_learners(profile: Mapping[str, Any], profile_id: str) -> set[str]:
    blocks = _sequence(profile.get("blocks"), f"execution profile={profile_id}.blocks")
    learners: set[str] = set()
    for index, raw_block in enumerate(blocks):
        block = _mapping(raw_block, f"execution profile={profile_id}.blocks[{index}]")
        learners.update(
            _string_sequence(
                block.get("learners"),
                f"execution profile={profile_id}.blocks[{index}].learners",
            )
        )
    return learners


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: mapping required")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object, context: str) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"{context}: nonempty sequence required")
    return list(value)


def _string_sequence(value: object, context: str) -> list[str]:
    values = _sequence(value, context)
    strings = [str(item) for item in values]
    if any(not item for item in strings) or len(strings) != len(set(strings)):
        raise ValueError(f"{context}: unique nonempty strings required")
    return strings


def _string_sequence_allow_empty(value: object, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{context}: sequence required")
    strings = [str(item) for item in value]
    if any(not item for item in strings) or len(strings) != len(set(strings)):
        raise ValueError(f"{context}: unique nonempty strings required")
    return strings


def _report(
    run_id: str,
    protocol_hash: str,
    registry: dict[str, object],
    p08_architecture: Mapping[str, object],
) -> str:
    calibration_targets = ", ".join(
        str(value)
        for value in cast(Sequence[object], p08_architecture["calibration_targets"])
    )
    return "\n".join(
        (
            "# P00 audit report",
            "",
            f"- Run ID: `{run_id}`",
            f"- Protocol hash: `{protocol_hash}`",
            f"- Compiler version: `{registry['compiler_version']}`",
            f"- Canonical Appendix B decisions: {len(cast(Sized, registry['appendix_b']))}",
            f"- Traceability rows: {len(cast(Sized, registry['decision_traceability']))}",
            f"- Schemas: {len(cast(Sized, registry['schemas']))}",
            f"- Artifacts: {len(cast(Sized, registry['artifacts']))}",
            f"- Steps: {len(cast(Sized, registry['steps']))}",
            f"- Executable test controls: {len(cast(Sized, registry['tests']))}",
            f"- Data snapshot ID: `{cast(dict[str, object], registry.get('data_snapshot', {})).get('snapshot_id')}`",
            f"- Data snapshot hash: `{cast(dict[str, object], registry.get('data_snapshot', {})).get('snapshot_hash')}`",
            "",
            "## P08 empirical-calibration architecture",
            "",
            f"- Architecture version: `{p08_architecture['architecture_version']}`",
            f"- Active execution profile: `{p08_architecture['active_profile']}`",
            f"- Registered execution profiles: {p08_architecture['registered_profiles']}",
            f"- Operational scenarios: {p08_architecture['operational_scenarios']}",
            f"- Semi-synthetic scenarios: {p08_architecture['semi_synthetic_scenarios']}",
            f"- Fully synthetic scenarios: {p08_architecture['fully_synthetic_scenarios']}",
            f"- Calibration targets: `{calibration_targets}`",
            "- Population/sample-size source: `P02.firm_year_panel`",
            "- Observed-label-rate source: `P05.source_channel_matrices`",
            "- Covariate source: `P07.feature_panel`",
            "- Empirical row policy: exact P02 development rows; no bootstrap population",
            "- Semi-synthetic prevalence policy: calibrated through the locked DGP; not researcher-fixed",
            "- Outer outcomes: SEALED",
            "- Known cases: SEALED",
            "- P08 empirical architecture: PASS",
            "",
            "- Methodological invariants: PASS",
            "- Artifact/access references: PASS",
            "- Test-node collection: PASS",
            "- Final status: SUCCESS",
            "",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
