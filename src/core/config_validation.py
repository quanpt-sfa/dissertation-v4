"""Cross-reference and methodological-invariant validation for P0 only."""

from __future__ import annotations

import re

from .errors import (
    AccessPolicyError,
    ArtifactPathCollisionError,
    MethodologicalInvariantError,
    UnknownReferenceError,
)


def _mapping(registry: dict[str, object], key: str) -> dict[str, object]:
    value = registry.get(key)
    if not isinstance(value, dict):
        raise UnknownReferenceError(f"namespace={key}: mapping required")
    return value


def validate_references(registry: dict[str, object]) -> None:
    """Validate artifact, schema, step, test, and artifact-template references."""
    artifacts = _mapping(registry, "artifacts")
    steps = _mapping(registry, "steps")
    schemas = _mapping(registry, "schemas")
    tests = _mapping(registry, "tests")
    seen_paths: dict[str, str] = {}
    for artifact_id, value in artifacts.items():
        if not isinstance(value, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: mapping required")
        for key in (
            "producer_step",
            "path_template",
            "schema_id",
            "format",
            "coordinates",
            "sensitivity_class",
            "immutability",
            "aggregation_behavior",
            "checkpoint_role",
        ):
            if key not in value:
                raise UnknownReferenceError(
                    f"artifact={artifact_id}, key={key}: required declaration missing"
                )
        producer = value["producer_step"]
        schema = value["schema_id"]
        if producer not in steps:
            raise UnknownReferenceError(
                f"artifact={artifact_id}, producer={producer}: unknown step"
            )
        if schema not in schemas:
            raise UnknownReferenceError(f"artifact={artifact_id}, schema={schema}: unknown schema")
        template = value["path_template"]
        coordinates = value["coordinates"]
        if not isinstance(template, str) or not isinstance(coordinates, list):
            raise UnknownReferenceError(
                f"artifact={artifact_id}: path_template string and coordinates list required"
            )
        if template.startswith("/") or ".." in template.replace("\\", "/").split("/"):
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: template escapes registered artifact root"
            )
        placeholders = set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)}", template))
        if placeholders != set(coordinates):
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: template placeholders must equal coordinates"
            )
        signature = re.sub(r"{[A-Za-z_][A-Za-z0-9_]*}", "{coordinate}", template.replace("\\", "/"))
        if signature in seen_paths:
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: path can collide with {seen_paths[signature]}"
            )
        seen_paths[signature] = str(artifact_id)
        producer_spec = steps[producer]
        if not isinstance(producer_spec, dict) or artifact_id not in producer_spec.get(
            "writes", []
        ):
            raise UnknownReferenceError(
                f"artifact={artifact_id}: producer step must declare the artifact in writes"
            )
    for step_id, value in steps.items():
        if not isinstance(value, dict):
            raise UnknownReferenceError(f"step={step_id}: mapping required")
        required = (
            "description",
            "cli_module",
            "unit_coordinates",
            "reads",
            "optional_reads",
            "writes",
            "outer_access",
            "known_case_access",
            "checkpoint_role",
            "permitted_states",
            "allowed_next_states",
            "test_ids",
            "decision_ids",
        )
        absent = [key for key in required if key not in value]
        if absent:
            raise UnknownReferenceError(f"step={step_id}: missing required fields {absent}")
        for artifact_id in [*value["reads"], *value["optional_reads"], *value["writes"]]:
            if artifact_id not in artifacts:
                raise UnknownReferenceError(
                    f"step={step_id}, artifact={artifact_id}: unknown artifact"
                )
        for test_id in value["test_ids"]:
            if test_id not in tests:
                raise UnknownReferenceError(f"step={step_id}, test={test_id}: unknown test")
        if value["outer_access"] == "sealed" and any("outer" in str(a) for a in value["reads"]):
            raise AccessPolicyError(
                f"step={step_id}: sealed outer access cannot read outer artifacts"
            )
        known_steps = _mapping(registry, "access_control").get("known_case_access_steps")
        if not isinstance(known_steps, list) or (
            step_id not in known_steps and value["known_case_access"] != "none"
        ):
            raise AccessPolicyError(
                f"step={step_id}: known-case content is not configured to open here"
            )
        if step_id == "P17":
            for artifact_id in value["writes"]:
                producer = artifacts[artifact_id]
                if isinstance(producer, dict) and producer.get("producer_step") != "P17":
                    raise AccessPolicyError(
                        f"step=P17, artifact={artifact_id}: cannot write prior-step artifact"
                    )


def validate_methodology(registry: dict[str, object]) -> None:
    """Enforce locked methodological invariants as configuration rules."""
    measurement = _mapping(registry, "measurement")
    evaluation = _mapping(registry, "evaluation")
    features = _mapping(registry, "features")
    weighting = _mapping(registry, "weighting")
    simulation = _mapping(registry, "simulation")
    learners = _mapping(registry, "learners")
    utility = _mapping(registry, "utility")
    inference = _mapping(registry, "inference")
    if measurement.get("track_a_primary_endpoint") != "L1":
        raise MethodologicalInvariantError(
            "measurement.track_a_primary_endpoint: Track A must use L1"
        )
    candidates = measurement.get("selection_candidates")
    if candidates != ["L2", "L3_fixed_pi", "none"]:
        raise MethodologicalInvariantError(
            "measurement.selection_candidates: Gate 1 permits only L2, L3_fixed_pi, none"
        )
    hierarchical = measurement.get("hierarchical_pi")
    if not isinstance(hierarchical, dict) or hierarchical.get("role") != "sensitivity_only":
        raise MethodologicalInvariantError(
            "measurement.hierarchical_pi.role: hierarchical-pi is sensitivity only"
        )
    if evaluation.get("ap_soft_targets"):
        raise MethodologicalInvariantError(
            "evaluation.ap_soft_targets: AP cannot evaluate L2/L3 soft targets"
        )
    if features.get("label_model_allows_content"):
        raise MethodologicalInvariantError(
            "features.label_model_allows_content: content predictors are forbidden"
        )
    ladder = evaluation.get("benchmark_ladder")
    if (
        not isinstance(ladder, list)
        or len(ladder) != 6
        or "PU" in ladder
        or evaluation.get("pu_branch") != "separate"
    ):
        raise MethodologicalInvariantError(
            "evaluation.benchmark_ladder: exactly six levels and a separate PU branch required"
        )
    if (
        weighting.get("full_sample_role") != "descriptive_only"
        or weighting.get("analytical_fit_scope") != "development_history"
    ):
        raise MethodologicalInvariantError(
            "weighting: full-sample weights are descriptive; analytical weights are fold-aware"
        )
    if weighting.get("ipcw_role") != "sensitivity_only":
        raise MethodologicalInvariantError("weighting.ipcw_role: IPCW is sensitivity only")
    if simulation.get("imports_production_labels") is not True:
        raise MethodologicalInvariantError(
            "simulation.imports_production_labels: production L0-L3 import required"
        )
    roster = simulation.get("model_roster")
    if not isinstance(roster, list) or not {
        "oracle",
        "observability_only",
        "content_only",
        "full",
        "anchor_pu",
    }.issubset(roster):
        raise MethodologicalInvariantError(
            "simulation.model_roster: oracle, observability-only, content-only, full, anchor-PU required"
        )
    if simulation.get("targets") != ["L1", "L2", "L3_fixed_pi"]:
        raise MethodologicalInvariantError("simulation.targets: L1, L2, L3_fixed_pi required")
    if simulation.get("methods") != [
        "oracle",
        "observability_only",
        "content_only",
        "full",
        "anchor_pu",
    ]:
        raise MethodologicalInvariantError("simulation.methods: registered five-method roster required")
    if simulation.get("learners") != ["logistic", "main_boosting"] or learners.get("registered") != [
        "logistic",
        "main_boosting",
    ]:
        raise MethodologicalInvariantError("learners: logistic and main_boosting required")
    if simulation.get("gate3_operating_characteristics") is not True:
        raise MethodologicalInvariantError("simulation.gate3_operating_characteristics: required")
    core = simulation.get("core")
    l3 = simulation.get("l3")
    if not isinstance(core, dict) or not isinstance(l3, dict):
        raise MethodologicalInvariantError("simulation: core and L3 policies required")
    if (
        core.get("minimum_replications") != 2500
        or not isinstance(core.get("pass_fail_mcse_max"), (int, float))
        or core["pass_fail_mcse_max"] <= 0
    ):
        raise MethodologicalInvariantError(
            "simulation.core: minimum 2,500 and positive MCSE threshold required"
        )
    if l3.get("initial_replications") != 1000 or l3.get("pass_fail_mcse_max") == core.get(
        "pass_fail_mcse_max"
    ):
        raise MethodologicalInvariantError(
            "simulation.l3: initial 1,000 and distinct threshold required"
        )
    prevalence = simulation.get("prevalence")
    dimensions = simulation.get("dimensions")
    if (
        not isinstance(prevalence, dict)
        or prevalence.get("independent_scenario_dimension") is not True
        or not isinstance(dimensions, list)
    ):
        raise MethodologicalInvariantError(
            "simulation: prevalence and scenario dimensions required"
        )
    if inference.get("chnc3_family_adjusted") is not True:
        raise MethodologicalInvariantError("inference.chnc3_family_adjusted: required")
    components = utility.get("components")
    if not isinstance(components, list) or not {
        "review_cost",
        "additional_false_positive_cost",
    }.issubset(components):
        raise MethodologicalInvariantError(
            "utility.components: separate review and additional false-positive costs required"
        )
