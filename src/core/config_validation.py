"""Cross-reference and exact Chapter 3 v19 methodological invariant validation."""

from __future__ import annotations

import re
from typing import Any, cast

from .errors import (
    AccessPolicyError,
    ArtifactPathCollisionError,
    MethodologicalInvariantError,
    UnknownReferenceError,
)


def _mapping(registry: dict[str, object], key: str) -> dict[str, Any]:
    value = registry.get(key)
    if not isinstance(value, dict):
        raise UnknownReferenceError(f"namespace={key}: mapping required")
    return cast(dict[str, Any], value)


def validate_references(registry: dict[str, object]) -> None:
    artifacts = _mapping(registry, "artifacts")
    steps = _mapping(registry, "steps")
    schemas = _mapping(registry, "schemas")
    tests = _mapping(registry, "tests")
    seen: dict[str, str] = {}
    format_contracts = {
        "parquet": {"dataframe"},
        "json": {"json_object", "json_array", "receipt"},
        "text": {"text"},
        "markdown": {"markdown"},
    }
    for artifact_id, raw in artifacts.items():
        if not isinstance(raw, dict):
            raise UnknownReferenceError(f"artifact={artifact_id}: mapping required")
        item = cast(dict[str, Any], raw)
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
            if key not in item:
                raise UnknownReferenceError(f"artifact={artifact_id}, key={key}: required")
        producer, schema_id = item["producer_step"], item["schema_id"]
        if producer not in steps or schema_id not in schemas:
            raise UnknownReferenceError(f"artifact={artifact_id}: unknown producer or schema")
        schema = schemas[schema_id]
        schema_spec = cast(dict[str, Any], schema) if isinstance(schema, dict) else {}
        if (
            not isinstance(schema, dict)
            or schema_spec.get("contract_type") not in format_contracts[item["format"]]
        ):
            raise UnknownReferenceError(f"artifact={artifact_id}: format/contract mismatch")
        template = str(item["path_template"])
        coordinates = item["coordinates"]
        if template.startswith("/") or ".." in template.replace("\\", "/").split("/"):
            raise ArtifactPathCollisionError(f"artifact={artifact_id}: path escapes root")
        placeholders = set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)}", template))
        if placeholders != set(coordinates):
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: coordinate placeholders differ"
            )
        signature = re.sub(
            r"{[A-Za-z_][A-Za-z0-9_]*}",
            "{coordinate}",
            template.replace("\\", "/"),
        )
        if signature in seen:
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: path collides with {seen[signature]}"
            )
        seen[signature] = artifact_id
        if artifact_id not in steps[producer].get("writes", []):
            raise UnknownReferenceError(f"artifact={artifact_id}: producer does not declare write")

    for step_id, raw in steps.items():
        if not isinstance(raw, dict):
            raise UnknownReferenceError(f"step={step_id}: mapping required")
        item = cast(dict[str, Any], raw)
        for artifact_id in [
            *item.get("reads", []),
            *item.get("optional_reads", []),
            *item.get("writes", []),
            *item.get("required_receipts", []),
        ]:
            if artifact_id not in artifacts:
                raise UnknownReferenceError(f"step={step_id}, artifact={artifact_id}: unknown")
        for test_id in item.get("test_ids", []):
            if test_id not in tests:
                raise UnknownReferenceError(f"step={step_id}, test={test_id}: unknown")
        if step_id == "P17" and any(
            artifacts[artifact_id]["producer_step"] != "P17"
            for artifact_id in item.get("writes", [])
        ):
            raise AccessPolicyError("P17 cannot write prior-stage artifacts")


def validate_methodology(registry: dict[str, object]) -> None:
    study = _mapping(registry, "study")
    sources = _mapping(registry, "data_sources")
    measurement = _mapping(registry, "measurement")
    evaluation = _mapping(registry, "evaluation")
    features = _mapping(registry, "features")
    learners = _mapping(registry, "learners")
    folds = _mapping(registry, "folds")
    weighting = _mapping(registry, "weighting")
    evidence = _mapping(registry, "evidence")
    risksets = _mapping(registry, "risksets")
    calibration = _mapping(registry, "calibration")
    inference = _mapping(registry, "inference")
    known = _mapping(registry, "known_cases")
    simulation = _mapping(registry, "simulation")

    required = [
        (study["horizons_months"]["primary"] == 12, "D02 primary horizon must be 12 months"),
        (
            study["horizons_months"]["sensitivity"] == [24],
            "D02 sensitivity horizon must be 24 months",
        ),
        (
            sources["anchor"]["false_positive_rate_grid"] == [0.0, 0.01, 0.03, 0.05],
            "D04 anchor grid differs",
        ),
        (measurement["track_a_primary_endpoint"] == "L1", "D05 primary endpoint must be L1"),
        (
            evaluation["review_budget"]["primary_fraction"] == 0.05,
            "D07 primary review budget must be 5%",
        ),
        (
            learners["confirmatory"] == ["elastic_net_logistic", "random_forest", "main_boosting"],
            "D08 learner roster differs",
        ),
        (
            learners["tuning"]["max_valid_configurations_per_learner_inner_fold"] == 50,
            "D09 tuning cap must be 50",
        ),
        (
            folds["initial_outer_year"] == 2020
            and folds["fully_nested_outer_years"] == [2021, 2022, 2023, 2024],
            "D10 fold calendar differs",
        ),
        (
            measurement["selection_candidates"] == ["L2", "L3_fixed_pi", "none"],
            "hierarchical-pi cannot enter Gate 1",
        ),
        (
            measurement["l3_model"]["hierarchical_pi"]["role"] == "sensitivity_only",
            "hierarchical-pi must be sensitivity only",
        ),
        (
            evaluation["ap_soft_targets"] is False
            and evaluation["track_b_metrics"]["ap_directly_on_soft_targets"] is False,
            "AP on soft target forbidden",
        ),
        (
            weighting["full_sample_role"] == "descriptive_only"
            and weighting["analytical_fit_scope"] == "development_history",
            "weights must be fold-aware",
        ),
        (weighting["ipcw_role"] == "sensitivity_only", "IPCW must be sensitivity only"),
        (
            evidence["missing_is_zero"] is False and evidence["immature_is_negative"] is False,
            "missing/immature cannot be negatives",
        ),
        (
            features["label_model_allows_content"] is False,
            "D20 content predictors cannot enter label models",
        ),
        (risksets["complete_followup_required"] is True, "D21 complete follow-up required"),
        (
            calibration["in_sample_predictions_after_refit_forbidden"] is True,
            "D35 in-sample calibration forbidden",
        ),
        (inference["families"] == ["CHNC2", "CHNC3"], "D36 family registry differs"),
        (known["soft_veto"]["minimum_cases"] == 4, "D30 known-case minimum differs"),
        (simulation["core"]["minimum_replications"] == 2500, "D44 core R must be at least 2500"),
        (simulation["l3"]["initial_replications"] == 1000, "D44 L3 initial R must be 1000"),
        (
            simulation["protocol_role"]["must_complete_before_outer_test"] is True,
            "D45 simulation must precede outer test",
        ),
        (
            simulation["protocol_role"]["gate_optimization_on_outer_performance_forbidden"] is True,
            "D45 outer gate optimization forbidden",
        ),
    ]
    failures = [message for passed, message in required if not passed]
    if failures:
        raise MethodologicalInvariantError("; ".join(failures))
