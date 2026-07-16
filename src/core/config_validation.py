"""Cross-reference and Chapter 3 methodological validation."""

from __future__ import annotations

import re
from typing import Any, cast

from .errors import (
    AccessPolicyError,
    ArtifactPathCollisionError,
    MethodologicalInvariantError,
    UnknownReferenceError,
)
from .semantic_keys import CHANNEL_ID

StringMap = dict[str, Any]


def _mapping(
    registry: dict[str, object],
    key: str,
) -> StringMap:
    value = registry.get(key)
    if not isinstance(value, dict):
        raise UnknownReferenceError(f"namespace={key}: mapping required")
    return cast(StringMap, value)


def _entry(value: object, context: str) -> StringMap:
    if not isinstance(value, dict):
        raise UnknownReferenceError(f"{context}: mapping required")
    return cast(StringMap, value)


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise UnknownReferenceError(f"{context}: list required")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise UnknownReferenceError(f"{context}: every item must be a string")
    return cast(list[str], items)


def validate_references(
    registry: dict[str, object],
) -> None:
    artifacts = _mapping(registry, "artifacts")
    steps = _mapping(registry, "steps")
    schemas = _mapping(registry, "schemas")
    tests = _mapping(registry, "tests")

    seen: dict[str, str] = {}
    format_contracts = {
        "parquet": {"dataframe"},
        "csv": {"dataframe"},
        "json": {
            "json_object",
            "json_array",
            "receipt",
        },
        "text": {"text"},
        "markdown": {"markdown"},
    }

    for artifact_id, raw_artifact in artifacts.items():
        item = _entry(
            raw_artifact,
            f"artifact={artifact_id}",
        )
        required_keys = (
            "producer_step",
            "path_template",
            "schema_id",
            "format",
            "coordinates",
            "sensitivity_class",
            "immutability",
            "aggregation_behavior",
            "checkpoint_role",
        )
        for key in required_keys:
            if key not in item:
                raise UnknownReferenceError(f"artifact={artifact_id}, key={key}: required")

        producer = item.get("producer_step")
        schema_id = item.get("schema_id")
        format_name = item.get("format")
        if not isinstance(producer, str):
            raise UnknownReferenceError(f"artifact={artifact_id}: producer_step must be a string")
        if not isinstance(schema_id, str):
            raise UnknownReferenceError(f"artifact={artifact_id}: schema_id must be a string")
        if not isinstance(format_name, str):
            raise UnknownReferenceError(f"artifact={artifact_id}: format must be a string")
        if producer not in steps or schema_id not in schemas:
            raise UnknownReferenceError(f"artifact={artifact_id}: unknown producer or schema")

        schema = _entry(
            schemas[schema_id],
            f"schema={schema_id}",
        )
        allowed_contracts = format_contracts.get(format_name)
        if allowed_contracts is None or schema.get("contract_type") not in allowed_contracts:
            raise UnknownReferenceError(f"artifact={artifact_id}: format/contract mismatch")

        template = item.get("path_template")
        if not isinstance(template, str):
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: path_template must be a string"
            )
        coordinates = _string_list(
            item.get("coordinates"),
            f"artifact={artifact_id}, coordinates",
        )
        normalized = template.replace("\\", "/")
        if template.startswith("/") or ".." in normalized.split("/"):
            raise ArtifactPathCollisionError(f"artifact={artifact_id}: path escapes root")

        placeholders = set(
            re.findall(
                r"{([A-Za-z_][A-Za-z0-9_]*)}",
                template,
            )
        )
        if placeholders != set(coordinates):
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: coordinate placeholders differ"
            )

        signature = re.sub(
            r"{[A-Za-z_][A-Za-z0-9_]*}",
            "{coordinate}",
            normalized,
        )
        if signature in seen:
            raise ArtifactPathCollisionError(
                f"artifact={artifact_id}: path collides with {seen[signature]}"
            )
        seen[signature] = artifact_id

        producer_step = _entry(
            steps[producer],
            f"step={producer}",
        )
        producer_writes = _string_list(
            producer_step.get("writes", []),
            f"step={producer}, writes",
        )
        if artifact_id not in producer_writes:
            raise UnknownReferenceError(f"artifact={artifact_id}: producer does not declare write")

    for step_id, raw_step in steps.items():
        item = _entry(raw_step, f"step={step_id}")
        referenced_artifacts = [
            *_string_list(
                item.get("reads", []),
                f"step={step_id}, reads",
            ),
            *_string_list(
                item.get("optional_reads", []),
                f"step={step_id}, optional_reads",
            ),
            *_string_list(
                item.get("writes", []),
                f"step={step_id}, writes",
            ),
            *_string_list(
                item.get("required_receipts", []),
                f"step={step_id}, required_receipts",
            ),
        ]
        for artifact_id in referenced_artifacts:
            if artifact_id not in artifacts:
                raise UnknownReferenceError(f"step={step_id}, artifact={artifact_id}: unknown")
        for test_id in _string_list(
            item.get("test_ids", []),
            f"step={step_id}, test_ids",
        ):
            if test_id not in tests:
                raise UnknownReferenceError(f"step={step_id}, test={test_id}: unknown")

        if step_id == "P17":
            for artifact_id in _string_list(
                item.get("writes", []),
                "step=P17, writes",
            ):
                artifact = _entry(
                    artifacts[artifact_id],
                    f"artifact={artifact_id}",
                )
                if artifact.get("producer_step") != "P17":
                    raise AccessPolicyError("P17 cannot write prior-stage artifacts")


def validate_methodology(
    registry: dict[str, object],
) -> None:
    study = _mapping(registry, "study")
    sources = _mapping(registry, "data_sources")
    entity_resolution = _mapping(registry, "entity_resolution")
    measurement = _mapping(registry, "measurement")
    evaluation = _mapping(registry, "evaluation")
    learners = _mapping(registry, "learners")
    folds = _mapping(registry, "folds")
    weighting = _mapping(registry, "weighting")
    evidence = _mapping(registry, "evidence")
    features = _mapping(registry, "features")
    risksets = _mapping(registry, "risksets")
    calibration = _mapping(registry, "calibration")
    inference = _mapping(registry, "inference")
    known = _mapping(registry, "known_cases")
    simulation = _mapping(registry, "simulation")
    s3_taxonomy = _mapping(registry, "s3_taxonomy")
    _validate_p02_configuration(sources, entity_resolution)
    _validate_s3_configuration(s3_taxonomy)

    horizons = _entry(
        study.get("horizons_months"),
        "study.horizons_months",
    )
    anchor = _entry(
        sources.get("anchor"),
        "data_sources.anchor",
    )
    review_budget = _entry(
        evaluation.get("review_budget"),
        "evaluation.review_budget",
    )
    tuning = _entry(
        learners.get("tuning"),
        "learners.tuning",
    )
    l3_model = _entry(
        measurement.get("l3_model"),
        "measurement.l3_model",
    )
    hierarchical = _entry(
        l3_model.get("hierarchical_pi"),
        "measurement.l3_model.hierarchical_pi",
    )
    track_b_metrics = _entry(
        evaluation.get("track_b_metrics"),
        "evaluation.track_b_metrics",
    )
    soft_veto = _entry(
        known.get("soft_veto"),
        "known_cases.soft_veto",
    )
    simulation_core = _entry(
        simulation.get("core"),
        "simulation.core",
    )
    simulation_l3 = _entry(
        simulation.get("l3"),
        "simulation.l3",
    )
    protocol_role = _entry(
        simulation.get("protocol_role"),
        "simulation.protocol_role",
    )

    required = [
        (
            horizons.get("primary") == 12,
            "D02 primary horizon must be 12 months",
        ),
        (
            horizons.get("sensitivity") == [24],
            "D02 sensitivity horizon must be 24 months",
        ),
        (
            anchor.get("false_positive_rate_grid") == [0.0, 0.01, 0.03, 0.05],
            "D04 anchor grid differs",
        ),
        (
            _candidate_target_sources_valid(measurement.get("candidate_targets"))
            and _primary_target_valid(measurement),
            "D05 primary target must be null or explicitly reference a registered candidate",
        ),
        (
            review_budget.get("primary_fraction") == 0.05,
            "D07 primary review budget must be 5%",
        ),
        (
            learners.get("confirmatory")
            == [
                "elastic_net_logistic",
                "random_forest",
                "main_boosting",
            ],
            "D08 learner roster differs",
        ),
        (
            tuning.get("max_valid_configurations_per_learner_inner_fold") == 50,
            "D09 tuning cap must be 50",
        ),
        (
            folds.get("initial_outer_year") == 2020
            and folds.get("fully_nested_outer_years") == [2021, 2022, 2023, 2024],
            "D10 fold calendar differs",
        ),
        (
            measurement.get("selection_candidates") == ["L2", "L3_fixed_pi", "none"],
            "hierarchical-pi cannot enter Gate 1",
        ),
        (
            hierarchical.get("role") == "sensitivity_only",
            "hierarchical-pi must be sensitivity only",
        ),
        (
            evaluation.get("ap_soft_targets") is False
            and track_b_metrics.get("ap_directly_on_soft_targets") is False,
            "AP on soft target forbidden",
        ),
        (
            weighting.get("full_sample_role") == "descriptive_only"
            and weighting.get("analytical_fit_scope") == "development_history",
            "weights must be fold-aware",
        ),
        (
            weighting.get("ipcw_role") == "sensitivity_only",
            "IPCW must be sensitivity only",
        ),
        (
            evidence.get("missing_is_zero") is False
            and evidence.get("immature_is_negative") is False,
            "missing or immature evidence cannot be negative",
        ),
        (
            features.get("label_model_allows_content") is False,
            "content predictors are forbidden in the label model",
        ),
        (
            risksets.get("complete_followup_required") is True,
            "D21 complete follow-up required",
        ),
        (
            calibration.get("in_sample_predictions_after_refit_forbidden") is True,
            "D35 in-sample calibration forbidden",
        ),
        (
            inference.get("families") == ["CHNC2", "CHNC3"],
            "D36 family registry differs",
        ),
        (
            soft_veto.get("minimum_cases") == 4,
            "D30 known-case minimum differs",
        ),
        (
            simulation_core.get("minimum_replications") == 2500,
            "D44 core R must be at least 2500",
        ),
        (
            simulation_l3.get("initial_replications") == 1000,
            "D44 L3 initial R must be 1000",
        ),
        (
            protocol_role.get("must_complete_before_outer_test") is True,
            "D45 simulation must precede outer test",
        ),
        (
            protocol_role.get("gate_optimization_on_outer_performance_forbidden") is True,
            "D45 outer gate optimization forbidden",
        ),
    ]
    failures = [message for passed, message in required if not passed]
    if failures:
        raise MethodologicalInvariantError("; ".join(failures))


def _candidate_target_sources_valid(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    candidates = cast(dict[object, object], value)
    required = {
        "L1_ANNUAL": [
            "S1_profit_adjustment",
            "S1_revenue_adjustment",
            "S2_audit_opinion",
        ],
        "S3_BROAD": ["S3_BROAD"],
        "S3_REPORTING": ["S3_REPORTING"],
        "S3_CONTENT": ["S3_CONTENT"],
        "S3_TIMELINESS": ["S3_TIMELINESS"],
        "L1_REPORTING": [
            "S1_profit_adjustment",
            "S1_revenue_adjustment",
            "S2_audit_opinion",
            "S3_REPORTING",
        ],
        "L1_CONTENT_STRICT": [
            "S1_profit_adjustment",
            "S1_revenue_adjustment",
            "S3_CONTENT",
        ],
    }
    if set(candidates) != set(required):
        return False
    for target_id, expected_sources in required.items():
        raw = candidates[target_id]
        if not isinstance(raw, dict):
            return False
        sources = cast(dict[object, object], raw).get("sources")
        if sources != expected_sources:
            return False
    return True


def _primary_target_valid(measurement: dict[str, Any]) -> bool:
    primary = measurement.get("primary_target_id")
    if primary is None:
        return True
    candidates = measurement.get("candidate_targets")
    return isinstance(primary, str) and isinstance(candidates, dict) and primary in candidates


def _validate_s3_configuration(configuration: dict[str, Any]) -> None:
    if configuration.get(CHANNEL_ID) != "S3":
        raise MethodologicalInvariantError("S3 taxonomy must remain one S3 channel")
    if configuration.get("target_year_rule") != "sanction_year_minus_one":
        raise MethodologicalInvariantError("S3 target year must be sanction year minus one")
    endpoints = _entry(configuration.get("endpoints"), "s3_taxonomy.endpoints")
    required = {"S3_BROAD", "S3_REPORTING", "S3_CONTENT", "S3_TIMELINESS"}
    if set(endpoints) != required:
        raise MethodologicalInvariantError("S3 endpoint registry differs")
    content = set(
        _string_list(
            _entry(endpoints["S3_CONTENT"], "S3_CONTENT").get("normalized_codes"),
            "S3_CONTENT.normalized_codes",
        )
    )
    timeliness = set(
        _string_list(
            _entry(endpoints["S3_TIMELINESS"], "S3_TIMELINESS").get("normalized_codes"),
            "S3_TIMELINESS.normalized_codes",
        )
    )
    if content & timeliness:
        raise MethodologicalInvariantError("S3 content and timeliness codes must be disjoint")
    completeness = _entry(
        configuration.get("sanction_source_completeness"),
        "s3_taxonomy.sanction_source_completeness",
    )
    if completeness.get("complete_through_year") != 2025:
        raise MethodologicalInvariantError("S3 source completeness must end at 2025")
    if completeness.get("incomplete_years") != [2026]:
        raise MethodologicalInvariantError("S3 source year 2026 must remain incomplete")


def _validate_p02_configuration(
    data_sources: dict[str, Any],
    entity_resolution: dict[str, Any],
) -> None:
    if entity_resolution.get("policy") != "registered_only":
        raise MethodologicalInvariantError("P02 requires entity_resolution.policy=registered_only")
    if entity_resolution.get("collision_policy") != "fail":
        raise MethodologicalInvariantError("P02 requires entity_resolution.collision_policy=fail")
    source_registry = _entry(
        data_sources.get("source_registry"),
        "data_sources.source_registry",
    )
    registered_sources = _entry(
        source_registry.get("sources", {}),
        "data_sources.source_registry.sources",
    )
    enabled_count = 0
    core_count = 0
    for source_id, raw_source in registered_sources.items():
        source = _entry(raw_source, f"source={source_id}")
        enabled = source.get("enabled", True)
        if not isinstance(enabled, bool):
            raise MethodologicalInvariantError(f"source={source_id}.enabled must be boolean")
        if not enabled:
            continue
        raw_panel = source.get("panel_mapping")
        if not isinstance(raw_panel, dict):
            raise MethodologicalInvariantError(f"source={source_id}.panel_mapping is required")
        panel = cast(dict[str, Any], raw_panel)
        panel_enabled = panel.get("enabled", True)
        if not isinstance(panel_enabled, bool):
            raise MethodologicalInvariantError(
                f"source={source_id}.panel_mapping.enabled must be boolean"
            )
        if not panel_enabled:
            continue
        enabled_count += 1
        for field_name in (
            "firm_id_field",
            "fiscal_year_field",
            "availability_date_field",
        ):
            value = panel.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise MethodologicalInvariantError(
                    f"source={source_id}.panel_mapping.{field_name} is required"
                )
        ticker = panel.get("ticker_field")
        if ticker is not None and not isinstance(ticker, str):
            raise MethodologicalInvariantError(
                f"source={source_id}.panel_mapping.ticker_field must be string or null"
            )
        if ticker is not None and panel.get("firm_id_field") == ticker:
            raise MethodologicalInvariantError(
                f"source={source_id}: ticker cannot be immutable firm identity"
            )
        if panel.get("availability_date_field") != source.get("availability_date_field"):
            raise MethodologicalInvariantError(
                f"source={source_id}: P01 and P02 availability fields must match"
            )
        contributes = panel.get("contributes_to_firm_master", True)
        core = panel.get("core_predictor", False)
        if not isinstance(contributes, bool) or not isinstance(core, bool):
            raise MethodologicalInvariantError(
                f"source={source_id}: P02 panel flags must be boolean"
            )
        if core:
            core_count += 1
    if enabled_count and core_count == 0:
        raise MethodologicalInvariantError(
            "P02 requires at least one enabled core predictor source"
        )
