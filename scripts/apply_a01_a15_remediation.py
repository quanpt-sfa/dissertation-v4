#!/usr/bin/env python3
"""Apply the verified A-01--A-15 audit remediations once.

This temporary script is removed before the pull request is finalized.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _replace(path: str, old: str, new: str) -> None:
    text = _read(path)
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"{path}: replacement anchor not found")
    _write(path, text.replace(old, new, 1))


def _append_once(path: str, marker: str, addition: str) -> None:
    text = _read(path)
    if addition.strip() in text:
        return
    if marker not in text:
        raise RuntimeError(f"{path}: append marker not found")
    _write(path, text.replace(marker, marker + addition, 1))


def _patch_feature_eligibility() -> None:
    _replace(
        "src/features/service.py",
        '_DECISION_STATUS = {"LOCKED", "RESEARCH_DECISION_REQUIRED", "UNAVAILABLE", "NOT_APPLICABLE"}\n',
        '_DECISION_STATUS = {"LOCKED", "RESEARCH_DECISION_REQUIRED", "UNAVAILABLE", "NOT_APPLICABLE"}\n'
        'MODEL_ELIGIBILITY_VALUES = frozenset(\n'
        '    {"eligible", "eligible_observability_view", "blocked_until_mapping_review"}\n'
        ')\n'
        'FIT_MODEL_ELIGIBILITY_VALUES = frozenset({"eligible", "eligible_observability_view"})\n',
    )
    _replace(
        "src/features/service.py",
        '    if item["research_decision_status"] not in _DECISION_STATUS:\n'
        '        raise ValueError(f"feature={feature_id}: invalid research_decision_status")\n',
        '    if item["research_decision_status"] not in _DECISION_STATUS:\n'
        '        raise ValueError(f"feature={feature_id}: invalid research_decision_status")\n'
        '    if item["model_eligibility"] not in MODEL_ELIGIBILITY_VALUES:\n'
        '        raise ValueError(f"feature={feature_id}: invalid model_eligibility")\n',
    )
    _replace(
        "src/modeling/service.py",
        'from core.metrics import average_precision\n',
        'from core.metrics import average_precision\n'
        'from features.service import FIT_MODEL_ELIGIBILITY_VALUES\n',
    )
    _replace(
        "src/modeling/service.py",
        '    maximum_valid_configurations: int = 50,\n) -> ModelFitResult:\n',
        '    maximum_valid_configurations: int = 50,\n'
        '    required_feature_groups: list[str] | None = None,\n'
        '    candidate_seed_offset: int = 1009,\n'
        ') -> ModelFitResult:\n',
    )
    _replace(
        "src/modeling/service.py",
        '    groups = _feature_groups(feature_registry)\n'
        '    models: list[dict[str, object]] = []\n',
        '    groups = _feature_groups(feature_registry)\n'
        '    required_groups = set(required_feature_groups or [])\n'
        '    unknown_required = sorted(required_groups - set(groups))\n'
        '    if unknown_required:\n'
        '        raise ValueError(f"unknown required feature groups: {unknown_required}")\n'
        '    empty_required = sorted(group for group in required_groups if not groups[group])\n'
        '    if empty_required:\n'
        '        raise RuntimeError(f"required feature groups are empty: {empty_required}")\n'
        '    if candidate_seed_offset < 1:\n'
        '        raise ValueError("candidate_seed_offset must be positive")\n'
        '    models: list[dict[str, object]] = []\n',
    )
    _replace(
        "src/modeling/service.py",
        '                    random_state + candidate_index * 1009,\n',
        '                    random_state + candidate_index * candidate_seed_offset,\n',
    )
    _replace(
        "src/modeling/service.py",
        '        and item.get("model_eligibility") in {None, ELIGIBLE}\n',
        '        and item.get("model_eligibility") in {None, *FIT_MODEL_ELIGIBILITY_VALUES}\n',
    )
    _replace(
        "src/selection/nested_refit.py",
        'from modeling.service import ModelFitResult, fit_fold_models\n',
        'from features.service import FIT_MODEL_ELIGIBILITY_VALUES\n'
        'from modeling.service import ModelFitResult, fit_fold_models\n',
    )
    _replace(
        "src/selection/nested_refit.py",
        '    random_state: int,\n) -> NestedRefitResult:\n',
        '    random_state: int,\n'
        '    channel_seed_offset: int = 100_003,\n'
        '    candidate_seed_offset: int = 1009,\n'
        ') -> NestedRefitResult:\n',
    )
    _replace(
        "src/selection/nested_refit.py",
        '    if gate1_feature_group not in {"full", "content_only", "observability_only"}:\n'
        '        raise ValueError("unsupported Gate 1 feature group")\n',
        '    if gate1_feature_group not in {"full", "content_only", "observability_only"}:\n'
        '        raise ValueError("unsupported Gate 1 feature group")\n'
        '    if channel_seed_offset < 1 or candidate_seed_offset < 1:\n'
        '        raise ValueError("nested-refit seed offsets must be positive")\n',
    )
    _replace(
        "src/selection/nested_refit.py",
        '                    random_state=random_state + channel_index * 100_003,\n',
        '                    random_state=random_state + channel_index * channel_seed_offset,\n',
    )
    _replace(
        "src/selection/nested_refit.py",
        '                    maximum_valid_configurations=maximum_valid_configurations,\n'
        '                )\n',
        '                    maximum_valid_configurations=maximum_valid_configurations,\n'
        '                    required_feature_groups=[gate1_feature_group],\n'
        '                    candidate_seed_offset=candidate_seed_offset,\n'
        '                )\n',
    )
    _replace(
        "src/selection/nested_refit.py",
        '        if item.get("model_eligibility") not in {None, ELIGIBLE}:\n',
        '        if item.get("model_eligibility") not in {None, *FIT_MODEL_ELIGIBILITY_VALUES}:\n',
    )


def _patch_locked_runtime_parameters() -> None:
    _replace(
        "config/execution/reproducibility.yaml",
        "    container_policy: record_digest_or_explicit_absence\n",
        "    container_policy: record_digest_or_explicit_absence\n"
        "  seed_offsets:\n"
        "    nested_channel: 100003\n"
        "    tuning_candidate: 1009\n",
    )
    _replace(
        "config/methodology/calibration.yaml",
        "  in_sample_predictions_after_refit_forbidden: true\n",
        "  in_sample_predictions_after_refit_forbidden: true\n"
        "  platt:\n"
        "    inverse_regularization: 1000000.0\n"
        "    maximum_iterations: 2000\n",
    )
    _replace(
        "config/methodology/known_cases.yaml",
        "    below_median_cases: 3\n",
        "    below_median_cases: 3\n"
        "    weak_percentile: 0.5\n",
    )
    _replace(
        "config/methodology/evaluation.yaml",
        "  benchmark_ladder:\n"
        "  - fixed_accounting\n"
        "  - reestimated_linear\n"
        "  - observability_only\n"
        "  - content_only\n"
        "  - full\n"
        "  - nonlinear\n",
        "  benchmark_ladder:\n"
        "  - fixed_accounting\n"
        "  - reestimated_linear\n"
        "  - observability_only\n"
        "  - content_only\n"
        "  - full\n"
        "  - nonlinear\n"
        "  benchmark_status:\n"
        "    fixed_accounting:\n"
        "      status: BLOCKED_UNTIL_MAPPING_REVIEW\n"
        "      reason_code: BENEISH_VN_MAPPING_REVIEW_REQUIRED\n"
        "      operational_results_may_be_claimed: false\n",
    )
    _replace(
        "config/methodology/evaluation.yaml",
        "    initial_fold_role: separate\n",
        "    initial_fold_role: separate\n"
        "    breakpoint_grid:\n"
        "      lower_quantile: 0.1\n"
        "      upper_quantile: 0.9\n"
        "      points: 17\n"
        "    logistic_fit:\n"
        "      inverse_regularization: 1000000.0\n"
        "      maximum_iterations: 2000\n",
    )


def _patch_p11_and_p12() -> None:
    _replace(
        "scripts/p11_freeze_models.py",
        '    maximum_configurations = int(tuning["max_valid_configurations_per_learner_inner_fold"])\n'
        '    columns = physical_columns(loaded.registry)\n',
        '    maximum_configurations = int(tuning["max_valid_configurations_per_learner_inner_fold"])\n'
        '    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")\n'
        '    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")\n'
        '    reference_by_candidate = mapping(\n'
        '        gate2.get("reference_by_candidate"), "evaluation.gate2.reference_by_candidate"\n'
        '    )\n'
        '    required_feature_groups = sorted(\n'
        '        {\n'
        '            str(model_id).rsplit(":", 1)[-1]\n'
        '            for model_id in [*reference_by_candidate.keys(), *reference_by_candidate.values()]\n'
        '        }\n'
        '    )\n'
        '    reproducibility = mapping(loaded.registry.get("reproducibility"), "reproducibility")\n'
        '    seed_offsets = mapping(reproducibility.get("seed_offsets"), "reproducibility.seed_offsets")\n'
        '    candidate_seed_offset = int(seed_offsets["tuning_candidate"])\n'
        '    columns = physical_columns(loaded.registry)\n',
    )
    _replace(
        "scripts/p11_freeze_models.py",
        '        maximum_valid_configurations=maximum_configurations,\n'
        '    )\n'
        '    fits = [track_l1]\n',
        '        maximum_valid_configurations=maximum_configurations,\n'
        '        required_feature_groups=required_feature_groups,\n'
        '        candidate_seed_offset=candidate_seed_offset,\n'
        '    )\n'
        '    fits = [track_l1]\n',
    )
    _replace(
        "scripts/p11_freeze_models.py",
        '            maximum_valid_configurations=maximum_configurations,\n'
        '        )\n'
        '        fits.append(optional_fit)\n',
        '            maximum_valid_configurations=maximum_configurations,\n'
        '            candidate_seed_offset=candidate_seed_offset,\n'
        '        )\n'
        '        fits.append(optional_fit)\n',
    )
    _replace(
        "src/evaluation/service.py",
        '    latent_risk_scenarios: dict[str, list[dict[str, float]]] | None = None,\n'
        ') -> EvaluationResult:\n',
        '    latent_risk_scenarios: dict[str, list[dict[str, float]]] | None = None,\n'
        '    required_reference_by_candidate: dict[str, str] | None = None,\n'
        '    platt_inverse_regularization: float = 1_000_000.0,\n'
        '    platt_maximum_iterations: int = 2000,\n'
        ') -> EvaluationResult:\n',
    )
    _replace(
        "src/evaluation/service.py",
        '        intercept, slope, status = _fit_platt(\n'
        '            development_frame[prediction], development_frame[outcome]\n'
        '        )\n',
        '        intercept, slope, status = _fit_platt(\n'
        '            development_frame[prediction],\n'
        '            development_frame[outcome],\n'
        '            inverse_regularization=platt_inverse_regularization,\n'
        '            maximum_iterations=platt_maximum_iterations,\n'
        '        )\n',
    )
    _replace(
        "src/evaluation/service.py",
        '    comparisons = _comparisons(outer, calibrated, learner, outcome)\n',
        '    comparisons = _comparisons(\n'
        '        outer,\n'
        '        calibrated,\n'
        '        learner,\n'
        '        outcome,\n'
        '        required_reference_by_candidate=required_reference_by_candidate,\n'
        '    )\n',
    )
    _replace(
        "src/evaluation/service.py",
        'def _fit_platt(scores: pd.Series, outcomes: pd.Series) -> tuple[float, float, str]:\n',
        'def _fit_platt(\n'
        '    scores: pd.Series,\n'
        '    outcomes: pd.Series,\n'
        '    *,\n'
        '    inverse_regularization: float,\n'
        '    maximum_iterations: int,\n'
        ') -> tuple[float, float, str]:\n',
    )
    _replace(
        "src/evaluation/service.py",
        '    logits = _logit(scores.to_numpy(dtype=float)).reshape(-1, 1)\n'
        '    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)\n',
        '    if inverse_regularization <= 0 or maximum_iterations < 1:\n'
        '        raise ValueError("Platt calibration controls must be positive")\n'
        '    logits = _logit(scores.to_numpy(dtype=float)).reshape(-1, 1)\n'
        '    model = LogisticRegression(\n'
        '        C=inverse_regularization, solver="lbfgs", max_iter=maximum_iterations\n'
        '    )\n',
    )
    _replace(
        "src/evaluation/service.py",
        'def _comparisons(\n'
        '    outer: pd.DataFrame,\n'
        '    calibrated: dict[str, np.ndarray],\n'
        '    learner: str,\n'
        '    outcome: str,\n'
        ') -> list[dict[str, object]]:\n'
        '    results: list[dict[str, object]] = []\n'
        '    by_id = {str(value): frame for value, frame in outer.groupby(learner, sort=True)}\n'
        '    for full_id, full_frame in by_id.items():\n'
        '        if not full_id.endswith(":full"):\n'
        '            continue\n'
        '        reference_id = full_id.removesuffix(":full") + ":observability_only"\n'
        '        reference = by_id.get(reference_id)\n'
        '        if reference is None or len(reference) != len(full_frame):\n'
        '            continue\n',
        'def _comparisons(\n'
        '    outer: pd.DataFrame,\n'
        '    calibrated: dict[str, np.ndarray],\n'
        '    learner: str,\n'
        '    outcome: str,\n'
        '    *,\n'
        '    required_reference_by_candidate: dict[str, str] | None,\n'
        ') -> list[dict[str, object]]:\n'
        '    results: list[dict[str, object]] = []\n'
        '    by_id = {str(value): frame for value, frame in outer.groupby(learner, sort=True)}\n'
        '    pairs = required_reference_by_candidate or {\n'
        '        model_id: model_id.removesuffix(":full") + ":observability_only"\n'
        '        for model_id in by_id\n'
        '        if model_id.endswith(":full")\n'
        '    }\n'
        '    for full_id, reference_id in pairs.items():\n'
        '        full_frame = by_id.get(full_id)\n'
        '        reference = by_id.get(reference_id)\n'
        '        if full_frame is None or reference is None:\n'
        '            raise RuntimeError(\n'
        '                f"required Gate 2 model pair is missing: candidate={full_id}, reference={reference_id}"\n'
        '            )\n'
        '        if len(reference) != len(full_frame):\n'
        '            raise RuntimeError(\n'
        '                f"required Gate 2 model pair has unequal support: candidate={full_id}, reference={reference_id}"\n'
        '            )\n',
    )
    _replace(
        "scripts/p12_evaluate.py",
        '    review_budget = mapping(evaluation.get("review_budget"), "evaluation.review_budget")\n',
        '    review_budget = mapping(evaluation.get("review_budget"), "evaluation.review_budget")\n'
        '    gate2 = mapping(evaluation.get("gate2"), "evaluation.gate2")\n'
        '    reference_by_candidate = {\n'
        '        str(key): str(value)\n'
        '        for key, value in mapping(\n'
        '            gate2.get("reference_by_candidate"), "evaluation.gate2.reference_by_candidate"\n'
        '        ).items()\n'
        '    }\n'
        '    calibration = mapping(loaded.registry.get("calibration"), "calibration")\n'
        '    platt = mapping(calibration.get("platt"), "calibration.platt")\n',
    )
    _replace(
        "scripts/p12_evaluate.py",
        '        latent_risk_scenarios=latent_risk_scenarios,\n'
        '        target_id=primary_target_id,\n',
        '        latent_risk_scenarios=latent_risk_scenarios,\n'
        '        required_reference_by_candidate=reference_by_candidate,\n'
        '        platt_inverse_regularization=float(platt["inverse_regularization"]),\n'
        '        platt_maximum_iterations=int(platt["maximum_iterations"]),\n'
        '        target_id=primary_target_id,\n',
    )


def _patch_l3_contract() -> None:
    _replace(
        "config/methodology/measurement.yaml",
        "    operational:\n"
        "      fixed_pi_grid: []\n"
        "      accuracy_priors_by_profile: {}\n",
        "    operational:\n"
        "      parameter_status: PENDING_EXTERNAL_ELICITATION\n"
        "      report_required: false\n"
        "      unavailable_reason_code: L3_PARAMETERS_NOT_EXTERNALLY_JUSTIFIED\n"
        "      fixed_pi_grid: []\n"
        "      accuracy_priors_by_profile: {}\n",
    )
    _replace(
        "scripts/p05_measurement_inputs.py",
        '    operational = mapping(model.get("operational"), "measurement.l3_model.operational")\n'
        '    grid = sequence(operational.get("fixed_pi_grid"), "l3 fixed_pi_grid")\n',
        '    operational = mapping(model.get("operational"), "measurement.l3_model.operational")\n'
        '    parameter_status = str(operational.get("parameter_status", "UNSPECIFIED"))\n'
        '    report_required = bool(operational.get("report_required", False))\n'
        '    grid = sequence(operational.get("fixed_pi_grid"), "l3 fixed_pi_grid")\n',
    )
    _replace(
        "scripts/p05_measurement_inputs.py",
        '    if not grid:\n'
        '        capability.update(\n'
        '            {\n'
        '                "status": "EMPIRICALLY_PENDING",\n'
        '                "pilot_executed": False,\n'
        '                "reason_code": "L3_FIXED_PI_GRID_NOT_LOCKED",\n'
        '                "required_config_key": "measurement.l3_model.operational.fixed_pi_grid",\n'
        '            }\n'
        '        )\n'
        '        return\n',
        '    if not grid:\n'
        '        if report_required:\n'
        '            raise RuntimeError("L3 is report-required but fixed_pi_grid is empty")\n'
        '        capability.update(\n'
        '            {\n'
        '                "status": "EMPIRICALLY_PENDING",\n'
        '                "pilot_executed": False,\n'
        '                "reason_code": str(\n'
        '                    operational.get(\n'
        '                        "unavailable_reason_code", "L3_FIXED_PI_GRID_NOT_LOCKED"\n'
        '                    )\n'
        '                ),\n'
        '                "parameter_status": parameter_status,\n'
        '                "report_required": report_required,\n'
        '                "required_config_key": "measurement.l3_model.operational.fixed_pi_grid",\n'
        '            }\n'
        '        )\n'
        '        return\n',
    )
    _replace(
        "scripts/p05_measurement_inputs.py",
        '    if missing_profiles:\n'
        '        capability.update(\n',
        '    if missing_profiles:\n'
        '        if report_required:\n'
        '            raise RuntimeError(\n'
        '                f"L3 is report-required but accuracy priors are missing for {missing_profiles}"\n'
        '            )\n'
        '        capability.update(\n',
    )
    _replace(
        "scripts/p05_measurement_inputs.py",
        '                "required_config_key": "measurement.l3_model.operational.accuracy_priors_by_profile",\n'
        '            }\n',
        '                "required_config_key": "measurement.l3_model.operational.accuracy_priors_by_profile",\n'
        '                "parameter_status": parameter_status,\n'
        '                "report_required": report_required,\n'
        '            }\n',
    )


def _patch_temporal_and_data_contracts() -> None:
    _replace(
        "config/methodology/study.yaml",
        "    default_anchor: audited_financial_statement_publication_date\n"
        "    calendar_standardization: true\n"
        "    availability_date_audit_required: true\n",
        "    default_anchor: synthetic_annual_anchor_31_march_fiscal_year_plus_one\n"
        "    calendar_standardization: true\n"
        "    availability_date_audit_required: true\n"
        "    observed_publication_date_available: false\n"
        "    anchor_assumption_disclosure_required: true\n"
        "    sensitivity_anchors:\n"
        "    - synthetic_annual_anchor_30_june_fiscal_year_plus_one\n"
        "    - synthetic_annual_anchor_30_september_fiscal_year_plus_one\n",
    )
    _replace(
        "config/methodology/study.yaml",
        "  sample_fiscal_years:\n"
        "    start: 2015\n"
        "    end: 2026\n",
        "  sample_fiscal_years:\n"
        "    start: 2015\n"
        "    end: 2025\n"
        "  prospective_target_year: 2026\n",
    )
    _replace(
        "config/methodology/folds.yaml",
        "  prospective_year: 2026\n"
        "  initial_in_confirmatory_pool: false\n",
        "  prospective_year: 2026\n"
        "  prospective_feature_status: unavailable_by_data_cutoff\n"
        "  primary_embargo_years: 0\n"
        "  sensitivity_embargo_years:\n"
        "  - 1\n"
        "  initial_in_confirmatory_pool: false\n",
    )
    _replace(
        "config/methodology/features.yaml",
        "    availability_policy: synthetic_annual_anchor_allowed_but_not_observed_publication_date\n",
        "    availability_policy: synthetic_annual_anchor_explicit_assumption\n"
        "    availability_contract:\n"
        "      observed_publication_dates_available: false\n"
        "      primary_synthetic_anchor_month_day: 03-31\n"
        "      sensitivity_synthetic_anchor_month_days:\n"
        "      - 06-30\n"
        "      - 09-30\n"
        "      limitation_disclosure_required: true\n"
        "    revision_contract:\n"
        "      point_in_time_vintages_available: false\n"
        "      revision_policy_source: extract_provenance.revision_policy\n"
        "      restatement_sensitivity_required: true\n"
        "    preprocessing:\n"
        "      winsorization: none\n"
        "      p1_p99_role: diagnostics_only\n"
        "      train_fold_only_if_enabled: true\n",
    )
    _replace(
        "config/methodology/data_sources.yaml",
        "    hash_policy: locked_sha256_required\n"
        "    sources: {}\n",
        "    hash_policy: locked_sha256_required\n"
        "    provenance_contract:\n"
        "      manifest_relative_path: extract_provenance.json\n"
        "      required: true\n"
        "      required_fields:\n"
        "      - vendor\n"
        "      - vendor_product\n"
        "      - pull_date\n"
        "      - vendor_version\n"
        "      - extract_query\n"
        "      - revision_policy\n"
        "      - point_in_time_vintages_available\n"
        "    sources: {}\n",
    )


def _patch_snapshot_provenance() -> None:
    _replace(
        "src/snapshot/builder.py",
        '    catalog = SourceCatalog.from_mapping(registry.get("source_catalog"))\n'
        '    entries: list[dict[str, object]] = []\n',
        '    catalog = SourceCatalog.from_mapping(registry.get("source_catalog"))\n'
        '    extract_provenance = _load_extract_provenance(registry, raw_root)\n'
        '    entries: list[dict[str, object]] = []\n',
    )
    _replace(
        "src/snapshot/builder.py",
        '        "root_environment_variable": catalog.root_environment_variable,\n'
        '        "raw_root_recorded": False,\n'
        '        "profiles": profile_summary,\n',
        '        "root_environment_variable": catalog.root_environment_variable,\n'
        '        "raw_root_recorded": True,\n'
        '        "raw_root": str(raw_root),\n'
        '        "extract_provenance": extract_provenance,\n'
        '        "profiles": profile_summary,\n',
    )
    _replace(
        "src/snapshot/builder.py",
        '        "raw_root_recorded": snapshot.get("raw_root_recorded"),\n'
        '        "profiles": snapshot.get("profiles"),\n',
        '        "raw_root_recorded": snapshot.get("raw_root_recorded"),\n'
        '        "extract_provenance": snapshot.get("extract_provenance"),\n'
        '        "profiles": snapshot.get("profiles"),\n',
    )
    marker = '\n\ndef write_snapshot(path: Path, snapshot: dict[str, object]) -> None:\n'
    addition = '''\n\ndef _load_extract_provenance(\n    registry: dict[str, object], raw_root: Path\n) -> dict[str, object]:\n    data_sources = registry.get("data_sources")\n    if not isinstance(data_sources, dict):\n        return {}\n    source_registry = cast(dict[str, object], data_sources).get("source_registry")\n    if not isinstance(source_registry, dict):\n        return {}\n    contract = cast(dict[str, object], source_registry).get("provenance_contract")\n    if not isinstance(contract, dict):\n        return {}\n    typed = cast(dict[str, object], contract)\n    relative = typed.get("manifest_relative_path")\n    if not isinstance(relative, str) or not relative:\n        raise ValueError("data source provenance manifest path must be registered")\n    path = (raw_root / relative).resolve()\n    try:\n        path.relative_to(raw_root)\n    except ValueError as exc:\n        raise ValueError("data source provenance manifest escapes raw root") from exc\n    required = typed.get("required") is True\n    if not path.is_file():\n        if required:\n            raise FileNotFoundError(\n                f"required extract provenance manifest is missing: {path}"\n            )\n        return {}\n    raw: object = json.loads(path.read_text(encoding="utf-8"))\n    provenance = _mapping(raw, "extract provenance")\n    raw_fields = typed.get("required_fields")\n    if not isinstance(raw_fields, list) or not all(\n        isinstance(value, str) and value for value in cast(list[object], raw_fields)\n    ):\n        raise ValueError("extract provenance required_fields must be registered")\n    required_fields = cast(list[str], raw_fields)\n    missing = [\n        field\n        for field in required_fields\n        if field not in provenance or provenance[field] in {None, ""}\n    ]\n    if missing:\n        raise ValueError(f"extract provenance is missing required fields: {missing}")\n    return {str(key): value for key, value in provenance.items()}\n'''
    _append_once("src/snapshot/builder.py", marker, addition)
    _write(
        "templates/extract_provenance.template.json",
        json.dumps(
            {
                "vendor": "FiinPro",
                "vendor_product": "REPLACE_WITH_PRODUCT_NAME",
                "pull_date": "YYYY-MM-DD",
                "vendor_version": "REPLACE_WITH_VENDOR_VERSION_OR_EXPLICIT_UNKNOWN",
                "extract_query": "REPLACE_WITH_EXPORT_QUERY_OR_REPORT_IDENTIFIER",
                "revision_policy": "REPLACE_AFTER_VENDOR_CONFIRMATION",
                "point_in_time_vintages_available": False,
                "notes": "Copy this file to <raw-root>/extract_provenance.json before a production run.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )


def _patch_known_cases_and_resampling() -> None:
    _replace(
        "src/known_cases/service.py",
        '    strong_percentile: float,\n'
        '    columns: dict[str, str],\n',
        '    strong_percentile: float,\n'
        '    weak_percentile: float,\n'
        '    columns: dict[str, str],\n',
    )
    _replace(
        "src/known_cases/service.py",
        '                and float(item["percentile"]) < 0.5\n',
        '                and float(item["percentile"]) < weak_percentile\n',
    )
    _replace(
        "scripts/p15_open_known_cases.py",
        '        strong_percentile=float(veto["strong_falsification_all_below_percentile"]),\n'
        '        columns=physical_columns(loaded.registry),\n',
        '        strong_percentile=float(veto["strong_falsification_all_below_percentile"]),\n'
        '        weak_percentile=float(veto["weak_percentile"]),\n'
        '        columns=physical_columns(loaded.registry),\n',
    )
    _replace(
        "src/simulation/service.py",
        '    except Exception as exc:\n'
        '        record_resampling_failure(type(exc).__name__)\n'
        '        return x, y, selection_weights, neutral_stats, 0.0\n',
        '    except (ImportError, ValueError, RuntimeError) as exc:\n'
        '        record_resampling_failure(type(exc).__name__)\n'
        '        return x, y, selection_weights, neutral_stats, 0.0\n',
    )


def _patch_config_validation() -> None:
    _replace(
        "src/core/config_validation.py",
        '    simulation = _mapping(registry, "simulation")\n'
        '    s3_taxonomy = _mapping(registry, "s3_taxonomy")\n',
        '    simulation = _mapping(registry, "simulation")\n'
        '    s3_taxonomy = _mapping(registry, "s3_taxonomy")\n'
        '    reproducibility = _mapping(registry, "reproducibility")\n',
    )
    _replace(
        "src/core/config_validation.py",
        '    protocol_role = _entry(\n'
        '        simulation.get("protocol_role"),\n'
        '        "simulation.protocol_role",\n'
        '    )\n',
        '    protocol_role = _entry(\n'
        '        simulation.get("protocol_role"),\n'
        '        "simulation.protocol_role",\n'
        '    )\n'
        '    prediction_time = _entry(study.get("prediction_time"), "study.prediction_time")\n'
        '    sample_years = _entry(study.get("sample_fiscal_years"), "study.sample_fiscal_years")\n'
        '    feature_store = _entry(features.get("store"), "features.store")\n'
        '    availability_contract = _entry(\n'
        '        feature_store.get("availability_contract"), "features.store.availability_contract"\n'
        '    )\n'
        '    revision_contract = _entry(\n'
        '        feature_store.get("revision_contract"), "features.store.revision_contract"\n'
        '    )\n'
        '    preprocessing = _entry(\n'
        '        feature_store.get("preprocessing"), "features.store.preprocessing"\n'
        '    )\n'
        '    l3_operational = _entry(l3_model.get("operational"), "measurement.l3_model.operational")\n'
        '    execution_tracks = _entry(\n'
        '        measurement.get("execution_tracks"), "measurement.execution_tracks"\n'
        '    )\n'
        '    source_registry = _entry(\n'
        '        sources.get("source_registry"), "data_sources.source_registry"\n'
        '    )\n'
        '    provenance_contract = _entry(\n'
        '        source_registry.get("provenance_contract"),\n'
        '        "data_sources.source_registry.provenance_contract",\n'
        '    )\n'
        '    benchmark_status = _entry(\n'
        '        evaluation.get("benchmark_status"), "evaluation.benchmark_status"\n'
        '    )\n'
        '    fixed_accounting_status = _entry(\n'
        '        benchmark_status.get("fixed_accounting"),\n'
        '        "evaluation.benchmark_status.fixed_accounting",\n'
        '    )\n'
        '    platt = _entry(calibration.get("platt"), "calibration.platt")\n'
        '    gate3 = _entry(evaluation.get("gate3"), "evaluation.gate3")\n'
        '    breakpoint_grid = _entry(gate3.get("breakpoint_grid"), "evaluation.gate3.breakpoint_grid")\n'
        '    gate3_logistic = _entry(gate3.get("logistic_fit"), "evaluation.gate3.logistic_fit")\n'
        '    seed_offsets = _entry(\n'
        '        reproducibility.get("seed_offsets"), "reproducibility.seed_offsets"\n'
        '    )\n',
    )
    _replace(
        "src/core/config_validation.py",
        '        (\n'
        '            review_budget.get("primary_fraction") == 0.05,\n'
        '            "D07 primary review budget must be 5%",\n'
        '        ),\n',
        '        # D07 is intentionally hard-guarded so protocol drift requires an explicit amendment.\n'
        '        (\n'
        '            review_budget.get("primary_fraction") == 0.05,\n'
        '            "D07 primary review budget must be 5%",\n'
        '        ),\n',
    )
    _replace(
        "src/core/config_validation.py",
        '        (\n'
        '            tuning.get("max_valid_configurations_per_learner_inner_fold") == 50,\n'
        '            "D09 tuning cap must be 50",\n'
        '        ),\n',
        '        # D09 is intentionally hard-guarded so protocol drift requires an explicit amendment.\n'
        '        (\n'
        '            tuning.get("max_valid_configurations_per_learner_inner_fold") == 50,\n'
        '            "D09 tuning cap must be 50",\n'
        '        ),\n',
    )
    _replace(
        "src/core/config_validation.py",
        '        (\n'
        '            folds.get("initial_outer_year") == 2020\n'
        '            and folds.get("fully_nested_outer_years") == [2021, 2022, 2023, 2024],\n'
        '            "D10 fold calendar differs",\n'
        '        ),\n',
        '        # D10 is intentionally hard-guarded so protocol drift requires an explicit amendment.\n'
        '        (\n'
        '            folds.get("initial_outer_year") == 2020\n'
        '            and folds.get("fully_nested_outer_years") == [2021, 2022, 2023, 2024],\n'
        '            "D10 fold calendar differs",\n'
        '        ),\n',
    )
    anchor = '        (\n            measurement.get("selection_candidates") == ["L2", "L3_fixed_pi", "none"],\n            "hierarchical-pi cannot enter Gate 1",\n        ),\n'
    addition = '''        (\n            prediction_time.get("default_anchor")\n            == "synthetic_annual_anchor_31_march_fiscal_year_plus_one"\n            and prediction_time.get("observed_publication_date_available") is False\n            and prediction_time.get("anchor_assumption_disclosure_required") is True,\n            "D01 synthetic annual anchor must be explicit and disclosed",\n        ),\n        (\n            availability_contract.get("observed_publication_dates_available") is False\n            and availability_contract.get("primary_synthetic_anchor_month_day") == "03-31"\n            and availability_contract.get("sensitivity_synthetic_anchor_month_days")\n            == ["06-30", "09-30"],\n            "feature availability anchor contract differs",\n        ),\n        (\n            revision_contract.get("point_in_time_vintages_available") is False\n            and revision_contract.get("restatement_sensitivity_required") is True,\n            "point-in-time limitation and restatement sensitivity must be explicit",\n        ),\n        (\n            preprocessing.get("winsorization") == "none"\n            and preprocessing.get("p1_p99_role") == "diagnostics_only"\n            and preprocessing.get("train_fold_only_if_enabled") is True,\n            "feature preprocessing policy must be explicit and leakage-safe",\n        ),\n        (\n            sample_years.get("end") == feature_store.get("allowed_fiscal_year_max")\n            and folds.get("prospective_year") == study.get("prospective_target_year")\n            and int(folds.get("prospective_year", 0)) == int(sample_years.get("end", 0)) + 1\n            and folds.get("prospective_feature_status") == "unavailable_by_data_cutoff"\n            and folds.get("prospective_in_retrospective_evaluation") is False,\n            "sample and prospective year ownership differs",\n        ),\n        (\n            folds.get("primary_embargo_years") == 0\n            and folds.get("sensitivity_embargo_years") == [1],\n            "one-year embargo sensitivity must be registered",\n        ),\n        (\n            (\n                bool(l3_operational.get("fixed_pi_grid"))\n                and bool(l3_operational.get("accuracy_priors_by_profile"))\n            )\n            or (\n                l3_operational.get("parameter_status") == "PENDING_EXTERNAL_ELICITATION"\n                and l3_operational.get("report_required") is False\n                and "L3_fixed_pi" in _string_list(\n                    execution_tracks.get("order"), "measurement.execution_tracks.order"\n                )\n            ),\n            "L3 parameters must be locked or explicitly pending and non-reportable",\n        ),\n        (\n            provenance_contract.get("required") is True\n            and provenance_contract.get("manifest_relative_path") == "extract_provenance.json"\n            and set(\n                _string_list(\n                    provenance_contract.get("required_fields"),\n                    "data source provenance required_fields",\n                )\n            )\n            >= {\n                "vendor",\n                "vendor_product",\n                "pull_date",\n                "vendor_version",\n                "extract_query",\n                "revision_policy",\n                "point_in_time_vintages_available",\n            },\n            "data extract provenance must be fail-closed",\n        ),\n        (\n            fixed_accounting_status.get("status") == "BLOCKED_UNTIL_MAPPING_REVIEW"\n            and fixed_accounting_status.get("operational_results_may_be_claimed") is False,\n            "fixed-accounting benchmark cannot be claimed before Beneish mapping review",\n        ),\n        (\n            float(platt.get("inverse_regularization", 0.0)) > 0\n            and int(platt.get("maximum_iterations", 0)) > 0\n            and breakpoint_grid.get("lower_quantile") == 0.1\n            and breakpoint_grid.get("upper_quantile") == 0.9\n            and breakpoint_grid.get("points") == 17\n            and float(gate3_logistic.get("inverse_regularization", 0.0)) > 0\n            and int(gate3_logistic.get("maximum_iterations", 0)) > 0\n            and int(seed_offsets.get("nested_channel", 0)) > 0\n            and int(seed_offsets.get("tuning_candidate", 0)) > 0,\n            "locked calibration, breakpoint, and seed controls differ",\n        ),\n'''
    _append_once("src/core/config_validation.py", anchor, addition)


def _write_tests_and_report() -> None:
    _write(
        "tests/test_audit_remediation_contracts.py",
        '''from __future__ import annotations\n\nfrom pathlib import Path\nfrom typing import Any, cast\n\nimport pytest\n\nfrom core.registry_compiler import compile_registry\nfrom features.service import _validate_definition\nfrom modeling.service import _feature_groups\n\nROOT = Path(__file__).resolve().parents[1]\n\n\ndef _registry() -> dict[str, object]:\n    return cast(dict[str, object], compile_registry(ROOT / "config" / "pipeline.yaml").registry)\n\n\ndef test_observability_features_enter_reference_and_full_groups() -> None:\n    registry = _registry()\n    features = cast(dict[str, Any], registry["features"])\n    definitions = cast(list[dict[str, Any]], features["registry"])\n    groups = _feature_groups(definitions)\n    assert groups["observability_only"]\n    assert set(groups["observability_only"]).issubset(groups["full"])\n    assert set(groups["full"]) != set(groups["content_only"])\n\n\ndef test_model_eligibility_is_a_closed_enum() -> None:\n    registry = _registry()\n    features = cast(dict[str, Any], registry["features"])\n    definition = dict(cast(list[dict[str, Any]], features["registry"])[0])\n    definition["model_eligibility"] = "silent_unknown_value"\n    with pytest.raises(ValueError, match="invalid model_eligibility"):\n        _validate_definition(definition)\n\n\ndef test_l3_unlocked_parameters_are_explicitly_nonreportable() -> None:\n    registry = _registry()\n    measurement = cast(dict[str, Any], registry["measurement"])\n    operational = cast(dict[str, Any], cast(dict[str, Any], measurement["l3_model"])["operational"])\n    assert operational["fixed_pi_grid"] == []\n    assert operational["accuracy_priors_by_profile"] == {}\n    assert operational["parameter_status"] == "PENDING_EXTERNAL_ELICITATION"\n    assert operational["report_required"] is False\n\n\ndef test_temporal_and_provenance_contracts_are_explicit() -> None:\n    registry = _registry()\n    study = cast(dict[str, Any], registry["study"])\n    folds = cast(dict[str, Any], registry["folds"])\n    features = cast(dict[str, Any], registry["features"])\n    store = cast(dict[str, Any], features["store"])\n    assert cast(dict[str, Any], study["sample_fiscal_years"])["end"] == 2025\n    assert store["allowed_fiscal_year_max"] == 2025\n    assert folds["prospective_year"] == 2026\n    assert folds["prospective_feature_status"] == "unavailable_by_data_cutoff"\n    assert cast(dict[str, Any], study["prediction_time"])["observed_publication_date_available"] is False\n    provenance = cast(\n        dict[str, Any],\n        cast(dict[str, Any], cast(dict[str, Any], registry["data_sources"])["source_registry"])[\n            "provenance_contract"\n        ],\n    )\n    assert provenance["required"] is True\n    assert "revision_policy" in provenance["required_fields"]\n\n\ndef test_stale_backup_source_is_absent() -> None:\n    assert not (ROOT / "data" / "source" / "firm_event_sanction_panel_backup.csv").exists()\n''',
    )
    _write(
        "docs/AUDIT_REMEDIATION_A01_A15.md",
        '''# A-01--A-15 methodological audit remediation\n\nThis note records the disposition of the external audit against the current `main` branch.\n\n- **A-01 fixed.** `eligible_observability_view` is accepted by the production and nested-refit feature selectors; model eligibility is now a closed enum; P11 and P12 fail closed when the Gate 2 candidate/reference groups are absent.\n- **A-02 controlled, not assumed away.** Production snapshots now require an extract-provenance manifest containing the vendor revision policy and whether point-in-time vintages exist. The locked feature contract states that point-in-time vintages are currently unavailable and that a restatement sensitivity is required. No claim is made that a current FiinPro snapshot is point-in-time.\n- **A-03 fixed as a fail-closed reporting contract.** L3 parameters remain empty because no external elicitation has yet justified them. The registry marks L3 `PENDING_EXTERNAL_ELICITATION` and non-reportable. A future change to `report_required: true` fails unless fixed-pi values and accuracy priors are locked.\n- **A-04 fixed as an explicit assumption.** The study no longer calls 31 March an observed publication date. It is registered as a synthetic annual anchor, with 30 June and 30 September sensitivity anchors. Implementing those sensitivity runs remains required before the final paper.\n- **A-05 fixed at the snapshot boundary.** Snapshots record the raw-root locator and a required extract-provenance manifest. The manifest is included in the source-content hash; moving the same raw files does not alter that content hash.\n- **A-06 hardened.** The current SMOTE/ADASYN regression test passes. The remaining broad exception was narrowed to expected import/value/runtime failures; diagnostic collectors retain the concrete failure class and affected replication IDs.\n- **A-07 fixed.** The retrospective feature sample ends in 2025. The 2026 year is separately registered as prospective-only and unavailable by the current data cutoff; it is not included in retrospective outer folds.\n- **A-08 registered but not falsely claimed complete.** The primary split has zero embargo and a one-year embargo sensitivity is now part of the locked protocol. The final paper still requires the corresponding refit/AP comparison; registration alone is not a result.\n- **A-09 retained by design.** Rolling panel forecasts may reuse earlier observations from the same firm. Firm-clustered bootstrap remains the inferential unit; final descriptive output must report both firm-years and distinct firms.\n- **A-10 declared explicitly.** No winsorization is applied. P1/P99 values are diagnostic only. Any future winsorization must be fitted inside the development fold.\n- **A-11 blocked from claims.** The fixed-accounting/Beneish benchmark is explicitly non-operational until the Vietnam mapping review is complete.\n- **A-12 partially eliminated.** Platt regularization/iterations, Gate 3 breakpoint grid/logistic controls, known-case weak percentile, and seed offsets are protocol configuration. Existing D07/D09/D10 hard guards are retained intentionally.\n- **A-13 retained.** Comments now identify D07/D09/D10 hard coding as deliberate preregistration guards.\n- **A-14 already resolved.** The backup sanction CSV is absent from `main`.\n- **A-15 deferred.** Long-function decomposition is maintainability work, not a methodological correction, and should be split into behavior-preserving PRs after the blockers above are stable.\n''',
    )


def main() -> None:
    _patch_feature_eligibility()
    _patch_locked_runtime_parameters()
    _patch_p11_and_p12()
    _patch_l3_contract()
    _patch_temporal_and_data_contracts()
    _patch_snapshot_provenance()
    _patch_known_cases_and_resampling()
    _patch_config_validation()
    _write_tests_and_report()


if __name__ == "__main__":
    main()
