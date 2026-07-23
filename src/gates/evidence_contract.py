"""Fail-closed evidence preparation for the locked Gate 2 family."""

from __future__ import annotations

import math
from typing import Any, cast

from core.semantic_keys import LEARNER_ID, OUTER_FOLD

JsonRow = dict[str, Any]


def prepare_gate2_inputs(
    *,
    evaluations: list[JsonRow],
    bootstraps: list[list[JsonRow]],
    gate: JsonRow,
    confirmatory_folds: list[str],
) -> dict[str, object]:
    """Validate and filter Gate 2 inputs to the preregistered candidate family."""
    blockers: list[str] = []
    candidate_ids = _string_list(gate.get("candidate_ids"), "candidate_ids", blockers)
    reference_by_candidate = _string_mapping(
        gate.get("reference_by_candidate"), "reference_by_candidate", blockers
    )
    expected_family_count = gate.get("candidate_family_count")
    minimum_bootstrap = gate.get("minimum_valid_bootstrap_replications")
    fold_count = gate.get("fold_count")

    if not isinstance(expected_family_count, int) or isinstance(expected_family_count, bool):
        blockers.append("GATE2_CANDIDATE_FAMILY_COUNT_INVALID")
    elif expected_family_count < 1:
        blockers.append("GATE2_CANDIDATE_FAMILY_COUNT_INVALID")
    elif len(candidate_ids) != expected_family_count:
        blockers.append("GATE2_CANDIDATE_FAMILY_COUNT_MISMATCH")
    if set(reference_by_candidate) != set(candidate_ids):
        blockers.append("GATE2_REFERENCE_BINDINGS_INCOMPLETE")
    minimum_bootstrap_count: int | None = None
    if not isinstance(minimum_bootstrap, int) or isinstance(minimum_bootstrap, bool):
        blockers.append("GATE2_MINIMUM_BOOTSTRAP_INVALID")
    elif minimum_bootstrap < 1:
        blockers.append("GATE2_MINIMUM_BOOTSTRAP_INVALID")
    else:
        minimum_bootstrap_count = minimum_bootstrap
    if not isinstance(fold_count, int) or isinstance(fold_count, bool):
        blockers.append("GATE2_CONFIRMATORY_FOLD_COUNT_MISMATCH")
    elif fold_count != len(confirmatory_folds):
        blockers.append("GATE2_CONFIRMATORY_FOLD_COUNT_MISMATCH")
    if len(set(confirmatory_folds)) != len(confirmatory_folds):
        blockers.append("GATE2_CONFIRMATORY_FOLDS_NOT_UNIQUE")

    evaluations_by_fold: dict[str, JsonRow] = {}
    for evaluation in evaluations:
        fold = str(evaluation.get(OUTER_FOLD, ""))
        if fold in evaluations_by_fold:
            blockers.append(f"GATE2_DUPLICATE_EVALUATION_FOLD:{fold}")
        evaluations_by_fold[fold] = evaluation
    if set(evaluations_by_fold) != set(confirmatory_folds):
        blockers.append("GATE2_EVALUATION_FOLD_COVERAGE_INCOMPLETE")

    bootstrap_by_fold: dict[str, list[JsonRow]] = {}
    for batch in bootstraps:
        folds = {str(item.get(OUTER_FOLD, "")) for item in batch}
        if len(folds) != 1:
            blockers.append("GATE2_BOOTSTRAP_BATCH_FOLD_AMBIGUOUS")
            continue
        fold = next(iter(folds))
        if fold in bootstrap_by_fold:
            blockers.append(f"GATE2_DUPLICATE_BOOTSTRAP_FOLD:{fold}")
        bootstrap_by_fold[fold] = batch
    if set(bootstrap_by_fold) != set(confirmatory_folds):
        blockers.append("GATE2_BOOTSTRAP_FOLD_COVERAGE_INCOMPLETE")

    filtered_evaluations: list[JsonRow] = []
    filtered_bootstraps: list[list[JsonRow]] = []
    for fold in confirmatory_folds:
        evaluation = evaluations_by_fold.get(fold)
        if evaluation is None:
            continue
        raw_models = evaluation.get("models")
        raw_comparisons = evaluation.get("comparisons")
        if not isinstance(raw_models, list) or not isinstance(raw_comparisons, list):
            blockers.append(f"GATE2_EVALUATION_STRUCTURE_INVALID:{fold}")
            continue

        model_rows = _dict_rows(cast(list[object], raw_models))
        comparison_rows = _dict_rows(cast(list[object], raw_comparisons))
        model_index = _unique_index(model_rows, LEARNER_ID, fold, "MODEL", blockers)
        comparison_index = _unique_index(comparison_rows, "candidate", fold, "COMPARISON", blockers)
        selected_models: list[JsonRow] = []
        selected_comparisons: list[JsonRow] = []
        for candidate in candidate_ids:
            reference = reference_by_candidate.get(candidate)
            if reference is None:
                blockers.append(f"GATE2_REFERENCE_BINDING_MISSING:{fold}:{candidate}")
                continue
            candidate_model = model_index.get(candidate)
            reference_model = model_index.get(reference)
            comparison = comparison_index.get(candidate)
            if candidate_model is None or reference_model is None or comparison is None:
                blockers.append(f"GATE2_REQUIRED_PAIR_MISSING:{fold}:{candidate}")
                continue
            if str(comparison.get("reference")) != reference:
                blockers.append(f"GATE2_REFERENCE_MISMATCH:{fold}:{candidate}")
            if _finite_float(comparison.get("delta_average_precision")) is None:
                blockers.append(f"GATE2_DELTA_AP_MISSING:{fold}:{candidate}")
            for model_id, model in (
                (candidate, candidate_model),
                (reference, reference_model),
            ):
                if _finite_float(model.get("average_precision")) is None:
                    blockers.append(f"GATE2_AP_MISSING:{fold}:{model_id}")
                if _finite_float(model.get("precision_at_primary_budget")) is None:
                    blockers.append(f"GATE2_YIELD_MISSING:{fold}:{model_id}")
            reference_precision = _finite_float(reference_model.get("precision_at_primary_budget"))
            if reference_precision is not None and reference_precision <= 0.0:
                blockers.append(f"GATE2_REFERENCE_YIELD_NONPOSITIVE:{fold}:{candidate}")
            positives = candidate_model.get("positives")
            if not isinstance(positives, int) or isinstance(positives, bool):
                blockers.append(f"GATE2_POSITIVE_COUNT_MISSING:{fold}:{candidate}")
            selected_models.extend([candidate_model, reference_model])
            selected_comparisons.append(comparison)
        filtered_evaluations.append(
            {
                **evaluation,
                "models": _deduplicate_models(selected_models),
                "comparisons": selected_comparisons,
            }
        )

        batch = bootstrap_by_fold.get(fold)
        if batch is None:
            continue
        bootstrap_index = _unique_index(batch, "candidate", fold, "BOOTSTRAP", blockers)
        selected_bootstraps: list[JsonRow] = []
        for candidate in candidate_ids:
            reference = reference_by_candidate.get(candidate)
            if reference is None:
                blockers.append(f"GATE2_BOOTSTRAP_REFERENCE_BINDING_MISSING:{fold}:{candidate}")
                continue
            item = bootstrap_index.get(candidate)
            if item is None:
                blockers.append(f"GATE2_BOOTSTRAP_PAIR_MISSING:{fold}:{candidate}")
                continue
            if str(item.get("reference")) != reference:
                blockers.append(f"GATE2_BOOTSTRAP_REFERENCE_MISMATCH:{fold}:{candidate}")
            samples = item.get("delta_ap_samples")
            if not isinstance(samples, list):
                blockers.append(f"GATE2_BOOTSTRAP_SAMPLES_MISSING:{fold}:{candidate}")
            else:
                sample_values = cast(list[object], samples)
                if (
                    minimum_bootstrap_count is not None
                    and len(sample_values) < minimum_bootstrap_count
                ):
                    blockers.append(f"GATE2_BOOTSTRAP_TOO_SHORT:{fold}:{candidate}")
                if any(_finite_float(value) is None for value in sample_values):
                    blockers.append(f"GATE2_BOOTSTRAP_NONFINITE:{fold}:{candidate}")
            selected_bootstraps.append(item)
        filtered_bootstraps.append(selected_bootstraps)

    return {
        "evaluations": filtered_evaluations,
        "bootstraps": filtered_bootstraps,
        "candidate_ids": candidate_ids,
        "reference_by_candidate": reference_by_candidate,
        "blockers": sorted(set(blockers)),
    }


def insufficient_gate2_verdict(blockers: list[str], candidate_ids: list[str]) -> dict[str, object]:
    return {
        "status": "INSUFFICIENT_EVIDENCE",
        "gate_id": "GATE2",
        "verdict": "INSUFFICIENT_EVIDENCE",
        "candidates": [],
        "locked_candidate_ids": candidate_ids,
        "domain_robust_scenario_fraction": None,
        "evidence_complete": False,
        "reason_code": "REQUIRED_GATE2_EVIDENCE_INCOMPLETE",
        "evidence_blockers": blockers,
        "criteria_source": "locked_registry",
    }


def _dict_rows(value: list[object]) -> list[JsonRow]:
    rows: list[JsonRow] = []
    for raw_row in value:
        if isinstance(raw_row, dict):
            rows.append(cast(JsonRow, raw_row))
    return rows


def _string_list(value: object, name: str, blockers: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        blockers.append(f"GATE2_{name.upper()}_INVALID")
        return []
    parsed: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str) or not item:
            blockers.append(f"GATE2_{name.upper()}_INVALID")
            return []
        parsed.append(item)
    return parsed


def _string_mapping(value: object, name: str, blockers: list[str]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        blockers.append(f"GATE2_{name.upper()}_INVALID")
        return {}
    parsed: dict[str, str] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
            blockers.append(f"GATE2_{name.upper()}_INVALID")
            return {}
        parsed[key] = item
    return parsed


def _unique_index(
    rows: list[JsonRow],
    key: str,
    fold: str,
    role: str,
    blockers: list[str],
) -> dict[str, JsonRow]:
    index: dict[str, JsonRow] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if value in index:
            blockers.append(f"GATE2_DUPLICATE_{role}:{fold}:{value}")
        index[value] = row
    return index


def _deduplicate_models(rows: list[JsonRow]) -> list[JsonRow]:
    unique: dict[str, JsonRow] = {}
    for row in rows:
        unique[str(row.get(LEARNER_ID, ""))] = row
    return [unique[key] for key in sorted(unique)]


def _finite_float(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None
