"""Full-refit channel-within-time measurement selection for P10."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from core.semantic_keys import (
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    OUTER_FOLD,
    PREDICTION,
    TARGET_VALUE,
)
from modeling.service import ModelFitResult, fit_fold_models

_HELDOUT_CHANNEL = "heldout_channel"
_REMOVED_INPUTS = ("target", "label_model", "features", "tuning", "calibration")


@dataclass(frozen=True)
class NestedRefitResult:
    candidate_results: list[dict[str, object]]
    receipt: dict[str, object]


def run_nested_channel_refit(
    *,
    matrices: Mapping[str, object],
    feature_panel: pd.DataFrame,
    feature_registry: Sequence[Mapping[str, object]],
    label_inputs: pd.DataFrame,
    weights: pd.DataFrame,
    outer_year: int,
    candidates: Sequence[str],
    l3_fold_result: Mapping[str, object] | None,
    minimum_observed_channels: int,
    gate1_learner_id: str,
    gate1_feature_group: str,
    learner_settings: Mapping[str, object],
    learner_search_spaces: Mapping[str, object],
    maximum_valid_configurations: int,
    columns: Mapping[str, str],
    random_state: int,
) -> NestedRefitResult:
    """Run the complete development-only learner procedure for each held channel."""
    if minimum_observed_channels < 1:
        raise ValueError("nested channel refit requires positive channel coverage")
    if maximum_valid_configurations < 1:
        raise ValueError("nested channel refit requires a positive tuning budget")
    if gate1_feature_group not in {"full", "content_only", "observability_only"}:
        raise ValueError("unsupported Gate 1 feature group")

    year = _physical(columns, FISCAL_YEAR)
    expected_channels = _string_list(matrices.get("expected_channels"), "expected_channels")
    rows = _matrix_rows(matrices)
    requested_candidates = [str(value) for value in candidates if str(value) != "none"]
    if len(expected_channels) < 2:
        candidate_results = [
            {
                "candidate": candidate,
                ELIGIBLE: False,
                "objective": None,
                "reason_code": "INSUFFICIENT_CHANNELS",
            }
            for candidate in requested_candidates
        ]
        return NestedRefitResult(
            candidate_results=candidate_results,
            receipt=_receipt(
                outer_year=outer_year,
                gate1_learner_id=gate1_learner_id,
                gate1_feature_group=gate1_feature_group,
                expected_channels=expected_channels,
                cell_results=[],
                candidate_results=candidate_results,
                status="INSUFFICIENT_EVIDENCE",
                reason_code="INSUFFICIENT_CHANNELS",
            ),
        )

    development_panel = feature_panel.loc[feature_panel[year] < outer_year].copy()
    development_weights = weights.loc[weights[year] < outer_year].copy()
    development_labels = label_inputs.loc[label_inputs[year] < outer_year].copy()
    if development_panel.empty:
        raise ValueError("nested channel refit has no development feature rows")
    if int(development_panel[year].max()) >= outer_year:
        raise ValueError("nested channel refit accessed outer-year features")
    if not development_weights.empty and int(development_weights[year].max()) >= outer_year:
        raise ValueError("nested channel refit accessed outer-year weights")

    cell_results: list[dict[str, object]] = []
    candidate_results: list[dict[str, object]] = []
    for candidate in requested_candidates:
        candidate_cells: list[dict[str, object]] = []
        for channel_index, heldout_channel in enumerate(expected_channels):
            target_frame = _candidate_target_frame(
                candidate=candidate,
                heldout_channel=heldout_channel,
                rows=rows,
                outer_year=outer_year,
                minimum_observed_channels=minimum_observed_channels,
                l3_fold_result=l3_fold_result,
                columns=columns,
            )
            heldout_outcomes = _heldout_outcome_frame(
                rows=rows,
                heldout_channel=heldout_channel,
                outer_year=outer_year,
                columns=columns,
            )
            filtered_registry = _heldout_feature_registry(
                feature_registry=feature_registry,
                feature_panel=development_panel,
                heldout_channel=heldout_channel,
            )
            eligible_features = _features_for_group(filtered_registry, gate1_feature_group)
            if target_frame.empty:
                cell = _failed_cell(
                    candidate,
                    heldout_channel,
                    "CANDIDATE_TARGET_UNAVAILABLE_AFTER_CHANNEL_HOLDOUT",
                    len(eligible_features),
                )
            elif heldout_outcomes.empty:
                cell = _failed_cell(
                    candidate,
                    heldout_channel,
                    "HELDOUT_CHANNEL_OUTCOMES_UNAVAILABLE",
                    len(eligible_features),
                )
            elif not eligible_features:
                cell = _failed_cell(
                    candidate,
                    heldout_channel,
                    "NO_FEATURES_AFTER_CHANNEL_HOLDOUT",
                    0,
                )
            else:
                fit = fit_fold_models(
                    feature_panel=development_panel,
                    feature_registry=filtered_registry,
                    label_inputs=development_labels,
                    weights=development_weights,
                    outer_year=outer_year,
                    learner_ids=[gate1_learner_id],
                    learner_settings={str(key): value for key, value in learner_settings.items()},
                    target_id=candidate,
                    measurement_id=candidate,
                    columns={str(key): str(value) for key, value in columns.items()},
                    random_state=random_state + channel_index * 100_003,
                    target_values=target_frame,
                    soft_target=True,
                    target_transform="training_ecdf" if candidate == "L2" else "identity",
                    track_id=f"gate1_{_token(candidate)}_{_token(heldout_channel)}",
                    learner_search_spaces={
                        str(key): value for key, value in learner_search_spaces.items()
                    },
                    maximum_valid_configurations=maximum_valid_configurations,
                )
                cell = _score_fit(
                    fit=fit,
                    candidate=candidate,
                    heldout_channel=heldout_channel,
                    heldout_outcomes=heldout_outcomes,
                    gate1_learner_id=gate1_learner_id,
                    gate1_feature_group=gate1_feature_group,
                    eligible_feature_count=len(eligible_features),
                    columns=columns,
                )
            candidate_cells.append(cell)
            cell_results.append(cell)

        complete = bool(candidate_cells) and all(
            item.get("status") == "PASS" for item in candidate_cells
        )
        objectives = [
            float(item["soft_cross_entropy"])
            for item in candidate_cells
            if item.get("soft_cross_entropy") is not None
        ]
        candidate_results.append(
            {
                "candidate": candidate,
                ELIGIBLE: complete,
                "objective": float(np.mean(objectives)) if complete and objectives else None,
                "reason_code": None if complete else "INCOMPLETE_NESTED_CHANNEL_GRID",
                "required_heldout_channels": len(expected_channels),
                "completed_heldout_channels": sum(
                    item.get("status") == "PASS" for item in candidate_cells
                ),
                "selection_procedure": "full_refit_channel_within_time",
                "gate1_learner_id": gate1_learner_id,
                "gate1_feature_group": gate1_feature_group,
            }
        )

    status = (
        "PASS"
        if any(item.get(ELIGIBLE) is True for item in candidate_results)
        else "INSUFFICIENT_EVIDENCE"
    )
    reason_code = None if status == "PASS" else "NO_COMPLETE_NESTED_CANDIDATE"
    receipt = _receipt(
        outer_year=outer_year,
        gate1_learner_id=gate1_learner_id,
        gate1_feature_group=gate1_feature_group,
        expected_channels=expected_channels,
        cell_results=cell_results,
        candidate_results=candidate_results,
        status=status,
        reason_code=reason_code,
    )
    return NestedRefitResult(candidate_results=candidate_results, receipt=receipt)


def validate_nested_refit_receipt(
    channel_selection: Mapping[str, object],
    *,
    outer_fold: str,
    required_optional_measurements: Sequence[str],
) -> dict[str, object]:
    """Validate the P10 nested-selection receipt before P11/P12 may proceed."""
    raw = channel_selection.get("nested_refit_receipt")
    if not isinstance(raw, Mapping):
        raise RuntimeError("NESTED_CHANNEL_REFIT_RECEIPT_MISSING")
    receipt = {str(key): value for key, value in raw.items()}
    if str(receipt.get(OUTER_FOLD)) != str(outer_fold):
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_FOLD_MISMATCH")
    if receipt.get("fit_scope") != "development_history_only":
        raise RuntimeError("NESTED_CHANNEL_REFIT_SCOPE_INVALID")
    if receipt.get("outer_outcomes_accessed") is not False:
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_OUTCOME_FIREWALL_BREACH")
    if receipt.get("outer_rows_used_in_selection") != 0:
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_ROWS_USED")
    removed = receipt.get("heldout_channel_removed_from")
    if not isinstance(removed, list) or set(str(value) for value in removed) != set(
        _REMOVED_INPUTS
    ):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")

    candidate_rows_raw = receipt.get("candidate_results")
    if not isinstance(candidate_rows_raw, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_CANDIDATES_MISSING")
    candidate_rows = [
        cast(Mapping[str, object], item)
        for item in candidate_rows_raw
        if isinstance(item, Mapping)
    ]
    by_candidate = {str(item.get("candidate")): item for item in candidate_rows}
    for measurement_id in required_optional_measurements:
        candidate = by_candidate.get(str(measurement_id))
        if candidate is None or candidate.get(ELIGIBLE) is not True:
            raise RuntimeError(
                f"NESTED_CHANNEL_REFIT_REQUIRED_CANDIDATE_INCOMPLETE:{measurement_id}"
            )
    if required_optional_measurements and receipt.get("status") != "PASS":
        raise RuntimeError("NESTED_CHANNEL_REFIT_REQUIRED_PASS_MISSING")
    return receipt


def _score_fit(
    *,
    fit: ModelFitResult,
    candidate: str,
    heldout_channel: str,
    heldout_outcomes: pd.DataFrame,
    gate1_learner_id: str,
    gate1_feature_group: str,
    eligible_feature_count: int,
    columns: Mapping[str, str],
) -> dict[str, object]:
    firm = _physical(columns, FIRM_ID)
    year = _physical(columns, FISCAL_YEAR)
    learner = _physical(columns, LEARNER_ID)
    prediction = _physical(columns, PREDICTION)
    target_value = _physical(columns, TARGET_VALUE)
    model_suffix = f":{gate1_learner_id}:{gate1_feature_group}"

    oof = fit.oof_predictions.loc[
        fit.oof_predictions[learner].astype(str).str.endswith(model_suffix)
    ].copy()
    target_rows_raw = fit.models.get("oof_training_targets")
    if not isinstance(target_rows_raw, list):
        return _failed_cell(
            candidate,
            heldout_channel,
            "OOF_TRAINING_TARGETS_MISSING",
            eligible_feature_count,
        )
    target_rows = pd.DataFrame(
        [cast(dict[str, object], item) for item in target_rows_raw if isinstance(item, dict)]
    )
    if target_rows.empty:
        return _failed_cell(
            candidate,
            heldout_channel,
            "NO_CROSS_FITTED_PREDICTIONS",
            eligible_feature_count,
        )
    target_rows = target_rows.rename(
        columns={
            FIRM_ID: firm,
            FISCAL_YEAR: year,
            LEARNER_ID: learner,
            TARGET_VALUE: target_value,
        }
    )
    target_rows = target_rows.loc[target_rows[learner].astype(str).str.endswith(model_suffix)]
    if oof.empty or target_rows.empty:
        return _failed_cell(
            candidate,
            heldout_channel,
            "NO_CROSS_FITTED_PREDICTIONS",
            eligible_feature_count,
        )
    merged = (
        oof[[firm, year, learner, prediction]]
        .merge(
            target_rows[[firm, year, learner, target_value]],
            on=[firm, year, learner],
            how="inner",
            validate="1:1",
        )
        .merge(heldout_outcomes, on=[firm, year], how="inner", validate="m:1")
    )
    if merged.empty:
        return _failed_cell(
            candidate,
            heldout_channel,
            "NO_PAIRED_HELDOUT_ROWS",
            eligible_feature_count,
        )
    intercept, slope, calibration_status = _fit_soft_platt(
        merged[prediction].to_numpy(dtype=float),
        merged[target_value].to_numpy(dtype=float),
    )
    calibrated = _apply_platt(
        merged[prediction].to_numpy(dtype=float),
        intercept=intercept,
        slope=slope,
    )
    truth = merged["_heldout_outcome"].to_numpy(dtype=float)
    return {
        "candidate": candidate,
        _HELDOUT_CHANNEL: heldout_channel,
        "status": "PASS",
        "reason_code": None,
        "rows": len(merged),
        "soft_cross_entropy": _cross_entropy(truth, calibrated),
        "calibration_status": calibration_status,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "fit_scope": "development_history_only",
        "outer_rows_used": 0,
        "outer_outcomes_accessed": False,
        "heldout_removed_from_target": True,
        "heldout_removed_from_label_model": True,
        "heldout_removed_from_features": True,
        "heldout_removed_from_tuning": True,
        "heldout_removed_from_calibration": True,
        "eligible_feature_count": eligible_feature_count,
        "gate1_learner_id": gate1_learner_id,
        "gate1_feature_group": gate1_feature_group,
    }


def _candidate_target_frame(
    *,
    candidate: str,
    heldout_channel: str,
    rows: Sequence[Mapping[str, object]],
    outer_year: int,
    minimum_observed_channels: int,
    l3_fold_result: Mapping[str, object] | None,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    firm = _physical(columns, FIRM_ID)
    year = _physical(columns, FISCAL_YEAR)
    target_value = _physical(columns, TARGET_VALUE)
    if candidate == "L3_fixed_pi":
        raw_targets = None if l3_fold_result is None else l3_fold_result.get(
            "heldout_target_rows"
        )
        if not isinstance(raw_targets, list):
            return pd.DataFrame(columns=[firm, year, target_value])
        parsed = [
            cast(Mapping[str, object], item)
            for item in raw_targets
            if isinstance(item, Mapping)
            and str(item.get(_HELDOUT_CHANNEL)) == heldout_channel
            and isinstance(item.get(FISCAL_YEAR), int)
            and int(cast(int, item[FISCAL_YEAR])) < outer_year
        ]
        return _target_frame_from_rows(parsed, columns)

    if candidate != "L2":
        return pd.DataFrame(columns=[firm, year, target_value])
    output: list[dict[str, object]] = []
    for row in rows:
        row_year = row.get(FISCAL_YEAR)
        if not isinstance(row_year, int) or row_year >= outer_year:
            continue
        if row.get(ELIGIBLE) is not True or row.get(MATURE) is not True:
            continue
        raw_scores = row.get("channel_evidence_scores")
        if not isinstance(raw_scores, Mapping):
            continue
        remaining = [
            float(value)
            for channel, value in raw_scores.items()
            if str(channel) != heldout_channel
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        if len(remaining) < minimum_observed_channels:
            continue
        output.append(
            {
                firm: str(row[FIRM_ID]),
                year: row_year,
                target_value: float(np.mean(remaining)),
            }
        )
    return _typed_target_frame(output, columns)


def _target_frame_from_rows(
    rows: Sequence[Mapping[str, object]],
    columns: Mapping[str, str],
) -> pd.DataFrame:
    firm = _physical(columns, FIRM_ID)
    year = _physical(columns, FISCAL_YEAR)
    target_value = _physical(columns, TARGET_VALUE)
    output: list[dict[str, object]] = []
    for row in rows:
        value = row.get(TARGET_VALUE)
        row_year = row.get(FISCAL_YEAR)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isinstance(row_year, int)
        ):
            continue
        output.append(
            {
                firm: str(row[FIRM_ID]),
                year: row_year,
                target_value: float(value),
            }
        )
    return _typed_target_frame(output, columns)


def _typed_target_frame(
    rows: list[dict[str, object]],
    columns: Mapping[str, str],
) -> pd.DataFrame:
    firm = _physical(columns, FIRM_ID)
    year = _physical(columns, FISCAL_YEAR)
    target_value = _physical(columns, TARGET_VALUE)
    frame = pd.DataFrame(rows, columns=[firm, year, target_value])
    return frame.astype({firm: "string", year: "int16", target_value: "float64"})


def _heldout_outcome_frame(
    *,
    rows: Sequence[Mapping[str, object]],
    heldout_channel: str,
    outer_year: int,
    columns: Mapping[str, str],
) -> pd.DataFrame:
    firm = _physical(columns, FIRM_ID)
    year = _physical(columns, FISCAL_YEAR)
    output: list[dict[str, object]] = []
    for row in rows:
        row_year = row.get(FISCAL_YEAR)
        if not isinstance(row_year, int) or row_year >= outer_year:
            continue
        raw_outcomes = row.get("channel_outcomes")
        if not isinstance(raw_outcomes, Mapping):
            continue
        value = raw_outcomes.get(heldout_channel)
        if value is None:
            continue
        output.append(
            {
                firm: str(row[FIRM_ID]),
                year: row_year,
                "_heldout_outcome": float(bool(value)),
            }
        )
    frame = pd.DataFrame(output, columns=[firm, year, "_heldout_outcome"])
    return frame.astype({firm: "string", year: "int16", "_heldout_outcome": "float64"})


def _heldout_feature_registry(
    *,
    feature_registry: Sequence[Mapping[str, object]],
    feature_panel: pd.DataFrame,
    heldout_channel: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in feature_registry:
        item = {str(key): value for key, value in raw.items()}
        feature_id = item.get("feature_id")
        if not isinstance(feature_id, str) or feature_id not in feature_panel.columns:
            continue
        if item.get("research_decision_status") not in {None, "LOCKED"}:
            continue
        if item.get("model_eligibility") not in {None, "eligible"}:
            continue
        if str(item.get("source_channel", "")) == heldout_channel:
            continue
        role = item.get("role", item.get("content_observability_role"))
        if role not in {"content", "observability", "ambiguous"}:
            continue
        item["role"] = role
        output.append(cast(dict[str, Any], item))
    return output


def _features_for_group(
    registry: Sequence[Mapping[str, object]],
    group_id: str,
) -> list[str]:
    content = [
        str(item["feature_id"]) for item in registry if item.get("role") == "content"
    ]
    observability = [
        str(item["feature_id"])
        for item in registry
        if item.get("role") == "observability"
    ]
    ambiguous = [
        str(item["feature_id"]) for item in registry if item.get("role") == "ambiguous"
    ]
    groups = {
        "content_only": content,
        "observability_only": observability,
        "full": [*observability, *content, *ambiguous],
    }
    return groups[group_id]


def _fit_soft_platt(
    scores: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, float, str]:
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if len(scores) < 2 or float(np.ptp(targets)) <= 1e-12:
        return 0.0, 1.0, "IDENTITY_INSUFFICIENT_TARGET_VARIATION"
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("calibration targets must be in [0, 1]")
    logits = _logit(scores)
    design = np.column_stack([np.ones(len(logits), dtype=float), logits])
    beta = np.asarray([0.0, 1.0], dtype=float)
    ridge = np.asarray([1e-8, 1e-8], dtype=float)
    for _ in range(100):
        linear = np.clip(design @ beta, -35.0, 35.0)
        probabilities = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (probabilities - targets) + ridge * beta
        curvature = probabilities * (1.0 - probabilities)
        hessian = design.T @ (design * curvature[:, None]) + np.diag(ridge)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return 0.0, 1.0, "IDENTITY_SINGULAR_CALIBRATION"
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            return float(beta[0]), float(beta[1]), "SOFT_PLATT"
    return float(beta[0]), float(beta[1]), "SOFT_PLATT_MAX_ITERATIONS"


def _apply_platt(
    scores: np.ndarray,
    *,
    intercept: float,
    slope: float,
) -> np.ndarray:
    linear = intercept + slope * _logit(scores)
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))


def _logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return float(
        np.mean(
            -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
        )
    )


def _matrix_rows(matrices: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw = matrices.get("rows")
    if not isinstance(raw, list):
        raise ValueError("nested channel refit requires matrix rows")
    return [
        cast(Mapping[str, object], item) for item in raw if isinstance(item, Mapping)
    ]


def _string_list(raw: object, context: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"nested channel refit requires {context}")
    values = [str(value) for value in raw]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"nested channel refit requires unique nonempty {context}")
    return values


def _physical(columns: Mapping[str, str], key: str) -> str:
    value = columns.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"physical column binding missing for {key}")
    return value


def _failed_cell(
    candidate: str,
    heldout_channel: str,
    reason_code: str,
    eligible_feature_count: int,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        _HELDOUT_CHANNEL: heldout_channel,
        "status": "INSUFFICIENT_EVIDENCE",
        "reason_code": reason_code,
        "rows": 0,
        "soft_cross_entropy": None,
        "fit_scope": "development_history_only",
        "outer_rows_used": 0,
        "outer_outcomes_accessed": False,
        "heldout_removed_from_target": True,
        "heldout_removed_from_label_model": True,
        "heldout_removed_from_features": True,
        "heldout_removed_from_tuning": True,
        "heldout_removed_from_calibration": True,
        "eligible_feature_count": eligible_feature_count,
    }


def _receipt(
    *,
    outer_year: int,
    gate1_learner_id: str,
    gate1_feature_group: str,
    expected_channels: Sequence[str],
    cell_results: Sequence[Mapping[str, object]],
    candidate_results: Sequence[Mapping[str, object]],
    status: str,
    reason_code: str | None,
) -> dict[str, object]:
    return {
        "status": status,
        "reason_code": reason_code,
        OUTER_FOLD: str(outer_year),
        "selection_procedure": "full_refit_channel_within_time",
        "fit_scope": "development_history_only",
        "outer_outcomes_accessed": False,
        "outer_rows_used_in_selection": 0,
        "heldout_channel_removed_from": list(_REMOVED_INPUTS),
        "complete_channel_grid_required": True,
        "expected_channels": list(expected_channels),
        "gate1_learner_id": gate1_learner_id,
        "gate1_feature_group": gate1_feature_group,
        "cell_results": [dict(item) for item in cell_results],
        "candidate_results": [dict(item) for item in candidate_results],
    }


def _token(value: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in value
    ).strip("_")
