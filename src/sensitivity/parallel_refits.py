"""Deterministic parallel execution for P13 source-exclusion refits."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pandas as pd

from core.metrics import average_precision
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
    TARGET_ID,
)
from labels.service import aggregate_l1
from modeling.service import fit_fold_models
from sensitivity.service import select_observed_target_outcomes


def _alternative_l1_labels(
    *,
    matrix_rows: list[object],
    excluded_sources: set[str],
    target_id: str,
    columns: dict[str, str],
) -> pd.DataFrame:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    target = columns[TARGET_ID]
    outcome = columns[OUTCOME]
    rows: list[dict[str, object]] = []
    for raw in matrix_rows:
        if not isinstance(raw, dict):
            continue
        row = cast(dict[str, Any], raw)
        source_outcomes = row.get("source_outcomes")
        if not isinstance(source_outcomes, dict):
            continue
        remaining = {
            str(source): cast(bool | None, value)
            for source, value in cast(dict[object, object], source_outcomes).items()
            if str(source) not in excluded_sources
        }
        value = aggregate_l1(remaining) if row.get(MATURE) is True else None
        rows.append(
            {
                firm: str(row[FIRM_ID]),
                year: int(row[FISCAL_YEAR]),
                target: target_id,
                outcome: value,
            }
        )
    frame = pd.DataFrame(rows, columns=[firm, year, target, outcome])
    return frame.astype(
        {
            firm: "string",
            year: "int16",
            target: "string",
            outcome: "boolean",
        }
    )


def parallel_source_exclusion_refits(
    *,
    matrices: dict[str, Any],
    feature_panel: pd.DataFrame,
    feature_registry: list[dict[str, Any]],
    weights_by_fold: dict[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    outer_folds: list[str],
    learner_ids: list[str],
    learner_settings: dict[str, Any],
    learner_search_spaces: dict[str, Any],
    maximum_valid_configurations: int,
    evaluation_target_id: str,
    columns: dict[str, str],
    seed_by_fold_and_exclusion: dict[tuple[str, str], int],
    workers: int,
) -> dict[str, object]:
    """Refit every exclusion/fold unit concurrently while preserving row order."""
    if workers < 1:
        raise ValueError("P13 workers must be positive")

    raw_rows = matrices.get("rows")
    expected_sources = matrices.get("expected_sources")
    if not isinstance(raw_rows, list) or not isinstance(expected_sources, dict):
        raise ValueError("source sensitivity requires source-channel matrices")

    observed_outcomes = select_observed_target_outcomes(
        outcomes,
        target_id=evaluation_target_id,
        columns=columns,
        context="P13 source exclusion",
    )
    source_channels = {
        str(key): str(value) for key, value in cast(dict[object, object], expected_sources).items()
    }
    exclusions: dict[str, set[str]] = {
        f"source:{source}": {source} for source in sorted(source_channels)
    }
    for channel in sorted(set(source_channels.values())):
        exclusions[f"channel:{channel}"] = {
            source
            for source, source_channel in source_channels.items()
            if source_channel == channel
        }

    labels_by_exclusion = {
        exclusion_id: _alternative_l1_labels(
            matrix_rows=cast(list[object], raw_rows),
            excluded_sources=excluded_sources,
            target_id=f"L1_without_{exclusion_id}",
            columns=columns,
        )
        for exclusion_id, excluded_sources in exclusions.items()
    }
    tasks = [(exclusion_id, fold_id) for exclusion_id in exclusions for fold_id in outer_folds]

    missing_seeds = [
        (fold_id, exclusion_id)
        for exclusion_id, fold_id in tasks
        if (fold_id, exclusion_id) not in seed_by_fold_and_exclusion
    ]
    if missing_seeds:
        raise ValueError(f"P13 source-refit seeds are incomplete: {missing_seeds[:5]}")

    def run_unit(task: tuple[str, str]) -> list[dict[str, object]]:
        exclusion_id, fold_id = task
        excluded_sources = exclusions[exclusion_id]
        training_target_id = f"L1_without_{exclusion_id}"
        weights = weights_by_fold.get(fold_id)
        if weights is None:
            raise ValueError(f"fold={fold_id}: sensitivity weights required")

        fit = fit_fold_models(
            feature_panel=feature_panel,
            feature_registry=feature_registry,
            label_inputs=labels_by_exclusion[exclusion_id],
            weights=weights,
            outer_year=int(fold_id),
            learner_ids=learner_ids,
            learner_settings=learner_settings,
            target_id=training_target_id,
            measurement_id=training_target_id,
            columns=columns,
            random_state=seed_by_fold_and_exclusion[(fold_id, exclusion_id)],
            track_id="source_sensitivity",
            learner_search_spaces=learner_search_spaces,
            maximum_valid_configurations=maximum_valid_configurations,
        )
        evaluated = fit.outer_predictions.merge(
            observed_outcomes,
            on=[columns[FIRM_ID], columns[FISCAL_YEAR]],
            how="inner",
            validate="m:1",
        )

        unit_rows: list[dict[str, object]] = []
        for model_id, frame in evaluated.groupby(columns[LEARNER_ID], sort=True):
            truth = frame[columns[OUTCOME]].astype(bool).tolist()
            scores = frame[columns[PREDICTION]].astype(float).tolist()
            unit_rows.append(
                {
                    "exclusion_id": exclusion_id,
                    "excluded_source_ids": sorted(excluded_sources),
                    OUTER_FOLD: fold_id,
                    "model_id": str(model_id),
                    "training_target_id": training_target_id,
                    "evaluation_target_id": evaluation_target_id,
                    "fit_status": fit.models["status"],
                    "rows": len(frame),
                    "positives": int(frame[columns[OUTCOME]].sum()),
                    "average_precision": average_precision(truth, scores),
                    "outer_outcomes_used_in_fit": False,
                    "tuning_scope": "development_history_only",
                    "tuning_budget_maximum": maximum_valid_configurations,
                }
            )
        return unit_rows

    worker_count = min(workers, len(tasks)) if tasks else 1
    if worker_count == 1:
        unit_results = [run_unit(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # executor.map preserves the deterministic task order.
            unit_results = list(executor.map(run_unit, tasks))

    rows = [row for unit_rows in unit_results for row in unit_rows]
    return {
        "status": "PASS" if rows else "SKIPPED",
        "reason_code": None if rows else "INSUFFICIENT_SOURCE_REFITS",
        "evaluation_target_id": evaluation_target_id,
        "refit_executed": bool(rows),
        "results": rows,
    }
