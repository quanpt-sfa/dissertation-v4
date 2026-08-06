"""Deterministic process execution and checkpoints for P13 source refits."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
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

P13_CHECKPOINT_IMPLEMENTATION_ID = "P13_PROCESS_REFITS_V1"
_WORKER_STATE: dict[str, Any] | None = None


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


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    task_index: int,
    exclusion_id: str,
    fold_id: str,
) -> Path:
    task_hash = hashlib.sha256(f"{exclusion_id}\0{fold_id}".encode()).hexdigest()[:12]
    return checkpoint_dir / f"{task_index:03d}_{task_hash}.json"


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _read_checkpoint(
    path: Path,
    *,
    protocol_hash: str,
    contract_hash: str,
    exclusion_id: str,
    fold_id: str,
    seed: int,
) -> list[dict[str, object]] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise RuntimeError(f"P13 checkpoint is not a file: {path}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"P13 checkpoint is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"P13 checkpoint must be an object: {path}")
    checkpoint = cast(dict[str, object], raw)
    expected = {
        "schema_version": 1,
        "implementation_id": P13_CHECKPOINT_IMPLEMENTATION_ID,
        "protocol_hash": protocol_hash,
        "contract_hash": contract_hash,
        "exclusion_id": exclusion_id,
        "outer_fold": fold_id,
        "seed": seed,
    }
    mismatched = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if mismatched:
        raise RuntimeError(f"P13 checkpoint contract mismatch at {path}; mismatched={mismatched}")
    results = checkpoint.get("results")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise RuntimeError(f"P13 checkpoint results are invalid: {path}")
    return [cast(dict[str, object], item) for item in cast(list[object], results)]


def _initialize_worker(state: dict[str, Any]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _run_unit_with_state(
    task: tuple[str, str],
    state: dict[str, Any],
) -> list[dict[str, object]]:
    exclusion_id, fold_id = task
    exclusions = cast(dict[str, set[str]], state["exclusions"])
    labels_by_exclusion = cast(dict[str, pd.DataFrame], state["labels_by_exclusion"])
    weights_by_fold = cast(dict[str, pd.DataFrame], state["weights_by_fold"])
    columns = cast(dict[str, str], state["columns"])
    seeds = cast(dict[tuple[str, str], int], state["seed_by_fold_and_exclusion"])
    excluded_sources = exclusions[exclusion_id]
    training_target_id = f"L1_without_{exclusion_id}"
    weights = weights_by_fold.get(fold_id)
    if weights is None:
        raise ValueError(f"fold={fold_id}: sensitivity weights required")

    fit = fit_fold_models(
        feature_panel=cast(pd.DataFrame, state["feature_panel"]),
        feature_registry=cast(list[dict[str, Any]], state["feature_registry"]),
        label_inputs=labels_by_exclusion[exclusion_id],
        weights=weights,
        outer_year=int(fold_id),
        learner_ids=cast(list[str], state["learner_ids"]),
        learner_settings=cast(dict[str, Any], state["learner_settings"]),
        target_id=training_target_id,
        measurement_id=training_target_id,
        columns=columns,
        random_state=seeds[(fold_id, exclusion_id)],
        track_id="source_sensitivity",
        learner_search_spaces=cast(dict[str, Any], state["learner_search_spaces"]),
        maximum_valid_configurations=int(state["maximum_valid_configurations"]),
    )
    evaluated = fit.outer_predictions.merge(
        cast(pd.DataFrame, state["observed_outcomes"]),
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
                "evaluation_target_id": str(state["evaluation_target_id"]),
                "fit_status": fit.models["status"],
                "rows": len(frame),
                "positives": int(frame[columns[OUTCOME]].sum()),
                "average_precision": average_precision(truth, scores),
                "outer_outcomes_used_in_fit": False,
                "tuning_scope": "development_history_only",
                "tuning_budget_maximum": int(state["maximum_valid_configurations"]),
            }
        )
    return unit_rows


def _run_worker_unit(task: tuple[str, str]) -> list[dict[str, object]]:
    if _WORKER_STATE is None:
        raise RuntimeError("P13 process worker was not initialized")
    return _run_unit_with_state(task, _WORKER_STATE)


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
    protocol_hash: str,
    checkpoint_dir: Path,
    workers: int,
) -> dict[str, object]:
    """Run exclusion/fold units in processes with deterministic resumable checkpoints."""
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

    contract_hash = _stable_hash(
        {
            "implementation_id": P13_CHECKPOINT_IMPLEMENTATION_ID,
            "protocol_hash": protocol_hash,
            "evaluation_target_id": evaluation_target_id,
            "columns": columns,
            "outer_folds": outer_folds,
            "learner_ids": learner_ids,
            "learner_settings": learner_settings,
            "learner_search_spaces": learner_search_spaces,
            "maximum_valid_configurations": maximum_valid_configurations,
            "source_channels": source_channels,
            "seeds": [
                {
                    "exclusion_id": exclusion_id,
                    "outer_fold": fold_id,
                    "seed": seed_by_fold_and_exclusion[(fold_id, exclusion_id)],
                }
                for exclusion_id, fold_id in tasks
            ],
        }
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        task: _checkpoint_path(
            checkpoint_dir,
            task_index=index,
            exclusion_id=task[0],
            fold_id=task[1],
        )
        for index, task in enumerate(tasks)
    }

    unit_results: dict[tuple[str, str], list[dict[str, object]]] = {}
    for exclusion_id, fold_id in tasks:
        cached = _read_checkpoint(
            checkpoints[(exclusion_id, fold_id)],
            protocol_hash=protocol_hash,
            contract_hash=contract_hash,
            exclusion_id=exclusion_id,
            fold_id=fold_id,
            seed=seed_by_fold_and_exclusion[(fold_id, exclusion_id)],
        )
        if cached is not None:
            unit_results[(exclusion_id, fold_id)] = cached

    total = len(tasks)
    completed = len(unit_results)
    print(
        f"P13 source refits progress={completed}/{total} cached={completed} "
        f"checkpoint_dir={checkpoint_dir}",
        flush=True,
    )
    pending = [task for task in tasks if task not in unit_results]

    state: dict[str, Any] = {
        "feature_panel": feature_panel,
        "feature_registry": feature_registry,
        "weights_by_fold": weights_by_fold,
        "observed_outcomes": observed_outcomes,
        "labels_by_exclusion": labels_by_exclusion,
        "exclusions": exclusions,
        "learner_ids": learner_ids,
        "learner_settings": learner_settings,
        "learner_search_spaces": learner_search_spaces,
        "maximum_valid_configurations": maximum_valid_configurations,
        "evaluation_target_id": evaluation_target_id,
        "columns": columns,
        "seed_by_fold_and_exclusion": seed_by_fold_and_exclusion,
    }

    def publish(task: tuple[str, str], rows: list[dict[str, object]]) -> None:
        nonlocal completed
        exclusion_id, fold_id = task
        payload: dict[str, object] = {
            "schema_version": 1,
            "implementation_id": P13_CHECKPOINT_IMPLEMENTATION_ID,
            "protocol_hash": protocol_hash,
            "contract_hash": contract_hash,
            "exclusion_id": exclusion_id,
            "outer_fold": fold_id,
            "seed": seed_by_fold_and_exclusion[(fold_id, exclusion_id)],
            "results": rows,
        }
        _write_checkpoint(checkpoints[task], payload)
        unit_results[task] = rows
        completed += 1
        print(
            f"P13 source refits progress={completed}/{total} "
            f"exclusion={exclusion_id} fold={fold_id}",
            flush=True,
        )

    worker_count = min(workers, len(pending)) if pending else 0
    if worker_count == 1:
        for task in pending:
            publish(task, _run_unit_with_state(task, state))
    elif worker_count > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_initialize_worker,
            initargs=(state,),
        ) as executor:
            futures = {executor.submit(_run_worker_unit, task): task for task in pending}
            try:
                for future in as_completed(futures):
                    task = futures[future]
                    publish(task, future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    rows = [row for task in tasks for row in unit_results[task]]
    return {
        "status": "PASS" if rows else "SKIPPED",
        "reason_code": None if rows else "INSUFFICIENT_SOURCE_REFITS",
        "evaluation_target_id": evaluation_target_id,
        "refit_executed": bool(rows),
        "execution_backend": "process",
        "checkpoint_implementation_id": P13_CHECKPOINT_IMPLEMENTATION_ID,
        "checkpoint_contract_hash": contract_hash,
        "checkpoint_tasks_reused": total - len(pending),
        "checkpoint_tasks_computed": len(pending),
        "results": rows,
    }
