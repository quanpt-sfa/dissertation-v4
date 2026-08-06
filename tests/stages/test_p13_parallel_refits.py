from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import pytest

import sensitivity.parallel_refits as parallel_refits
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION,
)
from modeling.service import ModelFitResult


class _ImmediateProcessPoolExecutor:
    def __init__(
        self,
        *,
        max_workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        self.max_workers = max_workers
        if initializer is not None:
            initializer(*initargs)

    def __enter__(self) -> _ImmediateProcessPoolExecutor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(
        self,
        function: Callable[[tuple[str, str]], list[dict[str, object]]],
        task: tuple[str, str],
    ) -> Future[list[dict[str, object]]]:
        future: Future[list[dict[str, object]]] = Future()
        try:
            future.set_result(function(task))
        except BaseException as exc:
            future.set_exception(exc)
        return future


def _arguments(checkpoint_dir: Path) -> dict[str, Any]:
    columns = {
        FIRM_ID: "firm_master_id",
        FISCAL_YEAR: "fiscal_year",
        LEARNER_ID: "model_id",
        OUTCOME: "outcome",
        PREDICTION: "prediction",
    }
    outer_folds = ["2021", "2022"]
    exclusion_ids = ["source:S1", "source:S2", "channel:C1", "channel:C2"]
    seeds = {
        (fold_id, exclusion_id): index + 1
        for index, (exclusion_id, fold_id) in enumerate(
            (exclusion_id, fold_id) for exclusion_id in exclusion_ids for fold_id in outer_folds
        )
    }
    return {
        "matrices": {
            "rows": [],
            "expected_sources": {"S1": "C1", "S2": "C2"},
        },
        "feature_panel": pd.DataFrame(),
        "feature_registry": [],
        "weights_by_fold": {
            "2021": pd.DataFrame(),
            "2022": pd.DataFrame(),
        },
        "outcomes": pd.DataFrame(),
        "outer_folds": outer_folds,
        "learner_ids": ["elastic_net_logistic"],
        "learner_settings": {},
        "learner_search_spaces": {},
        "maximum_valid_configurations": 1,
        "evaluation_target_id": "L1_ANNUAL",
        "columns": columns,
        "seed_by_fold_and_exclusion": seeds,
        "protocol_hash": "p" * 64,
        "checkpoint_dir": checkpoint_dir,
    }


def _patch_lightweight_refits(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []

    def fake_labels(**_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()

    def fake_observed_outcomes(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"firm_master_id": "F1", "fiscal_year": 2021, "outcome": True},
                {"firm_master_id": "F2", "fiscal_year": 2021, "outcome": False},
                {"firm_master_id": "F1", "fiscal_year": 2022, "outcome": True},
                {"firm_master_id": "F2", "fiscal_year": 2022, "outcome": False},
            ]
        )

    def fake_fit_fold_models(**kwargs: Any) -> ModelFitResult:
        outer_year = int(kwargs["outer_year"])
        target_id = str(kwargs["target_id"])
        calls.append((outer_year, target_id))
        predictions = pd.DataFrame(
            [
                {
                    "firm_master_id": "F1",
                    "fiscal_year": outer_year,
                    "model_id": f"model:{target_id}",
                    "prediction": 0.8,
                },
                {
                    "firm_master_id": "F2",
                    "fiscal_year": outer_year,
                    "model_id": f"model:{target_id}",
                    "prediction": 0.2,
                },
            ]
        )
        return ModelFitResult(
            models={"status": "PASS"},
            oof_predictions=pd.DataFrame(),
            outer_predictions=predictions,
        )

    monkeypatch.setattr(parallel_refits, "_alternative_l1_labels", fake_labels)
    monkeypatch.setattr(
        parallel_refits,
        "select_observed_target_outcomes",
        fake_observed_outcomes,
    )
    monkeypatch.setattr(parallel_refits, "fit_fold_models", fake_fit_fold_models)
    monkeypatch.setattr(
        parallel_refits,
        "ProcessPoolExecutor",
        _ImmediateProcessPoolExecutor,
    )
    return calls


def test_process_source_refits_are_worker_invariant_and_checkpointed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_lightweight_refits(monkeypatch)
    sequential_arguments = _arguments(tmp_path / "sequential")
    process_arguments = _arguments(tmp_path / "process")

    sequential = parallel_refits.parallel_source_exclusion_refits(
        **sequential_arguments,
        workers=1,
    )
    process = parallel_refits.parallel_source_exclusion_refits(
        **process_arguments,
        workers=4,
    )

    assert process["results"] == sequential["results"]
    result_rows = cast(list[dict[str, object]], process["results"])
    exclusion_ids = ["source:S1", "source:S2", "channel:C1", "channel:C2"]
    outer_folds = ["2021", "2022"]
    assert [(row["exclusion_id"], row[OUTER_FOLD]) for row in result_rows] == [
        (exclusion_id, fold_id) for exclusion_id in exclusion_ids for fold_id in outer_folds
    ]
    assert process["execution_backend"] == "process"
    assert process["checkpoint_tasks_reused"] == 0
    assert process["checkpoint_tasks_computed"] == 8
    assert len(list((tmp_path / "process").glob("*.json"))) == 8
    assert len(calls) == 16


def test_process_source_refits_reuse_complete_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_lightweight_refits(monkeypatch)
    arguments = _arguments(tmp_path / "checkpoints")
    first = parallel_refits.parallel_source_exclusion_refits(**arguments, workers=4)
    assert len(calls) == 8

    calls.clear()
    second = parallel_refits.parallel_source_exclusion_refits(**arguments, workers=4)
    assert second["results"] == first["results"]
    assert second["checkpoint_tasks_reused"] == 8
    assert second["checkpoint_tasks_computed"] == 0
    assert calls == []


def test_process_source_refits_fail_closed_on_checkpoint_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lightweight_refits(monkeypatch)
    arguments = _arguments(tmp_path / "checkpoints")
    parallel_refits.parallel_source_exclusion_refits(**arguments, workers=1)

    checkpoint = next((tmp_path / "checkpoints").glob("*.json"))
    checkpoint.write_text(checkpoint.read_text(encoding="utf-8").replace("p" * 64, "q" * 64))
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        parallel_refits.parallel_source_exclusion_refits(**arguments, workers=1)


def test_parallel_source_refits_require_positive_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        parallel_refits.parallel_source_exclusion_refits(
            matrices={"rows": [], "expected_sources": {}},
            feature_panel=pd.DataFrame(),
            feature_registry=[],
            weights_by_fold={},
            outcomes=pd.DataFrame(),
            outer_folds=[],
            learner_ids=[],
            learner_settings={},
            learner_search_spaces={},
            maximum_valid_configurations=1,
            evaluation_target_id="L1_ANNUAL",
            columns={},
            seed_by_fold_and_exclusion={},
            protocol_hash="p" * 64,
            checkpoint_dir=tmp_path,
            workers=0,
        )
