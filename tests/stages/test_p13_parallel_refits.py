from __future__ import annotations

from typing import Any

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


def test_parallel_source_refits_are_worker_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    columns = {
        FIRM_ID: "firm_master_id",
        FISCAL_YEAR: "fiscal_year",
        LEARNER_ID: "model_id",
        OUTCOME: "outcome",
        PREDICTION: "prediction",
    }

    monkeypatch.setattr(
        parallel_refits,
        "_alternative_l1_labels",
        lambda **_: pd.DataFrame(),
    )
    monkeypatch.setattr(
        parallel_refits,
        "select_observed_target_outcomes",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {"firm_master_id": "F1", "fiscal_year": 2021, "outcome": True},
                {"firm_master_id": "F2", "fiscal_year": 2021, "outcome": False},
                {"firm_master_id": "F1", "fiscal_year": 2022, "outcome": True},
                {"firm_master_id": "F2", "fiscal_year": 2022, "outcome": False},
            ]
        ),
    )

    def fake_fit_fold_models(**kwargs: Any) -> ModelFitResult:
        outer_year = int(kwargs["outer_year"])
        target_id = str(kwargs["target_id"])
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

    monkeypatch.setattr(parallel_refits, "fit_fold_models", fake_fit_fold_models)

    matrices = {
        "rows": [],
        "expected_sources": {"S1": "C1", "S2": "C2"},
    }
    outer_folds = ["2021", "2022"]
    exclusion_ids = ["source:S1", "source:S2", "channel:C1", "channel:C2"]
    seeds = {
        (fold_id, exclusion_id): index + 1
        for index, (exclusion_id, fold_id) in enumerate(
            (exclusion_id, fold_id) for exclusion_id in exclusion_ids for fold_id in outer_folds
        )
    }
    arguments: dict[str, Any] = {
        "matrices": matrices,
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
    }

    sequential = parallel_refits.parallel_source_exclusion_refits(
        **arguments,
        workers=1,
    )
    parallel = parallel_refits.parallel_source_exclusion_refits(
        **arguments,
        workers=4,
    )

    assert parallel == sequential
    assert [(row["exclusion_id"], row[OUTER_FOLD]) for row in parallel["results"]] == [
        (exclusion_id, fold_id) for exclusion_id in exclusion_ids for fold_id in outer_folds
    ]


def test_parallel_source_refits_require_positive_workers() -> None:
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
            workers=0,
        )
