from __future__ import annotations

from typing import Any, cast

import pandas as pd

import sensitivity.domain_transfer as domain_module
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    OUTCOME,
    PREDICTION,
    TARGET_ID,
    WEIGHT,
)
from modeling.service import ModelFitResult
from sensitivity.domain_transfer import candidate_domain_transfer

FIRM = "firm_master_id"
YEAR = "fiscal_year"
TARGET = "target_id"
Y = "outcome"
LEARNER = "learner_id"
PRED = "prediction"
WEIGHT_COL = "weight"


def _columns() -> dict[str, str]:
    return {
        FIRM_ID: FIRM,
        FISCAL_YEAR: YEAR,
        TARGET_ID: TARGET,
        OUTCOME: Y,
        LEARNER_ID: LEARNER,
        PREDICTION: PRED,
        WEIGHT: WEIGHT_COL,
    }


def test_candidate_domain_transfer_excludes_held_board_from_development(monkeypatch: Any) -> None:
    rows: list[dict[str, object]] = []
    outcomes: list[dict[str, object]] = []
    for year in (2020, 2021):
        for board in ("HOSE", "HNX"):
            for index, outcome in enumerate((False, True, False, True)):
                firm_id = f"{board}-{year}-{index}"
                rows.append(
                    {
                        FIRM: firm_id,
                        YEAR: year,
                        "exchange_or_board": board,
                        "obs_feature": float(index % 2),
                        "content_feature": float(outcome),
                    }
                )
                outcomes.append({FIRM: firm_id, YEAR: year, TARGET: "L1_ANNUAL", Y: outcome})

    seen: list[tuple[str, tuple[str, ...]]] = []

    def fake_fit_fold_models(**kwargs: Any) -> ModelFitResult:
        panel = cast(pd.DataFrame, kwargs["feature_panel"])
        outer_year = int(kwargs["outer_year"])
        train_boards = tuple(
            sorted(
                {
                    str(value)
                    for value in panel.loc[panel[YEAR] < outer_year, "exchange_or_board"].tolist()
                }
            )
        )
        test_boards = {
            str(value)
            for value in panel.loc[panel[YEAR] == outer_year, "exchange_or_board"].tolist()
        }
        assert len(test_boards) == 1
        held_board = next(iter(test_boards))
        assert held_board not in train_boards
        seen.append((held_board, train_boards))

        prediction_rows: list[dict[str, object]] = []
        for row in panel.loc[panel[YEAR] == outer_year].to_dict(orient="records"):
            prediction_rows.extend(
                [
                    {
                        FIRM: row[FIRM],
                        YEAR: row[YEAR],
                        LEARNER: "domain_transfer:elastic_net_logistic:full",
                        PRED: float(row["content_feature"]),
                    },
                    {
                        FIRM: row[FIRM],
                        YEAR: row[YEAR],
                        LEARNER: "domain_transfer:elastic_net_logistic:observability_only",
                        PRED: 0.5,
                    },
                ]
            )
        return ModelFitResult(
            models={"status": "PASS", "models": []},
            oof_predictions=pd.DataFrame(),
            outer_predictions=pd.DataFrame(prediction_rows),
        )

    monkeypatch.setattr(domain_module, "fit_fold_models", fake_fit_fold_models)
    result = candidate_domain_transfer(
        predictions=pd.DataFrame({YEAR: [2021]}),
        outcomes=pd.DataFrame(outcomes),
        feature_panel=pd.DataFrame(rows),
        feature_registry=[
            {"feature_id": "obs_feature", "role": "observability"},
            {"feature_id": "content_feature", "role": "content"},
        ],
        domain_bindings=[{"domain_id": "exchange_or_board", "column": "exchange_or_board"}],
        learner_ids=["elastic_net_logistic"],
        learner_settings={},
        learner_search_spaces={},
        maximum_valid_configurations=1,
        noninferiority_margin=0.05,
        support_fraction_minimum=0.8,
        support_bounds=(0.05, 0.95),
        support_crossfit_folds=5,
        minimum_domains=2,
        evaluation_target_id="L1_ANNUAL",
        columns=_columns(),
        seed_by_scenario={
            ("2021", "exchange_or_board", "HOSE"): 1,
            ("2021", "exchange_or_board", "HNX"): 2,
        },
    )

    assert set(seen) == {("HOSE", ("HNX",)), ("HNX", ("HOSE",))}
    assert result["status"] == "PASS"
    assert (
        result["support_method"] == "cross_fitted_balanced_logistic_domain_score_held_test_fraction"
    )
    assert result["support_score_interpretation"] == (
        "balanced_domain_membership_score_not_causal_propensity"
    )
    candidates = cast(list[dict[str, object]], result["candidate_results"])
    assert candidates[0]["evidence_complete"] is True
    assert candidates[0]["robust_scenario_fraction"] == 1.0
    domain_rows = cast(list[dict[str, object]], result["domains"])
    assert {row["support_crossfit_folds"] for row in domain_rows} == {4}
