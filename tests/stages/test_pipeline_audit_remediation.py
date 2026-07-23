from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import yaml

from core.mcse_contract import p08_completion_blockers, require_p08_completion
from core.semantic_keys import (
    FIRM_ID,
    FISCAL_YEAR,
    LEARNER_ID,
    OUTER_FOLD,
    PREDICTION,
)
from evaluation.pairing import validate_paired_prediction_keys
from gates.evidence_contract import prepare_gate2_inputs


CANDIDATES = [
    "track_l1:elastic_net_logistic:full",
    "track_l1:random_forest:full",
    "track_l1:main_boosting:full",
]
REFERENCES = {
    candidate: candidate.removesuffix(":full") + ":observability_only"
    for candidate in CANDIDATES
}
FOLDS = ["2021", "2022", "2023", "2024"]


def test_production_configuration_locks_target_and_tuning_grids() -> None:
    measurement = yaml.safe_load(
        Path("config/methodology/measurement.yaml").read_text(encoding="utf-8")
    )["measurement"]
    learners = yaml.safe_load(
        Path("config/execution/learners.yaml").read_text(encoding="utf-8")
    )["learners"]
    assert measurement["primary_target_id"] == "L1_ANNUAL"
    assert "primary_target_id" not in measurement["execution_tracks"]
    search_spaces = learners["tuning"]["search_spaces"]
    assert set(search_spaces) == set(learners["confirmatory"])
    maximum = learners["tuning"]["max_valid_configurations_per_learner_inner_fold"]
    for space in search_spaces.values():
        count = 1
        for options in space.values():
            count *= len(options)
        assert 1 <= count <= maximum


def test_p08_completion_contract_rejects_intermediate_statuses() -> None:
    assert p08_completion_blockers({"status": "PASS", "precision_target_met": True}) == []
    blockers = p08_completion_blockers(
        {"status": "CONTINUE", "precision_target_met": False}
    )
    assert blockers == ["P08_STATUS_NOT_PASS", "P08_PRECISION_TARGET_NOT_MET"]
    with pytest.raises(RuntimeError, match="P09_PRODUCTION_PATH_BLOCKED"):
        require_p08_completion(
            {"status": "INCOMPLETE", "precision_target_met": False}, "P09"
        )


def test_p12_pairing_requires_exact_firm_year_support() -> None:
    columns = {
        FIRM_ID: "firm_key",
        FISCAL_YEAR: "year_key",
        LEARNER_ID: "model_key",
        PREDICTION: "score",
    }
    candidate = CANDIDATES[0]
    reference = REFERENCES[candidate]
    exact = pd.DataFrame(
        [
            {"firm_key": "F1", "year_key": 2024, "model_key": candidate, "score": 0.8},
            {"firm_key": "F2", "year_key": 2024, "model_key": candidate, "score": 0.2},
            {"firm_key": "F1", "year_key": 2024, "model_key": reference, "score": 0.7},
            {"firm_key": "F2", "year_key": 2024, "model_key": reference, "score": 0.3},
        ]
    )
    audit = validate_paired_prediction_keys(exact, columns)
    assert audit["status"] == "PASS"
    assert audit["pair_count"] == 1

    mismatch = exact.loc[
        ~(exact["model_key"].eq(reference) & exact["firm_key"].eq("F2"))
    ].copy()
    with pytest.raises(RuntimeError, match="FIRM_YEAR_SUPPORT_MISMATCH"):
        validate_paired_prediction_keys(mismatch, columns)


def test_gate2_contract_requires_complete_yield_and_bootstrap_evidence() -> None:
    gate: dict[str, Any] = {
        "candidate_ids": CANDIDATES,
        "reference_by_candidate": REFERENCES,
        "candidate_family_count": len(CANDIDATES),
        "minimum_valid_bootstrap_replications": 1800,
        "fold_count": len(FOLDS),
    }
    evaluations: list[dict[str, Any]] = []
    bootstraps: list[list[dict[str, Any]]] = []
    for fold in FOLDS:
        models: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        batch: list[dict[str, Any]] = []
        for index, candidate in enumerate(CANDIDATES):
            reference = REFERENCES[candidate]
            models.extend(
                [
                    {
                        LEARNER_ID: candidate,
                        "average_precision": 0.30 + index * 0.01,
                        "precision_at_primary_budget": 0.25,
                        "positives": 30,
                    },
                    {
                        LEARNER_ID: reference,
                        "average_precision": 0.20 + index * 0.01,
                        "precision_at_primary_budget": 0.24,
                        "positives": 30,
                    },
                ]
            )
            comparisons.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "delta_average_precision": 0.10,
                }
            )
            batch.append(
                {
                    OUTER_FOLD: fold,
                    "candidate": candidate,
                    "reference": reference,
                    "delta_ap_samples": [0.05] * 1800,
                }
            )
        evaluations.append(
            {OUTER_FOLD: fold, "models": models, "comparisons": comparisons}
        )
        bootstraps.append(batch)

    complete = prepare_gate2_inputs(
        evaluations=evaluations,
        bootstraps=bootstraps,
        gate=gate,
        confirmatory_folds=FOLDS,
    )
    assert complete["blockers"] == []

    first_models = cast(list[dict[str, Any]], evaluations[0]["models"])
    first_models[0]["precision_at_primary_budget"] = None
    incomplete = prepare_gate2_inputs(
        evaluations=evaluations,
        bootstraps=bootstraps,
        gate=gate,
        confirmatory_folds=FOLDS,
    )
    assert any(
        str(blocker).startswith("GATE2_YIELD_MISSING:2021")
        for blocker in cast(list[object], incomplete["blockers"])
    )
