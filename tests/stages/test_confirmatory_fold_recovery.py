from __future__ import annotations

import pandas as pd

from core.semantic_keys import (
    ELIGIBLE,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    TARGET_ID,
)
from measurement.service import summarize_fold_eligibility
from scripts.run_stage_confirmatory_only import _confirmatory_outer_fold_ids


def test_initial_fold_remains_separate_while_nested_fold_is_confirmatory() -> None:
    columns = {
        FISCAL_YEAR: "fiscal_year",
        OUTCOME: "outcome",
        MATURE: "mature",
        ELIGIBLE: "eligible",
        TARGET_ID: "target_id",
    }
    sealed = pd.DataFrame(
        [
            {"fiscal_year": year, "target_id": "L1_ANNUAL", "outcome": value}
            for year in (2020, 2021)
            for value in (True, True, False, False)
        ]
    )
    maturity = pd.DataFrame(
        [
            {
                "fiscal_year": year,
                "target_id": "L1_ANNUAL",
                "mature": True,
                "eligible": True,
            }
            for year in (2020, 2021, 2022)
            for _ in range(4)
        ]
    )

    rows = summarize_fold_eligibility(
        sealed_outcomes=sealed,
        target_maturity=maturity,
        initial_outer_year=2020,
        confirmatory_years=[2021],
        prospective_year=2022,
        confirmatory_positive_minimum=2,
        sensitivity_positive_range=(1, 1),
        columns=columns,
    )
    roles = {
        str(row[OUTER_FOLD]): str(row["assigned_role"])
        for row in rows
        if row[TARGET_ID] == "L1_ANNUAL"
    }

    assert roles["2020"] == "initial_separate"
    assert roles["2021"] == "confirmatory"
    assert roles["2022"] == "prospective_separate"


def test_confirmatory_stage_wrapper_excludes_initial_fold() -> None:
    registry = {
        "folds": {
            "initial_outer_year": 2020,
            "fully_nested_outer_years": [2021, 2022, 2023, 2024],
        }
    }

    assert _confirmatory_outer_fold_ids(registry) == [
        "2021",
        "2022",
        "2023",
        "2024",
    ]
    assert _confirmatory_outer_fold_ids(
        registry,
        include_initial=True,
    ) == ["2021", "2022", "2023", "2024"]
