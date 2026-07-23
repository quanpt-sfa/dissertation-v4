"""Score sealed known cases only after Gate 2 has been frozen."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, LEARNER_ID, PREDICTION


def evaluate_known_cases(
    *,
    cases: list[dict[str, Any]],
    predictions: pd.DataFrame,
    expected_case_ids: list[str],
    minimum_cases: int,
    below_median_cases: int,
    scenario_fraction_min: float,
    strong_percentile: float,
    columns: dict[str, str],
    weak_percentile: float = 0.5,
    weak_percentile: float = 0.5,
) -> list[dict[str, Any]]:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    learner = columns[LEARNER_ID]
    prediction = columns[PREDICTION]
    observed_ids = {str(case["case_id"]) for case in cases}
    unexpected = observed_ids - set(expected_case_ids)
    if unexpected:
        raise ValueError(f"unregistered known case IDs: {sorted(unexpected)}")
    results: list[dict[str, Any]] = []
    for case in cases:
        case_frame = predictions.loc[
            (predictions[firm].astype(str) == str(case[firm]))
            & (predictions[year].astype(int) == int(case[year]))
        ]
        for _, row in case_frame.iterrows():
            reference = cast(
                pd.Series,
                pd.to_numeric(
                    predictions.loc[
                        (predictions[year] == row[year]) & (predictions[learner] == row[learner]),
                        prediction,
                    ]
                ),
            )
            percentile = (
                float(reference.le(float(row[prediction])).mean()) if len(reference) else None
            )
            exact_lower_tail = (
                (int(reference.le(float(row[prediction])).sum()) + 1) / (len(reference) + 1)
                if len(reference)
                else None
            )
            results.append(
                {
                    "record_type": "case_result",
                    "case_id": str(case["case_id"]),
                    "firm_id": str(case[firm]),
                    FISCAL_YEAR: int(case[year]),
                    LEARNER_ID: str(row[learner]),
                    PREDICTION: float(row[prediction]),
                    "percentile": percentile,
                    "exact_permutation_lower_tail_p": exact_lower_tail,
                    "permutation_reference_size": len(reference),
                }
            )
    scenario_ids = sorted({str(item[LEARNER_ID]) for item in results})
    eligible_case_ids = sorted({str(item["case_id"]) for item in results})
    below_counts = {
        scenario: len(
            {
                str(item["case_id"])
                for item in results
                if item[LEARNER_ID] == scenario
                and item["percentile"] is not None
                and float(item["percentile"]) < weak_percentile
            }
        )
        for scenario in scenario_ids
    }
    failing_scenarios = sum(value >= below_median_cases for value in below_counts.values())
    scenario_fraction = failing_scenarios / len(scenario_ids) if scenario_ids else None
    strong_counts = {
        scenario: len(
            {
                str(item["case_id"])
                for item in results
                if item[LEARNER_ID] == scenario
                and item["percentile"] is not None
                and float(item["percentile"]) < strong_percentile
            }
        )
        for scenario in scenario_ids
    }
    strong_scenario_fraction = (
        sum(value >= minimum_cases for value in strong_counts.values()) / len(scenario_ids)
        if scenario_ids
        else None
    )
    strong = bool(
        strong_scenario_fraction is not None and strong_scenario_fraction >= scenario_fraction_min
    )
    enough = len(eligible_case_ids) >= minimum_cases
    veto = bool(
        enough and scenario_fraction is not None and scenario_fraction >= scenario_fraction_min
    )
    results.append(
        {
            "record_type": "summary",
            "status": "PASS" if results else "INSUFFICIENT_EVIDENCE",
            "reason_code": None if results else "KNOWN_CASES_UNAVAILABLE",
            "configured_case_count": len(cases),
            "eligible_case_count": len(eligible_case_ids),
            "eligible_case_ids": eligible_case_ids,
            "minimum_cases_met": enough,
            "scenario_count": len(scenario_ids),
            "below_median_scenario_fraction": scenario_fraction,
            "soft_veto": veto,
            "strong_falsification": strong,
            "strong_falsification_scenario_fraction": strong_scenario_fraction,
            "fewer_than_minimum_role": None if enough else "casewise_reporting_only",
            "permutation_method": "exact_within_outer_year_model_rank",
            "may_upgrade_failed_gate": False,
        }
    )
    return results
