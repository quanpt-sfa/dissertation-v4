"""Pure P04 maturity and prospective-cohort classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.semantic_keys import ELIGIBLE, FIRM_ID, FISCAL_YEAR, MATURE, PREDICTION_TIME


@dataclass(frozen=True)
class RiskSetResult:
    risk_sets: pd.DataFrame
    prospective_set: pd.DataFrame
    censoring_registry: list[dict[str, object]]
    maturity_audit: dict[str, object]


def build_risk_set(
    *,
    panel: pd.DataFrame,
    data_cutoff: datetime,
    horizon_months: int,
    columns: dict[str, str],
) -> RiskSetResult:
    """Classify maturity without converting immature observations to negatives."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    required = [firm, year, prediction]
    missing = set(required) - set(panel.columns)
    if missing:
        raise ValueError(f"P04 panel missing columns {sorted(missing)}")
    if horizon_months <= 0:
        raise ValueError("horizon must be positive")
    result = panel.loc[:, required].copy()
    prediction_values = pd.to_datetime(result[prediction], errors="raise")
    horizon_end = prediction_values + pd.DateOffset(months=horizon_months)
    result[columns[ELIGIBLE]] = True
    result[columns[MATURE]] = horizon_end <= pd.Timestamp(data_cutoff)
    result[firm] = result[firm].astype("string")
    result[year] = result[year].astype("int16")
    result[prediction] = prediction_values.astype("datetime64[ns]")
    result[columns[ELIGIBLE]] = result[columns[ELIGIBLE]].astype(bool)
    result[columns[MATURE]] = result[columns[MATURE]].astype(bool)
    result = result.sort_values([firm, year], kind="stable").reset_index(drop=True)
    mature_count = int(result[columns[MATURE]].sum())
    prospective = result.loc[~result[columns[MATURE]]].copy().reset_index(drop=True)
    censoring_registry: list[dict[str, object]] = [
        {
            FIRM_ID: str(row[firm]),
            FISCAL_YEAR: int(row[year]),
            "classification": "complete_mature_followup"
            if bool(row[columns[MATURE]])
            else "prospective_immature",
            "eligible_for_retrospective_evaluation": bool(row[columns[MATURE]]),
            "assigned_negative_due_to_exit_or_immaturity": False,
        }
        for row in result.to_dict(orient="records")
    ]
    return RiskSetResult(
        risk_sets=result,
        prospective_set=prospective,
        censoring_registry=censoring_registry,
        maturity_audit={
            "data_cutoff": data_cutoff.isoformat(),
            "horizon_months": horizon_months,
            "firm_year_count": len(result),
            "eligible_count": int(result[columns[ELIGIBLE]].sum()),
            "mature_count": mature_count,
            "prospective_count": len(result) - mature_count,
            "immature_assigned_negative_count": 0,
            "exit_or_code_change_assigned_negative_count": 0,
        },
    )
