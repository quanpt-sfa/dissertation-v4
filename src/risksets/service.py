"""Pure P04 maturity and prospective-cohort classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    PREDICTION_TIME,
    SOURCE_ID,
)


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
    evidence: pd.DataFrame | None = None,
    sensitivity_horizons_months: list[int] | None = None,
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
    horizons = [horizon_months, *(sensitivity_horizons_months or [])]
    maturity_counts = {
        str(value): int(
            (prediction_values + pd.DateOffset(months=value) <= pd.Timestamp(data_cutoff)).sum()
        )
        for value in sorted(set(horizons))
    }
    source_curves = _source_maturity_curves(
        panel=panel,
        evidence=evidence,
        horizons_months=sorted(set(horizons)),
        data_cutoff=data_cutoff,
        columns=columns,
    )
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
            "mature_count_by_horizon_months": maturity_counts,
            "source_maturity_curves": source_curves,
            "source_maturity_curves_executed": evidence is not None,
        },
    )


def _source_maturity_curves(
    *,
    panel: pd.DataFrame,
    evidence: pd.DataFrame | None,
    horizons_months: list[int],
    data_cutoff: datetime,
    columns: dict[str, str],
) -> list[dict[str, object]]:
    if evidence is None:
        return []
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    availability = columns[AVAILABILITY_DATE]
    source = columns[SOURCE_ID]
    channel = columns[CHANNEL_ID]
    required = {firm, year, availability, source, channel}
    if not required.issubset(evidence.columns):
        raise ValueError("evidence ledger is incomplete for source maturity curves")
    linked = evidence.merge(
        panel.loc[:, [firm, year, prediction]],
        on=[firm, year],
        how="inner",
        validate="many_to_one",
    )
    linked["_detection_lag_days"] = (
        pd.to_datetime(linked[availability]) - pd.to_datetime(linked[prediction])
    ).dt.days
    rows: list[dict[str, object]] = []
    for (source_id, channel_id), frame in linked.groupby([source, channel], sort=True):
        lags = frame["_detection_lag_days"].to_numpy(dtype=float)
        prediction_dates = pd.to_datetime(frame[prediction])
        availability_dates = pd.to_datetime(frame[availability])
        cutoff = pd.Timestamp(data_cutoff)
        pre_prediction = availability_dates < prediction_dates
        same_day = availability_dates == prediction_dates
        after_prediction = availability_dates > prediction_dates
        after_cutoff = availability_dates > cutoff
        in_horizon_counts: dict[str, int] = {}
        post_horizon_counts: dict[str, int] = {}
        in_horizon_fractions: dict[str, float] = {}
        for horizon in horizons_months:
            horizon_end = prediction_dates + pd.DateOffset(months=horizon)
            in_horizon = after_prediction & (availability_dates <= horizon_end) & ~after_cutoff
            post_horizon = (availability_dates > horizon_end) & ~after_cutoff
            in_horizon_counts[str(horizon)] = int(in_horizon.sum())
            post_horizon_counts[str(horizon)] = int(post_horizon.sum())
            in_horizon_fractions[str(horizon)] = float(in_horizon.mean()) if len(frame) else 0.0
        rows.append(
            {
                SOURCE_ID: str(source_id),
                CHANNEL_ID: str(channel_id),
                "observed_event_count": len(lags),
                "pre_prediction_event_count": int(pre_prediction.sum()),
                "negative_detection_lag_count": int((lags < 0).sum()),
                "same_day_detection_count": int(same_day.sum()),
                "future_detection_event_count": int(after_prediction.sum()),
                "post_data_cutoff_event_count": int(after_cutoff.sum()),
                "median_detection_lag_days": float(pd.Series(lags).median()) if len(lags) else None,
                "in_horizon_event_count_by_months": in_horizon_counts,
                "post_horizon_event_count_by_months": post_horizon_counts,
                "linked_event_in_horizon_fraction_by_months": in_horizon_fractions,
                "metric_scope": "observed_linked_events_only",
                "source_opportunity_coverage_rate": None,
            }
        )
    return rows
