"""Pure P04 maturity and prospective-cohort classification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from core.semantic_keys import (
    ANNUAL_MEASUREMENT_MATURE,
    AVAILABILITY_BASIS,
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    MATURITY_STATUS,
    PREDICTION_TIME,
    REQUIRED_SANCTION_YEAR,
    S3_NEXT_YEAR_MATURE,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    SOURCE_YEAR_COMPLETE,
    TEMPORAL_ROLE,
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
    sanction_complete_through_year: int = 0,
    sanction_incomplete_years: set[int] | None = None,
    primary_target_id: str | None = None,
) -> RiskSetResult:
    """Classify endpoint-specific maturity without converting immaturity to negatives."""
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
    annual_mature_column = columns.get(ANNUAL_MEASUREMENT_MATURE, ANNUAL_MEASUREMENT_MATURE)
    s3_mature_column = columns.get(S3_NEXT_YEAR_MATURE, S3_NEXT_YEAR_MATURE)
    required_sanction_year_column = columns.get(REQUIRED_SANCTION_YEAR, REQUIRED_SANCTION_YEAR)
    source_year_complete_column = columns.get(SOURCE_YEAR_COMPLETE, SOURCE_YEAR_COMPLETE)
    maturity_status_column = columns.get(MATURITY_STATUS, MATURITY_STATUS)
    result[columns[ELIGIBLE]] = True
    annual_mature = prediction_values <= pd.Timestamp(data_cutoff)
    required_sanction_year = result[year].astype(int) + 1
    incomplete = sanction_incomplete_years or set()
    source_complete = required_sanction_year.map(
        lambda value: value <= sanction_complete_through_year and value not in incomplete
    )
    result[annual_mature_column] = annual_mature
    result[required_sanction_year_column] = required_sanction_year
    result[source_year_complete_column] = source_complete
    result[s3_mature_column] = source_complete
    legacy_delayed_mode = sanction_complete_through_year == 0 and primary_target_id is None
    if legacy_delayed_mode:
        mature_values = prediction_values + pd.DateOffset(months=horizon_months) <= pd.Timestamp(
            data_cutoff
        )
        maturity_status = pd.Series(
            [
                "LEGACY_DELAYED_MATURE" if bool(value) else "LEGACY_DELAYED_IMMATURE"
                for value in mature_values
            ],
            index=result.index,
            dtype="string",
        )
    else:
        mature_values, maturity_status = _primary_maturity(
            primary_target_id=primary_target_id,
            annual_mature=annual_mature,
            s3_mature=source_complete,
        )
    result[columns[MATURE]] = mature_values
    result[maturity_status_column] = maturity_status
    result[firm] = result[firm].astype("string")
    result[year] = result[year].astype("int16")
    result[prediction] = prediction_values.astype("datetime64[ns]")
    result[columns[ELIGIBLE]] = result[columns[ELIGIBLE]].astype(bool)
    result[columns[MATURE]] = result[columns[MATURE]].astype(bool)
    result[annual_mature_column] = result[annual_mature_column].astype(bool)
    result[s3_mature_column] = result[s3_mature_column].astype(bool)
    result[required_sanction_year_column] = result[required_sanction_year_column].astype("int16")
    result[source_year_complete_column] = result[source_year_complete_column].astype(bool)
    result[maturity_status_column] = result[maturity_status_column].astype("string")
    result = result.sort_values([firm, year], kind="stable").reset_index(drop=True)
    mature_count = int(result[columns[MATURE]].sum())
    prospective = result.loc[~result[columns[MATURE]]].copy().reset_index(drop=True)
    censoring_registry: list[dict[str, object]] = [
        {
            FIRM_ID: str(row[firm]),
            FISCAL_YEAR: int(row[year]),
            "classification": "complete_mature_followup"
            if bool(row[columns[MATURE]])
            else str(row[maturity_status_column]),
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
    annual_availability = _annual_measurement_availability(
        panel=panel,
        evidence=evidence,
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
            "primary_target_id": primary_target_id,
            "primary_target_status": (
                "LOCKED" if primary_target_id is not None else "PRIMARY_TARGET_NOT_LOCKED"
            ),
            "annual_measurement_mature_count": int(annual_mature.sum()),
            "s3_next_year_mature_count": int(source_complete.sum()),
            "sanction_source_completeness": {
                "complete_through_year": sanction_complete_through_year,
                "incomplete_years": sorted(incomplete),
            },
            "s3_maturity_rule": "required_sanction_year_equals_fiscal_year_plus_one",
            "source_maturity_curves": source_curves,
            "delayed_verification_maturity_curves": source_curves,
            "annual_measurement_availability": annual_availability,
            "source_maturity_curves_executed": evidence is not None,
        },
    )


def _primary_maturity(
    *,
    primary_target_id: str | None,
    annual_mature: pd.Series,
    s3_mature: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if primary_target_id is None:
        return (
            pd.Series(False, index=annual_mature.index, dtype=bool),
            pd.Series("PRIMARY_TARGET_NOT_LOCKED", index=annual_mature.index, dtype="string"),
        )
    if primary_target_id == "L1_ANNUAL":
        values = annual_mature.astype(bool)
        rule = "ANNUAL_MEASUREMENT_MATURE"
    elif primary_target_id in {"S3_BROAD", "S3_REPORTING", "S3_CONTENT", "S3_TIMELINESS"}:
        values = s3_mature.astype(bool)
        rule = "S3_NEXT_YEAR_MATURE"
    elif primary_target_id in {"L1_REPORTING", "L1_CONTENT_STRICT"}:
        values = annual_mature.astype(bool) & s3_mature.astype(bool)
        rule = "ANNUAL_AND_S3_MATURE"
    else:
        raise ValueError(f"unknown primary target={primary_target_id}")
    status = pd.Series(
        [rule if bool(value) else f"{rule}_INCOMPLETE" for value in values],
        index=values.index,
        dtype="string",
    )
    return values, status


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
    temporal = columns.get(TEMPORAL_ROLE, TEMPORAL_ROLE)
    if temporal in evidence.columns:
        delayed = evidence.loc[evidence[temporal].eq("delayed_verification")].copy()
    else:
        delayed = evidence
    linked = delayed.merge(
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


def _annual_measurement_availability(
    *,
    panel: pd.DataFrame,
    evidence: pd.DataFrame | None,
    columns: dict[str, str],
) -> list[dict[str, object]]:
    if evidence is None:
        return []
    temporal = columns.get(TEMPORAL_ROLE, TEMPORAL_ROLE)
    if temporal not in evidence.columns:
        return []
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    availability = columns[AVAILABILITY_DATE]
    source = columns[SOURCE_ID]
    channel = columns[CHANNEL_ID]
    opportunity = columns.get(SOURCE_OPPORTUNITY, SOURCE_OPPORTUNITY)
    availability_basis = columns.get(AVAILABILITY_BASIS, AVAILABILITY_BASIS)
    required = {firm, year, availability, source, channel, opportunity, availability_basis}
    if not required.issubset(evidence.columns):
        raise ValueError("annual evidence ledger is incomplete for availability audit")
    annual = evidence.loc[evidence[temporal].eq("annual_measurement_at_anchor")].copy()
    linked = annual.merge(
        panel.loc[:, [firm, year, prediction]],
        on=[firm, year],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (source_id, channel_id), frame in linked.groupby([source, channel], sort=True):
        opportunity_values = frame[opportunity].astype("boolean")
        anchor_match = pd.to_datetime(frame[availability]).eq(pd.to_datetime(frame[prediction]))
        rows.append(
            {
                SOURCE_ID: str(source_id),
                CHANNEL_ID: str(channel_id),
                "annual_measurement_record_count": len(frame),
                "available_at_common_anchor_count": int(anchor_match.sum()),
                "annual_anchor_mismatch_count": int((~anchor_match).sum()),
                "source_opportunity_observed_count": int(opportunity_values.eq(True).sum()),
                "source_opportunity_not_observed_count": int(opportunity_values.eq(False).sum()),
                "source_opportunity_unknown_count": int(opportunity_values.isna().sum()),
                AVAILABILITY_BASIS: sorted(
                    str(value) for value in frame[availability_basis].dropna().unique()
                ),
                "included_in_delayed_maturity_curve": False,
            }
        )
    return rows
