"""Pure P05 L0/L1/L2 construction with sealed mature outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION_TIME,
    SOURCE_ID,
    TARGET_ID,
    TARGET_VALUE,
)
from labels.service import aggregate_l1, evidence_score_l2


@dataclass(frozen=True)
class MeasurementResult:
    matrices: dict[str, Any]
    inputs: pd.DataFrame
    sealed_outcomes: pd.DataFrame
    l3_capability: dict[str, Any]
    measurement_variables: list[dict[str, object]]
    channel_capability: dict[str, object]
    anchor_capability: dict[str, object]


def build_measurement_inputs(
    *,
    risk_sets: pd.DataFrame,
    evidence: pd.DataFrame,
    expected_sources: dict[str, str],
    horizon_months: int,
    columns: dict[str, str],
    pending_status: str,
    unavailable_status: str,
    insufficient_channels_reason: str,
    anchor_source_ids: list[str],
    source_profiles: dict[str, str] | None = None,
    l2_scoring: dict[str, Any] | None = None,
) -> MeasurementResult:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    mature = columns[MATURE]
    eligible = columns[ELIGIBLE]
    source = columns[SOURCE_ID]
    channel = columns[CHANNEL_ID]
    availability = columns[AVAILABILITY_DATE]
    outcome = columns[OUTCOME]
    required_risk = {firm, year, prediction, mature, eligible}
    required_evidence = {firm, year, source, channel, availability, outcome}
    if not required_risk.issubset(risk_sets.columns):
        raise ValueError("P05 risk-set contract is incomplete")
    if not required_evidence.issubset(evidence.columns):
        raise ValueError("P05 evidence contract is incomplete")
    evidence_by_key: dict[tuple[str, int], list[dict[str, object]]] = {}
    for raw in evidence.to_dict(orient="records"):
        row = {str(key): value for key, value in raw.items()}
        evidence_by_key.setdefault((str(row[firm]), int(row[year])), []).append(row)

    input_rows: list[dict[str, object]] = []
    sealed_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    expected_source_ids = sorted(expected_sources)
    expected_channels = sorted(set(expected_sources.values()))
    l2_configuration = _l2_configuration(
        expected_source_ids=expected_source_ids,
        source_profiles=source_profiles or {},
        scoring=l2_scoring or {},
    )
    for raw in cast(list[dict[str, Any]], risk_sets.to_dict(orient="records")):
        risk = {str(key): value for key, value in raw.items()}
        key = (str(risk[firm]), int(risk[year]))
        prediction_time = pd.Timestamp(risk[prediction])
        horizon_end = prediction_time + pd.DateOffset(months=horizon_months)
        observed: dict[str, bool | None] = {item: None for item in expected_source_ids}
        observed_lag_days: dict[str, int | None] = {item: None for item in expected_source_ids}
        for event in evidence_by_key.get(key, []):
            source_id = str(event[source])
            event_time = pd.Timestamp(cast(Any, event[availability]))
            if event_time <= prediction_time or event_time > horizon_end:
                continue
            value = event[outcome]
            parsed = None if pd.isna(cast(Any, value)) else bool(value)
            observed[source_id] = parsed
            observed_lag_days[source_id] = int((event_time - prediction_time).days)
        for source_id in expected_source_ids:
            input_rows.append(
                {
                    firm: key[0],
                    year: key[1],
                    columns[TARGET_ID]: f"L0:{source_id}",
                    outcome: observed[source_id] if bool(risk[mature]) else None,
                }
            )
        l1 = aggregate_l1(observed) if bool(risk[mature]) and bool(risk[eligible]) else None
        input_rows.append({firm: key[0], year: key[1], columns[TARGET_ID]: "L1", outcome: l1})
        channel_values: dict[str, bool | None] = {item: None for item in expected_channels}
        for channel_id in expected_channels:
            values = [
                value
                for source_id, value in observed.items()
                if expected_sources[source_id] == channel_id
            ]
            channel_values[channel_id] = aggregate_l1(
                {str(index): value for index, value in enumerate(values)}
            )
        channel_scores = _l2_channel_scores(
            source_outcomes=observed,
            source_lag_days=observed_lag_days,
            expected_sources=expected_sources,
            source_profiles=source_profiles or {},
            configuration=l2_configuration,
        )
        matrix_rows.append(
            {
                FIRM_ID: key[0],
                FISCAL_YEAR: key[1],
                MATURE: bool(risk[mature]),
                "source_outcomes": observed,
                "channel_outcomes": channel_values,
                "channel_evidence_scores": channel_scores,
                "observed_source_count": sum(value is not None for value in observed.values()),
                "observed_channel_count": sum(
                    value is not None for value in channel_values.values()
                ),
                "l1": l1,
                "l2_score": evidence_score_l2(channel_scores)
                if l2_configuration["status"] == "AVAILABLE"
                else None,
                "l2_score_status": l2_configuration["status"],
                "l2_score_reason_code": l2_configuration["reason_code"],
            }
        )
        if l1 is not None:
            sealed_rows.append({firm: key[0], year: key[1], outcome: l1})

    inputs = pd.DataFrame(input_rows, columns=[firm, year, columns[TARGET_ID], outcome])
    inputs[firm] = inputs[firm].astype("string")
    inputs[year] = inputs[year].astype("int16")
    inputs[columns[TARGET_ID]] = inputs[columns[TARGET_ID]].astype("string")
    inputs[outcome] = inputs[outcome].astype("boolean")
    sealed = pd.DataFrame(sealed_rows, columns=[firm, year, outcome])
    if not sealed.empty:
        sealed[firm] = sealed[firm].astype("string")
        sealed[year] = sealed[year].astype("int16")
        sealed[outcome] = sealed[outcome].astype(bool)
    channel_count = len(expected_channels)
    capability: dict[str, Any] = {
        "status": pending_status if channel_count >= 2 else unavailable_status,
        "pilot_executed": False,
        "channel_count": channel_count,
        "reason": None if channel_count >= 2 else insufficient_channels_reason,
        "gate1_candidate": "L3_fixed_pi" if channel_count >= 2 else None,
        "hierarchical_pi_role": "sensitivity_only",
    }
    variables: list[dict[str, object]] = [
        {
            SOURCE_ID: source_id,
            CHANNEL_ID: expected_sources[source_id],
            "variable_id": f"L0:{source_id}",
            "role": "single_source_benchmark",
            "missing_source_encoded_as_zero": False,
        }
        for source_id in expected_source_ids
    ]
    variables.extend(
        [
            {
                "variable_id": "L1",
                "role": "primary_binary_union",
                "missing_source_encoded_as_zero": False,
            },
            {
                "variable_id": "L2",
                "role": "observed_channel_normalized_score",
                "normalization_denominator": "observed_channels_only",
                "status": l2_configuration["status"],
                "reason_code": l2_configuration["reason_code"],
                "formula": l2_configuration.get("formula"),
            },
        ]
    )
    anchor_ids = sorted(set(anchor_source_ids) & set(expected_source_ids))
    return MeasurementResult(
        matrices={
            "expected_sources": expected_sources,
            "expected_channels": expected_channels,
            "rows": matrix_rows,
            "missing_source_encoded_as_zero": False,
            "l2_scoring": l2_configuration,
        },
        inputs=inputs,
        sealed_outcomes=sealed,
        l3_capability=capability,
        measurement_variables=variables,
        channel_capability={
            "status": "AVAILABLE" if channel_count >= 2 else unavailable_status,
            "channel_count": channel_count,
            "strict_holdout_possible": channel_count >= 2,
            "reason_code": None if channel_count >= 2 else insufficient_channels_reason,
        },
        anchor_capability={
            "status": "AVAILABLE" if anchor_ids else unavailable_status,
            "anchor_source_ids": anchor_ids,
            "reason_code": None if anchor_ids else "ANCHOR_UNAVAILABLE",
            "clean_positive_assumption": False,
        },
    )


def _l2_configuration(
    *,
    expected_source_ids: list[str],
    source_profiles: dict[str, str],
    scoring: dict[str, Any],
) -> dict[str, Any]:
    formula = scoring.get("formula")
    if formula is None:
        return {
            "status": "EMPIRICALLY_PENDING",
            "reason_code": "L2_SCORING_FORMULA_NOT_LOCKED",
            "required_config_keys": [
                "measurement.l2_scoring.formula",
                "measurement.l2_scoring.source_quality_by_profile",
                "measurement.l2_scoring.delay_half_life_days",
            ],
        }
    if formula != scoring.get("supported_formula"):
        raise ValueError(f"unsupported L2 scoring formula: {formula}")
    if scoring.get("channel_weights") != "equal":
        raise ValueError("primary L2 requires equal channel weights")
    raw_quality = scoring.get("source_quality_by_profile")
    if not isinstance(raw_quality, dict):
        raise ValueError("L2 source quality profile mapping is required")
    quality = cast(dict[str, Any], raw_quality)
    missing_sources = sorted(set(expected_source_ids) - set(source_profiles))
    missing_profiles = sorted(set(source_profiles.values()) - set(quality))
    half_life = scoring.get("delay_half_life_days")
    if missing_sources or missing_profiles or not isinstance(half_life, (int, float)):
        return {
            "status": "EMPIRICALLY_PENDING",
            "reason_code": "L2_QUALITY_OR_DELAY_PARAMETERS_NOT_LOCKED",
            "missing_source_profiles": missing_sources,
            "missing_quality_profiles": missing_profiles,
            "required_config_keys": [
                "measurement.l2_scoring.source_quality_by_profile",
                "measurement.l2_scoring.delay_half_life_days",
            ],
        }
    if float(half_life) <= 0:
        raise ValueError("L2 delay half-life must be positive")
    for profile_id in set(source_profiles.values()):
        value = quality[profile_id]
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ValueError(f"L2 quality for profile={profile_id} must be in [0, 1]")
    return {
        "status": "AVAILABLE",
        "reason_code": None,
        "formula": formula,
        "channel_weights": "equal",
        "source_quality_by_profile": {
            profile: float(quality[profile]) for profile in sorted(set(source_profiles.values()))
        },
        "delay_half_life_days": float(half_life),
        "missing_source_encoded_as_zero": False,
    }


def _l2_channel_scores(
    *,
    source_outcomes: dict[str, bool | None],
    source_lag_days: dict[str, int | None],
    expected_sources: dict[str, str],
    source_profiles: dict[str, str],
    configuration: dict[str, Any],
) -> dict[str, float | None]:
    channels = sorted(set(expected_sources.values()))
    result: dict[str, float | None] = {channel: None for channel in channels}
    if configuration["status"] != "AVAILABLE":
        return result
    qualities = cast(dict[str, float], configuration["source_quality_by_profile"])
    half_life = float(configuration["delay_half_life_days"])
    for channel in channels:
        values: list[float] = []
        for source_id, source_channel in expected_sources.items():
            if source_channel != channel or source_outcomes[source_id] is None:
                continue
            lag = source_lag_days[source_id]
            if lag is None:
                continue
            quality = qualities[source_profiles[source_id]]
            timeliness = math.exp(-math.log(2.0) * max(0, lag) / half_life)
            values.append(float(bool(source_outcomes[source_id])) * quality * timeliness)
        if values:
            result[channel] = sum(values) / len(values)
    return result


def summarize_fold_eligibility(
    *,
    sealed_outcomes: pd.DataFrame,
    initial_outer_year: int,
    confirmatory_years: list[int],
    prospective_year: int,
    confirmatory_positive_minimum: int,
    sensitivity_positive_range: tuple[int, int],
    columns: dict[str, str],
) -> list[dict[str, object]]:
    """Assign only aggregate fold roles; row-level outer outcomes never leave P05."""
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    counts = (
        sealed_outcomes.groupby(year, sort=True)[outcome].agg(["count", "sum"])
        if not sealed_outcomes.empty
        else pd.DataFrame()
    )

    def row_for(fold_year: int, configured_role: str) -> dict[str, object]:
        row_count = int(str(counts.loc[fold_year, "count"])) if fold_year in counts.index else 0
        positive_count = int(str(counts.loc[fold_year, "sum"])) if fold_year in counts.index else 0
        if configured_role in {"initial", "prospective"}:
            role = f"{configured_role}_separate"
            reason = None
        elif positive_count >= confirmatory_positive_minimum:
            role = "confirmatory"
            reason = None
        elif sensitivity_positive_range[0] <= positive_count <= sensitivity_positive_range[1]:
            role = "sensitivity"
            reason = "POSITIVE_COUNT_BELOW_CONFIRMATORY"
        else:
            role = "prospective_or_descriptive"
            reason = "INSUFFICIENT_POSITIVES"
        return {
            OUTER_FOLD: str(fold_year),
            "configured_role": configured_role,
            "assigned_role": role,
            "mature_row_count": row_count,
            "positive_count": positive_count,
            "reason_code": reason,
            "aggregate_only": True,
            "row_level_outer_labels_exposed": False,
        }

    return [
        row_for(initial_outer_year, "initial"),
        *[row_for(value, "fully_nested") for value in confirmatory_years],
        row_for(prospective_year, "prospective"),
    ]


def measurement_target_frame(
    *,
    matrices: dict[str, Any],
    measurement_id: str,
    minimum_observed_channels: int,
    columns: dict[str, str],
) -> pd.DataFrame:
    """Materialize a soft target without converting unobserved channels to zero."""
    if minimum_observed_channels < 1:
        raise ValueError("minimum observed channels must be positive")
    raw_rows = matrices.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError("source-channel matrix rows are required")
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    value = columns[TARGET_VALUE]
    output: list[dict[str, object]] = []
    for raw in cast(list[object], raw_rows):
        if not isinstance(raw, dict):
            continue
        row = cast(dict[str, Any], raw)
        if int(row.get("observed_channel_count", 0)) < minimum_observed_channels:
            target: object = None
        elif measurement_id == "L2":
            target = row.get("l2_score")
        elif measurement_id == "L3_fixed_pi":
            target = row.get("l3_posterior_mean")
        else:
            raise ValueError(f"measurement={measurement_id}: unsupported soft target")
        output.append(
            {
                firm: str(row[FIRM_ID]),
                year: int(row[FISCAL_YEAR]),
                value: target,
            }
        )
    frame = pd.DataFrame(output, columns=[firm, year, value])
    frame[firm] = frame[firm].astype("string")
    frame[year] = frame[year].astype("int16")
    frame[value] = frame[value].astype("float64")
    observed = frame[value].dropna()
    if not observed.empty and ((observed < 0).any() or (observed > 1).any()):
        raise ValueError(f"measurement={measurement_id}: target values must be in [0, 1]")
    return frame
