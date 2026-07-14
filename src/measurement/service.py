"""Pure P05 L0/L1/L2 construction with sealed mature outcomes."""

from __future__ import annotations

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
    for raw in cast(list[dict[str, Any]], risk_sets.to_dict(orient="records")):
        risk = {str(key): value for key, value in raw.items()}
        key = (str(risk[firm]), int(risk[year]))
        prediction_time = pd.Timestamp(risk[prediction])
        horizon_end = prediction_time + pd.DateOffset(months=horizon_months)
        observed: dict[str, bool | None] = {item: None for item in expected_source_ids}
        for event in evidence_by_key.get(key, []):
            source_id = str(event[source])
            event_time = pd.Timestamp(cast(Any, event[availability]))
            if event_time <= prediction_time or event_time > horizon_end:
                continue
            value = event[outcome]
            parsed = None if pd.isna(cast(Any, value)) else bool(value)
            observed[source_id] = parsed
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
        matrix_rows.append(
            {
                FIRM_ID: key[0],
                FISCAL_YEAR: key[1],
                MATURE: bool(risk[mature]),
                "source_outcomes": observed,
                "channel_outcomes": channel_values,
                "observed_source_count": sum(value is not None for value in observed.values()),
                "l1": l1,
                "l2_score": evidence_score_l2(channel_values),
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
