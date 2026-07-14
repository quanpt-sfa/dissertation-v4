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
    SOURCE_OPPORTUNITY,
    TARGET_ID,
    TARGET_VALUE,
    TEMPORAL_ROLE,
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


def l3_channel_capability_allows_pilot(capability: dict[str, object]) -> bool:
    """Keep structural channel failure ahead of missing empirical parameters."""
    channel_count = capability.get("channel_count")
    if not isinstance(channel_count, int) or isinstance(channel_count, bool):
        raise ValueError("L3 capability requires an integer channel_count")
    if channel_count >= 2:
        return True
    capability.update(
        {
            "status": "UNAVAILABLE_BY_DESIGN",
            "pilot_executed": False,
            "reason_code": "INSUFFICIENT_CHANNELS",
        }
    )
    return False


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
    source_temporal_roles: dict[str, str] | None = None,
    explicit_negative_allowed: dict[str, bool] | None = None,
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
    temporal = columns.get(TEMPORAL_ROLE, TEMPORAL_ROLE)
    opportunity = columns.get(SOURCE_OPPORTUNITY, SOURCE_OPPORTUNITY)
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
    temporal_roles = source_temporal_roles or {
        source_id: "delayed_verification" for source_id in expected_source_ids
    }
    negative_policy = explicit_negative_allowed or {
        source_id: True for source_id in expected_source_ids
    }
    if set(temporal_roles) != set(expected_source_ids):
        raise ValueError("P05 temporal roles must cover every logical source")
    if set(negative_policy) != set(expected_source_ids):
        raise ValueError("P05 explicit-negative policy must cover every logical source")
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
        source_events: dict[str, list[tuple[bool | None, int]]] = {
            item: [] for item in expected_source_ids
        }
        source_opportunity_values: dict[str, list[bool | None]] = {
            item: [] for item in expected_source_ids
        }
        observed: dict[str, bool | None] = {item: None for item in expected_source_ids}
        observed_opportunities: dict[str, bool | None] = {
            item: None for item in expected_source_ids
        }
        observed_lag_days: dict[str, int | None] = {item: None for item in expected_source_ids}
        for event in evidence_by_key.get(key, []):
            source_id = str(event[source])
            if source_id not in source_events:
                raise ValueError(f"P05 evidence source is not registered: {source_id}")
            event_time = pd.Timestamp(cast(Any, event[availability]))
            row_temporal_role = str(event.get(temporal, temporal_roles[source_id]))
            if row_temporal_role != temporal_roles[source_id]:
                raise ValueError(f"source={source_id}: evidence temporal role drift")
            if row_temporal_role == "annual_measurement_at_anchor":
                if event_time != prediction_time:
                    raise ValueError(
                        f"source={source_id}: annual evidence must be available at anchor"
                    )
                lag_days = 0
            elif row_temporal_role == "delayed_verification":
                if event_time <= prediction_time or event_time > horizon_end:
                    continue
                lag_days = int((event_time - prediction_time).days)
            else:
                raise ValueError(f"source={source_id}: unsupported temporal role")
            value = event[outcome]
            parsed = None if pd.isna(cast(Any, value)) else bool(value)
            if parsed is False and negative_policy[source_id] is not True:
                raise ValueError(f"source={source_id}: explicit negative is not allowed")
            source_events[source_id].append((parsed, lag_days))
            raw_opportunity = event.get(opportunity)
            source_opportunity_values[source_id].append(
                None
                if raw_opportunity is None or pd.isna(cast(Any, raw_opportunity))
                else bool(raw_opportunity)
            )
        for source_id, events in source_events.items():
            values = {str(index): value for index, (value, _) in enumerate(events)}
            opportunity_values = source_opportunity_values[source_id]
            observed_opportunities[source_id] = _aggregate_opportunity(opportunity_values)
            aggregate = aggregate_l1(
                values,
                {str(index): opportunity_values[index] for index in range(len(opportunity_values))}
                if opportunity_values
                else None,
            )
            observed[source_id] = aggregate
            if aggregate is not None:
                observed_lag_days[source_id] = min(
                    lag for value, lag in events if value is aggregate
                )
        for source_id in expected_source_ids:
            input_rows.append(
                {
                    firm: key[0],
                    year: key[1],
                    columns[TARGET_ID]: f"L0:{source_id}",
                    outcome: observed[source_id]
                    if temporal_roles[source_id] == "annual_measurement_at_anchor"
                    or bool(risk[mature])
                    else None,
                }
            )
        l1 = (
            aggregate_l1(observed, observed_opportunities)
            if bool(risk[mature]) and bool(risk[eligible])
            else None
        )
        input_rows.append({firm: key[0], year: key[1], columns[TARGET_ID]: "L1", outcome: l1})
        channel_values: dict[str, bool | None] = {item: None for item in expected_channels}
        channel_opportunities: dict[str, bool | None] = {item: None for item in expected_channels}
        for channel_id in expected_channels:
            channel_source_ids = [
                source_id
                for source_id in expected_source_ids
                if expected_sources[source_id] == channel_id
            ]
            values = [observed[source_id] for source_id in channel_source_ids]
            opportunities = [observed_opportunities[source_id] for source_id in channel_source_ids]
            channel_opportunities[channel_id] = _aggregate_opportunity(opportunities)
            channel_values[channel_id] = aggregate_l1(
                {str(index): value for index, value in enumerate(values)},
                {str(index): value for index, value in enumerate(opportunities)},
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
                ELIGIBLE: bool(risk[eligible]),
                MATURE: bool(risk[mature]),
                "source_outcomes": observed,
                "source_opportunities": observed_opportunities,
                "channel_outcomes": channel_values,
                "channel_opportunities": channel_opportunities,
                "channel_evidence_scores": channel_scores,
                "observed_source_count": sum(value is not None for value in observed.values()),
                "observed_channel_count": sum(
                    value is not None for value in channel_values.values()
                ),
                "positive_source_count": sum(value is True for value in observed.values()),
                "explicit_negative_source_count": sum(
                    value is False for value in observed.values()
                ),
                "unknown_source_count": sum(value is None for value in observed.values()),
                "source_opportunity_count": sum(
                    value is True for value in observed_opportunities.values()
                ),
                "source_opportunity_not_observed_count": sum(
                    value is False for value in observed_opportunities.values()
                ),
                "source_opportunity_unknown_count": sum(
                    value is None for value in observed_opportunities.values()
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
    valid_channels = sorted(
        {
            channel_id
            for row in matrix_rows
            for channel_id, value in cast(dict[str, bool | None], row["channel_outcomes"]).items()
            if value is not None
        }
    )
    channel_count = len(valid_channels)
    capability: dict[str, Any] = {
        "status": pending_status if channel_count >= 2 else unavailable_status,
        "pilot_executed": False,
        "channel_count": channel_count,
        "reason": None if channel_count >= 2 else insufficient_channels_reason,
        "gate1_candidate": "L3_fixed_pi" if channel_count >= 2 else None,
        "hierarchical_pi_role": "sensitivity_only",
        "configured_channels": expected_channels,
        "valid_channels": valid_channels,
    }
    variables: list[dict[str, object]] = [
        {
            SOURCE_ID: source_id,
            CHANNEL_ID: expected_sources[source_id],
            TEMPORAL_ROLE: temporal_roles[source_id],
            "explicit_negative_allowed": negative_policy[source_id],
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
            "valid_channels": valid_channels,
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
            "configured_channel_count": len(expected_channels),
            "valid_channels": valid_channels,
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


def _aggregate_opportunity(values: list[bool | None]) -> bool | None:
    """Summarize source opportunity without turning absence into observed coverage."""
    if any(value is True for value in values):
        return True
    if values and all(value is False for value in values):
        return False
    return None


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
    risk_sets: pd.DataFrame,
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
    mature = columns[MATURE]
    eligible = columns[ELIGIBLE]
    required_risk = {year, mature, eligible}
    if not required_risk.issubset(risk_sets.columns):
        raise ValueError("P05 fold eligibility requires mature risk-set rows")
    counts = (
        sealed_outcomes.groupby(year, sort=True)[outcome].agg(["count", "sum"])
        if not sealed_outcomes.empty
        else pd.DataFrame()
    )
    mature_counts = (
        risk_sets.loc[risk_sets[mature].astype(bool) & risk_sets[eligible].astype(bool)]
        .groupby(year, sort=True)
        .size()
    )
    eligible_counts = (
        risk_sets.loc[risk_sets[eligible].astype(bool)].groupby(year, sort=True).size()
    )

    def row_for(fold_year: int, configured_role: str) -> dict[str, object]:
        labeled_count = int(str(counts.loc[fold_year, "count"])) if fold_year in counts.index else 0
        positive_count = int(str(counts.loc[fold_year, "sum"])) if fold_year in counts.index else 0
        mature_count = int(mature_counts.get(fold_year, 0))
        eligible_count = int(eligible_counts.get(fold_year, 0))
        explicit_negative_count = labeled_count - positive_count
        unknown_count = mature_count - labeled_count
        if unknown_count < 0:
            raise ValueError(f"fold={fold_year}: labeled outcomes exceed mature risk-set rows")
        observed_class_count = int(positive_count > 0) + int(explicit_negative_count > 0)
        if observed_class_count == 2:
            binary_reason = None
        elif positive_count == 0 and explicit_negative_count == 0:
            binary_reason = "NO_OBSERVED_BINARY_CLASSES"
        elif positive_count == 0:
            binary_reason = "POSITIVE_CLASS_MISSING"
        else:
            binary_reason = "EXPLICIT_NEGATIVE_CLASS_MISSING"
        if configured_role in {"initial", "prospective"}:
            role = f"{configured_role}_separate"
            reason = binary_reason or "NONCONFIRMATORY_FOLD_ROLE"
        elif observed_class_count < 2:
            role = "prospective_or_descriptive"
            reason = binary_reason
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
            "eligible_row_count": eligible_count,
            "mature_row_count": mature_count,
            "labeled_row_count": labeled_count,
            "observed_label_row_count": labeled_count,
            "positive_count": positive_count,
            "explicit_negative_count": explicit_negative_count,
            "unknown_count": unknown_count,
            "observed_binary_class_count": observed_class_count,
            "both_binary_classes_observed": positive_count > 0 and explicit_negative_count > 0,
            "binary_class_reason_code": binary_reason,
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


def fold_local_l3_target_frame(
    *,
    channel_selection: dict[str, Any],
    outer_year: int,
    columns: dict[str, str],
) -> pd.DataFrame:
    """Materialize only the development-only L3 posterior selected by P10."""
    if channel_selection.get("l3_fit_scope") != "development_history_only":
        raise ValueError("L3 Track B target requires a development-history-only P10 refit")
    if channel_selection.get("outer_outcomes_accessed") is not False:
        raise ValueError("L3 Track B target refuses selection with outer outcome access")
    if channel_selection.get("l3_selected_fixed_pi") is None:
        raise ValueError("L3 Track B target requires a selected fixed-pi scenario")
    raw_rows = channel_selection.get("l3_target_rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("L3 Track B target requires nonempty fold-local posterior rows")
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    value = columns[TARGET_VALUE]
    output: list[dict[str, object]] = []
    for raw in cast(list[object], raw_rows):
        if not isinstance(raw, dict):
            raise ValueError("L3 target row must be an object")
        row = cast(dict[str, Any], raw)
        row_year = int(row[FISCAL_YEAR])
        target = float(row[TARGET_VALUE])
        if row_year >= outer_year:
            raise ValueError("L3 Track B target contains an outer-or-future row")
        if not 0 <= target <= 1:
            raise ValueError("L3 Track B posterior target must be in [0, 1]")
        output.append({firm: str(row[FIRM_ID]), year: row_year, value: target})
    frame = pd.DataFrame(output, columns=[firm, year, value])
    if frame.duplicated([firm, year]).any():
        raise ValueError("L3 Track B target contains duplicate firm-year rows")
    frame[firm] = frame[firm].astype("string")
    frame[year] = frame[year].astype("int16")
    frame[value] = frame[value].astype("float64")
    return frame
