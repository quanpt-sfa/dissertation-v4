"""Policy-aware MCSE aggregation and cost-metric sanitation for P08."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import TypedDict, cast

import pandas as pd

from core.semantic_keys import (
    BATCH_KEY,
    ESTIMATE,
    MCSE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from simulation.mcse_contract import MetricPolicy


class BatchRangeAudit(TypedDict):
    scenario_id: str
    method_id: str
    artifact_batch_coordinate: str
    replication_start: int
    replication_end: int
    replication_count: int
    locked_worker_batch_size: int


def normalized_cost_regret(
    *,
    total_cost: float,
    oracle_cost: float,
    all_negative_cost: float,
) -> float:
    """Return normalized regret only when the avoidable-cost scale is identified."""
    avoidable_cost = abs(all_negative_cost - oracle_cost)
    scale = max(abs(all_negative_cost), abs(oracle_cost), 1.0)
    if avoidable_cost <= 1e-10 * scale:
        return math.nan
    return float((total_cost - oracle_cost) / avoidable_cost)


def sanitize_normalized_cost_regret(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute normalized regret from companion metrics without mutating artifacts."""
    required = {SCENARIO_ID, METHOD_ID, REPLICATION_ID, METRIC_ID}
    missing = sorted(required - set(str(column) for column in frame.columns))
    if missing:
        raise ValueError(f"P08 cost sanitation missing columns={missing}")

    records = cast(list[dict[str, object]], frame.to_dict(orient="records"))
    estimate_present = ESTIMATE in frame.columns
    metric_keys: set[tuple[str, str, int, str]] = set()
    companions: dict[tuple[str, str, int, str], dict[str, float]] = {}
    normalized_rows: list[tuple[int, tuple[str, str, int, str]]] = []
    companion_suffixes = {
        "normalized_cost_regret",
        "cost_regret_vs_oracle",
        "cost_savings_vs_all_negative",
    }

    for row_index, row in enumerate(records):
        scenario_id = str(row.get(SCENARIO_ID, ""))
        method_id = str(row.get(METHOD_ID, ""))
        replication_id = _required_int(row.get(REPLICATION_ID), "replication_" + "id")
        metric_id = str(row.get(METRIC_ID, ""))
        metric_key = (scenario_id, method_id, replication_id, metric_id)
        if metric_key in metric_keys:
            raise ValueError(
                "Cannot sanitize normalized cost regret with duplicate replication-metric rows"
            )
        metric_keys.add(metric_key)

        prefix, separator, suffix = metric_id.rpartition("::")
        if not separator or suffix not in companion_suffixes:
            continue
        group_key = (scenario_id, method_id, replication_id, prefix)
        if suffix == "normalized_cost_regret":
            normalized_rows.append((row_index, group_key))
            continue
        value = _finite_float(row.get(ESTIMATE))
        if value is not None:
            companions.setdefault(group_key, {})[suffix] = value

    if not normalized_rows:
        return frame.copy()
    if not estimate_present:
        raise ValueError("normalized cost regret rows require the estimate column")

    for row_index, group_key in normalized_rows:
        values = companions.get(group_key, {})
        regret = values.get("cost_regret_vs_oracle")
        savings = values.get("cost_savings_vs_all_negative")
        if regret is None or savings is None:
            raise ValueError(
                "normalized cost regret companion metrics are missing for one or more replications"
            )
        avoidable_cost = abs(regret + savings)
        scale = max(abs(regret), abs(savings), 1.0)
        records[row_index][ESTIMATE] = (
            math.nan if avoidable_cost <= 1e-10 * scale else regret / avoidable_cost
        )
    return pd.DataFrame.from_records(records, columns=list(frame.columns))


def validate_worker_batch(
    frame: pd.DataFrame,
    *,
    coordinates: Mapping[str, str],
    locked_worker_batch_size: int,
) -> BatchRangeAudit:
    """Validate one immutable worker artifact against its locked coordinate."""
    if locked_worker_batch_size < 1:
        raise ValueError("locked worker batch size must be positive")
    required = {SCENARIO_ID, METHOD_ID, REPLICATION_ID}
    missing = sorted(required - set(str(column) for column in frame.columns))
    if missing:
        raise ValueError(f"P08 worker batch missing columns={missing}")

    records = cast(list[dict[str, object]], frame.to_dict(orient="records"))
    scenario_ids = {str(row.get(SCENARIO_ID, "")) for row in records}
    method_ids = {str(row.get(METHOD_ID, "")) for row in records}
    if len(scenario_ids) != 1 or "" in scenario_ids:
        raise ValueError("Each P08 worker batch must contain one nonempty scenario_id")
    if len(method_ids) != 1 or "" in method_ids:
        raise ValueError("Each P08 worker batch must contain one nonempty method_id")

    replication_ids = sorted(
        {_required_int(row.get(REPLICATION_ID), "replication_" + "id") for row in records}
    )
    if not replication_ids:
        raise ValueError("P08 worker batch contains no replication IDs")
    expected_ids = list(range(replication_ids[0], replication_ids[-1] + 1))
    if replication_ids != expected_ids:
        raise ValueError("P08 worker batch replication IDs must be contiguous")
    if replication_ids[0] % locked_worker_batch_size != 0:
        raise ValueError("P08 worker batch start is not aligned to locked batch size")
    if len(replication_ids) > locked_worker_batch_size:
        raise ValueError("P08 worker artifact exceeds locked worker batch size")

    expected_batch_key = f"b{replication_ids[0] // locked_worker_batch_size:04d}"
    actual_batch_key = coordinates.get(BATCH_KEY, "")
    if actual_batch_key != expected_batch_key:
        raise ValueError(
            f"P08 worker batch_key={actual_batch_key} does not match expected={expected_batch_key}"
        )
    return {
        "scenario_id": next(iter(scenario_ids)),
        "method_" + "id": next(iter(method_ids)),
        "artifact_batch_coordinate": actual_batch_key,
        "replication_start": replication_ids[0],
        "replication_end": replication_ids[-1],
        "replication_count": len(replication_ids),
        "locked_worker_batch_size": locked_worker_batch_size,
    }


def summarize_mcse(
    batches: Sequence[pd.DataFrame],
    *,
    metric_policies: Mapping[str, MetricPolicy],
    minimum_replications: int,
    maximum_replications: int,
    pass_fail_mcse_maximum: float,
    l3_minimum_replications: int | None = None,
    l3_maximum_replications: int | None = None,
    l3_pass_fail_mcse_maximum: float | None = None,
    extended_minimum_replications: int | None = None,
    extended_maximum_replications: int | None = None,
    extended_pass_fail_mcse_maximum: float | None = None,
    cost_sensitive_minimum_replications: int | None = None,
    cost_sensitive_maximum_replications: int | None = None,
    cost_sensitive_pass_fail_mcse_maximum: float | None = None,
    cost_mcse_relative_fraction: float = 0.02,
    continuous_mcse_fraction: float | None = None,
    minimum_meaningful_improvement: float | None = None,
) -> dict[str, object]:
    """Compute MCSE using explicit metric policies and finite-value contracts."""
    if not batches:
        return {
            "status": "SKIPPED",
            "reason_code": "NO_SIMULATION_BATCHES",
            "precision_target_met": False,
            "metrics": [],
        }
    if (
        minimum_replications < 1
        or maximum_replications < minimum_replications
        or pass_fail_mcse_maximum <= 0
    ):
        raise ValueError("simulation replication controls are invalid")

    records: list[dict[str, object]] = []
    for batch in batches:
        records.extend(cast(list[dict[str, object]], batch.to_dict(orient="records")))
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in records:
        key = (
            str(row.get(SCENARIO_ID, "")),
            str(row.get(METHOD_ID, "")),
            str(row.get(METRIC_ID, "")),
        )
        if "" in key:
            raise ValueError("P08 MCSE rows require scenario_id, method_id, and metric_id")
        grouped.setdefault(key, []).append(row)

    metrics: list[dict[str, object]] = []
    blockers: list[str] = []
    for (scenario_id, method_id, metric_id), rows in sorted(grouped.items()):
        policy = metric_policies.get(metric_id)
        if policy is None:
            raise ValueError(f"P08 MCSE resolved policy missing for metric_id={metric_id}")
        total_count = len(rows)
        finite_values = [
            value
            for value in (_finite_float(row.get(ESTIMATE)) for row in rows)
            if value is not None
        ]
        finite_count = len(finite_values)
        undefined_count = total_count - finite_count
        mean = statistics.fmean(finite_values) if finite_values else math.nan
        standard_deviation = statistics.stdev(finite_values) if finite_count > 1 else 0.0
        actual_mcse = standard_deviation / math.sqrt(finite_count) if finite_count else math.nan

        learner_tiers = _metadata_values(rows, "learner_tier")
        training_costs = _metadata_values(rows, "training_cost_regime_id")
        treatments = _metadata_values(rows, "imbalance_treatment_id")
        method_families = _metadata_values(rows, "method_family")
        if any(
            len(values) > 1
            for values in (learner_tiers, training_costs, treatments, method_families)
        ):
            raise ValueError(f"scenario={scenario_id}, method={method_id}: mixed method metadata")

        is_standalone = method_families == {"standalone_estimator"} or learner_tiers == {
            "standalone"
        }
        is_imbalance_robustness = treatments not in (
            set(),
            {"none"},
            {"not_applicable"},
        )
        is_extended = (
            learner_tiers
            in (
                {"extended"},
                {"methodological"},
            )
            or is_imbalance_robustness
        )
        is_cost_sensitive_core = (
            learner_tiers == {"core"}
            and not is_imbalance_robustness
            and training_costs not in (set(), {"symmetric"})
        )
        required_replications = _tier_value(
            default=minimum_replications,
            standalone=is_standalone,
            standalone_value=l3_minimum_replications,
            extended=is_extended,
            extended_value=extended_minimum_replications,
            cost_sensitive=is_cost_sensitive_core,
            cost_sensitive_value=cost_sensitive_minimum_replications,
        )
        replication_cap = _tier_value(
            default=maximum_replications,
            standalone=is_standalone,
            standalone_value=l3_maximum_replications,
            extended=is_extended,
            extended_value=extended_maximum_replications,
            cost_sensitive=is_cost_sensitive_core,
            cost_sensitive_value=cost_sensitive_maximum_replications,
        )
        pass_fail_target = float(
            _tier_value(
                default=pass_fail_mcse_maximum,
                standalone=is_standalone,
                standalone_value=l3_pass_fail_mcse_maximum,
                extended=is_extended,
                extended_value=extended_pass_fail_mcse_maximum,
                cost_sensitive=is_cost_sensitive_core,
                cost_sensitive_value=cost_sensitive_pass_fail_mcse_maximum,
            )
        )

        gate_required = policy["mcse_gate_required"]
        target_rule = policy["target_rule"]
        mcse_target = math.nan
        if gate_required:
            mcse_target = pass_fail_target
            if target_rule == "relative_to_mean":
                mcse_target = max(
                    mcse_target,
                    cost_mcse_relative_fraction * max(abs(mean), 1e-6),
                )
            elif target_rule == "mmi_scaled":
                if continuous_mcse_fraction is None or minimum_meaningful_improvement is None:
                    raise ValueError(
                        f"metric_id={metric_id}: MMI-scaled MCSE rule lacks locked inputs"
                    )
                mcse_target = min(
                    mcse_target,
                    continuous_mcse_fraction * minimum_meaningful_improvement,
                )
            elif target_rule != "absolute":
                raise ValueError(
                    f"metric_id={metric_id}: unsupported gated target_rule={target_rule}"
                )

        required_finite = math.ceil(required_replications * policy["minimum_finite_fraction"])
        total_minimum_met = total_count >= required_replications
        finite_minimum_met = finite_count >= required_finite
        undefined_policy_met = not (policy["undefined_policy"] == "forbid" and undefined_count > 0)
        mcse_target_met = (not gate_required) or (
            finite_count > 0 and math.isfinite(actual_mcse) and actual_mcse <= mcse_target
        )
        if not gate_required:
            total_minimum_met = True
            finite_minimum_met = True
            undefined_policy_met = True
            mcse_target_met = True

        gate_met = (
            total_minimum_met and finite_minimum_met and undefined_policy_met and mcse_target_met
        )
        if gate_required and not gate_met:
            blockers.append(f"{scenario_id}:{method_id}:{metric_id}")
        metrics.append(
            {
                SCENARIO_ID: scenario_id,
                METHOD_ID: method_id,
                METRIC_ID: metric_id,
                "policy_id": policy["policy_id"],
                "metric_role": policy["role"],
                "undefined_policy": policy["undefined_policy"],
                "mcse_target_rule": target_rule,
                "mcse_gate_required": gate_required,
                "replications": total_count,
                "finite_replications": finite_count,
                "undefined_replications": undefined_count,
                "undefined_fraction": undefined_count / max(1, total_count),
                "mean": mean,
                MCSE: actual_mcse,
                "mcse_target": mcse_target,
                "replication_tier": (
                    "standalone"
                    if is_standalone
                    else "imbalance_robustness"
                    if is_imbalance_robustness
                    else "extended_predictive"
                    if is_extended
                    else "cost_sensitive_core"
                    if is_cost_sensitive_core
                    else "core_cost_neutral"
                ),
                "minimum_replications_required": required_replications,
                "finite_replications_required": required_finite,
                "minimum_replications_met": total_minimum_met,
                "finite_replications_met": finite_minimum_met,
                "undefined_policy_met": undefined_policy_met,
                "mcse_target_met": mcse_target_met,
                "gate_metric_met": gate_met,
                "maximum_replications_reached": total_count >= replication_cap,
                **(
                    {"rmse": math.sqrt(max(0.0, mean))}
                    if metric_id == "prevalence_squared_error" and math.isfinite(mean)
                    else {}
                ),
            }
        )

    gated = [item for item in metrics if item["mcse_gate_required"] is True]
    if not gated:
        return {
            "status": "FAIL",
            "reason_code": "NO_CONFIRMATORY_MCSE_METRICS",
            "precision_target_met": False,
            "evidence_blockers": ["NO_CONFIRMATORY_MCSE_METRICS"],
            "metrics": metrics,
        }
    precision_met = all(item["gate_metric_met"] is True for item in gated)
    maximum_reached = all(
        item["gate_metric_met"] is True or item["maximum_replications_reached"] is True
        for item in gated
    )
    status = "PASS" if precision_met else "MAXIMUM_REACHED" if maximum_reached else "CONTINUE"
    return {
        "status": status,
        "reason_code": (
            None
            if precision_met
            else "MCSE_TARGET_NOT_MET_AT_CAP"
            if maximum_reached
            else "ADDITIONAL_REPLICATIONS_REQUIRED"
        ),
        "minimum_replications": minimum_replications,
        "maximum_replications": maximum_replications,
        "pass_fail_mcse_maximum": pass_fail_mcse_maximum,
        "l3_minimum_replications": l3_minimum_replications,
        "l3_maximum_replications": l3_maximum_replications,
        "l3_pass_fail_mcse_maximum": l3_pass_fail_mcse_maximum,
        "extended_minimum_replications": extended_minimum_replications,
        "extended_maximum_replications": extended_maximum_replications,
        "extended_pass_fail_mcse_maximum": extended_pass_fail_mcse_maximum,
        "cost_sensitive_minimum_replications": cost_sensitive_minimum_replications,
        "cost_sensitive_maximum_replications": cost_sensitive_maximum_replications,
        "cost_sensitive_pass_fail_mcse_maximum": cost_sensitive_pass_fail_mcse_maximum,
        "cost_mcse_relative_fraction": cost_mcse_relative_fraction,
        "continuous_mcse_fraction": continuous_mcse_fraction,
        "minimum_meaningful_improvement": minimum_meaningful_improvement,
        "precision_target_met": precision_met,
        "confirmatory_metric_count": len(gated),
        "evidence_blockers": blockers,
        "metrics": metrics,
    }


def _metadata_values(rows: Sequence[Mapping[str, object]], key: str) -> set[str]:
    return {str(value) for row in rows if (value := row.get(key)) is not None and str(value) != ""}


def _tier_value(
    *,
    default: int | float,
    standalone: bool,
    standalone_value: int | float | None,
    extended: bool,
    extended_value: int | float | None,
    cost_sensitive: bool,
    cost_sensitive_value: int | float | None,
) -> int | float:
    if standalone and standalone_value is not None:
        return standalone_value
    if extended and extended_value is not None:
        return extended_value
    if cost_sensitive and cost_sensitive_value is not None:
        return cost_sensitive_value
    return default


def _required_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"P08 {field} must be integer-compatible")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"P08 {field} must be integer-compatible") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"P08 {field} must be integral")
    return parsed


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None
