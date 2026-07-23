"""Typed methodology and active-profile validation for P08 aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pandas as pd

from core.semantic_keys import (
    LEARNER_ID,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from simulation.method_contract import (
    ANALYSIS_ROLE,
    EXECUTION_PROFILE,
    IMBALANCE_TREATMENT_ID,
    PROTOCOL_STATUS,
    active_method_ids,
    method_by_id,
    profile_expected_counts,
    required_metric_ids,
)

_REQUIRED_METADATA = (
    "label_strategy_id",
    LEARNER_ID,
    "learner_tier",
    "training_cost_regime_id",
    IMBALANCE_TREATMENT_ID,
    "method_family",
    ANALYSIS_ROLE,
    EXECUTION_PROFILE,
    PROTOCOL_STATUS,
)


def filter_active_batches(
    batches: Sequence[pd.DataFrame],
    scenario_by_id: Mapping[str, Mapping[str, object]],
) -> list[pd.DataFrame]:
    """Keep only methods active under the locked execution profile."""
    output: list[pd.DataFrame] = []
    for batch in batches:
        records = _records(batch)
        if not records:
            continue
        scenario_ids = {str(row.get(SCENARIO_ID, "")) for row in records}
        method_ids = {str(row.get(METHOD_ID, "")) for row in records}
        if len(scenario_ids) != 1 or len(method_ids) != 1:
            raise ValueError("Each simulation batch must contain exactly one scenario and method")
        scenario_id = next(iter(scenario_ids))
        method_id = next(iter(method_ids))
        scenario = scenario_by_id.get(scenario_id)
        if scenario is not None and method_id in active_method_ids(scenario):
            output.append(batch)
    return output


def required_active_metric_ids(
    scenario_by_id: Mapping[str, Mapping[str, object]],
) -> set[str]:
    """Return all metric IDs required by active methods in registered scenarios."""
    metric_ids: set[str] = set()
    for scenario in scenario_by_id.values():
        for method_id in active_method_ids(scenario):
            metric_ids.update(required_metric_ids(method_by_id(scenario, method_id)))
    return metric_ids


def validate_methodology(
    *,
    batches: Sequence[pd.DataFrame],
    scenario_by_id: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Validate active method coverage, metric presence, metadata, and uniqueness."""
    if not scenario_by_id:
        return {"status": "FAIL", "reason_codes": ["NO_REGISTERED_SCENARIOS"]}

    first_scenario = next(iter(scenario_by_id.values()))
    first_registry_raw = first_scenario.get("method_registry")
    if not isinstance(first_registry_raw, list):
        raise ValueError("P08 scenario method_registry required")
    first_registry = [
        cast(dict[str, object], item)
        for item in cast(list[object], first_registry_raw)
        if isinstance(item, dict)
    ]
    counts = profile_expected_counts(first_registry, active_only=True)
    active_profile = str(first_scenario.get("execution_profile", ""))

    records: list[dict[str, object]] = []
    columns: set[str] = set()
    for batch in batches:
        columns.update(str(column) for column in batch.columns)
        records.extend(_records(batch))
    if not records:
        return {
            "status": "FAIL",
            "reason_codes": ["NO_SIMULATION_BATCHES"],
            "execution_profile": active_profile,
            "expected_counts": counts,
            "missing_methods": [],
            "unexpected_methods": [],
            "missing_metrics": [],
            "duplicate_rows": 0,
        }

    required_columns = {
        SCENARIO_ID,
        METHOD_ID,
        REPLICATION_ID,
        METRIC_ID,
        *_REQUIRED_METADATA,
    }
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        return {
            "status": "FAIL",
            "reason_codes": ["BATCH_METADATA_COLUMNS_MISSING"],
            "missing_columns": missing_columns,
            "execution_profile": active_profile,
            "expected_counts": counts,
        }

    row_keys: set[tuple[str, str, int, str]] = set()
    duplicate_rows = 0
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in records:
        scenario_id = str(row.get(SCENARIO_ID, ""))
        method_id = str(row.get(METHOD_ID, ""))
        replication_id = _required_int(row.get(REPLICATION_ID))
        metric_id = str(row.get(METRIC_ID, ""))
        key = (scenario_id, method_id, replication_id, metric_id)
        if key in row_keys:
            duplicate_rows += 1
        row_keys.add(key)
        grouped.setdefault((scenario_id, method_id), []).append(row)

    missing_methods: list[dict[str, str]] = []
    unexpected_methods: list[dict[str, str]] = []
    missing_metrics: list[dict[str, object]] = []
    metadata_conflicts: list[dict[str, str]] = []
    profile_conflicts: list[dict[str, str]] = []

    for scenario_id, scenario in sorted(scenario_by_id.items()):
        registry_raw = scenario.get("method_registry")
        if not isinstance(registry_raw, list):
            raise ValueError(f"scenario={scenario_id}: method_registry required")
        registry = [
            cast(dict[str, object], item)
            for item in cast(list[object], registry_raw)
            if isinstance(item, dict)
        ]
        expected_ids = active_method_ids(scenario)
        scenario_counts = profile_expected_counts(registry, active_only=True)
        if len(expected_ids) != int(cast(int, scenario_counts["method_total"])):
            raise ValueError(f"scenario={scenario_id}: active method count mismatch")
        actual_ids = {
            method_id
            for (record_scenario, method_id) in grouped
            if record_scenario == scenario_id
        }
        for method_id in sorted(expected_ids - actual_ids):
            missing_methods.append({SCENARIO_ID: scenario_id, METHOD_ID: method_id})
        for method_id in sorted(actual_ids - expected_ids):
            unexpected_methods.append({SCENARIO_ID: scenario_id, METHOD_ID: method_id})
        for method_id in sorted(expected_ids & actual_ids):
            spec = method_by_id(scenario, method_id, require_active=True)
            method_rows = grouped[(scenario_id, method_id)]
            present_metrics = {str(row.get(METRIC_ID, "")) for row in method_rows}
            absent = sorted(required_metric_ids(spec) - present_metrics)
            if absent:
                missing_metrics.append(
                    {SCENARIO_ID: scenario_id, METHOD_ID: method_id, "metric_ids": absent}
                )
            for field in _REQUIRED_METADATA:
                values = {str(row.get(field, "")) for row in method_rows}
                if values != {str(spec[field])}:
                    metadata_conflicts.append(
                        {SCENARIO_ID: scenario_id, METHOD_ID: method_id, "field": field}
                    )
            expected_profile = str(scenario.get("execution_profile", ""))
            profile_values = {str(row.get(EXECUTION_PROFILE, "")) for row in method_rows}
            if profile_values != {expected_profile}:
                profile_conflicts.append(
                    {SCENARIO_ID: scenario_id, METHOD_ID: method_id}
                )

    reason_codes: list[str] = []
    if duplicate_rows:
        reason_codes.append("DUPLICATE_REPLICATION_METRIC_ROWS")
    if missing_methods:
        reason_codes.append("ACTIVE_PROFILE_METHODS_INCOMPLETE")
    if unexpected_methods:
        reason_codes.append("INACTIVE_OR_UNREGISTERED_METHOD_EXECUTED")
    if missing_metrics:
        reason_codes.append("REQUIRED_COST_METRICS_MISSING")
    if metadata_conflicts:
        reason_codes.append("METHOD_METADATA_CONFLICT")
    if profile_conflicts:
        reason_codes.append("EXECUTION_PROFILE_CONFLICT")
    return {
        "status": "PASS" if not reason_codes else "FAIL",
        "reason_codes": reason_codes,
        "execution_profile": active_profile,
        "expected_counts": counts,
        "missing_methods": missing_methods,
        "unexpected_methods": unexpected_methods,
        "missing_metrics": missing_metrics,
        "metadata_conflicts": metadata_conflicts,
        "profile_conflicts": profile_conflicts,
        "duplicate_rows": duplicate_rows,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], frame.to_dict(orient="records"))


def _required_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("P08 replication_id must be integer-compatible")
    parsed = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("P08 replication_id must be integral")
    return parsed
