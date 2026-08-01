"""Reporting helpers for the P07 raw feature-computation contract."""

from __future__ import annotations

from typing import cast


def build_raw_computation_decision_report(base: str, summary: dict[str, object]) -> str:
    """Append raw-computation validation details to the P07 governance report."""

    store = cast(dict[str, object], summary["feature_store"])
    identity = cast(dict[str, object], summary["identifier_mapping"])
    unsupported_raw = store.get("unsupported_features")
    if not isinstance(unsupported_raw, list):
        raise ValueError("P07 feature_store.unsupported_features must be a list")
    unsupported = [str(value) for value in unsupported_raw]
    stage_status = _required_text(store, "stage_status")
    source = _required_text(store, "source")
    computation_mode = _required_text(store, "computation_mode")
    feature_count = _required_nonnegative_int(store, "feature_count")
    computed_pass = _required_nonnegative_int(store, "computed_pass")
    raw_rows = _required_nonnegative_int(store, "raw_rows")
    panel_features = _required_nonnegative_int(summary, "panel_features")
    unresolved_features = _required_nonnegative_int(summary, "unresolved_features")
    matched_identifiers = _required_nonnegative_int(identity, "matched_identifiers")
    unmatched_identifiers = _required_nonnegative_int(identity, "unmatched_identifiers")
    unsupported_text = ", ".join(unsupported) if unsupported else "none"
    return (
        base
        + "\n## Raw feature-computation validation\n\n"
        + f"- P07 stage status: {stage_status}\n"
        + f"- Registered source: {source}\n"
        + f"- Computation mode: {computation_mode}\n"
        + f"- Registered feature computations: {feature_count}\n"
        + f"- Computed PASS: {computed_pass}\n"
        + f"- Unsupported feature computations: {len(unsupported)}\n"
        + f"- Raw source rows: {raw_rows}\n"
        + f"- Operational panel features: {panel_features}\n"
        + f"- Registry unresolved features: {unresolved_features}\n"
        + f"- Matched identifiers: {matched_identifiers}\n"
        + f"- Unmatched identifiers: {unmatched_identifiers}\n"
        + "- Availability basis is a synthetic annual anchor, not an observed publication date.\n"
        + "\n## Registered unavailable or unresolved feature computations\n\n"
        + unsupported_text
        + "\n"
    )


def _required_text(values: dict[str, object], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"P07 report contract requires non-empty text field {key}")
    return value


def _required_nonnegative_int(values: dict[str, object], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"P07 report contract requires nonnegative integer field {key}")
    return value
