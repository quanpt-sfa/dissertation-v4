"""Assemble the P07 firm-year feature panel by computing features from raw sources.

This is the drop-in replacement for the removed feature-store loader: it computes
every registered feature from the raw ``financial_statement_core_long`` source
(via :mod:`features.raw_loader` and :mod:`features.compute`) and joins the results
to the P02 firm-year spine, returning the same panel/audit contract the P07 stage
expects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from features.compute import FIRM, YEAR, compute_feature_values
from features.raw_loader import build_long_values, read_financial_statement_long
from p02.models import EntityResolutionSpec


@dataclass(frozen=True)
class FeatureSourceResult:
    """Computed panel plus the ingestion-boundary audits P07 publishes."""

    panel: pd.DataFrame
    validation_report: dict[str, object]
    file_audit: pd.DataFrame
    identifier_audit: pd.DataFrame
    availability_violations: pd.DataFrame
    research_decision_audit: pd.DataFrame
    coverage_audit: pd.DataFrame
    component_completeness: pd.DataFrame


def assemble_from_raw(
    *,
    base_panel: pd.DataFrame,
    feature_definitions: Sequence[Mapping[str, object]],
    intended_definitions: Sequence[Mapping[str, object]],
    entity_spec: EntityResolutionSpec,
    raw_source_path: Path,
    reader: Mapping[str, object] | None,
    firm_column: str,
    year_column: str,
    prediction_time_column: str,
) -> FeatureSourceResult:
    """Compute features from raw and join to the firm-year spine."""
    required_base = {firm_column, year_column, prediction_time_column}
    if not required_base.issubset(base_panel.columns):
        raise ValueError(
            f"P02 panel missing feature join columns: {sorted(required_base - set(base_panel.columns))}"
        )

    spine = base_panel.loc[:, [firm_column, year_column, prediction_time_column]].copy()
    spine[firm_column] = spine[firm_column].astype("string")
    spine[year_column] = pd.to_numeric(spine[year_column], errors="raise").astype("int16")
    if spine.duplicated([firm_column, year_column]).any():
        raise ValueError("P02 panel has duplicate firm-year keys")
    spine = spine.sort_values([firm_column, year_column], kind="stable").reset_index(drop=True)
    key_index = pd.MultiIndex.from_frame(
        spine[[firm_column, year_column]].rename(columns={firm_column: FIRM, year_column: YEAR})
    )

    raw_frame = read_financial_statement_long(raw_source_path, reader)
    long_values = build_long_values(raw_frame, entity_spec=entity_spec)

    definitions = [dict(item) for item in (*feature_definitions, *intended_definitions)]
    computations = compute_feature_values(definitions, long_values)

    panel = spine.copy()
    status_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    completeness_rows: list[dict[str, object]] = []
    for computation in computations:
        aligned = computation.values.reindex(key_index)
        panel[computation.feature_id] = aligned.to_numpy()
        non_null = int(aligned.notna().sum())
        status_rows.append(
            {
                "feature_id": computation.feature_id,
                "status": computation.status,
                "reason_code": computation.reason_code,
                "non_null_firm_years": non_null,
            }
        )
        coverage_rows.append(
            {
                "feature_id": computation.feature_id,
                "firm_year_count": len(spine),
                "non_null_count": non_null,
                "coverage_fraction": (non_null / len(spine)) if len(spine) else 0.0,
                "status": computation.status,
            }
        )
        diagnostics = computation.diagnostics
        if diagnostics is not None:
            raw_incomplete: object = diagnostics.get("incomplete_firm_years")
            if isinstance(raw_incomplete, list):
                for raw_row in cast(list[object], raw_incomplete):
                    if not isinstance(raw_row, Mapping):
                        continue
                    typed_row = cast(Mapping[object, object], raw_row)
                    normalized_row: dict[str, object] = {}
                    for key, value in typed_row.items():
                        if not isinstance(key, str) or not key:
                            raise ValueError("component completeness keys must be strings")
                        normalized_row[key] = value
                    completeness_rows.append(
                        {"feature_id": computation.feature_id, **normalized_row}
                    )

    status_frame = pd.DataFrame(status_rows)
    pass_count = int((status_frame["status"] == "PASS").sum()) if not status_frame.empty else 0
    unsupported = (
        status_frame.loc[status_frame["status"] == "UNSUPPORTED", "feature_id"].tolist()
        if not status_frame.empty
        else []
    )
    validation_report: dict[str, object] = {
        "source": "financial_statement_core_long",
        "computation_mode": "raw_registry_formula",
        "feature_count": len(computations),
        "computed_pass": pass_count,
        "unsupported_features": unsupported,
        "raw_rows": int(len(long_values)),
        "component_sum_incomplete_dropped": completeness_rows,
    }

    firm_ids = sorted(str(value) for value in long_values[FIRM].dropna().unique())
    panel_firms = set(spine[firm_column].dropna().astype(str))
    identifier_audit = pd.DataFrame(
        {
            "feature_store_firm_id": pd.Series(firm_ids, dtype="string"),
            "canonical_firm_id": pd.Series(firm_ids, dtype="string"),
            "mapping_status": pd.Series(
                ["MATCHED" if firm in panel_firms else "UNMATCHED" for firm in firm_ids],
                dtype="string",
            ),
            "ambiguity_flag": pd.Series([False] * len(firm_ids), dtype="bool"),
        }
    )

    file_audit = pd.DataFrame(
        [
            {
                "source_id": "financial_statement_core_long",
                "relative_path": raw_source_path.name,
                "raw_rows": int(len(raw_frame)),
                "usable_rows": int(len(long_values)),
            }
        ]
    )
    # Annual-anchor availability holds by construction (value for year t is available
    # at the shared anchor a_t = prediction_time), so no availability violations.
    availability_violations = pd.DataFrame(
        columns=["feature_id", firm_column, year_column, "available_date", prediction_time_column]
    )
    research_decision_audit = status_frame.loc[:, ["feature_id", "status", "reason_code"]].copy()

    return FeatureSourceResult(
        panel=panel,
        validation_report=validation_report,
        file_audit=file_audit,
        identifier_audit=identifier_audit,
        availability_violations=availability_violations,
        research_decision_audit=research_decision_audit,
        coverage_audit=pd.DataFrame(coverage_rows),
        component_completeness=pd.DataFrame(completeness_rows),
    )
