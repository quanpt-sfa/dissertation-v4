"""Assemble the P07 firm-year feature panel from a registered physical source.

Production uses one final firm-year Parquet containing the approved feature
columns. The historical long-format computation path remains available only as
a compatibility path for unit tests and archived protocols.
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
    """Feature panel plus the ingestion-boundary audits P07 publishes."""

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
    """Dispatch to the final wide input or the archived long-format path."""

    if raw_source_path.suffix.casefold() == ".parquet":
        return _assemble_from_final(
            base_panel=base_panel,
            feature_definitions=feature_definitions,
            intended_definitions=intended_definitions,
            raw_source_path=raw_source_path,
            firm_column=firm_column,
            year_column=year_column,
            prediction_time_column=prediction_time_column,
        )
    return _assemble_from_legacy_long(
        base_panel=base_panel,
        feature_definitions=feature_definitions,
        intended_definitions=intended_definitions,
        entity_spec=entity_spec,
        raw_source_path=raw_source_path,
        reader=reader,
        firm_column=firm_column,
        year_column=year_column,
        prediction_time_column=prediction_time_column,
    )


def _spine(
    base_panel: pd.DataFrame,
    *,
    firm_column: str,
    year_column: str,
    prediction_time_column: str,
) -> pd.DataFrame:
    required = {firm_column, year_column, prediction_time_column}
    if not required.issubset(base_panel.columns):
        raise ValueError(
            f"P02 panel missing feature join columns: {sorted(required - set(base_panel.columns))}"
        )
    spine = base_panel.loc[:, [firm_column, year_column, prediction_time_column]].copy()
    spine[firm_column] = spine[firm_column].astype("string")
    spine[year_column] = pd.to_numeric(spine[year_column], errors="raise").astype("int16")
    spine[prediction_time_column] = pd.to_datetime(
        spine[prediction_time_column], errors="raise"
    ).astype("datetime64[ns]")
    if spine.duplicated([firm_column, year_column]).any():
        raise ValueError("P02 panel has duplicate firm-year keys")
    return spine.sort_values([firm_column, year_column], kind="stable").reset_index(drop=True)


def _assemble_from_final(
    *,
    base_panel: pd.DataFrame,
    feature_definitions: Sequence[Mapping[str, object]],
    intended_definitions: Sequence[Mapping[str, object]],
    raw_source_path: Path,
    firm_column: str,
    year_column: str,
    prediction_time_column: str,
) -> FeatureSourceResult:
    """Bind approved physical feature columns from the final firm-year Parquet."""

    spine = _spine(
        base_panel,
        firm_column=firm_column,
        year_column=year_column,
        prediction_time_column=prediction_time_column,
    )
    final_frame = pd.read_parquet(raw_source_path)
    required = {firm_column, year_column, prediction_time_column}
    if not required.issubset(final_frame.columns):
        raise ValueError(
            "final firm-year input is missing canonical panel columns: "
            f"{sorted(required - set(final_frame.columns))}"
        )
    final_frame = final_frame.copy()
    final_frame[firm_column] = final_frame[firm_column].astype("string")
    final_frame[year_column] = pd.to_numeric(
        final_frame[year_column], errors="raise"
    ).astype("int16")
    final_frame[prediction_time_column] = pd.to_datetime(
        final_frame[prediction_time_column], errors="raise"
    ).astype("datetime64[ns]")
    if final_frame.duplicated([firm_column, year_column]).any():
        raise ValueError("final firm-year input has duplicate firm-year keys")

    key_check = spine.loc[:, [firm_column, year_column]].merge(
        final_frame.loc[:, [firm_column, year_column]],
        on=[firm_column, year_column],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not key_check["_merge"].eq("both").all():
        counts = key_check["_merge"].value_counts().to_dict()
        raise ValueError(f"final input and P02 spine firm-year keys differ: {counts}")

    time_check = spine.merge(
        final_frame.loc[:, [firm_column, year_column, prediction_time_column]],
        on=[firm_column, year_column],
        how="left",
        validate="one_to_one",
        suffixes=("_panel", "_final"),
    )
    panel_time = f"{prediction_time_column}_panel"
    final_time = f"{prediction_time_column}_final"
    if not time_check[panel_time].equals(time_check[final_time]):
        raise ValueError("final input prediction_time differs from the P02 locked panel anchor")

    definitions = [dict(item) for item in (*feature_definitions, *intended_definitions)]
    status_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    selected_columns: list[str] = []
    for definition in definitions:
        feature_id = str(definition.get("feature_id", ""))
        physical = definition.get("physical_column")
        present = isinstance(physical, str) and bool(physical) and physical in final_frame.columns
        status = "PASS" if present else "UNSUPPORTED"
        reason = "FINAL_INPUT_COLUMN_PRESENT" if present else "FINAL_INPUT_COLUMN_MISSING"
        non_null = int(final_frame[cast(str, physical)].notna().sum()) if present else 0
        status_rows.append(
            {
                "feature_id": feature_id,
                "status": status,
                "reason_code": reason,
                "non_null_firm_years": non_null,
            }
        )
        coverage_rows.append(
            {
                "feature_id": feature_id,
                "firm_year_count": len(spine),
                "non_null_count": non_null,
                "coverage_fraction": (non_null / len(spine)) if len(spine) else 0.0,
                "status": status,
            }
        )
        if present and cast(str, physical) not in selected_columns:
            selected_columns.append(cast(str, physical))

    final_values = final_frame.loc[:, [firm_column, year_column, *selected_columns]].copy()
    panel = spine.merge(
        final_values,
        on=[firm_column, year_column],
        how="left",
        validate="one_to_one",
    )
    status_frame = pd.DataFrame(status_rows)
    pass_count = int((status_frame["status"] == "PASS").sum()) if not status_frame.empty else 0
    unsupported = (
        status_frame.loc[status_frame["status"] == "UNSUPPORTED", "feature_id"].tolist()
        if not status_frame.empty
        else []
    )
    validation_report: dict[str, object] = {
        "source": "final_firm_year_s1",
        "computation_mode": "final_firm_year_physical_columns",
        "feature_count": len(definitions),
        "computed_pass": pass_count,
        "unsupported_features": unsupported,
        "raw_rows": int(len(final_frame)),
        "component_sum_incomplete_dropped": [],
    }
    firm_ids = sorted(str(value) for value in final_frame[firm_column].dropna().unique())
    panel_firms = set(spine[firm_column].dropna().astype(str))
    status_audit = status_frame.loc[:, ["feature_id", "status", "reason_code"]].copy()
    return FeatureSourceResult(
        panel=panel,
        validation_report=validation_report,
        file_audit=build_file_audit(
            status_frame=status_frame,
            raw_source_path=raw_source_path,
            raw_rows=len(final_frame),
            usable_rows=len(panel),
            source_id="final_firm_year_s1",
        ),
        identifier_audit=build_identifier_audit(
            firm_ids=firm_ids,
            panel_firms=panel_firms,
            firm_column=firm_column,
        ),
        availability_violations=pd.DataFrame(
            columns=[
                "feature_id",
                firm_column,
                year_column,
                "available_date",
                prediction_time_column,
            ]
        ),
        research_decision_audit=status_audit,
        coverage_audit=pd.DataFrame(coverage_rows),
        component_completeness=pd.DataFrame(),
    )


def _assemble_from_legacy_long(
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
    """Archived long-format feature computation retained for compatibility."""

    spine = _spine(
        base_panel,
        firm_column=firm_column,
        year_column=year_column,
        prediction_time_column=prediction_time_column,
    )
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
    return FeatureSourceResult(
        panel=panel,
        validation_report=validation_report,
        file_audit=build_file_audit(
            status_frame=status_frame,
            raw_source_path=raw_source_path,
            raw_rows=len(raw_frame),
            usable_rows=len(long_values),
        ),
        identifier_audit=build_identifier_audit(
            firm_ids=firm_ids,
            panel_firms=panel_firms,
            firm_column=firm_column,
        ),
        availability_violations=pd.DataFrame(
            columns=[
                "feature_id",
                firm_column,
                year_column,
                "available_date",
                prediction_time_column,
            ]
        ),
        research_decision_audit=status_frame.loc[
            :, ["feature_id", "status", "reason_code"]
        ].copy(),
        coverage_audit=pd.DataFrame(coverage_rows),
        component_completeness=pd.DataFrame(completeness_rows),
    )


def build_file_audit(
    *,
    status_frame: pd.DataFrame,
    raw_source_path: Path,
    raw_rows: int,
    usable_rows: int,
    source_id: str = "financial_statement_core_long",
) -> pd.DataFrame:
    """Return one source-binding audit row per registered feature."""

    required = {"feature_id", "status", "reason_code"}
    if not required.issubset(status_frame.columns):
        raise ValueError(
            "P07 status frame missing file-audit columns: "
            f"{sorted(required - set(status_frame.columns))}"
        )
    audit = status_frame.loc[:, ["feature_id", "status", "reason_code"]].copy()
    if audit["feature_id"].duplicated().any():
        raise ValueError("P07 file audit requires unique feature_id rows")
    audit.insert(1, "source_id", source_id)
    audit.insert(2, "relative_path", raw_source_path.name)
    audit.insert(3, "raw_rows", int(raw_rows))
    audit.insert(4, "usable_rows", int(usable_rows))
    return audit


def build_identifier_audit(
    *,
    firm_ids: Sequence[str],
    panel_firms: set[str],
    firm_column: str,
) -> pd.DataFrame:
    """Return identifier audit with the canonical firm key first."""

    values = list(firm_ids)
    return pd.DataFrame(
        {
            firm_column: pd.Series(values, dtype="string"),
            "feature_store_firm_id": pd.Series(values, dtype="string"),
            "canonical_firm_id": pd.Series(values, dtype="string"),
            "mapping_status": pd.Series(
                ["MATCHED" if firm in panel_firms else "UNMATCHED" for firm in values],
                dtype="string",
            ),
            "ambiguity_flag": pd.Series([False] * len(values), dtype="bool"),
        }
    )
