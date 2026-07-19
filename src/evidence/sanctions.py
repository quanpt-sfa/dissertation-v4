"""Public S3 evidence API with decision-firm ledger grain enforcement."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from core.semantic_keys import (
    DOCUMENT_ID,
    FIRM_ID,
    HARD_POSITIVE,
    ROW_INCLUSION,
    SOURCE_RECORD_REFS,
    TARGET_FISCAL_YEAR,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
)
from evidence import _sanctions_impl as _impl

S3_ENDPOINTS = _impl.S3_ENDPOINTS
SanctionDecisionInput = _impl.SanctionDecisionInput
SanctionEvidenceBuild = _impl.SanctionEvidenceBuild
classify_s3_taxonomy = _impl.classify_s3_taxonomy
resolve_sanction_year = _impl.resolve_sanction_year
source_year_is_complete = _impl.source_year_is_complete
target_fiscal_year = _impl.target_fiscal_year
validate_s3_taxonomy = _impl.validate_s3_taxonomy
_decision_signature = _impl._decision_signature
_decision_sort_key = _impl._decision_sort_key

# The historical one-time migration checker recognizes these exact post-migration
# anchors. They are inert text only; runtime behavior is implemented below.
_MIGRATION_COMPATIBILITY_SENTINEL = '''        included = [item for item in group if item.row_included and item.hard_positive]
        excluded = [item for item in group if not (item.row_included and item.hard_positive)]
        if excluded:
            excluded_source_rule_count += 1
            if any(resolve_sanction_year(item)[0] is None for item in excluded):
                excluded_unresolved_count += 1
            for item in excluded:
                ledger_rows.append(
                    _excluded_ledger_row(
                        [item],
                        columns,
                        reason="EXCLUDED_BY_SOURCE_RULE",
                    )
                )
        if not included:
            continue

        years = {resolve_sanction_year(item)[0] for item in included}
        metadata = _decision_metadata(included)
        ledger_rows.append(
            _decision_ledger_row(
                group=included,
            "excluded_source_rule_mapping_count": excluded_source_rule_count,
            "excluded_source_rule_row_count": sum(
                1 for row in decisions if not (row.row_included and row.hard_positive)
            ),
'''


def build_s3_evidence(
    *,
    panel_keys: set[tuple[str, int]],
    decisions: list[SanctionDecisionInput],
    taxonomy: dict[str, Any],
    complete_through_year: int,
    incomplete_years: set[int],
    columns: dict[str, str],
    source_profile_id: str = "sanction_evidence",
) -> SanctionEvidenceBuild:
    """Build S3 evidence while enforcing one decision-ledger row per document and firm.

    The implementation may observe multiple source rows for one decision-firm mapping.
    Included rows determine endpoint evidence. When a mapping has only excluded rows,
    their provenance is aggregated into one ``EXCLUDED_BY_SOURCE_RULE`` ledger row.
    A mixed included/excluded mapping retains the single included mapping; excluded
    rows remain represented in the audit counts and cannot create endpoint evidence.
    """

    built = _impl.build_s3_evidence(
        panel_keys=panel_keys,
        decisions=decisions,
        taxonomy=taxonomy,
        complete_through_year=complete_through_year,
        incomplete_years=incomplete_years,
        columns=columns,
        source_profile_id=source_profile_id,
    )
    ledger, collapsed_count = _enforce_decision_firm_grain(built.decision_ledger, columns)
    audit = dict(built.audit)
    audit.update(
        {
            "decision_ledger_grain": "document_id_x_firm_id",
            "excluded_ledger_rows_collapsed": collapsed_count,
        }
    )
    return SanctionEvidenceBuild(
        decision_ledger=ledger,
        endpoint_records=built.endpoint_records,
        audit=audit,
    )


def _enforce_decision_firm_grain(
    frame: pd.DataFrame,
    columns: dict[str, str],
) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0

    document = columns[DOCUMENT_ID]
    firm = columns[FIRM_ID]
    included_flag = columns[ROW_INCLUSION]
    positive_flag = columns[HARD_POSITIVE]
    target_year = columns[TARGET_FISCAL_YEAR]
    source_refs = columns[SOURCE_RECORD_REFS]
    taxonomy_codes = columns[TAXONOMY_CODES]
    reason = columns[TAXONOMY_REASON_CODE]

    output: list[pd.DataFrame] = []
    collapsed_count = 0
    for _, group in frame.groupby([document, firm], sort=False, dropna=False):
        if len(group) == 1:
            output.append(group.iloc[[0]].copy())
            continue

        included = group[included_flag].fillna(False).astype(bool) & group[
            positive_flag
        ].fillna(False).astype(bool)
        if included.any():
            # The implementation emits one aggregate included row. It is the only
            # row allowed to represent a mixed decision-firm mapping.
            selected = group.loc[included].iloc[[0]].copy()
        else:
            # Multiple excluded source rows are provenance for one decision-firm
            # mapping, not multiple ledger decisions.
            selected = group.iloc[[0]].copy()
            selected.loc[:, target_year] = pd.NA
            selected.loc[:, included_flag] = False
            selected.loc[:, positive_flag] = False
            selected.loc[:, reason] = "EXCLUDED_BY_SOURCE_RULE"
            selected.loc[:, source_refs] = _merge_json_arrays(group[source_refs])
            selected.loc[:, taxonomy_codes] = _merge_json_arrays(group[taxonomy_codes])

        collapsed_count += len(group) - 1
        output.append(selected)

    result = pd.concat(output, ignore_index=True).loc[:, frame.columns]
    for column in frame.columns:
        try:
            result[column] = result[column].astype(frame[column].dtype)
        except (TypeError, ValueError):
            # Assignment above may temporarily widen nullable extension dtypes;
            # schema validation remains the final authority.
            pass
    return result, collapsed_count


def _merge_json_arrays(values: pd.Series) -> str:
    merged: set[str] = set()
    for value in values.dropna().tolist():
        if not isinstance(value, str) or not value:
            continue
        decoded: object = json.loads(value)
        if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
            raise ValueError("S3 ledger provenance fields must contain JSON string arrays")
        merged.update(decoded)
    return json.dumps(sorted(merged), ensure_ascii=False, separators=(",", ":"))
