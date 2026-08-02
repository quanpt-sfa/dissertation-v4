"""Adapters from internal evidence objects to physical artifact contracts."""

from __future__ import annotations

import hashlib
import json
from typing import cast

import pandas as pd

from core.semantic_keys import (
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    DOCUMENT_ID,
    FIRM_ID,
    HARD_POSITIVE,
    LEGACY_EVENT_ID,
    NORMALIZED_VIOLATION_CODE,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    ROW_INCLUSION,
    SOURCE_RECORD_REFS,
    TARGET_FISCAL_YEAR,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
)

_DECISION_LEDGER_TAIL = [
    TARGET_FISCAL_YEAR,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    NORMALIZED_VIOLATION_CODE,
    HARD_POSITIVE,
    ROW_INCLUSION,
    LEGACY_EVENT_ID,
    PERIOD_LINK_SOURCE,
    PERIOD_LINK_CONFIDENCE,
    SOURCE_RECORD_REFS,
    TAXONOMY_CODES,
    TAXONOMY_REASON_CODE,
]
_AMBIGUOUS_TARGET_YEAR_REASON = "MULTIPLE_TARGET_FISCAL_YEARS_IN_FINAL_PROVENANCE"
_AMBIGUOUS_TARGET_YEAR_CONFIDENCE = "unresolved_multiple_target_years"
_INVALID_DOCUMENT_ID_TOKENS = {
    "",
    "[]",
    "{}",
    "null",
    "none",
    "nan",
    "na",
    "n/a",
    "<na>",
}


def bind_sanction_decision_ledger_columns(
    frame: pd.DataFrame,
    *,
    firm_column: str,
) -> pd.DataFrame:
    """Bind the wide S3 ledger to the compiled physical artifact contract.

    The wide builder works with the logical firm key and initially emits one row
    per document, firm, and preaggregated target year. The production decision
    ledger is deliberately coarser: one row per document and firm. This adapter
    therefore binds the physical firm key, aggregates repeated endpoint
    provenance, and represents conflicting target years as an unresolved mapping
    instead of inventing a new document identity or weakening the schema.
    """

    if not firm_column:
        raise ValueError("sanction decision ledger requires a physical firm column")

    result = frame.copy()
    if firm_column != FIRM_ID:
        if firm_column in result.columns and FIRM_ID in result.columns:
            raise ValueError(
                "sanction decision ledger contains both internal and physical firm keys"
            )
        if firm_column not in result.columns:
            if FIRM_ID not in result.columns:
                raise ValueError("sanction decision ledger is missing its internal firm key")
            result = result.rename(columns={FIRM_ID: firm_column})
    elif firm_column not in result.columns:
        raise ValueError("sanction decision ledger is missing its physical firm key")

    ordered = [DOCUMENT_ID, firm_column, *_DECISION_LEDGER_TAIL]
    missing = [column for column in ordered if column not in result.columns]
    extras = [column for column in result.columns if column not in ordered]
    if missing or extras:
        raise ValueError(
            "sanction decision ledger columns differ from the production contract: "
            f"missing={missing}, extras={extras}"
        )
    bound = result.loc[:, ordered]
    repaired = _repair_placeholder_document_ids(bound, firm_column)
    return _enforce_document_firm_grain(repaired, firm_column)


def _repair_placeholder_document_ids(frame: pd.DataFrame, firm_column: str) -> pd.DataFrame:
    """Replace serialized empty document keys with stable provenance-derived keys.

    The final firm-year builder may receive nested JSON where an empty document-id
    array is serialized as the literal string ``"[]"``. Treating that placeholder
    as an actual decision identifier makes all years for a firm collide at the
    locked ``document_id × firm`` grain. A placeholder is therefore replaced only
    when non-empty source provenance can deterministically identify the embedded
    record. Missing provenance remains a hard error rather than an invented key.
    """

    if frame.empty:
        return frame.copy()

    result = frame.copy()
    document_text = result[DOCUMENT_ID].astype("string").str.strip()
    invalid = document_text.isna() | document_text.str.casefold().isin(_INVALID_DOCUMENT_ID_TOKENS)
    if not invalid.any():
        return result

    for index in result.index[invalid]:
        refs = _normalized_source_refs(result.at[index, SOURCE_RECORD_REFS])
        if not refs:
            firm_id = result.at[index, firm_column]
            target_year = result.at[index, TARGET_FISCAL_YEAR]
            raise ValueError(
                "sanction decision ledger contains an empty-container document identifier "
                "without non-empty provenance from which to derive a stable key: "
                f"firm={firm_id}, target_fiscal_year={target_year}"
            )
        canonical_refs = json.dumps(
            refs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical_refs.encode("utf-8")).hexdigest()[:16]
        result.at[index, DOCUMENT_ID] = f"embedded-provenance-{digest}"

    return result


def _normalized_source_refs(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        raise ValueError("S3 ledger source_record_refs must contain a JSON string array")
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("S3 ledger source_record_refs must contain valid JSON") from exc
    if not isinstance(decoded, list):
        raise ValueError("S3 ledger source_record_refs must contain a JSON string array")

    normalized: set[str] = set()
    for item in cast(list[object], decoded):
        fragment = _normalize_provenance_fragment(item)
        if fragment is None:
            continue
        if isinstance(fragment, (dict, list)):
            normalized.add(
                json.dumps(
                    fragment,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            text = str(fragment).strip()
            if text:
                normalized.add(text)
    return sorted(normalized)


def _normalize_provenance_fragment(value: object) -> object | None:
    """Decode nested JSON and remove null-like or empty-container fragments."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        normalized_dict: dict[str, object] = {}
        for key, child in cast(dict[object, object], value).items():
            normalized = _normalize_provenance_fragment(child)
            if normalized is not None:
                normalized_dict[str(key)] = normalized
        return normalized_dict or None
    if isinstance(value, (list, tuple)):
        normalized_list: list[object] = [
            normalized
            for child in cast(list[object] | tuple[object, ...], value)
            if (normalized := _normalize_provenance_fragment(child)) is not None
        ]
        return normalized_list or None
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in _INVALID_DOCUMENT_ID_TOKENS:
            return None
        if text.startswith(("[", "{")):
            try:
                decoded: object = json.loads(text)
            except json.JSONDecodeError:
                return text
            return _normalize_provenance_fragment(decoded)
        return text
    return value


def _enforce_document_firm_grain(frame: pd.DataFrame, firm_column: str) -> pd.DataFrame:
    """Collapse repeated wide-provenance rows to the locked document-firm grain."""

    if frame.empty or not frame.duplicated([DOCUMENT_ID, firm_column], keep=False).any():
        return frame.copy()

    output: list[pd.DataFrame] = []
    scalar_fields = [
        PRIMARY_VIOLATION_L1,
        PRIMARY_VIOLATION_L2,
        CONSTRUCT_FAMILY,
        NORMALIZED_VIOLATION_CODE,
        LEGACY_EVENT_ID,
        PERIOD_LINK_SOURCE,
    ]
    for key, group in frame.groupby([DOCUMENT_ID, firm_column], sort=False, dropna=False):
        if len(group) == 1:
            output.append(group.iloc[[0]].copy())
            continue

        selected = group.iloc[[0]].copy()
        for column in scalar_fields:
            values = _nonempty_strings(group[column])
            if len(values) > 1:
                raise ValueError(
                    "sanction decision ledger has conflicting scalar provenance for "
                    f"document-firm={key}, column={column}, values={values}"
                )
            if values:
                selected.loc[:, column] = values[0]

        years = sorted(
            {
                int(value)
                for value in pd.to_numeric(group[TARGET_FISCAL_YEAR], errors="coerce")
                .dropna()
                .tolist()
            }
        )
        selected.loc[:, TARGET_FISCAL_YEAR] = years[0] if len(years) == 1 else pd.NA
        selected.loc[:, CONSTRUCT_TARGET] = _merge_json_string_arrays(group[CONSTRUCT_TARGET])
        selected.loc[:, SOURCE_RECORD_REFS] = _merge_json_string_arrays(group[SOURCE_RECORD_REFS])
        selected.loc[:, TAXONOMY_CODES] = _merge_json_string_arrays(group[TAXONOMY_CODES])
        selected.loc[:, HARD_POSITIVE] = bool(group[HARD_POSITIVE].fillna(False).any())
        selected.loc[:, ROW_INCLUSION] = bool(group[ROW_INCLUSION].fillna(False).any())

        if len(years) > 1:
            selected.loc[:, PERIOD_LINK_CONFIDENCE] = _AMBIGUOUS_TARGET_YEAR_CONFIDENCE
            selected.loc[:, TAXONOMY_REASON_CODE] = _AMBIGUOUS_TARGET_YEAR_REASON
        else:
            confidence = _nonempty_strings(group[PERIOD_LINK_CONFIDENCE])
            reasons = _nonempty_strings(group[TAXONOMY_REASON_CODE])
            if len(confidence) > 1:
                raise ValueError(
                    "sanction decision ledger has conflicting period-link confidence for "
                    f"document-firm={key}, values={confidence}"
                )
            if len(reasons) > 1:
                raise ValueError(
                    "sanction decision ledger has conflicting taxonomy reasons for "
                    f"document-firm={key}, values={reasons}"
                )
            if confidence:
                selected.loc[:, PERIOD_LINK_CONFIDENCE] = confidence[0]
            if reasons:
                selected.loc[:, TAXONOMY_REASON_CODE] = reasons[0]

        output.append(selected)

    result = pd.concat(output, ignore_index=True).loc[:, frame.columns]
    for column in frame.columns:
        try:
            result[column] = result[column].astype(frame[column].dtype)
        except (TypeError, ValueError):
            pass
    if result.duplicated([DOCUMENT_ID, firm_column]).any():
        raise ValueError("sanction decision ledger document-firm grain remains non-unique")
    return result


def _merge_json_string_arrays(values: pd.Series) -> str:
    merged: set[str] = set()
    for value in values.dropna().tolist():
        if not isinstance(value, str) or not value:
            continue
        decoded: object = json.loads(value)
        if not isinstance(decoded, list):
            raise ValueError("S3 ledger aggregate fields must contain JSON string arrays")
        for item in cast(list[object], decoded):
            if not isinstance(item, str):
                raise ValueError("S3 ledger aggregate fields must contain JSON string arrays")
            merged.add(item)
    return json.dumps(sorted(merged), ensure_ascii=False, separators=(",", ":"))


def _nonempty_strings(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna().tolist() if str(value)})
