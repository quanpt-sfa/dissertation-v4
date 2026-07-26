"""Load raw financial-statement values into the long form the compute engine needs.

P07 computes features directly from the registered ``financial_statement_core_long``
source instead of a pre-built store. This module turns the raw long file into a
``(firm_master_id, fiscal_year, audit_status, source_item_id, value_numeric)`` frame,
reusing P02's entity resolution so the canonical firm id matches the P02 spine
exactly. It is the single sanctioned raw-reading boundary for feature computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from features.compute import AUDIT, FIRM, ITEM, VALUE, YEAR
from p02.builder import normalize_entity_field, resolve_entity_link
from p02.models import EntityResolutionSpec

# Physical column names in financial_statement_full_long.
_RAW_TICKER = "issuer_ticker"
_RAW_YEAR = "fiscal_year"
_RAW_AUDIT = "audit_status"
_RAW_SCOPE = "scope"
_RAW_ITEM = "source_item_id"
_RAW_VALUE = "value_numeric"
_RAW_UNIT = "unit"

_ANALYSED_SCOPE = "consolidated"
_ANALYSED_UNIT = "VND"
_SOURCE_ID = "financial_statement_core_long"


def read_financial_statement_long(
    path: Path, reader: Mapping[str, object] | None = None
) -> pd.DataFrame:
    """Read the raw long financial-statement file (sole raw-reading boundary)."""
    options = dict(reader or {})
    frame = pd.read_csv(  # noqa: PD901 - registered raw boundary
        path,
        compression="infer",
        dtype="string",
        keep_default_na=False,
        encoding=str(options.get("encoding", "utf-8-sig")),
    )
    required = {_RAW_TICKER, _RAW_YEAR, _RAW_AUDIT, _RAW_SCOPE, _RAW_ITEM, _RAW_VALUE, _RAW_UNIT}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"financial statement long file is missing columns: {sorted(missing)}")
    return frame


def build_long_values(
    raw_frame: pd.DataFrame,
    *,
    entity_spec: EntityResolutionSpec,
    source_id: str = _SOURCE_ID,
) -> pd.DataFrame:
    """Filter to the analysed scope/unit and resolve identity to firm_master_id."""
    frame = raw_frame.loc[
        (raw_frame[_RAW_SCOPE] == _ANALYSED_SCOPE) & (raw_frame[_RAW_UNIT] == _ANALYSED_UNIT)
    ].copy()

    tickers = frame[_RAW_TICKER].astype("string")
    canonical_by_ticker: dict[str, str] = {}
    for ticker in sorted(set(tickers.dropna())):
        raw_value = str(ticker)
        normalized = normalize_entity_field(raw_value, entity_spec)
        canonical, _method = resolve_entity_link(source_id, raw_value, normalized, entity_spec)
        canonical_by_ticker[raw_value] = canonical

    firm = tickers.map(canonical_by_ticker).astype("string")
    year = pd.to_numeric(frame[_RAW_YEAR], errors="coerce").astype("Int16")
    value = pd.to_numeric(frame[_RAW_VALUE].replace("", pd.NA), errors="coerce").astype("float64")

    long_values = pd.DataFrame(
        {
            FIRM: firm,
            YEAR: year,
            AUDIT: frame[_RAW_AUDIT].astype("string"),
            ITEM: frame[_RAW_ITEM].astype("string"),
            VALUE: value,
        }
    )
    long_values = long_values.loc[long_values[FIRM].notna() & long_values[YEAR].notna()].copy()
    long_values[YEAR] = long_values[YEAR].astype("int16")

    duplicated = long_values.duplicated([FIRM, YEAR, AUDIT, ITEM], keep=False)
    if duplicated.any():
        sample = long_values.loc[duplicated, [FIRM, YEAR, AUDIT, ITEM]].head(10).to_dict("records")
        raise ValueError(f"raw long values have duplicate firm-year-audit-item keys: {sample}")
    return long_values.reset_index(drop=True)
