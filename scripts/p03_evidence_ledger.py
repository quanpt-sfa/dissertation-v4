"""P03 entry point for the single final firm-year Parquet input.

The legacy sanction-row parser is retained as a compatibility seam for the
existing parser regression test. Production evidence construction is delegated
to :mod:`scripts.p03_final_evidence_ledger`.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.semantic_keys import (
    AFFECTED_FISCAL_YEAR,
    CONSTRUCT_FAMILY,
    CONSTRUCT_TARGET,
    DECISION_DATE,
    DECISION_NUMBER,
    DECISION_TOTAL_FINE,
    DOCUMENT_ID,
    FIRM_ID,
    HARD_POSITIVE,
    HAS_FINE,
    HAS_REMEDY,
    HAS_SUSPENSION,
    HAS_WARNING,
    LABEL_KNOWN_DATE,
    LEGACY_EVENT_ID,
    NORMALIZED_VIOLATION_CODE,
    PERIOD_LINK_CONFIDENCE,
    PERIOD_LINK_SOURCE,
    PRIMARY_VIOLATION_L1,
    PRIMARY_VIOLATION_L2,
    PUBLISH_DATE,
    ROW_INCLUSION,
    SANCTION_YEAR,
    SOURCE_FILE,
    SOURCE_ROW,
    TARGET_FISCAL_YEAR,
)
from evidence.sanctions import SanctionDecisionInput
from p01.models import SourceSpec
from p01.readers import iter_rows
from p02.builder import normalize_entity_field, resolve_entity_link
from p02.models import EntityResolutionSpec
from scripts.p03_final_evidence_ledger import main


def _sanction_rows(
    *,
    source_id: str,
    path: Path,
    spec: object,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
) -> list[SanctionDecisionInput]:
    """Parse an archived event-level sanction source for regression compatibility."""

    if not isinstance(spec, SourceSpec):
        raise TypeError("locked source specification required")
    required = {
        FIRM_ID,
        DOCUMENT_ID,
        ROW_INCLUSION,
        HARD_POSITIVE,
        TARGET_FISCAL_YEAR,
    }
    if not required.issubset(semantics):
        raise ValueError(
            f"source={source_id}: unresolved S3 semantics {sorted(required - set(semantics))}"
        )
    result: list[SanctionDecisionInput] = []
    for row in iter_rows(path, spec):
        firm_raw = _required(row.get(str(semantics[FIRM_ID])), FIRM_ID)
        normalized = normalize_entity_field(str(firm_raw), entity)
        canonical, _ = resolve_entity_link(source_id, str(firm_raw), normalized, entity)
        row_included = _boolean(row.get(str(semantics[ROW_INCLUSION])))
        hard_positive = _boolean(row.get(str(semantics[HARD_POSITIVE])))
        if row_included is None or hard_positive is None:
            raise ValueError(
                f"source={source_id}: S3 inclusion and hard-positive flags are required"
            )
        result.append(
            SanctionDecisionInput(
                document_id=str(_required(row.get(str(semantics[DOCUMENT_ID])), DOCUMENT_ID)),
                firm_id=canonical,
                decision_number=_optional_text(row, semantics, DECISION_NUMBER),
                sanction_year=_optional_int(row, semantics, SANCTION_YEAR),
                decision_date=_optional_datetime(row, semantics, DECISION_DATE),
                publish_date=_optional_datetime(row, semantics, PUBLISH_DATE),
                label_known_date=_optional_datetime(row, semantics, LABEL_KNOWN_DATE),
                affected_fiscal_year=_optional_int(row, semantics, AFFECTED_FISCAL_YEAR),
                primary_violation_l1=_optional_text(row, semantics, PRIMARY_VIOLATION_L1),
                primary_violation_l2=_optional_text(row, semantics, PRIMARY_VIOLATION_L2),
                construct_family=_optional_text(row, semantics, CONSTRUCT_FAMILY),
                construct_target=_optional_text(row, semantics, CONSTRUCT_TARGET),
                normalized_violation_code=_optional_text(row, semantics, NORMALIZED_VIOLATION_CODE),
                row_included=row_included,
                hard_positive=hard_positive,
                legacy_event_id=_optional_text(row, semantics, LEGACY_EVENT_ID),
                period_link_source=_optional_text(row, semantics, PERIOD_LINK_SOURCE),
                period_link_confidence=_optional_text(row, semantics, PERIOD_LINK_CONFIDENCE),
                total_fine=_optional_float(row, semantics, DECISION_TOTAL_FINE),
                has_fine=_optional_boolean(row, semantics, HAS_FINE),
                has_suspension=_optional_boolean(row, semantics, HAS_SUSPENSION),
                has_warning=_optional_boolean(row, semantics, HAS_WARNING),
                has_remedy=_optional_boolean(row, semantics, HAS_REMEDY),
                source_ref=_source_ref(row, semantics, SOURCE_FILE, SOURCE_ROW),
                target_fiscal_year=_optional_int(row, semantics, TARGET_FISCAL_YEAR),
            )
        )
    return result


def _required(value: object, field: str) -> object:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"evidence row missing {field}")
    return value


def _boolean(value: object) -> bool | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().casefold()
    if normalized in {"true", "yes", "y", "1"}:
        return True
    if normalized in {"false", "no", "n", "0"}:
        return False
    raise ValueError(f"evidence outcome is not a registered binary representation: {value}")


def _optional_text(row: dict[str, object], semantics: dict[str, Any], name: str) -> str | None:
    column = semantics.get(name)
    if not isinstance(column, str):
        return None
    value = row.get(column)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _optional_int(row: dict[str, object], semantics: dict[str, Any], name: str) -> int | None:
    value = _optional_text(row, semantics, name)
    return None if value is None else int(float(value))


def _optional_datetime(
    row: dict[str, object], semantics: dict[str, Any], name: str
) -> datetime | None:
    value = _optional_text(row, semantics, name)
    return None if value is None else datetime.fromisoformat(value)


def _optional_float(row: dict[str, object], semantics: dict[str, Any], name: str) -> float | None:
    value = _optional_text(row, semantics, name)
    return None if value is None else float(value)


def _optional_boolean(row: dict[str, object], semantics: dict[str, Any], name: str) -> bool | None:
    column = semantics.get(name)
    return None if not isinstance(column, str) else _boolean(row.get(column))


def _source_ref(
    row: dict[str, object],
    semantics: dict[str, Any],
    file_semantic: str,
    row_semantic: str,
) -> str:
    file_value = _optional_text(row, semantics, file_semantic) or "registered_source"
    row_value = _optional_text(row, semantics, row_semantic) or "unavailable"
    return f"{file_value}#{row_value}"


if __name__ == "__main__":
    raise SystemExit(main())
