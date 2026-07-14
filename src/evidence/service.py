"""Pure P03 event normalization, upstream deduplication, and lag decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    FIRM_ID,
    FISCAL_YEAR,
    OUTCOME,
    PREDICTION_TIME,
    SOURCE_ID,
)


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    channel_id: str
    firm_id: str
    fiscal_year: int
    availability_date: datetime
    outcome: bool | None
    event_id: str
    event_cluster_id: str


@dataclass(frozen=True)
class EvidenceBuildResult:
    ledger: pd.DataFrame
    availability_registry: list[dict[str, object]]
    lag_decomposition: dict[str, object]


def build_evidence_ledger(
    *,
    panel: pd.DataFrame,
    records: list[EvidenceRecord],
    columns: dict[str, str],
    fiscal_year_end_month_day: str,
    lag_tolerance_days: int,
) -> EvidenceBuildResult:
    """Build a source-level ledger without treating absent or immature data as zero."""
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    required = {firm, year, prediction}
    if not required.issubset(panel.columns):
        raise ValueError(f"P03 panel missing columns {sorted(required - set(panel.columns))}")
    panel_times: dict[tuple[str, int], datetime] = {}
    for raw in cast(list[dict[str, Any]], panel.to_dict(orient="records")):
        key = (str(raw[firm]), int(raw[year]))
        if key in panel_times:
            raise ValueError(f"P03 duplicate panel key: {key}")
        panel_times[key] = _as_datetime(raw[prediction])
    seen_clusters: set[tuple[str, int, str]] = set()
    accepted: list[tuple[EvidenceRecord, datetime]] = []
    availability_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []

    for record in sorted(
        records,
        key=lambda item: (
            item.firm_id,
            item.fiscal_year,
            item.event_cluster_id,
            item.source_id,
            item.event_id,
        ),
    ):
        key = (record.firm_id, record.fiscal_year)
        if key not in panel_times:
            availability_rows.append(_availability_row(record, "UNLINKED_FIRM_YEAR"))
            continue
        cluster_key = (record.firm_id, record.fiscal_year, record.event_cluster_id)
        if cluster_key in seen_clusters:
            availability_rows.append(_availability_row(record, "DUPLICATE_UPSTREAM_EVENT"))
            continue
        seen_clusters.add(cluster_key)
        prediction_time = panel_times[key]
        accepted.append((record, prediction_time))
        availability_rows.append(_availability_row(record, "ACCEPTED"))
        year_end = _fiscal_year_end(record.fiscal_year, fiscal_year_end_month_day)
        report_lag = (prediction_time.date() - year_end).days
        detection_lag = (record.availability_date.date() - prediction_time.date()).days
        total_lag = (record.availability_date.date() - year_end).days
        identity_error = total_lag - report_lag - detection_lag
        if abs(identity_error) > lag_tolerance_days:
            raise ValueError(
                f"lag identity violated for event={record.event_id}: error={identity_error}"
            )
        lag_rows.append(
            {
                "event_id": record.event_id,
                SOURCE_ID: record.source_id,
                "report_lag_days": report_lag,
                "detection_lag_days": detection_lag,
                "total_lag_days": total_lag,
                "identity_error_days": identity_error,
            }
        )

    grouped: dict[tuple[str, int, str, str], list[tuple[EvidenceRecord, datetime]]] = {}
    for item in accepted:
        record, _ = item
        grouped.setdefault(
            (record.firm_id, record.fiscal_year, record.source_id, record.channel_id), []
        ).append(item)
    ledger_rows: list[dict[str, Any]] = []
    for (firm_id, fiscal_year, source_id, channel_id), items in sorted(grouped.items()):
        outcomes = [record.outcome for record, _ in items]
        outcome: bool | None
        if any(value is True for value in outcomes):
            outcome = True
        elif outcomes and all(value is False for value in outcomes):
            outcome = False
        else:
            outcome = None
        availability = min(record.availability_date for record, _ in items)
        ledger_rows.append(
            {
                columns[FIRM_ID]: firm_id,
                columns[FISCAL_YEAR]: fiscal_year,
                columns[SOURCE_ID]: source_id,
                columns[CHANNEL_ID]: channel_id,
                columns[AVAILABILITY_DATE]: availability,
                columns[OUTCOME]: outcome,
            }
        )
    ledger = pd.DataFrame(
        ledger_rows,
        columns=[
            columns[FIRM_ID],
            columns[FISCAL_YEAR],
            columns[SOURCE_ID],
            columns[CHANNEL_ID],
            columns[AVAILABILITY_DATE],
            columns[OUTCOME],
        ],
    )
    if not ledger.empty:
        ledger[columns[FIRM_ID]] = ledger[columns[FIRM_ID]].astype("string")
        ledger[columns[FISCAL_YEAR]] = ledger[columns[FISCAL_YEAR]].astype("int16")
        ledger[columns[SOURCE_ID]] = ledger[columns[SOURCE_ID]].astype("string")
        ledger[columns[CHANNEL_ID]] = ledger[columns[CHANNEL_ID]].astype("string")
        ledger[columns[AVAILABILITY_DATE]] = pd.to_datetime(
            ledger[columns[AVAILABILITY_DATE]]
        ).astype("datetime64[ns]")
        ledger[columns[OUTCOME]] = ledger[columns[OUTCOME]].astype("boolean")
    return EvidenceBuildResult(
        ledger=ledger,
        availability_registry=availability_rows,
        lag_decomposition={
            "records": lag_rows,
            "accepted_event_count": len(lag_rows),
            "deduplicated_event_count": sum(
                row["status"] == "DUPLICATE_UPSTREAM_EVENT" for row in availability_rows
            ),
            "lag_identity_tolerance_days": lag_tolerance_days,
        },
    )


def _availability_row(record: EvidenceRecord, status: str) -> dict[str, object]:
    return {
        "event_id": record.event_id,
        "event_cluster_id": record.event_cluster_id,
        SOURCE_ID: record.source_id,
        CHANNEL_ID: record.channel_id,
        AVAILABILITY_DATE: record.availability_date.isoformat(),
        "status": status,
    }


def _as_datetime(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    return parsed.to_pydatetime()


def _fiscal_year_end(fiscal_year: int, month_day: str) -> date:
    month, day = (int(value) for value in month_day.split("-"))
    return date(fiscal_year, month, day)
