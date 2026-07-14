"""Reconcile event-level and label-capability artifacts across two immutable runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.artifact_store import ArtifactStore
from core.pipeline import mapping, physical_columns, sequence
from core.schema_registry import contract_registry
from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    OUTER_FOLD,
    PREDICTION_TIME,
    SOURCE_ID,
)
from p01.registry import load_locked_registry


@dataclass(frozen=True)
class AuditRun:
    root: Path
    registry: dict[str, Any]
    protocol_hash: str
    store: ArtifactStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run-root", type=Path, required=True)
    parser.add_argument("--new-run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    old = _open_run(args.old_run_root)
    new = _open_run(args.new_run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    event_rows, event_totals = _event_reconciliation(new)
    l0_rows = _l0_counts(new)
    l1_rows = _l1_counts(new)
    overlap_rows = _channel_overlap(new)
    _write_csv(args.output_dir / "event_reconciliation.csv", event_rows)
    _write_csv(args.output_dir / "l0_counts_by_source_year.csv", l0_rows)
    _write_csv(args.output_dir / "l1_counts_by_year_fold.csv", l1_rows)
    _write_csv(args.output_dir / "channel_overlap_capability.csv", overlap_rows)

    before_after = _before_after(old, new)
    summary = {
        "old_run_id": old.root.name,
        "new_run_id": new.root.name,
        "old_protocol_hash": old.protocol_hash,
        "new_protocol_hash": new.protocol_hash,
        "old_run_tree": _tree_digest(old.root),
        "new_run_tree": _tree_digest(new.root),
        "event_totals": event_totals,
        **before_after,
    }
    (args.output_dir / "label_pipeline_reconciliation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _open_run(run_root: Path) -> AuditRun:
    registry, protocol_hash, resolved_root = load_locked_registry(
        run_root / "P00" / "registry.lock.json"
    )
    artifacts = mapping(registry.get("artifacts"), "artifacts")
    reproducibility = mapping(registry.get("reproducibility"), "reproducibility")
    recovery = reproducibility.get("recovery_policy", "quarantine_incomplete")
    if not isinstance(recovery, str):
        raise ValueError("reproducibility.recovery_policy must be a string")
    return AuditRun(
        root=resolved_root,
        registry=registry,
        protocol_hash=protocol_hash,
        store=ArtifactStore(
            resolved_root,
            artifacts,
            contract_registry(registry),
            protocol_hash,
            recovery_policy=recovery,
        ),
    )


def _event_reconciliation(run: AuditRun) -> tuple[list[dict[str, object]], dict[str, int]]:
    registry_rows = sequence(run.store.read("availability_registry", {}), "availability rows")
    panel = run.store.read("firm_year_panel", {})
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("firm_year_panel must be a DataFrame")
    columns = physical_columns(run.registry)
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    prediction_by_key = {
        (str(row[firm]), int(row[year])): pd.Timestamp(row[prediction])
        for row in cast(list[dict[str, Any]], panel.to_dict(orient="records"))
    }
    cutoff_raw = mapping(run.registry.get("risksets"), "risksets").get("data_cutoff")
    if not isinstance(cutoff_raw, str):
        raise ValueError("risksets.data_cutoff required")
    cutoff = pd.Timestamp(datetime.fromisoformat(cutoff_raw))
    grouped: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)
    for value in registry_rows:
        row = mapping(value, "availability row")
        source_id = str(row[SOURCE_ID])
        channel_id = str(row[CHANNEL_ID])
        fiscal_year = int(row[FISCAL_YEAR])
        counts = grouped[(source_id, channel_id, fiscal_year)]
        counts["raw_event_rows"] += 1
        status = str(row.get("status"))
        if status in {"ACCEPTED", "DUPLICATE_UPSTREAM_EVENT"}:
            counts["linked_events"] += 1
        if status == "UNLINKED_FIRM_YEAR":
            counts["unlinked_events"] += 1
        if status == "DUPLICATE_UPSTREAM_EVENT":
            counts["duplicate_events"] += 1
        if status == "EXCLUDED_BY_SOURCE_RULE":
            counts["excluded_by_source_rule"] += 1
        available = pd.Timestamp(str(row[AVAILABILITY_DATE]))
        if available > cutoff:
            counts["post_cutoff_events"] += 1
        prediction_time = prediction_by_key.get((str(row[FIRM_ID]), fiscal_year))
        if prediction_time is None or status not in {
            "ACCEPTED",
            "DUPLICATE_UPSTREAM_EVENT",
        }:
            continue
        horizon12 = prediction_time + pd.DateOffset(months=12)
        horizon24 = prediction_time + pd.DateOffset(months=24)
        if available < prediction_time:
            counts["pre_prediction_events"] += 1
        elif available == prediction_time:
            counts["same_day_events"] += 1
        elif available <= horizon12 and available <= cutoff:
            counts["in_12_month_events"] += 1
            counts["in_24_month_events"] += 1
        elif available <= horizon24 and available <= cutoff:
            counts["month_12_to_24_events"] += 1
            counts["in_24_month_events"] += 1
        elif available <= cutoff:
            counts["post_horizon_events"] += 1
    fields = [
        SOURCE_ID,
        CHANNEL_ID,
        FISCAL_YEAR,
        "raw_event_rows",
        "linked_events",
        "unlinked_events",
        "duplicate_events",
        "excluded_by_source_rule",
        "pre_prediction_events",
        "same_day_events",
        "in_12_month_events",
        "month_12_to_24_events",
        "in_24_month_events",
        "post_horizon_events",
        "post_cutoff_events",
    ]
    output: list[dict[str, object]] = [
        {
            SOURCE_ID: source_id,
            CHANNEL_ID: channel_id,
            FISCAL_YEAR: fiscal_year,
            **{field: counts[field] for field in fields[3:]},
        }
        for (source_id, channel_id, fiscal_year), counts in sorted(grouped.items())
    ]
    totals = {field: sum(int(str(row[field])) for row in output) for field in fields[3:]}
    return output, totals


def _l0_counts(run: AuditRun) -> list[dict[str, object]]:
    matrices = mapping(run.store.read("source_channel_matrices", {}), "source matrices")
    rows = [mapping(value, "source matrix row") for value in sequence(matrices["rows"], "rows")]
    sources = mapping(matrices.get("expected_sources"), "expected sources")
    grouped: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get(ELIGIBLE) is not True:
            continue
        fiscal_year = int(row[FISCAL_YEAR])
        mature = row.get(MATURE) is True
        outcomes = mapping(row.get("source_outcomes"), "source outcomes")
        for source_id in sorted(sources):
            counts = grouped[(source_id, fiscal_year)]
            counts[ELIGIBLE] += 1
            counts["opportunity_unknown"] += 1
            if not mature:
                counts["immature"] += 1
                continue
            counts[MATURE] += 1
            value = outcomes.get(source_id)
            if value is True:
                counts["positive"] += 1
            elif value is False:
                counts["explicit_negative"] += 1
            else:
                counts["unknown"] += 1
    return [
        {
            SOURCE_ID: source_id,
            CHANNEL_ID: sources[source_id],
            FISCAL_YEAR: fiscal_year,
            ELIGIBLE: counts[ELIGIBLE],
            MATURE: counts[MATURE],
            "positive": counts["positive"],
            "explicit_negative": counts["explicit_negative"],
            "unknown": counts["unknown"],
            "immature": counts["immature"],
            "source_opportunity_observed": 0,
            "opportunity_unknown": counts["opportunity_unknown"],
            "opportunity_status": "UNKNOWN_NO_OPPORTUNITY_INDICATOR",
        }
        for (source_id, fiscal_year), counts in sorted(grouped.items())
    ]


def _l1_counts(run: AuditRun) -> list[dict[str, object]]:
    matrices = mapping(run.store.read("source_channel_matrices", {}), "source matrices")
    rows = [mapping(value, "source matrix row") for value in sequence(matrices["rows"], "rows")]
    fold_rows = sequence(run.store.read("fold_eligibility", {}), "fold eligibility")
    folds = {
        str(mapping(value, "fold row")[OUTER_FOLD]): mapping(value, "fold row")
        for value in fold_rows
    }
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get(ELIGIBLE) is not True:
            continue
        counts = grouped[int(row[FISCAL_YEAR])]
        counts[ELIGIBLE] += 1
        if row.get(MATURE) is not True:
            counts["immature"] += 1
            continue
        counts[MATURE] += 1
        value = row.get("l1")
        if value is True:
            counts["positive"] += 1
        elif value is False:
            counts["explicit_negative"] += 1
        else:
            counts["unknown"] += 1
    output: list[dict[str, object]] = []
    for fiscal_year, counts in sorted(grouped.items()):
        fold = folds.get(str(fiscal_year))
        classes = int(counts["positive"] > 0) + int(counts["explicit_negative"] > 0)
        output.append(
            {
                FISCAL_YEAR: fiscal_year,
                OUTER_FOLD: str(fiscal_year) if fold else "",
                "eligible_rows": counts[ELIGIBLE],
                "mature_rows": counts[MATURE],
                "immature_rows": counts["immature"],
                "l1_positive": counts["positive"],
                "l1_explicit_negative": counts["explicit_negative"],
                "l1_unknown": counts["unknown"],
                "observed_binary_class_count": classes,
                "fold_role": fold.get("assigned_role") if fold else "development_history",
                "reason_code": fold.get("reason_code") if fold else "NOT_AN_OUTER_FOLD",
            }
        )
    return output


def _channel_overlap(run: AuditRun) -> list[dict[str, object]]:
    matrices = mapping(run.store.read("source_channel_matrices", {}), "source matrices")
    rows = [mapping(value, "source matrix row") for value in sequence(matrices["rows"], "rows")]
    channels = [str(value) for value in sequence(matrices["expected_channels"], "channels")]
    l2 = mapping(matrices.get("l2_scoring"), "l2 scoring")
    l3 = mapping(run.store.read("l3_pilot_capability", {}), "L3 capability")
    anchor = mapping(run.store.read("anchor_capability", {}), "anchor capability")
    distribution = Counter(int(row.get("observed_channel_count", 0)) for row in rows)
    mature_distribution = Counter(
        int(row.get("observed_channel_count", 0)) for row in rows if row.get(MATURE) is True
    )
    l2_computable = sum(row.get("l2_score") is not None for row in rows)
    l3_computable = sum(row.get("l3_posterior_mean") is not None for row in rows)
    output: list[dict[str, object]] = [
        {
            "record_type": "summary",
            "channel_a": "",
            "channel_b": "",
            "observed_channel_count": "",
            "firm_year_count": len(rows),
            "mature_firm_year_count": sum(row.get(MATURE) is True for row in rows),
            "valid_channel_count": len(channels),
            "overlap_firm_year_count": 0,
            "l2_computable_firm_year_count": l2_computable,
            "l2_status": l2.get("status"),
            "l2_reason_code": l2.get("reason_code"),
            "l3_computable_firm_year_count": l3_computable,
            "l3_status": l3.get("status"),
            "l3_reason_code": l3.get("reason_code") or l3.get("reason"),
            "anchor_source_status": anchor.get("status"),
            "anchor_source_ids": ";".join(
                str(value) for value in anchor.get("anchor_source_ids", [])
            ),
        }
    ]
    for count in sorted(distribution):
        output.append(
            {
                "record_type": "observed_channel_distribution",
                "channel_a": "",
                "channel_b": "",
                "observed_channel_count": count,
                "firm_year_count": distribution[count],
                "mature_firm_year_count": mature_distribution[count],
                "valid_channel_count": len(channels),
                "overlap_firm_year_count": "",
                "l2_computable_firm_year_count": l2_computable,
                "l2_status": l2.get("status"),
                "l2_reason_code": l2.get("reason_code"),
                "l3_computable_firm_year_count": l3_computable,
                "l3_status": l3.get("status"),
                "l3_reason_code": l3.get("reason_code") or l3.get("reason"),
                "anchor_source_status": anchor.get("status"),
                "anchor_source_ids": ";".join(
                    str(value) for value in anchor.get("anchor_source_ids", [])
                ),
            }
        )
    for first, second in itertools.combinations(channels, 2):
        overlap = sum(
            mapping(row.get("channel_outcomes"), "channel outcomes").get(first) is not None
            and mapping(row.get("channel_outcomes"), "channel outcomes").get(second) is not None
            for row in rows
        )
        output.append(
            {
                "record_type": "pair_overlap",
                "channel_a": first,
                "channel_b": second,
                "observed_channel_count": "",
                "firm_year_count": len(rows),
                "mature_firm_year_count": "",
                "valid_channel_count": len(channels),
                "overlap_firm_year_count": overlap,
                "l2_computable_firm_year_count": l2_computable,
                "l2_status": l2.get("status"),
                "l2_reason_code": l2.get("reason_code"),
                "l3_computable_firm_year_count": l3_computable,
                "l3_status": l3.get("status"),
                "l3_reason_code": l3.get("reason_code") or l3.get("reason"),
                "anchor_source_status": anchor.get("status"),
                "anchor_source_ids": ";".join(
                    str(value) for value in anchor.get("anchor_source_ids", [])
                ),
            }
        )
    return output


def _before_after(old: AuditRun, new: AuditRun) -> dict[str, object]:
    old_values = _sealed_values(old)
    new_values = _sealed_values(new)
    added = sorted(new_values.items() - old_values.items())
    removed = sorted(old_values.items() - new_values.items())
    return {
        "old_sealed_outcome_count": len(old_values),
        "old_positive_count": sum(value is True for value in old_values.values()),
        "old_explicit_negative_count": sum(value is False for value in old_values.values()),
        "new_sealed_outcome_count": len(new_values),
        "new_positive_count": sum(value is True for value in new_values.values()),
        "new_explicit_negative_count": sum(value is False for value in new_values.values()),
        "changed_firm_years_added": [
            {FIRM_ID: key[0], FISCAL_YEAR: key[1], OUTCOME: value} for key, value in added
        ],
        "changed_firm_years_removed": [
            {FIRM_ID: key[0], FISCAL_YEAR: key[1], OUTCOME: value} for key, value in removed
        ],
        "old_fold_eligibility": run_value(old, "fold_eligibility"),
        "new_fold_eligibility": run_value(new, "fold_eligibility"),
    }


def _sealed_values(run: AuditRun) -> dict[tuple[str, int], bool]:
    frame = run.store.read("sealed_outcome_store", {})
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("sealed outcome store must be a DataFrame")
    columns = physical_columns(run.registry)
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    outcome = columns[OUTCOME]
    return {
        (str(row[firm]), int(row[year])): bool(row[outcome])
        for row in cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    }


def run_value(run: AuditRun, artifact_id: str) -> object:
    return run.store.read(artifact_id, {})


def _tree_digest(root: Path) -> dict[str, object]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    lines: list[str] = []
    total_bytes = 0
    for path in files:
        data = path.read_bytes()
        total_bytes += len(data)
        lines.append(
            f"{path.relative_to(root).as_posix()}\0{len(data)}\0{hashlib.sha256(data).hexdigest()}"
        )
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no audit rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
