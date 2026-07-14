"""P15 CLI: open and validate the snapshot-locked known-case ledger."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, outer_fold_ids, physical_columns, sequence
from core.semantic_keys import FIRM_ID, FISCAL_YEAR, OUTER_FOLD
from known_cases.service import evaluate_known_cases
from p01.readers import hash_file, iter_rows
from p01.registry import resolve_source
from p02.builder import normalize_entity_field, resolve_entity_link
from p02.models import EntityResolutionSpec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    loaded = load_run(registry_path=args.registry, run_id=args.run_id, step_id="P15", state="GATE2")
    mapping(loaded.context.read("gate2_verdict", {}), "Gate 2 receipt")
    cases = _read_cases(loaded.registry)
    if not cases:
        loaded.context.write(
            "known_case_results",
            [
                {
                    "record_type": "summary",
                    "status": "SKIPPED",
                    "reason_code": "KNOWN_CASES_UNAVAILABLE",
                    "case_count": 0,
                    "soft_veto": False,
                }
            ],
            {},
        )
        print("P15 status=SKIPPED reason=KNOWN_CASES_UNAVAILABLE")
        return 0
    predictions: list[pd.DataFrame] = []
    for fold_id in outer_fold_ids(loaded.registry):
        freeze = mapping(
            loaded.context.read("model_freeze_receipt", {OUTER_FOLD: fold_id}),
            "freeze receipt",
        )
        opened = mapping(
            loaded.context.read("outer_open_receipt", {OUTER_FOLD: fold_id}),
            "outer-open receipt",
        )
        loaded.context.read("evaluation_metrics", {OUTER_FOLD: fold_id})
        value = loaded.context.read("raw_outer_predictions", {OUTER_FOLD: fold_id})
        if freeze.get("status") != "PASS":
            raise RuntimeError(f"fold={fold_id}: PASS freeze receipt required")
        if opened.get("status") != "PASS":
            raise RuntimeError(f"fold={fold_id}: PASS outer-open receipt required")
        if freeze.get("protocol_hash") != loaded.protocol_hash:
            raise RuntimeError(f"fold={fold_id}: freeze receipt protocol hash mismatch")
        if opened.get("protocol_hash") != loaded.protocol_hash:
            raise RuntimeError(f"fold={fold_id}: outer-open receipt protocol hash mismatch")
        if isinstance(value, pd.DataFrame):
            predictions.append(value)
    known = mapping(loaded.registry.get("known_cases"), "known_cases")
    veto = mapping(known.get("soft_veto"), "known_cases.soft_veto")
    result = evaluate_known_cases(
        cases=cases,
        predictions=pd.concat(predictions, ignore_index=True),
        expected_case_ids=[
            str(value) for value in sequence(known.get("content_ids"), "known_cases.content_ids")
        ],
        minimum_cases=int(veto["minimum_cases"]),
        below_median_cases=int(veto["below_median_cases"]),
        scenario_fraction_min=float(veto["scenario_fraction_min"]),
        strong_percentile=float(veto["strong_falsification_all_below_percentile"]),
        columns=physical_columns(loaded.registry),
    )
    loaded.context.write("known_case_results", result, {})
    print(f"P15 status=PASS cases={len(cases)}")
    return 0


def _read_cases(registry: dict[str, Any]) -> list[dict[str, Any]]:
    sources = mapping(
        mapping(
            mapping(registry.get("data_sources"), "data_sources").get("source_registry"),
            "source_registry",
        ).get("sources"),
        "source_registry.sources",
    )
    matches = [
        (source_id, mapping(value, f"source={source_id}"))
        for source_id, value in sources.items()
        if mapping(value, f"source={source_id}").get("role") == "known_case"
        and mapping(value, f"source={source_id}").get("enabled") is True
    ]
    if not matches:
        return []
    if len(matches) != 1:
        raise ValueError("exactly one snapshot-locked known-case source is allowed")
    source_id, source = matches[0]
    spec, path = resolve_source(registry, source_id)
    if hash_file(path) != spec.locked_sha256:
        raise ValueError("known-case file does not match its P00 snapshot hash")
    semantics = mapping(source.get("resolved_semantics"), "known-case resolved semantics")
    required = {"case_id", FIRM_ID, FISCAL_YEAR}
    if not required.issubset(semantics):
        raise ValueError("known-case semantics are incomplete")
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    columns = physical_columns(registry)
    rows: list[dict[str, Any]] = []
    for row in iter_rows(path, spec):
        raw_firm = str(row[str(semantics[FIRM_ID])])
        normalized = normalize_entity_field(raw_firm, entity)
        canonical, _ = resolve_entity_link(source_id, raw_firm, normalized, entity)
        rows.append(
            {
                "case_id": str(row[str(semantics["case_id"])]),
                columns[FIRM_ID]: canonical,
                columns[FISCAL_YEAR]: int(str(row[str(semantics[FISCAL_YEAR])])),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
