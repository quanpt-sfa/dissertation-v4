"""P15 CLI: open known cases embedded in the final firm-year input."""

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
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P15",
        state="GATE2",
    )
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
            str(value)
            for value in sequence(known.get("content_ids"), "known_cases.content_ids")
        ],
        minimum_cases=int(veto["minimum_cases"]),
        below_median_cases=int(veto["below_median_cases"]),
        scenario_fraction_min=float(veto["scenario_fraction_min"]),
        strong_percentile=float(veto["strong_falsification_all_below_percentile"]),
        weak_percentile=float(veto["weak_percentile"]),
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
        raise ValueError("exactly one snapshot-locked known-case semantic view is allowed")
    source_id, source = matches[0]
    spec, path = resolve_source(registry, source_id)
    if hash_file(path) != spec.locked_sha256:
        raise ValueError("final firm-year file does not match its P00 snapshot hash")
    semantics = mapping(source.get("resolved_semantics"), "known-case resolved semantics")
    required = {
        "case_id",
        FIRM_ID,
        FISCAL_YEAR,
        "case_construct",
        "case_role",
        "external_validation_include_flag",
        "case_seal_status",
        "case_opens_at_step",
    }
    if not required.issubset(semantics):
        raise ValueError(
            "embedded known-case semantics are incomplete: "
            f"{sorted(required - set(semantics))}"
        )
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    columns = physical_columns(registry)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for row in iter_rows(path, spec):
        raw_case_id = row.get(str(semantics["case_id"]))
        if raw_case_id is None or not str(raw_case_id).strip():
            continue
        raw_firm_value = row.get(str(semantics[FIRM_ID]))
        if raw_firm_value is None or not str(raw_firm_value).strip():
            raise ValueError("known-case row requires firm_id")
        raw_firm = str(raw_firm_value)
        normalized = normalize_entity_field(raw_firm, entity)
        canonical, _ = resolve_entity_link(source_id, raw_firm, normalized, entity)
        case_id = str(raw_case_id).strip()
        fiscal_year = _parse_year(row.get(str(semantics[FISCAL_YEAR])))
        case_construct = str(row.get(str(semantics["case_construct"]), "")).strip()
        case_role = str(row.get(str(semantics["case_role"]), "")).strip()
        external_flag = _parse_bool(
            row.get(str(semantics["external_validation_include_flag"])),
            "external_validation_include_flag",
        )
        seal_status = str(row.get(str(semantics["case_seal_status"]), "")).strip()
        opens_at_step = str(row.get(str(semantics["case_opens_at_step"]), "")).strip()
        if case_construct != "CONFIRMED_FINANCIAL_REPORTING_CASE":
            raise ValueError("known case construct must be CONFIRMED_FINANCIAL_REPORTING_CASE")
        if case_role != "SIMULATION_EXTERNAL_VALIDATION":
            raise ValueError("known case role must be SIMULATION_EXTERNAL_VALIDATION")
        if external_flag is not True:
            raise ValueError("known case must be included only for external validation")
        if opens_at_step.upper() != "P15":
            raise ValueError("known case may open only at P15")
        if not _is_sealed_status(seal_status):
            raise ValueError("known case seal status must explicitly indicate sealed or locked")
        key = (case_id, canonical, fiscal_year)
        if key in seen:
            raise ValueError(f"duplicate embedded known-case row={key}")
        seen.add(key)
        rows.append(
            {
                "case_id": case_id,
                columns[FIRM_ID]: canonical,
                columns[FISCAL_YEAR]: fiscal_year,
            }
        )
    return rows


def _parse_year(value: object) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("known-case fiscal_year is required")
    text = str(value).strip()
    if not text:
        raise ValueError("known-case fiscal_year is required")
    numeric = float(text)
    if not numeric.is_integer():
        raise ValueError("known-case fiscal_year must be an integer")
    year = int(numeric)
    if not 1900 <= year <= 2200:
        raise ValueError("known-case fiscal_year is outside the supported range")
    return year


def _parse_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"known case field={field}: boolean value required")


def _is_sealed_status(value: str) -> bool:
    normalized = value.strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized or "unseal" in normalized or normalized.startswith("open"):
        return False
    return "seal" in normalized or "lock" in normalized


if __name__ == "__main__":
    raise SystemExit(main())
