"""P15 CLI: open the snapshot-locked, validation-only known-case registry."""

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

_EXPECTED_CASE_CONSTRUCT = "CONFIRMED_FINANCIAL_REPORTING_CASE"
_EXPECTED_CASE_ROLE = "SIMULATION_EXTERNAL_VALIDATION"
_EXPECTED_INCLUSION_FLAGS = {
    "training_include_flag": False,
    "calibration_include_flag": False,
    "model_selection_include_flag": False,
    "external_validation_include_flag": True,
}


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
    for fold_id in outer_fold_ids(loaded.registry, include_initial=False):
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
        weak_percentile=float(veto["weak_percentile"]),
        columns=physical_columns(loaded.registry),
    )
    loaded.context.write("known_case_results", result, {})
    print(f"P15 status=PASS cases={len(cases)}")
    return 0


def _read_cases(registry: dict[str, Any]) -> list[dict[str, Any]]:
    known_config = mapping(registry.get("known_cases"), "known_cases")
    if known_config.get("opening_step") != "P15":
        raise ValueError("known-case registry must open only at P15")
    if known_config.get("sealed_before_development") is not True:
        raise ValueError("known-case registry must be sealed before development")

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
        raise ValueError("exactly one snapshot-locked known-case registry is allowed")
    source_id, source = matches[0]
    spec, path = resolve_source(registry, source_id)
    if hash_file(path) != spec.locked_sha256:
        raise ValueError("known-case registry does not match its P00 snapshot hash")
    semantics = mapping(source.get("resolved_semantics"), "known-case resolved semantics")
    required = {
        "case_id",
        FIRM_ID,
        FISCAL_YEAR,
        "case_construct",
        "case_role",
        *_EXPECTED_INCLUSION_FLAGS,
    }
    if not required.issubset(semantics):
        raise ValueError(
            f"known-case registry semantics are incomplete: {sorted(required - set(semantics))}"
        )
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    columns = physical_columns(registry)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for source_row, row in enumerate(iter_rows(path, spec), start=1):
        case_id = _required_text(row.get(str(semantics["case_id"])), "case_id", source_row)
        raw_firm = _required_text(row.get(str(semantics[FIRM_ID])), FIRM_ID, source_row)
        normalized = normalize_entity_field(raw_firm, entity)
        canonical, _ = resolve_entity_link(source_id, raw_firm, normalized, entity)
        fiscal_year = _parse_year(row.get(str(semantics[FISCAL_YEAR])))
        case_construct = _required_text(
            row.get(str(semantics["case_construct"])), "case_construct", source_row
        )
        case_role = _required_text(row.get(str(semantics["case_role"])), "case_role", source_row)
        flags = {
            field: _parse_bool(row.get(str(semantics[field])), field)
            for field in _EXPECTED_INCLUSION_FLAGS
        }
        if case_construct != _EXPECTED_CASE_CONSTRUCT:
            raise ValueError(
                f"known case row={source_row}: case_construct must be {_EXPECTED_CASE_CONSTRUCT}"
            )
        if case_role != _EXPECTED_CASE_ROLE:
            raise ValueError(
                f"known case row={source_row}: case_role must be {_EXPECTED_CASE_ROLE}"
            )
        if flags != _EXPECTED_INCLUSION_FLAGS:
            raise ValueError(
                f"known case row={source_row}: inclusion flags violate the sealed "
                "external-validation role"
            )
        key = (case_id, canonical, fiscal_year)
        if key in seen:
            raise ValueError(f"duplicate known-case row={key}")
        seen.add(key)
        rows.append(
            {
                "case_id": case_id,
                columns[FIRM_ID]: canonical,
                columns[FISCAL_YEAR]: fiscal_year,
            }
        )
    return rows


def _required_text(value: object, field: str, source_row: int) -> str:
    if value is None:
        raise ValueError(f"known case row={source_row}: {field} is required")
    text = str(value).strip()
    if not text:
        raise ValueError(f"known case row={source_row}: {field} is required")
    return text


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


if __name__ == "__main__":
    raise SystemExit(main())
