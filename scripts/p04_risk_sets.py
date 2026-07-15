"""P04 CLI: build the primary-horizon risk set and maturity audit."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from risksets.service import build_risk_set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P04",
        state="LEDGERED",
    )
    risksets = mapping(loaded.registry.get("risksets"), "risksets")
    cutoff_raw = risksets.get("data_cutoff")
    if not isinstance(cutoff_raw, str):
        if args.dry_run:
            print("P04 dry-run: blocked until risksets.data_cutoff is registered")
            return 0
        raise ValueError("MISSING_DATA_CUTOFF: set risksets.data_cutoff in pipeline config")
    cutoff = datetime.fromisoformat(cutoff_raw.replace("Z", "+00:00")).replace(tzinfo=None)
    study = mapping(loaded.registry.get("study"), "study")
    horizons = mapping(study.get("horizons_months"), "study.horizons_months")
    horizon = horizons.get("primary")
    if not isinstance(horizon, int):
        raise ValueError("study.horizons_months.primary must be an integer")
    if args.dry_run:
        print(f"P04 dry-run: cutoff={cutoff.isoformat()} horizon_months={horizon}")
        return 0
    panel = loaded.context.read("firm_year_panel", {})
    evidence = loaded.context.read("evidence_ledger", {})
    if not isinstance(panel, pd.DataFrame) or not isinstance(evidence, pd.DataFrame):
        raise ValueError("P04 panel and evidence inputs must be DataFrames")
    result = build_risk_set(
        panel=panel,
        data_cutoff=cutoff,
        horizon_months=horizon,
        columns=physical_columns(loaded.registry),
        evidence=evidence,
        sensitivity_horizons_months=[
            int(value)
            for value in sequence(horizons.get("sensitivity"), "study horizon sensitivities")
        ],
        sanction_complete_through_year=int(
            mapping(
                mapping(loaded.registry.get("s3_taxonomy"), "s3_taxonomy").get(
                    "sanction_source_completeness"
                ),
                "s3_taxonomy.sanction_source_completeness",
            )["complete_through_year"]
        ),
        sanction_incomplete_years={
            int(value)
            for value in sequence(
                mapping(
                    mapping(loaded.registry.get("s3_taxonomy"), "s3_taxonomy").get(
                        "sanction_source_completeness"
                    ),
                    "s3_taxonomy.sanction_source_completeness",
                ).get("incomplete_years"),
                "s3_taxonomy.sanction_source_completeness.incomplete_years",
            )
        },
        primary_target_id=(
            str(value)
            if (
                value := mapping(loaded.registry.get("measurement"), "measurement").get(
                    "primary_target_id"
                )
            )
            is not None
            else None
        ),
    )
    if args.validate_only:
        return 0
    loaded.context.write("risk_sets", result.risk_sets, {})
    loaded.context.write("maturity_audit", result.maturity_audit, {})
    loaded.context.write("prospective_set", result.prospective_set, {})
    loaded.context.write("censoring_registry", result.censoring_registry, {})
    print(
        f"P04 status=PASS mature={result.maturity_audit['mature_count']} "
        f"prospective={result.maturity_audit['prospective_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
