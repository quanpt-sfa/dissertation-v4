"""Build development-only coverage, lag, and prevalence reports for L2/L3 calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import yaml

from core.evidence_registry import logical_evidence_sources
from core.pipeline import load_run, mapping, physical_columns, sequence
from core.semantic_keys import (
    AVAILABILITY_DATE,
    CHANNEL_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    PREDICTION_TIME,
    SOURCE_ID,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P06",
        state="MEASURED",
    )
    risk_sets = loaded.context.read("risk_sets", {})
    evidence = loaded.context.read("evidence_ledger", {})
    matrices = mapping(
        loaded.context.read("source_channel_matrices", {}),
        "source_channel_matrices",
    )
    if not isinstance(risk_sets, pd.DataFrame) or not isinstance(evidence, pd.DataFrame):
        raise TypeError("P04 risk sets and P03 evidence ledger must be DataFrames")

    columns = physical_columns(loaded.registry)
    initial_outer_year = int(mapping(loaded.registry.get("folds"), "folds")["initial_outer_year"])
    sources = logical_evidence_sources(loaded.registry)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows = [
        mapping(item, "source-channel matrix row")
        for item in sequence(matrices.get("rows"), "source-channel matrix rows")
    ]
    development_rows = [
        row
        for row in matrix_rows
        if int(row[FISCAL_YEAR]) < initial_outer_year
        and row.get(ELIGIBLE) is True
        and row.get(MATURE) is True
    ]

    source_coverage = _source_coverage(development_rows, sources)
    channel_coverage = _channel_coverage(development_rows, matrices)
    lag_distribution = _lag_distribution(
        risk_sets=risk_sets,
        evidence=evidence,
        columns=columns,
        initial_outer_year=initial_outer_year,
        source_profiles={source_id: source.profile_id for source_id, source in sources.items()},
    )
    empirical_anchors = _empirical_anchors(source_coverage, channel_coverage)
    prior_worksheet = _prior_worksheet(source_coverage, sources)

    source_coverage.to_csv(output_dir / "source_coverage_development.csv", index=False)
    channel_coverage.to_csv(output_dir / "channel_coverage_development.csv", index=False)
    lag_distribution.to_csv(output_dir / "source_lag_distribution_development.csv", index=False)
    (output_dir / "prevalence_anchors_development.json").write_text(
        json.dumps(empirical_anchors, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "l3_prior_worksheet.yaml").write_text(
        yaml.safe_dump(prior_worksheet, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = {
        "run_id": args.run_id,
        "protocol_hash": loaded.protocol_hash,
        "fit_scope": "mature_eligible_development_history_only",
        "initial_outer_year": initial_outer_year,
        "development_row_count": len(development_rows),
        "logical_source_count": len(sources),
        "channel_count": len(sequence(matrices.get("expected_channels"), "expected channels")),
        "l2_status": mapping(matrices.get("l2_scoring"), "l2_scoring").get("status"),
        "l3_lock_status": "REQUIRES_RESEARCHER_REVIEW",
        "outer_outcomes_accessed": False,
    }
    print("MEASUREMENT_CALIBRATION_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


def _source_coverage(
    rows: list[dict[str, Any]],
    sources: dict[str, Any],
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for source_id, source in sorted(sources.items()):
        outcomes = [
            cast(dict[str, object], row.get("source_outcomes", {})).get(source_id)
            for row in rows
        ]
        opportunities = [
            cast(dict[str, object], row.get("source_opportunities", {})).get(source_id)
            for row in rows
        ]
        observed = [value for value in outcomes if value is not None]
        positives = sum(value is True for value in observed)
        output.append(
            {
                SOURCE_ID: source_id,
                "profile_id": source.profile_id,
                CHANNEL_ID: source.channel_id,
                "development_rows": len(rows),
                "observed_outcomes": len(observed),
                "observed_opportunities": sum(value is True for value in opportunities),
                "positive": positives,
                "explicit_negative": sum(value is False for value in observed),
                "unknown": sum(value is None for value in outcomes),
                "coverage_fraction": len(observed) / len(rows) if rows else None,
                "observed_positive_rate": positives / len(observed) if observed else None,
            }
        )
    return pd.DataFrame(output)


def _channel_coverage(rows: list[dict[str, Any]], matrices: dict[str, Any]) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    channels = [str(value) for value in sequence(matrices.get("expected_channels"), "channels")]
    for channel_id in sorted(channels):
        outcomes = [
            cast(dict[str, object], row.get("channel_outcomes", {})).get(channel_id)
            for row in rows
        ]
        opportunities = [
            cast(dict[str, object], row.get("channel_opportunities", {})).get(channel_id)
            for row in rows
        ]
        observed = [value for value in outcomes if value is not None]
        positives = sum(value is True for value in observed)
        output.append(
            {
                CHANNEL_ID: channel_id,
                "development_rows": len(rows),
                "observed_outcomes": len(observed),
                "observed_opportunities": sum(value is True for value in opportunities),
                "positive": positives,
                "explicit_negative": sum(value is False for value in observed),
                "unknown": sum(value is None for value in outcomes),
                "coverage_fraction": len(observed) / len(rows) if rows else None,
                "observed_positive_rate": positives / len(observed) if observed else None,
            }
        )
    return pd.DataFrame(output)


def _lag_distribution(
    *,
    risk_sets: pd.DataFrame,
    evidence: pd.DataFrame,
    columns: dict[str, str],
    initial_outer_year: int,
    source_profiles: dict[str, str],
) -> pd.DataFrame:
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    prediction = columns[PREDICTION_TIME]
    mature = columns[MATURE]
    eligible = columns[ELIGIBLE]
    source = columns[SOURCE_ID]
    availability = columns[AVAILABILITY_DATE]
    outcome = columns[OUTCOME]

    development = risk_sets.loc[
        (risk_sets[year] < initial_outer_year)
        & risk_sets[mature].astype(bool)
        & risk_sets[eligible].astype(bool),
        [firm, year, prediction],
    ].copy()
    merged = evidence.merge(development, on=[firm, year], how="inner", validate="many_to_one")
    merged = merged.loc[merged[availability].notna() & merged[prediction].notna()].copy()
    merged["lag_days"] = (
        pd.to_datetime(merged[availability]) - pd.to_datetime(merged[prediction])
    ).dt.days

    output: list[dict[str, object]] = []
    for source_id, group in merged.groupby(source, sort=True):
        lags = group["lag_days"].dropna().astype(float)
        observed_lags = group.loc[group[outcome].notna(), "lag_days"].dropna().astype(float)
        output.append(
            {
                SOURCE_ID: str(source_id),
                "profile_id": source_profiles.get(str(source_id)),
                "evidence_records": len(group),
                "observed_outcome_records": int(group[outcome].notna().sum()),
                "lag_count": len(lags),
                "lag_mean": float(lags.mean()) if len(lags) else None,
                "lag_p25": float(lags.quantile(0.25)) if len(lags) else None,
                "lag_p50": float(lags.quantile(0.50)) if len(lags) else None,
                "lag_p75": float(lags.quantile(0.75)) if len(lags) else None,
                "lag_p90": float(lags.quantile(0.90)) if len(lags) else None,
                "observed_outcome_lag_p50": (
                    float(observed_lags.quantile(0.50)) if len(observed_lags) else None
                ),
            }
        )
    return pd.DataFrame(output)


def _empirical_anchors(
    source_coverage: pd.DataFrame,
    channel_coverage: pd.DataFrame,
) -> dict[str, object]:
    source_rates = [
        float(value)
        for value in source_coverage["observed_positive_rate"].dropna().tolist()
    ]
    channel_rates = [
        float(value)
        for value in channel_coverage["observed_positive_rate"].dropna().tolist()
    ]
    combined = source_rates + channel_rates
    quantiles = (
        {
            "p10": float(pd.Series(combined).quantile(0.10)),
            "p25": float(pd.Series(combined).quantile(0.25)),
            "p50": float(pd.Series(combined).quantile(0.50)),
            "p75": float(pd.Series(combined).quantile(0.75)),
            "p90": float(pd.Series(combined).quantile(0.90)),
        }
        if combined
        else {}
    )
    return {
        "fit_scope": "mature_eligible_development_history_only",
        "source_observed_positive_rates": source_rates,
        "channel_observed_positive_rates": channel_rates,
        "observed_rate_quantiles": quantiles,
        "interpretation": (
            "Observed positive rates are calibration anchors only; they are not latent prevalence "
            "estimates because sensitivity and specificity are unknown."
        ),
        "fixed_pi_lock_policy": (
            "Lock a broad grid before outer-fold access using external prevalence evidence, "
            "development-only anchors, and explicit sensitivity analysis."
        ),
    }


def _prior_worksheet(
    source_coverage: pd.DataFrame,
    sources: dict[str, Any],
) -> dict[str, object]:
    profiles = sorted({source.profile_id for source in sources.values()})
    rows: list[dict[str, object]] = []
    for profile_id in profiles:
        profile_rows = source_coverage.loc[source_coverage["profile_id"].eq(profile_id)]
        rows.append(
            {
                "profile_id": profile_id,
                "logical_sources": sorted(
                    source_id
                    for source_id, source in sources.items()
                    if source.profile_id == profile_id
                ),
                "development_observed_outcomes": int(profile_rows["observed_outcomes"].sum()),
                "prior_mean_sensitivity": None,
                "prior_mean_specificity": None,
                "prior_effective_sample_size": None,
                "sensitivity_alpha": None,
                "sensitivity_beta": None,
                "specificity_alpha": None,
                "specificity_beta": None,
                "evidence_basis": None,
            }
        )
    return {
        "status": "RESEARCHER_LOCK_REQUIRED",
        "parameterization": {
            "alpha": "prior_mean * prior_effective_sample_size",
            "beta": "(1 - prior_mean) * prior_effective_sample_size",
        },
        "requirements": [
            "Use external literature, validation subsamples, or documented expert elicitation.",
            "Keep the primary prior weak enough to avoid prior-dominated posteriors.",
            "Run weak, skeptical, and evidence-hierarchy prior sensitivity scenarios.",
            "Do not infer sensitivity or specificity directly from observed positive rates.",
        ],
        "profiles": rows,
    }


if __name__ == "__main__":
    raise SystemExit(main())