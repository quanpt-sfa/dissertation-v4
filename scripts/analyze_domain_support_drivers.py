"""Diagnose which predictors drive P13 HOSE/HNX common-support loss.

The command is post-hoc and read-only with respect to an immutable pipeline run.
It verifies that its reproduced baseline support equals the stored P13 result,
then writes diagnostic CSV/JSON/Markdown files to an external output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

# Each diagnostic fit is intentionally single-threaded internally. Parallelism is
# across leave-one-feature-out fits controlled by --workers.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from core.pipeline import load_run, mapping, physical_columns, sequence
from core.rng import derive_seed
from core.semantic_keys import FIRM_ID, FISCAL_YEAR, OUTER_FOLD
from modeling.service import feature_groups
from sensitivity.domain_support_diagnostics import (
    DomainScoreResult,
    domain_score_diagnostics,
    feature_driver_row,
)

_DIAGNOSTIC_VERSION = "1"
_METADATA_FIELDS = (
    "role",
    "content_observability_role",
    "construct",
    "description",
    "label",
    "family",
    "model_eligibility",
)


def _default_workers() -> int:
    return min(8, max(1, os.cpu_count() or 1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=1e-10,
        help="Maximum allowed absolute difference from stored P13 common support.",
    )
    parser.add_argument(
        "--only-failing-scenarios",
        action="store_true",
        help="Run leave-one-feature-out attribution only where baseline support fails.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.baseline_tolerance < 0:
        parser.error("--baseline-tolerance must be nonnegative")

    loaded = load_run(
        registry_path=args.registry,
        run_id=args.run_id,
        step_id="P13",
        state="EVALUATED",
    )
    store = loaded.context.store
    panel_raw = store.read("feature_panel", {})
    registry_raw = store.read("feature_registry", {})
    domain_raw = store.read("domain_transfer_outputs", {})
    if not isinstance(panel_raw, pd.DataFrame):
        raise ValueError("feature_panel must be a DataFrame")
    panel = panel_raw
    feature_registry = [
        mapping(item, "feature registry item")
        for item in sequence(registry_raw, "feature registry")
    ]
    domain = mapping(domain_raw, "domain transfer outputs")

    columns = physical_columns(loaded.registry)
    firm_column = columns[FIRM_ID]
    year_column = columns[FISCAL_YEAR]
    full_features = feature_groups(feature_registry).get("full", [])
    if not full_features:
        raise RuntimeError("DOMAIN_SUPPORT_DIAGNOSTIC_FULL_FEATURES_UNAVAILABLE")

    evaluation = mapping(loaded.registry.get("evaluation"), "evaluation")
    domain_config = mapping(evaluation.get("domain_transfer"), "evaluation.domain_transfer")
    bounds_raw = sequence(
        domain_config.get("support_score_bounds"),
        "evaluation.domain_transfer.support_score_bounds",
    )
    if len(bounds_raw) != 2:
        raise ValueError("domain support diagnostics require exactly two support bounds")
    lower, upper = float(bounds_raw[0]), float(bounds_raw[1])
    support_minimum = float(domain_config["support_fraction_minimum"])
    crossfit_folds = int(domain_config["support_crossfit_folds"])

    scenarios = _stored_scenarios(domain)
    metadata_by_feature = _feature_metadata(feature_registry)
    scenario_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    print(
        f"domain-support diagnostics scenarios={len(scenarios)} features={len(full_features)} "
        f"workers={min(args.workers, len(full_features))}",
        flush=True,
    )

    for scenario_index, scenario in enumerate(scenarios, start=1):
        domain_id = str(scenario["domain_id"])
        domain_column = str(scenario["domain_column"])
        level = str(scenario["level"])
        outer_fold = str(scenario[OUTER_FOLD])
        outer_year = int(outer_fold)
        if domain_column not in panel.columns:
            raise ValueError(f"domain column is absent from feature panel: {domain_column}")

        development = panel.loc[
            (panel[year_column].astype(int) < outer_year)
            & panel[domain_column].notna()
            & ~panel[domain_column].astype("string").eq(level)
        ].copy()
        held_test = panel.loc[
            (panel[year_column].astype(int) == outer_year)
            & panel[domain_column].astype("string").eq(level)
        ].copy()
        seed = derive_seed(
            loaded.protocol_hash,
            "P13",
            {OUTER_FOLD: outer_fold},
            f"domain_transfer:{domain_id}:{level}",
        ) % (2**32 - 1)
        baseline = domain_score_diagnostics(
            development=development,
            held_test=held_test,
            feature_ids=list(full_features),
            firm_column=firm_column,
            lower=lower,
            upper=upper,
            crossfit_folds=crossfit_folds,
            random_state=seed,
        )
        _require_matching_baseline(
            baseline=baseline,
            stored_support=float(scenario["stored_common_support_fraction"]),
            tolerance=float(args.baseline_tolerance),
            scenario=scenario,
        )
        if baseline.support_fraction is None or baseline.domain_auc is None:
            raise RuntimeError(
                f"domain support baseline is not estimable for {domain_id}/{level}/{outer_fold}: "
                f"{baseline.reason_code}"
            )

        below_lower = float(np.mean(baseline.held_scores < lower))
        above_upper = float(np.mean(baseline.held_scores > upper))
        failed = baseline.support_fraction < support_minimum
        scenario_rows.append(
            {
                "domain_id": domain_id,
                "domain_column": domain_column,
                "held_level": level,
                OUTER_FOLD: outer_fold,
                "development_rows_other_domain": int(len(development)),
                "held_test_rows": int(len(held_test)),
                "baseline_support_fraction": baseline.support_fraction,
                "stored_common_support_fraction": scenario["stored_common_support_fraction"],
                "support_fraction_minimum": support_minimum,
                "support_gap_to_minimum": float(baseline.support_fraction - support_minimum),
                "support_pass": not failed,
                "baseline_domain_auc": baseline.domain_auc,
                "share_held_below_lower_bound": below_lower,
                "share_held_above_upper_bound": above_upper,
                "held_domain_score_mean": float(np.mean(baseline.held_scores)),
                "held_domain_score_min": float(np.min(baseline.held_scores)),
                "held_domain_score_max": float(np.max(baseline.held_scores)),
                "crossfit_folds_used": baseline.crossfit_folds_used,
            }
        )
        print(
            f"[{scenario_index}/{len(scenarios)}] held={level} fold={outer_fold} "
            f"support={baseline.support_fraction:.6f} auc={baseline.domain_auc:.6f} "
            f"status={'FAIL' if failed else 'PASS'}",
            flush=True,
        )
        if args.only_failing_scenarios and not failed:
            continue

        worker = partial(
            feature_driver_row,
            development=development,
            held_test=held_test,
            all_feature_ids=list(full_features),
            baseline=baseline,
            firm_column=firm_column,
            lower=lower,
            upper=upper,
            crossfit_folds=crossfit_folds,
            random_state=seed,
            support_fraction_minimum=support_minimum,
        )
        with ThreadPoolExecutor(max_workers=min(args.workers, len(full_features))) as executor:
            rows = list(executor.map(worker, full_features))
        for row in rows:
            feature_id = str(row["feature_id"])
            feature_rows.append(
                {
                    "domain_id": domain_id,
                    "domain_column": domain_column,
                    "held_level": level,
                    OUTER_FOLD: outer_fold,
                    "baseline_support_pass": not failed,
                    **metadata_by_feature.get(feature_id, {}),
                    **row,
                }
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_frame = pd.DataFrame(scenario_rows).sort_values(
        ["domain_id", "held_level", OUTER_FOLD], ignore_index=True
    )
    feature_frame = pd.DataFrame(feature_rows)
    _write_csv(scenario_frame, output_dir / "domain_support_scenarios.csv")
    if not feature_frame.empty:
        feature_frame = feature_frame.sort_values(
            ["domain_id", "held_level", OUTER_FOLD, "support_gain_when_removed"],
            ascending=[True, True, True, False],
            na_position="last",
            ignore_index=True,
        )
        _write_csv(feature_frame, output_dir / "domain_support_feature_diagnostics.csv")

    aggregate = _aggregate_drivers(feature_frame, failing_only=True)
    _write_csv(aggregate, output_dir / "domain_support_feature_drivers_failing.csv")
    aggregate_all = _aggregate_drivers(feature_frame, failing_only=False)
    _write_csv(aggregate_all, output_dir / "domain_support_feature_drivers_all.csv")

    report = _markdown_report(
        run_id=args.run_id,
        protocol_hash=loaded.protocol_hash,
        scenario_frame=scenario_frame,
        aggregate=aggregate,
        support_minimum=support_minimum,
        lower=lower,
        upper=upper,
    )
    (output_dir / "DOMAIN_SUPPORT_DRIVER_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "diagnostic_version": _DIAGNOSTIC_VERSION,
        "run_id": args.run_id,
        "protocol_hash": loaded.protocol_hash,
        "source_run_root": str(loaded.run_root),
        "output_dir": str(output_dir),
        "support_bounds": [lower, upper],
        "support_fraction_minimum": support_minimum,
        "crossfit_folds": crossfit_folds,
        "full_feature_count": len(full_features),
        "scenario_count": len(scenario_frame),
        "feature_scenario_rows": len(feature_frame),
        "only_failing_scenarios": bool(args.only_failing_scenarios),
        "baseline_tolerance": float(args.baseline_tolerance),
        "interpretation_boundary": (
            "Diagnostic attribution only. Leave-one-feature-out support gains and domain-separation "
            "metrics are not causal effects and do not modify P13/P14 verdicts."
        ),
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"diagnostics written to {output_dir}", flush=True)
    return 0


def _stored_scenarios(domain: dict[str, Any]) -> list[dict[str, object]]:
    raw = sequence(domain.get("domains"), "domain_transfer_outputs.domains")
    grouped: dict[tuple[str, str, str, str], set[float]] = {}
    for raw_row in raw:
        row = mapping(raw_row, "domain transfer row")
        key = (
            str(row.get("domain_id", "")),
            str(row.get("domain_column", "")),
            str(row.get("level", "")),
            str(row.get(OUTER_FOLD, "")),
        )
        if any(not value for value in key):
            raise ValueError("domain transfer row has incomplete scenario coordinates")
        value = row.get("common_support_fraction")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"stored common support is unavailable for scenario={key}")
        grouped.setdefault(key, set()).add(float(value))

    scenarios: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        if len(values) != 1:
            raise ValueError(f"candidate rows disagree on common support for scenario={key}")
        domain_id, domain_column, level, outer_fold = key
        scenarios.append(
            {
                "domain_id": domain_id,
                "domain_column": domain_column,
                "level": level,
                OUTER_FOLD: outer_fold,
                "stored_common_support_fraction": next(iter(values)),
            }
        )
    if not scenarios:
        raise RuntimeError("DOMAIN_SUPPORT_DIAGNOSTIC_SCENARIOS_UNAVAILABLE")
    return scenarios


def _require_matching_baseline(
    *,
    baseline: DomainScoreResult,
    stored_support: float,
    tolerance: float,
    scenario: dict[str, object],
) -> None:
    reproduced = baseline.support_fraction
    if reproduced is None:
        raise RuntimeError(
            f"cannot reproduce stored P13 support for scenario={scenario}: {baseline.reason_code}"
        )
    difference = abs(float(reproduced) - float(stored_support))
    if difference > tolerance:
        raise RuntimeError(
            "DOMAIN_SUPPORT_BASELINE_MISMATCH: diagnostic code/environment does not reproduce "
            f"stored P13 support for scenario={scenario}; stored={stored_support:.12g}, "
            f"reproduced={reproduced:.12g}, abs_diff={difference:.3g}, tolerance={tolerance:.3g}"
        )


def _feature_metadata(feature_registry: list[dict[str, Any]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in feature_registry:
        feature_id = item.get("feature_id")
        if not isinstance(feature_id, str) or not feature_id:
            continue
        metadata: dict[str, object] = {"feature_id": feature_id}
        for field in _METADATA_FIELDS:
            value = item.get(field)
            if value is not None:
                metadata[field] = value
        result[feature_id] = metadata
    return result


def _aggregate_drivers(feature_frame: pd.DataFrame, *, failing_only: bool) -> pd.DataFrame:
    columns = [
        "feature_id",
        *_METADATA_FIELDS,
        "scenario_count",
        "mean_support_gain_when_removed",
        "median_support_gain_when_removed",
        "maximum_support_gain_when_removed",
        "scenarios_restoring_locked_support",
        "mean_domain_auc_drop_when_removed",
        "mean_absolute_standardized_mean_difference",
        "mean_univariate_domain_separation_auc",
        "mean_absolute_missing_rate_gap",
        "mean_unsupported_minus_supported_abs_robust_z",
    ]
    if feature_frame.empty:
        return pd.DataFrame(columns=columns)
    frame = feature_frame.copy()
    if failing_only:
        frame = frame.loc[~frame["baseline_support_pass"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    frame["absolute_missing_rate_gap"] = pd.to_numeric(
        frame["held_minus_development_missing_rate"], errors="coerce"
    ).abs()
    group_fields = [field for field in ("feature_id", *_METADATA_FIELDS) if field in frame.columns]
    aggregate = (
        frame.groupby(group_fields, dropna=False)
        .agg(
            scenario_count=("feature_id", "size"),
            mean_support_gain_when_removed=("support_gain_when_removed", "mean"),
            median_support_gain_when_removed=("support_gain_when_removed", "median"),
            maximum_support_gain_when_removed=("support_gain_when_removed", "max"),
            scenarios_restoring_locked_support=(
                "restores_locked_support_when_removed",
                "sum",
            ),
            mean_domain_auc_drop_when_removed=("domain_auc_drop_when_removed", "mean"),
            mean_absolute_standardized_mean_difference=(
                "absolute_standardized_mean_difference",
                "mean",
            ),
            mean_univariate_domain_separation_auc=(
                "univariate_domain_separation_auc",
                "mean",
            ),
            mean_absolute_missing_rate_gap=("absolute_missing_rate_gap", "mean"),
            mean_unsupported_minus_supported_abs_robust_z=(
                "unsupported_minus_supported_mean_abs_robust_z",
                "mean",
            ),
        )
        .reset_index()
    )
    return aggregate.sort_values(
        [
            "mean_support_gain_when_removed",
            "mean_domain_auc_drop_when_removed",
            "mean_absolute_standardized_mean_difference",
        ],
        ascending=False,
        na_position="last",
        ignore_index=True,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    fieldnames = [str(value) for value in frame.columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in frame.to_dict(orient="records"):
            writer.writerow({str(key): _csv_value(value) for key, value in record.items()})


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _markdown_report(
    *,
    run_id: str,
    protocol_hash: str,
    scenario_frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    support_minimum: float,
    lower: float,
    upper: float,
) -> str:
    lines = [
        "# Domain common-support driver diagnostic",
        "",
        f"- Run: `{run_id}`",
        f"- Protocol hash: `{protocol_hash}`",
        f"- Locked support-score interval: `[{lower:.3f}, {upper:.3f}]`",
        f"- Locked minimum held-domain support fraction: `{support_minimum:.3f}`",
        "- Diagnostic status: post-hoc, read-only, does not modify Gate 2.",
        "",
        "## Scenario reproduction",
        "",
        "| Held domain | Outer fold | Support | Domain AUC | Below lower | Above upper | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in scenario_frame.to_dict(orient="records"):
        status = "PASS" if bool(row["support_pass"]) else "FAIL"
        lines.append(
            f"| {row['held_level']} | {row[OUTER_FOLD]} | "
            f"{float(row['baseline_support_fraction']):.3f} | "
            f"{float(row['baseline_domain_auc']):.3f} | "
            f"{float(row['share_held_below_lower_bound']):.3f} | "
            f"{float(row['share_held_above_upper_bound']):.3f} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Leading feature drivers in failed scenarios",
            "",
            "Ranking is primarily by the mean increase in held-domain common support after removing "
            "one feature while holding the P13 cross-fitting design and seed fixed. A positive value "
            "means removing the feature makes the domains overlap more. Correlated predictors can "
            "share or mask attribution, so these are diagnostic contributions rather than causal effects.",
            "",
            "| Rank | Feature | Mean support gain if removed | Mean domain-AUC drop | Mean |SMD| | Restores support |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(aggregate.head(25).to_dict(orient="records"), start=1):
        feature = str(row.get("feature_id", ""))
        lines.append(
            f"| {index} | `{feature}` | {_format_float(row.get('mean_support_gain_when_removed'))} | "
            f"{_format_float(row.get('mean_domain_auc_drop_when_removed'))} | "
            f"{_format_float(row.get('mean_absolute_standardized_mean_difference'))} | "
            f"{int(row.get('scenarios_restoring_locked_support', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This diagnostic identifies variables associated with the HOSE/HNX feature-space separation "
            "that drives the P13 common-support calculation. It does not establish that a variable "
            "causes exchange membership, does not use outcome labels, and does not reopen or alter the "
            "sealed Gate 2 verdict.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_float(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)):
        return f"{float(value):.4f}"
    return "NA"


if __name__ == "__main__":
    raise SystemExit(main())
