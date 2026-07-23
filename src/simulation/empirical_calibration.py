"""Empirical P02/P05 calibration for P08 semi-synthetic scenarios.

P02 determines the development population and sample size. P05 determines the
observed positive/known-label rates on that same population. The latent
prevalence is then calibrated to the scenario's detection mechanism so the DGP
reproduces the empirical observed-positive rate. No arbitrary sample size or
positive prevalence is accepted for the empirical semi-synthetic tier.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, SCENARIO_ID


def attach_empirical_panel_calibration(
    *,
    scenarios: Sequence[Mapping[str, Any]],
    firm_year_panel: pd.DataFrame,
    feature_panel: pd.DataFrame,
    feature_registry: Sequence[Mapping[str, Any]],
    source_channel_matrices: Mapping[str, object],
    firm_column: str,
    year_column: str,
    development_year_maximum: int,
    calibration_repeats: int = 64,
) -> list[dict[str, Any]]:
    """Bind P08 scenarios to the exact P02 development population.

    Fully synthetic scenarios pass through unchanged. For the empirical tier,
    every P02 development firm-year must have exactly one P07 feature row. The
    function stores the exact standardized covariate matrix, derives the sample
    size, measures the P05 target rate, estimates the scenario-specific
    ascertainment probabilities P(observed positive | Y*=1/0), and solves for
    latent prevalence.
    """
    if calibration_repeats < 8:
        raise ValueError("empirical calibration requires at least 8 repeats")
    for name, frame in (("firm_year_panel", firm_year_panel), ("feature_panel", feature_panel)):
        if firm_column not in frame.columns or year_column not in frame.columns:
            raise ValueError(f"{name} is missing P02 firm/year bindings")

    p02 = firm_year_panel.loc[
        firm_year_panel[year_column].astype(int) <= development_year_maximum,
        [firm_column, year_column],
    ].copy()
    p02[firm_column] = p02[firm_column].astype("string")
    p02[year_column] = p02[year_column].astype(int)
    if p02.duplicated([firm_column, year_column]).any():
        raise ValueError("P02 development panel must have unique firm-year keys")
    p02 = p02.sort_values([year_column, firm_column], kind="stable").reset_index(drop=True)
    if len(p02) < 50:
        raise ValueError("P02 development population must contain at least 50 firm-years")

    feature_keys = feature_panel[[firm_column, year_column]].copy()
    feature_keys[firm_column] = feature_keys[firm_column].astype("string")
    feature_keys[year_column] = feature_keys[year_column].astype(int)
    if feature_keys.duplicated([firm_column, year_column]).any():
        raise ValueError("P07 feature panel must have unique firm-year keys")

    feature_bindings = {
        str(item.get("feature_id")): str(item.get("physical_column", item.get("feature_id")))
        for item in feature_registry
        if item.get("feature_id") is not None
    }
    registered = set(feature_bindings)
    matrix_rows_raw = source_channel_matrices.get("rows")
    if not isinstance(matrix_rows_raw, list):
        raise ValueError("P05 source_channel_matrices.rows must be a list")
    measurement_rows = _measurement_row_index(cast(list[object], matrix_rows_raw))

    output: list[dict[str, Any]] = []
    for raw_scenario in scenarios:
        scenario = dict(raw_scenario)
        if scenario.get("tier") != "semi_synthetic_development_covariates":
            output.append(scenario)
            continue
        if scenario.get("sample_design", "p02_development_panel") != "p02_development_panel":
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)}: empirical tier requires "
                "sample_design=p02_development_panel"
            )
        target_id = scenario.get("calibration_target_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)}: calibration_target_id must be locked"
            )
        feature_ids_raw = scenario.get("semi_synthetic_feature_ids")
        if not isinstance(feature_ids_raw, list) or not feature_ids_raw:
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)}: semi_synthetic_feature_ids required"
            )
        feature_ids = [str(item) for item in cast(list[object], feature_ids_raw)]
        missing_registry = sorted(set(feature_ids) - registered)
        feature_columns = [feature_bindings.get(item, item) for item in feature_ids]
        missing_columns = sorted(set(feature_columns) - set(feature_panel.columns))
        if missing_registry or missing_columns:
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)}: unavailable empirical features; "
                f"unregistered={missing_registry}, absent={missing_columns}"
            )

        empirical = p02.merge(
            feature_panel[[firm_column, year_column, *feature_columns]],
            on=[firm_column, year_column],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(empirical) != len(p02):
            raise ValueError("P02/P07 empirical population alignment changed row count")
        numeric = empirical[feature_columns].apply(pd.to_numeric, errors="coerce")
        if numeric.dropna(how="all").shape[0] == 0:
            raise ValueError("All P02 development rows have all simulation features missing")
        medians = numeric.median(axis=0)
        if medians.isna().any():
            raise ValueError("at least one empirical simulation feature is entirely missing")
        imputed = numeric.fillna(medians)
        scale = imputed.std(axis=0, ddof=0).replace(0.0, 1.0)
        standardized = (imputed - imputed.mean(axis=0)) / scale
        pool = standardized.to_numpy(dtype=float)

        label_summary = _empirical_label_summary(
            keys=p02,
            measurement_rows=measurement_rows,
            target_id=target_id,
            firm_column=firm_column,
            year_column=year_column,
        )
        q_observed = float(label_summary["observed_positive_rate_all_rows"])
        if q_observed <= 0.0:
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)} target={target_id}: "
                "no observed positives in the P02 development population"
            )

        p1, p0, calibration_mcse = _ascertainment_probabilities(
            scenario=scenario,
            standardized_pool=pool,
            repeats=calibration_repeats,
        )
        if p1 <= p0 + 1e-8:
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)}: detection model has no positive "
                f"information (p1={p1:.6g}, p0={p0:.6g})"
            )
        raw_pi = (q_observed - p0) / (p1 - p0)
        tolerance = max(5.0 * calibration_mcse, 1.0 / len(p02))
        if raw_pi < -tolerance or raw_pi > 1.0 + tolerance:
            raise ValueError(
                f"scenario={scenario.get(SCENARIO_ID)} target={target_id}: empirical "
                f"observed rate={q_observed:.6g} is incompatible with the locked "
                f"detection mechanism p0={p0:.6g}, p1={p1:.6g}"
            )
        calibrated_pi = float(np.clip(raw_pi, 1e-6, 1.0 - 1e-6))

        key_hash = hashlib.sha256(
            "\n".join(
                f"{firm}|{year}"
                for firm, year in p02[[firm_column, year_column]].itertuples(index=False, name=None)
            ).encode("utf-8")
        ).hexdigest()
        scenario.update(
            {
                "sample_size": int(len(p02)),
                "prevalence": calibrated_pi,
                "fixed_pi": calibrated_pi,
                "sample_size_source": "P02_firm_year_panel_development_rows",
                "prevalence_source": "P05_observed_rate_calibrated_through_DGP",
                "empirical_population_mode": "exact_P02_rows_no_bootstrap",
                "semi_synthetic_covariate_pool": pool.tolist(),
                "semi_synthetic_feature_count": int(pool.shape[1]),
                "semi_synthetic_feature_columns": feature_columns,
                "semi_synthetic_pool_rows": int(pool.shape[0]),
                "semi_synthetic_development_year_maximum": int(development_year_maximum),
                "semi_synthetic_preprocessing": "P02_aligned_development_median_zscore",
                "empirical_panel_key_hash": key_hash,
                "empirical_calibration_target_id": target_id,
                "empirical_panel_rows": int(len(p02)),
                "empirical_positive_count": int(label_summary["positive_count"]),
                "empirical_known_label_count": int(label_summary["known_count"]),
                "empirical_mature_target_count": int(label_summary["mature_count"]),
                "empirical_observed_positive_rate": q_observed,
                "empirical_positive_rate_known_labels": float(
                    label_summary["positive_rate_known_labels"]
                ),
                "empirical_known_label_rate": float(label_summary["known_label_rate"]),
                "empirical_target_maturity_rate": float(label_summary["maturity_rate"]),
                "ascertainment_probability_if_latent_positive": p1,
                "ascertainment_probability_if_latent_negative": p0,
                "ascertainment_calibration_mcse": calibration_mcse,
                "calibrated_latent_prevalence": calibrated_pi,
                "outer_rows_used_in_pool": 0,
            }
        )
        output.append(scenario)
    return output


def _measurement_row_index(rows: list[object]) -> dict[tuple[str, int], dict[str, object]]:
    indexed: dict[tuple[str, int], dict[str, object]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("P05 source-channel row must be a mapping")
        row = cast(dict[str, object], raw)
        firm = row.get(FIRM_ID)
        year = row.get(FISCAL_YEAR)
        if firm is None or year is None:
            raise ValueError("P05 source-channel row is missing logical firm/year keys")
        key = (str(firm), int(cast(Any, year)))
        if key in indexed:
            raise ValueError(f"duplicate P05 source-channel key: {key}")
        indexed[key] = row
    return indexed


def _empirical_label_summary(
    *,
    keys: pd.DataFrame,
    measurement_rows: Mapping[tuple[str, int], Mapping[str, object]],
    target_id: str,
    firm_column: str,
    year_column: str,
) -> dict[str, float | int]:
    positive = 0
    known = 0
    mature = 0
    for firm, year in keys[[firm_column, year_column]].itertuples(index=False, name=None):
        row = measurement_rows.get((str(firm), int(year)))
        if row is None:
            continue
        target_maturity = row.get("target_maturity")
        if isinstance(target_maturity, dict) and target_maturity.get(target_id) is True:
            mature += 1
        outcomes = row.get("candidate_outcomes")
        if not isinstance(outcomes, dict) or target_id not in outcomes:
            continue
        value = outcomes[target_id]
        if value is None or pd.isna(value):
            continue
        known += 1
        if bool(value):
            positive += 1
    total = len(keys)
    return {
        "positive_count": positive,
        "known_count": known,
        "mature_count": mature,
        "observed_positive_rate_all_rows": positive / total,
        "positive_rate_known_labels": positive / known if known else 0.0,
        "known_label_rate": known / total,
        "maturity_rate": mature / total,
    }


def _ascertainment_probabilities(
    *,
    scenario: Mapping[str, Any],
    standardized_pool: np.ndarray,
    repeats: int,
) -> tuple[float, float, float]:
    scenario_id = str(scenario.get(SCENARIO_ID, "scenario"))
    seed = int.from_bytes(
        hashlib.sha256(f"P08_CALIBRATION|{scenario_id}".encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    n = len(standardized_pool)
    rates1: list[float] = []
    rates0: list[float] = []
    for _ in range(repeats):
        order = rng.permutation(n)
        base = standardized_pool[order]
        rates1.append(_observed_positive_rate(scenario, base, True, rng))
        rates0.append(_observed_positive_rate(scenario, base, False, rng))
    p1 = float(np.mean(rates1))
    p0 = float(np.mean(rates0))
    mcse = float(
        max(
            np.std(rates1, ddof=1) / np.sqrt(repeats),
            np.std(rates0, ddof=1) / np.sqrt(repeats),
        )
    )
    return p1, p0, mcse


def _observed_positive_rate(
    scenario: Mapping[str, Any],
    base: np.ndarray,
    latent_value: bool,
    rng: np.random.Generator,
) -> float:
    n = len(base)
    latent = np.full(n, latent_value, dtype=bool)
    base_score = base.mean(axis=1) if base.ndim == 2 else base
    strength = float(scenario["content_signal"])
    structure = str(scenario.get("signal_structure", "linear"))
    if structure == "linear":
        effect = strength * latent.astype(float)
    elif structure == "nonlinear":
        effect = strength * latent.astype(float) * np.tanh(base_score)
    elif structure == "interaction":
        effect = (
            strength * latent.astype(float) * (base_score > np.median(base_score)).astype(float)
        )
    else:
        raise ValueError(f"unsupported signal structure: {structure}")
    content = base_score + effect
    standardized = (content - content.mean()) / max(content.std(), 1e-12)
    observable_risk = 1.0 / (1.0 + np.exp(-standardized))
    dependence = float(scenario.get("channel_dependence", 0.0))
    shared = rng.random(n)
    anchor = _source_draw(
        latent,
        sensitivity=float(scenario["anchor_sensitivity"]),
        false_positive=float(scenario["anchor_false_positive"]),
        shared_uniform=shared,
        dependence=dependence,
        rng=rng,
    )
    weak = _source_draw(
        latent,
        sensitivity=float(scenario["weak_sensitivity"]),
        false_positive=float(scenario["weak_false_positive"]),
        shared_uniform=shared,
        dependence=dependence,
        rng=rng,
    )
    selective = float(scenario.get("selective_verification_strength", 0.0))
    anchor_observed = _verification_draw(
        latent,
        observable_risk,
        float(scenario.get("anchor_verification_probability", 1.0)),
        selective,
        rng,
    )
    weak_observed = _verification_draw(
        latent,
        observable_risk,
        float(scenario.get("weak_verification_probability", 1.0)),
        selective,
        rng,
    )
    delay = rng.exponential(float(scenario.get("detection_delay_mean_days", 30.0)), n)
    mature = delay <= float(scenario.get("horizon_days", 365.0))
    observed_positive = mature & ((anchor & anchor_observed) | (weak & weak_observed))
    return float(observed_positive.mean())


def _source_draw(
    latent: np.ndarray,
    *,
    sensitivity: float,
    false_positive: float,
    shared_uniform: np.ndarray,
    dependence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.where(latent, sensitivity, false_positive)
    independent = rng.random(len(latent))
    use_shared = rng.random(len(latent)) < dependence
    return np.where(use_shared, shared_uniform, independent) < probability


def _verification_draw(
    latent: np.ndarray,
    observable_risk: np.ndarray,
    base_probability: float,
    selective_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.clip(
        base_probability
        + 0.25 * selective_strength * observable_risk
        + 0.75 * selective_strength * latent.astype(float),
        0.0,
        1.0,
    )
    return rng.random(len(latent)) < probability
