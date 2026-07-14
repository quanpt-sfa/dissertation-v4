"""Synthetic measurement simulation with deterministic, batch-local randomness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from core.metrics import average_precision, roc_auc
from core.semantic_keys import (
    ESTIMATE,
    MCSE,
    METHOD_ID,
    METRIC_ID,
    REPLICATION_ID,
    SCENARIO_ID,
)
from labels.service import aggregate_l1, evidence_score_l2, posterior_l3_fixed_pi

_REQUIRED = {
    SCENARIO_ID,
    "sample_size",
    "prevalence",
    "anchor_sensitivity",
    "anchor_false_positive",
    "weak_sensitivity",
    "weak_false_positive",
    "content_signal",
}


def validate_scenarios(raw: Sequence[object]) -> list[dict[str, Any]]:
    """Validate only explicitly registered operational scenarios."""
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise ValueError(f"simulation scenario {index}: mapping required")
        scenario = cast(dict[str, Any], value)
        missing = sorted(_REQUIRED - set(scenario))
        if missing:
            raise ValueError(f"simulation scenario {index}: missing {missing}")
        scenario_id = scenario[SCENARIO_ID]
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen:
            raise ValueError("simulation scenario_id must be unique and nonempty")
        seen.add(scenario_id)
        if not isinstance(scenario["sample_size"], int) or scenario["sample_size"] < 2:
            raise ValueError(f"scenario={scenario_id}: sample_size must be >= 2")
        for key in (
            "prevalence",
            "anchor_sensitivity",
            "anchor_false_positive",
            "weak_sensitivity",
            "weak_false_positive",
        ):
            number = scenario[key]
            if not isinstance(number, (int, float)) or not 0 <= float(number) <= 1:
                raise ValueError(f"scenario={scenario_id}: {key} must be in [0, 1]")
        if not 0 < float(scenario["prevalence"]) < 1:
            raise ValueError(f"scenario={scenario_id}: prevalence must be in (0, 1)")
        fixed_pi = scenario.get("fixed_pi", scenario["prevalence"])
        if not isinstance(fixed_pi, (int, float)) or not 0 < float(fixed_pi) < 1:
            raise ValueError(f"scenario={scenario_id}: fixed_pi must be in (0, 1)")
        if not isinstance(scenario["content_signal"], (int, float)):
            raise ValueError(f"scenario={scenario_id}: content_signal must be numeric")
        for key in (
            "anchor_verification_probability",
            "weak_verification_probability",
            "selective_verification_strength",
            "channel_dependence",
        ):
            number = scenario.get(key, 0.0 if key != "channel_dependence" else 0.0)
            if not isinstance(number, (int, float)) or not 0 <= float(number) <= 1:
                raise ValueError(f"scenario={scenario_id}: {key} must be in [0, 1]")
        for key in ("horizon_days", "detection_delay_mean_days"):
            number = scenario.get(key, 365 if key == "horizon_days" else 30.0)
            if not isinstance(number, (int, float)) or float(number) <= 0:
                raise ValueError(f"scenario={scenario_id}: {key} must be positive")
        scenarios.append(dict(scenario))
    return scenarios


def run_batch(
    scenario: Mapping[str, Any],
    *,
    method_id: str,
    replications: range,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run a simulation batch; L1 and L2 are imported from production code."""
    rows: list[dict[str, object]] = []
    scenario_id = str(scenario[SCENARIO_ID])
    n = int(scenario["sample_size"])
    for replication_id in replications:
        latent = rng.random(n) < float(scenario["prevalence"])
        shared_uniform = rng.random(n)
        dependence = float(scenario.get("channel_dependence", 0.0))
        anchor = _source_draw(
            latent,
            sensitivity=float(scenario["anchor_sensitivity"]),
            false_positive=float(scenario["anchor_false_positive"]),
            shared_uniform=shared_uniform,
            dependence=dependence,
            rng=rng,
        )
        weak = _source_draw(
            latent,
            sensitivity=float(scenario["weak_sensitivity"]),
            false_positive=float(scenario["weak_false_positive"]),
            shared_uniform=shared_uniform,
            dependence=dependence,
            rng=rng,
        )
        selective = float(scenario.get("selective_verification_strength", 0.0))
        anchor_observed = _verification_draw(
            latent,
            base_probability=float(scenario.get("anchor_verification_probability", 1.0)),
            selective_strength=selective,
            rng=rng,
        )
        weak_observed = _verification_draw(
            latent,
            base_probability=float(scenario.get("weak_verification_probability", 1.0)),
            selective_strength=selective,
            rng=rng,
        )
        delay = rng.exponential(float(scenario.get("detection_delay_mean_days", 30.0)), n)
        mature = delay <= float(scenario.get("horizon_days", 365.0))
        anchor_values = [
            bool(value) if observed and is_mature else None
            for value, observed, is_mature in zip(anchor, anchor_observed, mature, strict=True)
        ]
        weak_values = [
            bool(value) if observed and is_mature else None
            for value, observed, is_mature in zip(weak, weak_observed, mature, strict=True)
        ]
        source_rows = [
            {"anchor": anchor_value, "weak": weak_value}
            for anchor_value, weak_value in zip(anchor_values, weak_values, strict=True)
        ]
        l1_values = [aggregate_l1(value) for value in source_rows]
        l2_values = [evidence_score_l2(value) for value in source_rows]
        fixed_pi = float(scenario.get("fixed_pi", scenario["prevalence"]))
        accuracy = {
            "anchor": (
                float(scenario["anchor_sensitivity"]),
                1.0 - float(scenario["anchor_false_positive"]),
            ),
            "weak": (
                float(scenario["weak_sensitivity"]),
                1.0 - float(scenario["weak_false_positive"]),
            ),
        }
        l3 = np.asarray(
            [posterior_l3_fixed_pi(value, accuracy, fixed_pi) for value in source_rows],
            dtype=float,
        )
        l2 = np.asarray(
            [fixed_pi if value is None else float(value) for value in l2_values], dtype=float
        )
        content = rng.normal(float(scenario["content_signal"]) * latent, 1.0, n)
        standardized_content = (content - content.mean()) / max(content.std(), 1e-12)
        method_scores = {
            "observability_only": l2,
            "content_only": 1.0 / (1.0 + np.exp(-standardized_content)),
            "full": 1.0 / (1.0 + np.exp(-(standardized_content + l2))),
            "anchor_pu": np.asarray(
                [fixed_pi if value is None else float(value) for value in anchor_values]
            ),
        }
        if method_id not in method_scores:
            raise ValueError(f"method={method_id}: unsupported simulation method")
        l1_observed = np.asarray([value is not None for value in l1_values], dtype=bool)
        l1_numeric = np.asarray(
            [False if value is None else bool(value) for value in l1_values], dtype=bool
        )
        l2_observed = np.asarray([value is not None for value in l2_values], dtype=bool)
        estimates = {
            "l1_coverage": float(l1_observed.mean()),
            "l1_latent_agreement": float(
                np.asarray(l1_numeric[l1_observed] == latent[l1_observed], dtype=float).mean()
            )
            if l1_observed.any()
            else 0.0,
            "l2_coverage": float(l2_observed.mean()),
            "l2_latent_mae": float(
                np.asarray(np.abs(l2[l2_observed] - latent[l2_observed].astype(float))).mean()
            )
            if l2_observed.any()
            else 0.0,
            "l3_fixed_latent_mae": float(np.abs(l3 - latent.astype(float)).mean()),
            "verification_rate": float(
                np.asarray(anchor_observed | weak_observed, dtype=float).mean()
            ),
            "maturity_rate": float(np.asarray(mature, dtype=float).mean()),
            "realized_prevalence": float(np.asarray(latent, dtype=float).mean()),
        }
        l3_ap = average_precision(latent.tolist(), l3.tolist())
        l3_auc = roc_auc(latent.tolist(), l3.tolist())
        if l3_ap is not None:
            estimates["l3_fixed_latent_average_precision"] = l3_ap
        if l3_auc is not None:
            estimates["l3_fixed_latent_auc"] = l3_auc
        content_ap = average_precision(latent.tolist(), content.tolist())
        if content_ap is not None:
            estimates["content_average_precision"] = content_ap
        method_ap = average_precision(latent.tolist(), method_scores[method_id].tolist())
        if method_ap is not None:
            estimates["method_latent_average_precision"] = method_ap
        method_auc = roc_auc(latent.tolist(), method_scores[method_id].tolist())
        if method_auc is not None:
            estimates["method_latent_auc"] = method_auc
        for metric_id, estimate in estimates.items():
            rows.append(
                {
                    SCENARIO_ID: scenario_id,
                    METHOD_ID: method_id,
                    REPLICATION_ID: replication_id,
                    METRIC_ID: metric_id,
                    ESTIMATE: float(estimate),
                    MCSE: None,
                }
            )
    return pd.DataFrame(rows).astype(
        {
            SCENARIO_ID: "string",
            METHOD_ID: "string",
            REPLICATION_ID: "int64",
            METRIC_ID: "string",
            ESTIMATE: "float64",
            MCSE: "float64",
        }
    )


def summarize_mcse(
    batches: Sequence[pd.DataFrame],
    *,
    minimum_replications: int,
    maximum_replications: int,
    pass_fail_mcse_maximum: float,
) -> dict[str, object]:
    """Compute actual Monte Carlo standard errors over completed replications."""
    if not batches:
        return {"status": "SKIPPED", "reason_code": "NO_SIMULATION_BATCHES", "metrics": []}
    if (
        minimum_replications < 1
        or maximum_replications < minimum_replications
        or pass_fail_mcse_maximum <= 0
    ):
        raise ValueError("simulation replication controls are invalid")
    combined = pd.concat(batches, ignore_index=True)
    metrics: list[dict[str, object]] = []
    grouped = combined.groupby([SCENARIO_ID, METHOD_ID, METRIC_ID], sort=True)
    for (scenario_id, method_id, metric_id), frame in grouped:
        count = int(len(frame))
        standard_deviation = float(frame[ESTIMATE].std(ddof=1)) if count > 1 else 0.0
        metrics.append(
            {
                SCENARIO_ID: str(scenario_id),
                METHOD_ID: str(method_id),
                METRIC_ID: str(metric_id),
                "replications": count,
                "mean": float(frame[ESTIMATE].mean()),
                MCSE: standard_deviation / math.sqrt(count),
                "minimum_replications_met": count >= minimum_replications,
                "mcse_target_met": standard_deviation / math.sqrt(count) <= pass_fail_mcse_maximum,
                "maximum_replications_reached": count >= maximum_replications,
            }
        )
    precision_met = bool(metrics) and all(
        item["minimum_replications_met"] is True and item["mcse_target_met"] is True
        for item in metrics
    )
    maximum_reached = bool(metrics) and all(
        item["maximum_replications_reached"] is True for item in metrics
    )
    status = "PASS" if precision_met else "MAXIMUM_REACHED" if maximum_reached else "CONTINUE"
    return {
        "status": status,
        "reason_code": None
        if precision_met
        else "MCSE_TARGET_NOT_MET_AT_CAP"
        if maximum_reached
        else "ADDITIONAL_REPLICATIONS_REQUIRED",
        "minimum_replications": minimum_replications,
        "maximum_replications": maximum_replications,
        "pass_fail_mcse_maximum": pass_fail_mcse_maximum,
        "precision_target_met": precision_met,
        "metrics": metrics,
    }


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
    draws = np.where(use_shared, shared_uniform, independent)
    return draws < probability


def _verification_draw(
    latent: np.ndarray,
    *,
    base_probability: float,
    selective_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = np.clip(base_probability + selective_strength * latent.astype(float), 0.0, 1.0)
    return rng.random(len(latent)) < probability
