from __future__ import annotations

import numpy as np
import pytest

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, MATURE, TARGET_VALUE
from evaluation.service import build_latent_risk_scenarios
from labels.latent_class import (
    attach_l3_pilot_posterior,
    finalize_l3_pilot_posteriors,
    fit_fixed_pi_latent_class,
    predict_fixed_pi_latent_probability,
)
from measurement.service import fold_local_l3_target_frame
from selection.service import fit_l3_fold_candidate, select_measurement


def test_fixed_pi_l3_runs_mcmc_and_reports_fail_closed_diagnostics() -> None:
    rng = np.random.default_rng(42)
    latent = rng.random(40) < 0.2
    rows: list[dict[str, object]] = []
    for value in latent:
        anchor = bool(rng.random() < (0.85 if value else 0.03))
        weak = bool(rng.random() < (0.70 if value else 0.12))
        rows.append({"source_outcomes": {"anchor": anchor, "weak": weak}})
    result = fit_fixed_pi_latent_class(
        rows=rows,
        source_channels={"anchor": "official", "weak": "media"},
        accuracy_priors={
            "anchor": {
                "sensitivity_alpha": 8.0,
                "sensitivity_beta": 2.0,
                "specificity_alpha": 20.0,
                "specificity_beta": 1.0,
            },
            "weak": {
                "sensitivity_alpha": 5.0,
                "sensitivity_beta": 3.0,
                "specificity_alpha": 8.0,
                "specificity_beta": 2.0,
            },
        },
        fixed_pi=0.2,
        chains=2,
        warmup=20,
        draws=40,
        alpha_step=0.2,
        random_effect_step=0.15,
        rhat_maximum=2.0,
        ess_minimum=1.0,
        ppc_rate_error_maximum=1.0,
        minimum_observations_per_source=10,
        rng=np.random.default_rng(123),
    )
    assert len(result.posterior_mean) == 40
    assert len(result.posterior_draws) == 80
    assert len(result.parameter_draws) == 80
    assert set(result.source_accuracy) == {"anchor", "weak"}
    assert set(result.channel_random_effect_sd) == {"official", "media"}
    assert result.diagnostics["channel_random_effect_sampled"] is True
    assert all(0 <= value <= 1 for value in result.posterior_mean)


def test_l3_refuses_unbound_source_priors() -> None:
    with pytest.raises(ValueError, match="bind every source"):
        fit_fixed_pi_latent_class(
            rows=[{"source_outcomes": {"a": True, "b": False}}],
            source_channels={"a": "x", "b": "y"},
            accuracy_priors={
                "a": {
                    "sensitivity_alpha": 2.0,
                    "sensitivity_beta": 1.0,
                    "specificity_alpha": 2.0,
                    "specificity_beta": 1.0,
                }
            },
            fixed_pi=0.1,
            chains=2,
            warmup=2,
            draws=2,
            alpha_step=0.1,
            random_effect_step=0.1,
            rhat_maximum=1.1,
            ess_minimum=1.0,
            ppc_rate_error_maximum=1.0,
            minimum_observations_per_source=1,
            rng=np.random.default_rng(1),
        )


def _mcmc_controls() -> dict[str, object]:
    return {
        "chains": 2,
        "warmup_per_chain": 5,
        "draws_per_chain": 10,
        "alpha_proposal_sd": 0.15,
        "random_effect_proposal_sd": 0.15,
        "rhat_max": 100.0,
        "ess_min": 0.0,
        "posterior_predictive_source_rate_error_max": 1.0,
        "minimum_observations_per_source": 1,
    }


def _prior() -> dict[str, float]:
    return {
        "sensitivity_alpha": 8.0,
        "sensitivity_beta": 2.0,
        "specificity_alpha": 12.0,
        "specificity_beta": 1.0,
    }


def test_p05_writes_row_level_l3_pilot_posteriors() -> None:
    rows = [
        {
            FIRM_ID: f"F{index}",
            FISCAL_YEAR: 2019,
            MATURE: True,
            "source_outcomes": {"a": index % 3 == 0, "b": index % 4 == 0},
            "observed_channel_count": 2,
        }
        for index in range(12)
    ]
    fit = fit_fixed_pi_latent_class(
        rows=rows,
        source_channels={"a": "official", "b": "media"},
        accuracy_priors={"a": _prior(), "b": _prior()},
        fixed_pi=0.2,
        chains=2,
        warmup=5,
        draws=10,
        alpha_step=0.15,
        random_effect_step=0.15,
        rhat_maximum=100.0,
        ess_minimum=0.0,
        ppc_rate_error_maximum=1.0,
        minimum_observations_per_source=1,
        rng=np.random.default_rng(11),
    )
    attach_l3_pilot_posterior(
        rows=rows,
        fixed_pi=0.2,
        posterior_mean=fit.posterior_mean,
    )
    finalize_l3_pilot_posteriors(rows=rows, eligible_fixed_pi=[0.2])
    assert all("l3_posterior_by_fixed_pi" in row for row in rows)
    assert all(row["l3_posterior_mean"] is not None for row in rows)
    assert all(row["l3_posterior_scope"] == "p05_feasibility_pilot_only" for row in rows)


def test_l3_fold_refit_drives_selection_and_p11_target_without_outer_rows() -> None:
    rng = np.random.default_rng(22)
    source_channels = {"a": "official", "b": "media", "c": "exchange"}
    rows: list[dict[str, object]] = []
    for index in range(30):
        latent = index % 5 == 0
        outcomes = {
            source: bool(latent if rng.random() > 0.1 else not latent) for source in source_channels
        }
        rows.append(
            {
                FIRM_ID: f"F{index}",
                FISCAL_YEAR: 2018 + index % 3,
                MATURE: True,
                "source_outcomes": outcomes,
                "channel_outcomes": {
                    channel: outcomes[source] for source, channel in source_channels.items()
                },
                "observed_channel_count": 3,
            }
        )
    rows.append(
        {
            FIRM_ID: "OUTER",
            FISCAL_YEAR: 2021,
            MATURE: True,
            "source_outcomes": {source: True for source in source_channels},
            "channel_outcomes": {channel: True for channel in source_channels.values()},
            "observed_channel_count": 3,
        }
    )
    matrices = {
        "expected_channels": sorted(source_channels.values()),
        "l2_scoring": {"status": "EMPIRICALLY_PENDING"},
        "rows": rows,
    }
    fold_result = fit_l3_fold_candidate(
        matrices=matrices,
        outer_year=2021,
        source_channels=source_channels,
        accuracy_priors={source: _prior() for source in source_channels},
        fixed_pi_grid=[0.2],
        mcmc=_mcmc_controls(),
        minimum_observed_channels=1,
        robust_fraction=1.0,
        rng=np.random.default_rng(44),
    )
    assert fold_result["status"] == "PASS"
    assert fold_result["selection_objective"] is not None
    assert fold_result["fit_max_year"] == 2020
    assert all(row[FISCAL_YEAR] < 2021 for row in fold_result["target_rows"])
    selection = select_measurement(
        matrices=matrices,
        outer_year=2021,
        candidates=["L2", "L3_fixed_pi"],
        l3_capability={"status": "AVAILABLE", "pilot_executed": True},
        l3_fold_result=fold_result,
        minimum_observed_channels=1,
    )
    assert selection.selection["selected_measurement"] == "L3_fixed_pi"
    target = fold_local_l3_target_frame(
        channel_selection=selection.channel_selection,
        outer_year=2021,
        columns={
            FIRM_ID: "firm_key",
            FISCAL_YEAR: "year_key",
            TARGET_VALUE: "soft_target",
        },
    )
    assert len(target) == 30
    assert target["year_key"].max() == 2020
    assert target["soft_target"].between(0, 1).all()


def test_p11_rejects_p05_pilot_posterior_as_track_b_target() -> None:
    with pytest.raises(ValueError, match="development-history-only"):
        fold_local_l3_target_frame(
            channel_selection={
                "l3_fit_scope": "p05_feasibility_pilot_only",
                "outer_outcomes_accessed": False,
                "l3_selected_fixed_pi": 0.2,
                "l3_target_rows": [{FIRM_ID: "F1", FISCAL_YEAR: 2020, TARGET_VALUE: 0.5}],
            },
            outer_year=2021,
            columns={
                FIRM_ID: "firm_key",
                FISCAL_YEAR: "year_key",
                TARGET_VALUE: "soft_target",
            },
        )


def test_p12_latent_utility_risk_uses_frozen_l3_measurement_scenario() -> None:
    source_accuracy = {
        "a": {"sensitivity_mean": 0.9, "specificity_mean": 0.95},
        "b": {"sensitivity_mean": 0.8, "specificity_mean": 0.9},
    }
    probabilities = predict_fixed_pi_latent_probability(
        rows=[
            {"source_outcomes": {"a": True, "b": True}},
            {"source_outcomes": {"a": False, "b": False}},
        ],
        source_channels={"a": "official", "b": "media"},
        source_accuracy=source_accuracy,
        channel_random_effect_sd={"official": 0.2, "media": 0.2},
        fixed_pi=0.2,
    )
    assert probabilities[0] > probabilities[1]
    risks = build_latent_risk_scenarios(
        matrices={
            "expected_sources": {"a": "official", "b": "media"},
            "rows": [
                {
                    FIRM_ID: "OUTER",
                    FISCAL_YEAR: 2021,
                    MATURE: True,
                    "source_outcomes": {"a": True, "b": False},
                }
            ],
        },
        channel_selection={
            "l3_scenario_results": [
                {
                    "fixed_pi": 0.2,
                    "eligible": True,
                    "source_accuracy": source_accuracy,
                    "channel_random_effect_sd": {"official": 0.2, "media": 0.2},
                }
            ]
        },
        outer_year=2021,
        scenarios=[{"scenario_id": "theta-1", "measurement_fixed_pi": 0.2}],
    )
    assert 0 < risks["theta-1"][0]["OUTER"] < 1
