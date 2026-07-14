from __future__ import annotations

import numpy as np
import pytest

from labels.latent_class import fit_fixed_pi_latent_class


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
