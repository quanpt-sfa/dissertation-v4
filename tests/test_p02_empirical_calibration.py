from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "simulation" / "empirical_calibration.py"


def _load_module():
    core = types.ModuleType("core")
    semantic = types.ModuleType("core.semantic_keys")
    semantic.ELIGIBLE = "eligible"
    semantic.FIRM_ID = "firm_id"
    semantic.FISCAL_YEAR = "fiscal_year"
    semantic.SCENARIO_ID = "scenario_id"
    sys.modules.setdefault("core", core)
    sys.modules["core.semantic_keys"] = semantic
    spec = importlib.util.spec_from_file_location("p08_empirical_calibration", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(n: int = 1000):
    firms = [f"F{i:04d}" for i in range(n)]
    years = np.repeat(2019, n)
    p02 = pd.DataFrame({"firm_id": firms, "fiscal_year": years})
    features = pd.DataFrame(
        {
            "firm_id": firms,
            "fiscal_year": years,
            "f1": np.linspace(-2.0, 2.0, n),
            "f2": np.sin(np.linspace(0.0, 6.0, n)),
        }
    )
    rows = []
    for i, firm in enumerate(firms):
        rows.append(
            {
                "firm_id": firm,
                "fiscal_year": 2019,
                "eligible": True,
                "target_maturity": {"L1_CONTENT_STRICT": True},
                "candidate_outcomes": {"L1_CONTENT_STRICT": i < 10},
            }
        )
    matrices = {"rows": rows}
    scenario = {
        "scenario_id": "empirical_rare_label",
        "tier": "semi_synthetic_development_covariates",
        "sample_design": "p02_development_panel",
        "calibration_target_id": "L1_CONTENT_STRICT",
        "semi_synthetic_feature_ids": ["f1", "f2"],
        "anchor_sensitivity": 0.85,
        "anchor_false_positive": 0.0,
        "weak_sensitivity": 0.60,
        "weak_false_positive": 0.0,
        "content_signal": 1.0,
        "anchor_verification_probability": 0.70,
        "weak_verification_probability": 0.45,
        "selective_verification_strength": 0.10,
        "channel_dependence": 0.10,
        "horizon_days": 365,
        "detection_delay_mean_days": 120,
        "shift_strength": 0.0,
        "signal_structure": "linear",
    }
    return p02, features, matrices, scenario


def test_sample_size_and_prevalence_are_empirically_derived() -> None:
    module = _load_module()
    p02, features, matrices, scenario = _inputs()
    locked = module.attach_empirical_panel_calibration(
        scenarios=[scenario],
        firm_year_panel=p02,
        feature_panel=features,
        feature_registry=[{"feature_id": "f1"}, {"feature_id": "f2"}],
        source_channel_matrices=matrices,
        firm_column="firm_id",
        year_column="fiscal_year",
        development_year_maximum=2020,
        calibration_repeats=16,
    )[0]
    assert locked["sample_size"] == len(p02)
    assert locked["sample_size_source"] == "P02_firm_year_panel_development_rows"
    assert locked["empirical_positive_count"] == 10
    assert locked["empirical_observed_positive_rate"] == pytest.approx(0.01)
    assert 0.0 < locked["prevalence"] < 0.05
    assert locked["prevalence"] == locked["calibrated_latent_prevalence"]
    assert locked["semi_synthetic_pool_rows"] == len(p02)
    assert locked["empirical_population_mode"] == "exact_P02_rows_no_bootstrap"


def test_empirical_scenario_requires_explicit_target() -> None:
    module = _load_module()
    p02, features, matrices, scenario = _inputs()
    scenario.pop("calibration_target_id")
    with pytest.raises(ValueError, match="calibration_target_id"):
        module.attach_empirical_panel_calibration(
            scenarios=[scenario],
            firm_year_panel=p02,
            feature_panel=features,
            feature_registry=[{"feature_id": "f1"}, {"feature_id": "f2"}],
            source_channel_matrices=matrices,
            firm_column="firm_id",
            year_column="fiscal_year",
            development_year_maximum=2020,
            calibration_repeats=8,
        )
