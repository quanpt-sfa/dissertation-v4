# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from evidence.service import EvidenceRecord, _validate_measurement_process
from identification.service import (
    build_identification_summary,
    minimax_regret,
    model_selection_set,
    paired_difference_dominance,
    prevalence_bounds,
)
from observability.service import build_observability_registry
from simulation.v24_service import _generate_replication_v24


def test_prevalence_bounds_are_uninformative_without_restrictions() -> None:
    result = prevalence_bounds(0.08)
    assert result.lower == 0.0
    assert result.upper == 1.0
    assert result.domain_type == "sensitivity_region"
    assert result.status == "UNINFORMATIVE_WITHOUT_RESTRICTIONS"


def test_prevalence_bounds_require_independent_justification_for_identified_set() -> None:
    sensitivity = prevalence_bounds(
        0.10,
        false_positive_upper=0.02,
        sensitivity_lower=0.50,
        restrictions_independently_justified=False,
    )
    identified = prevalence_bounds(
        0.10,
        false_positive_upper=0.02,
        sensitivity_lower=0.50,
        restrictions_independently_justified=True,
    )
    assert sensitivity.domain_type == "sensitivity_region"
    assert identified.domain_type == "identified_set"
    assert identified.lower == pytest.approx((0.10 - 0.02) / 0.98)
    assert identified.upper == pytest.approx(0.20)


def test_robust_model_selection_uses_paired_differences() -> None:
    surface = {
        "model_a": {"theta_1": 0.70, "theta_2": 0.75},
        "model_b": {"theta_1": 0.65, "theta_2": 0.70},
        "model_c": {"theta_1": 0.72, "theta_2": 0.68},
    }
    dominance = paired_difference_dominance(surface)
    a_over_b = next(
        row for row in dominance if row["model_a"] == "model_a" and row["model_b"] == "model_b"
    )
    assert a_over_b["dominates"] is True
    assert model_selection_set(surface) == ["model_a", "model_c"]
    regret = minimax_regret(surface)
    assert regret["status"] == "AVAILABLE"
    assert regret["minimax_regret_set"] == ["model_a"]


def test_identification_summary_never_promotes_an_unjustified_anchor() -> None:
    summary = build_identification_summary(
        {
            "identification": {
                "high_specificity_anchor_assumed": False,
                "independent_justification_confirmed": True,
                "restrictions": {
                    "enforcement_false_positive_upper": 0.03,
                    "enforcement_sensitivity_lower": 0.50,
                },
            }
        },
        observed_positive_rate=0.08,
    )
    assert summary["status"] == "SENSITIVITY_ONLY"
    assert summary["high_specificity_anchor_assumed"] is False
    assert summary["legacy_global_justification_flag_ignored"] is True
    assert summary["independent_justification_confirmed"] is False
    assert summary["scenario_fraction_can_replace_identified_set_dominance"] is False


def test_identification_summary_requires_per_restriction_provenance() -> None:
    source = {
        "citation_key": "external_bound",
        "independent_of_study_outcomes": True,
        "provides_numeric_bound_for_vietnam": True,
    }
    restriction = {
        "restriction_type": "numeric_upper_bound",
        "analytical_role": "identified_set",
        "source_ids": ["external_bound"],
        "transportability_status": "established_for_target_population",
        "approval_status": "approved_for_identified_set",
        "eligible_for_identified_set": True,
        "blocking_reasons": [],
    }
    sensitivity_restriction = dict(restriction)
    sensitivity_restriction["restriction_type"] = "numeric_lower_bound"
    summary = build_identification_summary(
        {
            "identification": {
                "evidence_basis_registry": {"external_bound": source},
                "restriction_registry": {
                    "enforcement_false_positive_upper": restriction,
                    "enforcement_sensitivity_lower": sensitivity_restriction,
                },
                "restrictions": {
                    "enforcement_false_positive_upper": 0.02,
                    "enforcement_sensitivity_lower": 0.50,
                },
            }
        },
        observed_positive_rate=0.10,
    )
    assert summary["status"] == "IDENTIFIED_SET_AVAILABLE"
    assert summary["uncertainty_domain_type"] == "identified_set"
    assert summary["independent_justification_confirmed"] is True


def test_registered_project_literature_does_not_create_vietnam_bounds() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load((root / "config" / "methodology" / "measurement.yaml").read_text())
    measurement = cast(dict[str, Any], raw["measurement"])
    summary = build_identification_summary(measurement, observed_positive_rate=0.08)
    assert summary["status"] == "SENSITIVITY_ONLY"
    assert summary["prevalence_bounds"] == {
        "lower": 0.0,
        "upper": 1.0,
        "uncertainty_domain_type": "sensitivity_region",
        "restrictions_used": [],
        "status": "UNINFORMATIVE_WITHOUT_RESTRICTIONS",
    }
    assert set(summary["identification_basis_source_ids"]) >= {
        "kedia_rajgopal_2011",
        "kubic_2021",
        "dyck_morse_zingales_2024",
        "hahn_murray_manolopoulou_2016",
    }


def test_legacy_high_confirmation_metadata_is_not_observed_verification() -> None:
    registry = build_observability_registry(
        {
            "rows": [
                {
                    "eligible": True,
                    "mature": True,
                    "source_maturity": {"S3_CONTENT": True},
                    "source_outcomes": {"S3_CONTENT": True},
                    "source_opportunities": {"S3_CONTENT": True},
                    "channel_outcomes": {"S3": True},
                    "channel_opportunities": {"S3": True},
                }
            ]
        },
        {
            "S3_CONTENT": {
                "channel_id": "S3",
                "verification_status": "high_confirmation",
                "evidence_mapping": {"opportunity_semantic": "derived_source_year_completeness"},
            }
        },
    )
    sources = cast(dict[str, dict[str, Any]], registry["sources"])
    source = sources["S3_CONTENT"]
    terminology = cast(dict[str, Any], registry["terminology_contract"])
    assert source["verification_classification"] == "unknown"
    assert source["legacy_high_confirmation_is_not_verification"] is True
    assert terminology["high_confirmation_is_observed_verification"] is False


def test_measurement_process_rejects_o_zero_with_observed_s() -> None:
    record = EvidenceRecord(
        source_id="S3_CONTENT",
        channel_id="S3",
        firm_id="F1",
        fiscal_year=2020,
        availability_date=None,
        outcome=True,
        event_id=None,
        event_cluster_id=None,
        source_opportunity=False,
    )
    with pytest.raises(ValueError, match="O=0"):
        _validate_measurement_process(record)


def test_simulation_generates_distinct_ovdr_layers() -> None:
    scenario = {
        "sample_size": 200,
        "prevalence": 0.08,
        "fixed_pi": 0.08,
        "tier": "fully_synthetic",
        "anchor_sensitivity": 0.85,
        "anchor_false_positive": 0.03,
        "weak_sensitivity": 0.60,
        "weak_false_positive": 0.10,
        "content_signal": 1.0,
        "anchor_opportunity_probability": 0.90,
        "weak_opportunity_probability": 0.80,
        "anchor_verification_probability": 0.60,
        "weak_verification_probability": 0.40,
        "selective_verification_strength": 0.50,
        "anchor_recording_probability": 0.75,
        "weak_recording_probability": 0.65,
        "channel_dependence": 0.20,
        "horizon_days": 365,
        "detection_delay_mean_days": 120,
        "signal_structure": "linear",
        "simulation_feature_count": 12,
    }
    generated = _generate_replication_v24(scenario, np.random.default_rng(20260801))
    process = cast(dict[str, dict[str, np.ndarray]], generated["measurement_process"])
    for source in ("anchor", "weak"):
        layers = process[source]
        assert set(layers) == {"O", "V", "D", "R"}
        assert len(layers["O"]) == 200
        assert np.all(~layers["V"] | layers["O"])
        assert np.all(~layers["R"] | layers["V"])
    verification_rate = generated["verification_rate"]
    recording_rate = generated["recording_rate"]
    assert isinstance(verification_rate, (int, float))
    assert isinstance(recording_rate, (int, float))
    assert 0.0 <= float(verification_rate) <= 1.0
    assert 0.0 <= float(recording_rate) <= 1.0
