"""Tests for P11 pilot-safe feature roster preparation."""

from __future__ import annotations

import pandas as pd

from modeling.pilot_contracts import prepare_model_feature_contract


def _feature(
    feature_id: str,
    *,
    computation_status: str | None = "PASS",
    target_component: bool = False,
) -> dict[str, object]:
    return {
        "feature_id": feature_id,
        "research_decision_status": "LOCKED",
        "model_eligibility": "eligible",
        "computation_status": computation_status,
        "target_component_flag": target_component,
    }


def test_unusable_features_are_excluded_without_mutating_upstream_registry() -> None:
    panel = pd.DataFrame(
        {
            "year": [2019, 2020, 2021],
            "good": [1.0, 2.0, 3.0],
            "zero_coverage": [pd.NA, pd.NA, 4.0],
            "target_component": [1.0, 2.0, 3.0],
            "unsupported": [1.0, 2.0, 3.0],
        }
    )
    registry = [
        _feature("good"),
        _feature("missing_column"),
        _feature("zero_coverage"),
        _feature("target_component", target_component=True),
        _feature("unsupported", computation_status="UNSUPPORTED"),
    ]

    result = prepare_model_feature_contract(
        panel=panel,
        registry=registry,
        year_column="year",
        outer_year=2021,
    )

    by_id = {str(item["feature_id"]): item for item in result.registry}
    assert by_id["good"]["model_eligibility"] == "eligible"
    for feature_id in (
        "missing_column",
        "zero_coverage",
        "target_component",
        "unsupported",
    ):
        assert by_id[feature_id]["model_eligibility"] == "blocked_until_mapping_review"
    assert registry[1]["model_eligibility"] == "eligible"
    assert result.summary["effective_feature_count"] == 1
    assert result.summary["runtime_exclusion_count"] == 4
    assert {str(item["reason_code"]) for item in result.exclusions} == {
        "MODEL_FEATURE_COLUMN_MISSING",
        "MODEL_FEATURE_ZERO_DEVELOPMENT_COVERAGE",
        "DIRECT_TARGET_COMPONENT_FORBIDDEN",
        "MODEL_FEATURE_COMPUTATION_NOT_PASS",
    }


def test_outer_only_values_do_not_make_a_feature_trainable() -> None:
    panel = pd.DataFrame(
        {
            "year": [2020, 2021],
            "good": [1.0, 2.0],
            "outer_only": [pd.NA, 1.0],
        }
    )
    result = prepare_model_feature_contract(
        panel=panel,
        registry=[_feature("good"), _feature("outer_only")],
        year_column="year",
        outer_year=2021,
    )
    assert result.exclusions[0]["reason_code"] == "MODEL_FEATURE_ZERO_DEVELOPMENT_COVERAGE"
