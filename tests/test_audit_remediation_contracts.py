from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core.registry_compiler import compile_registry
from features.service import validate_feature_definition
from modeling.service import feature_groups

ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, object]:
    return cast(dict[str, object], compile_registry(ROOT / "config" / "pipeline.yaml").registry)


def test_observability_features_enter_reference_and_full_groups() -> None:
    registry = _registry()
    features = cast(dict[str, Any], registry["features"])
    definitions = cast(list[dict[str, Any]], features["registry"])
    groups = feature_groups(definitions)
    assert groups["observability_only"]
    assert set(groups["observability_only"]).issubset(groups["full"])
    assert set(groups["full"]) != set(groups["content_only"])


def test_model_eligibility_is_a_closed_enum() -> None:
    registry = _registry()
    features = cast(dict[str, Any], registry["features"])
    definition = dict(cast(list[dict[str, Any]], features["registry"])[0])
    definition["model_eligibility"] = "silent_unknown_value"
    with pytest.raises(ValueError, match="invalid model_eligibility"):
        validate_feature_definition(definition)


def test_every_production_feature_passes_validation() -> None:
    # P07 validates every registry (and intended-registry) definition before it
    # builds the panel. A single invalid feature aborts the whole stage, so the
    # real locked config -- not a curated fixture -- must satisfy the contract.
    registry = _registry()
    features = cast(dict[str, Any], registry["features"])
    definitions = [
        *cast(list[dict[str, Any]], features.get("registry", [])),
        *cast(list[dict[str, Any]], features.get("intended_registry", [])),
    ]
    for definition in definitions:
        validate_feature_definition(dict(definition))


def test_target_component_features_are_retained_but_never_fit() -> None:
    # Methodology 3.5: target components stay in the store as ABLATION_ONLY and
    # must never reach a model matrix.
    registry = _registry()
    features = cast(dict[str, Any], registry["features"])
    definitions = cast(list[dict[str, Any]], features["registry"])
    target_components = {
        str(item["feature_id"])
        for item in definitions
        if item.get("model_eligibility") == "ablation_only_by_target"
    }
    assert target_components, "expected at least one retained target-component feature"
    groups = feature_groups(definitions)
    fit_features = (
        set(groups["full"]) | set(groups["content_only"]) | set(groups["observability_only"])
    )
    assert target_components.isdisjoint(fit_features)


def test_l3_unlocked_parameters_are_explicitly_nonreportable() -> None:
    registry = _registry()
    measurement = cast(dict[str, Any], registry["measurement"])
    operational = cast(dict[str, Any], cast(dict[str, Any], measurement["l3_model"])["operational"])
    assert operational["fixed_pi_grid"] == []
    assert operational["accuracy_priors_by_profile"] == {}
    assert operational["parameter_status"] == "PENDING_EXTERNAL_ELICITATION"
    assert operational["report_required"] is False


def test_temporal_and_provenance_contracts_are_explicit() -> None:
    registry = _registry()
    study = cast(dict[str, Any], registry["study"])
    folds = cast(dict[str, Any], registry["folds"])
    features = cast(dict[str, Any], registry["features"])
    feature_panel = cast(dict[str, Any], features["feature_panel"])
    assert cast(dict[str, Any], study["sample_fiscal_years"])["end"] == 2025
    assert feature_panel["allowed_fiscal_year_max"] == 2025
    assert folds["prospective_year"] == 2026
    assert folds["prospective_feature_status"] == "unavailable_by_data_cutoff"
    assert (
        cast(dict[str, Any], study["prediction_time"])["observed_publication_date_available"]
        is False
    )
    provenance = cast(
        dict[str, Any],
        cast(dict[str, Any], cast(dict[str, Any], registry["data_sources"])["source_registry"])[
            "provenance_contract"
        ],
    )
    assert provenance["required"] is True
    assert "revision_policy" in provenance["required_fields"]


def test_stale_backup_source_is_absent() -> None:
    assert not (ROOT / "data" / "source" / "firm_event_sanction_panel_backup.csv").exists()
