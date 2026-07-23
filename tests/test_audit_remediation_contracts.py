from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from core.registry_compiler import compile_registry
from features.service import _validate_definition
from modeling.service import _feature_groups

ROOT = Path(__file__).resolve().parents[1]


def _registry() -> dict[str, object]:
    return cast(dict[str, object], compile_registry(ROOT / "config" / "pipeline.yaml").registry)


def test_observability_features_enter_reference_and_full_groups() -> None:
    registry = _registry()
    features = cast(dict[str, Any], registry["features"])
    definitions = cast(list[dict[str, Any]], features["registry"])
    groups = _feature_groups(definitions)
    assert groups["observability_only"]
    assert set(groups["observability_only"]).issubset(groups["full"])
    assert set(groups["full"]) != set(groups["content_only"])


def test_model_eligibility_is_a_closed_enum() -> None:
    registry = _registry()
    features = cast(dict[str, Any], registry["features"])
    definition = dict(cast(list[dict[str, Any]], features["registry"])[0])
    definition["model_eligibility"] = "silent_unknown_value"
    with pytest.raises(ValueError, match="invalid model_eligibility"):
        _validate_definition(definition)


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
    store = cast(dict[str, Any], features["store"])
    assert cast(dict[str, Any], study["sample_fiscal_years"])["end"] == 2025
    assert store["allowed_fiscal_year_max"] == 2025
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
