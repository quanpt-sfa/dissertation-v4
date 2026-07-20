from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from core.pipeline import mapping, sequence
from core.registry_compiler import compile_registry
from features.generator import (
    _atomic_definition,
    _classify_definitions,
    _dependency_parts,
    _dependency_tokens,
    _evaluate_registered_formula,
    _materialize_observability,
    _prepost_pairs,
)


def _registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(Path("config/pipeline.yaml")).registry,
    )


def test_all_locked_feature_definitions_fit_the_production_generation_grammar() -> None:
    registry = _registry()
    features = mapping(registry.get("features"), "features")
    definitions = [
        mapping(value, "features.registry item")
        for value in sequence(features.get("registry"), "features.registry")
        if mapping(value, "features.registry item").get("research_decision_status")
        == "LOCKED"
    ]
    atomic, derived, observability = _classify_definitions(definitions)

    assert len(definitions) == 142
    assert len(atomic) == 100
    assert len(derived) == 36
    assert len(observability) == 6

    atomic_specs = [_atomic_definition(definition) for definition in atomic]
    assert len({definition.feature_id for definition in atomic_specs}) == len(atomic_specs)

    panel = pd.DataFrame(
        {
            "firm_key": pd.Series(["A", "A", "A"], dtype="string"),
            "year_key": pd.Series([2020, 2021, 2023], dtype="int16"),
        }
    )
    feature_series: dict[str, pd.Series] = {
        definition.feature_id: pd.Series([1.0, 2.0, 3.0], dtype="float64")
        for definition in atomic_specs
    }

    pending = list(derived)
    while pending:
        progressed = False
        remaining: list[dict[str, object]] = []
        for definition in pending:
            dependencies = _dependency_tokens(definition)
            missing = {
                _dependency_parts(token)[0]
                for token in dependencies
                if _dependency_parts(token)[0] not in feature_series
            }
            if missing:
                remaining.append(definition)
                continue
            feature_id = str(definition["feature_id"])
            feature_series[feature_id] = _evaluate_registered_formula(
                definition=definition,
                dependencies=dependencies,
                feature_series=feature_series,
                panel=panel,
                firm_column="firm_key",
                year_column="year_key",
            )
            assert len(feature_series[feature_id]) == len(panel)
            progressed = True
        assert progressed, {
            str(definition.get("feature_id")): [
                _dependency_parts(token)[0]
                for token in _dependency_tokens(definition)
                if _dependency_parts(token)[0] not in feature_series
            ]
            for definition in remaining
        }
        pending = remaining

    pairs = _prepost_pairs(derived)
    assert len(pairs) == 10
    _materialize_observability(
        definitions=observability,
        atomic_definitions=atomic,
        pair_dependencies=pairs,
        feature_series=feature_series,
    )
    assert set(feature_series) == {str(definition["feature_id"]) for definition in definitions}
