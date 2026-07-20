from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from core.pipeline import mapping, physical_columns
from core.registry_compiler import compile_registry
from core.semantic_keys import (
    AUDIT_STATUS,
    FIRM_ID,
    FISCAL_YEAR,
    ITEM_ID,
    STATEMENT_SCOPE,
    UNIT,
    VALUE,
)
from features.generator import (
    _source_controls,
    feature_source_id,
    materialize_registered_features,
)
from p01.models import SourceSpec
from p02.models import EntityResolutionSpec


def _registry() -> dict[str, Any]:
    return cast(
        dict[str, Any],
        compile_registry(Path("config/pipeline.yaml")).registry,
    )


def _definition(
    feature_id: str,
    *,
    step: str,
    formula: str,
    dependencies: list[str],
    audit_status: str | None = None,
    items: list[str] | None = None,
) -> dict[str, object]:
    lineage: list[dict[str, object]] = []
    if items is not None:
        lineage = [
            {
                "source_item_id": item,
                "transformation_step": step,
                "transformation_order": index,
            }
            for index, item in enumerate(items, start=1)
        ]
    else:
        lineage = [
            {
                "source_column": dependency,
                "transformation_step": step,
                "transformation_order": index,
            }
            for index, dependency in enumerate(dependencies, start=1)
        ]
    result: dict[str, object] = {
        "feature_id": feature_id,
        "physical_column": feature_id,
        "formula": formula,
        "dependencies": dependencies,
        "lineage": lineage,
        "research_decision_status": "LOCKED",
        "confirmatory_status": "confirmatory",
        "model_eligibility": "eligible",
    }
    if audit_status is not None:
        result["audit_status"] = audit_status
    return result


def _definitions() -> list[dict[str, object]]:
    result = [
        _definition(
            "fs_aud_x",
            step="source_selection",
            formula="select item_x",
            dependencies=[],
            audit_status="audited",
            items=["item_x"],
        ),
        _definition(
            "fs_unaud_x",
            step="source_selection",
            formula="select item_x",
            dependencies=[],
            audit_status="unaudited",
            items=["item_x"],
        ),
        _definition(
            "fs_aud_y",
            step="component_sum",
            formula="item_y1 + item_y2",
            dependencies=["raw:item_y1", "raw:item_y2"],
            audit_status="audited",
            items=["item_y1", "item_y2"],
        ),
        _definition(
            "fs_unaud_y",
            step="component_sum",
            formula="item_y1 + item_y2",
            dependencies=["raw:item_y1", "raw:item_y2"],
            audit_status="unaudited",
            items=["item_y1", "item_y2"],
        ),
        _definition(
            "audit_adj_x_signed",
            step="prepost_difference",
            formula="fs_aud_x - fs_unaud_x",
            dependencies=["fs_aud_x", "fs_unaud_x"],
        ),
        _definition(
            "audit_adj_x_relative",
            step="prepost_difference",
            formula="(fs_aud_x - fs_unaud_x) / abs(fs_aud_x)",
            dependencies=["fs_aud_x", "fs_unaud_x"],
        ),
        _definition(
            "ratio_x_to_y",
            step="registered_ratio",
            formula="x_t / y_t",
            dependencies=["fs_aud_x", "fs_aud_y"],
        ),
        _definition(
            "ratio_x_growth",
            step="registered_ratio",
            formula="(x_t - x_t_minus_1) / abs(x_t_minus_1)",
            dependencies=["fs_aud_x[t]", "fs_aud_x[t-1]"],
        ),
    ]
    for feature_id in (
        "obs_audited_core_coverage_count",
        "obs_unaudited_core_coverage_count",
        "obs_prepost_pair_coverage_count",
        "obs_audited_core_complete_flag",
        "obs_unaudited_core_complete_flag",
        "obs_prepost_pair_complete_flag",
    ):
        result.append(
            _definition(
                feature_id,
                step="coverage_count",
                formula="registered coverage",
                dependencies=["registered_core_feature_files"],
            )
        )
    return result


def _source_setup() -> tuple[
    dict[str, Any],
    str,
    dict[str, Any],
    SourceSpec,
    EntityResolutionSpec,
]:
    registry = _registry()
    source_id = feature_source_id(registry)
    sources = mapping(
        mapping(registry.get("data_sources"), "data_sources").get("source_registry"),
        "source_registry",
    ).get("sources")
    source = mapping(mapping(sources, "sources").get(source_id), f"source={source_id}")
    spec = SourceSpec.from_mapping(source_id, source)
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    return registry, source_id, source, spec, entity


def _row(
    semantics: dict[str, Any],
    *,
    firm: str,
    year: int,
    status: str,
    item: str,
    value: float,
) -> dict[str, object]:
    return {
        str(semantics[FIRM_ID]): firm,
        str(semantics[FISCAL_YEAR]): year,
        str(semantics[AUDIT_STATUS]): status,
        str(semantics[ITEM_ID]): item,
        str(semantics[VALUE]): value,
        str(semantics[UNIT]): "VND",
        str(semantics[STATEMENT_SCOPE]): "consolidated",
    }


def _rows(semantics: dict[str, Any]) -> list[dict[str, object]]:
    values = {
        2020: {
            "audited": {"item_x": 10.0, "item_y1": 3.0, "item_y2": 2.0},
            "unaudited": {"item_x": 8.0, "item_y1": 2.0, "item_y2": 1.0},
        },
        2021: {
            "audited": {"item_x": 15.0, "item_y1": 4.0, "item_y2": 2.0},
            "unaudited": {"item_x": 12.0, "item_y1": 3.0, "item_y2": 1.0},
        },
        2023: {
            "audited": {"item_x": 30.0, "item_y1": 5.0, "item_y2": 5.0},
            "unaudited": {"item_x": 24.0, "item_y1": 4.0, "item_y2": 4.0},
        },
    }
    return [
        _row(
            semantics,
            firm="A",
            year=year,
            status=status,
            item=item,
            value=value,
        )
        for year, by_status in values.items()
        for status, by_item in by_status.items()
        for item, value in by_item.items()
    ]


def test_registered_source_generator_builds_atomic_derived_and_observability_features() -> None:
    registry, source_id, source, spec, entity = _source_setup()
    semantics = mapping(source.get("resolved_semantics"), "resolved_semantics")
    panel = pd.DataFrame(
        {
            physical_columns(registry)[FIRM_ID]: pd.Series(["A", "A", "A"], dtype="string"),
            physical_columns(registry)[FISCAL_YEAR]: pd.Series(
                [2020, 2021, 2023], dtype="int16"
            ),
        }
    )
    result = materialize_registered_features(
        base_panel=panel,
        feature_definitions=_definitions(),
        source_rows=_rows(semantics),
        source_id=source_id,
        source_spec=spec,
        semantics=semantics,
        entity=entity,
        controls=_source_controls(source),
        columns=physical_columns(registry),
    )
    generated = result.panel.set_index(physical_columns(registry)[FISCAL_YEAR])
    assert generated.loc[2020, "fs_aud_y"] == pytest.approx(5.0)
    assert generated.loc[2020, "audit_adj_x_signed"] == pytest.approx(2.0)
    assert generated.loc[2020, "audit_adj_x_relative"] == pytest.approx(0.2)
    assert generated.loc[2020, "ratio_x_to_y"] == pytest.approx(2.0)
    assert generated.loc[2021, "ratio_x_growth"] == pytest.approx(0.5)
    assert pd.isna(generated.loc[2023, "ratio_x_growth"])
    assert generated["obs_audited_core_coverage_count"].eq(2.0).all()
    assert generated["obs_unaudited_core_coverage_count"].eq(2.0).all()
    assert generated["obs_prepost_pair_coverage_count"].eq(1.0).all()
    assert generated["obs_audited_core_complete_flag"].eq(1.0).all()
    assert generated["obs_unaudited_core_complete_flag"].eq(1.0).all()
    assert generated["obs_prepost_pair_complete_flag"].eq(1.0).all()
    assert result.audit["generated_feature_count"] == 14
    assert result.audit["prepost_pair_count"] == 1
    assert result.audit["duplicate_measurement_count"] == 0


def test_registered_source_generator_rejects_duplicate_measurement_keys() -> None:
    registry, source_id, source, spec, entity = _source_setup()
    semantics = mapping(source.get("resolved_semantics"), "resolved_semantics")
    rows = _rows(semantics)
    rows.append(dict(rows[0]))
    panel = pd.DataFrame(
        {
            physical_columns(registry)[FIRM_ID]: pd.Series(["A", "A", "A"], dtype="string"),
            physical_columns(registry)[FISCAL_YEAR]: pd.Series(
                [2020, 2021, 2023], dtype="int16"
            ),
        }
    )
    with pytest.raises(ValueError, match="duplicate firm-year-status-item"):
        materialize_registered_features(
            base_panel=panel,
            feature_definitions=_definitions(),
            source_rows=rows,
            source_id=source_id,
            source_spec=spec,
            semantics=semantics,
            entity=entity,
            controls=_source_controls(source),
            columns=physical_columns(registry),
        )
