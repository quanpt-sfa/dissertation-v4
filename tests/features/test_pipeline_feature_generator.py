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
    _SourceControls,
    feature_source_id,
    materialize_registered_features,
)
from p01.models import RawSchemaSpec, SourceSpec
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


def _source_spec() -> SourceSpec:
    return SourceSpec(
        source_id="financial_statement_core_long",
        enabled=True,
        channel_id="S1",
        source_type="financial_statement_core_long",
        source_agency="fixture",
        original_unit="firm-year-audit-status-item",
        related_period_field="fiscal_year",
        availability_date_field=None,
        availability_date_source="fixture",
        coverage_dimensions=(),
        role="evidence",
        verification_status="observed_opportunity_only",
        data_risks=(),
        relative_path="fixture.csv",
        format="csv",
        encoding="utf-8",
        delimiter=",",
        sheet_name=None,
        header_row=None,
        locked_sha256="0" * 64,
        schema=RawSchemaSpec(
            required_columns=(),
            optional_columns=(),
            key_columns=(),
            date_columns=(),
            required_date_columns=(),
            numeric_columns={},
            allow_extra_columns=True,
            key_unique=False,
            row_count_min=1,
        ),
    )


def _source_entry() -> dict[str, object]:
    return {
        "source_id": "financial_statement_core_long",
        "profile_id": "financial_statement_core_long",
        "enabled": True,
        "channel_id": "S1",
        "source_type": "financial_statement_core_long",
        "source_agency": "fixture",
        "original_unit": "firm-year-audit-status-item",
        "related_period_field": "fiscal_year",
        "availability_date_field": None,
        "availability_date_source": "fixture",
        "coverage_dimensions": [],
        "role": "evidence",
        "verification_status": "observed_opportunity_only",
        "data_risks": [],
        "relative_path": "fixture.csv",
        "format": "csv",
        "encoding": "utf-8",
        "delimiter": ",",
        "sheet_name": None,
        "header_row": None,
        "locked_sha256": "0" * 64,
        "resolved_semantics": {
            FIRM_ID: "issuer_ticker",
            FISCAL_YEAR: "fiscal_year",
            AUDIT_STATUS: "audit_status",
            ITEM_ID: "source_item_id",
            VALUE: "value_numeric",
            UNIT: "unit",
            STATEMENT_SCOPE: "scope",
        },
        "schema": {
            "required_columns": [],
            "optional_columns": [],
            "key_columns": [],
            "date_columns": [],
            "required_date_columns": [],
            "numeric_columns": {},
            "allow_extra_columns": True,
            "key_unique": False,
            "row_count_min": 1,
        },
        "evidence_mapping": {
            "audit_adjustment": {
                "expected_scope": "consolidated",
                "expected_unit": "VND",
            }
        },
    }


def _source_setup() -> tuple[
    dict[str, Any],
    str,
    SourceSpec,
    EntityResolutionSpec,
    _SourceControls,
    dict[str, Any],
]:
    registry = _registry()
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    semantics: dict[str, Any] = {
        FIRM_ID: "issuer_ticker",
        FISCAL_YEAR: "fiscal_year",
        AUDIT_STATUS: "audit_status",
        ITEM_ID: "source_item_id",
        VALUE: "value_numeric",
        UNIT: "unit",
        STATEMENT_SCOPE: "scope",
    }
    return (
        registry,
        "financial_statement_core_long",
        _source_spec(),
        entity,
        _SourceControls(expected_scope="consolidated", expected_unit="VND"),
        semantics,
    )


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


def test_feature_source_resolver_requires_snapshot_injection() -> None:
    registry = _registry()
    with pytest.raises(ValueError, match="found=0"):
        feature_source_id(registry)

    source_registry = mapping(
        mapping(registry.get("data_sources"), "data_sources").get("source_registry"),
        "data_sources.source_registry",
    )
    sources = mapping(source_registry.get("sources"), "data_sources.source_registry.sources")
    sources["financial_statement_core_long"] = _source_entry()
    assert feature_source_id(registry) == "financial_statement_core_long"


def test_registered_source_generator_builds_atomic_derived_and_observability_features() -> None:
    registry, source_id, spec, entity, controls, semantics = _source_setup()
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
        controls=controls,
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
    registry, source_id, spec, entity, controls, semantics = _source_setup()
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
            controls=controls,
            columns=physical_columns(registry),
        )
