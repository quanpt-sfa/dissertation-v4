"""Deterministic P07 feature generation from the locked financial-statement source."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from core.semantic_keys import (
    AUDIT_STATUS,
    FIRM_ID,
    FISCAL_YEAR,
    ITEM_ID,
    STATEMENT_SCOPE,
    UNIT,
    VALUE,
)
from p01.models import SourceSpec
from p01.readers import iter_rows
from p01.registry import resolve_source
from p02.builder import normalize_entity_field, resolve_entity_link
from p02.models import EntityResolutionSpec

_FEATURE_SOURCE_TYPE = "financial_statement_core_long"
_ATOMIC_STEPS = frozenset({"source_selection", "component_sum"})
_DERIVED_STEPS = frozenset({"prepost_difference", "registered_ratio"})
_OBSERVABILITY_STEPS = frozenset({"coverage_count"})
_DEPENDENCY_PATTERN = re.compile(
    r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\[(?P<period>t|t-1)\])?$"
)


@dataclass(frozen=True)
class GeneratedFeatureInput:
    """Feature-enriched P02 spine and its generation audit."""

    panel: pd.DataFrame
    audit: dict[str, object]


@dataclass(frozen=True)
class _AtomicDefinition:
    feature_id: str
    physical_column: str
    audit_status: str
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SourceControls:
    expected_scope: str
    expected_unit: str


Value = pd.Series | float


def build_pipeline_feature_input(
    *,
    base_panel: pd.DataFrame,
    feature_definitions: list[dict[str, object]],
    registry: dict[str, Any],
    raw_audit: dict[str, Any],
    columns: dict[str, str],
) -> GeneratedFeatureInput:
    """Build all LOCKED P07 features from one P01-audited registered source."""
    source_id, source = _feature_source(registry)
    decision = _mapping(raw_audit.get("decision"), f"raw_audit source={source_id}.decision")
    if decision.get("pipeline_may_advance") is not True:
        raise ValueError(f"source={source_id}: passing P01 audit required before P07 generation")
    spec, path = resolve_source(registry, source_id)
    semantics = _mapping(source.get("resolved_semantics"), f"source={source_id}.resolved_semantics")
    controls = _source_controls(source)
    entity = EntityResolutionSpec.from_mapping(registry.get("entity_resolution"))
    result = materialize_registered_features(
        base_panel=base_panel,
        feature_definitions=feature_definitions,
        source_rows=iter_rows(path, spec),
        source_id=source_id,
        source_spec=spec,
        semantics=semantics,
        entity=entity,
        controls=controls,
        columns=columns,
    )
    audit = {
        **result.audit,
        "source_relative_path": spec.relative_path,
        "source_locked_sha256": spec.locked_sha256,
        "source_audit_status": raw_audit.get("status"),
    }
    return GeneratedFeatureInput(panel=result.panel, audit=audit)


def materialize_registered_features(
    *,
    base_panel: pd.DataFrame,
    feature_definitions: list[dict[str, object]],
    source_rows: Iterable[dict[str, object]],
    source_id: str,
    source_spec: SourceSpec,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    controls: _SourceControls,
    columns: dict[str, str],
) -> GeneratedFeatureInput:
    """Pure materialization boundary used by production and focused tests."""
    firm_column = columns[FIRM_ID]
    year_column = columns[FISCAL_YEAR]
    required_panel = {firm_column, year_column}
    if not required_panel.issubset(base_panel.columns):
        raise ValueError(
            f"P07 feature generation panel missing columns {sorted(required_panel - set(base_panel.columns))}"
        )
    panel = base_panel.copy()
    panel[firm_column] = panel[firm_column].astype("string")
    panel[year_column] = pd.to_numeric(panel[year_column], errors="raise").astype("int16")
    if panel.duplicated([firm_column, year_column]).any():
        raise ValueError("P07 feature generation requires unique firm-year panel rows")
    panel = panel.sort_values([firm_column, year_column], kind="stable").reset_index(drop=True)

    locked = [
        dict(definition)
        for definition in feature_definitions
        if definition.get("research_decision_status") == "LOCKED"
    ]
    atomic, derived, observability = _classify_definitions(locked)
    atomic_specs = [_atomic_definition(definition) for definition in atomic]
    feature_ids = {str(definition["feature_id"]) for definition in locked}
    if len(feature_ids) != len(locked):
        raise ValueError("P07 LOCKED feature definitions contain duplicate feature IDs")
    physical_columns = {
        str(definition.get("physical_column"))
        for definition in locked
        if isinstance(definition.get("physical_column"), str)
    }
    collisions = sorted(physical_columns & set(panel.columns))
    if collisions:
        raise ValueError(f"P07 refuses to overwrite pre-existing feature columns: {collisions}")

    source_values, source_audit = _collect_source_values(
        source_rows=source_rows,
        source_id=source_id,
        source_spec=source_spec,
        semantics=semantics,
        entity=entity,
        controls=controls,
        atomic_specs=atomic_specs,
        panel_keys=set(
            zip(panel[firm_column].astype(str), panel[year_column].astype(int), strict=True)
        ),
    )
    feature_series: dict[str, pd.Series] = {}
    panel_index = pd.MultiIndex.from_frame(panel[[firm_column, year_column]])
    for definition in atomic_specs:
        values = _atomic_series(definition, source_values, panel_index)
        feature_series[definition.feature_id] = values

    pending = list(derived)
    while pending:
        progressed = False
        remaining: list[dict[str, object]] = []
        for definition in pending:
            dependencies = _dependency_tokens(definition)
            missing = sorted(
                {
                    _dependency_parts(token)[0]
                    for token in dependencies
                    if _dependency_parts(token)[0] not in feature_series
                }
            )
            if missing:
                remaining.append(definition)
                continue
            feature_id = str(definition["feature_id"])
            feature_series[feature_id] = _evaluate_registered_formula(
                definition=definition,
                dependencies=dependencies,
                feature_series=feature_series,
                panel=panel,
                firm_column=firm_column,
                year_column=year_column,
            )
            progressed = True
        if not progressed:
            blockers = {
                str(definition.get("feature_id")): sorted(
                    {
                        _dependency_parts(token)[0]
                        for token in _dependency_tokens(definition)
                        if _dependency_parts(token)[0] not in feature_series
                    }
                )
                for definition in remaining
            }
            raise ValueError(f"P07 derived feature dependency graph is unresolved: {blockers}")
        pending = remaining

    pair_dependencies = _prepost_pairs(derived)
    _materialize_observability(
        definitions=observability,
        atomic_definitions=atomic,
        pair_dependencies=pair_dependencies,
        feature_series=feature_series,
    )

    missing_outputs = sorted(feature_ids - set(feature_series))
    unknown_outputs = sorted(set(feature_series) - feature_ids)
    if missing_outputs or unknown_outputs:
        raise ValueError(
            "P07 generated feature registry mismatch: "
            f"missing={missing_outputs}, unknown={unknown_outputs}"
        )
    definition_by_id = {str(definition["feature_id"]): definition for definition in locked}
    for feature_id in sorted(feature_series):
        definition = definition_by_id[feature_id]
        physical = definition.get("physical_column")
        if not isinstance(physical, str) or not physical:
            raise ValueError(f"feature={feature_id}: LOCKED physical_column required")
        panel[physical] = pd.to_numeric(feature_series[feature_id], errors="raise").astype("float64")

    audit: dict[str, object] = {
        "status": "PIPELINE_FEATURE_GENERATION_VALID",
        "generation_mode": "registered_source_to_feature_panel",
        "source_id": source_id,
        "panel_row_count": len(panel),
        "panel_firm_count": int(panel[firm_column].nunique()),
        "locked_feature_count": len(locked),
        "atomic_feature_count": len(atomic),
        "derived_feature_count": len(derived),
        "observability_feature_count": len(observability),
        "prepost_pair_count": len(pair_dependencies),
        "generated_feature_count": len(feature_series),
        "outer_outcomes_accessed": False,
        "known_cases_accessed": False,
        "preprocessing_fit_at_p07": False,
        **source_audit,
    }
    return GeneratedFeatureInput(panel=panel, audit=audit)


def _feature_source(registry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    data_sources = _mapping(registry.get("data_sources"), "data_sources")
    source_registry = _mapping(data_sources.get("source_registry"), "data_sources.source_registry")
    sources = _mapping(source_registry.get("sources"), "data_sources.source_registry.sources")
    candidates = [
        (str(source_id), _mapping(raw, f"source={source_id}"))
        for source_id, raw in sources.items()
        if isinstance(raw, dict)
        and raw.get("enabled") is True
        and raw.get("source_type") == _FEATURE_SOURCE_TYPE
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"P07 requires exactly one enabled {_FEATURE_SOURCE_TYPE} source; found={len(candidates)}"
        )
    return candidates[0]


def feature_source_id(registry: dict[str, Any]) -> str:
    """Expose the resolved source ID so P07 can read its P01 audit receipt."""
    return _feature_source(registry)[0]


def _source_controls(source: dict[str, Any]) -> _SourceControls:
    mapping = _mapping(source.get("evidence_mapping"), "financial-statement evidence_mapping")
    adjustment = _mapping(mapping.get("audit_adjustment"), "audit_adjustment")
    scope = adjustment.get("expected_scope")
    unit = adjustment.get("expected_unit")
    if not isinstance(scope, str) or not scope or not isinstance(unit, str) or not unit:
        raise ValueError("P07 source controls require locked expected scope and unit")
    return _SourceControls(expected_scope=scope, expected_unit=unit)


def _classify_definitions(
    definitions: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    atomic: list[dict[str, object]] = []
    derived: list[dict[str, object]] = []
    observability: list[dict[str, object]] = []
    unsupported: dict[str, list[str]] = {}
    for definition in definitions:
        feature_id = str(definition.get("feature_id"))
        steps = _transformation_steps(definition)
        if steps and steps.issubset(_ATOMIC_STEPS):
            atomic.append(definition)
        elif steps and steps.issubset(_DERIVED_STEPS):
            derived.append(definition)
        elif steps and steps.issubset(_OBSERVABILITY_STEPS):
            observability.append(definition)
        else:
            unsupported[feature_id] = sorted(steps)
    if unsupported:
        raise ValueError(f"P07 unsupported LOCKED transformation steps: {unsupported}")
    return atomic, derived, observability


def _transformation_steps(definition: dict[str, object]) -> set[str]:
    raw = definition.get("lineage")
    if not isinstance(raw, list):
        return set()
    result: set[str] = set()
    for value in cast(list[object], raw):
        lineage = _mapping(value, f"feature={definition.get('feature_id')}.lineage")
        step = lineage.get("transformation_step")
        if isinstance(step, str) and step:
            result.add(step)
    return result


def _atomic_definition(definition: dict[str, object]) -> _AtomicDefinition:
    feature_id = definition.get("feature_id")
    physical = definition.get("physical_column")
    audit_status = definition.get("audit_status")
    if not all(isinstance(value, str) and value for value in (feature_id, physical, audit_status)):
        raise ValueError(
            f"feature={feature_id}: atomic feature requires feature_id, physical_column, audit_status"
        )
    raw_lineage = definition.get("lineage")
    if not isinstance(raw_lineage, list):
        raise ValueError(f"feature={feature_id}: atomic lineage required")
    items: list[str] = []
    for raw in cast(list[object], raw_lineage):
        lineage = _mapping(raw, f"feature={feature_id}.lineage")
        item = lineage.get("source_item_id")
        if not isinstance(item, str) or not item:
            raise ValueError(f"feature={feature_id}: source_item_id required for atomic lineage")
        items.append(item)
    if not items or len(items) != len(set(items)):
        raise ValueError(f"feature={feature_id}: atomic source items must be unique and nonempty")
    return _AtomicDefinition(
        feature_id=cast(str, feature_id),
        physical_column=cast(str, physical),
        audit_status=cast(str, audit_status),
        item_ids=tuple(items),
    )


def _collect_source_values(
    *,
    source_rows: Iterable[dict[str, object]],
    source_id: str,
    source_spec: SourceSpec,
    semantics: dict[str, Any],
    entity: EntityResolutionSpec,
    controls: _SourceControls,
    atomic_specs: list[_AtomicDefinition],
    panel_keys: set[tuple[str, int]],
) -> tuple[dict[tuple[str, int, str, str], float | None], dict[str, object]]:
    required = {
        FIRM_ID,
        FISCAL_YEAR,
        AUDIT_STATUS,
        ITEM_ID,
        VALUE,
        UNIT,
        STATEMENT_SCOPE,
    }
    if not required.issubset(semantics):
        raise ValueError(
            f"source={source_id}: unresolved feature semantics {sorted(required - set(semantics))}"
        )
    needed_items = {item for definition in atomic_specs for item in definition.item_ids}
    needed_statuses = {definition.audit_status for definition in atomic_specs}
    values: dict[tuple[str, int, str, str], float | None] = {}
    duplicate_keys: list[tuple[str, int, str, str]] = []
    scanned = 0
    candidate = 0
    selected = 0
    excluded_scope = 0
    excluded_unit = 0
    excluded_panel = 0
    entity_cache: dict[str, str] = {}
    for row in source_rows:
        scanned += 1
        item = _text(row.get(str(semantics[ITEM_ID])))
        status = _text(row.get(str(semantics[AUDIT_STATUS])))
        if item not in needed_items or status not in needed_statuses:
            continue
        candidate += 1
        scope = _text(row.get(str(semantics[STATEMENT_SCOPE])))
        unit = _text(row.get(str(semantics[UNIT])))
        if scope.casefold() != controls.expected_scope.casefold():
            excluded_scope += 1
            continue
        if unit.casefold() != controls.expected_unit.casefold():
            excluded_unit += 1
            continue
        raw_firm = _text(row.get(str(semantics[FIRM_ID])))
        canonical = entity_cache.get(raw_firm)
        if canonical is None:
            normalized = normalize_entity_field(raw_firm, entity)
            canonical, _ = resolve_entity_link(source_id, raw_firm, normalized, entity)
            entity_cache[raw_firm] = canonical
        year = _integer(row.get(str(semantics[FISCAL_YEAR])), FISCAL_YEAR)
        if (canonical, year) not in panel_keys:
            excluded_panel += 1
            continue
        key = (canonical, year, status, item)
        if key in values:
            duplicate_keys.append(key)
            continue
        values[key] = _numeric_or_none(row.get(str(semantics[VALUE])))
        selected += 1
    if duplicate_keys:
        examples = sorted(set(duplicate_keys))[:20]
        raise ValueError(
            f"source={source_id}: duplicate firm-year-status-item rows block P07; "
            f"count={len(duplicate_keys)} examples={examples}"
        )
    if not values:
        raise ValueError(f"source={source_id}: no registered feature measurements matched the P02 panel")
    audit: dict[str, object] = {
        "raw_rows_scanned": scanned,
        "candidate_source_rows": candidate,
        "selected_unique_measurements": selected,
        "excluded_scope_rows": excluded_scope,
        "excluded_unit_rows": excluded_unit,
        "excluded_nonpanel_rows": excluded_panel,
        "duplicate_measurement_count": 0,
        "needed_source_item_count": len(needed_items),
        "source_format": source_spec.format,
        "scope_rule": controls.expected_scope,
        "unit_rule": controls.expected_unit,
    }
    return values, audit


def _atomic_series(
    definition: _AtomicDefinition,
    values: dict[tuple[str, int, str, str], float | None],
    panel_index: pd.MultiIndex,
) -> pd.Series:
    components: list[pd.Series] = []
    for item in definition.item_ids:
        mapped = {
            (firm, year): value
            for (firm, year, status, source_item), value in values.items()
            if status == definition.audit_status and source_item == item
        }
        components.append(
            pd.Series(mapped, index=pd.MultiIndex.from_tuples(mapped), dtype="float64").reindex(
                panel_index
            )
            if mapped
            else pd.Series(index=panel_index, dtype="float64")
        )
    frame = pd.concat(components, axis=1)
    if len(components) == 1:
        result = frame.iloc[:, 0]
    else:
        result = frame.sum(axis=1, min_count=len(components))
    result.index = pd.RangeIndex(len(result))
    return result.astype("float64")


def _dependency_tokens(definition: dict[str, object]) -> list[str]:
    raw = definition.get("dependencies")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"feature={definition.get('feature_id')}: derived dependencies required")
    tokens = [str(value) for value in cast(list[object], raw)]
    for token in tokens:
        _dependency_parts(token)
    return tokens


def _dependency_parts(token: str) -> tuple[str, int]:
    match = _DEPENDENCY_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"unsupported registered feature dependency token: {token}")
    period = match.group("period")
    return match.group("base"), 1 if period == "t-1" else 0


def _evaluate_registered_formula(
    *,
    definition: dict[str, object],
    dependencies: list[str],
    feature_series: dict[str, pd.Series],
    panel: pd.DataFrame,
    firm_column: str,
    year_column: str,
) -> pd.Series:
    formula = definition.get("formula")
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError(f"feature={definition.get('feature_id')}: registered formula required")
    environment: dict[str, pd.Series] = {}
    for token in dependencies:
        base, lag = _dependency_parts(token)
        series = feature_series[base]
        resolved = (
            _exact_one_year_lag(
                series=series,
                panel=panel,
                firm_column=firm_column,
                year_column=year_column,
            )
            if lag
            else series
        )
        if lag == 0:
            environment[base] = resolved
        alias = _formula_alias(base, lag)
        if alias in environment and not environment[alias].equals(resolved):
            raise ValueError(
                f"feature={definition.get('feature_id')}: ambiguous formula alias {alias}"
            )
        environment[alias] = resolved
    try:
        expression = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"feature={definition.get('feature_id')}: invalid registered formula syntax"
        ) from exc
    result = _evaluate_expression(expression.body, environment, panel.index)
    return _as_series(result, panel.index).astype("float64")


def _formula_alias(base: str, lag: int) -> str:
    for prefix in ("fs_aud_", "fs_unaud_"):
        if base.startswith(prefix):
            concept = base[len(prefix) :]
            return f"{concept}_t_minus_1" if lag else f"{concept}_t"
    return f"{base}_t_minus_1" if lag else f"{base}_t"


def _exact_one_year_lag(
    *,
    series: pd.Series,
    panel: pd.DataFrame,
    firm_column: str,
    year_column: str,
) -> pd.Series:
    frame = panel.loc[:, [firm_column, year_column]].copy()
    frame["__value"] = series.to_numpy()
    grouped = frame.groupby(firm_column, sort=False, observed=True)
    previous_value = grouped["__value"].shift(1)
    previous_year = grouped[year_column].shift(1)
    exact = panel[year_column].astype("float64") - previous_year.astype("float64") == 1.0
    return previous_value.where(exact).astype("float64")


def _evaluate_expression(node: ast.AST, environment: dict[str, pd.Series], index: pd.Index) -> Value:
    if isinstance(node, ast.Name):
        if node.id not in environment:
            raise ValueError(f"registered formula references undeclared dependency: {node.id}")
        return environment[node.id]
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("registered feature formulas allow numeric constants only")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand, environment, index)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _evaluate_expression(node.left, environment, index)
        right = _evaluate_expression(node.right, environment, index)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return _safe_divide(left, right, index)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.keywords:
            raise ValueError("registered feature formula functions do not accept keywords")
        arguments = [_evaluate_expression(argument, environment, index) for argument in node.args]
        if node.func.id == "abs" and len(arguments) == 1:
            return abs(arguments[0])
        if node.func.id == "average" and len(arguments) >= 2:
            frame = pd.concat([_as_series(value, index) for value in arguments], axis=1)
            return frame.mean(axis=1, skipna=False)
        raise ValueError(f"unsupported registered feature formula function: {node.func.id}")
    raise ValueError(f"unsupported registered feature formula node: {type(node).__name__}")


def _safe_divide(left: Value, right: Value, index: pd.Index) -> pd.Series:
    numerator = _as_series(left, index)
    denominator = _as_series(right, index)
    denominator = denominator.where(denominator.ne(0.0))
    return (numerator / denominator).astype("float64")


def _as_series(value: Value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        result = value.copy()
        result.index = index
        return result.astype("float64")
    return pd.Series(float(value), index=index, dtype="float64")


def _prepost_pairs(definitions: list[dict[str, object]]) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for definition in definitions:
        if "prepost_difference" not in _transformation_steps(definition):
            continue
        dependencies = [_dependency_parts(token)[0] for token in _dependency_tokens(definition)]
        if len(dependencies) != 2:
            raise ValueError(
                f"feature={definition.get('feature_id')}: prepost adjustment requires two dependencies"
            )
        audited = [value for value in dependencies if value.startswith("fs_aud_")]
        unaudited = [value for value in dependencies if value.startswith("fs_unaud_")]
        if len(audited) != 1 or len(unaudited) != 1:
            raise ValueError(
                f"feature={definition.get('feature_id')}: prepost dependencies must bind audited/unaudited pair"
            )
        pairs.add((audited[0], unaudited[0]))
    return sorted(pairs)


def _materialize_observability(
    *,
    definitions: list[dict[str, object]],
    atomic_definitions: list[dict[str, object]],
    pair_dependencies: list[tuple[str, str]],
    feature_series: dict[str, pd.Series],
) -> None:
    audited = [
        str(item["feature_id"])
        for item in atomic_definitions
        if item.get("audit_status") == "audited"
    ]
    unaudited = [
        str(item["feature_id"])
        for item in atomic_definitions
        if item.get("audit_status") == "unaudited"
    ]
    if not audited or not unaudited or not pair_dependencies:
        raise ValueError("P07 observability generation requires audited, unaudited, and pair groups")
    audited_count = pd.concat([feature_series[value] for value in audited], axis=1).notna().sum(axis=1)
    unaudited_count = (
        pd.concat([feature_series[value] for value in unaudited], axis=1).notna().sum(axis=1)
    )
    pair_count = sum(
        feature_series[audited_id].notna() & feature_series[unaudited_id].notna()
        for audited_id, unaudited_id in pair_dependencies
    )
    for definition in definitions:
        feature_id = str(definition["feature_id"])
        normalized = feature_id.casefold()
        if "unaudited_core" in normalized:
            count = unaudited_count
            required = len(unaudited)
        elif "audited_core" in normalized:
            count = audited_count
            required = len(audited)
        elif "prepost_pair" in normalized:
            count = pair_count
            required = len(pair_dependencies)
        else:
            raise ValueError(f"feature={feature_id}: unsupported observability group")
        if normalized.endswith("coverage_count"):
            feature_series[feature_id] = pd.Series(count, dtype="float64")
        elif normalized.endswith("complete_flag"):
            feature_series[feature_id] = pd.Series(count.eq(required), dtype="float64")
        else:
            raise ValueError(f"feature={feature_id}: unsupported observability operation")


def _mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: mapping required")
    return cast(dict[str, Any], value)


def _text(value: object) -> str:
    if value is None or not str(value).strip():
        return ""
    return str(value).strip()


def _integer(value: object, context: str) -> int:
    text = _text(value)
    if not text:
        raise ValueError(f"financial-statement row missing {context}")
    return int(float(text))


def _numeric_or_none(value: object) -> float | None:
    text = _text(value)
    if not text:
        return None
    parsed = float(text)
    return parsed if math.isfinite(parsed) else None
