"""Deterministic feature computation from raw long-format source values.

This module replaces the external pre-computed feature store: P07 now computes
candidate feature values directly from the registered raw sources, driven by the
feature registry's structured ``lineage`` (methodology 3.5, CODEX P07 contract).
No feature list or formula is hard-coded here; every operation reads the
registry definition.

Five registered transformation steps are supported:

* ``source_selection``   -- pick one raw ``source_item_id`` value at firm-year.
* ``component_sum``      -- sum registered raw component items.
* ``prepost_difference`` -- audited minus unaudited of two computed features.
* ``registered_ratio``   -- a bounded arithmetic expression over computed
  features with one-year lag support and ``denominator == 0 -> missing``.
* ``coverage_count``     -- observability counts over a registered core set.

The input ``long_values`` frame is already identity-resolved to the canonical
firm id and filtered to the analysed scope/unit; it holds one numeric value per
``(firm, year, audit_status, source_item_id)``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

FIRM = "firm_master_id"
YEAR = "fiscal_year"
AUDIT = "audit_status"
ITEM = "source_item_id"
VALUE = "value_numeric"

_LAG_SUFFIX = re.compile(r"\[t(-1)?\]$")
_OPERAND = re.compile(r"[A-Za-z_]+_t(?:_minus_1)?\b")


@dataclass(frozen=True)
class FeatureComputation:
    """One computed feature: values indexed by (firm, year) and a status."""

    feature_id: str
    values: pd.Series
    status: str
    reason_code: str | None
    diagnostics: Mapping[str, object] | None = None


# --------------------------------------------------------------------------- #
# Bounded ratio-formula expression evaluator
# --------------------------------------------------------------------------- #

_Token = tuple[str, str]


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c in "+-/*(),":
            tokens.append(("op", c))
            i += 1
            continue
        word = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[i:])
        if word:
            token = word.group(0)
            tokens.append(("func" if token in {"average", "abs"} else "name", token))
            i += len(token)
            continue
        number = re.match(r"\d+(\.\d+)?", text[i:])
        if number:
            tokens.append(("num", number.group(0)))
            i += len(number.group(0))
            continue
        raise ValueError(f"unexpected character {c!r} in ratio formula: {text!r}")
    tokens.append(("end", ""))
    return tokens


class _ExpressionParser:
    """Recursive-descent evaluator for the registered ratio grammar."""

    def __init__(self, tokens: list[_Token], resolve: Callable[[str], float]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._resolve = resolve

    def _peek(self) -> _Token:
        return self._tokens[self._pos]

    def _eat(self, value: str | None = None) -> _Token:
        token = self._tokens[self._pos]
        if value is not None and token[1] != value:
            raise ValueError(f"expected {value!r}, got {token!r}")
        self._pos += 1
        return token

    def evaluate(self) -> float:
        value = self._additive()
        if self._peek()[0] != "end":
            raise ValueError(f"unexpected trailing tokens: {self._tokens[self._pos :]}")
        return value

    def _additive(self) -> float:
        value = self._multiplicative()
        while self._peek() in {("op", "+"), ("op", "-")}:
            operator = self._eat()[1]
            right = self._multiplicative()
            value = value + right if operator == "+" else value - right
        return value

    def _multiplicative(self) -> float:
        value = self._atom()
        while self._peek() in {("op", "/"), ("op", "*")}:
            operator = self._eat()[1]
            right = self._atom()
            if operator == "/":
                value = math.nan if right == 0 else value / right
            else:
                value = value * right
        return value

    def _atom(self) -> float:
        kind, token = self._peek()
        if (kind, token) == ("op", "("):
            self._eat("(")
            value = self._additive()
            self._eat(")")
            return value
        if kind == "func" and token == "average":
            self._eat("average")
            self._eat("(")
            first = self._additive()
            self._eat(",")
            second = self._additive()
            self._eat(")")
            return (first + second) / 2.0
        if kind == "func" and token == "abs":
            self._eat("abs")
            self._eat("(")
            inner = self._additive()
            self._eat(")")
            return abs(inner)
        if kind == "num":
            self._eat()
            return float(token)
        if kind == "name":
            self._eat()
            return self._resolve(token)
        raise ValueError(f"unexpected token {self._peek()!r}")


def evaluate_ratio_formula(formula: str, operand_values: Mapping[str, float]) -> float:
    """Evaluate a registered ratio formula given resolved operand values."""

    def resolve(name: str) -> float:
        if name not in operand_values:
            raise KeyError(f"unresolved operand {name!r} in formula {formula!r}")
        return operand_values[name]

    return _ExpressionParser(_tokenize(formula), resolve).evaluate()


# --------------------------------------------------------------------------- #
# Operation primitives
# --------------------------------------------------------------------------- #


def _lineage(definition: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw: object = definition.get("lineage")
    if not isinstance(raw, list):
        return []
    result: list[Mapping[str, object]] = []
    for raw_item in cast(list[object], raw):
        if not isinstance(raw_item, Mapping):
            continue
        item_mapping = cast(Mapping[object, object], raw_item)
        normalized: dict[str, object] = {}
        for key, value in item_mapping.items():
            if not isinstance(key, str) or not key:
                raise ValueError("lineage keys must be nonempty strings")
            normalized[key] = value
        result.append(normalized)
    return result


def _transformation_step(definition: Mapping[str, object]) -> str | None:
    steps = {str(item.get("transformation_step")) for item in _lineage(definition)}
    steps.discard("None")
    if len(steps) > 1:
        raise ValueError(
            f"feature={definition.get('feature_id')}: mixed transformation steps {sorted(steps)}"
        )
    return next(iter(steps)) if steps else None


def _item_values(long_values: pd.DataFrame, item_id: str, audit_status: str | None) -> pd.Series:
    mask = long_values[ITEM] == item_id
    if audit_status is not None:
        mask &= long_values[AUDIT] == audit_status
    selected = long_values.loc[mask, [FIRM, YEAR, VALUE]]
    if selected.duplicated([FIRM, YEAR]).any():
        raise ValueError(f"raw item {item_id!r} ({audit_status}) has duplicate firm-year values")
    return selected.set_index([FIRM, YEAR])[VALUE].astype("float64")


def _source_selection(definition: Mapping[str, object], long_values: pd.DataFrame) -> pd.Series:
    entries = [item for item in _lineage(definition) if item.get("source_item_id")]
    if len(entries) != 1:
        raise ValueError(
            f"feature={definition.get('feature_id')}: source_selection needs exactly one item"
        )
    item_id = str(entries[0]["source_item_id"])
    return _item_values(long_values, item_id, _audit_status(definition))


def _component_sum(
    definition: Mapping[str, object], long_values: pd.DataFrame
) -> tuple[pd.Series, dict[str, object]]:
    item_ids = [
        str(item["source_item_id"]) for item in _lineage(definition) if item.get("source_item_id")
    ]
    if not item_ids:
        raise ValueError(f"feature={definition.get('feature_id')}: component_sum has no items")
    audit_status = _audit_status(definition)
    frame = pd.concat(
        [_item_values(long_values, item_id, audit_status) for item_id in item_ids],
        axis=1,
        keys=item_ids,
    )
    # A registered sum requires every component present. A firm-year missing any
    # component stays missing (never treated as zero); the dropped firm-years are
    # reported so incomplete coverage is visible rather than silent.
    present = frame.notna()
    all_present = present.all(axis=1)
    any_present = present.any(axis=1)
    incomplete = any_present & ~all_present
    values = frame.sum(axis=1).where(all_present).astype("float64")
    incomplete_rows: list[dict[str, object]] = []
    present_matrix = present.to_numpy(dtype=bool)
    index_values = frame.index.to_list()
    incomplete_positions = np.flatnonzero(incomplete.to_numpy(dtype=bool)).tolist()
    for position in incomplete_positions:
        raw_key: object = cast(object, index_values[int(position)])
        firm_value, year_value = _normalize_firm_year_key(raw_key, "component_sum")
        incomplete_rows.append(
            {
                FIRM: firm_value,
                YEAR: year_value,
                "present_component_count": int(present_matrix[int(position)].sum()),
            }
        )
    diagnostics: dict[str, object] = {
        "component_item_ids": item_ids,
        "required_component_count": len(item_ids),
        "firm_years_complete": int(all_present.sum()),
        "firm_years_incomplete_dropped": int(incomplete.sum()),
        "incomplete_firm_years": incomplete_rows,
    }
    return values, diagnostics


def _prepost_difference(
    definition: Mapping[str, object], computed: Mapping[str, pd.Series]
) -> pd.Series:
    deps = [str(dep) for dep in _dependencies(definition)]
    if len(deps) != 2:
        raise ValueError(
            f"feature={definition.get('feature_id')}: prepost_difference needs two features"
        )
    left = computed[deps[0]]
    right = computed[deps[1]]
    return left.subtract(right, fill_value=np.nan).astype("float64")


def _registered_ratio(
    definition: Mapping[str, object], computed: Mapping[str, pd.Series]
) -> pd.Series:
    formula = str(definition.get("formula"))
    operands = sorted(_OPERAND.findall(formula))
    # Map each operand token (e.g. total_assets_t_minus_1) to a computed feature
    # series (fs_aud_<base>) at the required lag.
    series_by_operand: dict[str, pd.Series] = {}
    for operand in operands:
        base = re.sub(r"_t(_minus_1)?$", "", operand)
        feature_id = f"fs_aud_{base}"
        if feature_id not in computed:
            raise KeyError(
                f"feature={definition.get('feature_id')}: operand {operand!r} -> {feature_id!r} absent"
            )
        series = computed[feature_id]
        if operand.endswith("_minus_1"):
            series = _shift_one_year(series)
        series_by_operand[operand] = series

    index = _union_index(series_by_operand.values())
    results: dict[tuple[str, int], float] = {}
    for untyped_key in index:
        raw_key: object = cast(object, untyped_key)
        key = _normalize_firm_year_key(raw_key, "registered_ratio")
        operand_values: dict[str, float] = {}
        for name, series in series_by_operand.items():
            raw_value: object = series.get(key, np.nan)
            operand_values[name] = float(cast(Any, raw_value))
        results[key] = evaluate_ratio_formula(formula, operand_values)
    return pd.Series(results, dtype="float64").rename_axis([FIRM, YEAR])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize_firm_year_key(raw_key: object, context: str) -> tuple[str, int]:
    if not isinstance(raw_key, tuple):
        raise ValueError(f"{context} requires a two-level firm-year index")
    key_values = cast(tuple[object, ...], raw_key)
    if len(key_values) != 2:
        raise ValueError(f"{context} requires a two-level firm-year index")
    firm_value = key_values[0]
    year_value = key_values[1]
    if isinstance(year_value, bool) or not isinstance(year_value, (int, np.integer)):
        raise ValueError(f"{context} fiscal year must be integer-like")
    return str(firm_value), int(cast(Any, year_value))


def _dependencies(definition: Mapping[str, object]) -> list[str]:
    raw: object = definition.get("dependencies")
    if not isinstance(raw, list):
        return []
    return [str(dep) for dep in cast(list[object], raw)]


def _audit_status(definition: Mapping[str, object]) -> str | None:
    status = definition.get("audit_status")
    return str(status) if status in {"audited", "unaudited"} else None


def _shift_one_year(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    frame = series.rename(VALUE).reset_index()
    frame[YEAR] = frame[YEAR] + 1  # value at t-1 lands on year t
    return frame.set_index([FIRM, YEAR])[VALUE].astype("float64")


def _union_index(series_iterable: Iterable[pd.Series]) -> pd.Index:
    index: pd.Index | None = None
    for series in series_iterable:
        index = series.index if index is None else index.union(series.index)
    return index if index is not None else pd.Index([])


def _feature_id(definition: Mapping[str, object]) -> str:
    return str(definition["feature_id"])


def _topological_order(definitions: Sequence[Mapping[str, object]]) -> list[str]:
    by_id = {_feature_id(d): d for d in definitions}
    resolved: list[str] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def visit(feature_id: str) -> None:
        if feature_id in seen or feature_id not in by_id:
            return
        if feature_id in visiting:
            raise ValueError(f"cyclic feature dependency at {feature_id!r}")
        visiting.add(feature_id)
        for dep in _dependencies(by_id[feature_id]):
            dep_id = _LAG_SUFFIX.sub("", str(dep))
            if dep_id in by_id:
                visit(dep_id)
        visiting.discard(feature_id)
        seen.add(feature_id)
        resolved.append(feature_id)

    for definition in definitions:
        visit(_feature_id(definition))
    return resolved


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

_SUPPORTED_STEPS = {
    "source_selection",
    "component_sum",
    "prepost_difference",
    "registered_ratio",
}


def compute_feature_values(
    definitions: Sequence[Mapping[str, object]],
    long_values: pd.DataFrame,
) -> list[FeatureComputation]:
    """Compute every registry feature from the raw long-format values.

    Returns one :class:`FeatureComputation` per definition. Unsupported steps
    (currently ``coverage_count``, which needs a registered core-feature set not
    present in the registry) are returned with status ``UNSUPPORTED`` rather than
    silently omitted.
    """
    required = {FIRM, YEAR, AUDIT, ITEM, VALUE}
    if not required.issubset(long_values.columns):
        raise ValueError(
            f"long_values missing columns: {sorted(required - set(long_values.columns))}"
        )

    order = _topological_order(definitions)
    by_id = {_feature_id(d): d for d in definitions}
    computed: dict[str, pd.Series] = {}
    results: dict[str, FeatureComputation] = {}

    for feature_id in order:
        definition = by_id[feature_id]
        step = _transformation_step(definition)
        diagnostics: Mapping[str, object] | None = None
        try:
            if step == "source_selection":
                values = _source_selection(definition, long_values)
            elif step == "component_sum":
                values, diagnostics = _component_sum(definition, long_values)
            elif step == "prepost_difference":
                values = _prepost_difference(definition, computed)
            elif step == "registered_ratio":
                values = _registered_ratio(definition, computed)
            else:
                results[feature_id] = FeatureComputation(
                    feature_id, pd.Series(dtype="float64"), "UNSUPPORTED", f"STEP_{step}"
                )
                continue
        except (KeyError, ValueError) as exc:
            results[feature_id] = FeatureComputation(
                feature_id, pd.Series(dtype="float64"), "ERROR", str(exc)
            )
            continue
        computed[feature_id] = values
        results[feature_id] = FeatureComputation(feature_id, values, "PASS", None, diagnostics)

    return [results[_feature_id(d)] for d in definitions]
