"""Regression coverage for the sealed P15 registry contract."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_p15() -> ModuleType:
    path = ROOT / "scripts" / "p15_open_known_cases.py"
    spec = importlib.util.spec_from_file_location("p15_known_case_registry", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load P15 from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p15_locks_external_validation_flags() -> None:
    module = _load_p15()
    flags = cast(dict[str, bool], getattr(module, "_EXPECTED_INCLUSION_FLAGS"))
    assert flags == {
        "training_include_flag": False,
        "calibration_include_flag": False,
        "model_selection_include_flag": False,
        "external_validation_include_flag": True,
    }
    assert getattr(module, "_EXPECTED_CASE_CONSTRUCT") == "CONFIRMED_FINANCIAL_REPORTING_CASE"
    assert getattr(module, "_EXPECTED_CASE_ROLE") == "SIMULATION_EXTERNAL_VALIDATION"


def test_p15_boolean_parser_is_explicit_and_fail_closed() -> None:
    module = _load_p15()
    parse_bool = cast(Callable[[object, str], bool], getattr(module, "_parse_bool"))
    assert parse_bool(True, "flag") is True
    assert parse_bool("1", "flag") is True
    assert parse_bool("false", "flag") is False
    assert parse_bool("N", "flag") is False
    with pytest.raises(ValueError, match="boolean value required"):
        parse_bool("unknown", "flag")


def test_p15_no_longer_requires_embedded_case_columns() -> None:
    source = (ROOT / "scripts" / "p15_open_known_cases.py").read_text(encoding="utf-8")
    assert "known_case_seal_status" not in source
    assert "known_case_opens_at_step" not in source
    assert "snapshot-locked, validation-only known-case registry" in source
