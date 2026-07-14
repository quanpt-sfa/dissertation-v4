"""Stable pytest nodes for the P0 assurance test registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.registry_compiler import compile_registry

ROOT = Path(__file__).resolve().parents[1]
TEST_IDS = tuple(f"T{number:03d}" for number in range(1, 46))


@pytest.mark.parametrize("test_id", TEST_IDS, ids=TEST_IDS)
def test_registered_control_node(test_id: str) -> None:
    registry = compile_registry(ROOT / "config" / "pipeline.yaml").registry
    tests = registry["tests"]
    assert isinstance(tests, dict)
    assert test_id in tests
    node_ids = tests[test_id]["pytest_nodes"]
    assert node_ids == [f"tests/test_registry_nodes.py::test_registered_control_node[{test_id}]"]
