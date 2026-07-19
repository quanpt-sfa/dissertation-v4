from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import ConfigurationError
from core.forbidden_patterns import validate_source_patterns


def _registry() -> dict[str, object]:
    return {
        "columns": {
            "scenario_id": {
                "physical_name": "scenario_id",
                "semantic_role": "simulation_scenario",
            }
        },
        "artifacts": {},
    }


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_scenario_metadata_literal_exception_is_path_specific(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scripts/p05_measurement_inputs.py",
        'payload = {"scenario_id": "neutral_pi_03"}\n',
    )
    _write(
        tmp_path,
        "src/selection/preregistered_l3.py",
        'payload = {"scenario_id": "neutral_pi_03"}\n',
    )

    validate_source_patterns(tmp_path, _registry())

    _write(
        tmp_path,
        "scripts/unregistered_production_stage.py",
        'payload = {"scenario_id": "neutral_pi_03"}\n',
    )
    with pytest.raises(ConfigurationError, match="registered physical columns"):
        validate_source_patterns(tmp_path, _registry())
