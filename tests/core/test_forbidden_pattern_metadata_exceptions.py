from __future__ import annotations

from pathlib import Path

import pytest

from core.errors import ConfigurationError
from core.forbidden_patterns import validate_source_patterns
from core.registry_compiler import compile_registry

ROOT = Path(__file__).resolve().parents[2]


def _registry() -> dict[str, object]:
    return {
        "columns": {
            "scenario_id": {
                "physical_name": "scenario_id",
                "semantic_role": "simulation_scenario",
            },
            "outer_fold": {
                "physical_name": "outer_fold",
                "semantic_role": "fold_key",
            },
            "firm_id": {
                "physical_name": "firm_master_id",
                "semantic_role": "entity_key",
            },
        },
        "artifacts": {},
    }


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_control_plane_metadata_roles_are_not_data_bindings(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "scripts/control_plane.py",
        'payload = {"scenario_id": "neutral_pi_03", "outer_fold": "2020"}\n',
    )
    validate_source_patterns(tmp_path, _registry())

    _write(
        tmp_path,
        "scripts/data_binding.py",
        'payload = {"firm_master_id": "F1"}\n',
    )
    with pytest.raises(ConfigurationError, match="registered physical columns"):
        validate_source_patterns(tmp_path, _registry())


def test_exception_is_semantic_role_based_not_name_based(tmp_path: Path) -> None:
    registry = {
        "columns": {
            "misclassified": {
                "physical_name": "scenario_id",
                "semantic_role": "entity_key",
            }
        },
        "artifacts": {},
    }
    _write(tmp_path, "scripts/stage.py", 'payload = {"scenario_id": "F1"}\n')
    with pytest.raises(ConfigurationError, match="registered physical columns"):
        validate_source_patterns(tmp_path, registry)


def test_static_guard_reports_all_files_with_portable_paths(tmp_path: Path) -> None:
    _write(tmp_path, "scripts/first.py", 'payload = {"firm_master_id": "F1"}\n')
    _write(tmp_path, "src/second.py", 'payload = {"firm_master_id": "F2"}\n')

    with pytest.raises(ConfigurationError) as captured:
        validate_source_patterns(tmp_path, _registry())

    message = str(captured.value)
    assert "2 violation(s)" in message
    assert "source=scripts/first.py" in message
    assert "source=src/second.py" in message
    assert str(tmp_path) not in message


def test_current_repository_passes_complete_static_source_scan() -> None:
    compiled = compile_registry(ROOT / "config" / "pipeline.yaml")
    validate_source_patterns(ROOT, dict(compiled.registry))
