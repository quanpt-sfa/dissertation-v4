from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(relative: str, module_name: str) -> ModuleType:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_runner_accepts_positive_execution_controls() -> None:
    module = _load_script(
        "scripts/run_final_l3_production.py",
        "run_final_l3_production_execution_test",
    )
    assert module._validate_workers(1) == 1
    assert module._validate_workers(8) == 8
    assert module.validate_batch_multiplier(1) == 1
    assert module.validate_batch_multiplier(5) == 5
    with pytest.raises(ValueError, match="positive integer"):
        module._validate_workers(0)
    with pytest.raises(ValueError, match="positive integer"):
        module._validate_workers(-2)
    with pytest.raises(ValueError, match="positive integer"):
        module.validate_batch_multiplier(0)


def test_execution_controls_are_propagated_without_changing_protocol() -> None:
    final_source = (
        ROOT / "scripts/run_final_l3_production.py"
    ).read_text(encoding="utf-8")
    pipeline_source = (
        ROOT / "scripts/run_pipeline.py"
    ).read_text(encoding="utf-8")
    orchestrator_source = (
        ROOT / "scripts/p08_profiled_orchestrator.py"
    ).read_text(encoding="utf-8")

    assert 'parser.add_argument("--workers", type=int, default=1)' in final_source
    assert 'os.environ["P08_WORKERS"] = str(workers)' in final_source
    assert 'os.environ["P08_BATCH_MULTIPLIER"] = str(batch_multiplier)' in final_source
    assert '"p08_workers": workers' in final_source
    assert '"p08_worker_count_protocol_hashed": False' in final_source
    assert '"p08_batch_multiplier": batch_multiplier' in final_source
    assert '"p08_batch_multiplier_protocol_hashed": False' in final_source
    assert '"p08_replication_budgets_changed": False' in final_source
    assert 'os.environ.get("P08_WORKERS", "1")' in pipeline_source
    assert 'os.environ.get(' in orchestrator_source
    assert '"P08_BATCH_MULTIPLIER"' in orchestrator_source
    assert '"replication_budgets_changed": False' in orchestrator_source
    assert '"rng_seeds_changed": False' in orchestrator_source


def test_powershell_workflow_exposes_and_forwards_execution_controls() -> None:
    source = (
        ROOT / "scripts/s3_l3_production_workflow.ps1"
    ).read_text(encoding="utf-8")

    assert "[int]$Workers = 1" in source
    assert "[int]$BatchMultiplier = 5" in source
    assert 'if ($Workers -lt 1)' in source
    assert 'if ($BatchMultiplier -lt 1)' in source
    assert 'Write-Host "P08 Workers:          $Workers"' in source
    assert 'Write-Host "P08 BatchMultiplier:  $BatchMultiplier"' in source
    assert '"--workers", [string]$Workers' in source
    assert '"--batch-multiplier", [string]$BatchMultiplier' in source
