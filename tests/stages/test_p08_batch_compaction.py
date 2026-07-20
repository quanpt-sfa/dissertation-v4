from __future__ import annotations

from pathlib import Path

import pytest

from simulation.execution_batching import (
    DEFAULT_BATCH_MULTIPLIER,
    execution_batch_size,
    planned_batch_count,
    validate_batch_multiplier,
)

ROOT = Path(__file__).resolve().parents[2]


def test_default_dissertation_batch_plan_preserves_replication_budgets() -> None:
    core_batch_size = execution_batch_size(
        configured_batch_size=250,
        minimum_replications=2500,
        maximum_replications=10000,
        batch_multiplier=DEFAULT_BATCH_MULTIPLIER,
    )
    l3_batch_size = execution_batch_size(
        configured_batch_size=100,
        minimum_replications=1000,
        maximum_replications=5000,
        batch_multiplier=DEFAULT_BATCH_MULTIPLIER,
    )

    assert DEFAULT_BATCH_MULTIPLIER == 5
    assert core_batch_size == 1250
    assert l3_batch_size == 500
    assert planned_batch_count(replications=2500, batch_size=core_batch_size) == 2
    assert planned_batch_count(replications=1000, batch_size=l3_batch_size) == 2

    scenario_count = 17
    predictive_method_count = 4
    standalone_method_count = 6
    initial_batches = scenario_count * (
        predictive_method_count * 2 + standalone_method_count * 2
    )
    maximum_batches = scenario_count * (
        predictive_method_count
        * planned_batch_count(replications=10000, batch_size=core_batch_size)
        + standalone_method_count
        * planned_batch_count(replications=5000, batch_size=l3_batch_size)
    )

    assert initial_batches == 340
    assert maximum_batches == 1564


def test_compaction_selects_largest_aligned_factor_not_exceeding_request() -> None:
    assert execution_batch_size(
        configured_batch_size=100,
        minimum_replications=600,
        maximum_replications=2400,
        batch_multiplier=5,
    ) == 300


def test_invalid_batch_plans_fail_closed() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_batch_multiplier(0)
    with pytest.raises(ValueError, match="align"):
        execution_batch_size(
            configured_batch_size=250,
            minimum_replications=2600,
            maximum_replications=10000,
            batch_multiplier=5,
        )
    with pytest.raises(ValueError, match="at least the minimum"):
        execution_batch_size(
            configured_batch_size=100,
            minimum_replications=1000,
            maximum_replications=900,
            batch_multiplier=5,
        )


def test_worker_records_execution_batching_without_touching_seed_keys() -> None:
    worker_source = (ROOT / "scripts/p08b_run_batch.py").read_text(
        encoding="utf-8"
    )
    orchestrator_source = (
        ROOT / "scripts/p08_profiled_orchestrator.py"
    ).read_text(encoding="utf-8")

    assert 'diagnostics["execution_batching"]' in worker_source
    assert '"changes_replication_ids_or_seeds": False' in worker_source
    assert '"protocol_hashed": False' in worker_source
    assert '"P08_REPLICATION_DATA"' in worker_source
    assert '"P08_REPLICATION_MODEL"' in worker_source
    assert '"--batch-multiplier"' in worker_source
    assert '"--batch-multiplier"' in orchestrator_source
    assert "_validate_resume_batch_multiplier" in orchestrator_source
