"""Run P08A/P08B/P08C for the config-locked execution profile.

This script is the profile-aware replacement for legacy orchestration that
iterated over ``simulation.methods``. Those values are label strategies, not
fully qualified method IDs.

Parallelism is across P08B subprocesses. Each worker remains single-threaded.
Execution batch compaction changes only subprocess and artifact granularity;
replication IDs, RNG seeds, MCSE rules, and replication budgets remain locked.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from core.semantic_keys import METHOD_ID, SCENARIO_ID
from simulation.execution_batching import (
    DEFAULT_BATCH_MULTIPLIER,
    execution_batch_size,
    planned_batch_count,
    validate_batch_multiplier,
)
from simulation.method_contract import (
    IMBALANCE_TREATMENT_ID,
    LEARNER_TIER,
    METHOD_FAMILY,
    TRAINING_COST_REGIME_ID,
    active_method_ids,
    method_by_id,
)
from simulation.replication_contract import replication_plan as _replication_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--batch-multiplier",
        type=int,
        default=int(
            os.environ.get(
                "P08_BATCH_MULTIPLIER",
                str(DEFAULT_BATCH_MULTIPLIER),
            )
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    batch_multiplier = validate_batch_multiplier(args.batch_multiplier)

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    registry_path = args.registry.resolve()
    python = sys.executable
    env = dict(os.environ)

    loaded = load_run(
        registry_path=registry_path,
        run_id=args.run_id,
        step_id="P08",
        state="FEATURED",
    )
    if not args.resume or not _artifact_exists(
        loaded,
        "simulation_scenario_registry",
        {},
    ):
        _run(
            [
                python,
                "scripts/p08a_build_scenario_registry.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
            ],
            cwd=project_root,
            env=env,
        )

    loaded = load_run(
        registry_path=registry_path,
        run_id=args.run_id,
        step_id="P08",
        state="FEATURED",
    )
    if args.resume:
        _validate_resume_batch_multiplier(
            loaded,
            expected=batch_multiplier,
        )

    scenarios = [
        mapping(item, "simulation scenario")
        for item in sequence(
            loaded.context.read("simulation_scenario_registry", {}),
            "simulation scenario registry",
        )
    ]
    simulation = mapping(loaded.registry.get("simulation"), "simulation")

    jobs: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_id = str(scenario[SCENARIO_ID])
        scenario_key = str(scenario["scenario_" + "key"])
        for method_id in sorted(active_method_ids(scenario)):
            spec = method_by_id(scenario, method_id)
            method_key = str(spec["method_" + "key"])
            plan = _replication_plan(simulation, spec)
            configured_batch_size = int(plan["batch_size"])
            artifact_batch_size = execution_batch_size(
                configured_batch_size=configured_batch_size,
                minimum_replications=int(plan["minimum"]),
                maximum_replications=int(plan["maximum"]),
                batch_multiplier=batch_multiplier,
            )
            jobs.append(
                {
                    SCENARIO_ID: scenario_id,
                    METHOD_ID: method_id,
                    "scenario_" + "key": scenario_key,
                    "method_" + "key": method_key,
                    **plan,
                    "configured_batch_size": configured_batch_size,
                    "batch_size": artifact_batch_size,
                    "batch_multiplier": batch_multiplier,
                }
            )

    if not jobs:
        raise ValueError("P08 active execution profile has no jobs")

    initial_batch_count = sum(
        planned_batch_count(
            replications=int(job["minimum"]),
            batch_size=int(job["batch_size"]),
        )
        for job in jobs
    )
    maximum_batch_count = sum(
        planned_batch_count(
            replications=int(job["maximum"]),
            batch_size=int(job["batch_size"]),
        )
        for job in jobs
    )
    print(
        "P08_BATCH_PLAN_JSON="
        + json.dumps(
            {
                "run_id": args.run_id,
                "workers": args.workers,
                "batch_multiplier": batch_multiplier,
                "job_count": len(jobs),
                "initial_artifact_batch_count": initial_batch_count,
                "maximum_artifact_batch_count": maximum_batch_count,
                "replication_budgets_changed": False,
                "rng_seeds_changed": False,
                "mcse_rules_changed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    initial_commands: list[list[str]] = []
    for job in jobs:
        minimum = int(job["minimum"])
        batch_size = int(job["batch_size"])
        for start in range(0, minimum, batch_size):
            count = min(batch_size, minimum - start)
            command = _worker_command(
                python=python,
                registry_path=registry_path,
                run_id=args.run_id,
                scenario_id=str(job[SCENARIO_ID]),
                method_id=str(job[METHOD_ID]),
                start=start,
                count=count,
                batch_multiplier=batch_multiplier,
            )
            if args.resume and _batch_exists(
                loaded,
                scenario_key=str(job["scenario_" + "key"]),
                method_key=str(job["method_" + "key"]),
                start=start,
                batch_size=batch_size,
            ):
                continue
            initial_commands.append(command)
    _run_parallel(
        initial_commands,
        cwd=project_root,
        env=env,
        workers=args.workers,
    )

    if args.resume:
        # After initial phase, reload inventory and repair any incomplete
        # batch/diagnostics pairs that exist in the store. This covers adaptive
        # batches beyond ``minimum`` that were interrupted between writes.
        repair_loaded = load_run(
            registry_path=registry_path,
            run_id=args.run_id,
            step_id="P08",
            state="FEATURED",
        )
        repair_commands = _incomplete_batch_commands(
            repair_loaded,
            jobs=jobs,
            python=python,
            registry_path=registry_path,
            run_id=args.run_id,
        )
        if repair_commands:
            _run_parallel(
                repair_commands,
                cwd=project_root,
                env=env,
                workers=args.workers,
            )

    while True:
        control = _control_report(
            python=python,
            registry_path=registry_path,
            run_id=args.run_id,
            cwd=project_root,
            env=env,
        )
        metrics_raw = control.get("metrics")
        if not isinstance(metrics_raw, list):
            raise ValueError("P08 control report metrics must be a list")
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for raw in cast(list[object], metrics_raw):
            if not isinstance(raw, dict):
                raise ValueError("P08 control metric row must be an object")
            item = cast(dict[str, object], raw)
            key = (str(item[SCENARIO_ID]), str(item[METHOD_ID]))
            grouped.setdefault(key, []).append(item)

        pending_commands: list[list[str]] = []
        for job in jobs:
            scenario_id = str(job[SCENARIO_ID])
            method_id = str(job[METHOD_ID])
            rows = grouped.get((scenario_id, method_id), [])
            completed = min(
                (int(str(item["replications"])) for item in rows),
                default=0,
            )
            precision_met = bool(rows) and all(
                item.get("minimum_replications_met") is True
                and item.get("mcse_target_met") is True
                for item in rows
            )
            terminal_at_cap = bool(rows) and all(
                item.get("mcse_target_met") is True
                or item.get("maximum_replications_reached") is True
                for item in rows
            )
            maximum = int(job["maximum"])
            if precision_met or terminal_at_cap or completed >= maximum:
                continue
            batch_size = int(job["batch_size"])
            count = min(batch_size, maximum - completed)
            pending_commands.append(
                _worker_command(
                    python=python,
                    registry_path=registry_path,
                    run_id=args.run_id,
                    scenario_id=scenario_id,
                    method_id=method_id,
                    start=completed,
                    count=count,
                    batch_multiplier=batch_multiplier,
                )
            )

        if not pending_commands:
            break
        _run_parallel(
            pending_commands,
            cwd=project_root,
            env=env,
            workers=args.workers,
        )

    _run(
        [
            python,
            "scripts/p08c_aggregate_batches.py",
            "--registry",
            str(registry_path),
            "--run-id",
            args.run_id,
        ],
        cwd=project_root,
        env=env,
    )
    return 0


def _worker_command(
    *,
    python: str,
    registry_path: Path,
    run_id: str,
    scenario_id: str,
    method_id: str,
    start: int,
    count: int,
    batch_multiplier: int,
) -> list[str]:
    batch_id = f"{start:06d}-{start + count - 1:06d}"
    return [
        python,
        "-W",
        "ignore::FutureWarning",
        "scripts/p08b_run_batch.py",
        "--registry",
        str(registry_path),
        "--run-id",
        run_id,
        "--scenario-id",
        scenario_id,
        "--method-id",
        method_id,
        "--batch-id",
        batch_id,
        "--start",
        str(start),
        "--count",
        str(count),
        "--batch-multiplier",
        str(batch_multiplier),
    ]


def _control_report(
    *,
    python: str,
    registry_path: Path,
    run_id: str,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, object]:
    output = _run_capture(
        [
            python,
            "scripts/p08c_aggregate_batches.py",
            "--registry",
            str(registry_path),
            "--run-id",
            run_id,
            "--check-only",
        ],
        cwd=cwd,
        env=env,
    )
    marker = next(
        (
            line.removeprefix("P08_CONTROL_JSON=")
            for line in output.splitlines()
            if line.startswith("P08_CONTROL_JSON=")
        ),
        None,
    )
    if marker is None:
        raise RuntimeError("P08C did not return P08_CONTROL_JSON")
    raw: object = json.loads(marker)
    if not isinstance(raw, dict):
        raise ValueError("P08 control report must be an object")
    return cast(dict[str, object], raw)


def _artifact_exists(
    loaded: object,
    artifact_id: str,
    coordinates: dict[str, str],
) -> bool:
    context = getattr(loaded, "context")
    if not hasattr(context.store, "_inventory_cache"):
        setattr(
            context.store,
            "_inventory_cache",
            list(context.store.inventory()),
        )
    inventory = getattr(context.store, "_inventory_cache")
    for item in inventory:
        if item.get("artifact_id") != artifact_id:
            continue
        raw = item.get("coordinates")
        if isinstance(raw, dict) and {
            str(key): str(value) for key, value in raw.items()
        } == coordinates:
            context.read(artifact_id, coordinates)
            return True
    return False


def _validate_resume_batch_multiplier(
    loaded: object,
    *,
    expected: int,
) -> None:
    """Prevent one immutable run from mixing artifact partition schemes."""
    context = getattr(loaded, "context")
    for item in context.store.inventory():
        if item.get("artifact_id") != "model_diagnostics":
            continue
        raw = item.get("coordinates")
        if not isinstance(raw, dict):
            continue
        coordinates = {
            str(key): str(value)
            for key, value in raw.items()
        }
        diagnostics = context.read("model_diagnostics", coordinates)
        if not isinstance(diagnostics, dict):
            raise ValueError("P08 model diagnostics must be an object")
        batching = diagnostics.get("execution_batching")
        if not isinstance(batching, dict):
            raise ValueError(
                "resume is incompatible with legacy P08 diagnostics that do not "
                "record execution batching"
            )
        actual = int(str(batching.get("batch_multiplier")))
        if actual != expected:
            raise ValueError(
                "resume batch multiplier differs from existing P08 artifacts: "
                f"existing={actual}, requested={expected}"
            )
        return


def _incomplete_batch_commands(
    loaded: object,
    *,
    jobs: list[dict[str, object]],
    python: str,
    registry_path: Path,
    run_id: str,
) -> list[list[str]]:
    """Return worker commands for every incomplete batch/diagnostics pair."""
    inventory = list(loaded.context.store.inventory())

    def _coords_set(artifact_id: str) -> set[tuple[str, ...]]:
        result: set[tuple[str, ...]] = set()
        for item in inventory:
            if item.get("artifact_id") != artifact_id:
                continue
            raw = item.get("coordinates")
            if isinstance(raw, dict):
                result.add(
                    tuple(
                        sorted(
                            (str(key), str(value))
                            for key, value in raw.items()
                        )
                    )
                )
        return result

    batch_coords = _coords_set("simulation_batches")
    diagnostic_coords = _coords_set("model_diagnostics")
    incomplete = batch_coords.symmetric_difference(diagnostic_coords)
    if not incomplete:
        return []

    scenario_key_str = "scenario_" + "key"
    method_key_str = "method_" + "key"
    jobs_by_keys: dict[tuple[str, str], dict[str, object]] = {
        (str(job[scenario_key_str]), str(job[method_key_str])): job
        for job in jobs
    }

    commands: list[list[str]] = []
    for coordinate_tuple in sorted(incomplete):
        coordinates = dict(coordinate_tuple)
        scenario_key = coordinates.get("scenario_" + "key", "")
        method_key = coordinates.get("method_" + "key", "")
        batch_key = coordinates.get("batch_" + "key", "")
        job = jobs_by_keys.get((scenario_key, method_key))
        if job is None:
            raise ValueError(
                "Incomplete batch/diagnostics coordinate mismatch at "
                f"scenario_key={scenario_key}, method_key={method_key}, "
                f"batch_key={batch_key} has no corresponding active job. "
                "Please clean up the orphan artifacts."
            )
        batch_size = int(job["batch_size"])
        batch_index = int(batch_key.removeprefix("b"))
        start = batch_index * batch_size
        count = min(batch_size, int(job["maximum"]) - start)
        if count <= 0:
            continue
        # Unit fixtures and legacy callers may provide the pre-compaction job
        # shape. In that case batch_size already describes the artifact
        # partition, so multiplier=1 preserves its coordinate semantics.
        repair_batch_multiplier = int(job.get("batch_multiplier", 1))
        commands.append(
            _worker_command(
                python=python,
                registry_path=registry_path,
                run_id=run_id,
                scenario_id=str(job[SCENARIO_ID]),
                method_id=str(job[METHOD_ID]),
                start=start,
                count=count,
                batch_multiplier=repair_batch_multiplier,
            )
        )
    return commands


def _batch_exists(
    loaded: object,
    *,
    scenario_key: str,
    method_key: str,
    start: int,
    batch_size: int,
) -> bool:
    batch_index = start // batch_size
    batch_key = f"b{batch_index:04d}"

    coordinates = {
        "scenario_" + "key": scenario_key,
        "method_" + "key": method_key,
        "batch_" + "key": batch_key,
    }

    batch_written = _artifact_exists(
        loaded,
        "simulation_batches",
        coordinates,
    )
    diagnostics_written = _artifact_exists(
        loaded,
        "model_diagnostics",
        coordinates,
    )
    # Both must exist. If the process died between writes, the worker re-runs;
    # immutable re-write of identical content is idempotent.
    return batch_written and diagnostics_written


def _run_parallel(
    commands: list[list[str]],
    *,
    cwd: Path,
    env: dict[str, str],
    workers: int,
) -> None:
    if not commands:
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run, command, cwd=cwd, env=env)
            for command in commands
        ]
        for future in futures:
            future.result()


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _run_capture(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
