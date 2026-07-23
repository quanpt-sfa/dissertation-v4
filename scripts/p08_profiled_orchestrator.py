"""Run P08A/P08B/P08C for the config-locked execution profile.

This script is the profile-aware replacement for legacy orchestration that
iterated over ``simulation.methods``. Those values are label strategies, not
fully qualified method IDs.

Parallelism is across P08B subprocesses. Each worker remains single-threaded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.pipeline import load_run, mapping, sequence
from core.semantic_keys import BATCH_KEY, METHOD_ID, METHOD_KEY, SCENARIO_ID, SCENARIO_KEY
from simulation.mcse_control import confirmatory_control_decision
from simulation.method_contract import active_method_ids, method_by_id
from simulation.replication_contract import replication_plan

CoordinateKey = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class P08Job:
    scenario_id: str
    method_id: str
    scenario_key: str
    method_key: str
    minimum: int
    batch_size: int
    maximum: int


class _StoreLike(Protocol):
    def inventory(self) -> list[dict[str, object]]: ...


class _ContextLike(Protocol):
    store: _StoreLike

    def read(self, artifact_id: str, coordinates: Mapping[str, str]) -> object: ...


class _LoadedLike(Protocol):
    context: _ContextLike


def _loaded_like(value: object) -> _LoadedLike:
    return cast(_LoadedLike, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args()

    registry_path = cast(Path, args.registry).resolve()
    run_id = cast(str, args.run_id)
    workers = cast(int, args.workers)
    resume = cast(bool, args.resume)
    project_root_arg = cast(Path | None, args.project_root)
    if workers < 1:
        raise ValueError("--workers must be positive")

    project_root = (
        project_root_arg.resolve()
        if project_root_arg is not None
        else Path(__file__).resolve().parents[1]
    )
    python = sys.executable
    env: dict[str, str] = dict(os.environ)

    loaded = load_run(
        registry_path=registry_path,
        run_id=run_id,
        step_id="P08",
        state="FEATURED",
    )
    if not resume or not _artifact_exists(
        loaded,
        loaded.context.store.inventory(),
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
                run_id,
            ],
            cwd=project_root,
            env=env,
        )

    loaded = load_run(
        registry_path=registry_path,
        run_id=run_id,
        step_id="P08",
        state="FEATURED",
    )
    scenarios = [
        mapping(item, "simulation scenario")
        for item in sequence(
            loaded.context.read("simulation_scenario_registry", {}),
            "simulation scenario registry",
        )
    ]
    simulation = mapping(loaded.registry.get("simulation"), "simulation")
    jobs = _build_jobs(scenarios, simulation)
    if not jobs:
        raise ValueError("P08 active execution profile has no jobs")

    initial_inventory = loaded.context.store.inventory()
    initial_commands: list[list[str]] = []
    for job in jobs:
        for start in range(0, job.minimum, job.batch_size):
            count = min(job.batch_size, job.minimum - start)
            command = _worker_command(
                python=python,
                registry_path=registry_path,
                run_id=run_id,
                scenario_id=job.scenario_id,
                method_id=job.method_id,
                start=start,
                count=count,
            )
            if resume and _batch_exists(
                loaded,
                inventory=initial_inventory,
                scenario_key=job.scenario_key,
                method_key=job.method_key,
                start=start,
                batch_size=job.batch_size,
            ):
                continue
            initial_commands.append(command)
    _run_parallel(
        initial_commands,
        cwd=project_root,
        env=env,
        workers=workers,
    )

    if resume:
        repair_loaded = load_run(
            registry_path=registry_path,
            run_id=run_id,
            step_id="P08",
            state="FEATURED",
        )
        repair_commands = _incomplete_batch_commands(
            repair_loaded,
            jobs=jobs,
            python=python,
            registry_path=registry_path,
            run_id=run_id,
        )
        if repair_commands:
            _run_parallel(
                repair_commands,
                cwd=project_root,
                env=env,
                workers=workers,
            )

    while True:
        control = _control_report(
            python=python,
            registry_path=registry_path,
            run_id=run_id,
            cwd=project_root,
            env=env,
        )
        grouped = _group_control_metrics(control)

        pending_commands: list[list[str]] = []
        for job in jobs:
            rows = grouped.get((job.scenario_id, job.method_id), [])
            decision = confirmatory_control_decision(
                rows,
                scenario_id=job.scenario_id,
                method_id=job.method_id,
            )
            completed = decision["completed"]
            if decision["precision_met"] or decision["terminal_at_cap"] or completed >= job.maximum:
                continue
            count = min(job.batch_size, job.maximum - completed)
            pending_commands.append(
                _worker_command(
                    python=python,
                    registry_path=registry_path,
                    run_id=run_id,
                    scenario_id=job.scenario_id,
                    method_id=job.method_id,
                    start=completed,
                    count=count,
                )
            )

        if not pending_commands:
            break
        _run_parallel(
            pending_commands,
            cwd=project_root,
            env=env,
            workers=workers,
        )

    _run(
        [
            python,
            "scripts/p08c_aggregate_batches.py",
            "--registry",
            str(registry_path),
            "--run-id",
            run_id,
        ],
        cwd=project_root,
        env=env,
    )
    return 0


def _build_jobs(
    scenarios: Sequence[Mapping[str, object]],
    simulation: dict[str, object],
) -> list[P08Job]:
    jobs: list[P08Job] = []
    for scenario_raw in scenarios:
        scenario = dict(scenario_raw)
        scenario_id = str(scenario[SCENARIO_ID])
        scenario_key = str(scenario[SCENARIO_KEY])
        for method_id in sorted(active_method_ids(scenario)):
            spec = method_by_id(scenario, method_id)
            plan = replication_plan(simulation, spec)
            jobs.append(
                P08Job(
                    scenario_id=scenario_id,
                    method_id=method_id,
                    scenario_key=scenario_key,
                    method_key=str(spec[METHOD_KEY]),
                    minimum=plan["minimum"],
                    batch_size=plan["batch_size"],
                    maximum=plan["maximum"],
                )
            )
    return jobs


def _group_control_metrics(
    control: Mapping[str, object],
) -> dict[tuple[str, str], list[dict[str, object]]]:
    metrics_raw = control.get("metrics")
    if not isinstance(metrics_raw, list):
        raise ValueError("P08 control report metrics must be a list")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for raw in cast(list[object], metrics_raw):
        if not isinstance(raw, dict):
            raise ValueError("P08 control metric row must be an object")
        item = cast(dict[str, object], raw)
        scenario_id = str(item.get(SCENARIO_ID, ""))
        method_id = str(item.get(METHOD_ID, ""))
        if not scenario_id or not method_id:
            raise ValueError("P08 control metric row requires scenario and method identifiers")
        grouped.setdefault((scenario_id, method_id), []).append(item)
    return grouped


def _worker_command(
    *,
    python: str,
    registry_path: Path,
    run_id: str,
    scenario_id: str,
    method_id: str,
    start: int,
    count: int,
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
    inventory: Sequence[Mapping[str, object]],
    artifact_id: str,
    coordinates: Mapping[str, str],
) -> bool:
    context = _loaded_like(loaded).context
    for item in inventory:
        if item.get("artifact_id") != artifact_id:
            continue
        if _coordinates(item) == dict(coordinates):
            context.read(artifact_id, coordinates)
            return True
    return False


def _incomplete_batch_commands(
    loaded: object,
    *,
    jobs: Sequence[object],
    python: str,
    registry_path: Path,
    run_id: str,
) -> list[list[str]]:
    """Return worker commands for every incomplete batch/diagnostics pair."""
    inventory = _loaded_like(loaded).context.store.inventory()
    normalized_jobs = [_coerce_job(value) for value in jobs]

    def coordinates_for(artifact_id: str) -> set[CoordinateKey]:
        return {
            _coordinate_key(_coordinates(item))
            for item in inventory
            if item.get("artifact_id") == artifact_id
        }

    incomplete = coordinates_for("simulation_batches").symmetric_difference(
        coordinates_for("model_diagnostics")
    )
    if not incomplete:
        return []

    jobs_by_keys = {(job.scenario_key, job.method_key): job for job in normalized_jobs}
    commands: list[list[str]] = []
    for coordinate_key in sorted(incomplete):
        coordinates = dict(coordinate_key)
        scenario_key = coordinates.get(SCENARIO_KEY, "")
        method_key = coordinates.get(METHOD_KEY, "")
        batch_key = coordinates.get(BATCH_KEY, "")
        job = jobs_by_keys.get((scenario_key, method_key))
        if job is None:
            raise ValueError(
                "Incomplete batch/diagnostics coordinate mismatch at "
                f"scenario_key={scenario_key}, method_key={method_key}, batch_key={batch_key}; "
                "no corresponding active job is configured"
            )
        batch_index = _batch_index(batch_key)
        start = batch_index * job.batch_size
        count = min(job.batch_size, job.maximum - start)
        if count <= 0:
            continue
        commands.append(
            _worker_command(
                python=python,
                registry_path=registry_path,
                run_id=run_id,
                scenario_id=job.scenario_id,
                method_id=job.method_id,
                start=start,
                count=count,
            )
        )
    return commands


def _batch_exists(
    loaded: object,
    *,
    inventory: Sequence[Mapping[str, object]] | None = None,
    scenario_key: str,
    method_key: str,
    start: int,
    batch_size: int,
) -> bool:
    batch_key = f"b{start // batch_size:04d}"
    coordinates = {
        SCENARIO_KEY: scenario_key,
        METHOD_KEY: method_key,
        BATCH_KEY: batch_key,
    }
    available = (
        inventory if inventory is not None else _loaded_like(loaded).context.store.inventory()
    )
    batch_written = _artifact_exists(loaded, available, "simulation_batches", coordinates)
    diagnostics_written = _artifact_exists(loaded, available, "model_diagnostics", coordinates)
    return batch_written and diagnostics_written


def _coerce_job(value: object) -> P08Job:
    if isinstance(value, P08Job):
        return value
    if not isinstance(value, dict):
        raise ValueError("P08 job must be a P08Job or mapping")
    raw = cast(dict[str, object], value)
    job = P08Job(
        scenario_id=_required_text(raw.get(SCENARIO_ID), SCENARIO_ID),
        method_id=_required_text(raw.get(METHOD_ID), METHOD_ID),
        scenario_key=_required_text(raw.get(SCENARIO_KEY), SCENARIO_KEY),
        method_key=_required_text(raw.get(METHOD_KEY), METHOD_KEY),
        minimum=_required_positive_integer(raw.get("minimum"), "minimum"),
        batch_size=_required_positive_integer(raw.get("batch_size"), "batch_size"),
        maximum=_required_positive_integer(raw.get("maximum"), "maximum"),
    )
    if job.maximum < job.minimum:
        raise ValueError("P08 job maximum must be at least minimum")
    return job


def _required_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"P08 job {context}: nonempty string required")
    return value


def _required_positive_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"P08 job {context}: positive integer required")
    return value


def _coordinates(item: Mapping[str, object]) -> dict[str, str]:
    raw = item.get("coordinates")
    if not isinstance(raw, dict):
        raise ValueError("artifact manifest coordinates required")
    return {str(key): str(value) for key, value in cast(dict[object, object], raw).items()}


def _coordinate_key(coordinates: Mapping[str, str]) -> CoordinateKey:
    return tuple(sorted(coordinates.items()))


def _batch_index(batch_key: str) -> int:
    if not batch_key.startswith("b") or not batch_key[1:].isdigit():
        raise ValueError(f"invalid P08 batch coordinate={batch_key}")
    return int(batch_key[1:])


def _run_parallel(
    commands: Sequence[list[str]],
    *,
    cwd: Path,
    env: dict[str, str],
    workers: int,
) -> None:
    if not commands:
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run, command, cwd=cwd, env=env) for command in commands]
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
