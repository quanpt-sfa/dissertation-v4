"""Single-command immutable production runner from snapshot through P17."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _run_capture(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
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


def _require_clean_tree(root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    )
    if result.stdout.strip():
        raise RuntimeError("operational pipeline requires a clean committed Git tree")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/pipeline.yaml"))
    stage_ids = [f"P{value:02d}" for value in range(18)]
    parser.add_argument("--through", choices=stage_ids, default="P17")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    if not args.allow_dirty:
        _require_clean_tree(project_root)
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / args.run_id
    if run_root.exists():
        raise FileExistsError(f"run root already exists and is immutable: {run_root}")

    snapshot_path = run_root / "SNAPSHOT" / "data_snapshot.json"
    env = dict(os.environ)
    env["DISSERTATION_RAW_ROOT"] = str(raw_root)
    python = sys.executable

    _run(
        [
            python,
            "scripts/create_data_snapshot.py",
            "--config",
            str(args.config),
            "--raw-root",
            str(raw_root),
            "--snapshot-id",
            args.run_id,
            "--output",
            str(snapshot_path),
        ],
        cwd=project_root,
        env=env,
    )
    _run(
        [
            python,
            "scripts/p00_lock_protocol.py",
            "--config",
            str(args.config),
            "--run-id",
            args.run_id,
            "--output-root",
            str(output_root),
            "--snapshot-manifest",
            str(snapshot_path),
        ],
        cwd=project_root,
        env=env,
    )
    if args.through == "P00":
        return 0

    snapshot_raw: object = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot_raw, dict):
        raise ValueError("snapshot must be an object")
    snapshot = cast(dict[str, object], snapshot_raw)
    sources_raw = snapshot.get("sources")
    if not isinstance(sources_raw, list):
        raise ValueError("snapshot.sources must be a list")
    registry_path = run_root / "P00" / "registry.lock.json"
    registry_raw: object = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry_raw, dict):
        raise ValueError("locked registry must be an object")
    registry = cast(dict[str, object], registry_raw)
    source_registry = cast(
        dict[str, object],
        cast(
            dict[str, object], cast(dict[str, object], registry["data_sources"])["source_registry"]
        )["sources"],
    )
    source_ids: list[str] = []
    for raw in cast(list[object], sources_raw):
        if not isinstance(raw, dict):
            raise ValueError("snapshot source entry must be an object")
        source = cast(dict[str, object], raw)
        source_id = source.get("source_id")
        if not isinstance(source_id, str):
            raise ValueError("snapshot source_id required")
        registered = source_registry.get(source_id)
        if not isinstance(registered, dict):
            raise ValueError(f"snapshot source={source_id}: missing from locked registry")
        registered = cast(dict[str, object], registered)
        if registered.get("role") != "known_case":
            source_ids.append(source_id)

    for source_id in sorted(source_ids):
        _run(
            [
                python,
                "scripts/p01_audit_raw.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
                "--source-id",
                source_id,
            ],
            cwd=project_root,
            env=env,
        )
    if args.through == "P01":
        return 0

    _run(
        [
            python,
            "scripts/p02_build_firm_panel.py",
            "--registry",
            str(registry_path),
            "--run-id",
            args.run_id,
        ],
        cwd=project_root,
        env=env,
    )
    if args.through == "P02":
        return 0

    simple_stages = {
        "P03": "scripts/p03_evidence_ledger.py",
        "P04": "scripts/p04_risk_sets.py",
        "P05": "scripts/p05_measurement_inputs.py",
        "P06": "scripts/p06_observability.py",
        "P07": "scripts/p07_features.py",
    }
    for stage_id, script in simple_stages.items():
        _run(
            [python, script, "--registry", str(registry_path), "--run-id", args.run_id],
            cwd=project_root,
            env=env,
        )
        if args.through == stage_id:
            return 0

    _run(
        [
            python,
            "scripts/p08_build_scenario_registry.py",
            "--registry",
            str(registry_path),
            "--run-id",
            args.run_id,
        ],
        cwd=project_root,
        env=env,
    )
    simulation = cast(dict[str, object], registry["simulation"])
    scenarios = simulation.get("operational_scenarios")
    methods = simulation.get("methods")
    core = simulation.get("core")
    if (
        not isinstance(scenarios, list)
        or not isinstance(methods, list)
        or not isinstance(core, dict)
    ):
        raise ValueError("simulation operational registry is invalid")
    core = cast(dict[str, object], core)
    minimum = int(str(core["minimum_replications"]))
    batch_size = int(str(core["batch_size"]))
    maximum = int(str(core["maximum_replications"]))
    jobs: list[tuple[str, str]] = []
    for raw_scenario in cast(list[object], scenarios):
        if not isinstance(raw_scenario, dict):
            raise ValueError("simulation scenario_id required")
        scenario = cast(dict[str, object], raw_scenario)
        if not isinstance(scenario.get("scenario_id"), str):
            raise ValueError("simulation scenario_id required")
        scenario_id = str(scenario["scenario_id"])
        for method in cast(list[object], methods):
            method_id = str(method)
            jobs.append((scenario_id, method_id))
            for start in range(0, minimum, batch_size):
                count = min(batch_size, minimum - start)
                batch_id = f"{start:06d}-{start + count - 1:06d}"
                _run(
                    [
                        python,
                        "scripts/p08_run_batch.py",
                        "--registry",
                        str(registry_path),
                        "--run-id",
                        args.run_id,
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
                    ],
                    cwd=project_root,
                    env=env,
                )
    while jobs:
        output = _run_capture(
            [
                python,
                "scripts/p08_aggregate_batches.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
                "--check-only",
            ],
            cwd=project_root,
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
            raise RuntimeError("P08 adaptive controller did not return a control report")
        control_raw: object = json.loads(marker)
        if not isinstance(control_raw, dict):
            raise ValueError("P08 adaptive control report must be an object")
        control = cast(dict[str, object], control_raw)
        metric_rows = control.get("metrics")
        if not isinstance(metric_rows, list):
            raise ValueError("P08 adaptive control metrics must be a list")
        grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
        for raw_metric in cast(list[object], metric_rows):
            if not isinstance(raw_metric, dict):
                raise ValueError("P08 adaptive metric must be an object")
            metric = cast(dict[str, object], raw_metric)
            key = (str(metric["scenario_id"]), str(metric["method_id"]))
            grouped.setdefault(key, []).append(metric)
        pending: list[tuple[str, str, int]] = []
        for scenario_id, method_id in jobs:
            job_metrics = grouped.get((scenario_id, method_id), [])
            completed = min(
                (int(str(item["replications"])) for item in job_metrics),
                default=0,
            )
            precision_met = bool(job_metrics) and all(
                item.get("minimum_replications_met") is True and item.get("mcse_target_met") is True
                for item in job_metrics
            )
            if not precision_met and completed < maximum:
                pending.append((scenario_id, method_id, completed))
        if not pending:
            break
        for scenario_id, method_id, start in pending:
            count = min(batch_size, maximum - start)
            batch_id = f"{start:06d}-{start + count - 1:06d}"
            _run(
                [
                    python,
                    "scripts/p08_run_batch.py",
                    "--registry",
                    str(registry_path),
                    "--run-id",
                    args.run_id,
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
                ],
                cwd=project_root,
                env=env,
            )
    _run(
        [
            python,
            "scripts/p08_aggregate_batches.py",
            "--registry",
            str(registry_path),
            "--run-id",
            args.run_id,
        ],
        cwd=project_root,
        env=env,
    )
    if args.through == "P08":
        return 0

    _run(
        [
            python,
            "scripts/p09_splits_weights.py",
            "--registry",
            str(registry_path),
            "--run-id",
            args.run_id,
        ],
        cwd=project_root,
        env=env,
    )
    if args.through == "P09":
        return 0

    folds = cast(dict[str, object], registry["folds"])
    raw_nested = folds.get("fully_nested_outer_years")
    if not isinstance(raw_nested, list):
        raise ValueError("folds.fully_nested_outer_years must be a list")
    outer_folds = [
        str(folds["initial_outer_year"]),
        *[str(value) for value in cast(list[object], raw_nested)],
    ]
    for fold_id in outer_folds:
        _run(
            [
                python,
                "scripts/p10_select_measurement.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
                "--outer-fold",
                fold_id,
            ],
            cwd=project_root,
            env=env,
        )
    if args.through == "P10":
        return 0
    for fold_id in outer_folds:
        _run(
            [
                python,
                "scripts/p11_freeze_models.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
                "--outer-fold",
                fold_id,
            ],
            cwd=project_root,
            env=env,
        )
    if args.through == "P11":
        return 0
    for fold_id in outer_folds:
        _run(
            [
                python,
                "scripts/p12_evaluate.py",
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
                "--outer-fold",
                fold_id,
            ],
            cwd=project_root,
            env=env,
        )
    if args.through == "P12":
        return 0

    final_stages = {
        "P13": "scripts/p13_sensitivity.py",
        "P14": "scripts/p14_gate2.py",
        "P15": "scripts/p15_open_known_cases.py",
        "P16": "scripts/p16_gate3.py",
        "P17": "scripts/p17_build_outputs.py",
    }
    for stage_id, script in final_stages.items():
        _run(
            [python, script, "--registry", str(registry_path), "--run-id", args.run_id],
            cwd=project_root,
            env=env,
        )
        if args.through == stage_id:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
