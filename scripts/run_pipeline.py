"""Single-command immutable production runner from snapshot through P17."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.resume import artifact_complete, json_object, verify_resume_inputs


def _unit_complete(
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifacts: list[tuple[str, dict[str, str]]],
) -> bool:
    return bool(artifacts) and all(
        artifact_complete(registry, run_root, protocol_hash, artifact_id, coordinates)
        for artifact_id, coordinates in artifacts
    )


def _step_outputs(
    registry: dict[str, object], stage_id: str, coordinates: dict[str, str]
) -> list[tuple[str, dict[str, str]]]:
    raw_steps = registry.get("steps")
    raw_artifacts = registry.get("artifacts")
    if not isinstance(raw_steps, dict) or not isinstance(raw_artifacts, dict):
        raise ValueError("locked step/artifact catalogs are unavailable")
    raw_step = cast(dict[str, object], raw_steps).get(stage_id)
    if not isinstance(raw_step, dict):
        raise ValueError(f"stage={stage_id}: absent from locked catalog")
    writes = cast(dict[str, object], raw_step).get("writes")
    if not isinstance(writes, list):
        raise ValueError(f"stage={stage_id}: writes must be a list")
    outputs: list[tuple[str, dict[str, str]]] = []
    for value in cast(list[object], writes):
        artifact_id = str(value)
        raw_artifact = cast(dict[str, object], raw_artifacts).get(artifact_id)
        if not isinstance(raw_artifact, dict):
            raise ValueError(f"artifact={artifact_id}: absent from locked catalog")
        names = cast(dict[str, object], raw_artifact).get("coordinates")
        if isinstance(names, list) and {str(name) for name in cast(list[object], names)} == set(
            coordinates
        ):
            outputs.append((artifact_id, dict(coordinates)))
    return outputs


def _run_unit(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifacts: list[tuple[str, dict[str, str]]],
) -> None:
    if _unit_complete(registry, run_root, protocol_hash, artifacts):
        print(f"= verified complete; skip {' '.join(command[1:3])}", flush=True)
        return
    _run(command, cwd=cwd, env=env)
    if not _unit_complete(registry, run_root, protocol_hash, artifacts):
        raise RuntimeError(f"unit completed without its full artifact contract: {command}")


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same immutable run after verifying code, config, snapshot, and artifacts.",
    )
    args = parser.parse_args()

    project_root = args.config.resolve().parent.parent
    if not args.allow_dirty:
        _require_clean_tree(project_root)
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / args.run_id
    if run_root.exists() and not args.resume:
        raise FileExistsError(f"run root already exists and is immutable: {run_root}")

    snapshot_path = run_root / "SNAPSHOT" / "data_snapshot.json"
    env = dict(os.environ)
    env["DISSERTATION_RAW_ROOT"] = str(raw_root)
    python = sys.executable

    registry_path = run_root / "P00" / "registry.lock.json"
    if args.resume:
        if not snapshot_path.is_file() or not registry_path.is_file():
            raise RuntimeError("resume requires a completed snapshot and P00 lock")
    else:
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

    snapshot = json_object(snapshot_path, "snapshot")
    sources_raw = snapshot.get("sources")
    if not isinstance(sources_raw, list):
        raise ValueError("snapshot.sources must be a list")
    registry = json_object(registry_path, "locked registry")
    protocol_hash = (run_root / "P00" / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    if args.resume:
        verify_resume_inputs(project_root, raw_root, snapshot, run_root)
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
        source_coordinates = {"source_id": source_id}
        _run_unit(
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
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=[("raw_audit", source_coordinates)],
        )
    if args.through == "P01":
        return 0

    _run_unit(
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
        registry=registry,
        run_root=run_root,
        protocol_hash=protocol_hash,
        artifacts=_step_outputs(registry, "P02", {}),
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
        _run_unit(
            [python, script, "--registry", str(registry_path), "--run-id", args.run_id],
            cwd=project_root,
            env=env,
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=_step_outputs(registry, stage_id, {}),
        )
        if args.through == stage_id:
            return 0

    p08_command = [
        python,
        "-W",
        "ignore::FutureWarning",
        "scripts/p08_profiled_orchestrator.py",
        "--registry",
        str(registry_path),
        "--run-id",
        args.run_id,
        "--workers",
        str(max(1, int(os.environ.get("P08_WORKERS", "1")))),
    ]
    if args.resume:
        p08_command.append("--resume")

    _run_unit(
        p08_command,
        cwd=project_root,
        env=env,
        registry=registry,
        run_root=run_root,
        protocol_hash=protocol_hash,
        artifacts=[("mcse_report", {})],
    )

    report = json_object(run_root / "P08" / "mcse_report.json", "P08 report")
    if report.get("status") == "FAIL":
        reason = report.get("reason_code", "UNKNOWN_REASON")
        raise RuntimeError(
            f"P08 precision target or methodology contract was not met; P09 is blocked. Reason: {reason}"
        )

    if args.through == "P08":
        return 0

    folds = cast(dict[str, object], registry["folds"])
    raw_nested = folds.get("fully_nested_outer_years")
    if not isinstance(raw_nested, list):
        raise ValueError("folds.fully_nested_outer_years must be a list")
    outer_folds = [
        str(folds["initial_outer_year"]),
        *[str(value) for value in cast(list[object], raw_nested)],
    ]
    p09_artifacts = _step_outputs(registry, "P09", {})
    for fold_id in outer_folds:
        p09_artifacts.extend(
            [
                ("fold_aware_weights", {"fold_id": fold_id}),
                ("weight_diagnostics", {"fold_id": fold_id}),
            ]
        )
    _run_unit(
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
        registry=registry,
        run_root=run_root,
        protocol_hash=protocol_hash,
        artifacts=p09_artifacts,
    )
    if args.through == "P09":
        return 0

    for fold_id in outer_folds:
        fold_coordinates = {"outer_fold": fold_id}
        _run_unit(
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
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=_step_outputs(registry, "P10", fold_coordinates),
        )
    if args.through == "P10":
        return 0
    for fold_id in outer_folds:
        fold_coordinates = {"outer_fold": fold_id}
        _run_unit(
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
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=_step_outputs(registry, "P11", fold_coordinates),
        )
    if args.through == "P11":
        return 0
    for fold_id in outer_folds:
        fold_coordinates = {"outer_fold": fold_id}
        _run_unit(
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
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=_step_outputs(registry, "P12", fold_coordinates),
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
        _run_unit(
            [python, script, "--registry", str(registry_path), "--run-id", args.run_id],
            cwd=project_root,
            env=env,
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=_step_outputs(registry, stage_id, {}),
        )
        if args.through == stage_id:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
