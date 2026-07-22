"""Resume a locked run after P09 using confirmatory folds only.

The initial outer year is intentionally retained in P09 split/weight artifacts
but excluded from P10-P17 confirmatory execution.  This runner is for logged
recovery of runs created before that orchestration distinction was enforced.
It never rewrites P00, P08 batches, replication budgets, seeds, or fold roles.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.resume import artifact_complete, json_object


def _step_outputs(
    registry: dict[str, object],
    stage_id: str,
    coordinates: dict[str, str],
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
        if isinstance(names, list) and {
            str(name) for name in cast(list[object], names)
        } == set(coordinates):
            outputs.append((artifact_id, dict(coordinates)))
    return outputs


def _unit_complete(
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifacts: list[tuple[str, dict[str, str]]],
) -> bool:
    return bool(artifacts) and all(
        artifact_complete(
            registry,
            run_root,
            protocol_hash,
            artifact_id,
            coordinates,
        )
        for artifact_id, coordinates in artifacts
    )


def _run_unit(
    command: list[str],
    *,
    registry: dict[str, object],
    run_root: Path,
    protocol_hash: str,
    artifacts: list[tuple[str, dict[str, str]]],
) -> None:
    if _unit_complete(registry, run_root, protocol_hash, artifacts):
        print(f"= verified complete; skip {' '.join(command[1:3])}", flush=True)
        return
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if not _unit_complete(registry, run_root, protocol_hash, artifacts):
        raise RuntimeError(
            f"unit completed without its full artifact contract: {command}"
        )


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_receipt(
    *,
    run_root: Path,
    run_id: str,
    initial_fold: str,
    confirmatory_folds: list[str],
    completed_through: str,
) -> Path:
    source_manifest = json_object(
        run_root / "P00" / "source_config_manifest.json",
        "source manifest",
    )
    correction_root = (
        run_root.parent.parent
        / "corrections"
        / run_id
        / "post_p09_confirmatory_fold_recovery"
    )
    correction_root.mkdir(parents=True, exist_ok=True)
    receipt_path = correction_root / "receipt.json"
    receipt = {
        "run_id": run_id,
        "correction_type": "LOGGED_POST_P09_CONFIRMATORY_FOLD_ORCHESTRATION_FIX",
        "locked_git_commit": source_manifest.get("git_commit"),
        "recovery_git_commit": _git_head(),
        "initial_fold_retained_for_p09": initial_fold,
        "initial_fold_excluded_from_p10_p17": True,
        "confirmatory_folds": confirmatory_folds,
        "completed_through": completed_through,
        "p00_rewritten": False,
        "p08_batches_recomputed": False,
        "replication_ids_changed": False,
        "rng_seeds_changed": False,
        "fold_roles_changed": False,
        "outer_outcomes_used_for_model_selection": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--through",
        choices=[f"P{value:02d}" for value in range(10, 18)],
        default="P17",
    )
    args = parser.parse_args()

    registry_path = args.registry.expanduser().resolve()
    run_root = registry_path.parents[1]
    if run_root.name != args.run_id:
        raise ValueError(
            f"run-id={args.run_id} does not match registry run directory {run_root.name}"
        )

    registry = json_object(registry_path, "locked registry")
    protocol_hash = (
        run_root / "P00" / "protocol_hash.txt"
    ).read_text(encoding="utf-8").strip()
    folds = registry.get("folds")
    if not isinstance(folds, dict):
        raise ValueError("locked folds registry is unavailable")
    initial_raw = cast(dict[str, object], folds).get("initial_outer_year")
    nested_raw = cast(dict[str, object], folds).get("fully_nested_outer_years")
    if not isinstance(initial_raw, int):
        raise ValueError("folds.initial_outer_year must be an integer")
    if not isinstance(nested_raw, list) or not all(
        isinstance(value, int) for value in nested_raw
    ):
        raise ValueError("folds.fully_nested_outer_years must contain integers")

    initial_fold = str(initial_raw)
    confirmatory_folds = [str(value) for value in cast(list[int], nested_raw)]
    if not confirmatory_folds:
        raise RuntimeError("no fully nested confirmatory folds are locked")
    if initial_fold in confirmatory_folds:
        raise RuntimeError("initial fold must remain separate from confirmatory folds")

    mcse = json_object(run_root / "P08" / "mcse_report.json", "P08 report")
    if mcse.get("status") != "PASS" or mcse.get("precision_target_met") is not True:
        raise RuntimeError("post-P09 recovery requires a verified PASS P08 report")

    python = sys.executable
    stage_order = [f"P{value:02d}" for value in range(10, 18)]

    for stage_id, script in (
        ("P10", "scripts/p10_select_measurement.py"),
        ("P11", "scripts/p11_freeze_models.py"),
        ("P12", "scripts/p12_evaluate.py"),
    ):
        for fold_id in confirmatory_folds:
            coordinates = {"outer_fold": fold_id}
            _run_unit(
                [
                    python,
                    script,
                    "--registry",
                    str(registry_path),
                    "--run-id",
                    args.run_id,
                    "--outer-fold",
                    fold_id,
                ],
                registry=registry,
                run_root=run_root,
                protocol_hash=protocol_hash,
                artifacts=_step_outputs(registry, stage_id, coordinates),
            )
        if args.through == stage_id:
            receipt = _write_receipt(
                run_root=run_root,
                run_id=args.run_id,
                initial_fold=initial_fold,
                confirmatory_folds=confirmatory_folds,
                completed_through=stage_id,
            )
            print(f"POST_P09_RECOVERY_RECEIPT={receipt}")
            return 0

    wrapper = "scripts/run_stage_confirmatory_only.py"
    final_stages = (
        ("P13", "scripts/p13_sensitivity.py", True),
        ("P14", "scripts/p14_gate2.py", False),
        ("P15", "scripts/p15_open_known_cases.py", True),
        ("P16", "scripts/p16_gate3.py", True),
        ("P17", "scripts/p17_build_outputs.py", True),
    )
    for stage_id, script, needs_wrapper in final_stages:
        command = [python]
        if needs_wrapper:
            command.extend([wrapper, "--script", script])
        else:
            command.append(script)
        command.extend(
            [
                "--registry",
                str(registry_path),
                "--run-id",
                args.run_id,
            ]
        )
        _run_unit(
            command,
            registry=registry,
            run_root=run_root,
            protocol_hash=protocol_hash,
            artifacts=_step_outputs(registry, stage_id, {}),
        )
        if args.through == stage_id:
            receipt = _write_receipt(
                run_root=run_root,
                run_id=args.run_id,
                initial_fold=initial_fold,
                confirmatory_folds=confirmatory_folds,
                completed_through=stage_id,
            )
            print(f"POST_P09_RECOVERY_RECEIPT={receipt}")
            return 0

    raise RuntimeError(f"unreachable through stage: {args.through}; order={stage_order}")


if __name__ == "__main__":
    raise SystemExit(main())
