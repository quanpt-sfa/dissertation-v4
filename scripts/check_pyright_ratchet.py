#!/usr/bin/env python3
"""Run Pyright once and enforce the locked diagnostic ratchet."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.pyright_ratchet import (  # noqa: E402
    BaselineEntry,
    Diagnostic,
    baseline_from_diagnostics,
    diagnostics_from_pyright,
    evaluate_ratchet,
    normalize_changed_files,
)

DEFAULT_BASELINE = ROOT / "config" / "quality" / "pyright_baseline.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Changed files that must have zero Pyright errors")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--diff-base", help="Git base SHA/ref used to discover changed Python files")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--generated-from", default="")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    diagnostics = _run_pyright()
    baseline_path = _resolve(cast(Path, args.baseline))
    if cast(bool, args.write_baseline):
        generated_from = cast(str, args.generated_from) or _git_output("rev-parse", "HEAD")
        document = baseline_from_diagnostics(diagnostics, generated_from)
        _write_json(baseline_path, document)
        print(
            "Pyright baseline written: "
            f"errors={document['total_errors']} path={baseline_path.relative_to(ROOT)}"
        )
        return 0

    if not baseline_path.is_file():
        raise SystemExit(f"Pyright baseline is missing: {baseline_path}")
    baseline_document = cast(object, json.loads(baseline_path.read_text(encoding="utf-8")))

    changed_paths = list(cast(list[str], args.files))
    diff_base = cast(str | None, args.diff_base)
    if diff_base:
        changed_paths.extend(_changed_files(diff_base))
    changed_files = normalize_changed_files(changed_paths, ROOT)
    result = evaluate_ratchet(diagnostics, baseline_document, changed_files)

    report = {
        "schema_version": 1,
        "diff_base": diff_base,
        "changed_python_files": sorted(changed_files),
        **result,
    }
    report_path = cast(Path | None, args.report)
    if report_path is not None:
        _write_json(_resolve(report_path), report)

    print(
        "Pyright ratchet: "
        f"current={result['current_error_count']} "
        f"baseline={result['baseline_error_count']} "
        f"new={result['new_error_count']} "
        f"removed={result['removed_error_count']} "
        f"changed_file_errors={result['changed_file_error_count']}"
    )
    if result["new_errors"]:
        _print_entries("New diagnostics above baseline", result["new_errors"])
    if result["changed_file_errors"]:
        _print_entries("Diagnostics in changed Python files", result["changed_file_errors"])
    return 0 if result["passed"] else 1


def _run_pyright() -> list[Diagnostic]:
    executable = shutil.which("pyright")
    if executable is None:
        raise SystemExit("pyright executable is not available")
    completed = subprocess.run(
        [executable, "--outputjson"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if not completed.stdout.strip():
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise SystemExit(f"Pyright did not return JSON output: {detail}")
    try:
        payload = cast(object, json.loads(completed.stdout))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid Pyright JSON output: {exc}") from exc
    if completed.returncode not in {0, 1}:
        raise SystemExit(f"Pyright execution failed with exit code {completed.returncode}")
    return diagnostics_from_pyright(payload, ROOT)


def _changed_files(diff_base: str) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{diff_base}...HEAD",
            "--",
            "*.py",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown git diff failure"
        raise SystemExit(f"Unable to determine changed Python files: {detail}")
    return [line for line in completed.stdout.splitlines() if line]


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _print_entries(title: str, entries: list[BaselineEntry]) -> None:
    print(f"{title} (showing up to 20):", file=sys.stderr)
    for entry in entries[:20]:
        print(
            f"  {entry['path']} [{entry['rule']}] x{entry['count']}: {entry['message']}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
