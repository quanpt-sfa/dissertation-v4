"""Diagnose which predictors drive P13 HOSE/HNX common-support loss."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensitivity.domain_support_report import run_domain_support_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--baseline-tolerance", type=float, default=1e-10)
    parser.add_argument("--only-failing-scenarios", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.baseline_tolerance < 0:
        parser.error("--baseline-tolerance must be nonnegative")

    run_domain_support_diagnostics(
        registry_path=args.registry,
        run_id=args.run_id,
        output_dir=args.output_dir,
        workers=args.workers,
        baseline_tolerance=args.baseline_tolerance,
        only_failing_scenarios=args.only_failing_scenarios,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
