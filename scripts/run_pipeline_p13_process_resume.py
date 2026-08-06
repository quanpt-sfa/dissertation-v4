"""Run the immutable pipeline with the registered P13 process compatibility verifier."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    import run_pipeline as pipeline_runner
    from core.resume_p13_v5 import verify_resume_inputs_v5

    pipeline_runner.verify_resume_inputs = verify_resume_inputs_v5
    return pipeline_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
