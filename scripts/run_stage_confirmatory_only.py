"""Run a pipeline stage with ``outer_fold_ids`` restricted to nested folds.

This wrapper is reserved for logged implementation-error recovery.  It does not
change the locked fold registry; it only prevents post-P09 stages from treating
the separately designated initial fold as a confirmatory outer-test fold.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import core.pipeline as pipeline


def _confirmatory_outer_fold_ids(
    registry: Any,
    *,
    include_initial: bool = True,
) -> list[str]:
    """Return only ``fully_nested_outer_years`` regardless of caller default."""
    del include_initial
    return _ORIGINAL_OUTER_FOLD_IDS(registry, include_initial=False)


_ORIGINAL_OUTER_FOLD_IDS = pipeline.outer_fold_ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    args, passthrough = parser.parse_known_args()

    target = (PROJECT_ROOT / args.script).resolve()
    if target == PROJECT_ROOT or PROJECT_ROOT not in target.parents:
        raise ValueError("stage script must remain inside the project root")
    if not target.is_file():
        raise FileNotFoundError(target)

    pipeline.outer_fold_ids = _confirmatory_outer_fold_ids
    sys.argv = [str(target), *passthrough]
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(str(exc.code), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
