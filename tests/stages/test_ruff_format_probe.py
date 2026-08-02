"""Temporary CI probe for the exact Ruff formatter patch."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = subprocess.run(
    [
        "ruff",
        "format",
        "--diff",
        "src/features/panel_source.py",
        "tests/stages/test_p07_final_wide_derivation.py",
    ],
    cwd=ROOT,
    check=False,
    capture_output=True,
    text=True,
)
raise RuntimeError(f"RUFF FORMAT DIFF\n{RESULT.stdout}\n{RESULT.stderr}")
