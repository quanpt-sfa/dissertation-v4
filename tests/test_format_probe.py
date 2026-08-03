from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "src" / "sensitivity" / "parallel_refits.py",
    ROOT / "tests" / "stages" / "test_p13_parallel_refits.py",
]
original = {target: target.read_text(encoding="utf-8") for target in TARGETS}
subprocess.run([sys.executable, "-m", "ruff", "format", *map(str, TARGETS)], check=True)
parts: list[str] = []
for target in TARGETS:
    formatted = target.read_text(encoding="utf-8")
    parts.extend(
        difflib.unified_diff(
            original[target].splitlines(),
            formatted.splitlines(),
            fromfile=f"a/{target.relative_to(ROOT).as_posix()}",
            tofile=f"b/{target.relative_to(ROOT).as_posix()}",
            lineterm="",
        )
    )
raise RuntimeError("\n".join(parts))
