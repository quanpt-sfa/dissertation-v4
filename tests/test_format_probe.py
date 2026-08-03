from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "src" / "sensitivity" / "parallel_refits.py",
    ROOT / "tests" / "stages" / "test_p13_parallel_refits.py",
]
subprocess.run([sys.executable, "-m", "ruff", "format", *map(str, TARGETS)], check=True)
for target in TARGETS:
    payload = base64.b64encode(target.read_bytes()).decode("ascii")
    print(f"FORMAT_PROBE::{target.relative_to(ROOT).as_posix()}::{payload}")
raise RuntimeError("FORMAT_PROBE_COMPLETE")
