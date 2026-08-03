from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "p16_gate3.py"
original = TARGET.read_text(encoding="utf-8")
subprocess.run([sys.executable, "-m", "ruff", "format", str(TARGET)], check=True)
formatted = TARGET.read_text(encoding="utf-8")
diff = "\n".join(
    difflib.unified_diff(
        original.splitlines(),
        formatted.splitlines(),
        fromfile="a/scripts/p16_gate3.py",
        tofile="b/scripts/p16_gate3.py",
        lineterm="",
    )
)
raise RuntimeError(diff)
