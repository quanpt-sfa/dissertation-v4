"""Temporary CI probe that exposes Ruff's exact formatting diff."""

from __future__ import annotations

import difflib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (
    Path("src/evidence/artifact_adapter.py"),
    Path("tests/stages/test_s3_placeholder_document_ids.py"),
)


def _formatted_diff() -> str:
    ruff = shutil.which("ruff")
    if ruff is None:
        raise RuntimeError("Ruff executable is unavailable")

    chunks: list[str] = []
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        for relative in TARGETS:
            source = ROOT / relative
            target = temporary_root / relative.name
            original = source.read_text(encoding="utf-8")
            target.write_text(original, encoding="utf-8")
            subprocess.run(
                [ruff, "format", "--config", str(ROOT / "pyproject.toml"), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            formatted = target.read_text(encoding="utf-8")
            chunks.extend(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    formatted.splitlines(keepends=True),
                    fromfile=str(relative),
                    tofile=str(relative),
                )
            )
    return "".join(chunks)


raise RuntimeError("RUFF_FORMAT_DIFF_BEGIN\n" + _formatted_diff() + "RUFF_FORMAT_DIFF_END")
