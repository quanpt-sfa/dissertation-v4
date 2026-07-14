"""Atomic, no-overwrite P0 artifact publication."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .errors import ConfigurationError


def publish_p00(final_directory: Path, files: dict[str, object]) -> None:
    """Write a complete P0 directory privately, rename it, then write success last."""
    if final_directory.exists():
        raise ConfigurationError(f"output={final_directory}: existing P0 run will not be overwritten; choose a new run ID")
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".p00-", dir=final_directory.parent))
    try:
        for name, content in files.items():
            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True) if not isinstance(content, str) else content, encoding="utf-8")
        shutil.move(str(temporary), str(final_directory))
        (final_directory / "_SUCCESS.json").write_text(json.dumps({"status": "SUCCESS"}, sort_keys=True), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        shutil.rmtree(temporary, ignore_errors=True)
        raise
