"""Restartable atomic publication for the P00 lock directory."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from .errors import ConfigurationError


def publish_p00(
    final_directory: Path,
    files: dict[str, object],
    validate: Callable[[str, object], None],
    receipt: dict[str, object],
) -> None:
    """Stage, validate, hash, receipt, then atomically publish without overwrite."""
    success_name = "_SUCCESS.json"
    if final_directory.exists():
        receipt = final_directory / success_name
        if receipt.is_file():
            raise ConfigurationError(f"output={final_directory}: successful run is immutable")
        quarantine = final_directory.with_name(final_directory.name + ".quarantine")
        if quarantine.exists():
            shutil.rmtree(quarantine)
        final_directory.replace(quarantine)
    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p00-staging-", dir=final_directory.parent))
    try:
        hashes: dict[str, str] = {}
        for name, content in files.items():
            validate(name, content)
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            rendered = (
                json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
                if not isinstance(content, str)
                else content
            )
            destination.write_text(rendered, encoding="utf-8")
            hashes[name] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        manifest_path = staging / "job_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["output_hashes"] = hashes
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
        validate(success_name, receipt)
        (staging / success_name).write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        staging.replace(final_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
