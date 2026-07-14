"""Restartable atomic publication for the P00 lock directory."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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
        success_path = final_directory / success_name
        if success_path.is_file():
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
            if name == "job_manifest.json":
                continue
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
        manifest_raw = files.get("job_manifest.json")
        if not isinstance(manifest_raw, dict):
            raise ConfigurationError("job_manifest.json: required for P00 publication")
        manifest = dict(cast(dict[str, Any], manifest_raw))
        manifest["output_hashes"] = hashes
        validate("job_manifest.json", manifest)
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
        (staging / "job_manifest.json").write_text(manifest_text, encoding="utf-8")
        receipt = dict(receipt)
        receipt["manifest_hash"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        validate(success_name, receipt)
        (staging / success_name).write_text(
            json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
        staging.replace(final_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
