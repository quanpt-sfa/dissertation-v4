"""Atomic P00 directory publication with detached manifest hash."""

from __future__ import annotations

import hashlib
import json
import os
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
    success_name = "_SUCCESS.json"
    if final_directory.exists():
        if (final_directory / success_name).is_file():
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
                content
                if isinstance(content, str)
                else json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
            )
            encoded = rendered.encode("utf-8")
            destination.write_bytes(encoded)
            hashes[name] = hashlib.sha256(encoded).hexdigest()

        raw_manifest = files.get("job_manifest.json")
        if not isinstance(raw_manifest, dict):
            raise ConfigurationError("job_manifest.json is required")
        manifest = dict(cast(dict[str, Any], raw_manifest))
        manifest["output_hashes"] = hashes
        validate("job_manifest.json", manifest)
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
        (staging / "job_manifest.json").write_bytes(manifest_text.encode("utf-8"))

        success = dict(receipt)
        success["manifest_hash"] = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        validate(success_name, success)
        (staging / success_name).write_bytes(
            json.dumps(success, indent=2, sort_keys=True).encode("utf-8")
        )
        if os.name != "nt":
            descriptor = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        staging.replace(final_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
