"""Finalize primary-source binding after the main S3/L3 hardening migration.

This small follow-up is separated so the P05 feasibility pilot is guaranteed to
use exactly the same primary source set as L2 and fold-local L3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class FinalizationError(RuntimeError):
    pass


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise FinalizationError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1), True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    path = root / "scripts" / "p05_measurement_inputs.py"
    text = path.read_text(encoding="utf-8")
    old = '''    _execute_l3_pilot(\n        matrices=result.matrices,\n        capability=result.l3_capability,\n        source_channels=expected_sources,\n        source_profiles=source_profiles,\n'''
    new = '''    primary_expected_sources = {\n        source_id: expected_sources[source_id] for source_id in primary_measurement_sources\n    }\n    primary_source_profiles = {\n        source_id: source_profiles[source_id] for source_id in primary_measurement_sources\n    }\n    _execute_l3_pilot(\n        matrices=result.matrices,\n        capability=result.l3_capability,\n        source_channels=primary_expected_sources,\n        source_profiles=primary_source_profiles,\n'''
    text, changed = _replace_once(text, old, new, "P05 L3 feasibility primary source binding")
    if args.apply and changed:
        path.write_text(text, encoding="utf-8")
    result = {
        "mode": "apply" if args.apply else "check",
        "changed": changed,
        "path": str(path),
        "primary_s3_endpoint": "S3_CONTENT",
    }
    print("S3_L3_FINALIZATION_JSON=" + json.dumps(result, sort_keys=True))
    return 2 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
