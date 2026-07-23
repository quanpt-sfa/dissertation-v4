"""Synchronize the S3/L3 migration checker with the final compatibility-safe patch.

The production migration was repaired after its first application. This one-time
repository-maintenance helper updates the migration generator itself so that its
``--check`` mode recognizes the repaired source as the canonical hardened state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class CheckerRepairError(RuntimeError):
    """Raised when the migration generator is not in an expected state."""


def _read(path: Path) -> str:
    if not path.is_file():
        raise CheckerRepairError(f"required file is missing: {path}")
    return path.read_text(encoding="utf-8")


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise CheckerRepairError(f"{label}: expected one checker anchor, found {count}")
    return text.replace(old, new, 1), True


def _repair(path: Path, *, apply: bool) -> dict[str, bool]:
    text = _read(path)

    old_l2_score = (
        '        "            source_profiles={\\n"\n'
        '        "                source_id: (source_profiles or {})[source_id]\\n"\n'
        '        "                for source_id in primary_source_ids\\n"\n'
        '        "            },\\n",\n'
    )
    new_l2_score = (
        '        "            source_profiles={\\n"\n'
        '        "                source_id: (source_profiles or {})[source_id]\\n"\n'
        '        "                for source_id in primary_source_ids\\n"\n'
        '        "                if source_id in (source_profiles or {})\\n"\n'
        '        "            },\\n",\n'
    )
    text, l2_changed = _replace_once(
        text,
        old_l2_score,
        new_l2_score,
        "compatibility-safe L2 score binding generator",
    )

    old_signature = '        "    allowed_source_ids: list[str],\\n"\n'
    new_signature = '        "    allowed_source_ids: list[str] | None = None,\\n"\n'
    text, signature_changed = _replace_once(
        text,
        old_signature,
        new_signature,
        "backward-compatible P10 binding signature generator",
    )

    old_allowed = '        "    allowed = set(allowed_source_ids)\\n"\n'
    new_allowed = '        "    allowed = set(sources) if allowed_source_ids is None else set(allowed_source_ids)\\n"\n'
    text, allowed_changed = _replace_once(
        text,
        old_allowed,
        new_allowed,
        "backward-compatible P10 allowed-source generator",
    )

    if apply and text != _read(path):
        path.write_text(text, encoding="utf-8")

    return {
        "l2_score_generator": l2_changed,
        "p10_signature_generator": signature_changed,
        "p10_allowed_source_generator": allowed_changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    path = root / "scripts" / "apply_s3_l3_production_hardening.py"
    operations = _repair(path, apply=bool(args.apply))
    changes_required = [name for name, changed in operations.items() if changed]
    result = {
        "mode": "apply" if args.apply else "check",
        "path": str(path),
        "changes_required": changes_required,
        "already_synchronized": [name for name, changed in operations.items() if not changed],
    }
    print("S3_L3_MIGRATION_CHECKER_REPAIR_JSON=" + json.dumps(result, sort_keys=True))
    if args.check and changes_required:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
