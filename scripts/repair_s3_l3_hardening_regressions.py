"""Repair compatibility regressions introduced by the one-time S3/L3 migration.

This repository-maintenance script is idempotent. It preserves the production
restriction to the configured primary source set while keeping generic service
callers and focused unit tests backward compatible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


class RepairError(RuntimeError):
    """Raised when a known regression pattern cannot be repaired safely."""


def _read(path: Path) -> str:
    if not path.is_file():
        raise RepairError(f"required file is missing: {path}")
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str, *, apply: bool) -> bool:
    current = _read(path)
    if current == text:
        return False
    if apply:
        path.write_text(text, encoding="utf-8")
    return True


def _replace_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if new in text:
        return text, False
    count = text.count(old)
    if count != 1:
        raise RepairError(f"{label}: expected one repair anchor, found {count}")
    return text.replace(old, new, 1), True


def _replace_remaining_once(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    """Replace one remaining bad block without whole-file false positives."""

    count = text.count(old)
    if count == 0:
        return text, False
    if count != 1:
        raise RepairError(f"{label}: expected at most one remaining repair anchor, found {count}")
    return text.replace(old, new, 1), True


def _repair_measurement_profiles(path: Path, *, apply: bool) -> bool:
    text = _read(path)

    # The migration creates two similar comprehensions at different indentation
    # levels. Both must tolerate source_profiles=None or partial profile maps.
    config_old = '''        source_profiles={
            source_id: (source_profiles or {})[source_id]
            for source_id in primary_source_ids
        },
'''
    config_new = '''        source_profiles={
            source_id: (source_profiles or {})[source_id]
            for source_id in primary_source_ids
            if source_id in (source_profiles or {})
        },
'''
    text, changed_config = _replace_remaining_once(
        text,
        config_old,
        config_new,
        "optional source-profile filtering in L2 configuration",
    )

    score_old = '''            source_profiles={
                source_id: (source_profiles or {})[source_id]
                for source_id in primary_source_ids
            },
'''
    score_new = '''            source_profiles={
                source_id: (source_profiles or {})[source_id]
                for source_id in primary_source_ids
                if source_id in (source_profiles or {})
            },
'''
    text, changed_score = _replace_remaining_once(
        text,
        score_old,
        score_new,
        "optional source-profile filtering in L2 score binding",
    )

    return _write(path, text, apply=apply) or changed_config or changed_score


def _repair_l3_binding_signature(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    old_signature = '''def _l3_bindings(
    *,
    registry: dict[str, object],
    priors_by_profile: dict[str, Any],
    allowed_source_ids: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
'''
    new_signature = '''def _l3_bindings(
    *,
    registry: dict[str, object],
    priors_by_profile: dict[str, Any],
    allowed_source_ids: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
'''
    text, changed_signature = _replace_once(
        text,
        old_signature,
        new_signature,
        "optional allowed-source binding parameter",
    )
    old_allowed = "    allowed = set(allowed_source_ids)\n"
    new_allowed = (
        "    allowed = set(sources) if allowed_source_ids is None else set(allowed_source_ids)\n"
    )
    text, changed_allowed = _replace_once(
        text,
        old_allowed,
        new_allowed,
        "default all logical sources for legacy callers",
    )
    return _write(path, text, apply=apply) or changed_signature or changed_allowed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    apply = bool(args.apply)
    operations = {
        "measurement_optional_profiles": _repair_measurement_profiles(
            root / "src" / "measurement" / "service.py",
            apply=apply,
        ),
        "p10_backward_compatible_l3_bindings": _repair_l3_binding_signature(
            root / "scripts" / "p10_select_measurement.py",
            apply=apply,
        ),
    }
    result = {
        "mode": "apply" if apply else "check",
        "repo_root": str(root),
        "changes_required": [name for name, changed in operations.items() if changed],
        "already_repaired": [name for name, changed in operations.items() if not changed],
    }
    print("S3_L3_REGRESSION_REPAIR_JSON=" + json.dumps(result, sort_keys=True))
    if args.check and result["changes_required"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
