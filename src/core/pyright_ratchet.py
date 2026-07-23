"""Stable baseline and changed-file contracts for Pyright diagnostics."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TypedDict, cast


class Diagnostic(TypedDict):
    path: str
    rule: str
    message: str


class BaselineEntry(TypedDict):
    path: str
    rule: str
    message: str
    count: int


class BaselineDocument(TypedDict):
    schema_version: int
    generated_from: str
    strict_since_commit: str
    total_errors: int
    diagnostics: list[BaselineEntry]


class RatchetResult(TypedDict):
    passed: bool
    current_error_count: int
    baseline_error_count: int
    new_error_count: int
    removed_error_count: int
    changed_file_error_count: int
    new_errors: list[BaselineEntry]
    removed_errors: list[BaselineEntry]
    changed_file_errors: list[BaselineEntry]


Fingerprint = tuple[str, str, str]


def diagnostics_from_pyright(payload: object, repo_root: Path) -> list[Diagnostic]:
    """Parse Pyright JSON output into repository-relative error fingerprints."""
    if not isinstance(payload, dict):
        raise ValueError("Pyright output must be a JSON object")
    raw_diagnostics = cast(dict[str, object], payload).get("generalDiagnostics")
    if not isinstance(raw_diagnostics, list):
        raise ValueError("Pyright output is missing generalDiagnostics")

    diagnostics: list[Diagnostic] = []
    root = repo_root.resolve()
    for raw_item in cast(list[object], raw_diagnostics):
        if not isinstance(raw_item, dict):
            raise ValueError("Pyright diagnostic entries must be objects")
        item = cast(dict[str, object], raw_item)
        if item.get("severity") != "error":
            continue
        raw_file = item.get("file")
        raw_message = item.get("message")
        raw_rule = item.get("rule")
        if not isinstance(raw_file, str) or not isinstance(raw_message, str):
            raise ValueError("Pyright errors require file and message strings")
        diagnostics.append(
            {
                "path": _relative_path(raw_file, root),
                "rule": raw_rule if isinstance(raw_rule, str) and raw_rule else "<unclassified>",
                "message": " ".join(raw_message.split()),
            }
        )
    return diagnostics


def baseline_from_diagnostics(
    diagnostics: list[Diagnostic],
    generated_from: str,
    strict_since_commit: str,
) -> BaselineDocument:
    counts = _diagnostic_counts(diagnostics)
    return {
        "schema_version": 2,
        "generated_from": generated_from,
        "strict_since_commit": strict_since_commit,
        "total_errors": sum(counts.values()),
        "diagnostics": _entries(counts),
    }


def baseline_cutover(document: object) -> str:
    """Return the commit after which changed-file strictness becomes mandatory."""
    if not isinstance(document, dict):
        raise ValueError("Pyright baseline must be a JSON object")
    baseline = cast(dict[str, object], document)
    if baseline.get("schema_version") != 2:
        raise ValueError("Unsupported Pyright baseline schema_version")
    generated_from = baseline.get("generated_from")
    strict_since_commit = baseline.get("strict_since_commit")
    if not isinstance(generated_from, str) or not generated_from:
        raise ValueError("Pyright baseline generated_from must be locked")
    if not isinstance(strict_since_commit, str) or not strict_since_commit:
        raise ValueError("Pyright baseline strict_since_commit must be locked")
    return strict_since_commit


def baseline_counts(document: object) -> Counter[Fingerprint]:
    """Validate a baseline document and return its multiset of fingerprints."""
    baseline_cutover(document)
    baseline = cast(dict[str, object], document)
    raw_entries = baseline.get("diagnostics")
    if not isinstance(raw_entries, list):
        raise ValueError("Pyright baseline is missing diagnostics")

    counts: Counter[Fingerprint] = Counter()
    for raw_entry in cast(list[object], raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError("Pyright baseline entries must be objects")
        entry = cast(dict[str, object], raw_entry)
        path = entry.get("path")
        rule = entry.get("rule")
        message = entry.get("message")
        count = entry.get("count")
        if (
            not isinstance(path, str)
            or not isinstance(rule, str)
            or not isinstance(message, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 1
        ):
            raise ValueError("Invalid Pyright baseline entry")
        counts[(path, rule, message)] += count

    declared_total = baseline.get("total_errors")
    if not isinstance(declared_total, int) or isinstance(declared_total, bool):
        raise ValueError("Pyright baseline total_errors must be an integer")
    if declared_total != sum(counts.values()):
        raise ValueError("Pyright baseline total_errors does not match diagnostics")
    return counts


def select_effective_diff_base(
    requested_base: str,
    strict_since_commit: str,
    *,
    requested_base_precedes_cutover: bool,
    cutover_precedes_head: bool,
) -> str:
    """Grandfather changes already contained in the one-time cutover snapshot."""
    if requested_base_precedes_cutover and cutover_precedes_head:
        return strict_since_commit
    return requested_base


def evaluate_ratchet(
    diagnostics: list[Diagnostic],
    baseline_document: object,
    changed_files: set[str],
) -> RatchetResult:
    """Reject new baseline errors and every error in a changed Python file."""
    current = _diagnostic_counts(diagnostics)
    baseline = baseline_counts(baseline_document)
    new_errors = current - baseline
    removed_errors = baseline - current
    changed = Counter(
        {
            fingerprint: count
            for fingerprint, count in current.items()
            if fingerprint[0] in changed_files
        }
    )
    return {
        "passed": not new_errors and not changed,
        "current_error_count": sum(current.values()),
        "baseline_error_count": sum(baseline.values()),
        "new_error_count": sum(new_errors.values()),
        "removed_error_count": sum(removed_errors.values()),
        "changed_file_error_count": sum(changed.values()),
        "new_errors": _entries(new_errors),
        "removed_errors": _entries(removed_errors),
        "changed_file_errors": _entries(changed),
    }


def normalize_changed_files(paths: list[str], repo_root: Path) -> set[str]:
    """Normalize changed file arguments to repository-relative Python paths."""
    root = repo_root.resolve()
    normalized: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        absolute = path if path.is_absolute() else root / path
        try:
            relative = absolute.resolve().relative_to(root)
        except ValueError:
            continue
        if relative.suffix == ".py":
            normalized.add(relative.as_posix())
    return normalized


def _diagnostic_counts(diagnostics: list[Diagnostic]) -> Counter[Fingerprint]:
    return Counter((item["path"], item["rule"], item["message"]) for item in diagnostics)


def _entries(counts: Counter[Fingerprint]) -> list[BaselineEntry]:
    return [
        {"path": path, "rule": rule, "message": message, "count": count}
        for (path, rule, message), count in sorted(counts.items())
    ]


def _relative_path(raw_file: str, repo_root: Path) -> str:
    path = Path(raw_file)
    absolute = path if path.is_absolute() else repo_root / path
    try:
        return absolute.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return absolute.resolve().as_posix()
