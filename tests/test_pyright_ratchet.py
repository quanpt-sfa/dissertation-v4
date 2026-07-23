"""Regression tests for the Pyright baseline ratchet."""

from __future__ import annotations

from pathlib import Path

from core.pyright_ratchet import (
    Diagnostic,
    baseline_from_diagnostics,
    diagnostics_from_pyright,
    evaluate_ratchet,
    normalize_changed_files,
)


def _diagnostic(path: str, rule: str = "reportUnknownVariableType") -> Diagnostic:
    return {"path": path, "rule": rule, "message": "Type of value is unknown"}


def test_ratchet_rejects_new_fingerprint_even_when_total_is_unchanged() -> None:
    baseline = baseline_from_diagnostics(
        [_diagnostic("src/legacy.py", "legacy-rule")], generated_from="base"
    )
    result = evaluate_ratchet(
        [_diagnostic("src/new.py", "new-rule")], baseline, changed_files=set()
    )

    assert result["passed"] is False
    assert result["current_error_count"] == result["baseline_error_count"] == 1
    assert result["new_error_count"] == 1
    assert result["removed_error_count"] == 1


def test_ratchet_allows_historical_errors_but_not_errors_in_changed_files() -> None:
    diagnostic = _diagnostic("src/legacy.py")
    baseline = baseline_from_diagnostics([diagnostic], generated_from="base")

    unchanged = evaluate_ratchet([diagnostic], baseline, changed_files=set())
    changed = evaluate_ratchet([diagnostic], baseline, changed_files={"src/legacy.py"})

    assert unchanged["passed"] is True
    assert changed["passed"] is False
    assert changed["new_error_count"] == 0
    assert changed["changed_file_error_count"] == 1


def test_pyright_parser_ignores_line_numbers_and_normalizes_messages(tmp_path: Path) -> None:
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("value = 1\n", encoding="utf-8")
    payload = {
        "generalDiagnostics": [
            {
                "file": str(source),
                "severity": "error",
                "message": "Type of  value\n is unknown",
                "rule": "reportUnknownVariableType",
                "range": {"start": {"line": 100, "character": 2}},
            },
            {
                "file": str(source),
                "severity": "warning",
                "message": "Warning is not ratcheted",
                "rule": "reportUnusedVariable",
            },
        ]
    }

    diagnostics = diagnostics_from_pyright(payload, tmp_path)

    assert diagnostics == [
        {
            "path": "src/module.py",
            "rule": "reportUnknownVariableType",
            "message": "Type of value is unknown",
        }
    ]


def test_changed_file_normalization_keeps_only_repository_python_files(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.py"

    changed = normalize_changed_files(
        ["src/a.py", "README.md", str(outside)],
        tmp_path,
    )

    assert changed == {"src/a.py"}
