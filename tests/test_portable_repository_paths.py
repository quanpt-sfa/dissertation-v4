from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SCAN_ROOTS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "config",
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "templates",
    ROOT / "tests",
    ROOT / "tools",
)
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]+)")
MACHINE_POSIX_ROOT = re.compile(r"(?<![A-Za-z0-9])/(?:home|Users|mnt/[A-Za-z]|Volumes)/")


def _text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in TEXT_SUFFIXES
    ]


def test_repository_contains_no_machine_specific_absolute_paths() -> None:
    violations: list[str] = []
    for scan_root in SCAN_ROOTS:
        for path in _text_files(scan_root):
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if WINDOWS_ABSOLUTE.search(line) or MACHINE_POSIX_ROOT.search(line):
                    relative = path.relative_to(ROOT).as_posix()
                    violations.append(f"{relative}:{line_number}: {line.strip()}")

    assert not violations, "machine-specific absolute paths found:\n" + "\n".join(violations)
