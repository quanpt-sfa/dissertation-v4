"""Locate and audit historical board sources without machine-specific paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

FINAL_NAME = "vn_pipeline_final_firm_year_2015_2025.parquet"
LISTING_GLOBS = (
    "listing_history_expanded*.csv",
    "*listing*history*.csv",
)
BOARD_KEYWORDS = (
    "exchange",
    "board",
    "market",
    "listing",
    "san",
    "sàn",
)
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_roots(extra_roots: list[Path]) -> list[Path]:
    repo = _repo_root()
    raw_env = os.environ.get("DISSERTATION_RAW_ROOT")
    candidates = [
        *(Path(value).expanduser() for value in [raw_env] if value),
        *extra_roots,
        repo,
        repo.parent,
        repo.parent.parent,
    ]
    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            path = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(path))
        if key in seen or not path.exists() or not path.is_dir():
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def _walk_files(root: Path) -> list[Path]:
    matches: list[Path] = []
    for current, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in PRUNE_DIRS]
        current_path = Path(current)
        for name in files:
            lowered = name.lower()
            if lowered == FINAL_NAME.lower() or (
                lowered.endswith(".csv")
                and "listing" in lowered
                and "history" in lowered
            ):
                matches.append(current_path / name)
    return matches


def _discover(extra_roots: list[Path]) -> dict[str, list[Path]]:
    final_files: dict[str, Path] = {}
    listing_files: dict[str, Path] = {}
    for root in _candidate_roots(extra_roots):
        for path in _walk_files(root):
            key = os.path.normcase(str(path.resolve()))
            if path.name.lower() == FINAL_NAME.lower():
                final_files[key] = path.resolve()
            elif any(path.match(pattern) for pattern in LISTING_GLOBS):
                listing_files[key] = path.resolve()
    return {
        "final": sorted(final_files.values()),
        "listing": sorted(listing_files.values()),
    }


def _parquet_audit(path: Path) -> dict[str, Any]:
    schema = pq.read_schema(path)
    return {
        "path": str(path),
        "columns": schema.names,
        "has_exchange_or_board": "exchange_or_board" in schema.names,
        "has_industry_code": "industry_code" in schema.names,
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


def _listing_audit(path: Path) -> dict[str, Any]:
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    columns = [str(column) for column in header.columns]
    board_candidates = [
        column
        for column in columns
        if any(keyword in column.casefold() for keyword in BOARD_KEYWORDS)
    ]
    result: dict[str, Any] = {
        "path": str(path),
        "columns": columns,
        "board_candidate_columns": board_candidates,
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    if board_candidates:
        sample = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=board_candidates,
            low_memory=False,
        )
        result["candidate_value_counts"] = {
            column: {
                str(key): int(value)
                for key, value in sample[column]
                .dropna()
                .astype(str)
                .str.strip()
                .value_counts()
                .head(100)
                .items()
            }
            for column in board_candidates
        }
    return result


def _builder_audit() -> dict[str, Any]:
    path = _repo_root() / "scripts" / "build_immutable_firm_year_input.py"
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        text = path.read_text(encoding="utf-8")
        result["mentions_exchange_or_board"] = "exchange_or_board" in text
        result["mentions_listing_history"] = "listing_history" in text
        result["mentions_industry"] = "industry" in text.lower()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        type=Path,
        help="Additional root to search recursively; may be supplied more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("board_source_audit.json"),
        help="JSON output path, relative to the current working directory by default.",
    )
    args = parser.parse_args()

    discovered = _discover(args.search_root)
    report = {
        "repo_root": str(_repo_root()),
        "environment_raw_root": os.environ.get("DISSERTATION_RAW_ROOT"),
        "searched_roots": [str(path) for path in _candidate_roots(args.search_root)],
        "final_parquets": [_parquet_audit(path) for path in discovered["final"]],
        "listing_histories": [_listing_audit(path) for path in discovered["listing"]],
        "local_builder": _builder_audit(),
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit_output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
