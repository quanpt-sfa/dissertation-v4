"""Migrate a persisted v6 evidence-ledger parquet file to schema v7."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from core.registry_compiler import compile_registry
from core.schema_registry import contract_registry
from evidence.migration import migrate_evidence_ledger_v6_to_v7


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {args.output}")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    compiled = compile_registry(args.config)
    registry = cast(dict[str, Any], compiled.registry)
    schema = cast(dict[str, Any], registry["schemas"]["evidence_ledger_frame"])
    if int(schema["version"]) != 7:
        raise ValueError("configured evidence ledger schema must be version 7")
    ordered_columns = [str(item["physical_name"]) for item in schema["columns"]]

    migration = migrate_evidence_ledger_v6_to_v7(
        pd.read_parquet(args.input),
        ordered_columns=ordered_columns,
    )
    contract_registry(registry).validate("evidence_ledger", migration.frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    migration.frame.to_parquet(temporary, index=False)
    temporary.replace(args.output)

    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(migration.audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "evidence-ledger migration status=PASS "
        f"rows={len(migration.frame)} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
