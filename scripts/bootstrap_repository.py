"""Validate modular configuration and regenerate only repository documentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.forbidden_patterns import validate_source_patterns
from core.generated_docs import render_documents, write_or_check_documents
from core.registry_compiler import compile_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    modes = sum((arguments.check, arguments.write, arguments.validate_only))
    if modes != 1:
        parser.error("select exactly one of --write, --check, or --validate-only")
    compiled = compile_registry(arguments.config)
    root = arguments.config.resolve().parent.parent
    validate_source_patterns(root, dict(compiled.registry))
    if not arguments.validate_only:
        write_or_check_documents(
            root, render_documents(dict(compiled.registry)), check=arguments.check
        )
    if arguments.verbose:
        print(compiled.protocol_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
