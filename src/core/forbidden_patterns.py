"""Static guard for direct I/O and copied registry literals."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

from .errors import ConfigurationError

FORBIDDEN_IO = (
    "pd.read_csv(",
    "pd.read_parquet(",
    ".to_csv(",
    ".to_parquet(",
    "np.random.seed(",
)
FORBIDDEN_APPEND = (
    'mode="a"',
    "mode='a'",
)
APPROVED_CORE_FILES = {
    "artifact_store.py",
    "config_loader.py",
    "forbidden_patterns.py",
    "semantic_keys.py",
}
# These semantic roles describe control-plane or registry metadata rather than
# physical empirical-data bindings. Their names may legitimately appear in
# receipts and scenario-control dictionaries. All entity, period, measurement,
# outcome, predictor, and evidence columns remain protected by the literal guard.
NON_DATA_COLUMN_SEMANTIC_ROLES = {
    "feature_registry_key",
    "feature_view_key",
    "feature_lineage_source",
    "fold_key",
    "simulation_scenario",
}
# P07A/P07B are explicitly non-production source-resolution audits. They can
# inspect external, researcher-supplied files but cannot consume runtime
# artifacts, outcomes, or known cases. Keeping this exception named and tiny
# preserves the production-stage I/O firewall.
NONPRODUCTION_AUDIT_SCRIPTS = {
    "scripts/p07a_feature_definition_audit.py",
    "scripts/validate_production_source_contracts.py",
}
NONPRODUCTION_AUDIT_MODULES: set[str] = set()
# These scripts intentionally maintain repository source/configuration rather
# than execute an empirical stage. They may reference registered schema names or
# non-manifest config paths only for migration/locking. The whitelist is exact;
# no production runner or analysis stage belongs here.
REPOSITORY_MAINTENANCE_SCRIPTS = {
    "scripts/apply_s3_l3_production_hardening.py",
    "scripts/finalize_s3_l3_production_hardening.py",
    "scripts/repair_s3_l3_hardening_regressions.py",
    "scripts/lock_l3_parameters.py",
}
PRODUCTION_DATA_BOUNDARIES = {
    "src/features/store.py",
    # Explicit documentation-export boundaries. These scripts read only verified
    # runtime artifacts and write researcher-facing audit/calibration tables outside
    # the immutable production artifact namespace.
    "scripts/report_measurement_calibration.py",
    "scripts/report_s3_year_audit.py",
}

StringMap = dict[str, Any]


def _mapping(value: object, context: str) -> StringMap:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context}: mapping required")
    return cast(StringMap, value)


def _is_registered_data_boundary(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return (
        relative
        in (
            NONPRODUCTION_AUDIT_SCRIPTS
            | NONPRODUCTION_AUDIT_MODULES
            | REPOSITORY_MAINTENANCE_SCRIPTS
            | PRODUCTION_DATA_BOUNDARIES
        )
        or relative.startswith("src/p01/")
        or relative.startswith("src/p02/")
        or relative.startswith("src/snapshot/")
        or relative
        in {
            "scripts/p01_audit_raw.py",
            "scripts/p02_build_firm_panel.py",
            "scripts/create_data_snapshot.py",
            "scripts/run_pipeline.py",
        }
    )


def validate_source_patterns(
    root: Path,
    registry: dict[str, object],
) -> None:
    columns = _mapping(registry.get("columns"), "columns")
    artifacts = _mapping(registry.get("artifacts"), "artifacts")

    physical_names: set[str] = set()
    artifact_paths: set[str] = set()

    for column_id, raw_column in columns.items():
        column = _mapping(raw_column, f"column={column_id}")
        physical_name = column.get("physical_name")
        if not isinstance(physical_name, str):
            raise ConfigurationError(f"column={column_id}: physical_name must be a string")
        semantic_role = column.get("semantic_role")
        if semantic_role not in NON_DATA_COLUMN_SEMANTIC_ROLES:
            physical_names.add(physical_name)

    for artifact_id, raw_artifact in artifacts.items():
        artifact = _mapping(raw_artifact, f"artifact={artifact_id}")
        path_template = artifact.get("path_template")
        if not isinstance(path_template, str):
            raise ConfigurationError(f"artifact={artifact_id}: path_template must be a string")
        artifact_paths.add(path_template)

    source_files = [
        *root.glob("scripts/*.py"),
        *root.glob("src/**/*.py"),
    ]
    for path in source_files:
        if path.name in APPROVED_CORE_FILES and path.parent.name == "core":
            continue

        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        registered_boundary = _is_registered_data_boundary(root, path)
        if relative not in NONPRODUCTION_AUDIT_SCRIPTS and not registered_boundary:
            for pattern in (*FORBIDDEN_IO, *FORBIDDEN_APPEND):
                if pattern in text:
                    raise ConfigurationError(
                        f"source={path}: forbidden pattern {pattern}; "
                        "use the registered raw-reader or core runtime layer"
                    )

        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise ConfigurationError(f"source={path}: Python syntax error") from exc

        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        if not registered_boundary:
            copied_columns = sorted(physical_names & literals)
            if copied_columns:
                raise ConfigurationError(
                    f"source={path}: registered physical columns "
                    f"copied into source: {copied_columns}"
                )

        copied_paths = sorted(artifact_paths & literals)
        if copied_paths:
            raise ConfigurationError(
                f"source={path}: registered artifact paths copied into source: {copied_paths}"
            )

        if relative not in REPOSITORY_MAINTENANCE_SCRIPTS:
            direct_config_paths = sorted(
                value
                for value in literals
                if (value.startswith("config/") or value.startswith("config\\"))
                and value
                not in {
                    "config/pipeline.yaml",
                    "config\\pipeline.yaml",
                }
            )
            if direct_config_paths:
                raise ConfigurationError(
                    f"source={path}: direct source-config paths are forbidden: {direct_config_paths}"
                )
