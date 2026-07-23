"""Apply the one-time S3/L3 production-hardening migration.

The migration is intentionally fail-closed and idempotent. It edits the current
repository checkout only when the expected pre-migration anchors are present.
Run with ``--check`` first, then ``--apply`` and review the resulting diff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


class MigrationError(RuntimeError):
    """Raised when repository contents do not match the expected migration base."""


def _read(path: Path) -> str:
    if not path.is_file():
        raise MigrationError(f"required file is missing: {path}")
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
        raise MigrationError(f"{label}: expected one migration anchor, found {count}")
    return text.replace(old, new, 1), True


def _patch_sanctions(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    old = """        included = [item for item in group if item.row_included and item.hard_positive]\n        if not included:\n            ledger_rows.append(\n                _excluded_ledger_row(\n                    group,\n                    columns,\n                    reason="EXCLUDED_BY_SOURCE_RULE",\n                )\n            )\n            excluded_source_rule_count += 1\n            years_group = {resolve_sanction_year(item)[0] for item in group}\n            years_group.discard(None)\n            if len(years_group) != 1 or any(resolve_sanction_year(item)[0] is None for item in group):\n                excluded_unresolved_count += 1\n            continue\n\n        years = {resolve_sanction_year(item)[0] for item in included}\n"""
    new = """        included = [item for item in group if item.row_included and item.hard_positive]\n        excluded = [item for item in group if not (item.row_included and item.hard_positive)]\n        if excluded:\n            excluded_source_rule_count += 1\n            if any(resolve_sanction_year(item)[0] is None for item in excluded):\n                excluded_unresolved_count += 1\n            for item in excluded:\n                ledger_rows.append(\n                    _excluded_ledger_row(\n                        [item],\n                        columns,\n                        reason="EXCLUDED_BY_SOURCE_RULE",\n                    )\n                )\n        if not included:\n            continue\n\n        years = {resolve_sanction_year(item)[0] for item in included}\n"""
    text, changed1 = _replace_once(text, old, new, "S3 included/excluded split")
    text, changed2 = _replace_once(
        text,
        "        metadata = _decision_metadata(raw_group)\n        ledger_rows.append(\n            _decision_ledger_row(\n                group=raw_group,\n",
        "        metadata = _decision_metadata(included)\n        ledger_rows.append(\n            _decision_ledger_row(\n                group=included,\n",
        "S3 eligible-only provenance",
    )
    text, changed3 = _replace_once(
        text,
        '            "excluded_source_rule_mapping_count": excluded_source_rule_count,\n',
        '            "excluded_source_rule_mapping_count": excluded_source_rule_count,\n'
        '            "excluded_source_rule_row_count": sum(\n'
        "                1 for row in decisions if not (row.row_included and row.hard_positive)\n"
        "            ),\n",
        "S3 excluded row audit",
    )
    return _write(path, text, apply=apply) or changed1 or changed2 or changed3


def _patch_measurement_yaml(path: Path, *, apply: bool) -> bool:
    raw = yaml.safe_load(_read(path))
    if not isinstance(raw, dict) or not isinstance(raw.get("measurement"), dict):
        raise MigrationError("measurement.yaml must contain a measurement mapping")
    measurement = raw["measurement"]
    source_sets = {
        "primary_misstatement": {
            "role": "primary_measurement",
            "sources": [
                "S1_profit_adjustment",
                "S1_revenue_adjustment",
                "S2_audit_opinion",
                "S3_CONTENT",
            ],
        },
        "reporting_sensitivity": {
            "role": "measurement_sensitivity",
            "sources": [
                "S1_profit_adjustment",
                "S1_revenue_adjustment",
                "S2_audit_opinion",
                "S3_REPORTING",
            ],
        },
        "timeliness_sensitivity": {
            "role": "measurement_sensitivity",
            "sources": [
                "S1_profit_adjustment",
                "S1_revenue_adjustment",
                "S2_audit_opinion",
                "S3_TIMELINESS",
            ],
        },
        "broad_descriptive": {
            "role": "descriptive_falsification_only",
            "sources": ["S3_BROAD"],
        },
    }
    measurement["primary_source_set_id"] = "primary_misstatement"
    measurement["primary_s3_endpoint"] = "S3_CONTENT"
    measurement["source_sets"] = source_sets
    rendered = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    return _write(path, rendered, apply=apply)


def _patch_measurement_service(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    text, c1 = _replace_once(
        text,
        "    candidate_targets: dict[str, list[str]] | None = None,\n) -> MeasurementResult:\n",
        "    candidate_targets: dict[str, list[str]] | None = None,\n"
        "    primary_measurement_sources: list[str] | None = None,\n"
        ") -> MeasurementResult:\n",
        "measurement service signature",
    )
    text, c2 = _replace_once(
        text,
        "    expected_source_ids = sorted(expected_sources)\n"
        "    expected_channels = sorted(set(expected_sources.values()))\n",
        "    expected_source_ids = sorted(expected_sources)\n"
        "    primary_source_ids = sorted(set(primary_measurement_sources or expected_source_ids))\n"
        "    unknown_primary_sources = sorted(set(primary_source_ids) - set(expected_source_ids))\n"
        "    if unknown_primary_sources:\n"
        "        raise ValueError(\n"
        '            f"primary measurement source set references unknown sources {unknown_primary_sources}"\n'
        "        )\n"
        "    if not primary_source_ids:\n"
        '        raise ValueError("primary measurement source set must not be empty")\n'
        "    primary_expected_sources = {\n"
        "        source_id: expected_sources[source_id] for source_id in primary_source_ids\n"
        "    }\n"
        "    expected_channels = sorted(set(primary_expected_sources.values()))\n",
        "primary measurement source set initialization",
    )
    text, c3 = _replace_once(
        text,
        "        expected_source_ids=expected_source_ids,\n"
        "        source_profiles=source_profiles or {},\n",
        "        expected_source_ids=primary_source_ids,\n"
        "        source_profiles={\n"
        "            source_id: (source_profiles or {})[source_id]\n"
        "            for source_id in primary_source_ids\n"
        "            if source_id in (source_profiles or {})\n"
        "        },\n",
        "L2 primary source configuration",
    )
    text, c4 = _replace_once(
        text,
        "                for source_id in expected_source_ids\n"
        "                if expected_sources[source_id] == channel_id\n",
        "                for source_id in primary_source_ids\n"
        "                if primary_expected_sources[source_id] == channel_id\n",
        "primary channel aggregation",
    )
    text, c5 = _replace_once(
        text,
        "            expected_sources=expected_sources,\n"
        "            source_profiles=source_profiles or {},\n",
        "            expected_sources=primary_expected_sources,\n"
        "            source_profiles={\n"
        "                source_id: (source_profiles or {})[source_id]\n"
        "                for source_id in primary_source_ids\n"
        "                if source_id in (source_profiles or {})\n"
        "            },\n",
        "primary L2 score bindings",
    )
    text, c6 = _replace_once(
        text,
        '            "expected_channels": expected_channels,\n',
        '            "expected_channels": expected_channels,\n'
        '            "primary_measurement_sources": primary_source_ids,\n',
        "matrix primary source receipt",
    )
    return _write(path, text, apply=apply) or any((c1, c2, c3, c4, c5, c6))


def _patch_p05(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    text, c1 = _replace_once(
        text,
        "    result = build_measurement_inputs(\n",
        "    primary_measurement_sources = _primary_measurement_sources(measurement)\n"
        "    result = build_measurement_inputs(\n",
        "P05 primary source resolution",
    )
    text, c2 = _replace_once(
        text,
        "        candidate_targets=candidate_targets,\n    )\n",
        "        candidate_targets=candidate_targets,\n"
        "        primary_measurement_sources=primary_measurement_sources,\n"
        "    )\n",
        "P05 primary source pass-through",
    )
    helper = """\n\ndef _primary_measurement_sources(measurement: dict[str, object]) -> list[str]:\n    source_set_id = measurement.get("primary_source_set_id")\n    if not isinstance(source_set_id, str) or not source_set_id:\n        raise ValueError("measurement.primary_source_set_id must be locked")\n    source_sets = mapping(measurement.get("source_sets"), "measurement.source_sets")\n    source_set = mapping(source_sets.get(source_set_id), f"measurement.source_sets.{source_set_id}")\n    sources = [\n        str(value)\n        for value in sequence(\n            source_set.get("sources"), f"measurement.source_sets.{source_set_id}.sources"\n        )\n    ]\n    s3_sources = sorted(source for source in sources if source.startswith("S3_"))\n    if s3_sources != ["S3_CONTENT"]:\n        raise ValueError("primary measurement source set must contain S3_CONTENT and no other S3 endpoint")\n    return sources\n"""
    text, c3 = _replace_once(
        text,
        "\ndef _evidence_sources(\n",
        helper + "\n\ndef _evidence_sources(\n",
        "P05 primary source helper",
    )
    return _write(path, text, apply=apply) or any((c1, c2, c3))


def _patch_p10(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    text, c1 = _replace_once(
        text,
        "        source_channels, accuracy_priors = _l3_bindings(\n"
        "            registry=loaded.registry,\n",
        "        source_channels, accuracy_priors = _l3_bindings(\n"
        "            registry=loaded.registry,\n"
        "            allowed_source_ids=_primary_measurement_sources(measurement),\n",
        "P10 primary L3 source binding",
    )
    text, c2 = _replace_once(
        text,
        "def _l3_bindings(\n"
        "    *,\n"
        "    registry: dict[str, object],\n"
        "    priors_by_profile: dict[str, Any],\n"
        ") -> tuple[dict[str, str], dict[str, dict[str, float]]]:\n",
        "def _l3_bindings(\n"
        "    *,\n"
        "    registry: dict[str, object],\n"
        "    priors_by_profile: dict[str, Any],\n"
        "    allowed_source_ids: list[str] | None = None,\n"
        ") -> tuple[dict[str, str], dict[str, dict[str, float]]]:\n",
        "P10 L3 binding signature",
    )
    text, c3 = _replace_once(
        text,
        "    for logical_source_id, source in sorted(sources.items()):\n"
        "        raw_prior: object = priors_by_profile.get(source.profile_id)\n",
        "    allowed = set(sources) if allowed_source_ids is None else set(allowed_source_ids)\n"
        "    unknown = sorted(allowed - set(sources))\n"
        "    if unknown:\n"
        '        raise ValueError(f"primary L3 source set contains unknown sources {unknown}")\n'
        "    for logical_source_id, source in sorted(sources.items()):\n"
        "        if logical_source_id not in allowed:\n"
        "            continue\n"
        "        raw_prior: object = priors_by_profile.get(source.profile_id)\n",
        "P10 filter logical sources",
    )
    helper = """\n\ndef _primary_measurement_sources(measurement: dict[str, object]) -> list[str]:\n    source_set_id = measurement.get("primary_source_set_id")\n    if not isinstance(source_set_id, str) or not source_set_id:\n        raise ValueError("measurement.primary_source_set_id must be locked")\n    source_sets = mapping(measurement.get("source_sets"), "measurement.source_sets")\n    source_set = mapping(source_sets.get(source_set_id), f"measurement.source_sets.{source_set_id}")\n    sources = [\n        str(value)\n        for value in sequence(\n            source_set.get("sources"), f"measurement.source_sets.{source_set_id}.sources"\n        )\n    ]\n    s3_sources = sorted(source for source in sources if source.startswith("S3_"))\n    if s3_sources != ["S3_CONTENT"]:\n        raise ValueError("primary L3 source set must contain S3_CONTENT and no other S3 endpoint")\n    return sources\n"""
    text, c4 = _replace_once(
        text,
        "\ndef _l3_bindings(\n",
        helper + "\n\ndef _l3_bindings(\n",
        "P10 primary source helper",
    )
    return _write(path, text, apply=apply) or any((c1, c2, c3, c4))


def _patch_known_case_catalog(path: Path, *, apply: bool) -> bool:
    raw = yaml.safe_load(_read(path))
    profiles = raw.get("source_catalog", {}).get("profiles", {})
    known = profiles.get("known_cases")
    if not isinstance(known, dict):
        raise MigrationError("source_catalog known_cases profile is missing")
    known["discovery"]["globs"] = ["data/source/known_case_registry.csv"]
    semantics = known.setdefault("semantic_fields", {})
    semantics.update(
        {
            "case_construct": ["case_construct"],
            "case_role": ["role"],
            "training_include_flag": ["training_include_flag"],
            "calibration_include_flag": ["calibration_include_flag"],
            "model_selection_include_flag": ["model_selection_include_flag"],
            "external_validation_include_flag": ["external_validation_include_flag"],
        }
    )
    known["required_semantic_fields"] = [
        "firm_id",
        "fiscal_year",
        "case_id",
        "case_construct",
        "case_role",
        "training_include_flag",
        "calibration_include_flag",
        "model_selection_include_flag",
        "external_validation_include_flag",
    ]
    rendered = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    return _write(path, rendered, apply=apply)


def _patch_p15(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    text, c1 = _replace_once(
        text,
        '    required = {"case_id", FIRM_ID, FISCAL_YEAR}\n',
        "    required = {\n"
        '        "case_id",\n'
        "        FIRM_ID,\n"
        "        FISCAL_YEAR,\n"
        '        "case_construct",\n'
        '        "case_role",\n'
        '        "training_include_flag",\n'
        '        "calibration_include_flag",\n'
        '        "model_selection_include_flag",\n'
        '        "external_validation_include_flag",\n'
        "    }\n",
        "P15 known-case semantic contract",
    )
    old = """        rows.append(\n            {\n                "case_id": str(row[str(semantics["case_id"])]),\n                columns[FIRM_ID]: canonical,\n                columns[FISCAL_YEAR]: int(str(row[str(semantics[FISCAL_YEAR])])),\n            }\n        )\n"""
    new = """        case_construct = str(row[str(semantics["case_construct"])]).strip()\n        case_role = str(row[str(semantics["case_role"])]).strip()\n        flags = {\n            name: _parse_bool(row[str(semantics[name])], name)\n            for name in (\n                "training_include_flag",\n                "calibration_include_flag",\n                "model_selection_include_flag",\n                "external_validation_include_flag",\n            )\n        }\n        if case_construct != "CONFIRMED_FINANCIAL_REPORTING_CASE":\n            raise ValueError("known case construct must be CONFIRMED_FINANCIAL_REPORTING_CASE")\n        if case_role != "SIMULATION_EXTERNAL_VALIDATION":\n            raise ValueError("known case role must be SIMULATION_EXTERNAL_VALIDATION")\n        if flags != {\n            "training_include_flag": False,\n            "calibration_include_flag": False,\n            "model_selection_include_flag": False,\n            "external_validation_include_flag": True,\n        }:\n            raise ValueError("known case inclusion flags violate the sealed external-validation role")\n        rows.append(\n            {\n                "case_id": str(row[str(semantics["case_id"])]),\n                columns[FIRM_ID]: canonical,\n                columns[FISCAL_YEAR]: int(str(row[str(semantics[FISCAL_YEAR])])),\n            }\n        )\n"""
    text, c2 = _replace_once(text, old, new, "P15 known-case row validation")
    helper = """\n\ndef _parse_bool(value: object, field: str) -> bool:\n    if isinstance(value, bool):\n        return value\n    normalized = str(value).strip().lower()\n    if normalized in {"true", "1", "yes", "y"}:\n        return True\n    if normalized in {"false", "0", "no", "n"}:\n        return False\n    raise ValueError(f"known case field={field}: boolean value required")\n"""
    text, c3 = _replace_once(
        text,
        '\n\nif __name__ == "__main__":\n',
        helper + '\n\nif __name__ == "__main__":\n',
        "P15 boolean parser",
    )
    return _write(path, text, apply=apply) or any((c1, c2, c3))


def _patch_s3_report(path: Path, *, apply: bool) -> bool:
    text = _read(path)
    text, c1 = _replace_once(
        text,
        '    development_positive = int(development["positive"].sum()) if not development.empty else 0\n',
        '    development_positive = int(development["positive"].sum()) if not development.empty else 0\n'
        "    development_positive_by_endpoint = (\n"
        "        {\n"
        '            str(endpoint): int(group["positive"].sum())\n'
        "            for endpoint, group in development.groupby(SOURCE_ID, sort=True)\n"
        "        }\n"
        "        if not development.empty\n"
        "        else {}\n"
        "    )\n"
        '    content_positive = int(development_positive_by_endpoint.get("S3_CONTENT", 0))\n'
        "    sanction_year_unresolved_firm_year_count = int(\n"
        "        unknown_reasons.loc[\n"
        '            unknown_reasons[OUTCOME_BASIS].astype(str).eq("SANCTION_YEAR_UNRESOLVED"),\n'
        '            "unknown_count",\n'
        "        ].sum()\n"
        "    ) if not unknown_reasons.empty else 0\n",
        "S3 endpoint-specific development counts",
    )
    text, c2 = _replace_once(
        text,
        '        else "BLOCK_L3_NO_DEVELOPMENT_POSITIVES"\n        if development_positive == 0\n',
        '        else "BLOCK_L3_NO_S3_CONTENT_DEVELOPMENT_POSITIVES"\n        if content_positive == 0\n',
        "S3 content readiness status",
    )
    text, c3 = _replace_once(
        text,
        '        "development_s3_positive_count": development_positive,\n',
        '        "development_s3_positive_count": development_positive,\n'
        '        "development_positive_count_by_endpoint": development_positive_by_endpoint,\n'
        '        "development_s3_content_positive_count": content_positive,\n'
        '        "sanction_year_unresolved_firm_year_count": sanction_year_unresolved_firm_year_count,\n',
        "S3 summary fields",
    )
    text, c4 = _replace_once(
        text,
        '        "excluded_source_rule_mapping_count": sanction_audit.get(\n'
        '            "excluded_source_rule_mapping_count"\n'
        "        ),\n",
        '        "excluded_source_rule_mapping_count": sanction_audit.get(\n'
        '            "excluded_source_rule_mapping_count"\n'
        "        ),\n"
        '        "excluded_source_rule_row_count": sanction_audit.get(\n'
        '            "excluded_source_rule_row_count"\n'
        "        ),\n",
        "S3 excluded row summary",
    )
    return _write(path, text, apply=apply) or any((c1, c2, c3, c4))


def _write_regression_tests(root: Path, *, apply: bool) -> bool:
    path = root / "tests" / "stages" / "test_s3_l3_production_hardening.py"
    content = """from __future__ import annotations\n\nfrom pathlib import Path\n\nimport yaml\n\n\ndef test_primary_source_set_uses_only_s3_content() -> None:\n    config = yaml.safe_load(Path("config/methodology/measurement.yaml").read_text(encoding="utf-8"))\n    measurement = config["measurement"]\n    primary = measurement["source_sets"][measurement["primary_source_set_id"]]["sources"]\n    assert measurement["primary_s3_endpoint"] == "S3_CONTENT"\n    assert sorted(source for source in primary if source.startswith("S3_")) == ["S3_CONTENT"]\n\n\ndef test_known_case_registry_contract_is_external_validation_only() -> None:\n    config = yaml.safe_load(Path("config/methodology/source_catalog.yaml").read_text(encoding="utf-8"))\n    known = config["source_catalog"]["profiles"]["known_cases"]\n    assert known["discovery"]["globs"] == ["data/source/known_case_registry.csv"]\n    required = set(known["required_semantic_fields"])\n    assert {\n        "case_construct",\n        "case_role",\n        "training_include_flag",\n        "calibration_include_flag",\n        "model_selection_include_flag",\n        "external_validation_include_flag",\n    }.issubset(required)\n"""
    if path.exists() and _read(path) == content:
        return False
    if apply:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return True


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
        "mixed_s3_groups": _patch_sanctions(root / "src/evidence/sanctions.py", apply=apply),
        "measurement_source_sets": _patch_measurement_yaml(
            root / "config/methodology/measurement.yaml", apply=apply
        ),
        "measurement_engine": _patch_measurement_service(
            root / "src/measurement/service.py", apply=apply
        ),
        "p05_primary_sources": _patch_p05(root / "scripts/p05_measurement_inputs.py", apply=apply),
        "p10_primary_l3_sources": _patch_p10(
            root / "scripts/p10_select_measurement.py", apply=apply
        ),
        "known_case_catalog": _patch_known_case_catalog(
            root / "config/methodology/source_catalog.yaml", apply=apply
        ),
        "known_case_validation": _patch_p15(root / "scripts/p15_open_known_cases.py", apply=apply),
        "s3_endpoint_audit": _patch_s3_report(
            root / "scripts/report_s3_year_audit.py", apply=apply
        ),
        "regression_tests": _write_regression_tests(root, apply=apply),
    }
    result = {
        "mode": "apply" if apply else "check",
        "repo_root": str(root),
        "changes_required": [name for name, changed in operations.items() if changed],
        "already_hardened": [name for name, changed in operations.items() if not changed],
    }
    print("S3_L3_HARDENING_JSON=" + json.dumps(result, sort_keys=True))
    if args.check and result["changes_required"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
