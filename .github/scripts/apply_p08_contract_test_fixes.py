from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Patch target not found: {label}")
    return text.replace(old, new, 1)


def patch_config() -> None:
    path = Path("config/execution/simulation_mcse.yaml")
    text = path.read_text(encoding="utf-8")
    marker = "  - policy_id: descriptive_training_and_sampling\n"
    policy = dedent(
        """
          - policy_id: descriptive_imbalance_treatment_applied
            priority: 20
            metric_pattern: imbalance_treatment_applied
            role: descriptive
            mcse_gate_required: false
            undefined_policy: report_only
            target_rule: none
            minimum_finite_fraction: 0.0
        """
    ).lstrip("\n")
    if policy not in text:
        text = replace_once(text, marker, policy + marker, "imbalance metric policy")
    path.write_text(text, encoding="utf-8")


def patch_service() -> None:
    path = Path("src/simulation/mcse_service.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    required = {SCENARIO_ID, METHOD_ID, REPLICATION_ID, METRIC_ID, ESTIMATE}\n",
        "    required = {SCENARIO_ID, METHOD_ID, REPLICATION_ID, METRIC_ID}\n",
        "sanitizer required columns",
    )
    text = replace_once(
        text,
        '    records = cast(list[dict[str, object]], frame.to_dict(orient="records"))\n',
        '    records = cast(list[dict[str, object]], frame.to_dict(orient="records"))\n'
        "    estimate_present = ESTIMATE in frame.columns\n",
        "sanitizer estimate availability",
    )
    text = replace_once(
        text,
        "    for row_index, group_key in normalized_rows:\n",
        "    if not normalized_rows:\n"
        "        return frame.copy()\n"
        "    if not estimate_present:\n"
        '        raise ValueError("normalized cost regret rows require the estimate column")\n'
        "\n"
        "    for row_index, group_key in normalized_rows:\n",
        "sanitizer normalized rows",
    )
    path.write_text(text, encoding="utf-8")


def patch_collector() -> None:
    path = Path("scripts/p08c_aggregate_batches.py")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from simulation.mcse_contract import compile_metric_policies, resolve_metric_policies\n",
        "from simulation.mcse_contract import (\n"
        "    MetricPolicy,\n"
        "    compile_metric_policies,\n"
        "    resolve_metric_policies,\n"
        ")\n",
        "collector policy imports",
    )
    alias_marker = "from simulation.replication_contract import replication_plan\n\n"
    if "_filter_active_batches = filter_active_batches" not in text:
        text = replace_once(
            text,
            alias_marker,
            alias_marker + "_filter_active_batches = filter_active_batches\n\n",
            "collector compatibility alias",
        )
    text = replace_once(
        text,
        "        clean = sanitize_normalized_cost_regret(value)\n"
        "        batches.append(clean)\n",
        "        batches.append(value)\n",
        "collector raw batch retention",
    )
    text = replace_once(
        text,
        "    methodology = validate_methodology(\n"
        "        batches=batches,\n"
        "        scenario_by_id=scenario_by_id,\n"
        "    )\n"
        "    active_batches = filter_active_batches(batches, scenario_by_id)\n",
        "    batches = [sanitize_normalized_cost_regret(batch) for batch in batches]\n"
        "\n"
        "    methodology = _validate_methodology(\n"
        "        batches=batches,\n"
        "        scenario_by_id=scenario_by_id,\n"
        "        simulation=simulation,\n"
        '        mcse_registry=loaded.registry.get("simulation_mcse"),\n'
        "    )\n"
        "    active_batches = _filter_active_batches(batches, scenario_by_id)\n"
        "    methodology_report = {\n"
        '        key: value for key, value in methodology.items() if key != "metric_policies"\n'
        "    }\n",
        "collector methodology integration",
    )
    text = text.replace(
        '"methodology_validation": methodology,',
        '"methodology_validation": methodology_report,',
    )
    policy_start = text.find(
        '\n    try:\n        mcse_registry = mapping(loaded.registry.get("simulation_mcse"), "simulation_mcse")\n'
    )
    policy_end = text.find('\n    core = mapping(simulation.get("core"), "simulation.core")\n')
    if policy_start < 0 or policy_end < 0 or policy_end <= policy_start:
        raise SystemExit("Patch target not found: collector standalone policy block")
    text = text[:policy_start] + "\n" + text[policy_end:]
    core_marker = '    core = mapping(simulation.get("core"), "simulation.core")\n'
    text = replace_once(
        text,
        core_marker,
        "    metric_policies_raw = methodology.get(\"metric_policies\")\n"
        "    metric_policies = (\n"
        "        cast(dict[str, MetricPolicy], metric_policies_raw)\n"
        "        if isinstance(metric_policies_raw, dict)\n"
        "        else {}\n"
        "    )\n"
        "\n"
        + core_marker,
        "collector policy binding",
    )
    text = replace_once(
        text,
        '        metric_policies=policy_coverage["policies"],\n',
        "        metric_policies=metric_policies,\n",
        "collector summary policies",
    )
    old_coverage = (
        '            "metric_policy_coverage": {\n'
        '                key: value for key, value in policy_coverage.items() if key != "policies"\n'
        "            },\n"
    )
    text = replace_once(
        text,
        old_coverage,
        '            "metric_policy_coverage": methodology.get("metric_policy_coverage", {}),\n',
        "collector coverage report",
    )
    helper_marker = "\ndef _coordinates(item: dict[str, object]) -> dict[str, str]:\n"
    helper = dedent(
        """

        def _validate_methodology(
            *,
            batches: list[pd.DataFrame],
            scenario_by_id: dict[str, dict[str, object]],
            simulation: dict[str, object],
            mcse_registry: object,
        ) -> dict[str, object]:
            result = validate_methodology(batches=batches, scenario_by_id=scenario_by_id)
            if result.get("status") != "PASS":
                return result
            try:
                compiled = compile_metric_policies(
                    mapping(mcse_registry, "simulation_mcse"),
                    simulation,
                )
                coverage = resolve_metric_policies(
                    required_active_metric_ids(scenario_by_id),
                    compiled,
                )
            except ValueError as exc:
                reason_codes_raw = result.get("reason_codes")
                reason_codes = (
                    [str(value) for value in cast(list[object], reason_codes_raw)]
                    if isinstance(reason_codes_raw, list)
                    else []
                )
                return {
                    **result,
                    "status": "FAIL",
                    "reason_codes": [
                        *reason_codes,
                        "P08_MCSE_POLICY_CONTRACT_INCOMPLETE",
                    ],
                    "policy_error": str(exc),
                }
            return {
                **result,
                "metric_policy_coverage": {
                    key: value for key, value in coverage.items() if key != "policies"
                },
                "metric_policies": coverage["policies"],
            }
        """
    )
    if "def _validate_methodology(" not in text:
        text = replace_once(
            text,
            helper_marker,
            helper + helper_marker,
            "collector methodology helper",
        )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    patch_config()
    patch_service()
    patch_collector()


if __name__ == "__main__":
    main()
