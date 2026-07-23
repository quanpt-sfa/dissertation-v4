from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    (ROOT / path).write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}: {old[:80]!r}")
    write(path, content.replace(old, new, 1))


# Lock the Gate 1 learner/procedure rather than choosing it from empirical outcomes.
replace_once(
    "config/methodology/measurement.yaml",
    """  selection:\n    time_only: M_f_star\n    strict_channel: M_f_c_star\n    development_only: true\n    no_candidate_value: none\n""",
    """  selection:\n    time_only: M_f_star\n    strict_channel: M_f_c_star\n    development_only: true\n    nested_refit_required: true\n    complete_channel_grid_required: true\n    gate1_learner_id: elastic_net_logistic\n    gate1_feature_group: full\n    no_candidate_value: none\n""",
)

# P10 now consumes the feature panel for development-only full refits.
replace_once(
    "config/foundation/steps.yaml",
    """    - mcse_report\n    - feature_registry\n    optional_reads: []\n""",
    """    - mcse_report\n    - feature_panel\n    - feature_registry\n    optional_reads: []\n""",
)

# Production registries use content_observability_role; retain legacy role fixtures.
replace_once(
    "src/modeling/service.py",
    """def _feature_groups(registry: list[dict[str, Any]]) -> dict[str, list[str]]:\n    content = [str(item[\"feature_id\"]) for item in registry if item.get(\"role\") == \"content\"]\n    observable = [\n        str(item[\"feature_id\"]) for item in registry if item.get(\"role\") == \"observability\"\n    ]\n    ambiguous = [str(item[\"feature_id\"]) for item in registry if item.get(\"role\") == \"ambiguous\"]\n""",
    """def _feature_groups(registry: list[dict[str, Any]]) -> dict[str, list[str]]:\n    def role(item: dict[str, Any]) -> object:\n        return item.get(\"role\", item.get(\"content_observability_role\"))\n\n    eligible = [\n        item\n        for item in registry\n        if item.get(\"research_decision_status\") in {None, \"LOCKED\"}\n        and item.get(\"model_eligibility\") in {None, \"eligible\"}\n    ]\n    content = [str(item[\"feature_id\"]) for item in eligible if role(item) == \"content\"]\n    observable = [\n        str(item[\"feature_id\"]) for item in eligible if role(item) == \"observability\"\n    ]\n    ambiguous = [str(item[\"feature_id\"]) for item in eligible if role(item) == \"ambiguous\"]\n""",
)

# Allow the full-refit results to authoritatively replace proxy-only Gate 1 scores.
replace_once(
    "src/selection/service.py",
    """    l3_fold_result: dict[str, Any] | None = None,\n    minimum_observed_channels: int | None = None,\n) -> SelectionResult:\n""",
    """    l3_fold_result: dict[str, Any] | None = None,\n    minimum_observed_channels: int | None = None,\n    nested_candidate_results: list[dict[str, Any]] | None = None,\n) -> SelectionResult:\n""",
)
replace_once(
    "src/selection/service.py",
    """    eligible = [item for item in results if item[ELIGIBLE] and item[\"objective\"] is not None]\n""",
    """    if nested_candidate_results is not None:\n        proxy_by_candidate = {str(item[\"candidate\"]): item for item in results}\n        nested_by_candidate = {\n            str(item[\"candidate\"]): item\n            for item in nested_candidate_results\n            if isinstance(item.get(\"candidate\"), str)\n        }\n        results = [\n            {**proxy_by_candidate.get(candidate, {}), **nested_by_candidate[candidate]}\n            for candidate in candidates\n            if candidate != \"none\" and candidate in nested_by_candidate\n        ]\n    eligible = [item for item in results if item[ELIGIBLE] and item[\"objective\"] is not None]\n""",
)
replace_once(
    "src/selection/service.py",
    """            \"l3_fit_scope\": (l3_fold_result or {}).get(\"fit_scope\"),\n            \"outer_outcomes_accessed\": False,\n""",
    """            \"l3_fit_scope\": (l3_fold_result or {}).get(\"fit_scope\"),\n            \"nested_candidate_results\": nested_candidate_results or [],\n            \"outer_outcomes_accessed\": False,\n""",
)

# Store heldout L3 posterior targets so the same full-refit procedure can be used when L3 is activated.
for old, new in [
    (
        '                        "heldout_removed_from_target_and_measurement": True,\n',
        '                        "heldout_removed_from_target_and_measurement": True,\n                        "target_rows": [],\n',
    ),
    (
        '                        "heldout_removed_from_target_and_measurement": True,\n                    }\n                )\n                continue\n            losses: list[float] = []\n',
        '                        "heldout_removed_from_target_and_measurement": True,\n                        "target_rows": [],\n                    }\n                )\n                continue\n            losses: list[float] = []\n            heldout_target_rows: list[dict[str, object]] = []\n',
    ),
]:
    content = read("src/selection/service.py")
    if old in content:
        write("src/selection/service.py", content.replace(old, new, 1))

replace_once(
    "src/selection/service.py",
    """                if heldout_value is None or remaining_observed < minimum_observed_channels:\n                    continue\n                clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))\n""",
    """                if heldout_value is None or remaining_observed < minimum_observed_channels:\n                    continue\n                heldout_target_rows.append(\n                    {\n                        \"heldout_channel\": heldout,\n                        FIRM_ID: str(row[FIRM_ID]),\n                        FISCAL_YEAR: int(row[FISCAL_YEAR]),\n                        TARGET_VALUE: float(probability),\n                    }\n                )\n                clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))\n""",
)
replace_once(
    "src/selection/service.py",
    """                    \"fit_diagnostics\": fit.diagnostics,\n                }\n""",
    """                    \"fit_diagnostics\": fit.diagnostics,\n                    \"target_rows\": heldout_target_rows,\n                }\n""",
)
replace_once(
    "src/selection/service.py",
    """        \"target_rows\": target_rows,\n        \"fit_scope\": \"development_history_only\",\n""",
    """        \"target_rows\": target_rows,\n        \"heldout_target_rows\": [\n            target\n            for item in all_strict_results\n            if item.get(\"fixed_pi\") == selected_pi\n            for target in cast(list[dict[str, object]], item.get(\"target_rows\", []))\n        ],\n        \"fit_scope\": \"development_history_only\",\n""",
)

# Integrate the nested service into P10.
replace_once(
    "scripts/p10_select_measurement.py",
    """from typing import Any\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"src\"))\n\nfrom core.evidence_registry import logical_evidence_sources\n""",
    """from typing import Any, cast\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"src\"))\n\nimport pandas as pd\n\nfrom core.evidence_registry import logical_evidence_sources\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """from core.pipeline import load_run, mapping, sequence\nfrom core.rng import generator\n""",
    """from core.pipeline import load_run, mapping, physical_columns, sequence\nfrom core.rng import derive_seed, generator\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """from selection.service import fit_l3_fold_candidate, select_measurement\n""",
    """from selection.nested_refit import run_nested_channel_refit\nfrom selection.service import fit_l3_fold_candidate, select_measurement\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """    loaded.context.read(\"temporal_split_registry\", {})\n    loaded.context.read(\"channel_time_split_registry\", {})\n    loaded.context.read(\"fold_aware_weights\", {\"fold_id\": args.outer_fold})\n""",
    """    loaded.context.read(\"temporal_split_registry\", {})\n    loaded.context.read(\"channel_time_split_registry\", {})\n    weights = loaded.context.read(\"fold_aware_weights\", {\"fold_id\": args.outer_fold})\n    feature_panel = loaded.context.read(\"feature_panel\", {})\n    label_inputs = loaded.context.read(\"l0_l1_inputs\", {})\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """    fold_eligibility = loaded.context.read(\"fold_eligibility\", {})\n    feature_registry = loaded.context.read(\"feature_registry\", {})\n""",
    """    fold_eligibility = loaded.context.read(\"fold_eligibility\", {})\n    feature_registry = loaded.context.read(\"feature_registry\", {})\n    if not all(\n        isinstance(value, pd.DataFrame) for value in (weights, feature_panel, label_inputs)\n    ):\n        raise ValueError(\"P10 nested-refit inputs must be DataFrames\")\n    feature_registry_rows = [\n        mapping(item, \"feature registry item\")\n        for item in sequence(feature_registry, \"feature registry\")\n    ]\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """    result = select_measurement(\n        matrices=matrices,\n""",
    """    selection_config = mapping(measurement.get(\"selection\"), \"measurement.selection\")\n    if selection_config.get(\"nested_refit_required\") is not True:\n        raise RuntimeError(\"P10_NESTED_REFIT_NOT_LOCKED\")\n    learners = mapping(loaded.registry.get(\"learners\"), \"learners\")\n    tuning = mapping(learners.get(\"tuning\"), \"learners.tuning\")\n    gate1_learner_id = str(selection_config[\"gate1_learner_id\"])\n    gate1_feature_group = str(selection_config[\"gate1_feature_group\"])\n    nested = run_nested_channel_refit(\n        matrices=matrices,\n        feature_panel=cast(pd.DataFrame, feature_panel),\n        feature_registry=feature_registry_rows,\n        label_inputs=cast(pd.DataFrame, label_inputs),\n        weights=cast(pd.DataFrame, weights),\n        outer_year=int(args.outer_fold),\n        candidates=candidates,\n        l3_fold_result=l3_fold_result,\n        minimum_observed_channels=minimum_channels\n        if isinstance(minimum_channels, int)\n        else 1,\n        gate1_learner_id=gate1_learner_id,\n        gate1_feature_group=gate1_feature_group,\n        learner_settings=mapping(learners.get(\"settings\"), \"learners.settings\"),\n        learner_search_spaces=mapping(\n            tuning.get(\"search_spaces\"), \"learners.tuning.search_spaces\"\n        ),\n        maximum_valid_configurations=int(\n            tuning[\"max_valid_configurations_per_learner_inner_fold\"]\n        ),\n        columns=physical_columns(loaded.registry),\n        random_state=derive_seed(\n            loaded.protocol_hash, coordinates={OUTER_FOLD: args.outer_fold}, step_id=\"P10\", purpose=\"nested_channel_refit\"\n        )\n        % (2**32 - 1),\n    )\n\n    result = select_measurement(\n        matrices=matrices,\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """        minimum_observed_channels=minimum_channels,\n    )\n""",
    """        minimum_observed_channels=minimum_channels,\n        nested_candidate_results=cast(list[dict[str, Any]], nested.candidate_results),\n    )\n""",
)
replace_once(
    "scripts/p10_select_measurement.py",
    """    channel_selection = dict(result.channel_selection)\n""",
    """    channel_selection = {\n        **dict(result.channel_selection),\n        \"nested_refit_receipt\": nested.receipt,\n    }\n""",
)

# P11 validates and freezes the nested-selection receipt.
replace_once(
    "scripts/p11_freeze_models.py",
    """from selection.track_plan import executable_tracks\n""",
    """from selection.nested_refit import validate_nested_refit_receipt\nfrom selection.track_plan import executable_tracks\n""",
)
replace_once(
    "scripts/p11_freeze_models.py",
    """    plan_tracks = executable_tracks(selection)\n    primary_track = next(\n""",
    """    plan_tracks = executable_tracks(selection)\n    optional_measurements = [\n        str(item[MEASUREMENT_ID])\n        for item in plan_tracks\n        if item.get(\"role\") != \"required_primary\"\n    ]\n    nested_receipt = validate_nested_refit_receipt(\n        channel_selection,\n        outer_fold=args.outer_fold,\n        required_optional_measurements=optional_measurements,\n    )\n    primary_track = next(\n""",
)
replace_once(
    "scripts/p11_freeze_models.py",
    """        \"channel_measurement_selection_hash\": stable_hash(channel_selection),\n        \"feature_registry_hash\": stable_hash(feature_registry),\n""",
    """        \"channel_measurement_selection_hash\": stable_hash(channel_selection),\n        \"nested_selection_receipt_hash\": stable_hash(nested_receipt),\n        \"nested_selection_status\": nested_receipt.get(\"status\"),\n        \"optional_measurement_tracks\": optional_measurements,\n        \"feature_registry_hash\": stable_hash(feature_registry),\n""",
)

# P12 verifies P10/P11 hashes before opening sealed outcomes.
replace_once(
    "scripts/p12_evaluate.py",
    """from core.pipeline import load_run, mapping, physical_columns, sequence\n""",
    """from core.pipeline import load_run, mapping, physical_columns, sequence, stable_hash\n""",
)
replace_once(
    "scripts/p12_evaluate.py",
    """from evaluation.service import build_latent_risk_scenarios, evaluate_outer_fold\n""",
    """from evaluation.service import build_latent_risk_scenarios, evaluate_outer_fold\nfrom selection.nested_refit import validate_nested_refit_receipt\n""",
)
replace_once(
    "scripts/p12_evaluate.py",
    """    if freeze.get(\"protocol_hash\") != loaded.protocol_hash:\n        raise RuntimeError(\"model-freeze receipt protocol hash mismatch\")\n    open_receipt = {\n""",
    """    if freeze.get(\"protocol_hash\") != loaded.protocol_hash:\n        raise RuntimeError(\"model-freeze receipt protocol hash mismatch\")\n    channel_selection = mapping(\n        loaded.context.read(\"channel_measurement_selection\", coordinates),\n        \"channel measurement selection\",\n    )\n    optional_tracks_raw = freeze.get(\"optional_measurement_tracks\", [])\n    optional_tracks = [\n        str(value)\n        for value in sequence(optional_tracks_raw, \"freeze optional measurement tracks\")\n    ]\n    nested_receipt = validate_nested_refit_receipt(\n        channel_selection,\n        outer_fold=args.outer_fold,\n        required_optional_measurements=optional_tracks,\n    )\n    if freeze.get(\"channel_measurement_selection_hash\") != stable_hash(channel_selection):\n        raise RuntimeError(\"channel measurement selection hash mismatch\")\n    if freeze.get(\"nested_selection_receipt_hash\") != stable_hash(nested_receipt):\n        raise RuntimeError(\"nested selection receipt hash mismatch\")\n    open_receipt = {\n""",
)
replace_once(
    "scripts/p12_evaluate.py",
    """        \"mcse_report_hash\": loaded.context.store.receipt_hash(\"mcse_report\", {}),\n        \"opened_at_state\": \"FROZEN\",\n""",
    """        \"mcse_report_hash\": loaded.context.store.receipt_hash(\"mcse_report\", {}),\n        \"channel_measurement_selection_hash\": stable_hash(channel_selection),\n        \"nested_selection_receipt_hash\": stable_hash(nested_receipt),\n        \"opened_at_state\": \"FROZEN\",\n""",
)
replace_once(
    "scripts/p12_evaluate.py",
    """    channel_selection = mapping(\n        loaded.context.read(\"channel_measurement_selection\", coordinates),\n        \"channel measurement selection\",\n    )\n    latent_risk_scenarios = build_latent_risk_scenarios(\n""",
    """    latent_risk_scenarios = build_latent_risk_scenarios(\n""",
)

# Update the implementation note without claiming an empirical production run has completed.
replace_once(
    "docs/PIPELINE_P00_P17.md",
    """Giới hạn implementation hiện còn mở: strict channel score chưa chạy lại toàn bộ\nlearner/feature/tuning/calibration procedure cho từng `M*_{f,c}` và chưa truyền\nposterior-draw robustness qua toàn pipeline. Row-level fold-local L3 posterior đã\nđược tạo tại P10; P10 vẫn phải giữ `none` nếu diagnostics/evidence không đạt.\n""",
    """P10 thực hiện full-refit channel-within-time cho từng candidate × held-out channel\nbằng learner Gate 1 đã khóa: target, label model, feature, tuning và calibration đều\nloại held-out channel; chỉ development-history OOF predictions được dùng để tính\nsoft cross-entropy. Candidate thiếu bất kỳ channel cell nào phải `INSUFFICIENT_EVIDENCE`.\nP11 khóa receipt/hash của phép chọn này và P12 xác minh lại trước khi mở outer outcomes.\nPosterior-draw robustness của L3 vẫn chỉ khả dụng khi fixed-π grid và priors được khóa.\n""",
)

print("Applied P10-P12 full nesting integration")
