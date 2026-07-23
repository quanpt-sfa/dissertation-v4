from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement, found {count}: {old[:90]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "src/selection/service.py",
    '''                if heldout_value is None or remaining_observed < minimum_observed_channels:
                    continue
                heldout_target_rows.append(
                    {
                        "heldout_channel": heldout,
                        FIRM_ID: str(row[FIRM_ID]),
                        FISCAL_YEAR: int(row[FISCAL_YEAR]),
                        TARGET_VALUE: float(probability),
                    }
                )
                clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
''',
    '''                if remaining_observed < minimum_observed_channels:
                    continue
                heldout_target_rows.append(
                    {
                        "heldout_channel": heldout,
                        FIRM_ID: str(row[FIRM_ID]),
                        FISCAL_YEAR: int(row[FISCAL_YEAR]),
                        TARGET_VALUE: float(probability),
                    }
                )
                if heldout_value is None:
                    continue
                clipped = min(1.0 - 1e-6, max(1e-6, float(probability)))
''',
)

replace_once(
    "scripts/p10_select_measurement.py",
    '''    channel_selection = {
        **dict(result.channel_selection),
        "nested_refit_receipt": nested.receipt,
    }
''',
    '''    nested_cells = nested.receipt.get("cell_results")
    if not isinstance(nested_cells, list):
        raise RuntimeError("P10_NESTED_REFIT_CELL_RESULTS_MISSING")
    proxy_channel_diagnostics = result.channel_selection.get("strict_channel_results", [])
    channel_selection = {
        **dict(result.channel_selection),
        "proxy_only_channel_diagnostics": proxy_channel_diagnostics,
        "strict_channel_results": nested_cells,
        "nested_refit_receipt": nested.receipt,
    }
''',
)

old_validator = '''def validate_nested_refit_receipt(
    channel_selection: Mapping[str, object],
    *,
    outer_fold: str,
    required_optional_measurements: Sequence[str],
) -> dict[str, object]:
    """Validate the P10 nested-selection receipt before P11/P12 may proceed."""
    raw = channel_selection.get("nested_refit_receipt")
    if not isinstance(raw, dict):
        raise RuntimeError("NESTED_CHANNEL_REFIT_RECEIPT_MISSING")
    receipt = _string_object_dict(cast(object, raw), "nested-refit receipt")
    if str(receipt.get(OUTER_FOLD)) != str(outer_fold):
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_FOLD_MISMATCH")
    if receipt.get("fit_scope") != "development_history_only":
        raise RuntimeError("NESTED_CHANNEL_REFIT_SCOPE_INVALID")
    if receipt.get("outer_outcomes_accessed") is not False:
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_OUTCOME_FIREWALL_BREACH")
    if receipt.get("outer_rows_used_in_selection") != 0:
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_ROWS_USED")
    removed = receipt.get("heldout_channel_removed_from")
    if not isinstance(removed, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")
    removed_values = [str(value) for value in cast(list[object], removed)]
    if set(removed_values) != set(_REMOVED_INPUTS):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")

    candidate_rows_raw = receipt.get("candidate_results")
    if not isinstance(candidate_rows_raw, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_CANDIDATES_MISSING")
    candidate_rows = [
        _string_object_dict(cast(object, item), "nested-refit candidate")
        for item in cast(list[object], candidate_rows_raw)
        if isinstance(item, dict)
    ]
    by_candidate = {str(item.get("candidate")): item for item in candidate_rows}
    for measurement_id in required_optional_measurements:
        candidate = by_candidate.get(str(measurement_id))
        if candidate is None or candidate.get(ELIGIBLE) is not True:
            raise RuntimeError(
                f"NESTED_CHANNEL_REFIT_REQUIRED_CANDIDATE_INCOMPLETE:{measurement_id}"
            )
    if required_optional_measurements and receipt.get("status") != "PASS":
        raise RuntimeError("NESTED_CHANNEL_REFIT_REQUIRED_PASS_MISSING")
    return receipt
'''
new_validator = '''def validate_nested_refit_receipt(
    channel_selection: Mapping[str, object],
    *,
    outer_fold: str,
    required_optional_measurements: Sequence[str],
) -> dict[str, object]:
    """Validate every nested candidate-by-channel cell before P11/P12 proceed."""
    raw = channel_selection.get("nested_refit_receipt")
    if not isinstance(raw, dict):
        raise RuntimeError("NESTED_CHANNEL_REFIT_RECEIPT_MISSING")
    receipt = _string_object_dict(cast(object, raw), "nested-refit receipt")
    if str(receipt.get(OUTER_FOLD)) != str(outer_fold):
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_FOLD_MISMATCH")
    if receipt.get("selection_procedure") != "full_refit_channel_within_time":
        raise RuntimeError("NESTED_CHANNEL_REFIT_PROCEDURE_INVALID")
    if receipt.get("fit_scope") != "development_history_only":
        raise RuntimeError("NESTED_CHANNEL_REFIT_SCOPE_INVALID")
    if receipt.get("outer_outcomes_accessed") is not False:
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_OUTCOME_FIREWALL_BREACH")
    if receipt.get("outer_rows_used_in_selection") != 0:
        raise RuntimeError("NESTED_CHANNEL_REFIT_OUTER_ROWS_USED")
    if receipt.get("complete_channel_grid_required") is not True:
        raise RuntimeError("NESTED_CHANNEL_REFIT_COMPLETE_GRID_NOT_REQUIRED")

    removed = receipt.get("heldout_channel_removed_from")
    if not isinstance(removed, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")
    removed_values = [str(value) for value in cast(list[object], removed)]
    if set(removed_values) != set(_REMOVED_INPUTS):
        raise RuntimeError("NESTED_CHANNEL_REFIT_REMOVAL_CONTRACT_INVALID")

    expected_raw = receipt.get("expected_channels")
    if not isinstance(expected_raw, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_EXPECTED_CHANNELS_MISSING")
    expected_channels = [str(value) for value in cast(list[object], expected_raw)]
    if not expected_channels or len(expected_channels) != len(set(expected_channels)):
        raise RuntimeError("NESTED_CHANNEL_REFIT_EXPECTED_CHANNELS_INVALID")

    candidate_rows_raw = receipt.get("candidate_results")
    cell_rows_raw = receipt.get("cell_results")
    if not isinstance(candidate_rows_raw, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_CANDIDATES_MISSING")
    if not isinstance(cell_rows_raw, list):
        raise RuntimeError("NESTED_CHANNEL_REFIT_CELLS_MISSING")
    candidate_rows = [
        _string_object_dict(cast(object, item), "nested-refit candidate")
        for item in cast(list[object], candidate_rows_raw)
        if isinstance(item, dict)
    ]
    cell_rows = [
        _string_object_dict(cast(object, item), "nested-refit cell")
        for item in cast(list[object], cell_rows_raw)
        if isinstance(item, dict)
    ]
    by_candidate = {str(item.get("candidate")): item for item in candidate_rows}
    for measurement_id_raw in required_optional_measurements:
        measurement_id = str(measurement_id_raw)
        candidate = by_candidate.get(measurement_id)
        if candidate is None or candidate.get(ELIGIBLE) is not True:
            raise RuntimeError(
                f"NESTED_CHANNEL_REFIT_REQUIRED_CANDIDATE_INCOMPLETE:{measurement_id}"
            )
        objective = candidate.get("objective")
        if (
            not isinstance(objective, (int, float))
            or isinstance(objective, bool)
            or not math.isfinite(float(objective))
        ):
            raise RuntimeError(f"NESTED_CHANNEL_REFIT_OBJECTIVE_INVALID:{measurement_id}")
        if (
            candidate.get("required_heldout_channels") != len(expected_channels)
            or candidate.get("completed_heldout_channels") != len(expected_channels)
        ):
            raise RuntimeError(f"NESTED_CHANNEL_REFIT_COUNTS_INVALID:{measurement_id}")

        cells = [item for item in cell_rows if str(item.get("candidate")) == measurement_id]
        heldout_channels = [str(item.get(_HELDOUT_CHANNEL)) for item in cells]
        if len(cells) != len(expected_channels) or set(heldout_channels) != set(
            expected_channels
        ):
            raise RuntimeError(f"NESTED_CHANNEL_REFIT_CHANNEL_GRID_INCOMPLETE:{measurement_id}")
        for cell in cells:
            heldout = str(cell.get(_HELDOUT_CHANNEL))
            row_count = cell.get("rows")
            cell_objective = cell.get("soft_cross_entropy")
            if cell.get("status") != "PASS":
                raise RuntimeError(
                    f"NESTED_CHANNEL_REFIT_CELL_NOT_PASS:{measurement_id}:{heldout}"
                )
            if (
                not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count < 1
            ):
                raise RuntimeError(
                    f"NESTED_CHANNEL_REFIT_CELL_ROWS_INVALID:{measurement_id}:{heldout}"
                )
            if (
                not isinstance(cell_objective, (int, float))
                or isinstance(cell_objective, bool)
                or not math.isfinite(float(cell_objective))
            ):
                raise RuntimeError(
                    f"NESTED_CHANNEL_REFIT_CELL_OBJECTIVE_INVALID:{measurement_id}:{heldout}"
                )
            if (
                cell.get("fit_scope") != "development_history_only"
                or cell.get("outer_rows_used") != 0
                or cell.get("outer_outcomes_accessed") is not False
            ):
                raise RuntimeError(
                    f"NESTED_CHANNEL_REFIT_CELL_SCOPE_INVALID:{measurement_id}:{heldout}"
                )
            if any(
                cell.get(flag) is not True
                for flag in (
                    "heldout_removed_from_target",
                    "heldout_removed_from_label_model",
                    "heldout_removed_from_features",
                    "heldout_removed_from_tuning",
                    "heldout_removed_from_calibration",
                )
            ):
                raise RuntimeError(
                    f"NESTED_CHANNEL_REFIT_CELL_REMOVAL_INVALID:{measurement_id}:{heldout}"
                )
    if required_optional_measurements and receipt.get("status") != "PASS":
        raise RuntimeError("NESTED_CHANNEL_REFIT_REQUIRED_PASS_MISSING")
    return receipt
'''
replace_once("src/selection/nested_refit.py", old_validator, new_validator)

replace_once(
    "scripts/p12_evaluate.py",
    '''    channel_selection = mapping(
        loaded.context.read("channel_measurement_selection", coordinates),
        "channel measurement selection",
    )
''',
    '''    measurement_selection = mapping(
        loaded.context.read("measurement_selection_registry", coordinates),
        "measurement selection",
    )
    if freeze.get("measurement_selection_hash") != stable_hash(measurement_selection):
        raise RuntimeError("measurement selection hash mismatch")
    channel_selection = mapping(
        loaded.context.read("channel_measurement_selection", coordinates),
        "channel measurement selection",
    )
''',
)
replace_once(
    "scripts/p12_evaluate.py",
    '''        "mcse_report_hash": loaded.context.store.receipt_hash("mcse_report", {}),
        "channel_measurement_selection_hash": stable_hash(channel_selection),
''',
    '''        "mcse_report_hash": loaded.context.store.receipt_hash("mcse_report", {}),
        "measurement_selection_hash": stable_hash(measurement_selection),
        "channel_measurement_selection_hash": stable_hash(channel_selection),
''',
)

# Extend regression tests for cell-level validation and L3 inclusion independence.
replace_once(
    "tests/stages/test_p10_nested_refit.py",
    "from typing import cast\n",
    "from typing import Any, cast\n",
)
replace_once(
    "tests/stages/test_p10_nested_refit.py",
    "import pandas as pd\nimport pytest\n",
    "import numpy as np\nimport pandas as pd\nimport pytest\n\nimport selection.service as selection_service\nfrom labels.latent_class import LatentClassResult\n",
)

helper = '''\n\ndef _valid_receipt() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for heldout, objective in (("S1", 0.41), ("S2", 0.39)):
        cells.append(
            {
                "candidate": "L2",
                "heldout_channel": heldout,
                "status": "PASS",
                "reason_code": None,
                "rows": 12,
                "soft_cross_entropy": objective,
                "fit_scope": "development_history_only",
                "outer_rows_used": 0,
                "outer_outcomes_accessed": False,
                "heldout_removed_from_target": True,
                "heldout_removed_from_label_model": True,
                "heldout_removed_from_features": True,
                "heldout_removed_from_tuning": True,
                "heldout_removed_from_calibration": True,
            }
        )
    return {
        "status": "PASS",
        "reason_code": None,
        OUTER_FOLD: "2021",
        "selection_procedure": "full_refit_channel_within_time",
        "fit_scope": "development_history_only",
        "outer_outcomes_accessed": False,
        "outer_rows_used_in_selection": 0,
        "complete_channel_grid_required": True,
        "heldout_channel_removed_from": [
            "target",
            "label_model",
            "features",
            "tuning",
            "calibration",
        ],
        "expected_channels": ["S1", "S2"],
        "gate1_learner_id": "elastic_net_logistic",
        "gate1_feature_group": "full",
        "candidate_results": [
            {
                "candidate": "L2",
                ELIGIBLE: True,
                "objective": 0.4,
                "required_heldout_channels": 2,
                "completed_heldout_channels": 2,
            }
        ],
        "cell_results": cells,
    }
'''
replace_once(
    "tests/stages/test_p10_nested_refit.py",
    "\ndef test_nested_receipt_fails_closed_before_freeze_or_outer_open() -> None:\n",
    helper + "\n\ndef test_nested_receipt_fails_closed_before_freeze_or_outer_open() -> None:\n",
)

old_receipt = '''    receipt = {
        "status": "PASS",
        OUTER_FOLD: "2021",
        "fit_scope": "development_history_only",
        "outer_outcomes_accessed": False,
        "outer_rows_used_in_selection": 0,
        "heldout_channel_removed_from": [
            "target",
            "label_model",
            "features",
            "tuning",
            "calibration",
        ],
        "candidate_results": [{"candidate": "L2", ELIGIBLE: True, "objective": 0.5}],
    }
'''
replace_once(
    "tests/stages/test_p10_nested_refit.py",
    old_receipt,
    "    receipt = _valid_receipt()\n",
)

extra_assertions = '''\n    missing_cell = {
        **receipt,
        "cell_results": cast(list[dict[str, object]], receipt["cell_results"])[:-1],
    }
    with pytest.raises(RuntimeError, match="CHANNEL_GRID_INCOMPLETE"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": missing_cell},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )

    bad_cells = [dict(item) for item in cast(list[dict[str, object]], receipt["cell_results"])]
    bad_cells[0]["heldout_removed_from_calibration"] = False
    bad_removal = {**receipt, "cell_results": bad_cells}
    with pytest.raises(RuntimeError, match="CELL_REMOVAL_INVALID"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": bad_removal},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )
'''
replace_once(
    "tests/stages/test_p10_nested_refit.py",
    '''    with pytest.raises(RuntimeError, match="REQUIRED_CANDIDATE_INCOMPLETE"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": incomplete},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )


def test_nested_results_override_proxy_only_measurement_scores() -> None:
''',
    '''    with pytest.raises(RuntimeError, match="REQUIRED_CANDIDATE_INCOMPLETE"):
        validate_nested_refit_receipt(
            {"nested_refit_receipt": incomplete},
            outer_fold="2021",
            required_optional_measurements=["L2"],
        )
''' + extra_assertions + '''\n\ndef test_nested_results_override_proxy_only_measurement_scores() -> None:
''',
)

l3_test = '''\n\ndef test_l3_target_inclusion_does_not_depend_on_heldout_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: list[dict[str, Any]] = [
        {
            FIRM_ID: "F0",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 1,
            "source_outcomes": {"a": None, "b": True},
            "channel_outcomes": {"S1": None, "S2": True},
        },
        {
            FIRM_ID: "F1",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 2,
            "source_outcomes": {"a": True, "b": False},
            "channel_outcomes": {"S1": True, "S2": False},
        },
        {
            FIRM_ID: "F2",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 1,
            "source_outcomes": {"a": False, "b": None},
            "channel_outcomes": {"S1": False, "S2": None},
        },
        {
            FIRM_ID: "F3",
            FISCAL_YEAR: 2020,
            MATURE: True,
            "observed_channel_count": 2,
            "source_outcomes": {"a": True, "b": True},
            "channel_outcomes": {"S1": True, "S2": True},
        },
    ]

    def fake_fit_l3(
        *,
        rows: list[dict[str, Any]],
        source_channels: dict[str, str],
        accuracy_priors: dict[str, dict[str, float]],
        fixed_pi: float,
        mcmc: dict[str, Any],
        rng: np.random.Generator,
    ) -> LatentClassResult:
        _ = (accuracy_priors, fixed_pi, mcmc, rng)
        probabilities = [0.2 + 0.15 * index for index in range(len(rows))]
        return LatentClassResult(
            posterior_mean=probabilities,
            posterior_draws=[[0 for _ in rows]],
            parameter_draws=[],
            source_accuracy={source: {} for source in source_channels},
            channel_random_effect_sd={
                channel: 0.0 for channel in set(source_channels.values())
            },
            diagnostics={"eligible_for_gate1": True},
        )

    monkeypatch.setattr(selection_service, "_fit_l3", fake_fit_l3)
    result = selection_service.fit_l3_fold_candidate(
        matrices={"rows": rows},
        outer_year=2021,
        source_channels={"a": "S1", "b": "S2"},
        accuracy_priors={"a": {}, "b": {}},
        fixed_pi_grid=[0.1],
        mcmc={},
        minimum_observed_channels=1,
        robust_fraction=1.0,
        rng=np.random.default_rng(9),
    )
    assert result["status"] == "PASS"
    targets = cast(list[dict[str, object]], result["heldout_target_rows"])
    included = {(str(item["heldout_channel"]), str(item[FIRM_ID])) for item in targets}
    assert ("S1", "F0") in included
    assert ("S2", "F2") in included
'''
with Path("tests/stages/test_p10_nested_refit.py").open("a", encoding="utf-8") as handle:
    handle.write(l3_test)

print("Applied P10-P12 semantic hardening")
