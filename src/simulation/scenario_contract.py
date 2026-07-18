"""Scenario verification contract."""

from collections.abc import Mapping
from typing import Any

from core.semantic_keys import SCENARIO_ID


def validate_scenario_target_identity(
    scenario: Mapping[str, Any],
) -> None:
    """Validate that the scenario target identity matches its calibration target.

    Raises ValueError if incorrect.
    """
    scenario_id = scenario.get(SCENARIO_ID)
    target_id = scenario.get("calibration_target_id")

    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a nonempty string")

    if not isinstance(target_id, str) or not target_id:
        raise ValueError(
            f"scenario={scenario_id}: calibration_target_id is required"
        )

    marker = "__target_"
    if marker not in scenario_id:
        raise ValueError(
            f"scenario={scenario_id}: scenario_id must contain "
            f"'{marker}{target_id}'"
        )

    embedded_target = scenario_id.split(marker, 1)[1].split("__", 1)[0]

    if embedded_target != target_id:
        raise ValueError(
            f"scenario={scenario_id}: embedded target={embedded_target} "
            f"differs from calibration_target_id={target_id}"
        )
