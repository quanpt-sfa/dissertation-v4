from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.semantic_keys import FIRM_ID, FISCAL_YEAR, OUTCOME, TARGET_ID
from sensitivity.service import select_observed_target_outcomes

ROOT = Path(__file__).resolve().parents[2]


def _columns() -> dict[str, str]:
    return {
        FIRM_ID: "firm_master_id",
        FISCAL_YEAR: "fiscal_year",
        TARGET_ID: "target_id",
        OUTCOME: "outcome",
    }


def test_target_aware_outcomes_select_only_primary_observed_rows() -> None:
    outcomes = pd.DataFrame(
        [
            {
                "firm_master_id": "E1031",
                "fiscal_year": 2015,
                "target_id": "L0:S1_profit_adjustment",
                "outcome": False,
            },
            {
                "firm_master_id": "E1031",
                "fiscal_year": 2015,
                "target_id": "L1_ANNUAL",
                "outcome": True,
            },
            {
                "firm_master_id": "E1031",
                "fiscal_year": 2015,
                "target_id": "S3_BROAD",
                "outcome": False,
            },
            {
                "firm_master_id": "E2042",
                "fiscal_year": 2015,
                "target_id": "L1_ANNUAL",
                "outcome": pd.NA,
            },
            {
                "firm_master_id": "E2042",
                "fiscal_year": 2015,
                "target_id": "S3_BROAD",
                "outcome": True,
            },
        ]
    )

    selected = select_observed_target_outcomes(
        outcomes,
        target_id="L1_ANNUAL",
        columns=_columns(),
        context="test",
    )

    assert selected.to_dict(orient="records") == [
        {"firm_master_id": "E1031", "fiscal_year": 2015, "outcome": True}
    ]
    assert not selected.duplicated(["firm_master_id", "fiscal_year"]).any()


def test_duplicate_rows_within_primary_target_fail_closed() -> None:
    outcomes = pd.DataFrame(
        [
            {
                "firm_master_id": "E1031",
                "fiscal_year": 2015,
                "target_id": "L1_ANNUAL",
                "outcome": True,
            },
            {
                "firm_master_id": "E1031",
                "fiscal_year": 2015,
                "target_id": "L1_ANNUAL",
                "outcome": False,
            },
        ]
    )

    with pytest.raises(RuntimeError, match="not unique by firm-year"):
        select_observed_target_outcomes(
            outcomes,
            target_id="L1_ANNUAL",
            columns=_columns(),
            context="test",
        )


def test_p13_and_p16_bind_sealed_outcomes_to_primary_target() -> None:
    p13 = (ROOT / "scripts" / "p13_sensitivity.py").read_text(encoding="utf-8")
    p16 = (ROOT / "scripts" / "p16_gate3.py").read_text(encoding="utf-8")

    assert 'primary_target_id = require_primary_target(measurement, "P13")' in p13
    assert p13.count("evaluation_target_id=primary_target_id") == 2
    assert 'primary_target_id = require_primary_target(measurement, "P16")' in p16
    assert "select_observed_target_outcomes(" in p16
    assert "outcomes=observed_outcomes" in p16
