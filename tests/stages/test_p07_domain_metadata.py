from __future__ import annotations

import pandas as pd
import pytest

from features.domain_metadata import restore_domain_metadata


def test_restore_domain_metadata_replaces_placeholder_values() -> None:
    feature_panel = pd.DataFrame(
        {
            "firm_master_id": ["A", "B"],
            "fiscal_year": [2021, 2021],
            "exchange_or_board": pd.Series([pd.NA, pd.NA], dtype="string"),
            "x": [1.0, 2.0],
        }
    )
    source_panel = pd.DataFrame(
        {
            "firm_master_id": ["A", "B"],
            "fiscal_year": [2021, 2021],
            "exchange_or_board": ["HOSE", "HNX"],
        }
    )

    restored = restore_domain_metadata(
        feature_panel=feature_panel,
        source_panel=source_panel,
        domain_columns=["exchange_or_board"],
        firm_column="firm_master_id",
        year_column="fiscal_year",
        allowed_values_by_column={"exchange_or_board": ["HOSE", "HNX"]},
    )

    assert restored["exchange_or_board"].tolist() == ["HOSE", "HNX"]
    assert restored["x"].tolist() == [1.0, 2.0]


def test_restore_domain_metadata_fails_closed_on_missing_values() -> None:
    feature_panel = pd.DataFrame(
        {
            "firm_master_id": ["A", "B"],
            "fiscal_year": [2021, 2021],
            "x": [1.0, 2.0],
        }
    )
    source_panel = pd.DataFrame(
        {
            "firm_master_id": ["A", "B"],
            "fiscal_year": [2021, 2021],
            "exchange_or_board": ["HOSE", pd.NA],
        }
    )

    with pytest.raises(ValueError, match="exchange_or_board.*incomplete.*1"):
        restore_domain_metadata(
            feature_panel=feature_panel,
            source_panel=source_panel,
            domain_columns=["exchange_or_board"],
            firm_column="firm_master_id",
            year_column="fiscal_year",
            allowed_values_by_column={"exchange_or_board": ["HOSE", "HNX"]},
        )


def test_restore_domain_metadata_rejects_values_outside_locked_vocabulary() -> None:
    feature_panel = pd.DataFrame(
        {
            "firm_master_id": ["A", "B"],
            "fiscal_year": [2021, 2021],
            "x": [1.0, 2.0],
        }
    )
    source_panel = pd.DataFrame(
        {
            "firm_master_id": ["A", "B"],
            "fiscal_year": [2021, 2021],
            "exchange_or_board": ["HOSE", "UPCOM"],
        }
    )

    with pytest.raises(ValueError, match="outside the locked vocabulary.*UPCOM"):
        restore_domain_metadata(
            feature_panel=feature_panel,
            source_panel=source_panel,
            domain_columns=["exchange_or_board"],
            firm_column="firm_master_id",
            year_column="fiscal_year",
            allowed_values_by_column={"exchange_or_board": ["HOSE", "HNX"]},
        )
