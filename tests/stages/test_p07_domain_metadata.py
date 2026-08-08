from __future__ import annotations

import pandas as pd

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
    )

    assert restored["exchange_or_board"].tolist() == ["HOSE", "HNX"]
    assert restored["x"].tolist() == [1.0, 2.0]
