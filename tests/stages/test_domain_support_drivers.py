from __future__ import annotations

import numpy as np
import pandas as pd

from sensitivity.domain_support_diagnostics import (
    domain_score_diagnostics,
    feature_driver_row,
)


def _synthetic_domains() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260810)
    development_rows = 80
    held_rows = 80
    development = pd.DataFrame(
        {
            "firm_id": [f"D{value:03d}" for value in range(development_rows)],
            "shift_driver": rng.normal(0.0, 0.6, development_rows),
            "stable_feature": rng.normal(0.0, 1.0, development_rows),
        }
    )
    held = pd.DataFrame(
        {
            "firm_id": [f"H{value:03d}" for value in range(held_rows)],
            "shift_driver": rng.normal(4.0, 0.6, held_rows),
            "stable_feature": rng.normal(0.0, 1.0, held_rows),
        }
    )
    return development, held


def test_leave_one_feature_out_identifies_support_driver() -> None:
    development, held = _synthetic_domains()
    feature_ids = ["shift_driver", "stable_feature"]
    baseline = domain_score_diagnostics(
        development=development,
        held_test=held,
        feature_ids=feature_ids,
        firm_column="firm_id",
        lower=0.05,
        upper=0.95,
        crossfit_folds=5,
        random_state=41,
    )
    assert baseline.reason_code is None
    assert baseline.support_fraction is not None
    assert baseline.domain_auc is not None
    assert baseline.support_fraction < 0.80
    assert baseline.domain_auc > 0.95

    shift = feature_driver_row(
        feature_id="shift_driver",
        development=development,
        held_test=held,
        all_feature_ids=feature_ids,
        baseline=baseline,
        firm_column="firm_id",
        lower=0.05,
        upper=0.95,
        crossfit_folds=5,
        random_state=41,
        support_fraction_minimum=0.80,
    )
    stable = feature_driver_row(
        feature_id="stable_feature",
        development=development,
        held_test=held,
        all_feature_ids=feature_ids,
        baseline=baseline,
        firm_column="firm_id",
        lower=0.05,
        upper=0.95,
        crossfit_folds=5,
        random_state=41,
        support_fraction_minimum=0.80,
    )

    assert shift["support_gain_when_removed"] is not None
    assert stable["support_gain_when_removed"] is not None
    assert float(shift["support_gain_when_removed"]) > 0.50
    assert float(shift["support_gain_when_removed"]) > float(
        stable["support_gain_when_removed"]
    )
    assert shift["restores_locked_support_when_removed"] is True
    assert float(shift["absolute_standardized_mean_difference"]) > 3.0
    assert float(shift["domain_auc_drop_when_removed"]) > 0.30


def test_domain_score_diagnostics_preserves_held_row_alignment() -> None:
    development, held = _synthetic_domains()
    result = domain_score_diagnostics(
        development=development,
        held_test=held,
        feature_ids=["shift_driver", "stable_feature"],
        firm_column="firm_id",
        lower=0.05,
        upper=0.95,
        crossfit_folds=5,
        random_state=19,
    )

    assert result.reason_code is None
    assert len(result.held_scores) == len(held)
    assert len(result.held_supported) == len(held)
    assert result.crossfit_folds_used == 5
