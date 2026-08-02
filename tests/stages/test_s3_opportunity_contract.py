"""Regression tests for endpoint-specific S3 opportunity semantics."""

from __future__ import annotations

import pytest

from evidence.final_firm_year import _effective_opportunity  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("source_opportunity", "observation_opportunity", "expected"),
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (False, False, False),
        (False, None, False),
        (None, False, False),
        (True, None, None),
        (None, True, None),
        (None, None, None),
    ],
)
def test_effective_opportunity_requires_both_observed_layers(
    source_opportunity: bool | None,
    observation_opportunity: bool | None,
    expected: bool | None,
) -> None:
    assert _effective_opportunity(source_opportunity, observation_opportunity) is expected
