"""Regression tests for endpoint-specific S3 opportunity semantics."""

from __future__ import annotations

import pytest

from evidence.final_firm_year import (
    _effective_opportunity,  # pyright: ignore[reportPrivateUsage]
    _resolve_wide_s3_outcome,  # pyright: ignore[reportPrivateUsage]
)


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


def test_observed_opportunity_preserves_provenance_backed_undetermined_endpoint() -> None:
    outcome, reason = _resolve_wide_s3_outcome(
        effective_opportunity=True,
        raw_outcome=None,
        has_provenance=True,
        context=("E706", 2024, "S3_CONTENT"),
    )

    assert outcome is None
    assert reason == "S3_ENDPOINT_UNDETERMINED_WITH_OBSERVED_PROVENANCE"


def test_observed_opportunity_missing_endpoint_without_provenance_fails_closed() -> None:
    with pytest.raises(
        ValueError,
        match="observed S3 opportunity has missing endpoint without provenance",
    ):
        _resolve_wide_s3_outcome(
            effective_opportunity=True,
            raw_outcome=None,
            has_provenance=False,
            context=("F1", 2024, "S3_CONTENT"),
        )


def test_positive_endpoint_without_observation_opportunity_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive S3 endpoint lacks observation opportunity"):
        _resolve_wide_s3_outcome(
            effective_opportunity=False,
            raw_outcome=True,
            has_provenance=True,
            context=("F1", 2024, "S3_CONTENT"),
        )
