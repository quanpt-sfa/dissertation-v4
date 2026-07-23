from __future__ import annotations

from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from core.fold_control import require_primary_target
from core.pipeline import physical_columns
from core.registry_compiler import compile_registry
from core.semantic_keys import (
    ANNUAL_MEASUREMENT_MATURE,
    AVAILABILITY_DATE,
    CHANNEL_ID,
    DOCUMENT_ID,
    ELIGIBLE,
    FIRM_ID,
    FISCAL_YEAR,
    MATURE,
    OUTCOME,
    PREDICTION_TIME,
    REQUIRED_SANCTION_YEAR,
    ROW_INCLUSION,
    S3_NEXT_YEAR_MATURE,
    SANCTION_YEAR,
    SOURCE_ID,
    SOURCE_OPPORTUNITY,
    SOURCE_YEAR_COMPLETE,
    TARGET_FISCAL_YEAR,
    TARGET_ID,
    TEMPORAL_ROLE,
)
from evidence.sanctions import (
    SanctionDecisionInput,
    build_s3_evidence,
    classify_s3_taxonomy,
    resolve_sanction_year,
    target_fiscal_year,
)
from measurement.service import build_measurement_inputs, summarize_fold_eligibility
from risksets.service import build_risk_set

ROOT = Path(__file__).resolve().parents[2]


@cache
def _registry() -> dict[str, object]:
    return cast(dict[str, object], compile_registry(ROOT / "config" / "pipeline.yaml").registry)


@cache
def _columns() -> dict[str, str]:
    return physical_columns(_registry())


@cache
def _taxonomy() -> dict[str, Any]:
    return cast(dict[str, Any], _registry()["s3_taxonomy"])


def _decision(
    document_id: str,
    *,
    firm_id: str = "F1",
    sanction_year: int | None = 2024,
    decision_date: datetime | None = None,
    publish_date: datetime | None = None,
    decision_number: str | None = None,
    label_known_date: datetime | None = None,
    affected_fiscal_year: int | None = None,
    level_1: str | None = "GOVERNANCE",
    code: str | None = None,
    source_ref: str | None = None,
    row_included: bool = True,
    hard_positive: bool = True,
) -> SanctionDecisionInput:
    return SanctionDecisionInput(
        document_id=document_id,
        firm_id=firm_id,
        sanction_year=sanction_year,
        decision_date=decision_date,
        publish_date=publish_date,
        decision_number=decision_number,
        label_known_date=label_known_date,
        affected_fiscal_year=affected_fiscal_year,
        primary_violation_l1=level_1,
        normalized_violation_code=code,
        source_ref=source_ref,
        row_included=row_included,
        hard_positive=hard_positive,
    )


def _build(
    decisions: list[SanctionDecisionInput],
    *,
    panel_keys: set[tuple[str, int]] | None = None,
) -> object:
    return build_s3_evidence(
        panel_keys=panel_keys or {("F1", 2023)},
        decisions=decisions,
        taxonomy=_taxonomy(),
        complete_through_year=2025,
        incomplete_years={2026},
        columns=_columns(),
    )


def _record_map(result: object) -> dict[tuple[str, str, int], object]:
    records = cast(Any, result).endpoint_records
    return {(row.source_id, row.firm_id, row.fiscal_year): row for row in records}


def test_sanction_year_2024_maps_to_target_fiscal_year_2023() -> None:
    row = _decision("DOC-1")
    assert target_fiscal_year(row) == 2023
    ledger = cast(Any, _build([row])).decision_ledger
    assert ledger.loc[0, _columns()[TARGET_FISCAL_YEAR]] == 2023


def test_february_and_october_decisions_map_to_the_same_prior_fiscal_year() -> None:
    rows = [
        _decision("FEB", sanction_year=None, decision_date=datetime(2024, 2, 1)),
        _decision("OCT", sanction_year=None, decision_date=datetime(2024, 10, 1)),
    ]
    assert [target_fiscal_year(row) for row in rows] == [2023, 2023]


def test_march_31_boundary_does_not_change_s3_target_year() -> None:
    before = _decision("BEFORE", sanction_year=None, decision_date=datetime(2024, 3, 30))
    on_anchor = _decision("ON", sanction_year=None, decision_date=datetime(2024, 3, 31))
    after = _decision("AFTER", sanction_year=None, decision_date=datetime(2024, 4, 1))
    assert {target_fiscal_year(row) for row in (before, on_anchor, after)} == {2023}


def test_label_known_date_is_provenance_and_does_not_change_target_year() -> None:
    early = _decision("EARLY", label_known_date=datetime(2024, 1, 1))
    late = _decision("LATE", label_known_date=datetime(2025, 12, 31))
    assert target_fiscal_year(early) == target_fiscal_year(late) == 2023


def test_affected_fiscal_year_never_overrides_next_calendar_year_target() -> None:
    row = _decision("DOC", affected_fiscal_year=2021)
    ledger = cast(Any, _build([row])).decision_ledger
    assert target_fiscal_year(row) == 2023
    assert ledger.loc[0, _columns()[TARGET_FISCAL_YEAR]] == 2023


def test_s3_target_does_not_apply_prediction_date_or_horizon_filter() -> None:
    row = _decision("DOC", decision_date=datetime(2024, 2, 1))
    result = _build([row], panel_keys={("F1", 2023)})
    broad = _record_map(result)[("S3_BROAD", "F1", 2023)]
    assert cast(Any, broad).outcome is True


def test_document_id_is_the_decision_key_and_number_is_only_provenance() -> None:
    result = cast(
        Any,
        _build(
            [
                _decision("DOC-A", decision_number="01/QD"),
                _decision("DOC-B", decision_number="01/QD"),
            ]
        ),
    )
    assert result.audit["unique_decision_count"] == 2
    assert set(result.decision_ledger[_columns()[DOCUMENT_ID]]) == {"DOC-A", "DOC-B"}
    broad = _record_map(result)[("S3_BROAD", "F1", 2023)]
    assert cast(Any, broad).decision_count == 2


def test_same_document_and_firm_is_deduplicated_deterministically() -> None:
    rows = [
        _decision("DOC-1", source_ref="row-2"),
        _decision("DOC-1", source_ref="row-1"),
        _decision("DOC-1", source_ref="row-1"),
    ]
    first = cast(Any, _build(rows))
    second = cast(Any, _build(list(reversed(rows))))
    pd.testing.assert_frame_equal(first.decision_ledger, second.decision_ledger)
    assert first.audit["duplicate_source_row_count"] == 2
    assert len(first.decision_ledger) == 1


def test_same_document_for_two_firms_preserves_mappings_but_one_decision_count() -> None:
    result = cast(
        Any,
        _build(
            [_decision("DOC-1", firm_id="F1"), _decision("DOC-1", firm_id="F2")],
            panel_keys={("F1", 2023), ("F2", 2023)},
        ),
    )
    assert len(result.decision_ledger) == 2
    assert result.audit["firm_mapping_count"] == 2
    assert result.audit["unique_decision_count"] == 1
    records = _record_map(result)
    assert cast(Any, records[("S3_BROAD", "F1", 2023)]).decision_count == 1
    assert cast(Any, records[("S3_BROAD", "F2", 2023)]).decision_count == 1


def test_multiple_decisions_collapse_to_one_positive_firm_year_endpoint() -> None:
    result = cast(Any, _build([_decision("DOC-1"), _decision("DOC-2")]))
    broad = _record_map(result)[("S3_BROAD", "F1", 2023)]
    assert cast(Any, broad).outcome is True
    assert cast(Any, broad).decision_count == 2


def test_complete_source_year_without_event_creates_endpoint_specific_false() -> None:
    records = _record_map(_build([], panel_keys={("F1", 2024)}))
    assert {cast(Any, record).outcome for record in records.values()} == {False}
    assert all(cast(Any, record).source_opportunity is True for record in records.values())


def test_incomplete_source_year_without_event_remains_unknown() -> None:
    records = _record_map(_build([], panel_keys={("F1", 2025)}))
    assert all(cast(Any, record).outcome is None for record in records.values())
    assert all(cast(Any, record).source_opportunity is None for record in records.values())


def test_unresolved_sanction_year_cannot_create_complete_year_false() -> None:
    row = _decision("DOC", sanction_year=None, decision_date=None, publish_date=None)
    records = _record_map(_build([row]))
    assert all(cast(Any, record).outcome is None for record in records.values())
    assert all(
        cast(Any, record).outcome_basis == "SANCTION_YEAR_UNRESOLVED" for record in records.values()
    )


def test_riskset_2024_requires_complete_2025_source_year() -> None:
    result = build_risk_set(
        panel=pd.DataFrame(
            [
                {
                    _columns()[FIRM_ID]: "F1",
                    _columns()[FISCAL_YEAR]: 2024,
                    _columns()[PREDICTION_TIME]: "2025-03-31",
                }
            ]
        ),
        data_cutoff=datetime(2026, 3, 31),
        horizon_months=12,
        columns=_columns(),
        sanction_complete_through_year=2025,
        sanction_incomplete_years={2026},
    )
    row = result.risk_sets.iloc[0]
    assert row[_columns()[REQUIRED_SANCTION_YEAR]] == 2025
    assert bool(row[_columns()[SOURCE_YEAR_COMPLETE]]) is True
    assert bool(row[_columns()[S3_NEXT_YEAR_MATURE]]) is True


def test_riskset_2025_requires_incomplete_2026_source_year() -> None:
    result = build_risk_set(
        panel=pd.DataFrame(
            [
                {
                    _columns()[FIRM_ID]: "F1",
                    _columns()[FISCAL_YEAR]: 2025,
                    _columns()[PREDICTION_TIME]: "2026-03-31",
                }
            ]
        ),
        data_cutoff=datetime(2026, 3, 31),
        horizon_months=12,
        columns=_columns(),
        sanction_complete_through_year=2025,
        sanction_incomplete_years={2026},
    )
    row = result.risk_sets.iloc[0]
    assert row[_columns()[REQUIRED_SANCTION_YEAR]] == 2026
    assert bool(row[_columns()[SOURCE_YEAR_COMPLETE]]) is False
    assert bool(row[_columns()[S3_NEXT_YEAR_MATURE]]) is False


def test_content_is_always_a_subset_of_reporting() -> None:
    values, reason = classify_s3_taxonomy(
        _decision("DOC", level_1=None, code="FS_FALSE_MISLEADING"), _taxonomy()
    )
    assert reason is None
    assert values["S3_CONTENT"] is True
    assert values["S3_REPORTING"] is True
    assert values["S3_BROAD"] is True


def test_timeliness_is_always_a_subset_of_reporting() -> None:
    values, reason = classify_s3_taxonomy(
        _decision("DOC", level_1=None, code="LATE_PERIODIC_DISCLOSURE"), _taxonomy()
    )
    assert reason is None
    assert values["S3_TIMELINESS"] is True
    assert values["S3_REPORTING"] is True
    assert values["S3_BROAD"] is True


def test_unmapped_taxonomy_enters_broad_only_and_other_endpoints_remain_unknown() -> None:
    values, reason = classify_s3_taxonomy(
        _decision("DOC", level_1="UNMAPPED", code="UNMAPPED_CODE"), _taxonomy()
    )
    assert reason == "UNMAPPED_S3_TAXONOMY"
    assert values == {
        "S3_BROAD": True,
        "S3_REPORTING": None,
        "S3_CONTENT": None,
        "S3_TIMELINESS": None,
    }


def test_unmapped_broad_positive_retains_taxonomy_reason() -> None:
    records = _record_map(_build([_decision("DOC", level_1="UNMAPPED", code="UNMAPPED_CODE")]))
    broad = cast(Any, records[("S3_BROAD", "F1", 2023)])
    reporting = cast(Any, records[("S3_REPORTING", "F1", 2023)])
    assert broad.outcome is True
    assert broad.taxonomy_reason_code == "UNMAPPED_S3_TAXONOMY"
    assert reporting.outcome is None
    assert reporting.decision_count == 1


def test_excluded_decision_does_not_suppress_complete_year_endpoint_false() -> None:
    result = cast(Any, _build([_decision("DOC", row_included=False)]))
    records = _record_map(result)
    assert all(cast(Any, record).outcome is False for record in records.values())
    assert all(cast(Any, record).decision_count == 0 for record in records.values())
    assert bool(result.decision_ledger.loc[0, _columns()[ROW_INCLUSION]]) is False


def test_nonendpoint_decision_is_not_counted_in_false_endpoint_metadata() -> None:
    records = _record_map(_build([_decision("DOC", level_1="GOVERNANCE", code=None)]))
    broad = cast(Any, records[("S3_BROAD", "F1", 2023)])
    reporting = cast(Any, records[("S3_REPORTING", "F1", 2023)])
    assert broad.outcome is True
    assert broad.decision_count == 1
    assert reporting.outcome is False
    assert reporting.decision_count == 0


def test_governance_and_issuance_do_not_enter_reporting_without_override() -> None:
    for level_1 in ("GOVERNANCE", "ISSUANCE_OFFERING"):
        values, reason = classify_s3_taxonomy(
            _decision("DOC", level_1=level_1, code=None), _taxonomy()
        )
        assert reason is None
        assert values["S3_BROAD"] is True
        assert values["S3_REPORTING"] is False


def test_l1_annual_is_independent_of_unknown_s3() -> None:
    columns = {
        FIRM_ID: "firm",
        FISCAL_YEAR: "year",
        PREDICTION_TIME: "anchor",
        MATURE: "mature",
        ANNUAL_MEASUREMENT_MATURE: "annual_mature",
        S3_NEXT_YEAR_MATURE: "s3_mature",
        ELIGIBLE: "eligible",
        SOURCE_ID: "source",
        CHANNEL_ID: "channel",
        AVAILABILITY_DATE: "available",
        OUTCOME: "outcome",
        SOURCE_OPPORTUNITY: "opportunity",
        TEMPORAL_ROLE: "temporal",
        TARGET_ID: "target",
    }
    risk = pd.DataFrame(
        [
            {
                "firm": "F1",
                "year": 2023,
                "anchor": "2024-03-31",
                "mature": False,
                "annual_mature": True,
                "s3_mature": False,
                "eligible": True,
            }
        ]
    )
    evidence = pd.DataFrame(
        [
            {
                "firm": "F1",
                "year": 2023,
                "source": source,
                "channel": channel,
                "available": "2024-03-31",
                "outcome": False,
                "opportunity": True,
                "temporal": "annual_measurement_at_anchor",
            }
            for source, channel in (("S1", "S1"), ("S2", "S2"))
        ]
        + [
            {
                "firm": "F1",
                "year": 2023,
                "source": "S3_BROAD",
                "channel": "S3",
                "available": pd.NaT,
                "outcome": pd.NA,
                "opportunity": pd.NA,
                "temporal": "next_calendar_year_regulatory_event",
            }
        ]
    )
    result = build_measurement_inputs(
        risk_sets=risk,
        evidence=evidence,
        expected_sources={"S1": "S1", "S2": "S2", "S3_BROAD": "S3"},
        horizon_months=12,
        columns=columns,
        pending_status="EMPIRICALLY_PENDING",
        unavailable_status="UNAVAILABLE_BY_DESIGN",
        insufficient_channels_reason="INSUFFICIENT_CHANNELS",
        anchor_source_ids=["S3_BROAD"],
        source_temporal_roles={
            "S1": "annual_measurement_at_anchor",
            "S2": "annual_measurement_at_anchor",
            "S3_BROAD": "next_calendar_year_regulatory_event",
        },
        explicit_negative_allowed={"S1": True, "S2": True, "S3_BROAD": True},
        candidate_targets={"L1_ANNUAL": ["S1", "S2"], "S3_BROAD": ["S3_BROAD"]},
    )
    rows = result.inputs.set_index("target")["outcome"]
    assert bool(rows["L1_ANNUAL"]) is False
    assert pd.isna(rows["S3_BROAD"])
    assert not result.sealed_outcomes.duplicated(["firm", "year", "target"]).any()


def test_sealed_outcomes_are_unique_by_firm_year_target() -> None:
    result = cast(Any, _build([_decision("DOC")]))
    records = result.endpoint_records
    assert len({(row.firm_id, row.fiscal_year, row.source_id) for row in records}) == len(records)


def test_fold_eligibility_counts_are_bound_to_target_id() -> None:
    columns = _columns()
    firm = columns[FIRM_ID]
    year = columns[FISCAL_YEAR]
    target = columns[TARGET_ID]
    outcome = columns[OUTCOME]
    mature = columns[MATURE]
    eligible = columns[ELIGIBLE]
    target_maturity = pd.DataFrame(
        [
            {firm: "F1", year: 2021, target: target_id, mature: True, eligible: True}
            for target_id in ("L1_ANNUAL", "S3_BROAD")
        ]
        + [
            {firm: "F2", year: 2021, target: target_id, mature: True, eligible: True}
            for target_id in ("L1_ANNUAL", "S3_BROAD")
        ]
    )
    sealed = pd.DataFrame(
        [
            {firm: "F1", year: 2021, target: "L1_ANNUAL", outcome: True},
            {firm: "F2", year: 2021, target: "L1_ANNUAL", outcome: False},
            {firm: "F1", year: 2021, target: "S3_BROAD", outcome: True},
        ]
    )
    rows = summarize_fold_eligibility(
        sealed_outcomes=sealed,
        target_maturity=target_maturity,
        initial_outer_year=2020,
        confirmatory_years=[2021],
        prospective_year=2026,
        confirmatory_positive_minimum=1,
        sensitivity_positive_range=(1, 1),
        columns=columns,
    )
    bound = {str(row[TARGET_ID]): row for row in rows if row["outer_fold"] == "2021"}
    assert bound["L1_ANNUAL"]["observed_binary_class_count"] == 2
    assert bound["L1_ANNUAL"]["assigned_role"] == "confirmatory"
    assert bound["S3_BROAD"]["observed_binary_class_count"] == 1
    assert bound["S3_BROAD"]["assigned_role"] == "prospective_or_descriptive"


def test_missing_primary_target_blocks_confirmatory_production() -> None:
    with pytest.raises(RuntimeError, match="PRIMARY_TARGET_NOT_LOCKED"):
        require_primary_target({"primary_target_id": None}, "P10")


def test_resolve_sanction_year_precedence() -> None:
    from core.semantic_keys import DECISION_DATE, PUBLISH_DATE, TARGET_FISCAL_YEAR

    # Case 1: All fields populated -> sanction_year
    row1 = SanctionDecisionInput(
        document_id="DOC-1",
        firm_id="F1",
        sanction_year=2020,
        decision_date=datetime(2021, 5, 5),
        publish_date=datetime(2022, 6, 6),
        target_fiscal_year=2018,
    )
    val, source = resolve_sanction_year(row1)
    assert val == 2020
    assert source == SANCTION_YEAR

    # Case 2: sanction_year is None, others populated -> decision_date
    row2 = SanctionDecisionInput(
        document_id="DOC-2",
        firm_id="F1",
        sanction_year=None,
        decision_date=datetime(2021, 5, 5),
        publish_date=datetime(2022, 6, 6),
        target_fiscal_year=2018,
    )
    val, source = resolve_sanction_year(row2)
    assert val == 2021
    assert source == DECISION_DATE

    # Case 3: sanction_year and decision_date are None, others populated -> publish_date
    row3 = SanctionDecisionInput(
        document_id="DOC-3",
        firm_id="F1",
        sanction_year=None,
        decision_date=None,
        publish_date=datetime(2022, 6, 6),
        target_fiscal_year=2018,
    )
    val, source = resolve_sanction_year(row3)
    assert val == 2022
    assert source == PUBLISH_DATE

    # Case 4: Only target_fiscal_year is populated -> target_fiscal_year + 1
    row4 = SanctionDecisionInput(
        document_id="DOC-4",
        firm_id="F1",
        sanction_year=None,
        decision_date=None,
        publish_date=None,
        target_fiscal_year=2018,
    )
    val, source = resolve_sanction_year(row4)
    assert val == 2019
    assert source == TARGET_FISCAL_YEAR


def test_sanction_rows_parser_with_target_fiscal_year(tmp_path: Path) -> None:
    import importlib.util

    from core.semantic_keys import (
        DOCUMENT_ID,
        FIRM_ID,
        HARD_POSITIVE,
        ROW_INCLUSION,
        TARGET_FISCAL_YEAR,
    )
    from p01.models import SourceSpec
    from p02.models import EntityResolutionSpec

    path = ROOT / "scripts" / "p03_evidence_ledger.py"
    spec_location = importlib.util.spec_from_file_location("test_p03_script", path)
    assert spec_location is not None and spec_location.loader is not None
    module = importlib.util.module_from_spec(spec_location)
    spec_location.loader.exec_module(module)
    _sanction_rows = getattr(module, "_sanction_rows")

    # Write temporary CSV file
    csv_file = tmp_path / "test_sanctions.csv"
    csv_file.write_text(
        "ticker,doc_id,include_flag,positive_flag,target_year\nAAA,DOC-100,1,1,2019\n",
        encoding="utf-8",
    )

    entity = EntityResolutionSpec.from_mapping(
        {
            "policy": "registered_only",
            "allow_identity_mapping": True,
            "collision_policy": "fail",
            "normalization": {
                "unicode_nfkc": True,
                "trim": True,
                "uppercase": True,
                "collapse_internal_whitespace": True,
            },
            "aliases": {},
            "reporting_calendar": {
                "default_fiscal_year_end_month_day": "12-31",
                "firm_exceptions": {},
                "early_report_exceptions": [],
            },
        }
    )

    source_spec = SourceSpec.from_mapping(
        "sanction_evidence",
        {
            "enabled": True,
            "channel_id": "S3",
            "source_type": "official",
            "source_agency": "Agency",
            "original_unit": "firm-year",
            "related_period_field": None,
            "availability_date_field": None,
            "availability_date_source": "test",
            "coverage_dimensions": [],
            "role": "evidence",
            "verification_status": "observed",
            "data_risks": [],
            "relative_path": "test_sanctions.csv",
            "format": "csv",
            "encoding": "utf-8",
            "delimiter": ",",
            "locked_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "schema": {
                "required_columns": [
                    "ticker",
                    "doc_id",
                    "include_flag",
                    "positive_flag",
                    "target_year",
                ],
                "optional_columns": [],
                "key_columns": ["ticker", "doc_id"],
                "date_columns": [],
                "required_date_columns": [],
                "numeric_columns": {},
                "allow_extra_columns": True,
                "key_unique": False,
                "row_count_min": 1,
            },
        },
    )

    semantics = {
        FIRM_ID: "ticker",
        DOCUMENT_ID: "doc_id",
        ROW_INCLUSION: "include_flag",
        HARD_POSITIVE: "positive_flag",
        TARGET_FISCAL_YEAR: "target_year",
    }

    parsed_list = _sanction_rows(
        source_id="sanction_evidence",
        path=csv_file,
        spec=source_spec,
        semantics=semantics,
        entity=entity,
    )

    assert len(parsed_list) == 1
    parsed = parsed_list[0]
    assert parsed.target_fiscal_year == 2019
    assert target_fiscal_year(parsed) == 2019


def test_decision_date_precedes_target_fiscal_year_fallback() -> None:
    row = SanctionDecisionInput(
        document_id="DOC",
        firm_id="F1",
        decision_date=datetime(2024, 7, 1),
        target_fiscal_year=2018,
    )

    from core.semantic_keys import DECISION_DATE

    assert resolve_sanction_year(row) == (2024, DECISION_DATE)
    assert target_fiscal_year(row) == 2023


def test_target_fiscal_year_is_final_fallback() -> None:
    row = SanctionDecisionInput(
        document_id="DOC",
        firm_id="F1",
        target_fiscal_year=2019,
    )

    from core.semantic_keys import TARGET_FISCAL_YEAR

    assert resolve_sanction_year(row) == (2020, TARGET_FISCAL_YEAR)
    assert target_fiscal_year(row) == 2019


def test_excluded_unresolved_decision_does_not_contaminate_firm_years() -> None:
    result = _build(
        [
            _decision(
                "DISCLOSURE-ONLY",
                sanction_year=None,
                decision_date=None,
                publish_date=None,
                row_included=False,
                hard_positive=False,
                level_1="DISCLOSURE",
            )
        ],
        panel_keys={("F1", 2019), ("F1", 2020)},
    )

    records = _record_map(result)

    assert all(record.outcome is False for record in records.values())
    assert result.audit["unresolved_sanction_year_mapping_count"] == 0
    assert result.audit["excluded_source_rule_mapping_count"] == 1
