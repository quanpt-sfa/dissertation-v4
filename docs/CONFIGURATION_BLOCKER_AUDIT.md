# Configuration blocker audit

This audit intentionally records unresolved research choices. No value in this
table is imputed from outer outcomes or changed by this implementation.

| Setting | Current value | Status | Effect |
| --- | --- | --- | --- |
| `measurement.primary_target_id` | `null` | `LEGACY_NULL_MIGRATION` | P10-P12 resolve the locked primary from `measurement.execution_tracks.primary_target_id`; the legacy field no longer blocks sequential execution. |
| `measurement.execution_tracks.primary_target_id` | `L1_ANNUAL` | `LOCKED_PRIMARY_TRACK` | L1 is required and executes first; outer-performance target selection is forbidden. |
| S1 materiality thresholds | `null` | `BLOCKING_PRIMARY_TRACK` | L1 audit-adjustment components cannot become confirmatory until the thresholds are locked. |
| S1 denominator floor | `null` | `BLOCKING_PRIMARY_TRACK` | S1 materiality denominator remains unresolved. |
| L2 formula | `null` | `SKIP_OPTIONAL_TRACK` | L2 is recorded as unavailable and skipped without blocking L1. |
| L2 minimum observed channels | `null` | `SKIP_OPTIONAL_TRACK` | L2 eligibility is not locked; only the L2 track is skipped. |
| L2 source quality | empty mapping | `SKIP_OPTIONAL_TRACK` | Quality-weighted L2 scoring is unavailable; only the L2 track is skipped. |
| L2 delay half-life | `null` | `SKIP_OPTIONAL_TRACK` | Delay weighting is unavailable; only the L2 track is skipped. |
| L3 fixed-pi grid | empty list | `SKIP_OPTIONAL_TRACK` | L3 fixed-pi cannot become operational; only the L3 track is skipped. |
| L3 accuracy priors | empty mapping | `SKIP_OPTIONAL_TRACK` | L3 accuracy model cannot become operational; only the L3 track is skipped. |
| `s3_taxonomy.sanction_source_completeness.source_year_close_date` | `null` | `SAFE_TO_REMAIN_NULL` | Completeness is explicitly controlled by year registry. |
