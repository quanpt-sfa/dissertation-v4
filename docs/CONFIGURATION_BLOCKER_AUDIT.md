# Configuration blocker audit

This audit intentionally records unresolved research choices. No value in this
table is imputed from outer outcomes or changed by this implementation.

| Setting | Current value | Status | Effect |
| --- | --- | --- | --- |
| `measurement.primary_target_id` | `null` | `BLOCKING_CONFIRMATORY` | P10-P12 remain fail-closed. |
| S1 materiality thresholds | `null` | `BLOCKING_CONFIRMATORY` | S1 adjustment endpoint cannot become confirmatory. |
| S1 denominator floor | `null` | `BLOCKING_CONFIRMATORY` | S1 materiality denominator remains unresolved. |
| L2 formula | `null` | `BLOCKING_OPTIONAL_TRACK_B` | L2 remains empirically pending. |
| L2 minimum observed channels | `null` | `BLOCKING_OPTIONAL_TRACK_B` | L2 eligibility is not locked. |
| L2 source quality | empty mapping | `BLOCKING_OPTIONAL_TRACK_B` | Quality-weighted L2 scoring is unavailable. |
| L2 delay half-life | `null` | `BLOCKING_OPTIONAL_TRACK_B` | Delay weighting is unavailable. |
| L3 fixed-pi grid | empty list | `BLOCKING_OPTIONAL_TRACK_B` | L3 fixed-pi pilot cannot become operational. |
| L3 accuracy priors | empty mapping | `BLOCKING_OPTIONAL_TRACK_B` | L3 accuracy model cannot become operational. |
| `s3_taxonomy.sanction_source_completeness.source_year_close_date` | `null` | `SAFE_TO_REMAIN_NULL` | Completeness is explicitly controlled by year registry. |
