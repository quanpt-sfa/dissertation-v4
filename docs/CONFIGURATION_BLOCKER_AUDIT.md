# Configuration blocker audit

This audit intentionally records unresolved research choices. No value in this
table is imputed from outer outcomes or changed by this implementation.

| Setting | Current value | Status | Effect |
| --- | --- | --- | --- |
| `measurement.primary_target_id` | `null` | `LEGACY_NULL_MIGRATION` | P09-P12 resolve the locked primary from `measurement.execution_tracks.primary_target_id`; the legacy field no longer blocks sequential execution. |
| `measurement.execution_tracks.primary_target_id` | `L1_ANNUAL` | `LOCKED_PRIMARY_TRACK` | L1 is required and executes first; outer-performance target selection is forbidden. |
| S1 profit materiality threshold | `0.10` | `LOCKED_PRIMARY_RULE` | A profit adjustment is positive when the absolute pre/post-audit difference exceeds 10% of absolute post-audit profit. |
| S1 revenue materiality threshold | `0.01` | `LOCKED_PRIMARY_RULE` | A revenue adjustment is positive when the absolute pre/post-audit difference exceeds 1% of absolute post-audit revenue. |
| S1 denominator floor | `0.0` | `LOCKED_ZERO_ONLY_GUARD` | No arbitrary VND floor is imposed; exact zero denominators remain invalid under the existing fail-closed rule. |
| L2 formula | `quality_delay_weighted_observed_source_mean` | `LOCKED_NEUTRAL_L2` | L2 scores observed evidence only and preserves missingness. |
| L2 minimum observed channels | `1` | `LOCKED_NEUTRAL_L2` | A row requires at least one observed channel; stricter complete-channel subsets remain sensitivity analyses. |
| L2 source quality | all registered evidence profiles `1.0` | `LOCKED_NEUTRAL_L2` | No source-quality ranking is imposed without external validation or a ground truth. |
| L2 delay half-life | `365` days | `LOCKED_NEUTRAL_L2` | Delay discounting is aligned with the locked 12-month primary horizon. |
| L3 logical-source binding | logical endpoint IDs | `IMPLEMENTED` | Physical sources with multiple endpoints, especially S3, expand to separate logical outcomes before latent-class fitting. |
| L3 fixed-pi grid | empty list | `DATA_CALIBRATION_REQUIRED` | Run the development-only calibration report and review external prevalence evidence before locking a broad fixed-pi grid. |
| L3 accuracy priors | empty mapping | `THEORY_AND_EVIDENCE_REQUIRED` | Beta priors require documented literature, validation evidence, or expert elicitation plus prior-sensitivity analysis. |
| `s3_taxonomy.sanction_source_completeness.source_year_close_date` | `null` | `SAFE_TO_REMAIN_NULL` | Completeness is explicitly controlled by year registry. |
