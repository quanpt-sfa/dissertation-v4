# S1/S2 label counts and capability

## Run identity

- Run ID: `dissertation-2015-2026-s1-s2-annual-v1`
- Protocol hash: `0385ad907cc4a17cdc53b4723a2c29fd34cf540ae5935381d2fcf5f00fb2e913`
- Firm-year rows: 23.411
- Mature at 12 months: 21.703
- Prospective: 1.708
- Evidence ledger: 70.618 rows = 70.233 annual results + 385 delayed events

## Source-level L0

| Source | Opportunity observed | Positive | Explicit negative | Unknown | Status |
|---|---:|---:|---:|---:|---|
| S1 profit adjustment | 12.578 | 0 | 0 | 23.411 | Empirical rule not locked |
| S1 revenue adjustment | 12.209 | 0 | 0 | 23.411 | Empirical rule not locked |
| S2 audit opinion | 16.489 | 2.413 | 14.076 | 6.922 | Operational |
| S3 sanction evidence | Opportunity unknown | 60 | 0 | 23.351 | Delayed verification operational |

S1 opportunity counts exclude 75 matched pairs có denominator bằng 0. S1 outcomes remain unknown until denominator floor and endpoint thresholds are locked.

## L1 by year and fold

| Fiscal year | Mature rows | Positive | Explicit negative | Unknown | Fold role |
|---:|---:|---:|---:|---:|---|
| 2015 | 2.244 | 12 | 0 | 2.232 | Development history |
| 2016 | 2.552 | 180 | 0 | 2.372 | Development history |
| 2017 | 2.604 | 189 | 0 | 2.415 | Development history |
| 2018 | 2.582 | 197 | 0 | 2.385 | Development history |
| 2019 | 2.033 | 222 | 0 | 1.811 | Development history |
| 2020 | 2.002 | 271 | 0 | 1.731 | Initial separate |
| 2021 | 1.965 | 267 | 0 | 1.698 | Prospective/descriptive |
| 2022 | 1.952 | 310 | 0 | 1.642 | Prospective/descriptive |
| 2023 | 1.899 | 326 | 0 | 1.573 | Prospective/descriptive |
| 2024 | 1.870 | 285 | 0 | 1.585 | Prospective/descriptive |
| 2025 | 0 | 0 | 0 | 1.708 | Prospective, no outer fold |

Toàn bộ L1 có 2.259 positive, 0 explicit negative và 21.152 unknown. S2 có clean negatives, nhưng L1 vẫn unknown khi S3 opportunity unknown và không có positive. Vì vậy mọi evaluated outer fold chỉ có một observed class và không được đi vào confirmatory Track A.

Machine-readable table: [l1_counts_after_s1_s2.csv](docs/audits/l1_counts_after_s1_s2.csv).

## Channel overlap

Observed-outcome patterns:

| Pattern | Eligible | Mature | Prospective |
|---|---:|---:|---:|
| None | 6.914 | 6.719 | 195 |
| S2 only | 16.437 | 14.924 | 1.513 |
| S3 only | 8 | 8 | 0 |
| S2 ∩ S3 | 52 | 52 | 0 |

Observed-opportunity patterns:

| Pattern | Eligible | Mature | Prospective |
|---|---:|---:|---:|
| None | 5.838 | 5.717 | 121 |
| S1 only | 1.084 | 1.010 | 74 |
| S2 only | 4.991 | 4.578 | 413 |
| S1 ∩ S2 | 11.498 | 10.398 | 1.100 |

S3 không xuất hiện trong opportunity overlap vì source opportunity chưa quan sát được. Bảng đầy đủ và observed-channel-count distribution nằm tại [channel_overlap_after_s1_s2.csv](docs/audits/channel_overlap_after_s1_s2.csv).

## L2 feasibility

Implementation L2 đã tồn tại nhưng status là `EMPIRICALLY_PENDING` với reason `L2_SCORING_FORMULA_NOT_LOCKED`. Nếu các scoring parameters được khóa, 16.497 firm-years hiện có ít nhất một observed channel; chỉ 52 rows có hai observed channels. Không có S1 observed outcome trước khi materiality rules được khóa.

Các config keys còn thiếu:

- `measurement.l2_scoring.formula`;
- `measurement.l2_scoring.source_quality_by_profile`;
- `measurement.l2_scoring.delay_half_life_days`;
- `measurement.l2_missingness.minimum_observed_channels` nếu protocol yêu cầu coverage floor.

## L3 feasibility

S2 và S3 là hai valid channels, nên lỗi cấu trúc `INSUFFICIENT_CHANNELS` đã được giải quyết. Có 52 rows overlap S2–S3. Tuy nhiên L3 vẫn `EMPIRICALLY_PENDING` vì:

- `measurement.l3_model.operational.fixed_pi_grid` rỗng;
- `measurement.l3_model.operational.accuracy_priors_by_profile` rỗng.

Không có posterior giả được tạo. Hierarchical π vẫn sensitivity-only.

## Anchor-PU

S3 vẫn là high-confirmation anchor: 60 mature positives, zero explicit negatives và unknown opportunity. Anchor-PU có anchor positives về mặt dữ liệu, nhưng production modeling chưa khả thi vì feature registry rỗng và P08 simulation status là `SKIPPED/NO_SIMULATION_BATCHES`. Không có confirmatory model nào được fit.

