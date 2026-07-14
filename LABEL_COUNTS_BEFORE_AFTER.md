# Label Counts Before–After

## Tổng quan

| Metric | Run cũ | Run mới |
|---|---:|---:|
| Protocol hash | `cad146317...3013e95` | `2f254055...45f7df8` |
| Raw S3 event rows | 535 | 535 |
| Accepted linked events | 385 | 385 |
| Mature L1 positives | 56 | 60 |
| Mature L1 explicit negatives | 0 | 0 |
| Mature L1 unknown | 21.647 | 21.643 |
| Mature risk-set rows | 21.703 | 21.703 |
| Prospective/immature rows | 1.708 | 1.708 |

Kết quả 60 không được ép bằng threshold. Nó phát sinh từ 61 event records trong primary 12-month horizon, trong đó NVL–2024 có hai events nên còn 60 unique firm-years.

## Event reconciliation

| Timing/status | Event rows |
|---|---:|
| Raw | 535 |
| Linked | 385 |
| Unlinked firm-year | 150 |
| Duplicate cluster | 0 |
| Excluded by source rule | 0 |
| Pre-prediction | 273 |
| Same-day | 0 |
| Trong 12 tháng | 61 |
| Thêm trong tháng 12–24 | 41 |
| Cumulative trong 24 tháng | 102 |
| Sau 24 tháng | 10 |
| Sau data cutoff | 2 |

Các timing category của 385 linked events reconcile đúng: `273 + 61 + 41 + 10 = 385`. Hai post-cutoff events thuộc raw/unlinked records và được báo riêng.

## Bốn firm-year thay đổi

| Firm-year | Pre-prediction event | In-horizon event | Nguyên nhân thay đổi |
|---|---|---|---|
| C69–2018 | 2018-10-08 | 2019-10-18 | Run cũ aggregate availability về event sớm trước khi lọc horizon |
| CVN–2024 | 2024-12-10 | 2025-07-18 | Tương tự; event trong horizon bị event sớm che |
| GKM–2024 | 2024-04-04 | 2025-05-28 | Tương tự |
| PSH–2024 | 2024-06-06 | 2025-10-09 | Tương tự |

Không firm-year nào bị remove hoặc đổi thành negative.

## L1 theo fiscal year

| Year | Mature | Old positive | New positive | Explicit negative | New mature unknown |
|---:|---:|---:|---:|---:|---:|
| 2015 | 2.244 | 4 | 4 | 0 | 2.240 |
| 2016 | 2.552 | 6 | 6 | 0 | 2.546 |
| 2017 | 2.604 | 3 | 3 | 0 | 2.601 |
| 2018 | 2.582 | 3 | 4 | 0 | 2.578 |
| 2019 | 2.033 | 4 | 4 | 0 | 2.029 |
| 2020 | 2.002 | 7 | 7 | 0 | 1.995 |
| 2021 | 1.965 | 10 | 10 | 0 | 1.955 |
| 2022 | 1.952 | 3 | 3 | 0 | 1.949 |
| 2023 | 1.899 | 9 | 9 | 0 | 1.890 |
| 2024 | 1.870 | 7 | 10 | 0 | 1.860 |
| 2025 | 0 | 0 | 0 | 0 | 0; 1.708 rows immature |

## Fold eligibility trước–sau

Run cũ ghi sai `mature_row_count` bằng số observed labels. New artifact lấy count từ risk set và báo đủ classes.

| Fold | Old artifact mature field | New mature rows | Old → new positives | New negatives | Classes | New role/reason |
|---:|---:|---:|---:|---:|---:|---|
| 2020 | 7 | 2.002 | 7 → 7 | 0 | 1 | `initial_separate / EXPLICIT_NEGATIVE_CLASS_MISSING` |
| 2021 | 10 | 1.965 | 10 → 10 | 0 | 1 | `prospective_or_descriptive / EXPLICIT_NEGATIVE_CLASS_MISSING` |
| 2022 | 3 | 1.952 | 3 → 3 | 0 | 1 | tương tự |
| 2023 | 9 | 1.899 | 9 → 9 | 0 | 1 | tương tự |
| 2024 | 7 | 1.870 | 7 → 10 | 0 | 1 | tương tự |
| 2026 | 0 | 0 | 0 → 0 | 0 | 0 | `prospective_separate / NO_OBSERVED_BINARY_CLASSES` |

Không fold nào đủ confirmatory Track A; positive count không thể bù cho missing explicit-negative class.

## Protocol và old-run immutability

- Old protocol hash: `cad146317f18d83dabd19ff8544f0518ce3da3c140492ab2b3fbce4ba3013e95`.
- New protocol hash: `2f254055decdca8a8d185e61e82586b971c939787c7992fd22da8190d45f7df8`.
- Old-run tree trước thay đổi: 102 files, 19.155.686 bytes, SHA-256 `8901475cc3397b07ea5c970b2bcd080f815a7270bb2f51393cc9972d3a6239f0`.
- Old-run tree sau hai run mới và audit: cùng file count, bytes và SHA-256 `8901475cc3397b07ea5c970b2bcd080f815a7270bb2f51393cc9972d3a6239f0`.

Protocol hash thay đổi do source outcome semantics, event-level schema, locked snapshot mapping, fold/access controls và implementation commit thay đổi. Raw source file hashes không bị sửa.

## Machine-readable evidence

- `docs/audits/event_reconciliation.csv`
- `docs/audits/l0_counts_by_source_year.csv`
- `docs/audits/l1_counts_by_year_fold.csv`
- `docs/audits/channel_overlap_capability.csv`
- `docs/audits/label_taxonomy_mapping.csv`
- `docs/audits/label_pipeline_reconciliation_summary.json`

