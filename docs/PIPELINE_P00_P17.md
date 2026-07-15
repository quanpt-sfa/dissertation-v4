# Đặc tả pipeline P00–P17

Tài liệu này mô tả implementation hiện tại trong repo. Các danh sách reads/writes
chính xác được sinh từ registry tại `docs/generated/`; nếu tài liệu thủ công và
generated catalog khác nhau, generated catalog cùng `config/pipeline.yaml` là
nguồn kiểm tra cuối cùng.

## 1. Nguyên tắc xuyên suốt

- Mỗi run có một `run-id` và thư mục bất biến.
- Nội dung semantic của snapshot (file/hash/schema/header/binding) là một phần của
  protocol hash. `snapshot_id`, thời điểm capture và detached integrity hash không
  làm thay đổi semantic protocol hash khi bytes/schema/binding giống hệt nhau.
- Sau P00, stage chỉ đọc `registry.lock.json`, không đọc module YAML trực tiếp.
- Mọi artifact chính thức đi qua `RunContext` và `ArtifactStore`.
- Physical column được giải từ registry; stage chỉ dùng logical semantic keys.
- Thiếu nguồn không phải bằng chứng zero.
- Follow-up chưa trưởng thành không phải outcome âm.
- Content predictors không được vào label model.
- Hierarchical-π chỉ là sensitivity, không được vào Gate 1 selection.
- Outer outcomes được niêm phong đến P12.
- K1–K4 được niêm phong đến P15.
- P17 chỉ báo cáo; không fit, calibrate hoặc tạo nhãn lại.

## 2. Lệnh vận hành một lần

Chạy trong root repo:

```powershell
uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-start-v1 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P17
```

`--through Pxx` dừng sau stage tương ứng. Runner mặc định yêu cầu Git tree sạch;
`--allow-dirty` chỉ dùng cho phát triển có chủ ý, không dùng để chấp nhận run cuối.
`--resume` chỉ tiếp tục cùng run sau khi runner xác minh lại raw SHA-256, code,
config, snapshot, manifest và content hash của từng unit đã hoàn tất.

Luồng runner:

```text
snapshot → P00 → P01 từng source → P02 → ... → P17
```

Runner không gọi script đăng ký nguồn/panel/alias thủ công. Source được nhận diện
chỉ từ `config/methodology/source_catalog.yaml` và file do người dùng đặt đúng chỗ.

## 3. Trạng thái và đơn vị chạy lại

| Stage | Trạng thái vào | Trạng thái ra | Đơn vị partition/rerun |
| --- | --- | --- | --- |
| P00 | `CONFIGURED` | `LOCKED` | một run |
| P01 | `LOCKED` | `AUDITED` | một `source_id` |
| P02 | `AUDITED` | `PANELLED` | một run |
| P03 | `PANELLED` | `LEDGERED` | một run, tổng hợp mọi evidence source |
| P04 | `LEDGERED` | `RISK_SET` | một run/horizon chính |
| P05 | `RISK_SET` | `MEASURED` | một run/horizon chính |
| P06 | `MEASURED` | `OBSERVABLE` | một run |
| P07 | `OBSERVABLE` | `FEATURED` | một run |
| P08 | `FEATURED` | `SIMULATED` | `scenario_id × method_id × batch_id` |
| P09 | `SIMULATED` | `SPLIT` | registry chung; weight theo `fold_id` |
| P10 | `SPLIT` | `SELECTED` | một `outer_fold` |
| P11 | `SELECTED` | `FROZEN` | một `outer_fold` |
| P12 | `FROZEN` | `EVALUATED` | một `outer_fold` |
| P13 | `EVALUATED` | `SENSITIVITY` | một run; đọc mọi outer fold đã hoàn tất |
| P14 | `SENSITIVITY` | `GATE2` | một run |
| P15 | `GATE2` | `KNOWN_CASES_OPEN` | một run |
| P16 | `KNOWN_CASES_OPEN` | `GATE3` | một run |
| P17 | `GATE3` | `REPORTED` | một run |

Các state là nhãn access-control. Artifact và receipt mới là bằng chứng vật lý để
một stage downstream được phép tiếp tục.

## 4. Snapshot trước P00

Script: `scripts/create_data_snapshot.py`.

Snapshot engine:

1. đọc duy nhất các glob đã đăng ký;
2. kiểm tra cardinality;
3. đọc sheet/header/schema thật;
4. ánh xạ semantic field;
5. tính SHA-256 và kích thước;
6. ghi relative path, row count, columns và schema hash;
7. tính `snapshot_content_hash` semantic và detached `snapshot_hash` bảo vệ chính
   file manifest;
8. đưa snapshot path cho P00.

Thêm file, thay nội dung, đổi schema/header hoặc đổi semantic binding đều làm
snapshot/protocol thay đổi. Snapshot không copy dữ liệu sang vị trí khác.

## 5. P00 — Khóa protocol

Script: `scripts/p00_lock_protocol.py`.

CLI trực tiếp:

```powershell
uv run python scripts/p00_lock_protocol.py `
  --config config/pipeline.yaml `
  --run-id <run-id> `
  --output-root artifacts/runs `
  --snapshot-manifest <data_snapshot.json>
```

Mục tiêu:

- compile toàn bộ module từ `config/pipeline.yaml`;
- kiểm tra mỗi semantic setting chỉ có một owner;
- kiểm tra artifact path/schema/coordinate collision;
- dựng access matrix và decision traceability;
- quan sát môi trường, Git, package lock và source hashes;
- khóa known-case seal và protocol hash.

Reads: config source và snapshot manifest trước runtime registry.

Outputs chính:

- `registry_lock`, `protocol_hash`;
- `source_config_manifest`;
- `capability_seed`, `decision_traceability`;
- artifact/schema/step/access catalogs;
- `known_cases_seal`;
- environment expectation/observation;
- P00 audit, job manifest và `_SUCCESS` receipt.

Hàng rào:

- run root đã tồn tại thì không ghi đè;
- config hoặc generated docs drift làm P00/check thất bại;
- snapshot đã cấp phải được đưa vào registry và protocol hash;
- P00 không đọc outcome empirical để điều chỉnh tiêu chí.

## 6. P01 — Audit từng nguồn thật

Script: `scripts/p01_audit_raw.py`.

```powershell
uv run python scripts/p01_audit_raw.py `
  --registry artifacts/runs/<run-id>/P00/registry.lock.json `
  --run-id <run-id> `
  --source-id <source-id>
```

Mục tiêu:

- xác minh file vẫn khớp snapshot SHA-256;
- kiểm tra schema/header/date/key/unit/coverage;
- kiểm tra semantic fields đã resolve;
- tạo quyết định `pipeline_may_advance` theo source.

Reads: file nguồn qua snapshot-locked source spec. P01 không đọc artifact upstream.

Writes: `P01/raw_audit/<source_id>.json`.

Hàng rào:

- một source không được thay file sau snapshot;
- required semantic field thiếu phải fail;
- audit không sửa raw file;
- known-case content không được audit/mở như source bình thường trước P15.

## 7. P02 — Firm master và as-of panel

Script: `scripts/p02_build_firm_panel.py`.

Mục tiêu:

- dựng firm master từ source được phép đóng góp identity;
- chuẩn hóa issuer/ticker và áp dụng entity-resolution registry;
- kiểm tra alias/ticker reuse/overlapping spell;
- tạo panel firm-year theo prediction-time rule;
- loại duplicate/exclusion có reason rõ ràng.

Reads: toàn bộ `raw_audit` cần thiết và source snapshot-locked qua P02 reader.

Writes:

- `firm_master.parquet`;
- `firm_year_panel.parquet`;
- `duplicate_map.json`.

Hàng rào:

- firm-year key phải duy nhất;
- prediction time phải hợp lệ theo lịch báo cáo;
- không đoán alias không có bằng chứng;
- không dùng direct pandas artifact I/O ngoài core store.

## 8. P03 — Evidence ledger và lag decomposition

Script: `scripts/p03_evidence_ledger.py`.

Mục tiêu:

- giữ nguyên S1/S2 tại common annual anchor;
- dùng `document_id` làm khóa quyết định S3 và giữ `decision_number` làm provenance;
- gắn quyết định S3 năm `y` cho firm-year `y-1`, độc lập với vị trí ngày so với 31/3;
- giữ decision-level ledger riêng và tạo source-result theo firm-year cho bốn endpoint
  `S3_BROAD`, `S3_REPORTING`, `S3_CONTENT`, `S3_TIMELINESS` trong cùng channel S3;
- chỉ dùng normalized taxonomy registry, không phân loại bằng keyword văn bản tự do;
- giữ delayed-event lag decomposition chỉ cho endpoint sensitivity được đăng ký riêng.

Reads: `firm_year_panel`, `raw_audit` và source spec đã khóa.

Writes:

- `evidence_ledger.parquet`;
- `sanction_decision_ledger.parquet`;
- `availability_registry.json`;
- `lag_decomposition.json`.

Hàng rào:

- S3 source year complete và không có quyết định endpoint mới tạo explicit `False`;
- S3 source year incomplete, ngoài universe hoặc taxonomy không đủ vẫn là unknown;
- cùng `document_id` cho nhiều firm giữ nhiều firm mappings nhưng decision count vẫn theo
  unique `document_id`;
- event không link được ghi `UNLINKED_FIRM_YEAR`, không ép vào endpoint;
- lag identity phải đúng trong tolerance từ `evidence.yaml`;
- delisting/merger không tự động là negative.

## 9. P04 — Risk set, maturity, prospective và censoring

Script: `scripts/p04_risk_sets.py`.

Mục tiêu:

- phân loại riêng `annual_measurement_mature` cho S1/S2;
- phân loại `s3_next_year_mature` theo completeness của sanction year `t+1`;
- giữ `required_sanction_year` và `source_year_complete` trong risk-set contract;
- chỉ tạo cột `mature` chung từ `primary_target_id` đã khóa;
- tạo retrospective risk set;
- ghi censoring classification mà không tạo false negative.

Reads: `firm_year_panel`, `evidence_ledger`.

Writes:

- `risk_sets.parquet`;
- `maturity_audit.json`;
- `prospective_set.parquet`;
- `censoring_registry.json`.

Điều kiện trước run thật: `risksets.data_cutoff` phải là ngày được người dùng
khóa. Repo để `null` thay vì tự đoán ngày.

Hàng rào:

- khi `primary_target_id` còn null, production path fail-closed với
  `PRIMARY_TARGET_NOT_LOCKED`;
- incomplete source year không phải negative;
- prospective rows không vào retrospective evaluation;
- exit/code change không tự gán outcome;
- P04 chỉ phân loại maturity, không dùng positive count để làm đẹp fold role.

## 10. P05 — Measurement inputs và sealed outcomes

Script: `scripts/p05_measurement_inputs.py`.

Mục tiêu:

- tạo source binary matrix và channel matrix;
- xây L0 theo source và các candidate target config-driven;
- giữ `L1_ANNUAL` độc lập S3, đồng thời đăng ký các S3 endpoint,
  `L1_REPORTING` và `L1_CONTENT_STRICT`;
- tính L2 theo `g_c(S,T,Q)` khi công thức, quality theo profile và delay half-life
  đã được khóa; nếu chưa khóa thì giữ `EMPIRICALLY_PENDING`, không dùng proxy;
- chạy pilot fixed-π L3 bằng MCMC với Se/Sp theo source, random effect theo channel,
  R-hat, ESS và posterior-predictive diagnostics khi grid/prior đã được khóa;
- ghi posterior mean theo từng fixed-π trở lại đúng row của source/channel matrix;
  posterior tổng hợp này được đánh dấu `p05_feasibility_pilot_only`, không được
  dùng thay cho fold-local target của P10;
- đánh giá channel/anchor/L3 capability;
- chỉ công bố positive count aggregate theo fold;
- niêm phong row-level outcomes theo khóa `firm_id × fiscal_year × target_id`.

Reads: `risk_sets`, `evidence_ledger`.

Writes:

- `source_channel_matrices.json`;
- `l0_l1_inputs.parquet`;
- `measurement_variable_registry.json`;
- `channel_capability.json`, `anchor_capability.json`;
- `l3_pilot_capability.json`;
- `fold_eligibility.json`;
- `sealed_outcomes.parquet`.

Hàng rào:

- mỗi candidate target: có ít nhất một required source positive thì `True`; mọi required
  source có opportunity và false mới `False`; còn required source unknown thì target unknown;
  còn missing/false hỗn hợp là unknown;
- L2 chuẩn hóa trên observed channels, lưu `observed_channel_count`, quality/delay
  components và không dùng tổng số channel cố định;
- content predictor không được vào label model;
- hierarchical-π không được trở thành Gate 1 candidate;
- `fold_eligibility` tính riêng theo `target_id` và chỉ chứa aggregate count;
- P10/P11/P12 không tự chọn target có kết quả tốt nhất khi `primary_target_id` còn null.

## 11. P06 — Observability và verification registry

Script: `scripts/p06_observability.py`.

Mục tiêu:

- phân loại channel thành observed verification, observed opportunity only hoặc unknown;
- tính coverage diagnostics mô tả;
- ghi rõ full-sample diagnostics không được dùng phân tích.

Reads: `l3_pilot_capability`, `source_channel_matrices`.

Writes: `observability_registry.json`.

Hàng rào:

- `fit_scope = descriptive_full_sample`;
- `analytical_use = prohibited`;
- P06 không tạo IPW/IPCW dùng cho P10–P16.

## 12. P07 — Feature panel và leakage registry

Script: `scripts/p07_features.py`.

Mục tiêu:

- đọc feature definitions được đăng ký;
- bind `feature_id` với physical column tồn tại trong panel;
- lọc eligible firm-years;
- kiểm tra role, theoretical block và availability rule;
- tạo leakage audit, chưa fit preprocessing.

Reads: `observability_registry`, `firm_year_panel`, `risk_sets`, `raw_audit`.

Writes:

- `feature_panel.parquet`;
- `feature_registry.json`;
- `leakage_registry.json`.

Điều kiện trước run thật: điền `features.registry`. Nếu registry rỗng, stage tạo
artifact hợp lệ nhưng báo `SKIPPED/FEATURE_REGISTRY_EMPTY`; P11 không thể PASS.

Hàng rào:

- feature content bắt buộc `allowed_in_label_model: false`;
- physical column chỉ xuất hiện trong binding config;
- preprocessing fit tại inner-development fold ở P11, không fit ở P07;
- không dùng future/outer availability.

## 13. P08 — Mô phỏng phương pháp và adaptive MCSE

Scripts:

- `p08_build_scenario_registry.py`;
- `p08_run_batch.py`;
- `p08_aggregate_batches.py`.

Mục tiêu:

- chạy scenario được khai báo rõ, không tự tạo Cartesian product;
- sinh latent state, content signal, verification, source errors, dependence và delay;
- gọi lại production L1/L2/L3 fixed-π từ `src/labels/`;
- chạy batch deterministic theo protocol hash và coordinates;
- tăng replication đến khi đạt MCSE hoặc maximum cap;
- báo MCSE thực tế nếu cap đã chạm.

Reads: `feature_registry`, `source_channel_matrices`, scenario/batch artifacts.

Writes:

- `simulation_scenario_registry.json`;
- `batches/<scenario>/<method>/<batch>.parquet`;
- `mcse_report.json`.

Coordinates: `scenario_id`, `method_id`, `batch_id`.

Hàng rào:

- không đọc outer outcomes hoặc K1–K4;
- fixed-π không báo bias/RMSE của π;
- hierarchical-π metrics chỉ dành cho tham số thực sự estimated;
- số replication và MCSE threshold chỉ lấy từ registry;
- scenario list rỗng tạo `SKIPPED`, không tự bịa scenario.

Giới hạn implementation hiện còn mở: P08 đã có DGP, adaptive MCSE và một số
operating-characteristic metrics, nhưng chưa tái chạy toàn bộ production procedure
cho mọi nhánh D38–D45 và chưa có semi-synthetic development-covariate tier. Vì vậy
không được dùng việc script tồn tại hoặc unit test pass để gọi P08 là method-complete.

## 14. P09 — Temporal splits và fold-aware weights

Script: `scripts/p09_splits_weights.py`.

Mục tiêu:

- tạo rolling-origin temporal splits;
- tạo strict channel-within-time units;
- lấy fold role aggregate từ P05;
- fit verification weighting chỉ bằng development history;
- kiểm tra ESS và propensity support;
- fallback về unweighted mature cohort nếu IPW diagnostics không đạt.

Reads: feature/risk panels, observability, source matrices và fold eligibility.

Writes:

- `temporal_splits.json`;
- `channel_splits.json`;
- `weights/<fold_id>.parquet`;
- `weight_diagnostics/<fold_id>.json`.

Hàng rào:

- outer rows used in weight fit luôn bằng 0;
- diagnostics ghi fit scope, development years, ESS, support, estimand và reason;
- P10/P11 từ chối weight không có development-only diagnostics;
- IPCW giữ vai trò sensitivity.

## 15. P10 — Gate 1 measurement selection

Script: `scripts/p10_select_measurement.py`.

Mục tiêu:

- đánh giá L2 và L3 fixed-π chỉ khi target/capability substantive cho phép;
- sử dụng development years trước outer fold;
- thực hiện channel-within-time holdout;
- với L3, refit MCMC chỉ trên development history của outer fold, loại toàn bộ
  source thuộc held-out channel, tính soft cross-entropy theo channel giữ lại để
  chọn fixed-π, sau đó refit full-source trên development history và ghi row-level
  target vào `channel_measurement_selection`;
- ghi lựa chọn theo fold và strict channel;
- dùng giá trị `none` khi không ứng viên nào hợp lệ.

Reads: weights/diagnostics, split registries, source matrices, measurement inputs,
capabilities, fold eligibility và MCSE report.

Writes:

- `candidates/<outer_fold>.json`;
- `selection/<outer_fold>.json`;
- `channel_selection/<outer_fold>.json`.

Hàng rào:

- không được đọc `sealed_outcome_store`;
- held-out channel bị loại khỏi target, label model, feature, tuning và calibration;
- AP không tính trực tiếp trên L2/L3 soft targets;
- hierarchical-π không thể được chọn.

Giới hạn implementation hiện còn mở: strict channel score chưa chạy lại toàn bộ
learner/feature/tuning/calibration procedure cho từng `M*_{f,c}` và chưa truyền
posterior-draw robustness qua toàn pipeline. Row-level fold-local L3 posterior đã
được tạo tại P10; P10 vẫn phải giữ `none` nếu diagnostics/evidence không đạt.

## 16. P11 — Fit models và freeze

Script: `scripts/p11_freeze_models.py`.

Mục tiêu:

- fit learner đã đăng ký trong từng outer fold;
- chạy search space đã khóa với tối đa 50 valid configurations, ghi runtime và
  valid/evaluated counts; thiếu search space của learner confirmatory thì dừng;
- luôn fit Track A trên L1; fit Track B thật trên L2/L3 được P10 chọn;
- fit bagging Anchor-PU chỉ từ positive của source neo đã đăng ký;
- tạo observability-only, content-only và full comparisons;
- fit imputation/scaling/model trong temporal development folds;
- tạo development OOF predictions;
- tạo raw outer scores mà không đọc outer outcomes;
- serialize model và khóa hashes vào freeze receipt.

Reads: selection/split/feature/label/weight artifacts và P00 provenance.

Writes:

- `models/<outer_fold>.json`;
- `oof/<outer_fold>.parquet`;
- `predictions/<outer_fold>.parquet`;
- `freeze/<outer_fold>.json`.

Freeze receipt khóa protocol, split, measurement selection, channel measurement
selection (bao gồm L3 target provenance), feature registry,
preprocessing, learner settings, weight diagnostics, model, predictions, Git và
environment hashes.

Hàng rào:

- P11 không đọc sealed outer outcomes;
- outer prediction không được dùng để tune;
- preprocessing chỉ fit trong development/inner training;
- feature registry rỗng hoặc insufficient classes tạo SKIPPED receipt, không PASS giả.

## 17. P12 — Outer open, calibration và evaluation

Script: `scripts/p12_evaluate.py`.

Mục tiêu:

- xác minh PASS freeze receipt và protocol hash;
- tạo immutable outer-open receipt;
- fit Platt calibrator từ pooled cross-fitted development predictions;
- áp dụng calibrator lên raw outer predictions;
- tính registered discrimination/calibration/top-budget metrics;
- chạy paired firm bootstrap;
- đánh giá utility scenarios hoặc ghi explicit skip.
- scenario utility không rỗng phải bind `measurement_fixed_pi`, áp dụng L3 đã fit
  và đóng băng từ development history để tính `r_i(theta)` sau outer-open; không
  thay `r_i(theta)` bằng outer outcome hoặc calibrated model risk;
- từ latent risk đó phải tính reviewed cases, expected TP/FP/FN, review cost,
  additional false-positive cost, false-negative cost, true-positive benefit,
  net utility, incremental utility so với model observability-only tương ứng và
  uncertainty kết hợp L3 posterior-parameter draws và firm bootstrap; không được
  chỉ ghi `REGISTERED`;

Reads: freeze receipt, OOF predictions, raw outer predictions và sealed outcomes.

Writes:

- `outer_open/<outer_fold>.json`;
- `calibration/<outer_fold>.json`;
- `metrics/<outer_fold>.json`;
- `bootstrap/<outer_fold>.json`;
- `utility/<outer_fold>.json`.

Hàng rào:

- không mở outer outcomes nếu freeze receipt thiếu/sai hash;
- calibrator không dùng in-sample refit predictions;
- bootstrap unit là firm và mỗi outer year được xử lý riêng;
- review cost và false-positive cost là thành phần tách biệt;
- P10/P11 không được ghi lại sau outer open.

## 18. P13 — Sensitivity và transfer

Script: `scripts/p13_sensitivity.py`.

Mục tiêu:

- tổng hợp source-set, channel holdout, dedup và lag sensitivity capability;
- đánh giá domain transfer trên các domain feature đã đăng ký;
- kiểm tra IPCW sensitivity chỉ ở fold có diagnostics đạt;
- ghi hierarchical-π sensitivity status;
- tạo block-ablation summary từ outer evaluations.

Reads: P03/P04/P05/P07/P09/P12 artifacts cần thiết.

Writes:

- `source_sensitivity.json`;
- `domain_transfer.json`;
- `censoring_sensitivity.json`;
- `hierarchical_pi.json`;
- `ablations.json`.

Hàng rào:

- sensitivity không đổi P10 selection hoặc P11 model freeze;
- hierarchical-π không quay lại Gate 1;
- domain result thiếu binding/support phải `SKIPPED`;
- IPCW không có development diagnostics phải `SKIPPED`.

## 19. P14 — Gate 2

Script: `scripts/p14_gate2.py`.

Mục tiêu:

- gom fully nested outer folds;
- so full model với observability-only reference;
- dùng paired bootstrap samples để tạo family-adjusted lower bound;
- kiểm tra MMI, direction consistency, yield decline, positive count và domain robustness;
- gán `PASS`, `FAIL` hoặc `INSUFFICIENT_EVIDENCE`.

Reads: transfer/sensitivity/ablation/evaluation/bootstrap artifacts.

Writes: `P14/gate2.json`.

Hàng rào:

- tiêu chí chỉ lấy từ locked registry;
- initial fold không tự nhập confirmatory pool;
- thiếu comparison/domain evidence không được gọi PASS;
- criteria không được tune theo outer results.

## 20. P15 — Mở known cases K1–K4

Script: `scripts/p15_open_known_cases.py`.

Mục tiêu:

- chỉ mở source có `role: known_case` sau Gate 2;
- kiểm tra file hash vẫn khớp snapshot P00;
- xác minh freeze, outer-open, evaluation và protocol hashes của mọi fold;
- tính within-year/model percentile và exact permutation lower-tail rank;
- áp dụng soft-veto rule đã khóa.

Reads: Gate 2, freeze/open receipts, metrics và raw predictions. Known-case source
được mở trực tiếp qua snapshot-locked reader chỉ tại stage này.

Writes: `P15/known_cases.json`.

Hàng rào:

- known cases không được nâng cấp Gate 2 thất bại;
- chúng chỉ có thể downgrade/soft-veto theo rule;
- ít hơn minimum case chỉ được casewise reporting;
- file known cases không có là explicit `SKIPPED`, không phải validation PASS.

## 21. P16 — Threshold/interaction và Gate 3

Script: `scripts/p16_gate3.py`.

Mục tiêu:

- yêu cầu parent Gate 2 PASS;
- resolve pressure, monitoring, domain và parent-model bindings;
- kiểm tra breakpoint stability qua folds;
- breakpoint stability dùng dispersion `std(breakpoints) <= tolerance`; vị trí
  breakpoint so với zero chỉ thuộc một giả thuyết vị trí riêng nếu protocol khóa;
- kiểm tra cross-domain difference và common support;
- kiểm tra pressure × monitoring direction;
- áp dụng known-case soft veto;
- gán Gate 3 verdict.

Reads: known-case results, Gate 2, evaluations, feature panel/registry, raw
predictions và sealed outcomes.

Writes:

- `threshold_interactions.json`;
- `gate3.json`.

Điều kiện trước run thật: `inference.interaction_library.operational_bindings`
phải trỏ tới feature/model IDs đã đăng ký.

Hàng rào:

- parent Gate 2 không PASS thì Gate 3 ineligible;
- binding thiếu tạo `INSUFFICIENT_EVIDENCE`;
- insufficient positives/support không được PASS;
- chỉ các claims trong locked interaction library được xem là confirmatory.

## 22. P17 — Final reporting

Script: `scripts/p17_build_outputs.py`.

Mục tiêu:

- đọc artifact hoàn tất;
- dựng final result ledger và verdict matrix;
- xuất decision log và Chapter 4 machine-readable tables;
- tạo SVG gate summary;
- inventory và xác minh artifact manifests;
- tạo final report và audit report.

Reads bắt buộc: `gate3_verdict`. Các artifact Gate 2/evaluation/sensitivity/
known-case/calibration/bootstrap/utility là optional reads có kiểm soát.

Writes:

- `final_ledger.json`;
- `final_verdict_matrix.json`;
- `final_artifact_manifest.json`;
- `final_decision_log.json`;
- `chapter4_input_tables.json`;
- `final_gate_figure.svg`;
- `FINAL_AUDIT_REPORT.md`;
- `final_report.md`.

Hàng rào:

- không import modeling, labels, selection hoặc simulation;
- không fit model hoặc calibrator;
- không tạo lại label;
- không thay selection/gate;
- không mở file raw/known-case mới;
- final artifact manifest loại chính nó để tránh recursive self-hash.

## 23. Các giá trị phải khóa trước run empirical

Pipeline cố ý fail-closed khi các giá trị sau chưa được người dùng xác nhận:

| Setting | File owner | Hệ quả khi chưa có |
| --- | --- | --- |
| `risksets.data_cutoff` | `config/methodology/risksets.yaml` | P04 dừng |
| `features.registry` | `config/methodology/features.yaml` | P07 SKIPPED; P11 không freeze PASS |
| `measurement.l2_missingness.minimum_observed_channels` | `config/methodology/measurement.yaml` | L2 không eligible |
| `measurement.l2_scoring.formula`, quality profile và delay half-life | `config/methodology/measurement.yaml` | L2 giữ `EMPIRICALLY_PENDING` |
| fixed-π grid và accuracy priors theo profile | `config/methodology/measurement.yaml` | L3 pilot giữ `EMPIRICALLY_PENDING` |
| `learners.tuning.search_spaces` | `config/execution/learners.yaml` | P11 dừng trước freeze |
| `simulation.operational_scenarios` | `config/execution/simulation.yaml` | P08 SKIPPED |
| `utility.operational_scenarios` với `measurement_fixed_pi`, benefit/cost và budget đã khóa | `config/methodology/utility.yaml` | utility SKIPPED, descriptive yield vẫn ghi |
| Hai threshold feature, pressure/monitoring/domain/model bindings | `config/methodology/inference.yaml` | Gate 3 insufficient evidence |
| `known_cases.csv` | `data/source/known_cases/` | P15 SKIPPED; không soft veto |

Không điền các setting này bằng suy đoán từ outer results hoặc K1–K4.

## 24. Chạy lại và phục hồi

- Artifact là immutable; exact retry chỉ idempotent khi content và dependency
  provenance giống hệt nhau.
- P01/P08/P09/P10/P11/P12 có đơn vị partition nên có thể rerun trong một run mới
  với coordinates tương ứng.
- Nếu một run dừng giữa chừng, chạy lại đúng lệnh với `--resume`. Recovery policy
  chỉ quarantine artifact/manifest pair không hoàn chỉnh; unit đủ contract và hash
  được skip, unit chưa đủ được chạy lại.
- Resume bị từ chối nếu raw source, code, module config hoặc snapshot đã drift.
- Đổi source, schema, config hoặc code được chấp nhận phải tạo run-id/protocol mới.
- P17 lỗi trình bày có thể rerun từ finalized upstream artifacts trong run mới;
  không refit P11.

## 25. Kiểm tra implementation

Focused invariant tests nằm trong:

- `tests/stages/test_remaining_stage_invariants.py`;
- `tests/stages/test_p17_report_only.py`;
- `tests/access/test_firewalls.py`;
- `tests/runtime/test_artifact_store.py`.

Quality gate chuẩn:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
uv run pre-commit run --all-files
```
