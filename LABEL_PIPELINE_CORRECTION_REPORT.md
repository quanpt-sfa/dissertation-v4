# Label Pipeline Correction Report

## Kết luận

Pipeline nhãn P03–P09 đã được sửa và chạy lại bằng run bất biến mới:

`dissertation-2015-2026-label-correction-v2`

Protocol hash: `2f254055decdca8a8d185e61e82586b971c939787c7992fd22da8190d45f7df8`.

Run đạt P09. P03 giữ 385 event-level records; P05 tạo 60 mature L1 positives, 0 explicit negatives và 21.643 mature unknowns. P10 không được chạy vì P07 và P08 đều `SKIPPED`, feature registry rỗng và không fold nào có hai observed classes.

## Những sửa đổi đã triển khai và kiểm thử

### P03 — evidence ledger

- Bỏ aggregate source-year trước horizon; ledger hiện là event-level và giữ `event_id`, `event_cluster_id`, firm, fiscal year, source, channel, availability date, outcome, outcome basis, period-link source và period-link confidence.
- `train_include_flag` chỉ còn là row inclusion. Hard positive được khóa riêng bằng `is_direct_label`; false hoặc missing của positive indicator được giữ là unknown, không phải explicit negative.
- Event ID fallback được hash từ nội dung canonical của record, không phụ thuộc row number.
- Duplicate cluster được xử lý bằng rule khóa `identical_signature_then_source_event_id`; representative được chọn deterministic. Cluster có timing/outcome semantics mâu thuẫn fail-closed.
- Availability registry giữ accepted, unlinked, duplicate, excluded và provenance của outcome rule.

### P04 — maturity và source timing

- Maturity và horizon đều dùng calendar-month arithmetic qua `DateOffset`.
- Source timing tách pre-prediction, same-day, in-horizon, post-horizon và post-cutoff.
- Lag âm không được tính vào in-horizon fraction.
- Metric được đặt đúng scope `observed_linked_events_only`; không gọi incidence là coverage.

### P05 — L0/L1 và fold eligibility

- Mỗi event được lọc bằng `(prediction_time, horizon_end]` trước khi aggregate source rồi channel.
- Aggregate nhiều event cùng source-year độc lập row order: có positive thì positive; chỉ khi mọi observed result đều false mới là explicit negative; còn lại unknown.
- Matrix giữ riêng `eligible` và `mature`; absence và immaturity không thành negative.
- Fold artifact báo eligible rows, mature rows, observed-label rows, positive, explicit negative, unknown và số observed classes.
- Positive-only fold không thể trở thành confirmatory dù vượt positive threshold.
- L3 structural status `INSUFFICIENT_CHANNELS/UNAVAILABLE_BY_DESIGN` được giữ ưu tiên trước missing empirical parameters.

### P06 — observability

- Báo riêng source và channel.
- Báo event incidence, observed-outcome fraction, mature-cohort observed fraction và prospective observations.
- Source opportunity coverage là `UNKNOWN_NO_OPPORTUNITY_INDICATOR`; không suy coverage từ absence of sanction.
- Verification classification của S3 là `observed_verification`.

### P07–P12 — fail-closed production control

- P10 tổng hợp blocker từ fold role, class counts, feature registry và P08 MCSE trước selection.
- P11 kiểm tra lại fold eligibility và feature registry; không còn ghi freeze receipt `SKIPPED` rồi trả exit code thành công.
- P12 vốn đã yêu cầu P08 MCSE `PASS` và model-freeze receipt `PASS` trước khi mở outer outcome.
- Reason codes mới được đăng ký trong vocabulary thay vì phát sinh tự do ở downstream artifact.

## Regression và mutation coverage

Đã có tests cho:

1. pre-prediction event không che in-horizon positive;
2. post-horizon positive không bị backdate;
3. duplicate representative độc lập row/source order;
4. duplicate cluster có timing mâu thuẫn fail-closed;
5. event đúng horizon end được nhận;
6. event sau horizon một ngày bị loại;
7. lag âm không vào in-horizon fraction;
8. event sau cutoff không vào horizon count;
9. nhiều event source-year độc lập row order;
10. mature count lấy từ risk set;
11. positive-only/missing-negative fold không confirmatory;
12. empty features và P08 SKIPPED chặn P10;
13. descriptive, prospective và initial-separate roles bị P11 confirmatory từ chối;
14. absence of sanction vẫn unknown;
15. false positive-indicator vẫn unknown;
16. L3 structural unavailability không bị ghi đè;
17. immutable artifact contract và old-run tree digest.

## Run kết quả

| Stage | Kết quả | Chi tiết |
|---|---:|---|
| Snapshot | PASS | 7 sources; snapshot hash `84828b0c7c3bcf763f511ae0e4539dbc6c74c6cab32daae554fd2a7cbfdb13a3` |
| P00 | PASS | protocol hash `2f254055...f7df8` |
| P01 | PASS | 7/7 sources |
| P02 | PASS | 3.040 firms; 23.411 firm-years; excluded 0 |
| P03 | PASS | 385 linked accepted event records |
| P04 | PASS | 21.703 mature; 1.708 prospective |
| P05 | PASS | 60 sealed positive firm-years; 0 explicit negative |
| P06 | PASS | 1 valid evidence channel: S3 |
| P07 | SKIPPED | `FEATURE_REGISTRY_EMPTY` |
| P08 | SKIPPED | `NO_OPERATIONAL_SCENARIOS/NO_SIMULATION_BATCHES` |
| P09 | PASS | 5 split/weight units |
| P10+ | NOT RUN | fail-closed blockers đã được chứng minh trước khi selection |

Run v1 `dissertation-2015-2026-label-correction-v1` được giữ nguyên nhưng superseded: taxonomy audit phát hiện policy `included_event_positive` chưa tách outcome đủ rõ khỏi inclusion. Run v2 khóa `is_direct_label` làm explicit hard-positive indicator.

## Operational blockers còn lại

- `features.registry` rỗng: P07 chỉ có thể tạo artifact `SKIPPED`.
- `simulation.operational_scenarios` rỗng: P08 không thể MCSE PASS.
- Không có explicit negative cho S3 endpoint và absence không được coi là negative; mọi outer fold chỉ có một observed class.
- L2 thiếu formula, source quality, delay half-life và minimum observed channels.
- L3 chỉ có một channel và đồng thời thiếu fixed-pi grid/accuracy priors; structural blocker được ưu tiên.
- S1/S2 chưa đủ availability/opportunity/outcome mapping để trở thành evidence sources.
- Known-case IDs K1–K4 có config seal nhưng không có source binding/case data.

## Quality gates

Gate cuối sau khi sinh và kiểm tra deliverables:

- Bootstrap `--write`: PASS
- Bootstrap `--check`: PASS
- Ruff check: PASS
- Ruff format check: PASS
- Pyright: 0 errors, 0 warnings, 0 informations
- Pytest: 147 passed; 12 upstream sklearn deprecation warnings
- Pre-commit: tất cả hooks PASS
- `git diff --check`: PASS

## Exact command log

```powershell
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --write
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run pre-commit run --all-files
git diff --check

git add config docs scripts src tests EMPIRICAL_OUTCOME_CAPABILITY_AUDIT.md
git commit -m "Correct event-level label pipeline contracts"

uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-label-correction-v1 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P09

git add config docs/generated scripts src tests
git commit -m "Bind sanctions to explicit hard-positive indicator"

uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-label-correction-v2 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P09

uv run python scripts/audit_label_pipeline.py `
  --old-run-root "D:\Works\dissertation\dissertation-v4\artifacts\runs\dissertation-2015-2026-cutoff-20260331" `
  --new-run-root "D:\Works\dissertation\dissertation-v4\artifacts\runs\dissertation-2015-2026-label-correction-v2" `
  --output-dir "D:\Works\dissertation\dissertation-v4\docs\audits"
```

## Files changed

Implementation/config:

- `config/foundation/columns.yaml`
- `config/foundation/steps.yaml`
- `config/foundation/vocabulary.yaml`
- `config/methodology/source_catalog.yaml`
- `config/schemas/core.yaml`
- `scripts/p03_evidence_ledger.py`
- `scripts/p05_measurement_inputs.py`
- `scripts/p10_select_measurement.py`
- `scripts/p11_freeze_models.py`
- `scripts/audit_label_pipeline.py`
- `src/core/fold_control.py`
- `src/core/semantic_keys.py`
- `src/evidence/service.py`
- `src/measurement/service.py`
- `src/observability/service.py`
- `src/risksets/service.py`
- `src/snapshot/builder.py`
- `src/snapshot/models.py`

Tests/generated docs/audits:

- `tests/runtime/test_artifact_store.py`
- `tests/stages/test_remaining_stage_invariants.py`
- `docs/generated/ACCESS_MATRIX.md`
- `docs/generated/SCHEMA_CATALOG.md`
- `docs/generated/STEP_CARDS.md`
- `EMPIRICAL_OUTCOME_CAPABILITY_AUDIT.md`
- toàn bộ deliverables `LABEL_*.md` và các bảng mới trong `docs/audits/`.
