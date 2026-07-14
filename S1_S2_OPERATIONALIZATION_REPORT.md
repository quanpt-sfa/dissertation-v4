# S1/S2 operationalization report

## Outcome

S1 before/after audit adjustments và S2 audit opinions đã được operationalize theo annual firm-year semantics. S1/S2 dùng common annual anchor; S3 giữ delayed-verification semantics. Pipeline mới chạy thành công và được runner xác minh đến P09.

Không có missing/absence nào bị chuyển thành negative. S1 giữ fail-closed do materiality inputs chưa được khóa. S2 clean opinions tạo source-specific explicit negatives, nhưng L1 vẫn bảo toàn unknown của S3.

## Implementation status

| Workstream | Status | Evidence |
|---|---|---|
| S1 deterministic pair builder | Implemented and tested | Pair key, duplicate/conflict, unit/scope/family, denominator and provenance checks |
| S1 empirical classification | Implemented, awaiting empirical input | Threshold/floor null làm outcome unknown |
| S2 normalization | Implemented and tested | Clean/qualified/disclaimer/adverse taxonomy và raw provenance |
| Annual/delayed temporal split | Implemented and tested | 70.233 annual records, 0 anchor mismatch; S3 only in lag curve |
| L0/L1 three-state aggregation | Implemented and tested | 2.259 L1 positive, 0 negative, 21.152 unknown |
| P06 coverage/overlap | Implemented and tested | Opportunity, outcomes, incidence và overlap tách riêng |
| Leakage firewall | Implemented and tested | Same-year label components bị reject |
| L2 | Implemented but awaiting empirical inputs | Formula, quality, delay và coverage floor chưa khóa |
| L3 | Structurally feasible, awaiting empirical inputs | S2/S3 valid; 52 overlap rows; fixed π/priors chưa khóa |
| P08 simulation | Intentionally unavailable under current config | `SKIPPED/NO_SIMULATION_BATCHES`; config không bị sửa |
| Confirmatory Track A | Unavailable with current labels | Mọi outer fold thiếu explicit-negative class |

## Files changed in implementation commit

Implementation commit: `7dce0ac` (`Operationalize annual S1 and S2 evidence`).

Các nhóm chính:

- config/contracts: `config/foundation/{artifacts,columns,steps}.yaml`, `config/methodology/{source_catalog,features}.yaml`, `config/schemas/core.yaml`;
- registry/snapshot: `src/core/evidence_registry.py`, `src/core/semantic_keys.py`, `src/snapshot/{models,builder}.py`;
- P03–P07: `scripts/p03_evidence_ledger.py`, `scripts/p05_measurement_inputs.py`, `scripts/p06_observability.py`, `scripts/p07_features.py`;
- substantive services: `src/evidence/{annual,service}.py`, `src/labels/service.py`, `src/measurement/service.py`, `src/risksets/service.py`, `src/observability/service.py`, `src/features/service.py`;
- tests/contracts: `tests/stages/test_annual_evidence_channels.py` và các regression updates;
- generated catalogs: `docs/generated/`.

Post-run reporting thêm `scripts/report_s1_s2_audit.py`, sáu CSV audit và năm báo cáo markdown; script chỉ đọc artifact qua verified RunContext/ArtifactStore và không sửa run.

## Regression coverage

Các tests mới bảo vệ:

1. S1/S2 bằng annual anchor được nhận;
2. S3 cùng ngày anchor không được coi là delayed evidence;
3. S1 pair trên/dưới threshold fixture cho True/False;
4. missing pair và zero denominator cho Unknown;
5. duplicate conflict deterministic và fail-closed;
6. clean opinion False, non-clean True, missing/conflict Unknown;
7. L1 `False + False + Unknown = Unknown`;
8. bất kỳ source True làm L1 True;
9. chỉ khi mọi opportunity observed và mọi source false mới có L1 False;
10. S1/S2 không đi vào delayed maturity curve;
11. same-year label-derived feature bị chặn;
12. channel-overlap counts chính xác.

Full suite trước run: 160 passed, 12 upstream sklearn deprecation warnings.

## Final quality gates

| Gate | Result |
|---|---|
| Bootstrap `--write` | PASS |
| Bootstrap `--check` | PASS |
| Ruff check | PASS |
| Ruff format check | PASS, 113 files formatted |
| Pyright | PASS, 0 errors / 0 warnings / 0 informations |
| Pytest | PASS, 160 tests; 12 upstream sklearn deprecation warnings |
| Pre-commit all files | PASS, 6/6 hooks |
| `git diff --check` | PASS |
| CSV spreadsheet read-back | PASS, 6/6 tables |

Quality command log:

```powershell
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --write
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run pre-commit run --all-files
git diff --check
```

## Run result

- Run ID: `dissertation-2015-2026-s1-s2-annual-v1`
- Protocol hash: `0385ad907cc4a17cdc53b4723a2c29fd34cf540ae5935381d2fcf5f00fb2e913`
- P00–P09: artifact contracts verified complete
- P08: `SKIPPED`, reason `NO_SIMULATION_BATCHES`
- P09: PASS, five time folds materialized
- Feature registry: empty; production modeling remains fail-closed

Run command:

```powershell
uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-s1-s2-annual-v1 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P09
```

The desktop command window timed out after the P09 artifacts had been written. Không có code/config/data change; runner được gọi lại đúng run với `--resume --through P09` và xác minh mọi P01–P09 unit complete bằng raw hashes, protocol hash, manifests và content hashes. Không artifact nào được recompute.

```powershell
uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-s1-s2-annual-v1 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --resume `
  --through P09
```

Audit-table command:

```powershell
uv run python scripts/report_s1_s2_audit.py `
  --registry "artifacts/runs/dissertation-2015-2026-s1-s2-annual-v1/P00/registry.lock.json" `
  --run-id dissertation-2015-2026-s1-s2-annual-v1 `
  --output-dir docs/audits
```

## Old-run immutability

Run `dissertation-2015-2026-label-correction-v2` được hash trước implementation/run mới bằng sorted relative-path + per-file SHA-256 tree digest:

- file count: 102;
- total bytes: 19.955.784;
- tree SHA-256: `9bac72ccb96caf29400eec172e78885d6dd31ef6a6f63db58d3166b629099cd4`.

Digest sau toàn bộ implementation, run mới và reporting khớp tuyệt đối: 102 files, 19.955.784 bytes và cùng tree SHA-256 `9bac72ccb96caf29400eec172e78885d6dd31ef6a6f63db58d3166b629099cd4`. Run v2 không được resume, sửa hoặc ghi đè.

## Empirical blockers

S1:

- minimum absolute denominator;
- profit materiality threshold;
- revenue materiality threshold.

L2:

- supported scoring formula selection;
- source-quality values by profile;
- delay half-life;
- minimum observed-channel rule nếu Chapter 3 yêu cầu.

L3:

- fixed π grid;
- accuracy priors by source profile.

Downstream production:

- feature registry;
- operational simulation scenarios và MCSE PASS;
- learner tuning search spaces.

Mọi blocker được giữ nguyên; không có threshold, quality weight, Se/Sp, prevalence hoặc simulation scenario nào được tự điền.
