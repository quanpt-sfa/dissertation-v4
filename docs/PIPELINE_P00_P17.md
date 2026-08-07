# Đặc tả pipeline P00–P17

Tài liệu này mô tả implementation vận hành. Danh sách reads/writes chính xác được
sinh từ registry tại `docs/generated/`; nếu tài liệu thủ công và generated catalog
khác nhau, `config/pipeline.yaml` cùng generated catalog là nguồn kiểm tra cuối.

## Estimand và nguyên tắc xuyên suốt

Đơn vị phân tích là firm fiscal year `(i, t)`. S1 và S2 là annual measurements tại
common annual anchor. S3 dự báo endpoint trong calendar year `t+1` và dùng
`target_fiscal_year = sanction_year - 1`. Decision/publication dates được giữ cho
provenance và sensitivity, không thay đổi target fiscal year.

Các contract chung:

- mỗi run có một `run-id` và run root bất biến;
- snapshot khóa file, hash, schema, header và semantic binding;
- sau P00, stage chỉ đọc `registry.lock.json`;
- mọi artifact chính thức đi qua `RunContext` và `ArtifactStore`;
- artifact templates và source paths phải là relative, portable paths;
- raw root và output root chỉ được truyền qua CLI hoặc environment;
- thiếu nguồn không phải bằng chứng zero;
- `S=0` là observed source-endpoint zero, không phải latent non-fraud;
- outer outcomes được niêm phong đến P12;
- known cases được niêm phong đến P15;
- P17 chỉ báo cáo, không fit hoặc calibrate lại.

## Lệnh vận hành một lần

Chạy từ root repo:

```powershell
$projectRoot = (Resolve-Path ".").Path
$rawRoot = (Resolve-Path (Join-Path $projectRoot "..\final-input")).Path
$outputRoot = Join-Path $projectRoot "artifacts\runs"

uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-start-v1 `
  --raw-root $rawRoot `
  --output-root $outputRoot `
  --workers 5 `
  --through P17
```

`--through Pxx` dừng sau stage tương ứng. `--workers N` bật song song cho các
partition độc lập. Runner yêu cầu Git tree sạch; `--allow-dirty` chỉ dành cho phát
triển. `--resume` chỉ tiếp tục cùng immutable run sau khi code, config, raw hashes,
snapshot và artifact manifests đều được xác minh lại.

Luồng runner:

```text
snapshot → P00 → P01 [per source] → P02 → P03–P07 →
P08 [per batch] → P09 → P10–P12 [per fold] → P13–P17
```

## Trạng thái và đơn vị chạy lại

| Stage | Trạng thái vào | Trạng thái ra | Đơn vị partition/rerun |
| --- | --- | --- | --- |
| P00 | `CONFIGURED` | `LOCKED` | một run |
| P01 | `LOCKED` | `AUDITED` | một `source_id` |
| P02 | `AUDITED` | `PANELLED` | một run |
| P03 | `PANELLED` | `LEDGERED` | một run |
| P04 | `LEDGERED` | `RISK_SET` | một run |
| P05 | `RISK_SET` | `MEASURED` | một run |
| P06 | `MEASURED` | `OBSERVABLE` | một run |
| P07 | `OBSERVABLE` | `FEATURED` | một run |
| P08 | `FEATURED` | `SIMULATED` | scenario × method × batch |
| P09 | `SIMULATED` | `SPLIT` | fold registry |
| P10 | `SPLIT` | `SELECTED` | một outer fold |
| P11 | `SELECTED` | `FROZEN` | một outer fold |
| P12 | `FROZEN` | `EVALUATED` | một outer fold |
| P13 | `EVALUATED` | `SENSITIVITY` | một run |
| P14 | `SENSITIVITY` | `GATE2` | một run |
| P15 | `GATE2` | `KNOWN_CASES_OPEN` | một run |
| P16 | `KNOWN_CASES_OPEN` | `GATE3` | một run |
| P17 | `GATE3` | `REPORTED` | một run |

State là access-control label. Artifact và receipt mới là bằng chứng vật lý cho
phép stage downstream tiếp tục.

## P00 — Snapshot và khóa protocol

`scripts/create_data_snapshot.py` chỉ đọc glob đã đăng ký, kiểm tra cardinality,
schema/header, semantic binding, SHA-256 và row metadata. Snapshot không copy raw
data sang vị trí khác.

`scripts/p00_lock_protocol.py` compile config, kiểm tra path/schema/coordinate
collision, dựng access matrix, khóa source snapshot, known-case seal, environment
observation và protocol hash. Run root đã tồn tại không được ghi đè.

CLI trực tiếp dùng relative paths:

```powershell
uv run python scripts/p00_lock_protocol.py `
  --config config/pipeline.yaml `
  --run-id <run-id> `
  --output-root artifacts/runs `
  --snapshot-manifest artifacts/runs/<run-id>/SNAPSHOT/data_snapshot.json
```

## P01 — Audit raw sources

`scripts/p01_audit_raw.py` xác minh source hash, schema, semantic fields, dates,
keys, unit và coverage. Audit không sửa raw files và không mở known-case content.

```powershell
uv run python scripts/p01_audit_raw.py `
  --registry artifacts/runs/<run-id>/P00/registry.lock.json `
  --run-id <run-id> `
  --source-id <source-id>
```

Output:

```text
artifacts/runs/<run-id>/P01/raw_audit/<source-id>.json
```

## P02 — Firm master và as-of panel

`scripts/p02_build_firm_panel.py` chuẩn hóa entity, kiểm tra alias/ticker reuse,
tạo firm master và firm-year panel theo prediction-time rule. Firm-year keys phải
duy nhất; alias không có bằng chứng không được đoán.

Outputs chính:

```text
P02/firm_master.parquet
P02/firm_year_panel.parquet
P02/duplicate_map.json
```

## P03 — Evidence ledger

P03 giữ riêng observation, verification, determination, recording và observed
source label. S3 decision year `y` được gắn cho firm-year `y-1`; missing opportunity
không được chuyển thành negative.

Outputs chính:

```text
P03/evidence_ledger.parquet
P03/sanction_decision_ledger.parquet
P03/availability_registry.json
P03/lag_decomposition.json
```

## P04 — Risk sets và maturity

P04 phân loại annual maturity cho S1/S2 và next-calendar-year maturity cho S3.
Incomplete source year, prospective rows và exits không tự động trở thành negative.

Outputs chính:

```text
P04/risk_sets.parquet
P04/maturity_audit.json
P04/prospective_set.parquet
P04/censoring_registry.json
```

## P05 — Measurement inputs

P05 tạo source/channel matrices, candidate targets, capability receipts và sealed
outcomes. Label families không được dùng ngoài role đã khóa.

Outputs chính:

```text
P05/source_channel_matrices.json
P05/l0_l1_inputs.parquet
P05/measurement_variable_registry.json
P05/sealed_outcomes.parquet
```

## P06 — Observability

P06 dựng availability/observability registry và không mở outer outcomes. Missing
components phải giữ unknown với reason code.

## P07 — Features

P07 bind predictors đã được data-build materialize, kiểm tra coverage, missingness,
scale, lineage và leakage. P07 không tái tính feature từ nguồn long và không đưa
known-case identifiers vào model matrix.

Outputs chính:

```text
P07/feature_panel.parquet
P07/feature_registry.json
P07/leakage_registry.json
P07/feature_views.json
P07/p07_decision_report.md
```

## P08 — Simulation

P08 chạy scenario × method × batch, ghi batch artifacts và aggregate metrics.
Simulation không đọc empirical outer outcomes hoặc known cases.

## P09 — Splits và weights

P09 khóa fold assignment, overlap/IPW weights và diagnostic fold receipts. Fold
roles không được thay đổi theo kết quả downstream.

## P10 — Measurement/model selection

P10 chỉ dùng development information của outer fold tương ứng, chạy nested refits
và khóa lựa chọn. Confirmatory folds được xác định trước trong fold registry.

## P11 — Freeze

P11 materialize frozen model and preprocessing contracts cho từng outer fold. Sau
freeze, hyperparameters và feature set không được thay đổi.

## P12 — Outer evaluation

P12 mở outer outcomes theo access contract, đánh giá locked model/target pair và
ghi compatibility receipts cho resume. Không fit hoặc calibrate lại trên outer fold.

## P13 — Sensitivity

P13 chạy source exclusions, target variants và các sensitivity refits đã đăng ký.
Checkpoint/resume phải giữ deterministic process state và input hashes.

## P14 — Gate 2

P14 tổng hợp confirmatory evidence và trả verdict theo threshold đã khóa. Verdict
không được thay đổi để phù hợp known cases.

## P15 — Known-case external validation

P15 mới mở snapshot-locked `known_case_registry.csv`. Nó kiểm tra exact construct,
role, exclusion flags, external-validation inclusion, duplicate keys và hash seal.
Nếu registry hợp lệ nhưng không có rows, stage ghi `KNOWN_CASES_UNAVAILABLE`.

## P16 — Gate 3

P16 chỉ chạy khi parent gate đủ điều kiện. Nếu Gate 2 không PASS, Gate 3 ghi
`PARENT_GATE_FAILED` thay vì cố đánh giá threshold downstream.

## P17 — Reporting

P17 chỉ đọc locked artifacts, receipts và gate decisions để tạo outputs cuối. Không
được fit, select, calibrate hoặc tái xây nhãn.

## Portability contract

- Code/config/docs không được chứa drive path hoặc user-home path cố định.
- Source catalog và artifact catalog chỉ chứa relative paths dùng forward slash.
- External roots được truyền bằng `--raw-root`, `--output-root` hoặc environment.
- Runtime chặn POSIX absolute paths, Windows drive paths, UNC paths, `..`, và
  symlink/junction redirect ra ngoài declared root.
- Mọi thay đổi code/config sau P00 yêu cầu run ID mới; không resume snapshot cũ với
  implementation mới.
