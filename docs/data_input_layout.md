# Bố trí dữ liệu đầu vào

## Hợp đồng vật lý

Production sử dụng **một file Parquet cho modeling** và **một CSV tách biệt cho sealed external validation**:

```text
<RAW_ROOT>/
|-- extract_provenance.json
`-- data/
    `-- source/
        |-- vn_pipeline_final_firm_year_2015_2025.parquet
        `-- known_case_registry.csv
```

`extract_provenance.json` là manifest nguồn, không phải dataset phân tích. `--raw-root` và biến
`DISSERTATION_RAW_ROOT` phải cùng trỏ tới `<RAW_ROOT>`.

Ba source ID phục vụ measurement và modeling cùng resolve tới Parquet và cùng một SHA-256:

- `financial_statement_core_long`;
- `audit_annual_long`;
- `sanction_evidence`.

`known_cases` resolve riêng tới `known_case_registry.csv`. Registry này được snapshot-lock tại P00,
không phải panel source, không tham gia training, calibration, model selection hoặc Gate 2, và chỉ
được mở tại P15 sau khi Gate 2 đã đóng. Tách vật lý registry tránh hai rủi ro: data-build bỏ quên phép
merge làm mất external validation, và known-case identifiers xuất hiện trong bytes mà các stage mô
hình hóa trước P15 có thể đọc.

## Grain và phạm vi của final Parquet

File final phải có đúng một dòng cho mỗi:

```text
firm_master_id x fiscal_year
```

Phạm vi production hiện hành:

- doanh nghiệp phi tài chính trên HOSE/HNX;
- năm tài chính 2015–2025;
- báo cáo năm hợp nhất;
- chỉ các firm-year thỏa rule listing overlap đã khóa;
- `prediction_time` là annual anchor được data-build job materialize và pipeline kiểm tra lại.

Trùng `firm_master_id x fiscal_year`, thiếu key, lệch prediction anchor hoặc khác tập firm-year giữa
P02 và P07 đều làm pipeline dừng fail-closed.

## Nhóm cột bắt buộc trong final Parquet

### Key, mẫu và provenance

```text
firm_master_id
issuer_ticker
fiscal_year
prediction_time
source_snapshot_hash
exchange_at_fye
industry_code
```

### Thành phần S1

```text
pat_unaudited
pat_audited
revenue_unaudited
revenue_audited
```

Pipeline vẫn tự áp dụng ngưỡng materiality đã khóa. Data-build job không được tạo nhãn S1 cuối cùng.
Thiếu một cặp trước/sau kiểm toán phải giữ là unknown.

### Thành phần S2

```text
audit_report_observed
audit_opinion_raw
audit_report_version_id
```

`audit_report_observed=false` không được chuyển thành clean opinion. Khi report được quan sát,
`audit_opinion_raw` phải map được vào taxonomy đã khóa hoặc được ghi nhận là normalization failure.

### Endpoint và provenance S3

```text
s3_source_opportunity
s3_broad_endpoint
s3_reporting_endpoint
s3_content_endpoint
s3_timeliness_endpoint
s3_document_ids_json
s3_first_label_known_date
s3_last_label_known_date
s3_taxonomy_codes_json
```

Quy tắc ba trạng thái:

- opportunity true: bốn endpoint phải là true/false;
- opportunity không true: bốn endpoint phải null;
- endpoint positive: `s3_document_ids_json` phải chứa ít nhất một document ID.

False ở đây chỉ là observed endpoint zero trong source-year hoàn chỉnh; không phải latent non-fraud.

### Predictors

File final phải chứa các predictor columns đã được data-build job tính. Tên cột vật lý phải khớp
`physical_column` trong `config/methodology/features.yaml`.

P07 không tính lại feature từ BCTC long. P07 chỉ:

- bind feature columns đã đăng ký;
- kiểm tra coverage, missingness, scale và redundancy;
- áp leakage firewall và model eligibility;
- tạo feature views và model matrix manifest.

Direct S1/S2/S3 components không được tự động đưa vào model matrix chỉ vì có mặt trong file final.
Known-case columns không được nhúng trong Parquet.

## Hợp đồng `known_case_registry.csv`

Registry có grain `case_id x firm_id x fiscal_year` và phải có các cột semantic sau:

```text
case_id
firm_id
fiscal_year
case_construct
role
training_include_flag
calibration_include_flag
model_selection_include_flag
external_validation_include_flag
```

P15 fail-closed khi:

- registry không khớp SHA-256 đã khóa tại P00;
- `case_construct` khác `CONFIRMED_FINANCIAL_REPORTING_CASE`;
- `role` khác `SIMULATION_EXTERNAL_VALIDATION`;
- training, calibration hoặc model-selection flag không phải false;
- external-validation flag không phải true;
- trùng `case_id x firm_id x fiscal_year`;
- registry không được cấu hình sealed trước development hoặc opening step khác P15.

Các cột review bổ sung như `confirmation_status`, `source_document_ids`, `source_datasets` và
`move_reason` có thể được giữ trong CSV để audit nhưng không thay đổi inclusion contract của P15.

## `extract_provenance.json`

Manifest phải nằm trực tiếp dưới `<RAW_ROOT>` và có đủ:

```json
{
  "vendor": "...",
  "vendor_product": "...",
  "pull_date": "YYYY-MM-DD",
  "vendor_version": "...",
  "extract_query": "...",
  "revision_policy": "...",
  "point_in_time_vintages_available": false
}
```

Không ghi password, connection string hoặc secret vào manifest.

## Trách nhiệm của data-build và case-review jobs

Việc nối SQL Server, listing, audit, enforcement và ownership diễn ra **ngoài production modeling
pipeline**. Data-build job phải xuất:

```text
data/source/vn_pipeline_final_firm_year_2015_2025.parquet
```

Case-review job phải xuất riêng:

```text
data/source/known_case_registry.csv
```

Data-build job phải fail khi:

- grain firm-year bị trùng;
- population hoặc annual scope sai;
- source components xung đột;
- feature crosswalk chưa được phê duyệt;
- missing evidence bị chuyển thành negative;
- S3 endpoint không nhất quán với source opportunity;
- row provenance hoặc source snapshot hash bị thiếu.

Case-review job phải fail khi role hoặc inclusion flags không đúng external-validation contract.
Không job nào được merge known-case identifiers vào final modeling Parquet.

## Tính bất biến

- Final Parquet và known-case registry đều bất biến trong một run và bị khóa SHA-256 tại P00.
- Ba semantic measurement views phải cùng trỏ tới một relative Parquet path và cùng hash.
- Known cases phải trỏ tới CSV riêng và chỉ được mở ở P15.
- Đổi bytes, schema, semantic binding, feature set, evidence construction hoặc known-case registry tạo run mới.
- Artifact chính thức nằm trong `artifacts/runs/<run-id>/` và không vào Git.
- `--resume` chỉ hợp lệ khi code, config, snapshot và cả hai physical inputs không đổi.

## Chuẩn bị raw root và chạy

```powershell
$rawRoot = "D:\Works\dissertation\final-input"

Copy-Item `
  "D:\Works\dissertation\dissertation-v4\data\source\known_case_registry.csv" `
  "$rawRoot\data\source\known_case_registry.csv" `
  -Force

uv run python scripts/run_pipeline.py `
  --run-id final-firm-year-2015-2025-v2 `
  --raw-root $rawRoot `
  --output-root "D:\Works\dissertation\artifacts\runs" `
  --through P17 `
  --workers 12
```

Trước khi chạy, cả hai file phải tồn tại:

```text
D:\Works\dissertation\final-input\data\source\vn_pipeline_final_firm_year_2015_2025.parquet
D:\Works\dissertation\final-input\data\source\known_case_registry.csv
```
