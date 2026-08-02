# Bố trí dữ liệu đầu vào

## Hợp đồng vật lý

Production pipeline chỉ nhận **một file dữ liệu vật lý**:

```text
<RAW_ROOT>/
|-- extract_provenance.json
`-- data/
    `-- source/
        `-- vn_pipeline_final_firm_year_2015_2025.parquet
```

`extract_provenance.json` là manifest nguồn, không phải dataset thứ hai. File Parquet là dataset duy
nhất được snapshot, audit và đọc trong P01–P15. `--raw-root` và biến
`DISSERTATION_RAW_ROOT` phải cùng trỏ tới `<RAW_ROOT>`.

Bốn source ID logic vẫn được giữ để bảo toàn contract của S1, S2, S3 và known-case validation:

- `financial_statement_core_long`;
- `audit_annual_long`;
- `sanction_evidence`;
- `known_cases`.

Cả bốn source ID cùng resolve tới đúng một đường dẫn Parquet và cùng một SHA-256. Đây là bốn
semantic views của một file, không phải bốn file đầu vào.

## Grain và phạm vi

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

## Nhóm cột bắt buộc

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

### Known cases nhúng trong firm-year

Các cột sau phải tồn tại để P15 chạy, nhưng được để null cho firm-year không thuộc known cases:

```text
known_case_id
known_case_construct
known_case_role
known_case_training_include_flag
known_case_calibration_include_flag
known_case_model_selection_include_flag
known_case_external_validation_include_flag
```

Known case hợp lệ phải có role `SIMULATION_EXTERNAL_VALIDATION`, không đi vào training,
calibration hoặc model selection và chỉ được mở ở P15.

### Predictors

File final phải chứa các predictor columns đã được data-build job tính. Tên cột vật lý phải khớp
`physical_column` trong `config/methodology/features.yaml`.

P07 không tính lại feature từ BCTC long. P07 chỉ:

- bind feature columns đã đăng ký;
- kiểm tra coverage, missingness, scale và redundancy;
- áp leakage firewall và model eligibility;
- tạo feature views và model matrix manifest.

Direct S1/S2/S3 components và known-case columns không được tự động đưa vào model matrix chỉ vì có
mặt trong file final.

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

## Trách nhiệm của data-build job

Việc nối SQL Server, listing, audit, enforcement, ownership và known-case sources diễn ra **ngoài
production modeling pipeline**. Data-build job phải xuất:

```text
data/source/vn_pipeline_final_firm_year_2015_2025.parquet
```

Nó phải fail khi:

- grain firm-year bị trùng;
- population hoặc annual scope sai;
- source components xung đột;
- feature crosswalk chưa được phê duyệt;
- missing evidence bị chuyển thành negative;
- known cases bị đưa vào modeling roles;
- S3 endpoint không nhất quán với source opportunity;
- row provenance hoặc source snapshot hash bị thiếu.

## Tính bất biến

- Parquet final là bất biến trong một run và bị khóa SHA-256 tại snapshot.
- Bốn semantic views phải cùng trỏ tới một relative path và cùng hash.
- Đổi bytes, schema, semantic binding, feature set hoặc evidence construction tạo run mới.
- Artifact chính thức nằm trong `artifacts/runs/<run-id>/` và không vào Git.
- `--resume` chỉ hợp lệ khi code, config, snapshot và Parquet không đổi.

## Lệnh chạy

```powershell
uv run python scripts/run_pipeline.py `
  --run-id final-firm-year-2015-2025-v1 `
  --raw-root "D:\Works\dissertation\final-input" `
  --output-root "D:\Works\dissertation\artifacts\runs" `
  --through P17 `
  --workers 12
```

Trước khi chạy, file phải nằm tại:

```text
D:\Works\dissertation\final-input\data\source\vn_pipeline_final_firm_year_2015_2025.parquet
```
