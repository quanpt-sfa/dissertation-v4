# Bố trí dữ liệu đầu vào

## Hợp đồng vật lý

Production tối thiểu sử dụng **một file Parquet cho modeling** và **một CSV tách biệt cho sealed external validation**:

```text
<RAW_ROOT>/
`-- data/
    `-- source/
        |-- vn_pipeline_final_firm_year_2015_2025.parquet
        `-- known_case_registry.csv
```

`--raw-root` và biến `DISSERTATION_RAW_ROOT` phải cùng trỏ tới `<RAW_ROOT>`.

Một manifest vendor-level `extract_provenance.json` có thể được đặt trực tiếp dưới `<RAW_ROOT>`, nhưng
không còn là file phải copy thủ công để một production bundle đã materialize có thể chạy. Nếu manifest
ngoài tồn tại, snapshot dùng nó và kiểm tra đầy đủ các trường đã khóa. Nếu manifest ngoài không tồn tại,
snapshot yêu cầu final Parquet phải chứa lineage nhúng duy nhất, không rỗng trong
`source_provenance_json`; pipeline khóa lineage đó cùng SHA-256 của final input và ghi rõ vendor-level
metadata là `UNAVAILABLE_NOT_RETAINED_IN_DERIVED_INPUT`. Pipeline không tự suy diễn hoặc bịa vendor,
pull date, query hay revision policy đã không được giữ lại trong derived input.

Nếu cả manifest ngoài lẫn embedded lineage đều không có, snapshot vẫn fail-closed trước P00.

Ba source ID phục vụ measurement và modeling cùng resolve tới Parquet và cùng một SHA-256:

- `financial_statement_core_long`;
- `audit_annual_long`;
- `sanction_evidence`.

`known_cases` resolve riêng tới `known_case_registry.csv`. Registry này được snapshot-lock tại P00
và được P01 audit về hash, schema, key và coverage, nhưng không phải panel source và không tham gia
training, calibration, model selection hoặc Gate 2. Nội dung case chỉ được diễn giải và chấm tại P15
sau khi Gate 2 đã đóng. Tách vật lý registry tránh hai rủi ro: data-build bỏ quên phép merge làm mất
external validation, và known-case identifiers xuất hiện trong modeling Parquet.

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
- `prediction_time` là annual anchor được data-build job materialize và pipeline kiểm tra lại;
- `exchange_or_board` là sàn tại cuối năm tài chính, được data-build job đổi tên từ trường upstream
  `exchange_at_fye`; vocabulary production được khóa ở `HOSE` và `HNX`.

Trùng `firm_master_id x fiscal_year`, thiếu key, lệch prediction anchor, thiếu board, board ngoài
vocabulary đã khóa hoặc khác tập firm-year giữa P02 và P07 đều làm pipeline dừng fail-closed.

## Nhóm cột bắt buộc trong final Parquet

### Key, mẫu và provenance

```text
firm_master_id
issuer_ticker
fiscal_year
prediction_time
source_snapshot_hash
exchange_or_board
industry_code
source_provenance_json
source_protocol_hash
source_run_id
source_unified_population_sha256
```

`source_provenance_json` là lineage của derived production input. Nó không được diễn giải lại thành
vendor-level extract metadata. Khi `extract_provenance.json` không có, snapshot chỉ dùng embedded
lineage này để chứng minh nguồn build và khóa provenance ở mức derived input; các vendor fields không
được giữ lại sẽ được ghi rõ là unavailable.

`exchange_at_fye` là tên trường ở supplemental unified source của data-build job; nó không phải tên
cột vật lý cuối cùng. Builder materialize trường này thành `exchange_or_board` trước khi publish final
Parquet và kiểm tra toàn bộ giá trị thuộc `HOSE`/`HNX`.

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
- giữ và kiểm tra domain metadata ngoài predictor matrix;
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

Các cột review bổ sung như `confirmation_status`, `known_case_role`, `source_document_ids`,
`source_datasets` và `move_reason` có thể được giữ trong CSV để audit nhưng không thay đổi inclusion
contract của P15. Cột canonical dùng để bind role là `role`; việc cùng tồn tại `known_case_role` không
tạo semantic ambiguity.

## `extract_provenance.json`

Manifest vendor-level là nguồn provenance ưu tiên khi có sẵn. Nó nằm trực tiếp dưới `<RAW_ROOT>` và
có đủ:

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

Nếu manifest này không được chuyển sang máy mới, snapshot không yêu cầu người chạy tái tạo hoặc đoán
metadata. Thay vào đó, final Parquet phải cung cấp embedded derived-input lineage. Snapshot ghi:

- `provenance_origin = embedded_derived_input_lineage`;
- SHA-256 của final Parquet;
- `source_protocol_hash`, `source_run_id`, `source_unified_population_sha256` nếu có;
- toàn bộ `source_provenance_json` đã nhúng;
- trạng thái vendor metadata là không được giữ lại trong derived input.

Không ghi password, connection string hoặc secret vào manifest hay embedded lineage.

## Trách nhiệm của data-build và case-review jobs

Việc nối SQL Server, listing, audit, enforcement và ownership diễn ra **ngoài production modeling
pipeline**. Data-build job phải xuất:

```text
data/source/vn_pipeline_final_firm_year_2015_2025.parquet
```

Data-build boundary dùng supplemental unified source có metadata population, trong đó
`exchange_at_fye` được đổi tên thành `exchange_or_board` ở final output. Builder phải fail nếu
metadata firm-year xung đột hoặc board không thuộc HOSE/HNX.

Case-review job phải xuất riêng:

```text
data/source/known_case_registry.csv
```

Data-build job phải fail khi:

- grain firm-year bị trùng;
- population hoặc annual scope sai;
- `exchange_or_board` thiếu hoặc ngoài HOSE/HNX;
- source components xung đột;
- feature crosswalk chưa được phê duyệt;
- missing evidence bị chuyển thành negative;
- S3 endpoint không nhất quán với source opportunity;
- row provenance hoặc source snapshot hash bị thiếu;
- `source_provenance_json` không đồng nhất trong final input.

Case-review job phải fail khi role hoặc inclusion flags không đúng external-validation contract.
Không job nào được merge known-case identifiers vào final modeling Parquet.

## Tính bất biến

- Final Parquet và known-case registry đều bất biến trong một run và bị khóa SHA-256 tại P00.
- Ba semantic measurement views phải cùng trỏ tới một relative Parquet path và cùng hash.
- External manifest, nếu có, được ưu tiên; nếu không có thì embedded lineage phải duy nhất và được khóa
  cùng final input hash.
- Known-case rows chỉ được diễn giải tại P15; các stage trước chỉ được dùng metadata audit đã khóa.
- Đổi bytes, schema, semantic binding, feature set, evidence construction, embedded lineage hoặc
  known-case registry tạo run mới.
- Artifact chính thức nằm trong `artifacts/runs/<run-id>/` và không vào Git.
- `--resume` chỉ hợp lệ khi code, config, snapshot và cả hai physical inputs không đổi.

## Chuẩn bị raw root và chạy

Nếu `data/source` nằm ngay trong repo như production bundle hiện hành, không cần tạo một raw-root khác
hoặc copy manifest thủ công. Chạy từ root repo:

```powershell
$projectRoot = (Resolve-Path ".").Path
$workspaceRoot = Split-Path $projectRoot -Parent
$rawRoot = $projectRoot
$outputRoot = Join-Path $workspaceRoot "dissertation-artifacts\runs"
$runId = "full-pipeline-" + (Get-Date -Format "yyyyMMdd-HHmmss")

uv run python scripts/run_pipeline.py `
  --run-id $runId `
  --raw-root $rawRoot `
  --output-root $outputRoot `
  --through P17 `
  --workers 4
```

Trước khi chạy, hai physical inputs sau phải tồn tại tương đối dưới raw root:

```text
<RAW_ROOT>/data/source/vn_pipeline_final_firm_year_2015_2025.parquet
<RAW_ROOT>/data/source/known_case_registry.csv
```

Nếu final Parquet không có embedded lineage, khi đó `extract_provenance.json` trở lại thành bắt buộc và
snapshot sẽ dừng trước P00 thay vì tạo provenance giả.