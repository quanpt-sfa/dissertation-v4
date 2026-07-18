# Bố trí dữ liệu đầu vào

## Gốc dữ liệu

Dữ liệu nằm trong chính repo hiện tại. `--raw-root` và biến
`DISSERTATION_RAW_ROOT` phải trỏ tới:

```text
D:\Works\dissertation\dissertation-v4
```

Source catalog chỉ đọc các đường dẫn chính xác dưới `data/source/`. Pipeline không dò file
ngoài catalog và không dùng bảng tích hợp hoặc bảng dẫn xuất làm nguồn phân tích độc lập.

## Cây thư mục

```text
data/
|-- source/                         # nguồn chính thức được discovery
|   |-- firm_identity_master_augmented.csv
|   |-- listing_history_expanded_step31_vietstock_profile_timeline.csv
|   |-- firm_event_sanction_panel.csv
|   |-- financial_statement_full_long.csv.gz
|   |-- bctc_audit_annual_long.csv
|   |-- bctc_industry_icb.csv
|   |-- ownership_year_latest_snapshot_long.csv.gz
|   |-- macro_cpi_annual.csv
|   |-- dividend_annual_long.csv.gz
|   `-- known_cases.csv
|-- config_reference/
|   `-- locked_data_assumptions.csv
|-- validation_only/
|   |-- firm_year_pipeline_base_2015_2026.csv.gz
|   |-- financial_statement_pre_post_pairs.csv.gz
|   `-- state_ownership_inference.csv
|-- README_START_INPUTS.md
|`-- manifest_sha256.csv
```

## Trạng thái discovery

| Profile | File | Vai trò | Trạng thái mặc định |
| --- | --- | --- | --- |
| `firm_identity_master` | `data/source/firm_identity_master_augmented.csv` | master định danh | bật, bắt buộc |
| `listing_history` | `data/source/listing_history_expanded_step31_vietstock_profile_timeline.csv` | lịch sử niêm yết | bật, bắt buộc |
| `sanction_evidence` | `data/source/firm_event_sanction_panel.csv` | evidence sự kiện | bật, bắt buộc |
| `financial_statement_core_long` | `data/source/financial_statement_full_long.csv.gz` | BCTC trước/sau kiểm toán | bật, bắt buộc |
| `audit_annual_long` | `data/source/bctc_audit_annual_long.csv` | ý kiến và công ty kiểm toán | bật, bắt buộc |
| `industry_icb` | `data/source/bctc_industry_icb.csv` | phân ngành | bật, bắt buộc |
| `ownership_snapshots` | `data/source/ownership_year_latest_snapshot_long.csv.gz` | sở hữu | bật, bắt buộc |
| `macro_cpi` | `data/source/macro_cpi_annual.csv` | quy đổi danh nghĩa | tắt đến khi feature registry yêu cầu |
| `dividend_annual_long` | `data/source/dividend_annual_long.csv.gz` | chính sách cổ tức | tắt đến khi feature registry khóa |
| `known_cases` | `data/source/known_cases.csv` | K1–K4 niêm phong | tùy chọn; chỉ mở tại P15 |

`config_reference/` không phải dữ liệu phân tích. Các rule đã khóa được chuyển vào module config
tương ứng. `validation_only/` chỉ dùng cho golden comparison/regression test và không có glob trong
source catalog.

## Các rule đầu vào đã khóa

- BCTC được coi là khả dụng ngày 31/03 năm `t+1`; snapshot ghi rule dẫn xuất này vào protocol hash.
- Năm 2026 là prospective và không tham gia retrospective evaluation.
- Price, market capitalization, enterprise value, return và liquidity bị loại khỏi predictor set.
- Sở hữu nhà nước chỉ được giữ nguyên hoặc giảm do thoái vốn; tăng bất thường tạo review flag.
- VSM/VSMS và VTS/VTSC dùng rule phân giải theo tên đã khóa.
- Audit firm bị thiếu được giữ là missing và có missing indicator; không mã hóa thành 0.

## Tính bất biến và Git

- File trong `data/` là đầu vào cục bộ, bất biến trong một run và bị `.gitignore` toàn bộ.
- `manifest_sha256.csv` được xác minh trước khi giải nén; snapshot tính lại SHA-256 cho từng source.
- Thêm file, đổi bytes, schema, semantic binding hoặc rule availability đều tạo protocol hash mới.
- Artifact chính thức nằm trong `artifacts/runs/<run-id>/` và cũng không vào Git.

## Lệnh chạy

```powershell
uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2026-start-v1 `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P17
```

Mỗi `run-id` là bất biến. Dùng `--resume` chỉ khi code, config và snapshot không đổi. Nếu thiếu
`risksets.data_cutoff` hoặc empirical binding khác, production runner phải dừng fail-closed tại đúng
blocker; không tự suy đoán giá trị.
