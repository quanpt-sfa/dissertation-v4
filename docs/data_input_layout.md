# Bố trí dữ liệu đầu vào

## Gốc dữ liệu

Dữ liệu nằm trong chính repo hiện tại. `--raw-root` và biến
`DISSERTATION_RAW_ROOT` phải trỏ tới:

```text
D:\Works\dissertation\dissertation-v4
```

Source catalog chỉ đọc các đường dẫn tương đối bên dưới `data/`. Pipeline không dò
sang `D:\Works\dissertation\v2\repo`, thư mục Downloads hoặc bất kỳ repo nào khác.

## Cây thư mục cần dùng

```text
data/
|-- processed/
|   |-- panel_wide_clean.parquet
|   `-- panel_long_clean.parquet
|-- raw/
|   |-- fiinpro/
|   |   |-- audit_opinions/
|   |   |   `-- *.xlsx
|   |   |-- capital_ownership/
|   |   |   `-- *.xlsx
|   |   `-- quarterly_exports/
|   |       `-- *.xlsx
|   |-- vietstock/
|   |   `-- listing_history.csv
|   |-- regulatory/
|   |   `-- research_labels.csv
|   |-- reference/
|   |   `-- ThongTinCoBan.xlsx
|   `-- known_cases/
|       `-- known_cases.csv
`-- standardized/
    |-- audit_opinions.parquet
    |-- capital_ownership.parquet
    |-- quarterly_financials/
    |-- listing_history.parquet
    |-- regulatory_labels.parquet
    `-- firm_reference.parquet
```

## Danh mục file

| Profile | Vị trí trong repo | Số lượng | Bắt buộc |
| --- | --- | ---: | --- |
| Core wide panel | `data/processed/panel_wide_clean.parquet` | 1 | Có |
| Core long panel | `data/processed/panel_long_clean.parquet` | 1 | Có |
| Ý kiến kiểm toán | `data/raw/fiinpro/audit_opinions/*.xlsx` | 1+ | Có |
| Vốn và sở hữu | `data/raw/fiinpro/capital_ownership/*.xlsx` | 0+ | Không |
| BCTC xuất theo quý | `data/raw/fiinpro/quarterly_exports/*.xlsx` | 1+ | Có |
| Lịch sử niêm yết | `data/raw/vietstock/listing_history.csv` | 1 | Có |
| Nhãn sự kiện quản lý | `data/raw/regulatory/research_labels.csv` | 1 | Có |
| Thông tin doanh nghiệp | `data/raw/reference/ThongTinCoBan.xlsx` | 1 | Có |
| Known cases K1–K4 | `data/raw/known_cases/known_cases.csv` | 0 hoặc 1 | Không; chỉ mở tại P15 |

`ThongTinCoBan.xlsx` được đọc ở `Sheet1`, header row 8. Các file XLSX còn lại
được kiểm tra header thực tế trong snapshot và P01; pipeline không đổi tên hoặc di
chuyển file để làm cho catalog khớp.

## Nguyên tắc lưu dữ liệu

- `data/raw/` là đầu vào bất biến; không ghi đè từ notebook hoặc stage phân tích.
- `data/processed/` chứa hai panel lõi do người dùng đặt vào.
- `data/standardized/` dành cho lớp chuyển đổi ổn định nếu bổ sung bước chuẩn hóa;
  artifact chính thức của một run vẫn nằm trong `artifacts/runs/<run-id>/`.
- Dữ liệu thật bị `.gitignore`; các file `.gitkeep` chỉ giữ cấu trúc thư mục.
- File thiếu, schema sai, header sai hoặc semantic field không ánh xạ được phải làm
  discovery/P01 dừng. Không được âm thầm chọn file gần giống.

## Lệnh chạy

```powershell
uv run python scripts/run_pipeline.py `
  --run-id dissertation-2015-2025-final `
  --raw-root "D:\Works\dissertation\dissertation-v4" `
  --output-root "D:\Works\dissertation\dissertation-v4\artifacts\runs" `
  --through P17
```

Mỗi `run-id` là bất biến. Nếu thư mục run đã tồn tại, dùng một `run-id` mới; không
xóa hoặc ghi đè run cũ để tái sử dụng tên.
