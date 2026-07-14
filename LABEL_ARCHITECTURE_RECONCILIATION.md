# Label Architecture Reconciliation

## Kiến trúc nhiều tầng sau sửa

| Tầng | Semantics hiện tại | Khả thi | Không được suy diễn |
|---|---|---:|---|
| Row inclusion | `train_include_flag=True` | Có; 535/535 raw S3 rows | Không phải outcome |
| S3 hard positive | `is_direct_label=True` trên accepted event | Có; 385 linked events, 60 mature firm-year positives ở horizon 12 tháng | False/missing không phải negative |
| L0 S3 | Source-year result sau event horizon filtering | Có cho positive/unknown | Absence không phải false |
| L1 | Union của observed L0 sources/channels | Có ở mức descriptive/PU: 60 positive, 0 negative, 21.643 mature unknown | Không phải confirmatory binary endpoint |
| L2 | Quality-delay weighted observed-source score | Chưa khả thi | Không tự đặt formula, quality, half-life hoặc minimum channels |
| L3 | Fixed-pi latent class | Structural unavailable | Không chạy với một channel và không tạo posterior giả |
| Anchor-PU | S3 direct positives làm anchor | Anchor formation khả thi; model chưa chạy được | Không tuyên bố clean positives; không chạy khi features/P08/P11 path bị chặn |
| Known cases | K1–K4 sealed IDs | Intentionally unavailable | Không có binding/data; không mở trước P15 |

## S3 — regulatory/sanction evidence

S3 đủ điều kiện tạo hard-positive event:

- outcome semantics rõ: `is_direct_label=True`;
- availability date: `publish_date`;
- period link: `label_year` kèm `label_year_source` và `label_confidence`;
- row inclusion tách riêng: `train_include_flag`;
- false/missing positive indicator và event absence đều giữ unknown;
- source/channel giữ nguyên S3, không tách các taxonomy subtype thành independent channels.

Nguồn không có opportunity indicator đáng tin cậy. Vì vậy source opportunity coverage là unknown cho toàn bộ 23.411 eligible firm-years. Event incidence chỉ là 60/23.411 firm-years; nó không phải coverage.

Các trường `primary_violation_l1`, `primary_violation_l2`, `construct_family` và `construct_target` đang là taxonomy/provenance của cùng cơ chế quyết định S3. Pipeline không dùng chúng để tạo nhiều channel. `TRADING_MARKET_CONDUCT` hoặc `market_conduct_violation` chỉ là separate-outcome candidates; chưa có endpoint mapping khóa để gọi là `SECURITIES_MANIPULATION`.

## S1 — before/after audit adjustments

Dữ liệu có 1.739.246 item rows, gồm 1.033.471 audited và 705.775 unaudited observations trên 50 canonical items. Đây là nguyên liệu cho matched before/after adjustments nhưng chưa đủ để operationalize label source.

Các yêu cầu còn thiếu:

- endpoint/outcome definition cụ thể;
- materiality hoặc direction rule;
- availability date semantics cho adjustment;
- firm-year source opportunity rule cho matched pair;
- positive/explicit-negative/unknown mapping;
- period-link rule và tests.

Do đó S1 vẫn là predictor source. Không tạo L0 source hoặc channel giả từ audit status.

## S2 — audit opinions

Dữ liệu có 17.370 clean, 2.039 qualified, 395 disclaimer và 16 adverse opinion values trong 20.136 opinion rows; 316 opinion rows còn missing. Tuy nhiên catalog S2 chưa có availability date và chưa có endpoint mapping positive/explicit-negative/unknown.

Không được coi clean opinion là explicit negative, cũng không được coi missing opinion/audit firm là zero. S2 vẫn là auxiliary source cho đến khi các semantics trên được khóa.

## Taxonomy layers được yêu cầu nhưng không tồn tại

Không tìm thấy trong raw source, config hoặc artifacts hiện tại:

- `FSF_STRICT`;
- `FSF_PROSECUTED`;
- `REGULATORY_FS_MISSTATEMENT`;
- `SECURITIES_MANIPULATION`;
- `HIGH_SUSPICION`.

Vì không có định nghĩa/binding, các tên này được ghi `intentionally unavailable`, không được suy ra từ `construct_target`, severity hoặc sanction subtype.

## Severity, confidence, inclusion và affected year

- `label_confidence` được giữ làm period-link/quality provenance; không chuyển thành numeric score.
- `has_fine`, `has_suspension`, `has_warning`, `has_remedy` và các sanction component khác là severity/quality inputs; không tự đổi nhãn.
- `affected_fiscal_year` có ở 142 records; 393 records dùng fallback được khai báo bằng `label_year_source=fiscal_year`.
- `train_include_flag` và `event_include_flag` chỉ là inclusion decisions.
- `is_direct_label` là hard-positive indicator; `is_indirect_label=0` không phải explicit negative.

Mapping đầy đủ theo raw field/value nằm tại `docs/audits/label_taxonomy_mapping.csv`.

## Soft-label capability

New run chỉ có một valid evidence channel, S3:

- 60 mature firm-years có 1 observed channel;
- 21.643 mature firm-years có 0 observed channels;
- không có pairwise channel overlap;
- L2 computable production rows: 0 (`L2_SCORING_FORMULA_NOT_LOCKED`);
- L3 computable rows: 0 (`INSUFFICIENT_CHANNELS`);
- strict nested channel selection: không khả thi.

L1 hiện phù hợp cho mô tả positive-unlabeled và làm anchor input. Nó không đủ cho confirmatory Track A binary modeling vì không có explicit negative và mọi fold chỉ có một observed class.

Anchor-PU có source anchor `sanction_evidence` và 60 mature positives. Model vẫn chưa chạy được vì feature registry rỗng; ngoài ra production path P08/P10/P11 đang fail-closed. Đây là `anchor input available`, không phải `Anchor-PU analysis complete`.
