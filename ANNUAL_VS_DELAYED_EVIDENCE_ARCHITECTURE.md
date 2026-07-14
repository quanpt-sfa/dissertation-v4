# Annual versus delayed evidence architecture

## Quyết định thời gian đã khóa

Pipeline hiện phân biệt hai cơ chế evidence thay vì áp dụng một bộ lọc thời gian chung.

| Channel | Nguồn logic | Granularity | Temporal role | Availability rule | Maturity curve | Explicit negative |
|---|---|---|---|---|---|---|
| S1 | `S1_profit_adjustment`, `S1_revenue_adjustment` | source-result × firm-year | `annual_measurement_at_anchor` | `common_annual_anchor` | Không | Có, nhưng chỉ sau khi pair hợp lệ và rule empirical được khóa |
| S2 | `S2_audit_opinion` | source-result × firm-year | `annual_measurement_at_anchor` | `common_annual_anchor` | Không | Có: clean/unmodified |
| S3 | `S3_sanction_evidence` | event-level | `delayed_verification` | `actual_publish_date` | Có, 12/24 tháng | Không |

Với fiscal year `t`, common annual anchor là `31/03/t+1`. S1 và S2 được ghi nhận đúng tại anchor; không cần ngày công bố riêng của doanh nghiệp. S3 chỉ được quan sát khi `annual_anchor < publish_date <= annual_anchor + horizon`.

Run `dissertation-2015-2026-s1-s2-annual-v1` xác nhận 70.233 annual source-result records đều khớp anchor, với `annual_anchor_mismatch_count = 0`. Chỉ 385 S3 event records đi vào lag decomposition; S1/S2 không đi vào detection-lag curve.

## Contract theo stage

### P03

P03 giữ cả hai granularities trong cùng evidence ledger mà không tạo event giả:

- annual result có `evidence_record_id`, `evidence_record_kind=annual_source_result`, event identifiers null;
- delayed event có `event_id`, `event_cluster_id`, actual publication date và event-level deduplication;
- mọi row giữ `temporal_role`, `availability_basis`, `source_opportunity`, `opportunity_basis`, `outcome_basis`, raw record references và period-link provenance;
- annual result phải bằng prediction/annual anchor; mismatch làm pipeline fail-closed;
- S3 vẫn yêu cầu actual publication date và event identifiers.

### P04

P04 báo hai phần riêng:

- `annual_measurement_availability`: anchor match và opportunity counts cho S1/S2;
- `delayed_verification_maturity_curves`: pre-anchor, in-horizon, post-horizon và post-cutoff counts chỉ cho S3.

Không có annual record nào được dùng để tính detection lag.

### P05

P05 tạo bốn L0 sources trong ba channels:

- `L0:S1_profit_adjustment`;
- `L0:S1_revenue_adjustment`;
- `L0:S2_audit_opinion`;
- `L0:S3_sanction_evidence`.

L1 dùng quy tắc ba trạng thái:

| Điều kiện | L1 |
|---|---|
| Có ít nhất một source `True` | `True` |
| Mọi source đều có opportunity quan sát và mọi outcome đều `False` | `False` |
| Không có `True`, nhưng có outcome/opportunity unknown | `Unknown` |

Vì S3 chưa có opportunity indicator, `S1=False + S2=False + S3=Unknown` luôn là `Unknown`, không phải negative.

Các audit fields row-level gồm observed/positive/negative/unknown source counts, observed channel count và source-opportunity counts. Channel S1 vẫn là một mechanism dù có hai endpoints; S2 opinion subtypes và S3 sanction subtypes không bị tách thành channels giả.

### P06

P06 tách riêng:

- opportunity coverage;
- observed outcome fraction;
- positive incidence;
- explicit-negative incidence;
- unknown fraction;
- mature/prospective counts;
- observed-outcome và observed-opportunity overlap.

S3 event incidence không được gọi là coverage. Khi thiếu opportunity indicator, coverage là `NOT_ESTIMABLE_FROM_EVENT_ABSENCE`.

## Leakage firewall

Feature registry chặn các same-year semantics sau khi S1/S2 tham gia target:

- audit opinion;
- audit adjustment ratio;
- pre/post audit profit after tax;
- pre/post audit net revenue.

Audit firm không được dùng làm outcome. Nó chỉ có thể được đăng ký sau này như predictor/observability metadata nếu đáp ứng availability và feature-registry contract. Audited financial-statement components có khả năng reconstruct S1 cũng phải qua firewall; không có ngoại lệ benchmark mechanical nào được tự mở.

## Fail-closed boundaries

- Clean opinion chỉ nghĩa là không quan sát non-clean opinion.
- Below-threshold adjustment chỉ có thể nghĩa là không quan sát material adjustment sau khi threshold được khóa.
- Missing pair, invalid denominator, missing opinion và absence of sanction không bao giờ là negative.
- L2/L3 parameters, source quality và prevalence priors không được suy từ dữ liệu để làm stage chạy xa hơn.

