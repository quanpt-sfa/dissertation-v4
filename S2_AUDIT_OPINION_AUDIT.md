# S2 audit-opinion audit

## Nguồn và cấu trúc thực tế

Nguồn đã khóa là `data/source/audit/bctc_audit_annual_long.csv`.

Audit toàn nguồn cho thấy:

- 20.136 audit-opinion rows;
- key firm × fiscal year không có duplicate/conflict trong nguồn hiện tại;
- toàn bộ records là annual, consolidated và audited;
- raw taxonomy gồm 17.370 clean, 2.039 qualified, 395 disclaimer, 16 adverse và 316 missing;
- audit firm được giữ như metadata, không tham gia outcome normalization.

P03 materialize đúng một S2 source result cho mỗi firm-year trong panel. Record có opinion hợp lệ dùng annual anchor `31/03/t+1`; không yêu cầu actual audit-report publication date.

## Deterministic normalization

| Raw semantic | Normalized category | S2 outcome |
|---|---|---|
| Chấp nhận toàn phần | `clean` | `False` |
| Ý kiến ngoại trừ | `qualified` | `True` |
| Từ chối ra ý kiến | `disclaimer` | `True` |
| Ý kiến trái ngược | `adverse` | `True` |
| Missing/unmapped/conflicting | `unknown` | `Unknown` |

Warning, emphasis-of-matter và other-matter chưa được tự gán positive. Nếu xuất hiện mà chưa có config rule, chúng đi `EMPIRICALLY_PENDING`/unknown.

Duplicate opinions chỉ được chấp nhận khi tất cả normalize về cùng category và không có missing member. Nhiều categories hoặc duplicate lẫn missing đều fail-closed.

## Kết quả trong panel 2015–2025

| Metric | Count |
|---|---:|
| Firm-year source results | 23.411 |
| Opinion opportunity observed | 16.489 |
| Clean / explicit negative | 14.076 |
| Qualified | 2.009 |
| Disclaimer | 388 |
| Adverse | 16 |
| Non-clean / positive | 2.413 |
| Unknown | 6.922 |

Unknown gồm 6.912 firm-years không có audit-opinion record và 10 records có raw opinion missing. Không có conflicting hoặc unmapped opinion trong panel hiện tại.

Explicit `False` ở S2 chỉ có nghĩa “không quan sát non-clean audit opinion”; không phải bằng chứng doanh nghiệp không gian lận. S2 negative không thể tự biến L1 thành negative khi S1/S3 còn unknown.

Chi tiết theo fiscal year nằm tại [s2_opinion_counts_by_year.csv](docs/audits/s2_opinion_counts_by_year.csv). Normalization exceptions nằm tại [s2_normalization_exceptions.csv](docs/audits/s2_normalization_exceptions.csv).

