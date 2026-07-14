# S1 matched-pair audit

## Nguồn và cấu trúc thực tế

Nguồn đã khóa là `data/source/financial/financial_statement_core_long.csv.gz`.

Audit trước operationalization cho thấy:

- 1.739.246 item rows;
- key tự nhiên: firm × fiscal year × canonical item × audit status;
- hai audit statuses cần dùng: `unaudited` và `audited`;
- toàn bộ records dùng VND và consolidated scope;
- không có duplicate exact-key, conflicting unit hoặc conflicting statement family trong nguồn hiện tại;
- không suy số firm-year từ tổng item rows.

Hai endpoints được Chapter/config cho phép nhưng chưa khóa empirical threshold:

| Endpoint | Canonical item | Firm-year item keys | Matched pre/post pairs | Missing/unmatched | Invalid zero denominator | Valid ratio computed |
|---|---|---:|---:|---:|---:|---:|
| Profit adjustment | `profit_after_tax` | 23.134 | 12.581 | 10.553 | 3 | 12.578 |
| Revenue adjustment | `net_revenue` | 22.702 | 12.281 | 10.421 | 72 | 12.209 |

Các matched counts trên là firm-year pairs, không phải item-row counts. Trong panel chính 2015–2025, P03 materialize một source result cho mỗi endpoint × mỗi firm-year, tổng cộng 46.822 S1 records.

## Deterministic pair construction

Pair key là `firm_id × fiscal_year × canonical_item`. Builder yêu cầu đúng một unaudited row và đúng một audited row. Nó fail-closed khi:

- thiếu pre, post hoặc cả hai;
- duplicate cùng audit status, kể cả duplicate giống nhau;
- conflicting value/unit/scope/family;
- value missing;
- denominator bằng 0 hoặc không qua floor đã khóa;
- period/scope/family mismatch.

Raw record references được sắp xếp deterministic; kết quả không phụ thuộc file order hay row order.

Ratio đã implemented là:

```text
abs(pre_audit_value - post_audit_value) / abs(post_audit_value)
```

## Kết quả run mới

| Reason code | Count |
|---|---:|
| `S1_EMPIRICAL_RULE_NOT_LOCKED` | 24.787 |
| `S1_PAIR_MISSING_PRE_AUDIT` | 16.873 |
| `S1_PAIR_MISSING_POST_AUDIT` | 4.101 |
| `S1_PAIR_MISSING_BOTH` | 986 |
| `S1_INVALID_DENOMINATOR_ZERO` | 75 |

Không có duplicate/conflict trong panel hiện tại. S1 opportunity hợp lệ sau denominator protection là 12.578 profit rows và 12.209 revenue rows. Vì threshold và denominator floor chưa được khóa, các valid ratios vẫn giữ outcome `Unknown`; S1 hiện có 0 positive và 0 explicit negative. Đây là fail-closed empirical blocker, không phải thiếu implementation.

Chi tiết theo year/endpoint nằm tại [s1_counts_by_year_endpoint.csv](docs/audits/s1_counts_by_year_endpoint.csv). Failure counts theo year/reason nằm tại [s1_pair_failures.csv](docs/audits/s1_pair_failures.csv).

## Empirical values còn phải khóa

Không tự suy các giá trị sau:

1. `source_catalog.profiles.financial_statement_core_long.evidence_mapping.audit_adjustment.minimum_absolute_denominator`;
2. `source_catalog.profiles.financial_statement_core_long.evidence_mapping.logical_sources[source_id=S1_profit_adjustment].materiality_threshold`;
3. `source_catalog.profiles.financial_statement_core_long.evidence_mapping.logical_sources[source_id=S1_revenue_adjustment].materiality_threshold`.

Sau khi khóa, phải tạo run mới vì protocol hash thay đổi. Không được resume run hiện tại.

