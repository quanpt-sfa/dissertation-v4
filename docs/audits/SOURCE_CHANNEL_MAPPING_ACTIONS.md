# Source/channel mapping actions

Audit target: `dissertation-2015-2026-cutoff-20260331`.

This file separates required corrections from empirical choices. None of the proposed source/config changes below has been applied. In particular, no absent sanction has been converted to an explicit negative and no synthetic evidence channel has been created.

## Priority 0 — applied implementation corrections

1. Preserve one row per deduplicated event in `evidence_ledger` by carrying `event_id` and `event_cluster_id`. P05 must apply `prediction_time < availability_date <= horizon_end` before aggregating events to source and channel outcomes.
2. Calculate P04 horizon fractions only from strictly positive detection lags. A pre-prediction event is not “available within the future horizon.”
3. Report P05 mature fold counts from `risk_sets`, not from the number of non-missing sealed outcomes, and fail closed when only one binary class is observed.
4. P06 must not label event incidence as source coverage. Coverage remains unavailable until an opportunity/verification indicator exists.

These are code/contract fixes. They require a new run; the audited run remains unchanged.

## Priority 1 — sanction evidence mapping, proposed only

Current catalog mapping:

```yaml
resolved_semantics:
  fiscal_year: label_year
  availability_date: publish_date
  outcome: train_include_flag
  event_id: firm_event_id
  event_cluster_id: bundle_id
```

Problems:

- `train_include_flag` is a row-selection field, not an empirical negative/positive observation. It is `True` for all 535 rows.
- `bundle_id` is unique in all 535 rows, while the raw `event_cluster_id` column is completely missing. Therefore no upstream duplicates can currently be demonstrated or collapsed.
- 393/535 period links use `fiscal_year` fallback rather than `affected_fiscal_year`; 30 rows have medium label confidence, including 7 accepted rows. Neither link basis nor confidence reaches P03.

Required patch design:

1. Add a dedicated input field such as `evidence_outcome` for event-level observed outcomes. Included sanction events may be explicitly positive, but excluded rows must be filtered or isolated and never interpreted as negative.
2. Register `train_include_flag` as a row-inclusion semantic, separate from `outcome`.
3. Register and carry `label_year_source`, `label_confidence`, `ticker_match_method`, `ticker_match_confidence`, `label_link_type`, and the real upstream cluster identifier into the availability/link audit.
4. Add a locked primary-endpoint eligibility rule for entity-link and fiscal-year-link confidence. The exact accepted values must be supplied from the empirical protocol; this audit does not guess them.
5. Add a source-opportunity table only if the collection process can prove that a firm-year was actually searched/observable. An opportunity with an explicit no-event result can support an explicit negative; mere absence from the event file remains unknown.

Files/owners that would need a later empirical patch:

- `config/methodology/source_catalog.yaml`: semantic aliases and required fields;
- `config/methodology/evidence.yaml`: locked link-eligibility and opportunity rules;
- raw normalized sanction export: dedicated outcome, inclusion, link-confidence and cluster fields;
- P03 contract/tests: propagation and fail-closed filtering for the newly locked fields.

## Priority 2 — candidate S1 financial channel, proposed only

`financial_statement_core_long` is discovered as S1 but registered as `role: predictor`. It contains 23,411 firm-years; 12,736 have both audited and unaudited records. That is sufficient to investigate pre/post adjustment construction, but not sufficient to declare an evidence outcome.

Before S1 can enter P03–P06, lock all of the following:

- the item/scope pairing and duplicate-resolution rule;
- the substantive adjustment outcome and threshold;
- the event availability date distinct from, or explicitly related to, prediction time;
- source opportunity/coverage semantics;
- period-link confidence and missing-pair treatment.

The current derived S1 availability date is 31 March of year `t+1`, which is also the current prediction time. An S1 value available exactly at prediction time cannot be introduced as a future endpoint under the strict rule `prediction_time < event_time`. It may instead be a baseline predictor unless Chapter 3 and the empirical timestamps support a later evidence event.

## Priority 3 — candidate S2 audit channel, proposed only

`audit_annual_long` is discovered as S2 but registered as `role: auxiliary`. It contains 20,136 firm-years and opinion categories, but has no registered availability-date semantic. It cannot enter P03 without an actual or protocol-authorized derived availability date and a locked outcome rule.

Required empirical decisions:

- which audit indicator/opinion constitutes evidence;
- whether and how a no-signal audit record becomes an explicit negative;
- the event availability date;
- repeated-row collapse and audit-firm missingness treatment;
- source opportunity coverage.

Do not assign S2 merely from its catalog channel label. Channel membership becomes analytical only after the source satisfies the evidence contract.

## Sources intentionally outside P03–P06

| Source | Current role | Reason not to map as evidence |
|---|---|---|
| `firm_identity_master` | reference | Entity vocabulary, not an outcome process. |
| `listing_history` | reference | Lifecycle/risk-set support; exit is not a negative outcome. |
| `industry_icb` | predictor | Industry classification, not evidence. |
| `ownership_snapshots` | predictor | Ownership feature source, not evidence. |

There is a separate P02 design issue: the atemporal identity master cannot currently contribute to `firm_master` because P02 only accepts year-bearing `panel_mapping` sources. The five sanction tickers absent from the P02 master are present in the identity input but have no financial panel rows. Extend P02 with an atemporal master mapping rather than fabricating firm-years.
