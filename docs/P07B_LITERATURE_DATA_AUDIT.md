# P07B — Literature-Backed Feature Data Sufficiency Audit

P07B is a non-production, immutable researcher audit. It resolves the supplied
literature bundle, inventories authorised and prohibited data sources, and reports
literature-to-data mappings, coverage, temporal availability, and named gaps.

It does not read outcomes or known cases, rank variables by predictive performance,
fit preprocessing, or make a feature operational. In particular,
`data/validation_only/financial_statement_pre_post_pairs.csv.gz` is reported as
`PREPOST_DATA_NOT_AUTHORIZED_FOR_PRODUCTION` unless the protocol owner authorises a
separate derived production source.

Run it only with an extracted, externally stored literature bundle:

```powershell
uv run python scripts/p07b_literature_data_audit.py `
  --run-id <new-run-id> `
  --literature-root <extracted-fsf-directory> `
  --literature-archive D:\Works\MarkItDown\output\fsf.zip
```

P07B leaves `measurement.primary_target_id`, all P07 placeholders, and the S3
next-calendar-year estimand unchanged. Researcher approval of the generated YAML
decision template is required before a later operational P07 change.
