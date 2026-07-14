# Empirical outcome capability audit

Run audited: `dissertation-2015-2026-cutoff-20260331`  
Locked protocol hash: `cad146317f18d83dabd19ff8544f0518ce3da3c140492ab2b3fbce4ba3013e95`  
Data cutoff: `2026-03-31`  
Audit mode: read-only validation of the existing run; no run artifact was changed or regenerated.

## 1. Executive conclusion

The current run does not contain 56 binary observations with both classes. It contains 56 observed positive sanction outcomes, zero explicit negatives, and 21,647 mature firm-years whose L1 outcome remains unknown. A further 1,708 firm-years from fiscal year 2025 are prospective/immature at the 12-month horizon.

The claim that folds 2021–2024 have no positive class is factually incorrect. Their current positive counts are 10, 3, 9 and 7. They are ineligible because every fold is positive-only, all positive counts are below the locked sensitivity threshold of 15, and the unobserved remainder cannot be treated as negative.

Five implementation bugs were confirmed and fixed in code without changing this run:

1. P03 collapsed all same-source events for a firm-year before P05 applied the prediction horizon. A pre-prediction event could therefore hide a valid future event. Four 12-month positives were lost: `C69/2018`, `CVN/2024`, `GKM/2024`, and `PSH/2024`.
2. P04 counted negative detection lags as “available within horizon” and approximated calendar months with 365.25-day arithmetic. Its reported 87.06% and 97.57% are not valid future-horizon coverage rates.
3. P05 populated `mature_row_count` with the number of non-missing sealed outcomes rather than the number of mature risk-set rows. It also lacked an explicit one-class fail-closed check.
4. P06 labeled the fraction of rows with an observed event outcome as `coverage_rate`. That quantity is event incidence, not source opportunity coverage. Code now reports coverage as unavailable unless an opportunity indicator exists.
5. P05 overwrote the structural L3 failure (`INSUFFICIENT_CHANNELS`) with `EMPIRICALLY_PENDING` when the fixed-π grid was empty. Structural unavailability now takes precedence over unlocked empirical parameters.

After the event-ledger fix, an in-memory counterfactual using the same snapshot yields 60 rather than 56 primary-horizon positives. It still yields zero explicit negatives. Therefore the fix does not make confirmatory binary Track A feasible, and it does not make L2 or L3 feasible.

## 2. Authority and evidence reviewed

Method authority:

- the latest available Chapter 3 design and Appendix B decisions in `C:\Users\quanp\Downloads\Chapter3_Python_Pipeline_Design_Agent_Safe_v21.md`;
- `C:\Users\quanp\Downloads\Chapter3_Methodology_Simulation_Technical_Fixes_v19.md`;
- locked D01–D45 traceability, especially D01–D06, D10, D14, D17–D22, D25, D27 and D34.

Repository/run evidence:

- `config/pipeline.yaml`, source catalog, study, evidence, risk-set, measurement and fold modules;
- the P00 locked registry and snapshot manifest;
- every official artifact and manifest from P03 through P09;
- the raw normalized sanction, audit and financial inputs for capability diagnostics only;
- the P02 panel/master needed to audit entity-year linkage.

The interpretation follows the locked rules:

- endpoint evidence must satisfy `prediction_time < availability_date <= prediction_time + h`;
- absent evidence is not zero;
- immature follow-up is not negative;
- source coverage requires observed opportunity/verification, not merely an event file;
- L2/L3 require substantive independent-channel support and locked empirical parameters.

## 3. How the 56 sealed outcomes were formed

The complete chain is:

| Step | Count | Interpretation |
|---|---:|---|
| Snapshot sanction rows | 535 | All have `train_include_flag=True`; all are event positives under the current mapping. |
| P03 accepted entity-year links | 385 | The firm-year exists in the P02 panel. |
| P03 unlinked rows | 150 | 111 pre-sample, 32 missing a within-sample panel firm-year, 7 beyond the available panel. |
| P03 upstream duplicates | 0 | `bundle_id`, used as cluster key, is unique in all 535 rows. |
| Current P03 ledger rows | 371 | P03 collapsed multiple events to firm-year × source × channel before horizon filtering. |
| Current ledger rows before prediction time | 267 | Correctly excluded by P05, but incorrectly counted by the old P04 maturity fraction. |
| Current ledger rows in `(prediction, +12m]` | 56 | All have outcome `True`. |
| Current ledger rows in `(12m, 24m]` | 39 | Outside the primary horizon. |
| Current ledger rows after 24m | 9 | Outside both locked horizons. |
| P05 sealed outcomes | 56 | Only non-null L1 rows are sealed; all 56 are positive. |

P05 constructs one `L0:sanction_evidence` row and one L1 row for each of 23,411 panel rows, producing 46,822 rows in `l0_l1_inputs`. For a mature row:

- an in-window positive sanction makes L0 and L1 `True`;
- an explicit source `False` could contribute an explicit negative, but none exists;
- no in-window observed source result leaves L0 and L1 null;
- only non-null L1 is copied to `sealed_outcomes`.

This is why the sealed store has 56 rows rather than 21,703 mature rows. It is a positive-event store under the current data, not a complete binary-outcome cohort.

## 4. Positive, explicit negative and unknown by fiscal year

The machine-readable year × source × channel table is [outcome_counts_by_year_source_channel.csv](docs/audits/outcome_counts_by_year_source_channel.csv). Because the locked registry has only one evidence source, every row is `sanction_evidence × S3`.

| FY | Mature | Positive in run | Explicit negative | Mature unknown | Immature | Positive after event fix |
|---:|---:|---:|---:|---:|---:|---:|
| 2015 | 2,244 | 4 | 0 | 2,240 | 0 | 4 |
| 2016 | 2,552 | 6 | 0 | 2,546 | 0 | 6 |
| 2017 | 2,604 | 3 | 0 | 2,601 | 0 | 3 |
| 2018 | 2,582 | 3 | 0 | 2,579 | 0 | 4 |
| 2019 | 2,033 | 4 | 0 | 2,029 | 0 | 4 |
| 2020 | 2,002 | 7 | 0 | 1,995 | 0 | 7 |
| 2021 | 1,965 | 10 | 0 | 1,955 | 0 | 10 |
| 2022 | 1,952 | 3 | 0 | 1,949 | 0 | 3 |
| 2023 | 1,899 | 9 | 0 | 1,890 | 0 | 9 |
| 2024 | 1,870 | 7 | 0 | 1,863 | 0 | 10 |
| 2025 | 0 | 0 | 0 | 0 | 1,708 | 0 |
| **Total** | **21,703** | **56** | **0** | **21,647** | **1,708** | **60** |

“Unknown” here means mature but without a source result sufficient to construct L1. It does not mean a negative sanction finding. Fiscal year 2025 is reported separately as immature rather than mixed into mature unknown.

## 5. Fold eligibility and class counts

The machine-readable table is [fold_eligibility_class_counts.csv](docs/audits/fold_eligibility_class_counts.csv).

| Fold | Actual mature rows | P05 reported mature | Positive in run | Positive after fix | Explicit negative | Mature unknown after fix | Binary classes present |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2020 | 2,002 | 7 | 7 | 7 | 0 | 1,995 | No |
| 2021 | 1,965 | 10 | 10 | 10 | 0 | 1,955 | No |
| 2022 | 1,952 | 3 | 3 | 3 | 0 | 1,949 | No |
| 2023 | 1,899 | 9 | 9 | 9 | 0 | 1,890 | No |
| 2024 | 1,870 | 7 | 7 | 10 | 0 | 1,860 | No |
| 2026 | 0 | 0 | 0 | 0 | 0 | 0 | No |

The old `mature_row_count` exactly equaled the sealed positive count, demonstrating the P05 counting bug. The corrected fold summary derives maturity from `risk_sets` and marks fully nested folds fail-closed when either binary class is absent.

The locked thresholds are 25 positives for confirmatory status and 15–24 for sensitivity status. Thus 2021–2024 fail twice:

- their positive counts are 10, 3, 9 and 7 in the run (10, 3, 9 and 10 after the event fix), all below 15;
- there are no explicit negatives, so standard binary training/evaluation has only one observed class.

## 6. Source/channel horizon diagnostics

Only `sanction_evidence × S3` can be evaluated. Two distinct concepts must not be conflated:

1. **event timing/incidence**: among linked event records, when did an event occur relative to prediction time?;
2. **source opportunity coverage**: for how many risk-set rows was the source actually observable/searched, including explicit no-event results?

The run has evidence for the first concept but not the second.

| Measure | 12 months | 24 months |
|---|---:|---:|
| Old P04 reported fraction | 87.06% | 97.57% |
| Current-ledger strict future-window fraction | 56/371 = 15.09% | 95/371 = 25.61% |
| Raw accepted event rows in strict future window | 61/385 = 15.84% | 102/385 = 26.49% |
| Unique linked firm-years with an in-window event after fix | 60 | 101 |
| Mature analysis positives in the run | 56/21,703 = 0.2580% | 88/19,833 = 0.4437% |
| Mature analysis positives after event fix | 60/21,703 = 0.2765% | 91/19,833 = 0.4588% |
| Source opportunity coverage | Unknown | Unknown |

The 24-month mature denominator is 19,833 because fiscal year 2024 is not mature at 24 months under the same cutoff. The 101 unique event-linked firm-years include 10 fiscal-year-2024 rows; only 91 belong to the mature 24-month cohort.

The old P04 percentages equal all records with `lag <= horizon`, so they include 267 negative-lag rows. They describe neither future evidence incidence nor source coverage.

## 7. Why P06 has one channel

Snapshot discovery found seven enabled sources, but P03 and P06 intentionally select only sources whose locked role is `evidence`.

| Discovered source | Channel label | Locked role | P03–P06 status |
|---|---|---|---|
| `sanction_evidence` | S3 | evidence | Included. |
| `financial_statement_core_long` | S1 | predictor | Excluded; no locked event/outcome transformation. |
| `audit_annual_long` | S2 | auxiliary | Excluded; no availability-date semantic or locked outcome. |
| `firm_identity_master` | S3 | reference | Excluded by design. |
| `listing_history` | S3 | reference | Excluded by design; exit is not negative. |
| `industry_icb` | S1 | predictor | Excluded by design. |
| `ownership_snapshots` | S1 | predictor | Excluded by design. |

The channel labels S1/S2/S3 are descriptive source metadata. They do not override source role or manufacture evidence. P06 receives one expected channel because `_evidence_sources()` and P06 both filter `role == evidence`.

The macro file exists on disk but `macro_cpi` is disabled in the source catalog, so it is not part of this snapshot. Dividend is disabled, and the optional known-case source matched no file. None is an outcome channel for this audit.

Detailed proposed mapping work is in [SOURCE_CHANNEL_MAPPING_ACTIONS.md](docs/audits/SOURCE_CHANNEL_MAPPING_ACTIONS.md).

## 8. Is absence of sanction preserved as missing?

Yes, in the current P05 outcome construction:

- raw sanction outcome values: 535 `True`, 0 `False`, 0 null;
- L1 sealed outcomes: 56 `True`, 0 `False`;
- mature rows without an in-window observed result: 21,647 null;
- immature rows assigned negative: 0;
- exit/code-change rows assigned negative: 0.

The source/channel matrix initializes every expected source to null. `aggregate_l1()` returns `False` only when every expected source is explicitly observed `False`; a mix of missing and false remains null. With the present source, absence from the sanction event file is therefore retained as unknown.

However, this also means the current data cannot support an ordinary positive-versus-negative binary endpoint. A future explicit-negative channel must be backed by source opportunity evidence; it cannot be created by filling these nulls with zero.

## 9. Entity, fiscal-year, availability-date and horizon audit

### 9.1 Entity links

- All 535 sanction rows have high ticker-match confidence and direct label links.
- P03 accepts 385 rows because their canonical firm-year exists in the P02 panel.
- Of 150 unlinked rows, 111 predate the sample, 32 target a 2015–2025 firm-year absent from the financial panel, and 7 target fiscal year 2026, for which no panel row exists.
- Five unlinked tickers (`ATP`, `FBT`, `SD8`, `VCH`, `VKP`) are absent from the P02 `firm_master`, but all five are present in the identity input. They have no financial-statement panel rows. This is not evidence of a ticker-match error; it exposes that P02 builds its master from the financial source and cannot currently ingest the atemporal identity master.
- The 32 within-sample unlinked events need a source-by-source panel coverage review. They must not be forced onto a neighboring fiscal year.

Verdict: no demonstrated normalization/alias error among accepted records; a P02 master-design limitation and genuine missing firm-years remain.

### 9.2 Fiscal-year links

- 142/535 rows use `affected_fiscal_year`; it equals `label_year` in every one of those rows.
- 393/535 use a generic `fiscal_year` fallback.
- Among the 385 accepted events, 113 use explicit affected fiscal year, 265 use high-confidence fallback, and 7 use medium-confidence fallback.
- P03 does not preserve `label_year_source` or `label_confidence`, and therefore cannot enforce or reproduce a primary-endpoint confidence threshold.

Verdict: no arithmetic mismatch was found where affected fiscal year exists, but the fallback links are not sufficiently evidenced in the artifact. This is a source-semantic/config gap, not a reason to guess or relabel periods.

### 9.3 Availability dates

- All 535 `publish_date` values parse successfully; range `2007-03-30` to `2026-07-02`.
- All 23,411 prediction times equal 31 March of year `t+1`.
- Two raw events have publication dates after the cutoff; both are fiscal-year-2026 rows and are unlinked, so they do not enter the 56 outcomes.
- The endpoint filter is strict: same-day evidence would be excluded. No accepted ledger event has exactly zero detection lag.
- P03's old availability registry omitted canonical firm and fiscal year, weakening reproducibility of link failures. Code now includes both keys for future runs.

Verdict: no future-after-cutoff event entered a mature outcome. The principal date error was the P04 negative-lag maturity calculation, now fixed.

### 9.4 Horizon filtering

The strict P05 date predicate is correct. The failure occurred earlier: current P03 retained only the minimum availability date for a firm-year/source/channel. For six firm-years at 24 months, including four at 12 months, that minimum was pre-prediction while a later event was in-window.

Lost at 12 months:

- `C69/2018`;
- `CVN/2024`;
- `GKM/2024`;
- `PSH/2024`.

Additional losses visible at 24 months:

- `ABW/2022`;
- `SSN/2019`.

The fix makes the evidence ledger event-level and aggregates only after horizon filtering.

## 10. P07–P09 downstream capability evidence

- P07 has an empty feature registry. The leakage registry passes because no features were materialized, not because a production feature set is ready.
- P08 has an empty scenario registry and `mcse_report.status=SKIPPED`. No simulation config was changed during this audit.
- P09 creates the configured temporal folds, but every fold uses the unweighted mature-cohort estimand with `NO_OBSERVED_VERIFICATION`. P06 has no observed verification channel or pre-decision verification features, so propensity/IPW and overlap-weight sensitivity are skipped.
- P09 inherits the descriptive fold roles from P05. It does not repair class capability.

These artifacts are internally consistent with the current locked registry, apart from the counting/coverage bugs identified above.

## 11. Track A feasibility verdict

| Measurement | Current empirical feasibility | Verdict |
|---|---|---|
| L1 | Positive event identification works and the anchor source is registered. Ordinary supervised L1 has zero explicit negatives; every fully nested fold has fewer than 15 positives. Anchor-PU could be explored because unlabeled rows remain unlabeled, but the current feature registry is empty and confirmatory binary evaluation is unavailable. | **Partially feasible for positive/PU development only; not feasible for confirmatory Track A.** |
| L2 | One evidence channel only; scoring formula, quality mapping and delay half-life are unlocked; no opportunity coverage. | **Not feasible with current data/config.** |
| L3 fixed-π | One evidence channel; fixed-π grid and source-accuracy priors are unlocked; no independent overlap and no fold-local estimable target. The run artifact says `EMPIRICALLY_PENDING`, but this is the confirmed status-precedence bug; the correct status is structural `UNAVAILABLE_BY_DESIGN`. | **Not feasible with current data/config.** |

Hierarchical-π remains sensitivity-only and cannot be used to manufacture identification. Adding a channel label without a valid independent evidence process would not change these conclusions.

## 12. Applied code corrections and protocol-hash impact

Applied implementation changes:

- event-level P03 ledger with `event_id` and `event_cluster_id` contract fields;
- P05 horizon-aware multi-event aggregation;
- strict positive-lag, calendar-month P04 horizon fractions with explicit unknown opportunity coverage;
- corrected P05 mature/class counts and one-class fail-closed role;
- structural L3 channel failure taking precedence over missing fixed-π parameters;
- P06 separation of event-outcome fraction from source coverage;
- tests for each failure mode.

No source role, empirical link-confidence rule, outcome mapping, L2/L3 parameter, feature registry or simulation scenario was changed.

The schema/column contract changes are registry-owned configuration changes. Therefore the semantic protocol hash for a future run changes from:

```text
cad146317f18d83dabd19ff8544f0518ce3da3c140492ab2b3fbce4ba3013e95
```

to the currently compiled value:

```text
88c6eeedd02b87849bdfa26d864cffeb8937a00c1d73b266cf69ccfeb8a3ab13
```

Code-only fixes would not by themselves change the semantic registry hash, but they change producer behavior and code provenance. The event-level schema v3 and new semantic columns do change the protocol hash. The existing run must not be resumed under the corrected implementation; a new run is required after any empirical source-mapping decisions are locked.

## 13. Reproducibility and quality-gate status

Before and after quality gates, the audited run tree contained 102 files, 19,155,686 bytes, with the same deterministic whole-tree digest:

```text
224b4ae24d2a546332a4f1e20f4faa47deceb285078e5000cec7ba86b7ae34e8
```

This confirms that the audit and code corrections did not mutate any artifact of the audited run.

| Gate | Result |
|---|---|
| `bootstrap_repository.py --write` | PASS; generated schema catalog updated for evidence-ledger v3. |
| `bootstrap_repository.py --check` | PASS. |
| `ruff check .` | PASS. |
| `ruff format --check .` | PASS; 107 files already formatted. |
| `pyright` | PASS; 0 errors, 0 warnings, 0 informations. |
| `pytest -q` | PASS; 133 passed, 12 third-party deprecation warnings. |
| `pre-commit run --all-files` | PASS; all configured hooks passed. |
| `git diff --check` | PASS. |

No pipeline stage was rerun and no simulation configuration was edited.
