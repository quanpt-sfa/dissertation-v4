# P00–P17 completion report

Audit date: 2026-07-14  
Authority: `Chapter3_Methodology_Simulation_Technical_Fixes_v19.md`, Appendix B
D01–D45, then the locked registry compiled from `config/pipeline.yaml`.

## Overall verdict

`PARTIAL — NOT METHOD-COMPLETE`

This patch closes several high-risk implementation defects and makes additional
paths fail closed. It does **not** claim that every substantive requirement in
D01–D45 is complete. In particular, the production data directory contains only
`.gitkeep` files, operational empirical settings remain intentionally null/empty,
and the fully fitted nested Gate 1 plus the complete D38–D45 simulation procedure
are still incomplete. Consequently, no P00→P17 end-to-end PASS is claimed.

## Implemented and tested

- Semantic protocol hashing now ignores capture-only snapshot metadata while
  retaining file SHA-256, schema, header, semantic bindings and source inventory.
- Snapshot integrity remains independently protected by a detached snapshot hash.
- Artifact manifests record verified upstream dependency identities and hashes;
  exact immutable retries require identical content and provenance.
- `run_pipeline.py --resume` verifies raw, code, config, snapshot and completed
  artifact hashes before skipping units; drift is rejected.
- P02 carries only registered predictor bindings from the raw core panel into the
  as-of firm-year panel, with conflict detection and missingness preservation.
- P04 emits source/channel maturity curves and 12/24-month mature counts without
  converting immature observations to negatives.
- P08 semi-synthetic scenarios now lock and resample a covariate pool built only
  from development-year feature rows; outer rows cannot enter that pool.
- P05 stores `observed_channel_count`; L2 supports a locked quality/delay formula
  and otherwise remains `EMPIRICALLY_PENDING`. Fixed-π L3 now runs actual MCMC
  with source-specific Se/Sp, channel random effects, R-hat, ESS and PPC checks.
- P09 fits verification propensity from pre-decision observability features,
  reports SMD/support/ESS, applies stabilized IPW only when diagnostics pass and
  labels overlap weights as a different estimand.
- P11 executes bounded inner tuning, records runtime/configuration counts, fits
  Track A on L1, fits a selected soft-target Track B, and supports anchor-only
  bagging-PU without treating L1 as clean PU positives.
- P12 requires both a PASS simulation/MCSE artifact and a PASS freeze receipt
  before outer opening. Track B soft loss, rank correlation and concordance are
  separated from AP on the independent binary endpoint.
- P13 performs leave-one-domain-out refits and source/channel-exclusion refits
  without using outer outcomes in fitting.
- Gate 2 requires complete per-fold bootstrap evidence. Gate 3 requires exactly
  two distinct threshold bindings plus the pressure×monitoring block and returns
  `INSUFFICIENT_EVIDENCE` when required evidence is absent.
- Gate 3 also requires the same non-`none` measurement selection in at least 3/4
  fully nested folds before aggregating threshold/shape evidence.
- Gate 3 breakpoint stability now tests dispersion (`std(breakpoints)`) against
  the locked tolerance. It no longer substitutes distance of each breakpoint
  from zero for stability; any zero-location hypothesis must be a separate claim.
- P12 utility scenarios now execute scenario-based latent utility from frozen
  development-only L3 parameters. After the outer-open checkpoint, P12 integrates
  the channel random effects, derives `r_i(theta)` for each outer firm and reports
  reviewed cases, expected TP/FP/FN, review cost, additional false-positive cost,
  false-negative cost, net utility, matched full-vs-observability incremental
  utility and uncertainty combining L3 posterior-parameter draws with firm
  bootstrap. It does not substitute the
  observed endpoint or calibrated prediction for latent risk. An empty scenario
  registry still yields an explicit `SKIPPED` result.
- P05 writes aligned row-level fixed-pi posterior means into the source/channel
  matrices as pilot-only evidence. P10 no longer reads a metadata-only L3
  objective: it refits L3 within each outer-fold development history, removes the
  held-out channel's sources, computes strict held-channel cross-entropy, selects
  fixed-pi using development data and writes the selected fold-local posterior
  targets. P11 reads those targets from `channel_measurement_selection`, rejects
  outer/future rows and hashes that artifact into the freeze receipt.
- Known-case handling preserves the asymmetric downgrade-only/soft-veto rule.
- P17 remains report-only and does not import modeling, label, selection or
  simulation code.

## Implemented but awaiting empirical input

| Owner file | Required key/value |
| --- | --- |
| `config/methodology/risksets.yaml` | `risksets.data_cutoff`: approved cutoff date |
| `config/methodology/features.yaml` | `features.registry`: approved feature IDs, physical bindings, roles, availability, blocks and domain metadata |
| `config/methodology/measurement.yaml` | `measurement.l2_missingness.minimum_observed_channels`: positive integer |
| same | `measurement.l2_scoring.formula`: `quality_delay_weighted_observed_source_mean` only if approved |
| same | `measurement.l2_scoring.source_quality_by_profile`: value in [0,1] for every evidence profile |
| same | `measurement.l2_scoring.delay_half_life_days`: positive approved value |
| same | `measurement.l3_model.operational.fixed_pi_grid`: approved fixed-π grid |
| same | `measurement.l3_model.operational.accuracy_priors_by_profile`: complete beta-prior parameters for every evidence profile |
| `config/execution/learners.yaml` | `learners.tuning.search_spaces`: nonempty approved space for every confirmatory learner, ≤50 combinations |
| `config/execution/simulation.yaml` | `simulation.operational_scenarios`: locked D38–D45 scenario registry |
| `config/methodology/utility.yaml` | `utility.operational_scenarios`: locked scenarios containing `scenario_id`, `measurement_fixed_pi`, `true_positive_benefit`, `review_cost`, `additional_false_positive_cost`, `false_negative_cost`, and optional review-budget override |
| `config/methodology/inference.yaml` | two `threshold_feature_ids`, plus pressure, monitoring, parent-model and domain bindings |
| `config/methodology/entity_resolution.yaml` | aliases needed by actual unresolved source identities, if any |
| source catalog / actual data | empirically justified high-confirmation evidence anchor, if available; never infer it from known cases |

## Intentionally unavailable by design

- Hierarchical-π is sensitivity-only and may never enter primary Gate 1 selection.
- Known cases are optional, sealed until P15 and cannot upgrade Gate 2.
- Verification weighting remains unweighted when V is not observed or
  support/ESS diagnostics fail; missing verification is not evidence zero.
- Censoring sensitivity remains unavailable until a defensible exit/censoring-time
  source and bindings exist; delisting/merger is never automatically negative.
- L3 remains unavailable to Gate 1 when MCMC diagnostics, prior robustness or
  fold-local posterior evidence are insufficient.

## Still incomplete

- P03 does not yet carry a complete empirical quality variable Q and period-link
  confidence decision for every evidence profile.
- L3 row-level posterior means now follow P05 -> fold-local P10 -> P11 Track B,
  but posterior-draw robustness is not yet propagated through the complete nested
  learner/tuning/calibration procedure.
- P08 does not yet execute the entire production learner/selection/gate procedure
  for every D38–D45 scenario. Several operating-characteristic metrics remain
  reduced models even though both synthetic tiers now execute.
- P10 strict channel selection is not yet a fully fitted nested
  target→feature→tuning→calibration procedure for each `M*_{f,c}`. The current
  channel score is therefore not sufficient evidence for method completion.
- P12/P14 do not yet implement the complete pooled firm-cluster,
  outer-year-stratified simultaneous max-T/Holm family procedure across all CHNC2
  claims and the full six-rung benchmark ladder.
- P13 does not yet execute hierarchical-π, pairwise-dependence, L2
  mean/neutral/complete-channel, IPCW and worst–best sensitivity reruns.
- P16 has the two-threshold contract and substantive logistic/hinge/bootstrap
  implementation, but training-fold standardization/frozen interaction-library
  provenance still needs to be carried from P11 for a final confirmatory claim.
- A synthetic end-to-end runner fixture reaching P17 has not been established.
  The actual repo data tree currently has no source files, and production config
  correctly stops at source discovery or the first empirical blocker.

## Quality-gate evidence

Results after the last analytical code change. These are local results produced
in this workspace; no GitHub workflow run or commit status independently confirms
them yet.

| Gate | Result |
| --- | --- |
| Ruff check | PASS |
| Ruff format check | PASS — 107 files formatted |
| Pyright strict | PASS — 0 errors, 0 warnings, 0 informations |
| Pytest | PASS — 122 passed; 12 sklearn deprecation warnings |
| Bootstrap/config check | PASS |
| Pre-commit | PASS — all 6 hooks |
| `git diff --check` | PASS; only Git CRLF conversion notices were emitted |

Commands executed:

```text
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
uv run python scripts/bootstrap_repository.py --config config/pipeline.yaml --check
uv run pre-commit run --all-files
git diff --check
```

## End-to-end fixture evidence

- Production actual-source discovery: **FAIL-CLOSED as expected** at
  `profile=audit_opinions: expected at least one file`; required files are absent
  from `data/`.
- Full-capability P00→P17 fixture: **still incomplete**.
- Fail-closed behavior is covered by unit/invariant tests for missing source
  evidence, missing L2/L3 settings, missing positive classes, MCSE not met,
  incomplete Gate 2 evidence, unavailable known cases, missing Gate 3 bindings,
  artifact tampering and resume drift.

## Patch inventory

This targeted corrective patch modifies 19 tracked files: methodology/config
contracts, P05/P10/P11 orchestration, measurement/L3/selection/evaluation/gate
services, generated access/step catalogs, operational documentation and tests.
Use `git status --short` as the authoritative complete file list.
