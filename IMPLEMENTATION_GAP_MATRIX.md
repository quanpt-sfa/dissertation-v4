# Implementation gap matrix — P00–P17

Audit baseline: 2026-07-14. Authority order: `Chapter3_Methodology_Simulation_Technical_Fixes_v19.md`, Appendix B D01–D45, then the locked registry compiled from `config/pipeline.yaml`. Generated catalogs describe contracts but are not evidence that a method is implemented.

The `Baseline` column records the implementation before this completion work. `Closure` is deliberately fail-closed and is updated only when substantive code and mutation/failure tests exist.

## Current closure status

This overlay is the current status after the patch and is authoritative over the
baseline work-log values below.

| Status | Decisions | Evidence summary |
| --- | --- | --- |
| complete | D08, D18, D19, D20, D28, D33, D39 | Registered supervised learners plus anchor-only PU; development-only propensity/support diagnostics; seals/firewalls; content exclusion; 3/4 M* stability enforcement; development-only semi-synthetic covariate pools; tests cover success and mutation paths. D08/D18/D28/D39 still require their empirical source/feature inputs before a production run can exercise them. |
| partial | D01–D07, D09–D17, D21–D27, D29–D32, D34–D38, D40–D45 | Substantive pieces and fail-closed contracts exist, but one or more method components, empirical locks or end-to-end evidence listed in `P00_P17_COMPLETION_REPORT.md` remain. |
| missing | none currently known | Remaining gaps have substantive partial implementations but are not method-complete. |
| inconsistent | none currently known | Earlier false-PASS/propagation inconsistencies were converted to fail-closed or partial paths; this does not promote the remaining partial methods to complete. |

| Requirement | Config owner | Stage | Input → output artifact | Baseline implementation and protecting test | Baseline | Closure |
|---|---|---|---|---|---|---|
| D01 prediction time/as-of availability | `study.prediction_time` | P01–P03 | snapshot/raw audit/panel → `availability_registry`, `evidence_ledger` | Physical availability is audited, but period-link confidence and negative-lag decisions are incomplete; T001 checks config more than execution. | partial | open |
| D02 primary 12-month horizon and sensitivity horizons | `study.horizons_months` | P04, P12–P13 | panel/evidence → `risk_sets`, horizon metrics | P04 implements the primary horizon only; 24-month rerun/source maturity evidence is absent. | partial | open |
| D03 eligible source set and LOCO | `data_sources.eligibility`, `leave_one_channel_out_required` | P01, P03, P09–P10 | snapshot/evidence → channel selections | Catalog audit exists; strict LOCO selection is an arithmetic proxy and not a fitted nested procedure. | partial | open |
| D04 high-confirmation anchor with false-positive uncertainty | `data_sources.anchor` | P01, P05, P08 | raw audit/matrix → anchor/L3/simulation evidence | Anchor IDs are detected, but the locked 0/.01/.03/.05 grid does not drive empirical L3/PU fitting. | partial | open |
| D05 L1 primary, L0/L2/L3 secondary roles | `measurement.roles`, `track_a_primary_endpoint` | P05, P10–P13 | matrices → labels/models/metrics | Roles are recorded; L3 is not estimated and selected L2 is not propagated into P11. | inconsistent | open |
| D06 fixed-π grid primary, hierarchical-π sensitivity | `measurement.prior_accuracy_domain` | P05, P08, P13 | matrices/scenarios → L3 posterior/sensitivity | Only a conditional-independence posterior helper exists; no empirical grid/MCMC/hierarchical rerun. | missing | open |
| D07 5% review budget plus 1%, 10%, fixed-count sensitivities | `evaluation.review_budget` | P12–P13 | predictions/outcomes → utility/yield | Primary precision@5% exists; registered utility is metadata-only and budget sensitivities do not rerun. | partial | open |
| D08 elastic-net/RF/boosting plus Anchor-PU | `learners.*` | P11 | features/targets → `model_artifacts` | Three supervised learners fit; Anchor-PU is not fit. | partial | open |
| D09 ≤50 valid configs per learner/inner fold and runtime ledger | `learners.tuning` | P11 | inner splits → freeze receipt | A single fixed configuration is fit and no search/runtime/valid-count evidence is emitted. | missing | open |
| D10 rolling outer folds and 2025 prospective | `folds.*` | P09–P12 | maturity/feature panel → split registry | Outer years exist, but explicit inner origins/calibration origins and maturity-dependent prospective conversion are incomplete. | partial | open |
| D11 Gate 2 simultaneous inference/MMI/3-of-4/yield/positive counts | `evaluation.gate2` | P12, P14 | pooled predictions/bootstrap → `gate2_verdict` | Some thresholds exist; bootstrap is not pooled year-stratified simultaneous family inference. | partial | open |
| D12 Gate 3 breakpoint/support/domain/fold criteria | `evaluation.gate3` | P16 | frozen features/predictions → Gate 3 | Current least-squares approximation lacks family-adjusted uncertainty and full domain replication. | partial | open |
| D13 prespecified internal–external domains | `domains.*` | P13 | predictions/features → domain transfer | Descriptive summaries exist; leave-one-domain-out refits and common-support gates are absent. | partial | open |
| D14 formal L0–L3 roles | `measurement.roles` | P05, P10 | evidence → candidate/selection registry | L0/L1/L2 artifacts exist; L3 is capability-only. | partial | open |
| D15 isolate measurement × learner | `measurement.selection`, `learners.confirmatory` | P10–P12 | candidates → models/metrics | Selection records L2, but P11 always trains on L1; this breaks the required causal chain. | inconsistent | open |
| D16 soft CE and L3 posterior-draw robustness | `measurement.objectives` | P08, P10–P12 | soft targets/draws → candidate and Track B metrics | No fitted soft-target learner or posterior-draw propagation. | missing | open |
| D17 fixed-π latent class with source Se/Sp, channel RE, MCMC/PPC | `measurement.l3_model` | P05, P08, P13 | source matrices → L3 pilot/posteriors/diagnostics | No substantive MCMC; `EMPIRICALLY_PENDING` is emitted without attempting estimation. | missing | open |
| D18 selective verification model and support-aware weights | `weighting.*` | P06, P09 | matrix/features → observability/weights/diagnostics | Classification is descriptive; propensity is smoothed by year only, not modeled from pre-decision covariates. | partial | open |
| D19 K1–K4 sealing and forbidden early access | `known_cases.*`, access matrix | P00, P15 | seal → known-case result | Seal/access rules and mutation tests exist; runtime receipt semantics still require stronger status checking. | partial | open |
| D20 label model uses only S/M/T/Q/ZM | `evidence.label_model_allowed_blocks`, `features.*` | P05–P07 | registry/matrix → leakage evidence | Content-feature firewall is implemented and tested, but future L3 code must preserve it. | complete | verify |
| D21 τ+h≤cutoff plus complete follow-up | `risksets.*` | P04 | panel → risk sets | Calendar maturity is implemented; source-specific follow-up completeness is not. | partial | open |
| D22 common horizon and source maturity curves | `study.horizons_months`, `source_maturity_curves_required` | P04, P12–P13 | evidence lags → maturity curves/metrics | No source maturity curves or 24-month rerun. | missing | open |
| D23 common gate thresholds, ESS/support and power | `evaluation.gate_common` | P08, P09, P14, P16 | simulation/weights/results → gates | Constants are configured; power/operating-characteristic evidence is not required for opening outer outcomes. | inconsistent | open |
| D24 two threshold claims and pressure×monitoring block | `inference.interaction_library` | P16 | bindings/features/models → threshold result | Bindings fail closed when null; substantive fit is an approximation and no interaction-block model refit exists. | partial | open |
| D25 exit/merger is not negative | `evidence.exit_events` | P03–P04, P13 | events/risk sets → censoring sensitivity | Negative assignment is prohibited, but actual exit status/bounds are not modeled. | partial | open |
| D26 Track A always; Track B only for selected M* | `measurement.selection` | P10–P12 | selection → model tracks | Track A fits, but selected Track B is mislabeled L1 rather than trained on M*. | inconsistent | open |
| D27 channel nested within time and removed everywhere | `measurement.strict_selection` | P09–P11 | channel split → selection/freeze | Held channel is omitted from a score average, but not from a fitted target/feature/tuning/calibration pipeline. | partial | open |
| D28 Anchor-PU positive set only | `learners.pu_branch`, `sensitivity` | P11, P13 | anchor outcomes → PU models | No production PU estimator or positive-set audit. | missing | open |
| D29 paired firm bootstrap stratified by outer year | `inference.bootstrap` | P12 | pooled OOF → bootstrap batches | Per-fold row-position resampling exists; it is neither pooled nor explicitly year-stratified/cluster-preserving. | partial | open |
| D30 known-case asymmetric soft veto | `known_cases.soft_veto` | P15 | sealed cases/predictions → veto | Percentile rules exist; locked sensitivity-scenario and stratified permutation coverage is incomplete. | partial | open |
| D31 Track B soft loss/concordance and binary transfer only | `evaluation.track_b_metrics` | P12 | Track B predictions → fit/transfer metrics | Track B is not genuinely trained; no soft loss/concordance evidence. | missing | open |
| D32 fold-local M*f and M*f,c or none | `measurement.selection` | P10 | development evidence → selection | Fold-local registry exists; candidate estimator/criterion is incomplete and L3 objective can be copied from capability metadata. | partial | open |
| D33 shape aggregation only when same M in ≥3/4 folds | `measurement.selection.stability_*` | P13, P16 | selection registry → ablation/shape | No cross-fold stability enforcement in Gate 3. | missing | open |
| D34 L2 observed-channel normalization/min coverage/sensitivities | `measurement.l2_missingness` | P05, P10, P13 | channel matrix → L2 targets/sensitivities | Observed-channel mean exists; g_c(S,T,Q), train-fold ECDF, minimum coverage and real sensitivity reruns are absent. | partial | open |
| D35 pooled cross-fitted development calibrator | `calibration` | P11–P12 | OOF predictions → calibrator | OOF Platt fit exists, but selection between methods and minimum-positive stability rules are incomplete. | partial | open |
| D36 CHNC2/CHNC3 familywise inference | `inference.*` | P12, P14, P16 | bootstrap families → gates | No max-T/simultaneous/Holm family implementation across claims. | missing | open |
| D37 complete mature main; IPCW diagnostic sensitivity | `risksets.ipcw_role`, `weighting.ipcw_role` | P04, P13 | censoring/verification → ablation | Main mature cohort exists; IPCW sensitivity is status metadata rather than a refit. | partial | open |
| D38 simulate L1–L3, verification, gates, utility under known truth | `simulation.objective` | P08 | scenarios → batches/MCSE | DGP covers a subset; gate operating characteristics and utility regret are not simulated. | missing | open |
| D39 fully synthetic plus semi-synthetic development covariates | `simulation.tiers` | P08 | development feature distribution → simulations | Only fully synthetic scalar content is used. | missing | open |
| D40 Y*, V/O, sources, dependence, delay, shift, correct/misspecified L3 | `simulation.dgp` | P08 | scenario → simulated data | Y*/sources/verification/dependence/delay exist; shift and explicit correct/misspecified fitted L3 are incomplete. | partial | open |
| D41 baseline plus locked stress blocks over all dimensions | `simulation.scenario_space` | P08 | operational scenario registry → batches | Validator accepts a small subset and does not prove required dimension/block coverage. | partial | open |
| D42 L0–L3/learners/obs-content-full/PU/oracle/noisy sensitivity | `simulation.targets/learners/methods/sensitivities` | P08 | simulated data → method results | Scores are hand-built proxies, not the production estimation procedures; oracle/noisy-positive branches absent. | missing | open |
| D43 fixed/hierarchical L3, selection, transfer/gate/utility metrics | `simulation.metrics` | P08 | repetitions → MCSE report | Only coverage/MAE/AP/AUC diagnostics are emitted. | missing | open |
| D44 tier-specific R and MCSE plus continuous 10%-MMI rule | `simulation.core/l3/continuous_metrics` | P08 | batches → MCSE controller | Core count/one MCSE threshold exists; L3 tier and continuous-MMI rules are ignored. | partial | open |
| D45 simulation completes before outer; only logged demotion/fix | `simulation.protocol_role`, reproducibility | P00, P08, P12 | MCSE/freeze → outer-open receipt | P12 checks freeze but not MCSE PASS; runner can continue after P08 SKIPPED. | inconsistent | open |

## Dependency order

1. Semantic protocol hash, manifest dependency provenance, receipt semantics, resume/incomplete-run handling.
2. P03–P06 measurement inputs, maturity curves, L2 and L3/verification estimators.
3. P08 production-procedure simulation and tier-specific MCSE controller.
4. P09–P10 nested origins, fitted verification weights and substantive measurement selection.
5. P11–P12 Track A/Track B/PU, nested tuning, calibration, pooled inference and utility.
6. P13–P16 real sensitivity refits, domain validation, family inference and gates.
7. P17 derived-only reporting, end-to-end and fail-closed fixtures, final quality gates.
