# Chapter 3 requirement traceability

This document is the human-readable companion to the machine-generated D01–D45 catalog. It traces methodological obligations to their semantic owner and executable evidence; a configured value alone is not implementation evidence.

| Method block | Decisions | Single owners | Responsible stages | Required executable evidence |
|---|---|---|---|---|
| Prediction date, source lineage, event deduplication | D01, D03, D25 | `study.prediction_time`, `data_sources.*`, `evidence.*` | P01–P03 | Hash-verified source partitions; availability decisions; upstream event clusters; period-link confidence; lag identity; exit events never coded negative. |
| Horizon, maturity and censoring | D02, D21, D22, D37 | `study.horizons_months`, `risksets.*` | P04, P12–P13 | Complete mature main cohort; source maturity curves; 12/24-month outputs; prospective cohorts; IPCW and worst/best sensitivity with changed-estimand labels. |
| L0–L3 measurement | D04–D06, D14–D17, D34 | `measurement.*`, anchor source registry | P05, P08, P10–P13 | L0/L1; L2 g_c and fold-ECDF soft targets with coverage patterns; fixed-π L3 MCMC with source Se/Sp and channel dependence; R-hat/ESS/PPC/prior diagnostics; posterior draws; hierarchical-π sensitivity only. |
| Verification/observability | D18, D20 | `weighting.*`, `features.*`, label-model blocks | P06–P09 | V versus O classification; pre-decision propensity model; overlap/SMD/weight/ESS diagnostics; stabilized IPW only under support; overlap/trimming estimand labels; content firewall mutation tests. |
| Simulation and protocol gate | D23, D38–D45 | `simulation.*`, `evaluation.gate_common` | P08, P12 | Two simulation tiers; all DGP dimensions; production procedures plus oracle; all locked metrics; core/L3/continuous MCSE rules; operating characteristics for Gates 1–3; no outer opening unless simulation PASS. |
| Nested time/channel validation | D10, D27, D32 | `folds.*`, `measurement.strict_selection/selection` | P09–P11 | Outer/inner/calibration origins; maturity roles; strict channel exclusion from target, label model, features, tuning and calibration; M*f/M*f,c/none decision provenance. |
| Learners, tuning and tracks | D08, D09, D15, D26, D28 | `learners.*`, `measurement.selection` | P11 | Equal-budget search with ≤50 valid configurations and runtime; Track A always; selected Track B only when eligible; anchor-only PU; fold preprocessing; freeze hashes and PASS receipt. |
| Evaluation, calibration, bootstrap, utility | D07, D11, D29, D31, D35–D36 | `evaluation.*`, `calibration`, `inference.*`, `utility.*` | P12, P14 | Cross-fitted calibrator; binary metrics only on independent binary endpoint; Track B soft metrics; paired firm/year-stratified bootstrap; simultaneous family inference; actual utility scenarios. |
| Transfer, sensitivities and gates | D12–D13, D24, D33, D37 | `domains.*`, `evaluation.gate3`, `inference.interaction_library` | P13–P16 | Leave-one-domain-out refits/support; source/dedup/lag/L2/L3/censoring/uncertainty reruns; Gate 2 fail-closed; threshold and interaction refits with adjusted uncertainty; ≥3/4 measurement stability before pooled shape. |
| Known cases | D19, D30 | `known_cases.*`, access policy | P00, P15 | Pre-development seal/hash; forbidden-use tests; post-freeze stratified percentile/permutation evidence across locked sensitivities; downgrade/soft-veto only. |
| Derived reporting | all | `reporting.*`, artifact catalog | P17 | Verified manifest inventory and dependency chain; result ledger/gate matrix/decision log/tables/figure/report derived without importing modeling, selection, labels or simulation modules. |

## Acceptance mapping

- Protocol reproducibility: two independently created snapshots with identical source bytes/schema/mappings must produce the same semantic protocol hash; their detached snapshot integrity hashes may differ.
- Provenance: every post-P00 artifact manifest must list the verified content hashes of artifacts read by its producer.
- Firewalls: P12 requires a PASS freeze receipt and a PASS simulation/MCSE receipt before writing `outer_open_receipt`; outer data remain unread on either failure.
- Fail-closed gates: missing/insufficient evidence yields `INSUFFICIENT_EVIDENCE` or `SKIPPED`, never PASS.
- L3 chain: P05 pilot row posteriors are explicitly pilot-only; P10 must create the
  development-only held-channel objective and selected fixed-pi posterior rows;
  P11 must reject any L3 target not carried by that fold's selection artifact.
- Utility evidence: a nonempty scenario must bind a frozen measurement fixed-pi,
  derive `r_i(theta)` without substituting outer outcomes or calibrated model
  risk, and produce expected counts, component costs, net/incremental latent
  utility and uncertainty combining L3 posterior-parameter draws with firm
  bootstrap; `REGISTERED` metadata is
  not an analytical result.
- Gate 3 stability: the breakpoint criterion is dispersion across folds, not the
  absolute distance of breakpoint locations from zero.
- Empirical blockers: null/empty operational inputs remain explicit and stop at the first stage that requires them; fixture-only overrides must never be silently copied into production config.
