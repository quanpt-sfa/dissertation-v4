# L2/L3 calibration protocol

## Scope

This protocol uses only eligible development-history rows before the initial outer year. Source-specific observability and missingness determine whether an individual source contributes. Outer-fold outcomes and known cases are never used to define L2 coefficients, choose an L3 prevalence scenario, set L3 accuracy priors, or modify the measurement model.

L1 annual remains the mandatory primary track. L2 and L3 are optional, capability-gated tracks. Failure of an optional track means that track is unavailable by design; it does not replace or invalidate L1.

## Neutral L2

The primary L2 configuration is deliberately neutral:

- formula: `quality_delay_weighted_observed_source_mean`;
- equal channel weights;
- quality weight `1.0` for every registered evidence profile;
- delay half-life of 365 days, aligned with the locked 12-month primary horizon;
- at least one observed channel;
- missing sources remain missing and are not recoded to zero.

This is a measurement baseline, not a claim that all sources have equal diagnostic accuracy. Alternative quality weights and stricter coverage rules belong in sensitivity analyses and require external justification.

## P0-registered L3 scenarios

L3 prevalence assumptions and source-accuracy priors are declared in `config/methodology/l3_scenarios.yaml`. That module is loaded by `config/pipeline.yaml`, included in the protocol hash, validated during P00 compilation, and forbidden from modification after P00.

The registered fixed-prevalence scenarios are:

- `low_pi_01`: fixed π = 0.01, robustness;
- `neutral_pi_03`: fixed π = 0.03, preregistered primary;
- `high_pi_05`: fixed π = 0.05, robustness.

All registered scenarios are executed when L3 capability is available. Diagnostic losses are used only for capability assessment and reporting. They may not select a scenario, replace `neutral_pi_03`, or revise the scenario registry.

The primary scenario is therefore determined by protocol registration, not by minimum loss, outer-fold performance, known-case behavior, or post-Preparation review.

## P0-registered accuracy priors

Each evidence profile receives Beta priors for sensitivity and specificity in the same P0 module. The baseline registered prior set is:

- financial-statement core: sensitivity Beta(8, 2), specificity Beta(12, 1);
- annual audit evidence: sensitivity Beta(7, 3), specificity Beta(10, 2);
- sanction evidence: sensitivity Beta(6, 4), specificity Beta(15, 2).

These are measurement-sensitivity assumptions, not claims that S3 is fraud ground truth. S3 remains evidence of administrative sanctions. Logical endpoints from one physical source are not independent prior-information units and cannot be summed to inflate prior precision.

Hierarchical π remains sensitivity-only. It cannot replace the registered fixed-π primary analysis.

## Development-only capability report

After P05, `scripts/report_measurement_calibration.py` reports:

- logical-source coverage and observed source counts;
- channel coverage;
- one physical-channel coverage row per S1/S2/S3 measurement channel;
- development-only positive rates by source and channel;
- lag distributions by source;
- prevalence-anchor diagnostics;
- prior and scenario diagnostics.

These outputs assess observability, numerical feasibility, and capability. They do not authorize any change to the P0 scenario registry or priors.

Observed positive rates are not treated as latent prevalence estimates. They depend jointly on latent prevalence, sensitivity, specificity, opportunity, and detection delay. Prevalence-anchor quantiles use each physical measurement channel once, so multiple logical S3 endpoints cannot multiply the sanction source's influence.

## S3 year audit

`scripts/report_s3_year_audit.py` reads verified artifacts and produces:

- decision-ledger counts by target fiscal year;
- P03 and P05 endpoint outcomes by year and S3 endpoint;
- unknown-outcome reason counts;
- P03-to-P05 reconciliation summaries;
- row-level mismatch details;
- a JSON capability and data-contract summary.

The audit must establish whether eligible S3 decisions exist before the initial outer year and whether P03 positives propagate unchanged into P05 matrices. Missing keys, outcome mismatches, or unresolved eligible sanction-year mappings are data-contract blockers.

No development-history S3_CONTENT positives, insufficient channel coverage, or failed L3 diagnostics make L3 unavailable by design. They do not trigger post-data parameter revision and do not block the mandatory L1 pipeline.

Example:

```powershell
uv run python scripts/report_s3_year_audit.py `
  --registry $registry `
  --run-id $runId `
  --output-dir "$outputRoot\$runId\S3_AUDIT"
```

The audit does not re-read the raw CSV. It reconciles verified P03 and P05 artifacts and uses the verified sanction decision ledger plus the annual evidence audit to summarize pre-P03 exclusions.

## No-gold-standard cautions

Bayesian latent-class models can be unidentified or weakly identified without informative restrictions. Prior precision can create misleading posterior precision when prior means are inaccurate. Conditional dependence between sources can also inflate estimated accuracy. The implementation therefore retains channel random effects, posterior predictive checks, R-hat and ESS gates, minimum source counts, fixed-π robustness scenarios, and capability-based skipping.

## Assurance decision D06

D06 is locked at P00. Its executable assurance contract requires:

- `measurement.prior_accuracy_domain.primary_prior = preregistered_l3_scenario_registry`;
- `l3_scenarios.status = LOCKED_AT_P0`;
- `neutral_pi_03` as the sole primary scenario;
- all registered scenarios to run when capability permits;
- performance-based scenario selection to remain forbidden;
- outer outcomes and known cases to remain inaccessible;
- hierarchical π to remain sensitivity-only.

The D06 amendment is a named manifest module, included in source hashes and applied before decision traceability is compiled. Generated Appendix B and D01–D45 traceability therefore use the same P0 decision as runtime code.

## Methodological references

- Sun, A., and Zhou, X.-H. (2025), estimation of diagnostic-test accuracy without gold standards.
- Albert, P. S. (2004), caution on robustness of latent-class estimates without a gold standard.
- Spencer, B. D. (2012), conditions under which latent-class models overstate classifier accuracy.
- Dendukuri and colleagues, Bayesian latent-class analysis using informative Beta priors.
- Johnson et al. (2019), Bayesian hierarchical latent-class models for multiple tests or raters.
- Prior-precision simulation evidence showing that precise but inaccurate priors can worsen inference.
