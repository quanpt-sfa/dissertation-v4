# L2/L3 calibration protocol

## Scope

This protocol uses only eligible development-history rows before the initial outer year. Source-specific observability and missingness determine whether an individual source contributes. Outer-fold outcomes are never used to set L2 coefficients, the L3 fixed-prevalence grid, or L3 accuracy priors.

## Neutral L2

The primary L2 configuration is deliberately neutral:

- formula: `quality_delay_weighted_observed_source_mean`;
- equal channel weights;
- quality weight `1.0` for every registered evidence profile;
- delay half-life of 365 days, aligned with the locked 12-month primary horizon;
- at least one observed channel;
- missing sources remain missing and are not recoded to zero.

This is a measurement baseline, not a claim that all sources have equal diagnostic accuracy. Alternative quality weights and stricter coverage rules belong in sensitivity analyses and require external validation.

## Development-only calibration report

After P05, run `scripts/report_measurement_calibration.py`. It produces:

- logical-source coverage and observed source counts for diagnostics;
- channel coverage;
- one physical-channel coverage row per S1/S2/S3 measurement channel;
- development-only positive rates by source and channel;
- lag distributions by source;
- a prevalence-anchor JSON file;
- an L3 prior-elicitation worksheet.

Prevalence-anchor quantiles use each measurement channel exactly once. Logical endpoints from the same physical source remain visible for diagnostics but do not enter the anchor distribution separately. Thus the four S3 logical endpoints cannot multiply the influence of the sanction source.

Observed positive rates are not treated as latent prevalence estimates. They depend jointly on latent prevalence, sensitivity, specificity, opportunity, and detection delay.

## S3 year audit

Before locking L3, run `scripts/report_s3_year_audit.py` against the same P00-P05 run. The script reads verified artifacts and produces:

- decision-ledger counts by target fiscal year;
- P03 endpoint outcomes by year and S3 endpoint;
- P05 endpoint outcomes by year and S3 endpoint;
- unknown-outcome reason counts;
- P03-to-P05 reconciliation summaries;
- row-level mismatch details;
- a JSON lock-status summary.

The audit must establish whether positive S3 decisions exist before the initial outer year and whether P03 positives propagate unchanged into P05 matrices. Any missing key or outcome mismatch blocks L3. If development-history S3 contains no positive, S3 sensitivity is not empirically informed and L3 remains blocked pending a defensible design decision.

Example:

```powershell
uv run python scripts/report_s3_year_audit.py `
  --registry $registry `
  --run-id $runId `
  --output-dir "$outputRoot\$runId\S3_AUDIT"
```

The audit does not re-read the raw CSV. It reconciles verified P03 and P05 artifacts and uses the verified sanction decision ledger plus the annual evidence audit to summarize pre-P03 exclusions.

## Fixed-pi grid

The fixed-prevalence grid must be locked before outer-fold access. The grid should be justified from three inputs:

1. external evidence on the plausible incidence or prevalence of material reporting problems;
2. development-only unique-channel observed-rate anchors from the calibration report;
3. a deliberately broad sensitivity range that does not assume any observed source is a gold standard.

The final grid should include low, central, and high plausible scenarios. It must not be selected using outer-fold predictive performance.

## Accuracy priors

Each evidence profile receives Beta priors for sensitivity and specificity. Priors are parameterized using a prior mean `m` and prior effective sample size `kappa`:

- `alpha = m * kappa`;
- `beta = (1 - m) * kappa`.

The evidence basis for each prior must be recorded as one of:

- external published validation evidence;
- an independently reviewed validation subsample;
- structured expert elicitation;
- a weakly informative prior used because stronger evidence is unavailable.

A stronger prior effective sample size is permitted only when the evidence basis is correspondingly stronger. Logical endpoints from one physical source are not independent prior-information units; their observed counts must not be summed to inflate `kappa`. The primary analysis must be accompanied by weak, skeptical, and evidence-hierarchy sensitivity scenarios.

## No-gold-standard cautions

Bayesian latent-class models can be unidentified or weakly identified without informative restrictions. Prior precision can create misleading posterior precision when prior means are inaccurate. Conditional dependence between sources can also inflate estimated accuracy. The implementation therefore retains channel random effects, posterior predictive checks, R-hat and ESS gates, minimum source counts, and fixed-pi sensitivity scenarios.

## Lock decision

L3 remains `EMPIRICALLY_PENDING` until the calibration report, S3 year audit, and prior worksheet have been reviewed. Only then should `measurement.l3_model.operational.fixed_pi_grid` and `accuracy_priors_by_profile` be committed in a new protocol-locking change.

## Methodological references

- Sun, A., and Zhou, X.-H. (2025), estimation of diagnostic-test accuracy without gold standards.
- Albert, P. S. (2004), caution on robustness of latent-class estimates without a gold standard.
- Spencer, B. D. (2012), conditions under which latent-class models overstate classifier accuracy.
- Dendukuri and colleagues, Bayesian latent-class analysis using informative Beta priors.
- Johnson et al. (2019), Bayesian hierarchical latent-class models for multiple tests or raters.
- Prior-precision simulation evidence showing that precise but inaccurate priors can worsen inference.
