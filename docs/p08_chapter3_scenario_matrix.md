# P08 Chapter 3 scenario matrix

## Purpose

This matrix implements the baseline-plus-locked-stress-block design in Chapter 3 without taking a full Cartesian product. The measurement-focused run holds the learner fixed at logistic regression and compares four feasible label strategies plus two standalone identification estimators. The existing `core` profile remains the separately locked learner-comparison run.

## Execution profiles

| Profile | Role | Active methods |
|---|---|---:|
| `chapter3_measurement` | Confirmatory measurement simulation across all operational scenarios | 6 |
| `core` | Confirmatory learner comparison on baseline scenarios in a separately locked run | 22 |

The active profile in this branch is `chapter3_measurement`.

## Operational scenarios

| Block | Scenario ID | Tier | Primary change |
|---|---|---|---|
| S00 | `empirical_baseline__target_L1_ANNUAL` | Semi-synthetic | Exact P02 development covariates; P05-calibrated prevalence |
| S01 | `synthetic_baseline` | Fully synthetic | Reference DGP |
| S02 | `synthetic_low_prevalence` | Fully synthetic | Prevalence 0.02 |
| S03 | `synthetic_high_prevalence` | Fully synthetic | Prevalence 0.20 |
| S04 | `synthetic_noisy_anchor` | Fully synthetic | Anchor false-positive rate 0.03 |
| S05 | `synthetic_noisy_weak_source` | Fully synthetic | Weak-source false-positive rate 0.10 |
| S06 | `synthetic_strong_source_dependence` | Fully synthetic | Within-channel dependence 0.70 |
| S07 | `synthetic_strong_selective_verification_full_support` | Fully synthetic | Strong selection with retained base support |
| S08 | `synthetic_strong_selective_verification_limited_support` | Fully synthetic | Strong selection and low base verification |
| S09 | `synthetic_slow_detection` | Fully synthetic | Mean detection delay 365 days |
| S10 | `synthetic_severe_censoring` | Fully synthetic | 90-day horizon with 365-day mean delay |
| S11 | `synthetic_null_signal` | Fully synthetic | No content signal |
| S12 | `synthetic_nonlinear_signal` | Fully synthetic | Nonlinear content signal |
| S13 | `synthetic_interaction_signal` | Fully synthetic | Interaction content signal |
| S14 | `synthetic_small_sample` | Fully synthetic | Sample size 500 |
| S15 | `synthetic_large_sample` | Fully synthetic | Sample size 12,000 |
| S16 | `synthetic_combined_adverse` | Fully synthetic | Rare outcome, noisy sources, dependence, limited support and censoring |

## Computational size

Under `chapter3_measurement`, each scenario activates four predictive methods and two standalone estimators. At the locked minimum replications this gives:

- predictive: 4 methods × 2,500 replications;
- standalone: 2 methods × 1,000 replications;
- 60 initial batch artifacts per scenario;
- 1,020 initial batch artifacts across 17 scenarios.

Adaptive replication may extend methods that do not meet their MCSE target.

## Deliberately deferred engine blocks

The following Chapter 3 commitments are not represented as operational scenarios yet because the current DGP does not implement them correctly:

1. S3 complete-next-calendar-year maturity distinct from generic delayed maturity;
2. covariate shift, concept drift and label-policy drift (`shift_strength` is currently validated but not applied);
3. Gate 1–3 selection error, type-I error, power and breakpoint recovery metrics;
4. separate L3 correct and misspecified variants: ignore dependence, wrong fixed prevalence and clean-anchor assumption.

These blocks must be implemented in the simulation engine before being moved from `deferred_engine_blocks` into `operational_scenarios`. They must not be represented by inert YAML parameters.

## Run policy

Create a fresh P00 lock and run ID after merging this branch. Do not resume an artifact tree generated under the former one-scenario protocol.
